"""V3 stagewise reasoner orchestrator.

This module never modifies v2 code. It uses v2 building blocks (``V2ArtifactsWriter``,
``search_handlers``, ``FrontierModel``, ``SearchClient``, ``ConversationManager``) as libraries
and runs four independent stage pools (S1, S1D1, S2, S3+S4) so different upstream services
(text LLM, SERP+CDN, VLM) can be saturated in parallel.

Resume policy: ``adopt_in_place``. The output_dir for each row IS the destination
``eval_row_NNN`` directory; ``V2ArtifactsWriter`` writes new stage files into the existing
``artifacts_files/`` directly. No staging round-trip in this MVP.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .frontier_model import FrontierModel, normalize_prompt_analysis_hint_context
from .search_client import SearchClient
from .conversation import ConversationManager
from .search_stages import (
    run_image_download_for_serp_rows,
    run_image_search_for_query,
    run_image_search_only_for_query,
    run_image_selection_from_s1d1_bucket,
    run_text_search_for_query,
)
from .artifacts import V2ArtifactsWriter, plan_step_key
from .row_state import RowStage, RowState, classify_row, enqueue_plan, histogram
from .finalize import finalize_row


# ─────────────────────────── Helpers ───────────────────────────


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_row_logger(name: str, row_dir: Path) -> logging.Logger:
    """Per-row logger that writes to ``row_dir/v3_pipeline.log`` (separate from v2's ``pipeline_debug.log``)."""
    row_dir.mkdir(parents=True, exist_ok=True)
    log_path = row_dir / "v3_pipeline.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _reconstruct_reference_images_from_s2(s2_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Inverse of ``flush_s2_reference_selection``: turn s2 ``selections`` back into the runtime list."""
    refs: List[Dict[str, Any]] = []
    selections = s2_doc.get("selections") if isinstance(s2_doc, dict) else None
    if not isinstance(selections, dict):
        return refs
    for plan_key, sel in selections.items():
        if not isinstance(sel, dict):
            continue
        if sel.get("selection_skipped"):
            continue
        si = sel.get("selected_image")
        if not isinstance(si, dict) or not si.get("url"):
            continue
        refs.append(
            {
                "query": str(sel.get("query") or ""),
                "query_index": sel.get("query_index"),
                "selected_image": dict(si),
                "selection_reasoning": str(sel.get("selection_reasoning") or ""),
                "identified_knowledge_gaps": str(sel.get("identified_knowledge_gaps") or ""),
                "visual_description": str(sel.get("visual_description") or ""),
                "candidate_image_mappings": sel.get("candidate_image_mappings") or [],
            }
        )
    return refs


def _writer_for_row(row_dir: Path, *, s1_doc: Optional[Dict[str, Any]] = None,
                    s1d1_doc: Optional[Dict[str, Any]] = None,
                    s2_doc: Optional[Dict[str, Any]] = None,
                    s3_doc: Optional[Dict[str, Any]] = None) -> V2ArtifactsWriter:
    """Construct a writer whose in-memory state mirrors what's on disk (needed by later writers)."""
    w = V2ArtifactsWriter(row_dir)
    if s1_doc is not None:
        w.s1_data = s1_doc
        # _runtime_needs_search defaults to False; write_s4 falls back to parsed_llm_output.needs_search
        # so this is fine for resume.
    if s1d1_doc is not None:
        w.search_results_data = s1d1_doc
    if s2_doc is not None:
        w.s2_document = s2_doc
    if s3_doc is not None:
        w.s3_data = s3_doc
    return w


# ─────────────────────────── Runner ───────────────────────────


class V3StagewiseRunner:
    """Single-process asyncio orchestrator with four stage pools.

    Args:
        rows: pre-validated 1-based ordered list of evalset rows (from ``rows_for_evalset``).
        gen_root: output root.
        evalset: evalset name (registry key).
        reasoner: reasoner lane name.
        tree_subdir: filesystem subdir (``evalset_fs_subdir(evalset)``).
        model_name: LLM model name for FrontierModel.
        mode: ``full`` or ``stop_s2``.
        workers_*: per-stage concurrency caps.
        hint_mode: forward dataset hints to ``analyze_prompt``.
        max_downloaded_images_per_query: forwarded to ``run_image_search_for_query``.
        max_search_results: forwarded to text/image search.
        limit: process at most this many *non-terminal* rows.
        only_row_indices: optional 1-based indices to restrict to.
    """

    def __init__(
        self,
        *,
        rows: List[Dict[str, Any]],
        gen_root: Path,
        evalset: str,
        reasoner: str,
        tree_subdir: str,
        model_name: str,
        model_name_s1: Optional[str] = None,
        mode: str,
        workers_s1: int = 16,
        workers_s1d1: int = 24,            # Back-compat: if set & per-half not set, splits equally.
        workers_s1d1_search: Optional[int] = None,
        workers_s1d1_download: Optional[int] = None,
        workers_s2: int = 8,
        workers_s3: int = 12,
        hint_mode: bool = False,
        max_downloaded_images_per_query: int = 3,
        max_search_results: int = 10,
        limit: Optional[int] = None,
        only_row_indices: Optional[List[int]] = None,
        staging_dir: Optional[Path] = None,
        row_dirs: Optional[List[Path]] = None,
    ) -> None:
        assert mode in ("full", "stop_s2")
        self.rows = rows
        self._row_dir_overrides: Dict[int, Path] = {}
        if row_dirs is not None:
            assert len(row_dirs) == len(rows), f"row_dirs length {len(row_dirs)} != rows length {len(rows)}"
            for i, p in enumerate(row_dirs):
                self._row_dir_overrides[i + 1] = p  # runner uses 1-based indices
        self.gen_root = gen_root.resolve()
        self.evalset = evalset
        self.reasoner = reasoner
        self.tree_subdir = tree_subdir
        self.model_name = model_name
        self.model_name_s1 = model_name_s1 or model_name
        self.mode = mode
        self.w_s1 = workers_s1
        # Split S1D1 into S1D1a (search-only) + S1D1b (download). If the caller passes a
        # legacy ``workers_s1d1`` without per-half overrides, split it equally so default
        # CLI behavior stays sane: e.g. 24 → 12+12.
        if workers_s1d1_search is None and workers_s1d1_download is None:
            self.w_s1d1_search = max(1, workers_s1d1 // 2)
            self.w_s1d1_download = max(1, workers_s1d1 - self.w_s1d1_search)
        else:
            self.w_s1d1_search = int(workers_s1d1_search) if workers_s1d1_search is not None else max(1, workers_s1d1 // 2)
            self.w_s1d1_download = int(workers_s1d1_download) if workers_s1d1_download is not None else max(1, workers_s1d1 - self.w_s1d1_search)
        self.w_s1d1 = self.w_s1d1_search + self.w_s1d1_download  # back-compat aggregate
        self.w_s2 = workers_s2
        self.w_s3 = workers_s3
        self.hint_mode = bool(hint_mode)
        self.max_dl = int(max_downloaded_images_per_query)
        self.max_search = int(max_search_results)
        self.limit = limit
        self.only = set(only_row_indices or [])

        self.staging_dir = (staging_dir or (self.gen_root / "_staging" /
                            f"{evalset}_{reasoner}_v3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")).resolve()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.state_log_path = self.staging_dir / "_v3_row_state.jsonl"

        # Shared lazy clients. SearchClient is built per row in v2 (the constructor sets output_dir
        # later); we follow that and rebuild per row when needed.
        # FrontierModel: cheap to construct; create per-stage-call to avoid Session contention.

        # Per-row state cache (in-memory). The classifier (disk) is always re-run on resume.
        self.states: Dict[int, RowState] = {}

        # Counters
        self.counters: Dict[str, int] = {
            "s1_ok": 0, "s1d1a_ok": 0, "s1d1b_ok": 0, "s1d1_ok": 0, "s1d1_skipped": 0,
            "s2_ok": 0, "s3_ok": 0, "s4_written": 0,
            "stop_marker_written": 0, "moved_noop": 0, "failed": 0, "started_at": int(time.time()),
        }

    # ─── disk row dir ──────────────────────────────────────────────
    def _row_dir(self, idx: int) -> Path:
        if idx in self._row_dir_overrides:
            return self._row_dir_overrides[idx]
        return self.gen_root / self.tree_subdir / self.reasoner / f"eval_row_{idx:03d}"

    def _art_dir(self, idx: int) -> Path:
        return self._row_dir(idx) / "artifacts_files"

    # ─── state log (append-only JSONL, best-effort) ────────────────
    def _log_state(self, row_index: int, new_stage: str, reason: str = "", error: str = "") -> None:
        try:
            with self.state_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "row_index": row_index,
                    "stage": new_stage,
                    "reason": reason,
                    "error": error,
                    "ts": _utc_iso(),
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ─── classification ────────────────────────────────────────────
    def scan_all(self) -> List[RowState]:
        states: List[RowState] = []
        for idx in range(1, len(self.rows) + 1):
            if self.only and idx not in self.only:
                continue
            st = classify_row(idx, self._art_dir(idx), mode=self.mode)
            self.states[idx] = st
            states.append(st)
        return states

    # ─── stage workers ─────────────────────────────────────────────
    async def _stage_s1(self, idx: int, row: Dict[str, Any]) -> Tuple[bool, str]:
        art = self._art_dir(idx)
        row_dir = self._row_dir(idx)
        prompt = str(row.get("user_request") or row.get("user_prompt") or "").strip()
        if not prompt:
            return False, "empty_user_prompt"
        hint_ctx = None
        if self.hint_mode:
            ec = row.get("evaluation_context") if isinstance(row.get("evaluation_context"), dict) else {}
            hint_ctx = normalize_prompt_analysis_hint_context({
                "text_search_queries": row.get("text_search_queries") or ec.get("text_search_queries"),
                "why_text_search_needed": row.get("why_text_search_needed") or ec.get("why_text_search_needed"),
                "no_search_improvement_context": row.get("no_search_improvement_context") or ec.get("no_search_improvement_context"),
            })

        def _do_s1() -> Dict[str, Any]:
            fm = FrontierModel(model_name=self.model_name_s1)
            return fm.analyze_prompt(prompt, hint_ctx)

        try:
            res = await asyncio.to_thread(_do_s1)
        except Exception as e:
            self._log_state(idx, "S1_FAILED", error=f"{type(e).__name__}: {e}")
            return False, f"s1_exception:{e}"
        if not res.get("success"):
            self._log_state(idx, "S1_FAILED", error=str(res.get("error", "")))
            return False, f"s1_failed:{res.get('error')}"
        analysis = res["analysis"]
        writer = _writer_for_row(row_dir)
        writer.write_s1_analysis(
            analysis,
            res.get("raw_llm_input", {}),
            str(res.get("raw_response") or ""),
            prompt,
        )
        # Update state cache so S1D1 can reuse without re-reading disk.
        st = self.states.get(idx) or RowState(row_index=idx, art_path=art, stage=RowStage.S1_OK)
        st.has_s1 = True
        st.analysis = analysis
        # execution_search_plan is in analysis but classifier doesn't read it from s1
        # (we keep it in memory only — disk shape doesn't need it).
        plan = analysis.get("execution_search_plan") or []
        st.needs_image_s2 = any(str(s.get("search_type", "")).strip().lower() == "image" for s in plan)
        st.stage = RowStage.S1_OK
        self.states[idx] = st
        self.counters["s1_ok"] += 1
        self._log_state(idx, "S1_OK", reason=f"plan_len={len(plan)}")
        return True, "s1_ok"

    # ─── Plan rehydration shared by S1D1a / S1D1b / S2 / S3 on resume ───────────
    def _rehydrate_analysis(self, idx: int) -> Optional[Dict[str, Any]]:
        """Reconstruct the runtime analysis dict from on-disk s1_analysis.json.

        Returns None if s1 is missing or unparseable. Adds execution_search_plan
        via FrontierModel.build_execution_search_plan (no LLM call).
        """
        st = self.states[idx]
        # Only trust the cached analysis if it carries a runtime execution_search_plan.
        # classify_row caches the bare parsed_llm_output (knowledge_gaps/search_queries but
        # NO execution_search_plan), so trusting it here would skip plan-building entirely and
        # incorrectly route rows with critical/important gaps to "empty plan → text-only".
        # Falling through rebuilds the plan via build_execution_search_plan (knowledge_gaps fallback).
        if st.analysis is not None and "execution_search_plan" in st.analysis:
            return st.analysis
        art = self._art_dir(idx)
        p = art / "s1_analysis.json"
        if not p.is_file():
            return None
        try:
            s1 = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(s1.get("parsed_llm_output"), dict):
            return None
        parsed = s1["parsed_llm_output"]
        candidates = parsed.get("knowledge_gaps") if isinstance(parsed.get("knowledge_gaps"), list) else []
        limited = [c for c in candidates if isinstance(c, dict)
                   and str(c.get("severity", "")).strip().lower() in {"critical", "important"}]
        sq = parsed.get("search_queries") if isinstance(parsed.get("search_queries"), list) else []
        sq = [str(q).strip() for q in sq if str(q).strip()]
        fm = FrontierModel(model_name=self.model_name)
        analysis = {
            "analysis_reasoning": str(parsed.get("analysis_reasoning") or "").strip(),
            "visual_reference_candidates": candidates,
            "search_queries": sq,
            "search_justification": str(parsed.get("search_justification") or "").strip(),
            "knowledge_gaps": candidates,
            "needs_search": bool(parsed.get("needs_search")),
            "execution_candidates": limited,
        }
        analysis["execution_search_plan"] = fm.build_execution_search_plan(analysis, max_steps=3)
        st.analysis = analysis
        return analysis

    # ─── S1D1a (SERP only) — fast, lightweight, scale workers high ──────────────
    async def _stage_s1d1a_search(self, idx: int, row: Dict[str, Any]) -> Tuple[bool, str]:
        """Phase A: for each step in execution_search_plan, fetch SERP (text or image)
        WITHOUT downloading any images. Persist the result as ``s1d1_serp_pending.json``.

        On success transitions the row state to ``S1D1_SEARCHED``. A subsequent
        ``_stage_s1d1b_download`` call (in the S1D1b worker pool) reads the pending
        file, runs the slow image downloads, and writes the canonical
        ``s1d1_search_results.json`` (promoting state to S1D1_OK).
        """
        art = self._art_dir(idx)
        row_dir = self._row_dir(idx)
        st = self.states[idx]
        prompt = str(row.get("user_request") or row.get("user_prompt") or "").strip()

        analysis = self._rehydrate_analysis(idx)
        if analysis is None:
            return False, "s1d1a_missing_s1_for_resume"
        plan = analysis.get("execution_search_plan") or []

        # Empty plan → write skip log (same path as v1 monolithic _stage_s1d1); no
        # serp_pending or canonical s1d1 file. Route to next stage based on mode.
        if not plan:
            writer = _writer_for_row(row_dir)
            writer.write_s1d1_materialization_log(
                outcome="skipped",
                reason_code=("empty_execution_search_plan_reasoner_stop_stub" if self.mode == "stop_s2"
                             else "empty_execution_search_plan_refine_without_serp"),
                detail={"branch": "v3_stage_s1d1a_empty_plan", "mode": self.mode,
                        "execution_search_plan_len": 0},
            )
            st.has_s1d1_skip = True
            st.stage = RowStage.S1D1_SKIPPED
            self.counters["s1d1_skipped"] += 1
            self._log_state(idx, "S1D1_SKIPPED", reason="empty_plan")
            return True, "s1d1a_skipped_empty"

        sc = SearchClient()
        sc.output_dir = row_dir  # not used for SERP-only, but required to be set for later download
        conversation = ConversationManager()
        logger = _setup_row_logger(f"v3.s1d1a.row{idx:03d}", row_dir)

        def _do_step(slot_idx: int, step: Dict[str, Any], image_slot: int) -> Tuple[str, List[Dict[str, Any]], int]:
            q = str(step.get("query", "")).strip()
            stp = str(step.get("search_type", "web")).strip().lower()
            if not q:
                return stp, [], image_slot
            if stp == "web":
                pack = run_text_search_for_query(
                    q, search_client=sc, max_results=self.max_search,
                    query_id=slot_idx, gap_reasoning="",
                    conversation=conversation, stage_hooks=False,
                )
                return stp, list(pack.get("serp_rows") or []), image_slot
            if stp == "image":
                pack = run_image_search_only_for_query(
                    q, search_client=sc, max_results=self.max_search,
                    query_id=slot_idx, query_index=image_slot,
                    conversation=conversation, logger=logger, stage_hooks=False,
                )
                return stp, list(pack.get("serp_rows") or []), image_slot + 1
            return stp, [], image_slot

        try:
            # SERP rows organised by plan-step key (same convention as v2 s1d1_search_results)
            serp_by_key: Dict[str, List[Dict[str, Any]]] = {}
            image_slot = 0
            for slot_idx, step in enumerate(plan):
                stp, rows_part, image_slot = await asyncio.to_thread(_do_step, slot_idx, step, image_slot)
                key = plan_step_key(slot_idx, stp)
                serp_by_key[key] = rows_part
            pending_doc = {
                "schema_version": 1,
                "stage": "s1d1_serp_pending",
                "created_at": _utc_iso(),
                "execution_search_plan": plan,
                "knowledge_gaps": analysis.get("knowledge_gaps") or [],
                "search_results": serp_by_key,
            }
            art.mkdir(parents=True, exist_ok=True)
            (art / "s1d1_serp_pending.json").write_text(
                json.dumps(pending_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        except Exception as e:
            (art / "s1d1a_failure.json").write_text(
                json.dumps({"error": str(e), "traceback": traceback.format_exc(), "ts": _utc_iso()}, indent=2),
                encoding="utf-8",
            )
            self._log_state(idx, "S1D1A_FAILED", error=str(e))
            return False, f"s1d1a_exception:{e}"

        st.has_s1d1_pending = True
        st.s1d1_pending_doc = pending_doc
        st.stage = RowStage.S1D1_SEARCHED
        self.counters["s1d1a_ok"] += 1
        self._log_state(idx, "S1D1_SEARCHED", reason=f"plan_len={len(plan)}")
        return True, "s1d1a_ok"

    # ─── S1D1b (downloads only) — slow, network-bound, scale workers high ───────
    async def _stage_s1d1b_download(self, idx: int, row: Dict[str, Any]) -> Tuple[bool, str]:
        """Phase B: read s1d1_serp_pending.json, download images for each image-type
        plan step (text steps pass through unchanged), then write the canonical
        ``s1d1_search_results.json`` via V2ArtifactsWriter so all downstream code is
        unchanged.
        """
        art = self._art_dir(idx)
        row_dir = self._row_dir(idx)
        st = self.states[idx]

        # Rehydrate pending doc from disk if needed (resume).
        if st.s1d1_pending_doc is None:
            p = art / "s1d1_serp_pending.json"
            if not p.is_file():
                return False, "s1d1b_missing_pending"
            try:
                st.s1d1_pending_doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                return False, "s1d1b_unreadable_pending"

        pending = st.s1d1_pending_doc or {}
        plan = pending.get("execution_search_plan") or []
        serp_by_key: Dict[str, List[Dict[str, Any]]] = dict(pending.get("search_results") or {})

        analysis = self._rehydrate_analysis(idx)
        if analysis is None:
            return False, "s1d1b_missing_analysis"

        sc = SearchClient()
        sc.output_dir = row_dir
        logger = _setup_row_logger(f"v3.s1d1b.row{idx:03d}", row_dir)

        def _download_step(slot_idx: int, step: Dict[str, Any], image_slot: int) -> Tuple[str, List[Dict[str, Any]], int]:
            stp = str(step.get("search_type", "web")).strip().lower()
            q = str(step.get("query", "")).strip()
            key = plan_step_key(slot_idx, stp)
            existing = serp_by_key.get(key) or []
            if stp != "image":
                return stp, existing, image_slot  # text step passes through unchanged
            pack = run_image_download_for_serp_rows(
                q, existing, search_client=sc, query_id=slot_idx, query_index=image_slot,
                max_downloaded_images=self.max_dl, logger=logger,
            )
            return stp, list(pack.get("serp_rows") or []), image_slot + 1

        # Read s1 from disk for writer rehydration
        s1_disk = None
        s1p = art / "s1_analysis.json"
        if s1p.is_file():
            try:
                s1_disk = json.loads(s1p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                s1_disk = None
        writer = _writer_for_row(row_dir, s1_doc=s1_disk)

        try:
            # Run sequentially over plan steps but each step's _download_images_parallel is
            # already internally parallel; running plan steps in parallel here would race
            # on per-query filename slot (q{query_id}_refN.jpg). Keep per-row plan steps
            # serial; cross-row parallelism comes from the worker pool size.
            search_results: List[Dict[str, Any]] = []
            image_slot = 0
            for slot_idx, step in enumerate(plan):
                stp, rows_part, image_slot = await asyncio.to_thread(
                    _download_step, slot_idx, step, image_slot,
                )
                search_results.extend(rows_part)
            writer.write_s1d1_search_results(search_results, plan, analysis)
        except Exception as e:
            (art / "s1d1b_failure.json").write_text(
                json.dumps({"error": str(e), "traceback": traceback.format_exc(), "ts": _utc_iso()}, indent=2),
                encoding="utf-8",
            )
            self._log_state(idx, "S1D1B_FAILED", error=str(e))
            return False, f"s1d1b_exception:{e}"

        # Successful: leave s1d1_serp_pending.json on disk for audit (cheap, kb-scale).
        st.has_s1d1 = True
        st.s1d1_doc = writer.search_results_data
        st.stage = RowStage.S1D1_OK
        self.counters["s1d1b_ok"] += 1
        self.counters["s1d1_ok"] += 1
        self._log_state(idx, "S1D1_OK", reason=f"download_done,hits={len(search_results)}")
        return True, "s1d1b_ok"

    # ─── Legacy monolithic S1D1 (unused by new flow; kept for back-compat / tests) ──
    async def _stage_s1d1(self, idx: int, row: Dict[str, Any]) -> Tuple[bool, str]:
        """DEPRECATED — combined SERP+download. Kept for back-compat; the new
        flow uses ``_stage_s1d1a_search`` + ``_stage_s1d1b_download``."""
        art = self._art_dir(idx)
        row_dir = self._row_dir(idx)
        st = self.states[idx]
        prompt = str(row.get("user_request") or row.get("user_prompt") or "").strip()
        # Need analysis. If resumed from disk we may not have it in memory — re-read s1.
        if st.analysis is None:
            s1 = None
            p = art / "s1_analysis.json"
            if p.is_file():
                try:
                    s1 = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                except (OSError, json.JSONDecodeError):
                    s1 = None
            # Re-derive analysis: write_s1_analysis stored parsed_llm_output. For execution plan
            # we'd need a runtime build. Easiest path: re-run S1 to regenerate analysis +
            # execution_search_plan. Cheaper alternative: re-derive plan from parsed_llm_output.
            # We pick re-derive to avoid an LLM call.
            if s1 is None or not isinstance(s1.get("parsed_llm_output"), dict):
                return False, "s1d1_missing_s1_for_resume"
            fm = FrontierModel(model_name=self.model_name)
            parsed = s1["parsed_llm_output"]
            # Reconstruct minimum analysis shape used by handlers + writer
            candidates = parsed.get("knowledge_gaps") if isinstance(parsed.get("knowledge_gaps"), list) else []
            limited = [c for c in candidates if isinstance(c, dict)
                       and str(c.get("severity", "")).strip().lower() in {"critical", "important"}]
            sq = parsed.get("search_queries") if isinstance(parsed.get("search_queries"), list) else []
            sq = [str(q).strip() for q in sq if str(q).strip()]
            analysis = {
                "analysis_reasoning": str(parsed.get("analysis_reasoning") or "").strip(),
                "visual_reference_candidates": candidates,
                "search_queries": sq,
                "search_justification": str(parsed.get("search_justification") or "").strip(),
                "knowledge_gaps": candidates,
                "needs_search": bool(parsed.get("needs_search")),
                "execution_candidates": limited,
            }
            analysis["execution_search_plan"] = fm.build_execution_search_plan(analysis, max_steps=3)
            st.analysis = analysis

        analysis = st.analysis
        plan = analysis.get("execution_search_plan") or []

        # Rehydrate writer with s1 (needed for materialization log to count knowledge_gaps).
        s1_doc = {"parsed_llm_output": analysis.get("knowledge_gaps") and {"knowledge_gaps": analysis["knowledge_gaps"]} or {}}
        # Simpler: read the on-disk s1
        s1_disk = None
        s1p = art / "s1_analysis.json"
        if s1p.is_file():
            try:
                s1_disk = json.loads(s1p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                s1_disk = None
        writer = _writer_for_row(row_dir, s1_doc=s1_disk)

        # Empty plan → log skip and finish. Routing to S3/STOP handled by enqueue.
        if not plan:
            writer.write_s1d1_materialization_log(
                outcome="skipped",
                reason_code=("empty_execution_search_plan_reasoner_stop_stub" if self.mode == "stop_s2"
                             else "empty_execution_search_plan_refine_without_serp"),
                detail={"branch": "v3_stage_s1d1_empty_plan", "mode": self.mode,
                        "execution_search_plan_len": 0},
            )
            st.has_s1d1_skip = True
            st.stage = RowStage.S1D1_SKIPPED
            self.counters["s1d1_skipped"] += 1
            self._log_state(idx, "S1D1_SKIPPED", reason="empty_plan")
            return True, "s1d1_skipped_empty"

        # Build a SearchClient pinned to this row's dir (image downloads land under it).
        sc = SearchClient()
        sc.output_dir = row_dir
        conversation = ConversationManager()
        logger = _setup_row_logger(f"v3.s1d1.row{idx:03d}", row_dir)

        def _do_step(slot_idx: int, step: Dict[str, Any], image_slot: int) -> Tuple[List[Dict[str, Any]], int]:
            q = str(step.get("query", "")).strip()
            stp = str(step.get("search_type", "web")).strip().lower()
            if not q:
                return [], image_slot
            if stp == "web":
                fm_local = FrontierModel(model_name=self.model_name)
                pack = run_text_search_for_query(
                    q, search_client=sc, max_results=self.max_search,
                    query_id=slot_idx, gap_reasoning="",
                    conversation=conversation, stage_hooks=False,
                )
                return list(pack.get("serp_rows") or []), image_slot
            if stp == "image":
                # NB: skip_reference_selection=True — selection deferred to S2 pool.
                fm_local = FrontierModel(model_name=self.model_name)
                pack = run_image_search_for_query(
                    q, search_client=sc, frontier_model=fm_local,
                    user_prompt=prompt, analysis=analysis,
                    max_results=self.max_search, query_id=slot_idx, query_index=image_slot,
                    max_downloaded_images=self.max_dl,
                    conversation=conversation, logger=logger, stage_hooks=False,
                    skip_reference_selection=True,
                )
                return list(pack.get("serp_rows") or []), image_slot + 1
            return [], image_slot

        try:
            search_results: List[Dict[str, Any]] = []
            image_slot = 0
            for slot_idx, step in enumerate(plan):
                rows_part, image_slot = await asyncio.to_thread(_do_step, slot_idx, step, image_slot)
                search_results.extend(rows_part)
            writer.write_s1d1_search_results(search_results, plan, analysis)
        except Exception as e:
            tb = traceback.format_exc()
            (art / "s1d1_failure.json").write_text(
                json.dumps({"error": str(e), "traceback": tb, "ts": _utc_iso()}, indent=2),
                encoding="utf-8",
            )
            self._log_state(idx, "S1D1_FAILED", error=str(e))
            return False, f"s1d1_exception:{e}"

        st.has_s1d1 = True
        st.s1d1_doc = writer.search_results_data
        st.stage = RowStage.S1D1_OK
        self.counters["s1d1_ok"] += 1
        self._log_state(idx, "S1D1_OK", reason=f"plan_len={len(plan)},hits={len(search_results)}")
        return True, "s1d1_ok"

    async def _stage_s2(self, idx: int, row: Dict[str, Any]) -> Tuple[bool, str]:
        art = self._art_dir(idx)
        row_dir = self._row_dir(idx)
        st = self.states[idx]
        prompt = str(row.get("user_request") or row.get("user_prompt") or "").strip()
        if st.s1d1_doc is None:
            s1d1p = art / "s1d1_search_results.json"
            if not s1d1p.is_file():
                return False, "s2_missing_s1d1"
            try:
                st.s1d1_doc = json.loads(s1d1p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                return False, "s2_unreadable_s1d1"
        if st.analysis is None:
            # Re-derive from s1 (cheap; same logic as S1D1)
            s1p = art / "s1_analysis.json"
            try:
                s1_disk = json.loads(s1p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                return False, "s2_unreadable_s1"
            parsed = s1_disk.get("parsed_llm_output") or {}
            candidates = parsed.get("knowledge_gaps") if isinstance(parsed.get("knowledge_gaps"), list) else []
            sq = parsed.get("search_queries") if isinstance(parsed.get("search_queries"), list) else []
            sq = [str(q).strip() for q in sq if str(q).strip()]
            limited = [c for c in candidates if isinstance(c, dict)
                       and str(c.get("severity", "")).strip().lower() in {"critical", "important"}]
            fm = FrontierModel(model_name=self.model_name)
            analysis = {
                "analysis_reasoning": str(parsed.get("analysis_reasoning") or "").strip(),
                "visual_reference_candidates": candidates,
                "search_queries": sq,
                "search_justification": str(parsed.get("search_justification") or "").strip(),
                "knowledge_gaps": candidates,
                "needs_search": bool(parsed.get("needs_search")),
                "execution_candidates": limited,
            }
            analysis["execution_search_plan"] = fm.build_execution_search_plan(analysis, max_steps=3)
            st.analysis = analysis

        analysis = st.analysis
        plan = analysis.get("execution_search_plan") or []
        s1d1_doc = st.s1d1_doc
        s1_disk = None
        s1p = art / "s1_analysis.json"
        if s1p.is_file():
            try:
                s1_disk = json.loads(s1p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                s1_disk = None
        writer = _writer_for_row(row_dir, s1_doc=s1_disk, s1d1_doc=s1d1_doc)
        writer.reset_s2_slots()
        conversation = ConversationManager()
        logger = _setup_row_logger(f"v3.s2.row{idx:03d}", row_dir)

        image_slot = 0

        def _do_image_step(slot_idx: int, step: Dict[str, Any], image_slot_in: int) -> Optional[Dict[str, Any]]:
            q = str(step.get("query", "")).strip()
            stp = str(step.get("search_type", "web")).strip().lower()
            if stp != "image" or not q:
                return None
            bucket_rows_raw = (s1d1_doc.get("search_results") or {}).get(plan_step_key(slot_idx, "image")) or []
            bucket = [r for r in bucket_rows_raw if isinstance(r, dict)]
            fm_local = FrontierModel(model_name=self.model_name)
            pack = run_image_selection_from_s1d1_bucket(
                q, bucket, frontier_model=fm_local, user_prompt=prompt, analysis=analysis,
                query_index=image_slot_in, conversation=conversation, logger=logger, stage_hooks=False,
                skip_reference_selection=False,
            )
            return pack

        try:
            for slot_idx, step in enumerate(plan):
                stp = str(step.get("search_type", "web")).strip().lower()
                if stp != "image":
                    continue
                pack = await asyncio.to_thread(_do_image_step, slot_idx, step, image_slot)
                pk = plan_step_key(slot_idx, "image")
                if pack and pack.get("selection"):
                    writer.append_s2_slot(pack["selection"], pk)
                else:
                    writer.append_s2_slot({
                        "query": str(step.get("query") or ""),
                        "query_index": image_slot,
                        "selection_prompt": "", "selection_response": "",
                        "selected_image": None, "selection_reasoning": "",
                        "identified_knowledge_gaps": "", "visual_description": "",
                        "candidate_image_mappings": [], "retry_count": 0,
                        "selection_input_source": "", "selection_input_size_bytes": None,
                        "selection_skipped": True,
                    }, pk)
                image_slot += 1
            writer.flush_s2_reference_selection()
        except Exception as e:
            (art / "s2_failure.json").write_text(
                json.dumps({"error": str(e), "traceback": traceback.format_exc(), "ts": _utc_iso()}, indent=2),
                encoding="utf-8",
            )
            self._log_state(idx, "S2_FAILED", error=str(e))
            return False, f"s2_exception:{e}"

        st.has_s2 = True
        st.s2_doc = writer.s2_document
        st.stage = RowStage.S2_OK
        self.counters["s2_ok"] += 1
        self._log_state(idx, "S2_OK")
        return True, "s2_ok"

    async def _stage_s3_s4(self, idx: int, row: Dict[str, Any]) -> Tuple[bool, str]:
        art = self._art_dir(idx)
        row_dir = self._row_dir(idx)
        st = self.states[idx]
        prompt = str(row.get("user_request") or row.get("user_prompt") or "").strip()

        # Rehydrate analysis + s1 + s2 + s1d1 from disk if needed (resume).
        s1_disk = None
        s1p = art / "s1_analysis.json"
        if s1p.is_file():
            try:
                s1_disk = json.loads(s1p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                s1_disk = None
        if s1_disk is None:
            return False, "s3_missing_s1"
        if st.analysis is None:
            parsed = s1_disk.get("parsed_llm_output") or {}
            candidates = parsed.get("knowledge_gaps") if isinstance(parsed.get("knowledge_gaps"), list) else []
            fm = FrontierModel(model_name=self.model_name)
            limited = [c for c in candidates if isinstance(c, dict)
                       and str(c.get("severity", "")).strip().lower() in {"critical", "important"}]
            sq = parsed.get("search_queries") if isinstance(parsed.get("search_queries"), list) else []
            sq = [str(q).strip() for q in sq if str(q).strip()]
            analysis = {
                "analysis_reasoning": str(parsed.get("analysis_reasoning") or "").strip(),
                "visual_reference_candidates": candidates,
                "search_queries": sq,
                "search_justification": str(parsed.get("search_justification") or "").strip(),
                "knowledge_gaps": candidates,
                "needs_search": bool(parsed.get("needs_search")),
                "execution_candidates": limited,
            }
            analysis["execution_search_plan"] = fm.build_execution_search_plan(analysis, max_steps=3)
            st.analysis = analysis
        analysis = st.analysis

        s2_disk = None
        s2p = art / "s2_reference_selection.json"
        if s2p.is_file():
            try:
                s2_disk = json.loads(s2p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                s2_disk = None
        reference_images = _reconstruct_reference_images_from_s2(s2_disk) if isinstance(s2_disk, dict) else []

        s1d1_disk = None
        s1d1p = art / "s1d1_search_results.json"
        if s1d1p.is_file():
            try:
                s1d1_disk = json.loads(s1d1p.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                s1d1_disk = None

        # Flat web search_results list for refine_prompt
        web_rows: List[Dict[str, Any]] = []
        if isinstance(s1d1_disk, dict):
            for key, hits in (s1d1_disk.get("search_results") or {}).items():
                if not key.startswith("web-"):
                    continue
                for h in hits if isinstance(hits, list) else []:
                    if not isinstance(h, dict):
                        continue
                    row_h = dict(h)
                    row_h.setdefault("type", "web")
                    row_h.setdefault("query", row_h.get("search_query"))
                    web_rows.append(row_h)

        def _do_s3() -> Dict[str, Any]:
            fm = FrontierModel(model_name=self.model_name)
            if reference_images:
                res = fm.refine_prompt_with_multiple_references(
                    prompt, analysis, reference_images, web_results=web_rows,
                )
                if not res.get("success") and len(reference_images) >= 1:
                    primary = reference_images[0]
                    res = fm.refine_prompt_with_reference(
                        prompt, analysis,
                        primary.get("selected_image") or {},
                        primary.get("visual_description") or "",
                        web_results=web_rows,
                    )
                return res
            return fm.refine_prompt(user_prompt=prompt, analysis=analysis,
                                    search_results=web_rows or None)

        try:
            ref_res = await asyncio.to_thread(_do_s3)
        except Exception as e:
            (art / "s3_failure.json").write_text(
                json.dumps({"error": str(e), "traceback": traceback.format_exc(), "ts": _utc_iso()}, indent=2),
                encoding="utf-8",
            )
            self._log_state(idx, "S3_FAILED", error=str(e))
            return False, f"s3_exception:{e}"

        raw_resp = str(ref_res.get("raw_response", "") or ref_res.get("raw_llm_response", "") or "")
        parsed_out = ref_res.get("parsed_llm_output") or ref_res.get("parsed_output") or {}
        if (raw_resp.startswith("Unexpected") or "'code': -" in raw_resp
                or (isinstance(parsed_out, dict) and "_unparsed_text" in parsed_out)
                or any(pat in raw_resp for pat in ("429 ", "500 ", "502 ", "503 ",
                                                    "Client Error", "Server Error",
                                                    "timeout", "ConnectionError", "_error"))):
            self._log_state(idx, "S3_LLM_ERROR", error=raw_resp[:200])
            return False, "s3_llm_error"

        if not ref_res.get("success"):
            (art / "s3_failure.json").write_text(
                json.dumps({"error": ref_res.get("error"), "ts": _utc_iso()}, indent=2), encoding="utf-8",
            )
            self._log_state(idx, "S3_FAILED", error=str(ref_res.get("error")))
            return False, f"s3_failed:{ref_res.get('error')}"

        writer = _writer_for_row(row_dir, s1_doc=s1_disk, s1d1_doc=s1d1_disk, s2_doc=s2_disk)
        # Mirror v2's _v2_write_refined_and_manifest payload normalization.
        payload = dict(ref_res)
        rp = str(ref_res.get("refined_prompt") or "").strip()
        payload.setdefault("refined_prompt_final", rp)
        if "parsed_refinement_json" not in payload:
            payload["parsed_refinement_json"] = {
                "refined_prompt": rp,
                "selected_reference_indices": ref_res.get("selected_reference_indices", []),
                "borrow_from_references": ref_res.get("borrow_from_references", []),
            }
        # _pack_reference_images_for_s3 equivalent inline:
        packed: List[Dict[str, Any]] = []
        for pack in reference_images:
            sel = pack.get("selected_image") or {}
            packed.append({
                "query": str(pack.get("query") or ""),
                "title": str(sel.get("title") or ""),
                "url": str(sel.get("url") or sel.get("imageUrl") or ""),
                "local_path": str(sel.get("local_path") or ""),
                "selection_reasoning": str(pack.get("selection_reasoning") or ""),
            })
        writer.write_s3_refined_prompt(payload, prompt, packed)
        writer.write_s4_generation_manifest(
            prompt,
            evalset=self.evalset,
            reasoner=self.reasoner,
            row_index=idx - 1,  # 0-based per v2 convention; try_move uses +1 to get dest
        )
        self.counters["s3_ok"] += 1
        self.counters["s4_written"] += 1
        self._log_state(idx, "S3_S4_DONE")
        st.has_s3 = True
        st.has_s4 = True
        st.stage = RowStage.MOVED_NOOP
        return True, "s3_s4_done"

    async def _stage_stop_marker(self, idx: int, row: Dict[str, Any]) -> Tuple[bool, str]:
        art = self._art_dir(idx)
        row_dir = self._row_dir(idx)
        st = self.states[idx]
        prompt = str(row.get("user_request") or row.get("user_prompt") or "").strip()
        marker = {
            "schema_version": 1,
            "artifact": "reasoner_stop_before_s2",
            "stop_mode": "before_reference_selection",
            "created_at": _utc_iso(),
            "search_lane_ran": bool(st.has_s1d1),
            "user_prompt_raw": prompt,
            "row_identity": {
                "eval_row_index": idx - 1,
                "evalset": self.evalset,
                "reasoner": self.reasoner,
            },
        }
        art.mkdir(parents=True, exist_ok=True)
        for name in ("s3_refined_prompt.json", "s4_generation_manifest.json"):
            p = art / name
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
        (art / "reasoner_stop_before_s2.json").write_text(
            json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        st.has_stop_marker = True
        st.stage = RowStage.MOVED_NOOP
        self.counters["stop_marker_written"] += 1
        self._log_state(idx, "STOP_MARKER_WRITTEN")
        return True, "stop_marker_written"

    async def _finalize(self, idx: int, row: Dict[str, Any]) -> None:
        """Side effects after a row reaches a terminal-good state: refined_prompt.txt sync,
        ROW_MANIFEST.json, reasoner_inference_meta.json (best-effort)."""
        row_dir = self._row_dir(idx)
        try:
            finalize_row(
                row_dir,
                dataset_name=self.evalset,
                lane_name=self.reasoner,
                row_index=idx,
                row=row,
                model_name=self.model_name,
                model_name_s1=self.model_name_s1,
            )
        except Exception as e:
            self._log_state(idx, "FINALIZE_WARN", error=f"{type(e).__name__}: {e}")

    # ─── orchestration ─────────────────────────────────────────────
    async def run(self) -> Dict[str, Any]:
        states = self.scan_all()
        # Build initial queues by walking bootstrap routing per state.
        q_s1: asyncio.Queue = asyncio.Queue()
        q_s1d1a: asyncio.Queue = asyncio.Queue()   # SERP-only (cheap, high concurrency)
        q_s1d1b: asyncio.Queue = asyncio.Queue()   # downloads (slow, network-bound, high concurrency)
        q_s2: asyncio.Queue = asyncio.Queue()
        q_s3: asyncio.Queue = asyncio.Queue()
        q_stop: asyncio.Queue = asyncio.Queue()
        q_finalize: asyncio.Queue = asyncio.Queue()

        enqueued = 0
        for st in states:
            if self.limit is not None and enqueued >= self.limit and st.stage != RowStage.MOVED_NOOP:
                continue
            idx = st.row_index
            if st.stage == RowStage.MOVED_NOOP:
                self.counters["moved_noop"] += 1
                continue
            if st.stage == RowStage.STOP_MARKER_WRITTEN:
                self.counters["moved_noop"] += 1
                continue
            if st.stage in (RowStage.FAILED_PINNED, RowStage.CORRUPT):
                continue
            if st.stage == RowStage.NEW:
                await q_s1.put(idx)
                enqueued += 1
            elif st.stage == RowStage.S1_OK:
                # S1 done; route to S1D1a (SERP-only first).
                await q_s1d1a.put(idx)
                enqueued += 1
            elif st.stage == RowStage.S1D1_SEARCHED:
                # SERP done from a prior run; route directly to S1D1b (downloader).
                await q_s1d1b.put(idx)
                enqueued += 1
            elif st.stage == RowStage.S1D1_OK:
                if self.mode == "stop_s2":
                    await q_stop.put(idx)
                elif st.needs_image_s2:
                    await q_s2.put(idx)
                else:
                    await q_s3.put(idx)
                enqueued += 1
            elif st.stage == RowStage.S1D1_SKIPPED:
                if self.mode == "stop_s2":
                    await q_stop.put(idx)
                else:
                    await q_s3.put(idx)
                enqueued += 1
            elif st.stage == RowStage.S2_OK:
                if self.mode == "stop_s2":
                    await q_stop.put(idx)  # downgraded — write marker (mode mismatch §5.4 T7)
                else:
                    await q_s3.put(idx)
                enqueued += 1
            elif st.stage == RowStage.S3_OK:
                # Only s4 left; s4 has no LLM cost. Route through S3 worker which will detect
                # has_s3 already and just write s4. Simpler: re-run s3 anyway. We choose:
                # write only s4 here.
                await q_s3.put(idx)  # handler is idempotent — will rewrite s3+s4
                enqueued += 1
            elif st.stage == RowStage.MOVE_READY:
                if self.mode == "stop_s2":
                    await q_stop.put(idx)
                else:
                    await q_finalize.put(idx)
                enqueued += 1

        sem_s1 = asyncio.Semaphore(self.w_s1)
        sem_s1d1a = asyncio.Semaphore(self.w_s1d1_search)
        sem_s1d1b = asyncio.Semaphore(self.w_s1d1_download)
        sem_s2 = asyncio.Semaphore(self.w_s2)
        sem_s3 = asyncio.Semaphore(self.w_s3)

        async def _worker(q_in: asyncio.Queue, q_out: Optional[asyncio.Queue], sem: asyncio.Semaphore,
                          stage_fn, name: str, terminal_routing=None):
            while True:
                idx = await q_in.get()
                if idx is None:
                    q_in.task_done()
                    return
                row = self.rows[idx - 1]
                async with sem:
                    try:
                        ok, reason = await stage_fn(idx, row)
                    except Exception as e:
                        ok = False
                        reason = f"{name}_unhandled:{type(e).__name__}:{e}"
                        self._log_state(idx, f"{name}_FAILED", error=reason)
                if ok and terminal_routing is not None:
                    next_q = terminal_routing(self.states[idx])
                    if next_q is not None:
                        await next_q.put(idx)
                if not ok:
                    self.counters["failed"] += 1
                q_in.task_done()

        def _route_after_s1(st: RowState) -> Optional[asyncio.Queue]:
            # S1 always feeds into S1D1a (SERP-only) — even in stop_s2 mode, since the
            # SERP-only stage may detect an empty plan and route directly to STOP via
            # _route_after_s1d1a's skip branch.
            return q_s1d1a

        def _route_after_s1d1a(st: RowState) -> Optional[asyncio.Queue]:
            # SERP-only finished. If plan was empty, _stage_s1d1a_search set S1D1_SKIPPED
            # already → route per mode like the legacy s1d1_skip path. Otherwise → S1D1b
            # downloader.
            if st.has_s1d1_skip and not st.has_s1d1_pending:
                if self.mode == "stop_s2":
                    return q_stop
                return q_s3
            return q_s1d1b

        def _route_after_s1d1b(st: RowState) -> Optional[asyncio.Queue]:
            # Downloads done → same routing as legacy after-S1D1.
            if self.mode == "stop_s2":
                return q_stop
            if st.has_s1d1_skip and not st.has_s1d1:
                return q_s3
            if st.needs_image_s2:
                return q_s2
            return q_s3

        def _route_after_s2(st: RowState) -> Optional[asyncio.Queue]:
            return q_s3

        def _route_after_s3(st: RowState) -> Optional[asyncio.Queue]:
            return q_finalize

        def _route_after_stop(st: RowState) -> Optional[asyncio.Queue]:
            return q_finalize

        # Spawn worker tasks. We use one worker coroutine per slot; the semaphore is somewhat
        # redundant (n workers = n concurrent) but keeps the option open for future spawn-on-demand.
        tasks: List[asyncio.Task] = []
        for _ in range(self.w_s1):
            tasks.append(asyncio.create_task(_worker(q_s1, None, sem_s1, self._stage_s1, "S1", _route_after_s1)))
        for _ in range(self.w_s1d1_search):
            tasks.append(asyncio.create_task(_worker(q_s1d1a, None, sem_s1d1a, self._stage_s1d1a_search, "S1D1A", _route_after_s1d1a)))
        for _ in range(self.w_s1d1_download):
            tasks.append(asyncio.create_task(_worker(q_s1d1b, None, sem_s1d1b, self._stage_s1d1b_download, "S1D1B", _route_after_s1d1b)))
        for _ in range(self.w_s2):
            tasks.append(asyncio.create_task(_worker(q_s2, None, sem_s2, self._stage_s2, "S2", _route_after_s2)))
        for _ in range(self.w_s3):
            tasks.append(asyncio.create_task(_worker(q_s3, None, sem_s3, self._stage_s3_s4, "S3", _route_after_s3)))
        # STOP marker pool can be small (file write only)
        for _ in range(4):
            tasks.append(asyncio.create_task(_worker(q_stop, None, asyncio.Semaphore(4), self._stage_stop_marker, "STOP", _route_after_stop)))

        # Finalize coroutine (single, since shutil ops are cheap)
        async def _finalize_loop():
            while True:
                idx = await q_finalize.get()
                if idx is None:
                    q_finalize.task_done()
                    return
                row = self.rows[idx - 1]
                await self._finalize(idx, row)
                q_finalize.task_done()
        finalize_task = asyncio.create_task(_finalize_loop())

        # Wait for primary queues to drain in order.
        for q in (q_s1, q_s1d1a, q_s1d1b, q_s2, q_s3, q_stop):
            await q.join()
        await q_finalize.join()

        # Send sentinels to stop workers.
        for q, count in (
            (q_s1, self.w_s1),
            (q_s1d1a, self.w_s1d1_search),
            (q_s1d1b, self.w_s1d1_download),
            (q_s2, self.w_s2), (q_s3, self.w_s3),
            (q_stop, 4),
        ):
            for _ in range(count):
                await q.put(None)
        await q_finalize.put(None)
        await asyncio.gather(*tasks, finalize_task, return_exceptions=True)

        elapsed = int(time.time()) - self.counters["started_at"]
        return {
            "evalset": self.evalset,
            "reasoner": self.reasoner,
            "mode": self.mode,
            "total_rows": len(self.rows),
            "by_resume_state": histogram(list(self.states.values())),
            "enqueue_plan": enqueue_plan(list(self.states.values()), mode=self.mode),
            "counters": dict(self.counters),
            "elapsed_seconds": elapsed,
            "staging_dir": str(self.staging_dir),
        }
