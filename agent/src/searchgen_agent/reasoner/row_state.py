"""V3 stagewise pipeline — per-row state classifier.

This module is a **pure**
disk-shape classifier: it never writes, deletes, or moves anything. The classifier is the
source of truth for resume routing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class RowStage(str, Enum):
    """Bootstrap state for a row, derived from on-disk artifacts.

    Routing (see ``v3_stagewise_runner._enqueue_bootstrap``):
        NEW                    → S1 queue
        S1_OK                  → S1D1 queue
        S1D1_OK                → S2 queue (if any image steps) else S3 queue
        S1D1_SKIPPED           → S3-noref queue OR STOP-marker (per --mode)
        S2_OK                  → S3 queue
        S3_OK                  → S4 write (cheap, in MOVE coro)
        MOVE_READY             → MOVE queue
        STOP_MARKER_WRITTEN    → MOVE queue
        MOVED_NOOP             → terminal (skip)
        FAILED_PINNED          → terminal unless --retry-failed
        CORRUPT                → terminal unless --repair-corrupt
    """

    NEW = "NEW"
    S1_OK = "S1_OK"
    S1D1_SEARCHED = "S1D1_SEARCHED"  # serp_pending.json written by S1D1a; downloads not yet done
    S1D1_OK = "S1D1_OK"
    S1D1_SKIPPED = "S1D1_SKIPPED"
    S2_OK = "S2_OK"
    S3_OK = "S3_OK"
    MOVE_READY = "MOVE_READY"
    STOP_MARKER_WRITTEN = "STOP_MARKER_WRITTEN"
    MOVED_NOOP = "MOVED_NOOP"
    FAILED_PINNED = "FAILED_PINNED"
    CORRUPT = "CORRUPT"


SKIP_REASON_CODES = {
    # Canonical S1D1 materialization outcomes.
    "augmented_lane_skipped_after_baseline",
    "reasoner_only_no_augmented_search_lane",
    "empty_execution_search_plan_refine_without_serp",
    "empty_execution_search_plan_reasoner_stop_stub",
}


@dataclass
class RowState:
    """Disk-derived classification of one eval row."""

    row_index: int  # 1-based
    art_path: Path  # absolute path to .../eval_row_NNN/artifacts_files
    stage: RowStage
    has_s1: bool = False
    has_s1d1_pending: bool = False  # s1d1_serp_pending.json present (S1D1a done, S1D1b pending)
    has_s1d1: bool = False
    has_s1d1_skip: bool = False
    has_s2: bool = False
    has_s3: bool = False
    has_s4: bool = False
    has_stop_marker: bool = False
    needs_image_s2: bool = False  # True iff any execution_search_plan step has search_type==image
    analysis: Optional[Dict[str, Any]] = None  # parsed s1.analysis when has_s1
    s1d1_pending_doc: Optional[Dict[str, Any]] = None  # parsed s1d1_serp_pending.json when has_s1d1_pending
    s1d1_doc: Optional[Dict[str, Any]] = None  # parsed s1d1 doc when has_s1d1
    s2_doc: Optional[Dict[str, Any]] = None  # parsed s2 doc when has_s2
    s3_doc: Optional[Dict[str, Any]] = None  # parsed s3 doc when has_s3
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.stage in (RowStage.MOVED_NOOP, RowStage.FAILED_PINNED, RowStage.CORRUPT)


def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _analysis_from_s1(s1_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort reconstruction of the analysis dict the next stage needs.

    s1_analysis.json's ``parsed_llm_output`` mirrors ``frontier_model.analyze_prompt``'s parsed
    JSON (knowledge_gaps, search_queries, needs_search, etc.) but does NOT include
    ``execution_search_plan`` — that field is derived at runtime. We DO persist the parsed
    representation, however ``s1_data`` is consulted by ``write_s4_generation_manifest`` only
    via ``parsed_llm_output_as_dict``. For stage routing we just need enough fields to walk
    the execution plan recorded in s1d1 (when present) or to know there is no plan.
    """
    if not isinstance(s1_doc, dict):
        return None
    parsed = s1_doc.get("parsed_llm_output")
    if not isinstance(parsed, dict):
        return None
    return parsed


def _execution_plan_from_s1d1(s1d1_doc: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(s1d1_doc, dict):
        return []
    plan = s1d1_doc.get("execution_search_plan")
    return [s for s in plan if isinstance(s, dict)] if isinstance(plan, list) else []


def _has_image_step(plan: List[Dict[str, Any]]) -> bool:
    return any(str(s.get("search_type", "")).strip().lower() == "image" for s in plan)


def _validate_s2(s2_doc: Optional[Dict[str, Any]], plan: List[Dict[str, Any]]) -> bool:
    """§4 C2/C3 — empty selections OK iff plan has no image step."""
    if not isinstance(s2_doc, dict) or s2_doc.get("stage") != "s2_reference_selection":
        return False
    selections = s2_doc.get("selections") or {}
    if not isinstance(selections, dict):
        return False
    if selections:
        return True
    # empty selections — valid only if plan has no image step
    return not _has_image_step(plan)


def _validate_stop_marker(marker: Dict[str, Any], art: Path, has_s1d1: bool) -> bool:
    """Check whether marker rows indicate a finished reasoner lane."""
    if marker.get("artifact") != "reasoner_stop_before_s2":
        return False
    if not (art / "s1_analysis.json").is_file():
        return False
    if marker.get("search_lane_ran") and not has_s1d1:
        return False
    return True


def _validate_s3(s3_doc: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(s3_doc, dict) or s3_doc.get("stage") != "s3_refined_prompt":
        return False
    # parsed_llm_output.refined_prompt is the authoritative refinement output.
    # refined_prompt_final may contain a raw-prompt fallback when the LLM call failed,
    # so we require the parsed field to be non-empty when it exists.
    parsed = s3_doc.get("parsed_llm_output")
    if isinstance(parsed, dict):
        rp_parsed = str(parsed.get("refined_prompt") or "").strip()
        if rp_parsed:
            return True
        # parsed_llm_output exists but refined_prompt is empty → stub/fallback
        return False
    # No parsed_llm_output at all — fall back to refined_prompt_final
    rp = str(s3_doc.get("refined_prompt_final") or "").strip()
    return bool(rp)


def _validate_s4(s4_doc: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(s4_doc, dict) or s4_doc.get("stage") != "s4_generation_manifest":
        return False
    prompts = s4_doc.get("prompts") or {}
    if not isinstance(prompts, dict):
        return False
    return bool(str(prompts.get("refined_prompt") or "").strip())


def classify_row(row_index: int, art_path: Path, *, mode: str) -> RowState:
    """Classify on-disk state. ``mode`` ∈ {``full``, ``stop_s2``}. Pure function.

    Routing rules — match §3.7.2 of the plan.
    """
    st = RowState(row_index=row_index, art_path=art_path, stage=RowStage.NEW)
    if not art_path.is_dir():
        return st

    s1_doc = _safe_load(art_path / "s1_analysis.json")
    s1d1_pending_doc = _safe_load(art_path / "s1d1_serp_pending.json")
    s1d1_doc = _safe_load(art_path / "s1d1_search_results.json")
    s1d1_log = _safe_load(art_path / "s1d1_materialization_log.json")
    s2_doc = _safe_load(art_path / "s2_reference_selection.json")
    s3_doc = _safe_load(art_path / "s3_refined_prompt.json")
    s4_doc = _safe_load(art_path / "s4_generation_manifest.json")
    stop_marker = _safe_load(art_path / "reasoner_stop_before_s2.json")

    # s1 gate (per MUST_READ §1.1: file exists, parses, stage == s1_analysis, raw_user_prompt non-empty)
    if isinstance(s1_doc, dict) and s1_doc.get("stage") == "s1_analysis":
        rup = str(s1_doc.get("raw_user_prompt") or "").strip()
        if rup:
            st.has_s1 = True
            st.analysis = _analysis_from_s1(s1_doc)

    # s1d1 pending gate (S1D1a written serp_pending.json but S1D1b has not run yet)
    if isinstance(s1d1_pending_doc, dict) and s1d1_pending_doc.get("stage") == "s1d1_serp_pending":
        st.has_s1d1_pending = True
        st.s1d1_pending_doc = s1d1_pending_doc

    # s1d1 gate (canonical, post-download)
    if isinstance(s1d1_doc, dict) and s1d1_doc.get("stage") == "s1d1_search_results":
        st.has_s1d1 = True
        st.s1d1_doc = s1d1_doc

    # s1d1 skip log gate
    if (
        isinstance(s1d1_log, dict)
        and s1d1_log.get("outcome") == "skipped"
        and str(s1d1_log.get("reason_code") or "") in SKIP_REASON_CODES
    ):
        st.has_s1d1_skip = True

    # Plan is recorded in s1d1_search_results.json AND in s1d1_serp_pending.json (whichever exists).
    plan = _execution_plan_from_s1d1(st.s1d1_doc) or _execution_plan_from_s1d1(st.s1d1_pending_doc)
    st.needs_image_s2 = _has_image_step(plan)

    # s2 gate (§4 C2/C3)
    if isinstance(s2_doc, dict):
        if _validate_s2(s2_doc, plan):
            st.has_s2 = True
            st.s2_doc = s2_doc
        else:
            # Plan has image step but selections empty → C3 INVALID
            # We do NOT mark the row CORRUPT for this — instead we treat s2 as absent so
            # the S2 stage will re-run. If s2 keeps coming back empty, S3 will produce an
            # empty-refs refinement which is still valid.
            pass

    # s3 gate
    if _validate_s3(s3_doc):
        st.has_s3 = True
        st.s3_doc = s3_doc

    # s4 gate
    if _validate_s4(s4_doc):
        st.has_s4 = True

    # stop-marker gate
    if isinstance(stop_marker, dict) and _validate_stop_marker(stop_marker, art_path, st.has_s1d1):
        st.has_stop_marker = True

    # ── Routing ──
    # Terminal: destination already has a finished reasoner lane.
    if (st.has_s3 and st.has_s4) or st.has_stop_marker:
        st.stage = RowStage.MOVED_NOOP
        return st

    # Mode-specific routing for stop_s2
    if mode == "stop_s2":
        # In stop-s2 mode we never want s2/s3/s4. We only need s1 (+ optionally s1d1) and then
        # write the marker. If a previous full-mode run already produced s2/s3/s4, that row is
        # already complete (handled above as MOVED_NOOP); we wouldn't downgrade it.
        #
        # IMPORTANT: ``execution_search_plan`` is built at runtime by ``FrontierModel`` and is
        # NOT stored in ``s1_analysis.json`` directly — only ``parsed_llm_output`` (analyzer
        # JSON) is persisted. So we cannot determine plan emptiness from s1 alone. We therefore
        # always route s1-only rows through the S1D1 worker, which is responsible for re-deriving
        # the plan and either running search OR writing the skip log + routing to STOP.
        if st.has_s1:
            if st.has_s1d1 or st.has_s1d1_skip:
                st.stage = RowStage.MOVE_READY
                return st
            if st.has_s1d1_pending:
                # SERP done, need download to finish before stop marker (so post-stop tools
                # that inspect search_results.json find the proper file, not just pending).
                st.stage = RowStage.S1D1_SEARCHED
                return st
            st.stage = RowStage.S1_OK
            return st
        st.stage = RowStage.NEW
        return st

    # mode == "full"
    if st.has_s3:
        st.stage = RowStage.S3_OK
        return st
    if st.has_s2:
        st.stage = RowStage.S2_OK
        return st
    if st.has_s1d1 or st.has_s1d1_skip:
        st.stage = RowStage.S1D1_OK if st.has_s1d1 else RowStage.S1D1_SKIPPED
        return st
    if st.has_s1d1_pending:
        # S1D1a (SERP only) has written serp_pending.json but the downloader hasn't run.
        st.stage = RowStage.S1D1_SEARCHED
        return st
    if st.has_s1:
        st.stage = RowStage.S1_OK
        return st
    return st  # NEW


def histogram(states: List[RowState]) -> Dict[str, int]:
    counts: Dict[str, int] = {s.value: 0 for s in RowStage}
    for st in states:
        counts[st.stage.value] += 1
    return counts


def enqueue_plan(states: List[RowState], *, mode: str) -> Dict[str, int]:
    """Per-stage enqueue plan: which queue each non-terminal row goes into first.

    With the V3 split, S1D1 splits into S1D1A (search) and S1D1B (download). Older
    callers that read ``S1D1`` get the sum (back-compat).
    """
    plan = {"S1": 0, "S1D1A": 0, "S1D1B": 0, "S1D1": 0,
            "S2": 0, "S3": 0, "S4_ONLY": 0, "MOVE": 0}
    for st in states:
        s = st.stage
        if s == RowStage.NEW:
            plan["S1"] += 1
        elif s == RowStage.S1_OK:
            # S1_OK rows go to S1D1a (search-only) first
            plan["S1D1A"] += 1
            plan["S1D1"] += 1
        elif s == RowStage.S1D1_SEARCHED:
            # SERP done; needs download
            plan["S1D1B"] += 1
            plan["S1D1"] += 1
        elif s in (RowStage.S1D1_OK, RowStage.S1D1_SKIPPED):
            if mode == "stop_s2":
                plan["MOVE"] += 1
            else:
                if st.needs_image_s2 and s == RowStage.S1D1_OK:
                    plan["S2"] += 1
                else:
                    plan["S3"] += 1
        elif s == RowStage.S2_OK:
            plan["S3"] += 1
        elif s == RowStage.S3_OK:
            plan["S4_ONLY"] += 1
        elif s == RowStage.MOVE_READY:
            plan["MOVE"] += 1
        # terminal states (MOVED_NOOP, FAILED_PINNED, CORRUPT, STOP_MARKER_WRITTEN) contribute 0
    return plan
