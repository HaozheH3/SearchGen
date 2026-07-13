"""
V2 Artifacts Writer — strict JSON layout for reasoner stages (s1–s4).

Contract (schema_version 6):
- ``s1.parsed_llm_output`` = ``json.loads(raw_llm_response)`` only (no merged analysis fields).
- ``s1d1``: ``knowledge_gaps`` copied from s1; ``search_results`` keyed by plan step; each hit includes ``search_query``.
- ``s2.selections``: keyed by plan-step id; no raw selection prompt/response blobs; slim ``candidate_image_mappings``.
- ``s3`` / ``s4``: unchanged intent from v5.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 6
SAVE_TOP_SEARCH_RESULTS = 10


def env_stop_before_reference_selection() -> bool:
    v = os.environ.get("SEARCHGEN_STOP_BEFORE_REFERENCE_SELECTION") or ""
    return v.strip().lower() in ("1", "true", "yes")


def parsed_llm_output_as_dict(val: Any) -> dict[str, Any]:
    """Normalize ``parsed_llm_output`` from disk or memory to a dict."""
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, str) and val.strip():
        try:
            v = json.loads(val)
            return v if isinstance(v, dict) else {"_value": v}
        except json.JSONDecodeError:
            return {}
    return {}


def _parsed_refinement_block(s3_data: dict[str, Any]) -> dict[str, Any]:
    """Refinement JSON lives under ``parsed_llm_output`` (v6); legacy rows may use ``parsed_refinement_json``."""
    p = parsed_llm_output_as_dict(s3_data.get("parsed_llm_output"))
    if p:
        return p
    pr = s3_data.get("parsed_refinement_json")
    return dict(pr) if isinstance(pr, dict) else {}


def parse_visual_slot_index(slot: Any) -> int | None:
    """Map ``visual-1``, ``1``, etc. to a 1-based reference index (``None`` if unknown)."""
    if isinstance(slot, int):
        return slot if slot >= 1 else None
    s = str(slot or "").strip().lower()
    if not s:
        return None
    if s.startswith("visual-"):
        tail = s.split("-", 1)[1].strip()
        try:
            n = int(tail)
            return n if n >= 1 else None
        except ValueError:
            return None
    try:
        n = int(s)
        return n if n >= 1 else None
    except ValueError:
        return None


def refinement_selected_indices_one_based(s3_data: dict[str, Any]) -> set[int]:
    """Explicit 1-based indices: top-level ``selected_reference_indices``, parsed block, and ``borrow_from_references``."""
    out: set[int] = set()
    if not isinstance(s3_data, dict):
        return out
    p = _parsed_refinement_block(s3_data)
    for bucket in (p.get("selected_reference_indices"), s3_data.get("selected_reference_indices")):
        if not isinstance(bucket, list):
            continue
        for x in bucket:
            if isinstance(x, int) and x >= 1:
                out.add(x)
    for borrow in p.get("borrow_from_references") or []:
        if not isinstance(borrow, dict) or not borrow.get("used"):
            continue
        ri = borrow.get("reference_index")
        if ri is None:
            ri = borrow.get("image_index")
        if isinstance(ri, int) and ri >= 1:
            out.add(ri)
            continue
        vis = parse_visual_slot_index(borrow.get("index"))
        if vis is not None:
            out.add(vis)
    return out


def refinement_borrow_traits_by_ref_index(s3_data: dict[str, Any]) -> dict[int, list[str]]:
    """``borrowed_traits`` / ``reference_focus`` keyed by 1-based reference index."""
    if not isinstance(s3_data, dict):
        return {}
    p = _parsed_refinement_block(s3_data)
    borrow_map: dict[int, list[str]] = {}
    for borrow in p.get("borrow_from_references") or []:
        if not isinstance(borrow, dict) or not borrow.get("used"):
            continue
        ri = borrow.get("reference_index")
        if ri is None:
            ri = borrow.get("image_index")
        ref_n: int | None = None
        if isinstance(ri, int) and ri >= 1:
            ref_n = ri
        else:
            ref_n = parse_visual_slot_index(borrow.get("index"))
        if ref_n is None:
            continue
        traits: list[str] = []
        bt = borrow.get("borrowed_traits")
        if isinstance(bt, list):
            for x in bt:
                if isinstance(x, str) and x.strip():
                    traits.append(x.strip())
        rf = borrow.get("reference_focus")
        if isinstance(rf, str) and rf.strip():
            traits.append(rf.strip())
        borrow_map[ref_n] = traits
    return borrow_map


def plan_step_key(plan_index: int, search_type: str) -> str:
    st = (search_type or "web").strip().lower()
    if st not in ("image", "web"):
        st = "web"
    return f"{st}-{plan_index}"


def _parse_raw_llm_json(raw: str) -> dict[str, Any]:
    if not (raw or "").strip():
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {"_value": v}
    except json.JSONDecodeError:
        return {"_unparsed_text": raw}


class V2ArtifactsWriter:
    """Writes v6 artifacts under ``artifacts_files/`` during v2 pipeline execution."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.artifacts_dir = self.output_dir / "artifacts_files"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.s1_data: dict[str, Any] | None = None
        self.s2_document: dict[str, Any] | None = None
        self.s3_data: dict[str, Any] | None = None
        self.search_results_data: dict[str, Any] | None = None

        self._s2_slots: list[tuple[str, dict[str, Any]]] = []
        self._runtime_needs_search: bool = False

    def reset_s2_slots(self) -> None:
        self._s2_slots = []

    def append_s2_slot(self, selection_bundle: dict[str, Any], plan_step_key_str: str) -> None:
        """Queue one selection bundle under ``plan_step_key_str`` (e.g. ``image-0``)."""
        self._s2_slots.append((plan_step_key_str, dict(selection_bundle)))

    def flush_s2_reference_selection(self) -> None:
        """Write ``s2_reference_selection.json`` and remove shard files."""
        for p in self.artifacts_dir.glob("s2_reference_selection*.json"):
            try:
                p.unlink()
            except OSError:
                pass

        if env_stop_before_reference_selection():
            self.s2_document = None
            return

        selections: dict[str, Any] = {}
        for plan_key, slot in self._s2_slots:
            parsed = _parse_raw_llm_json(str(slot.get("selection_response") or ""))
            raw_maps = slot.get("candidate_image_mappings")
            slim_maps: list[dict[str, Any]] = []
            if isinstance(raw_maps, list):
                for m in raw_maps:
                    if not isinstance(m, dict):
                        continue
                    slim_maps.append(
                        {
                            "index": m.get("index"),
                            "title": m.get("title"),
                            "url": m.get("url"),
                            "local_path": m.get("local_path"),
                        }
                    )
            # ``visual_description`` is produced in ``FrontierModel.select_reference_image`` (title + source summary).
            selections[plan_key] = {
                "query": str(slot.get("query") or ""),
                "query_index": slot.get("query_index"),
                "parsed_llm_output": parsed,
                "selected_image": slot.get("selected_image"),
                "selection_reasoning": str(slot.get("selection_reasoning") or ""),
                "identified_knowledge_gaps": str(slot.get("identified_knowledge_gaps") or ""),
                "visual_description": str(slot.get("visual_description") or ""),
                "candidate_image_mappings": slim_maps,
                "selection_skipped": bool(slot.get("selection_skipped", False)),
            }

        doc = {
            "schema_version": SCHEMA_VERSION,
            "stage": "s2_reference_selection",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selections": selections,
        }
        doc, _ = backfill_s2_document_from_s1d1(doc, self.search_results_data)
        out = self.artifacts_dir / "s2_reference_selection.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.s2_document = doc
        print(f"  ✓ Wrote v2: {out.name} ({len(selections)} selection key(s))")

    def write_s1_analysis(
        self,
        analysis: dict[str, Any],
        raw_llm_input: dict[str, Any],
        raw_llm_response: str,
        user_prompt: str,
    ) -> None:
        raw_s = str(raw_llm_response or "")
        self._runtime_needs_search = bool(analysis.get("needs_search", False)) if isinstance(analysis, dict) else False
        s1 = {
            "schema_version": SCHEMA_VERSION,
            "stage": "s1_analysis",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_llm_input": raw_llm_input,
            "raw_user_prompt": user_prompt,
            "raw_llm_response": raw_s,
            "parsed_llm_output": _parse_raw_llm_json(raw_s),
        }
        path = self.artifacts_dir / "s1_analysis.json"
        path.write_text(json.dumps(s1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.s1_data = s1
        print(f"  ✓ Wrote v2: {path.name}")

    def _compact_search_hits(
        self, rows: list[dict[str, Any]], search_type: str, plan_query: str
    ) -> list[dict[str, Any]]:
        st = (search_type or "web").strip().lower()
        q = str(plan_query or "").strip()
        top = rows[:SAVE_TOP_SEARCH_RESULTS]
        out: list[dict[str, Any]] = []
        for r in top:
            if not isinstance(r, dict):
                continue
            if st == "image":
                lp = r.get("local_path")
                lp_s = str(lp).strip() if lp else ""
                out.append(
                    {
                        "search_query": q,
                        "title": r.get("title"),
                        "url": r.get("url") or r.get("imageUrl"),
                        "thumbnail_url": r.get("thumbnailUrl"),
                        "local_path": lp_s if lp_s else None,
                        "downloaded": bool(lp_s) or bool(r.get("download_success")),
                        "position": r.get("position"),
                        "source": r.get("source"),
                    }
                )
            else:
                out.append(
                    {
                        "search_query": q,
                        "title": r.get("title"),
                        "url": r.get("url") or r.get("link"),
                        "link": r.get("link"),
                        "snippet": r.get("snippet") or r.get("description"),
                        "position": r.get("position"),
                        "source": r.get("source"),
                    }
                )
        return out

    def _search_results_by_plan_step_topk(
        self,
        search_results: list[dict[str, Any]],
        execution_search_plan: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for i, step in enumerate(execution_search_plan):
            if not isinstance(step, dict):
                continue
            q = str(step.get("query", "")).strip()
            if not q:
                continue
            st = str(step.get("search_type", "web")).strip().lower()
            key = plan_step_key(i, st)
            rows: list[dict[str, Any]] = []
            for r in search_results:
                if not isinstance(r, dict):
                    continue
                rq = str(r.get("query", "")).strip()
                if rq != q:
                    continue
                rt = str(r.get("type", "web")).lower()
                if st == "image" and (rt == "image" or r.get("search_type") == "image"):
                    rows.append(r)
                elif st == "web" and rt == "web":
                    rows.append(r)
            out[key] = self._compact_search_hits(rows, st, q)
        return out

    def write_s1d1_search_results(
        self,
        search_results: list[dict[str, Any]],
        execution_search_plan: list[dict[str, Any]],
        _analysis: dict[str, Any],
    ) -> None:
        s1p = parsed_llm_output_as_dict((self.s1_data or {}).get("parsed_llm_output"))
        kg = s1p.get("knowledge_gaps")
        s1d1 = {
            "schema_version": SCHEMA_VERSION,
            "stage": "s1d1_search_results",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_search_plan": execution_search_plan,
            "knowledge_gaps": copy.deepcopy(kg) if isinstance(kg, list) else [],
            "search_results": self._search_results_by_plan_step_topk(search_results, execution_search_plan),
        }
        path = self.artifacts_dir / "s1d1_search_results.json"
        path.write_text(json.dumps(s1d1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.search_results_data = s1d1
        print(f"  ✓ Wrote v2: {path.name}")
        self.write_s1d1_materialization_log(
            outcome="written",
            reason_code="s1d1_search_results_json_persisted",
            detail={
                "execution_search_plan_len": len(execution_search_plan)
                if isinstance(execution_search_plan, list)
                else 0,
                "raw_search_hit_rows": len(search_results) if isinstance(search_results, list) else 0,
                "s1d1_plan_keys": list((s1d1.get("search_results") or {}).keys())
                if isinstance(s1d1.get("search_results"), dict)
                else [],
            },
        )

    def write_s1d1_materialization_log(
        self,
        *,
        outcome: str,
        reason_code: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Explain whether ``s1d1_search_results.json`` was written and why (audit / debugging).

        Written to ``artifacts_files/s1d1_materialization_log.json`` so rows without ``s1d1_search_results.json``
        still record the pipeline branch (empty plan, search disabled, reasoner-only refine, etc.).
        """
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact": "s1d1_materialization_log",
            "outcome": str(outcome),
            "reason_code": str(reason_code),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "detail": dict(detail or {}),
        }
        out = self.artifacts_dir / "s1d1_materialization_log.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  ✓ Wrote v2: {out.name} ({outcome})")

    def write_s3_refined_prompt(
        self,
        refinement_data: dict[str, Any],
        user_prompt: str,
        reference_images: list[dict[str, Any]],
    ) -> None:
        raw_in = (
            refinement_data.get("raw_input_prompt")
            or refinement_data.get("raw_llm_input")
            or refinement_data.get("prompt")
            or ""
        )
        raw_out = refinement_data.get("raw_output_response") or refinement_data.get("raw_llm_response") or ""
        raw_in_s = raw_in if isinstance(raw_in, str) else (json.dumps(raw_in, ensure_ascii=False) if raw_in is not None else "")
        raw_out_s = raw_out if isinstance(raw_out, str) else (json.dumps(raw_out, ensure_ascii=False) if raw_out is not None else "")

        parsed = _parse_raw_llm_json(raw_out_s)

        s3 = {
            "schema_version": SCHEMA_VERSION,
            "stage": "s3_refined_prompt",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_llm_input": raw_in_s,
            "raw_llm_response": raw_out_s,
            "parsed_llm_output": parsed,
            "refinement_vlm_submission": refinement_data.get("refinement_vlm_submission"),
            "selected_reference_indices": refinement_data.get("selected_reference_indices", []),
            "refined_prompt_final": str(refinement_data.get("refined_prompt_final") or refinement_data.get("refined_prompt") or "").strip(),
            "prompt_context_metadata": {
                "schema_version": SCHEMA_VERSION,
                "description": "Context for prompt refinement",
                "original_user_request": user_prompt,
                "reference_images": reference_images,
                "visual_reference_candidates_rendered": self._extract_visual_candidates(),
                "web_search_evidence": [],
            },
        }
        path = self.artifacts_dir / "s3_refined_prompt.json"
        path.write_text(json.dumps(s3, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.s3_data = s3
        print(f"  ✓ Wrote v2: {path.name}")

    def write_s4_generation_manifest(
        self,
        user_prompt: str,
        evalset: str = "unknown",
        reasoner: str = "unknown",
        row_index: int = -1,
    ) -> None:
        if not self.s1_data or not self.s3_data:
            print("  ⚠ Cannot write s4: missing s1 or s3 data")
            return

        refined_prompt = self._extract_refined_prompt()
        augmented_mode = self._determine_augmented_mode()
        references = self._extract_reference_images()
        knowledge_gaps = self._extract_knowledge_gaps()

        s4 = {
            "schema_version": SCHEMA_VERSION,
            "stage": "s4_generation_manifest",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_identity": {
                "eval_row_index": row_index,
                "evalset": evalset,
                "reasoner": reasoner,
                "dataset_source": f"{evalset}_dataset.json",
                "prompt_norm": user_prompt[:100],
            },
            "prompts": {
                "user_prompt_raw": user_prompt,
                "refined_prompt": refined_prompt,
                "augmented_mode": augmented_mode,
            },
            "reference_images": {
                "ordered_references": references,
                "total_count": len(references),
            },
            "knowledge_gaps": knowledge_gaps,
            "generation_context": {
                "needs_search": bool(parsed_llm_output_as_dict(self.s1_data.get("parsed_llm_output")).get("needs_search"))
                or self._runtime_needs_search,
                "search_executed": self.search_results_data is not None,
                "reference_selection_strategy": "top_k",
                "refinement_strategy": "multimodal" if references else "text_only",
            },
            "provenance": {
                "s1_analysis_file": "s1_analysis.json",
                "s1d1_search_results_file": "s1d1_search_results.json",
                "s2_reference_selection_file": "s2_reference_selection.json",
                "s3_refined_prompt_file": "s3_refined_prompt.json",
                "reasoner_complete_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        path = self.artifacts_dir / "s4_generation_manifest.json"
        path.write_text(json.dumps(s4, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  ✓ Wrote v2: {path.name}")

    def _extract_refined_prompt(self) -> str:
        if not self.s3_data:
            return ""
        fin = str(self.s3_data.get("refined_prompt_final") or "").strip()
        if fin:
            return fin
        p = parsed_llm_output_as_dict(self.s3_data.get("parsed_llm_output"))
        return str(p.get("refined_prompt") or "").strip()

    def _determine_augmented_mode(self) -> str:
        if not self.s3_data:
            return "t2i"
        top = self.s3_data.get("selected_reference_indices")
        if isinstance(top, list) and any(isinstance(x, int) and x >= 1 for x in top):
            return "i2i"
        p = _parsed_refinement_block(self.s3_data)
        if p.get("selected_reference_indices"):
            return "i2i"
        for b in p.get("borrow_from_references") or []:
            if isinstance(b, dict) and b.get("used"):
                return "i2i"
        return "t2i"

    def _extract_reference_images(self) -> list[dict[str, Any]]:
        if not self.s3_data:
            return []
        ctx = self.s3_data.get("prompt_context_metadata") or {}
        ref_images = ctx.get("reference_images") if isinstance(ctx.get("reference_images"), list) else []
        s3 = self.s3_data if isinstance(self.s3_data, dict) else {}
        selected_indices = refinement_selected_indices_one_based(s3)
        borrow_map = refinement_borrow_traits_by_ref_index(s3)

        references = []
        for idx, ref in enumerate(ref_images):
            if not isinstance(ref, dict):
                continue
            local_path = str(ref.get("local_path") or "")
            local_path_relative = ""
            if local_path:
                path_obj = Path(local_path)
                fname = path_obj.name
                # Always derive both paths from output_dir to stay consistent
                # regardless of where the source local_path originally pointed
                if path_obj.is_absolute():
                    try:
                        local_path_relative = str(path_obj.relative_to(self.output_dir))
                    except ValueError:
                        local_path_relative = f"reference_images/{fname}"
                else:
                    local_path_relative = local_path
            references.append(
                {
                    "index": idx,
                    "query": str(ref.get("query") or ""),
                    "title": str(ref.get("title") or ""),
                    "url": str(ref.get("url") or ref.get("imageUrl") or ""),
                    "local_path_relative": local_path_relative,
                    "selection_reasoning": str(ref.get("selection_reasoning") or ""),
                    "used_in_refinement": (idx + 1) in selected_indices,
                    "borrowed_traits": borrow_map.get(idx + 1, []),
                }
            )
        return references

    def _extract_knowledge_gaps(self) -> dict[str, list[dict[str, Any]]]:
        gaps: dict[str, list[dict[str, Any]]] = {"critical": [], "important": [], "moderate": [], "minor": []}
        if not self.s1_data:
            return gaps
        parsed = parsed_llm_output_as_dict(self.s1_data.get("parsed_llm_output"))
        for gap in parsed.get("knowledge_gaps") or []:
            if not isinstance(gap, dict):
                continue
            severity = str(gap.get("severity", "moderate")).lower()
            if severity not in gaps:
                severity = "moderate"
            gaps[severity].append(
                {
                    "entity": gap.get("entity", ""),
                    "category": gap.get("category", "OTHER"),
                    "severity": severity,
                    "reasoning": gap.get("reasoning", ""),
                }
            )
        return gaps

    def _extract_visual_candidates(self) -> dict[str, list[dict[str, Any]]]:
        candidates: dict[str, list[dict[str, Any]]] = {"critical": [], "important": [], "moderate": [], "minor": []}
        if not self.s1_data:
            return candidates
        parsed = parsed_llm_output_as_dict(self.s1_data.get("parsed_llm_output"))
        for ref in parsed.get("visual_reference_candidates") or []:
            if not isinstance(ref, dict):
                continue
            severity = str(ref.get("severity", "moderate")).lower()
            if severity not in candidates:
                severity = "moderate"
            candidates[severity].append(
                {
                    "severity": severity,
                    "entity": ref.get("entity", ""),
                    "reasoning": ref.get("reasoning", ""),
                }
            )
        return candidates


def _s2_http_url(u: Any) -> bool:
    return isinstance(u, str) and u.startswith(("http://", "https://"))


def _s1d1_hits_for_plan(s1d1_doc: dict[str, Any], plan_key: str) -> list[dict[str, Any]]:
    sr = s1d1_doc.get("search_results")
    if not isinstance(sr, dict):
        return []
    raw = sr.get(plan_key)
    if not isinstance(raw, list):
        return []
    return [h for h in raw if isinstance(h, dict)]


def _s1_hit_as_selected_image(hit: dict[str, Any]) -> dict[str, Any]:
    u = hit.get("url") or hit.get("imageUrl")
    return {
        "search_query": hit.get("search_query"),
        "title": hit.get("title"),
        "url": str(u).strip() if u else None,
        "thumbnail_url": hit.get("thumbnail_url") or hit.get("thumbnailUrl"),
        "local_path": hit.get("local_path"),
        "downloaded": hit.get("downloaded"),
        "position": hit.get("position"),
        "source": hit.get("source"),
        "download_error": hit.get("download_error"),
    }


def backfill_s2_document_from_s1d1(
    s2_doc: dict[str, Any],
    s1d1_doc: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Fill missing / non-http ``selected_image.url`` from ``s1d1_search_results`` (same-step hits).

    Uses ``parsed_llm_output.selected_index`` (1-based) when present; otherwise the first SERP hit
    with an http(s) URL. Returns a deep-copied document and whether any selection block changed.
    """
    if not isinstance(s2_doc, dict) or not isinstance(s1d1_doc, dict):
        return s2_doc, False
    out = copy.deepcopy(s2_doc)
    selections = out.get("selections")
    if not isinstance(selections, dict):
        return s2_doc, False
    changed = False
    for plan_key, block in selections.items():
        if not isinstance(block, dict):
            continue
        sim = block.get("selected_image")
        if isinstance(sim, dict) and _s2_http_url(sim.get("url")):
            continue
        hits = _s1d1_hits_for_plan(s1d1_doc, str(plan_key))
        if not hits:
            continue
        pick: dict[str, Any] | None = None
        plo = block.get("parsed_llm_output")
        if isinstance(plo, dict):
            si = plo.get("selected_index")
            if isinstance(si, int) and si >= 1 and si <= len(hits):
                cand = hits[si - 1]
                if isinstance(cand, dict) and (
                    _s2_http_url(cand.get("url")) or _s2_http_url(cand.get("imageUrl"))
                ):
                    pick = cand
        if pick is None:
            for h in hits:
                if not isinstance(h, dict):
                    continue
                if _s2_http_url(h.get("url")) or _s2_http_url(h.get("imageUrl")):
                    pick = h
                    break
        if pick is None:
            continue
        block["selected_image"] = _s1_hit_as_selected_image(pick)
        block["selection_skipped"] = False
        changed = True
    return out, changed
