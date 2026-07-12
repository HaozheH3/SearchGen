# Copyright 2026 Jayce-Ping, Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared ToolGen SearchBetter judge prompt layout and XML parsing (``evaluation`` schema; JSON fallback).

The default judge reply is strict XML under ``<evaluation>`` (reference_alignment, checklist,
rubric, visual_reference, text_reference). Legacy ``<judge_output>`` / ``<tgj>`` / ``<toolgen_judge_output>`` is
still parsed for backward compatibility with older model outputs.

Aligned with ``phase5_data_collection/evaluate_searchbetter_hard_direct.py`` and
real saved prompts such as ``.../augmented_prompt_context.txt`` (same Instructions
block, Evaluation context headers, interleaved reference context lines).

Training contract: the judge ``Task prompt`` / instructions ``user_prompt`` field comes from
``row["user_prompt"]`` and must be the **original** user task (checklist + rubric + faithfulness).
The phase4 **refined** prompt is **not** read here; it belongs on the training sample as ``prompt``
for the generator only (see :func:`build_eval_prompt_text`).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .image_utils import pil_image_to_base64

logger = logging.getLogger(__name__)

_UNPARSED_JUDGE_RESPONSE_DUMP_LOCK = threading.Lock()


def resolve_toolgen_judge_unparsed_response_dump_dir(
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Directory for persisting raw judge assistant text when XML/JSON parsing fails.

    Resolution order:

    1. ``extra_kwargs["toolgen_judge_unparsed_dump_dir"]`` (non-empty string)
    2. Environment variable ``FLOW_FACTORY_JUDGE_UNPARSED_DUMP_DIR``
    3. Environment variable ``TOOLGEN_JUDGE_UNPARSED_DUMP_DIR``
    """
    if isinstance(extra_kwargs, dict):
        raw = extra_kwargs.get("toolgen_judge_unparsed_dump_dir")
        if isinstance(raw, str) and raw.strip():
            return Path(raw).expanduser().resolve()
    for env_key in ("FLOW_FACTORY_JUDGE_UNPARSED_DUMP_DIR", "TOOLGEN_JUDGE_UNPARSED_DUMP_DIR"):
        raw = os.environ.get(env_key)
        if isinstance(raw, str) and raw.strip():
            return Path(raw).expanduser().resolve()
    return None


def write_toolgen_judge_unparsed_response_dump(
    dump_dir: Path,
    *,
    stem: str,
    raw_response: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write the exact model response for offline debugging.

    Creates ``dump_dir`` if needed. Writes ``{stem}.judge_unparsed.txt`` (UTF-8) and
    ``{stem}.judge_unparsed_meta.json`` (small JSON sidecar). Returns the path to the text file.

    Thread-safe across concurrent judge calls (single module lock around the write pair).
    """
    if not isinstance(dump_dir, Path):
        raise TypeError(f"dump_dir must be pathlib.Path, got {type(dump_dir).__name__}: {dump_dir!r}")
    stem_safe = str(stem).replace(os.sep, "_").replace("/", "_").strip()
    if not stem_safe:
        raise ValueError("stem must be non-empty after sanitization")

    payload: Dict[str, Any] = {"raw_char_len": len(raw_response)}
    if meta:
        for key, val in meta.items():
            payload[str(key)] = val
    payload["stem"] = stem_safe

    def _write() -> Path:
        dump_dir.mkdir(parents=True, exist_ok=True)
        text_path = dump_dir / f"{stem_safe}.judge_unparsed.txt"
        meta_path = dump_dir / f"{stem_safe}.judge_unparsed_meta.json"
        text_path.write_text(raw_response, encoding="utf-8")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        marked = (
            f"!!!! ToolGen judge UNPARSED_RESPONSE_DUMP "
            f"path_txt={text_path} path_meta={meta_path} chars={len(raw_response)} !!!!"
        )
        print(marked, flush=True)
        logger.warning("%s", marked)
        return text_path

    with _UNPARSED_JUDGE_RESPONSE_DUMP_LOCK:
        return _write()

JUDGE_SYSTEM_PROMPT = (
    "You are an image quality evaluator. Be rigorous and evidence-based. "
    "Output strict XML only, using the root element <evaluation> and the required tag names.\n\n"
    "Follow these steps in order:\n\n"
    "Step 1 — Task understanding\n"
    "- user_prompt: identify required subjects, attributes, actions, scene, style, quality, and in-image text.\n"
    "- verification_checklist: map each item to concrete visual evidence that satisfies or violates it.\n"
    "- evaluation_rubric: clarify what each dimension judges.\n"
    "- visual_reference_context: identify reference-dependent criteria.\n"
    "- textual_knowledge_gap_context: critical/important textual gaps from prompt analysis (metadata only); "
    "what factual or textual elaboration the target should reflect.\n\n"
    "Step 2 — Image roles\n"
    "- 'Reference image N' = comparison evidence only. Never score reference image quality.\n"
    "- 'Image to assess' = the only image scored.\n\n"
    "Step 3 — Evidence extraction (observable facts only, no inference)\n"
    "  a) Subjects & identity — entities, distinguishing traits, count.\n"
    "  b) Attributes — clothing, colors, materials, textures, proportions.\n"
    "  c) Actions & poses — gestures, interactions, body language.\n"
    "  d) Scene & environment — setting, background, weather, time of day.\n"
    "  e) Composition — framing, spatial arrangement, depth, balance.\n"
    "  f) Color & lighting — palette, light direction, shadows, consistency.\n"
    "  g) Style — artistic medium, realism level, stylistic cues.\n"
    "  h) In-image text — content, legibility, spelling, placement.\n"
    "  i) Physical plausibility — anatomy (hands, faces, limbs), object physics, spatial consistency.\n"
    "  j) Artifacts — blur, aliasing, seams, duplications, impossible geometry.\n"
    "  k) AI tells — unnatural smoothness, plastic textures, warped backgrounds, uncanny features.\n\n"
    "Step 4 — Reference comparison\n"
    "- Compare target against references for reference-dependent criteria.\n"
    "- Note which identity traits are preserved, altered, or missing.\n"
    "- Map each reference to the checklist items and rubric dimensions it informs.\n\n"
    "Step 5 — Scoring scale (0–3, half-points allowed)\n"
    "  0 = Not met / contradicted.  1 = Weak, major issues.\n"
    "  2 = Mostly met, minor issues.  3 = Fully met, correct and coherent.\n\n"
    "Step 5b — Auditable scores (each scored row's <reason>)\n"
    "Use three lines in order: (1) Anchor: <0…3 in 0.5 steps> — why, with Step 3 cite. "
    "(2) Adjustments: bullets \"Δ=±0.5|±1 …\" each with evidence, or \"none\". "
    "(3) Computed final: X — X must equal Anchor+ΣΔ (clamp to [0,3], 0.5 steps) and MUST match <score>. "
    "reference_alignment: 2–4 sentences only, no Anchor template.\n\n"
    "Step 6 — Score checklist and rubric\n"
    "Score each checklist item, each task-specific rubric dimension, and ALL generic dimensions below.\n"
    "Every scored row's <reason> follows Step 5b.\n\n"
    "  prompt_faithfulness — Are all requested subjects, attributes, actions, scene, and style present and accurate?\n\n"
    "  image_quality — Visual clarity/sharpness? Artifacts or defects? "
    "Style/lighting/perspective coherence? Physical plausibility and anatomy correctness?\n\n"
    "  text_rendering — If no in-image text is required and none is present, leave the score element empty in XML; "
    "otherwise give a score in 0–3. Does rendered text match required content? Readable? Spelled? Placed well?\n\n"
    "  ai_naturalness — Does the image look AI-generated? Check: organic texture realism vs AI smoothness? "
    "Fine-detail plausibility vs uncanny uniformity? Environment grounded vs dreamlike/generic? "
    "Would a careful viewer readily identify this as AI-generated?\n\n"
    "  composition_and_aesthetics — Well-framed and balanced? Clear depth and spatial arrangement? "
    "Color harmony and consistent lighting? Visually appealing overall?\n\n"
    "  physical_plausibility — Physical rules adherence: anatomy (fingers, limbs, proportions, pose)? "
    "Object physics (gravity, support, balance)? Spatial consistency (perspective, scale, occlusion)? "
    "Material properties (reflections, transparency, texture)? Lighting/shadow consistency? "
    "Leave score empty if not applicable (abstract/stylized content).\n\n"
    "Step 7 — Visual reference fidelity\n"
    "- Set visual_reference applicable=true only if visual references are provided in the evaluation context.\n"
    "- When applicable, compare the target image against each reference for identity preservation, attribute "
    "consistency, and style fidelity. Check: are key facial features / distinguishing traits maintained? "
    "Are colors, textures, and proportions consistent? Is the overall style preserved?\n"
    "- Score preservation of critical identity, attributes, and style from references onto the target.\n"
    "- When applicable, visual_reference <reason> follows Step 5b.\n\n"
    "Step 7b — Textual knowledge / textual elaboration fidelity\n"
    "- Set text_reference applicable=true only if textual knowledge gaps are listed in the evaluation context.\n"
    "- Using gap metadata (entity, reasoning, suggested lookup intent) and visible target evidence only, score how "
    "correctly the image renders what those gaps require (facts, labels, standards, wording, numbers, symbols, etc.).\n"
    "- Do not use live web search.\n"
    "- When applicable, text_reference <reason> follows Step 5b.\n\n"
    "Step 8 — Consistency checks\n"
    "- Computed final = <score> for every scored row; reasons must match scores.\n"
    "- Reasons and scores must not contradict. Major failures → score < 2. Minor-only issues → score > 1.\n"
    "- When a rubric score applies, use values in {0, 0.5, 1, 1.5, 2, 2.5, 3}. When a dimension does not apply, "
    "leave that dimension's <score> empty (no characters between the tags).\n"
    "- Include a dimension element for every required rubric row; all numeric scores are for the target image only.\n\n"
    "Output the target image evaluation as a single XML block. "
    "No preface, no code fences, no text after the root element. "
    "Escape <, >, & in text (e.g. &amp;)."
)

# Default judge XML root (parse path: :func:`parse_judge_xml_to_dict`).
JUDGE_XML_ROOT_TAG = "evaluation"

JUDGE_XML_REFERENCE_ALIGNMENT = "reference_alignment"
JUDGE_XML_ALIGNMENT_ITEM = "item"
JUDGE_XML_REF_ATTR = "ref"
JUDGE_XML_CHECKLIST_LINKS = "checklist_links"
JUDGE_XML_RUBRIC_LINKS = "rubric_links"

JUDGE_XML_CHECKLIST = "checklist"
JUDGE_XML_CHECKLIST_ITEM = "item"
JUDGE_XML_CRITERION = "criterion"

JUDGE_XML_RUBRIC = "rubric"
JUDGE_XML_DIMENSION = "dimension"

JUDGE_XML_VISUAL_REFERENCE = "visual_reference"
JUDGE_XML_TEXT_REFERENCE = "text_reference"
JUDGE_XML_APPLICABLE = "applicable"

JUDGE_XML_REASON = "reason"
JUDGE_XML_SCORE = "score"
JUDGE_XML_NAME_ATTR = "name"
JUDGE_XML_LINK = "link"
JUDGE_XML_I_ATTR = "i"

# Legacy ``<judge_output>`` tags (still accepted by :func:`parse_judge_xml_to_dict`).
LEGACY_JUDGE_XML_ROOT = "judge_output"
LEGACY_JUDGE_XML_REF_ALIGN = "ref_align"
LEGACY_JUDGE_XML_REF_ROW = "ref_row"
LEGACY_JUDGE_XML_REF_LABEL = "ref_label"
LEGACY_JUDGE_XML_LINKED_CL = "linked_checklist"
LEGACY_JUDGE_XML_LINKED_RUB = "linked_rubric"
LEGACY_JUDGE_XML_REF_REASON = "ref_reason"
LEGACY_JUDGE_XML_CL_ROW = "cl_row"
LEGACY_JUDGE_XML_LINE_TEXT = "line_text"
LEGACY_JUDGE_XML_RUBRIC_LINE = "rubric_line"
LEGACY_JUDGE_XML_VISUAL_REF = "visual_ref"
LEGACY_JUDGE_XML_APPLIES = "applies"

JUDGE_OUTPUT_XML_SHAPE_FENCED = f"""```xml
<{JUDGE_XML_ROOT_TAG}>
  <{JUDGE_XML_REFERENCE_ALIGNMENT}>
    <{JUDGE_XML_ALIGNMENT_ITEM} {JUDGE_XML_REF_ATTR}="Reference image 1">
      <{JUDGE_XML_CHECKLIST_LINKS}>…</{JUDGE_XML_CHECKLIST_LINKS}>
      <{JUDGE_XML_RUBRIC_LINKS}>…</{JUDGE_XML_RUBRIC_LINKS}>
      <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
    </{JUDGE_XML_ALIGNMENT_ITEM}>
  </{JUDGE_XML_REFERENCE_ALIGNMENT}>
  <{JUDGE_XML_CHECKLIST}>
    <{JUDGE_XML_CHECKLIST_ITEM} {JUDGE_XML_I_ATTR}="0">
      <{JUDGE_XML_CRITERION}>…</{JUDGE_XML_CRITERION}>
      <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
      <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
    </{JUDGE_XML_CHECKLIST_ITEM}>
  </{JUDGE_XML_CHECKLIST}>
  <{JUDGE_XML_RUBRIC}>
    <{JUDGE_XML_DIMENSION} {JUDGE_XML_NAME_ATTR}="…">
      <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
      <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
    </{JUDGE_XML_DIMENSION}>
  </{JUDGE_XML_RUBRIC}>
  <{JUDGE_XML_VISUAL_REFERENCE}>
    <{JUDGE_XML_APPLICABLE}>…</{JUDGE_XML_APPLICABLE}>
    <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
    <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
  </{JUDGE_XML_VISUAL_REFERENCE}>
  <{JUDGE_XML_TEXT_REFERENCE}>
    <{JUDGE_XML_APPLICABLE}>…</{JUDGE_XML_APPLICABLE}>
    <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
    <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
  </{JUDGE_XML_TEXT_REFERENCE}>
</{JUDGE_XML_ROOT_TAG}>
```"""

JUDGE_OUTPUT_XML_COMPACT_EXAMPLE = (
    f"<{JUDGE_XML_ROOT_TAG}>"
    f"<{JUDGE_XML_REFERENCE_ALIGNMENT}>"
    f'<{JUDGE_XML_ALIGNMENT_ITEM} {JUDGE_XML_REF_ATTR}="Reference image 1">'
    f"<{JUDGE_XML_CHECKLIST_LINKS}>checklist phrase this ref supports</{JUDGE_XML_CHECKLIST_LINKS}>"
    f"<{JUDGE_XML_RUBRIC_LINKS}>rubric dimension name</{JUDGE_XML_RUBRIC_LINKS}>"
    f"<{JUDGE_XML_REASON}>how the reference informs target judgment</{JUDGE_XML_REASON}>"
    f"</{JUDGE_XML_ALIGNMENT_ITEM}></{JUDGE_XML_REFERENCE_ALIGNMENT}>"
    f"<{JUDGE_XML_CHECKLIST}>"
    f'<{JUDGE_XML_CHECKLIST_ITEM} {JUDGE_XML_I_ATTR}="0">'
    f"<{JUDGE_XML_CRITERION}>exact checklist line being scored</{JUDGE_XML_CRITERION}>"
    f"<{JUDGE_XML_REASON}>rationale (cite Step 3 evidence)</{JUDGE_XML_REASON}>"
    f"<{JUDGE_XML_SCORE}>2</{JUDGE_XML_SCORE}></{JUDGE_XML_CHECKLIST_ITEM}></{JUDGE_XML_CHECKLIST}>"
    f"<{JUDGE_XML_RUBRIC}>"
    f'<{JUDGE_XML_DIMENSION} {JUDGE_XML_NAME_ATTR}="custom_dim">'
    f"<{JUDGE_XML_REASON}>rationale</{JUDGE_XML_REASON}><{JUDGE_XML_SCORE}>1</{JUDGE_XML_SCORE}></{JUDGE_XML_DIMENSION}>"
    f'<{JUDGE_XML_DIMENSION} {JUDGE_XML_NAME_ATTR}="prompt_faithfulness">'
    f"<{JUDGE_XML_REASON}>rationale</{JUDGE_XML_REASON}><{JUDGE_XML_SCORE}>3</{JUDGE_XML_SCORE}></{JUDGE_XML_DIMENSION}>"
    f"</{JUDGE_XML_RUBRIC}>"
    f"<{JUDGE_XML_VISUAL_REFERENCE}>"
    f"<{JUDGE_XML_APPLICABLE}>false</{JUDGE_XML_APPLICABLE}>"
    f"<{JUDGE_XML_REASON}>reference fidelity notes</{JUDGE_XML_REASON}>"
    f"<{JUDGE_XML_SCORE}>0</{JUDGE_XML_SCORE}></{JUDGE_XML_VISUAL_REFERENCE}>"
    f"<{JUDGE_XML_TEXT_REFERENCE}>"
    f"<{JUDGE_XML_APPLICABLE}>false</{JUDGE_XML_APPLICABLE}>"
    f"<{JUDGE_XML_REASON}>web knowledge gap fidelity notes</{JUDGE_XML_REASON}>"
    f"<{JUDGE_XML_SCORE}>0</{JUDGE_XML_SCORE}></{JUDGE_XML_TEXT_REFERENCE}></{JUDGE_XML_ROOT_TAG}>"
)

TOOLGEN_JUDGE_LABELED_SCORES_KEY = "_toolgen_judge_labeled_scores"
# Preserved when ``toolgen_group_drop_constant_score_dims`` strips live labeled vectors (for logging).
TOOLGEN_JUDGE_LABELED_SCORES_LOG_CACHE = "_toolgen_judge_labeled_scores_log_cache"
# Full judge call record (Frontier / HTTP ToolGen judges) for SFT replay; stored on every call.
TOOLGEN_JUDGE_TRANSCRIPT_KEY = "_toolgen_judge_transcript"

# Required generic rubric keys always requested in the judge prompt; may be absent from dataset YAML.
REQUIRED_GENERIC_RUBRIC_DIMS = frozenset(
    {
        "prompt_faithfulness",
        "image_quality",
        "text_rendering",
        "ai_naturalness",
        "composition_and_aesthetics",
        "physical_plausibility",
    }
)

DEFAULT_MAJOR_ASPECT_WEIGHTS: Dict[str, float] = {
    "physical_plausibility": 1.5,
}

# If a sub-aspect weight is non-positive, use this value before renormalizing within the group.
FALLBACK_NONPOSITIVE_SUB_WEIGHT = 0.2


def normalize_rubric(rubric: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rubric, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in rubric.items():
        if not isinstance(val, dict):
            continue
        weight = val.get("weight", 1.0)
        if isinstance(weight, (int, float)):
            w = float(weight)
        else:
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = 1.0
        out[str(key)] = {
            "weight": w,
            "description": str(val.get("description", "")),
        }
    return out


def get_reference_count_fields_default(used_count: int) -> Dict[str, int]:
    n = max(0, int(used_count))
    return {
        "expected_reference_images_count": n,
        "selected_reference_images_count": n,
        "used_reference_images_count": n,
    }


def build_visual_reference_context_minimal(has_refs: bool) -> Dict[str, Any]:
    return {
        "has_critical_references": bool(has_refs),
        "critical_candidate_indices": [],
        "critical_candidates": [],
        "eval_text_knowledge_slots": [],
        "has_text_knowledge_gaps_for_eval": False,
    }


def build_reference_image_metadata_map(visual_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    url_to_meta: Dict[str, Dict[str, Any]] = {}
    candidates = visual_context.get("critical_candidates")
    if not isinstance(candidates, list):
        return url_to_meta

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        matched = candidate.get("matched_search_results")
        if not isinstance(matched, list):
            continue
        for match in matched:
            if not isinstance(match, dict):
                continue
            image_url = match.get("image_url")
            if not isinstance(image_url, str):
                continue
            image_url = image_url.strip()
            if not image_url or image_url in url_to_meta:
                continue
            url_to_meta[image_url] = {
                "candidate_index": candidate.get("candidate_index"),
                "entity": candidate.get("entity"),
                "severity": candidate.get("severity"),
                "candidate_reasoning": candidate.get("reasoning"),
                "query": match.get("query"),
                "selected_image_title": match.get("selected_image_title"),
                "source_file": match.get("source_file"),
                "reference_local_path": match.get("local_path"),
            }
    return url_to_meta


def get_reference_count_fields_from_row(row: Dict[str, Any]) -> Dict[str, int]:
    details = row.get("augmented_generation_details")
    if not isinstance(details, dict):
        return get_reference_count_fields_default(0)

    def to_non_negative_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value if value >= 0 else 0
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return 0
        return ivalue if ivalue >= 0 else 0

    return {
        "expected_reference_images_count": to_non_negative_int(
            details.get("expected_reference_images_count", 0)
        ),
        "selected_reference_images_count": to_non_negative_int(
            details.get("selected_reference_images_count", 0)
        ),
        "used_reference_images_count": to_non_negative_int(details.get("used_reference_images_count", 0)),
    }


def build_visual_reference_context_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load judge context from the training row. Expects ``augmented_generation_details`` with:

    - ``eval_reference_slots`` — I2I reference inputs from ``generation_params.json`` plus metadata
      (from dataset build / ToolGen ``build_eval_reference_slots_from_generation_params``).
    - ``eval_text_knowledge_slots`` / ``has_text_knowledge_gaps_for_eval`` — web knowledge gaps
      (from analysis ``knowledge_gaps`` or, when absent, from ``visual_reference_candidates`` with
      ``search_type=web`` and severity important|critical), produced at dataset build
      (``build_eval_text_knowledge_slots_from_analysis`` in ToolGen phase5).
    """
    details = row.get("augmented_generation_details")
    if not isinstance(details, dict):
        return build_visual_reference_context_minimal(False)

    tk_list = details.get("eval_text_knowledge_slots")
    if not isinstance(tk_list, list):
        tk_list = []
    has_tk = len(tk_list) > 0

    def _with_text(
        has_visual: bool,
        *,
        n_vis: int = 0,
        cands: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if cands is None:
            cands = []
        return {
            "has_critical_references": bool(has_visual),
            "critical_candidate_indices": list(range(n_vis)) if has_visual else [],
            "critical_candidates": cands,
            "eval_text_knowledge_slots": tk_list,
            "has_text_knowledge_gaps_for_eval": has_tk,
        }

    slots = details.get("eval_reference_slots")
    if not isinstance(slots, list) or not slots:
        return _with_text(False)

    normalized: List[Dict[str, Any]] = []
    for idx, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        image_url = slot.get("image_url")
        if not isinstance(image_url, str) or not image_url.strip():
            continue
        image_url = image_url.strip()
        matched_row = {
            "query": slot.get("query"),
            "image_url": image_url,
            "local_path": slot.get("reference_local_path") or slot.get("generation_input"),
            "selected_image_title": slot.get("selected_image_title"),
            "source_file": slot.get("source_file"),
        }
        normalized.append(
            {
                "candidate_index": idx,
                "entity": slot.get("entity"),
                "severity": slot.get("severity"),
                "reasoning": slot.get("reasoning"),
                "suggested_search": None,
                "search_type": None,
                "matched_search_results": [matched_row],
            }
        )
    n = len(normalized)
    if n == 0:
        return _with_text(False)
    return _with_text(True, n_vis=n, cands=normalized)


def _slot_reference_focus(
    visual_context: Dict[str, Any],
    slot_index: int,
    url_to_meta: Dict[str, Dict[str, Any]],
    ref_slot_url: str,
) -> Tuple[str, str, str]:
    if isinstance(ref_slot_url, str) and ref_slot_url.startswith("flowfactory://"):
        cands = visual_context.get("critical_candidates")
        if isinstance(cands, list) and 0 <= slot_index < len(cands):
            c = cands[slot_index]
            if isinstance(c, dict):
                entity = str(c.get("entity") or "unknown").strip()
                severity = str(c.get("severity") or "unknown").strip()
                reasoning = str(c.get("reasoning") or "").strip()
                return entity, severity, reasoning or "No reasoning provided."
        return "unknown", "unknown", "No reasoning provided."

    meta = url_to_meta.get(ref_slot_url, {})
    entity = str(meta.get("entity") or "").strip()
    severity = str(meta.get("severity") or "").strip()
    reasoning = str(meta.get("candidate_reasoning") or "").strip()

    # Fallback: if URL didn't match the map, use critical_candidates by slot index
    if not entity:
        cands = visual_context.get("critical_candidates")
        if isinstance(cands, list) and 0 <= slot_index < len(cands):
            c = cands[slot_index]
            if isinstance(c, dict):
                entity = str(c.get("entity") or "unknown").strip()
                severity = str(c.get("severity") or "unknown").strip()
                reasoning = str(c.get("reasoning") or "").strip()

    return entity or "unknown", severity or "unknown", reasoning or "No reasoning provided."


def interleaved_ref_context_block(
    *,
    slot_index: int,
    visual_context: Dict[str, Any],
    url_to_meta: Dict[str, Dict[str, Any]],
    ref_slot_url: str,
) -> str:
    entity, severity, reasoning = _slot_reference_focus(
        visual_context, slot_index, url_to_meta, ref_slot_url
    )
    if len(reasoning) > 220:
        reasoning = reasoning[:217].rstrip() + "..."
    return (
        f"Reference image {slot_index + 1} context:\n"
        f"- entity: {entity}\n"
        f"- severity: {severity}\n"
        f"- why this reference matters: {reasoning or 'No reasoning provided.'}"
    )


def build_eval_prompt_text(
    *,
    row: Dict[str, Any],
    verification_checklist: Sequence[Any],
    evaluation_rubric: Dict[str, Any],
    visual_context: Dict[str, Any],
    variant: str,
    reference_slot_urls: Sequence[str],
    include_physical_plausibility: bool = False,
) -> str:
    """
    Build the judge user-text block (instructions + evaluation context).

    ``row["user_prompt"]`` must be the **original** user-facing task string. ToolGen checklist and
    rubric are defined against that text, and ``prompt_faithfulness`` in the judge protocol means
    alignment with this instruction — not the phase4 **refined** prompt used only as the generator's
    conditioning text (stored separately on training samples as ``prompt`` in JSONL / dataset rows).

    Callers (e.g. :class:`~flow_factory.rewards.toolgen_searchbetter_judge_reward.ToolGenSearchBetterJudgeRewardModel`)
    must keep that split: refined prompt → policy ``prompt``; original → ``user_prompt`` here.
    """
    checklist = list(verification_checklist)
    rubric = normalize_rubric(evaluation_rubric)
    reference_counts = get_reference_count_fields_from_row(row)
    if reference_counts["used_reference_images_count"] <= 0 and len(reference_slot_urls) > 0:
        reference_counts = get_reference_count_fields_default(len(reference_slot_urls))

    user_prompt = str(row.get("user_prompt", "") or "").strip()
    url_to_meta = build_reference_image_metadata_map(visual_context=visual_context)

    image_layout_lines: List[str] = []
    for i, ref_url in enumerate(reference_slot_urls):
        entity, severity, reasoning = _slot_reference_focus(visual_context, i, url_to_meta, ref_url)
        image_layout_lines.append(f"Reference image {i + 1} (entity={entity}): <image>")
    image_layout_lines.append("Image to assess: <image>")
    image_layout_text = "\n".join(image_layout_lines)

    checklist_lines = [f"{j}. {str(item)}" for j, item in enumerate(checklist, start=1)]
    checklist_text = "\n".join(checklist_lines) if checklist_lines else "None provided."

    rubric_lines: List[str] = []
    for dimension, dim_meta in rubric.items():
        desc = str(dim_meta.get("description", "") or "").strip()
        weight = dim_meta.get("weight")
        weight_text = f", weight={weight}" if weight is not None else ""
        rubric_lines.append(f"- {dimension}{weight_text}: {desc or 'No description provided.'}")
    rubric_text = "\n".join(rubric_lines) if rubric_lines else "None provided."
    rubric_text_with_required_generic = (
        f"{rubric_text}\n"
        "- prompt_faithfulness: Required generic — subjects, attributes, actions, scene, and style vs user prompt.\n"
        "- image_quality: Required generic — clarity, artifacts, style/lighting/perspective coherence, plausibility.\n"
        "- text_rendering: Required generic — text accuracy, readability, spelling, font, placement (leave score empty in output if no text required).\n"
        "- ai_naturalness: Required generic — organic realism vs AI smoothness, detail plausibility, grounded environment.\n"
        "- composition_and_aesthetics: Required generic — framing, balance, depth, color harmony, overall appeal."
    )
    if include_physical_plausibility:
        rubric_text_with_required_generic += (
            "\n- physical_plausibility: Required generic — physical plausibility of depicted content. "
            "Evaluate ONLY aspects that should follow physical rules given the user's request. "
            "Ground every claim with specific visual evidence. "
            "Check: (a) Human/creature anatomy — finger count, hand structure, facial symmetry, "
            "limb proportions, joint articulation, body-part connectivity, pose biomechanical "
            "soundness (grip on held objects, weight-bearing stance, natural posture)? "
            "(b) Object physics — gravity, support/balance, weight distribution; do objects float "
            "without support or intersect impossibly? "
            "(c) Spatial consistency — perspective correctness, vanishing points, scale between "
            "entities (character height vs environment, foreground vs background proportions), "
            "occlusion ordering, depth coherence? "
            "(d) Material properties — do metals reflect, glass transmit, fabric drape, liquids "
            "flow correctly? Are surface textures physically plausible under depicted lighting? "
            "(e) Lighting/shadow physics — shadow direction consistent with light source(s), "
            "reflection angles correct, no impossible self-shadowing? "
            "If the request is for abstract, stylized, or intentionally surreal content where "
            "physical rules do not apply, leave score empty in output."
        )

    critical_focus_lines: List[str] = []
    critical_candidates = visual_context.get("critical_candidates")
    if isinstance(critical_candidates, list):
        for candidate in critical_candidates:
            if not isinstance(candidate, dict):
                continue
            entity = str(candidate.get("entity") or "unknown").strip()
            severity = str(candidate.get("severity") or "unknown").strip()
            reasoning = str(candidate.get("reasoning") or "").strip()
            if len(reasoning) > 220:
                reasoning = reasoning[:217].rstrip() + "..."
            critical_focus_lines.append(
                f"- entity='{entity}', severity='{severity}', reason='{reasoning or 'No reasoning provided.'}'"
            )
    critical_focus_text = "\n".join(critical_focus_lines) if critical_focus_lines else "None."

    text_knowledge_lines: List[str] = []
    tk_slots = visual_context.get("eval_text_knowledge_slots")
    if isinstance(tk_slots, list):
        for slot in tk_slots:
            if not isinstance(slot, dict):
                continue
            entity = str(slot.get("entity") or "unknown").strip()
            severity = str(slot.get("severity") or "unknown").strip()
            reasoning = str(slot.get("reasoning") or "").strip()
            sug = str(slot.get("suggested_search") or "").strip()
            if len(reasoning) > 220:
                reasoning = reasoning[:217].rstrip() + "..."
            if len(sug) > 160:
                sug = sug[:157].rstrip() + "..."
            text_knowledge_lines.append(
                f"- entity='{entity}', severity='{severity}', "
                f"suggested_lookup_intent='{sug or 'N/A'}', reason='{reasoning or 'No reasoning provided.'}'"
            )
    text_knowledge_focus_text = "\n".join(text_knowledge_lines) if text_knowledge_lines else "None."
    has_text_knowledge = bool(visual_context.get("has_text_knowledge_gaps_for_eval"))

    # --- Build conditional sections ---
    has_visual_refs = bool(visual_context.get("has_critical_references")) and len(reference_slot_urls) > 0

    visual_ref_section = ""
    if has_visual_refs:
        visual_ref_section = (
            "\nVisual-reference setup:\n"
            f"- attached_reference_images: {len(reference_slot_urls)}\n"
            f"- used_reference_images_count (from generation): {reference_counts.get('used_reference_images_count')}\n"
            "- critical reference focus:\n"
            f"{critical_focus_text}\n"
            "- Guideline: Compare the target image against each reference for identity preservation (facial features, "
            "distinguishing traits), attribute consistency (colors, textures, proportions), and style fidelity. "
            "Penalize missing or altered identity traits; reward faithful reproduction.\n"
        )

    textual_knowledge_section = ""
    if has_text_knowledge and text_knowledge_lines:
        textual_knowledge_section = (
            "\nTextual knowledge gap (textual reference):\n"
            f"- gap_count: {len(text_knowledge_lines)}\n"
            "- gaps to assess (metadata only; no retrieved search text):\n"
            f"{text_knowledge_focus_text}\n"
            "- Guideline: Using the gap metadata (entity, reasoning, suggested lookup intent) and visible evidence "
            "in the target image only, assess whether the image correctly renders the factual/textual content "
            "these gaps describe (labels, numbers, symbols, standards, proper nouns, etc.).\n"
        )

    # --- Build XML output format block ---
    xml_format_block = (
        "XML output blocks:\n"
        f"1) reference_alignment — one {JUDGE_XML_ALIGNMENT_ITEM} per reference slot "
        f"(attribute {JUDGE_XML_REF_ATTR}=\"Reference image k\" matching the image layout): "
        f"{JUDGE_XML_CHECKLIST_LINKS} and {JUDGE_XML_RUBRIC_LINKS} (plain text or multiple {JUDGE_XML_LINK} children "
        "for checklist phrases / rubric dimension names), then reason (how the reference should influence judgment "
        "of the target only).\n"
        f"2) checklist — one {JUDGE_XML_CHECKLIST_ITEM} per verification line (optional {JUDGE_XML_I_ATTR}="
        f"0-based index): {JUDGE_XML_CRITERION} (the checklist line being scored), reason (Step 5b), "
        "score (always present and numeric).\n"
        f"3) rubric — one {JUDGE_XML_DIMENSION} per task-specific and generic row (attribute {JUDGE_XML_NAME_ATTR}="
        "exact dimension name): reason (Step 5b), then score; if N/A, empty <score> and brief reason.\n"
        f"4) visual_reference — {JUDGE_XML_APPLICABLE} true/false; when true: Step 5b reason + score; when false: short reason, score=0.\n"
        f"5) text_reference — {JUDGE_XML_APPLICABLE} true/false; when true: Step 5b reason + score; when false: short reason, score=0.\n\n"
        f"Row ordering: if all {JUDGE_XML_CHECKLIST_ITEM} rows carry a non-empty {JUDGE_XML_I_ATTR} attribute, "
        "ascending index order; otherwise document order.\n\n"
        f"{JUDGE_OUTPUT_XML_SHAPE_FENCED}\n"
    )

    # --- Assemble user message (query-specific only) ---
    parts: List[str] = []

    parts.append(
        "Evaluation context:\n\n"
        "Task prompt:\n"
        f"{user_prompt or 'N/A'}\n\n"
        "Verification checklist:\n"
        f"{checklist_text}\n\n"
        "Evaluation rubric:\n"
        f"{rubric_text_with_required_generic}"
    )

    if visual_ref_section:
        parts.append(visual_ref_section)

    if textual_knowledge_section:
        parts.append(textual_knowledge_section)

    parts.append(f"\n\n{xml_format_block}")

    parts.append(
        f"\nImage layout (slot order):\n{image_layout_text}\n\n"
        f"Return strict XML only (single <{JUDGE_XML_ROOT_TAG}> document; no legacy <{LEGACY_JUDGE_XML_ROOT}>).\n"
    )

    return "\n".join(parts)


def build_judge_interleaved_user_content(
    *,
    user_text_prompt: str,
    reference_slot_urls: Sequence[str],
    reference_image_urls: Sequence[str],
    visual_context: Dict[str, Any],
    assess_image_data_url: str,
    max_pixels: int,
) -> List[Dict[str, Any]]:
    """
    Same multimodal slot order as ``evaluate_searchbetter_hard_direct.prepare_evaluate_one_judge_multimodal``:
    full text block, then per reference (context text + image), then ``Image to assess:`` + target image.
    """
    url_to_meta = build_reference_image_metadata_map(visual_context=visual_context)
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text_prompt}]
    for i, ref_slot in enumerate(reference_slot_urls):
        ctx_text = interleaved_ref_context_block(
            slot_index=i,
            visual_context=visual_context,
            url_to_meta=url_to_meta,
            ref_slot_url=str(ref_slot),
        )
        content.append({"type": "text", "text": ctx_text})
        url = reference_image_urls[i] if i < len(reference_image_urls) else ""
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                f"reference_image_urls[{i}] missing or empty while building judge interleaved content "
                f"(expected {len(reference_slot_urls)} URLs)"
            )
        img_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url.strip()}}
        if max_pixels > 0:
            img_part["max_pixels"] = max_pixels
        content.append(img_part)
    content.append({"type": "text", "text": "Image to assess:"})
    assess_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": assess_image_data_url}}
    if max_pixels > 0:
        assess_part["max_pixels"] = max_pixels
    content.append(assess_part)
    return content


def _strip_assistant_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```xml"):
        cleaned = cleaned[6:]
    elif cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _repair_whitespace_in_xml_close_tags(text: str) -> str:
    r"""
    Normalize ``</ rubric_links>``-style closings to ``</rubric_links>``.

    Invalid XML only; well-formed judge output is unchanged.
    """
    return re.sub(r"</\s+([A-Za-z][\w:-]*)\s*>", r"</\1>", text)


def _extract_balanced_root_fragment(text: str, local_tag: str) -> Optional[Tuple[int, int]]:
    """
    Return ``(start, end)`` slice indices of the first balanced ``<local_tag>...</local_tag>`` subtree.

    Matching is case-insensitive for both open and close. Returns ``None`` if the root is unclosed
    (truncated output).
    """
    tag_re_open = re.compile(rf"<{re.escape(local_tag)}\b[^>]*>", re.IGNORECASE)
    m = tag_re_open.search(text)
    if not m:
        return None
    start = m.start()
    i = m.end()
    depth = 1
    tag_re_open2 = re.compile(rf"<{re.escape(local_tag)}\b[^>]*>", re.IGNORECASE)
    tag_re_close = re.compile(rf"</{re.escape(local_tag)}\s*>", re.IGNORECASE)
    while i < len(text) and depth > 0:
        next_open = tag_re_open2.search(text, i)
        next_close = tag_re_close.search(text, i)
        if next_close is None:
            return None
        next_open_start = next_open.start() if next_open is not None else len(text) + 1
        close_start = next_close.start()
        if next_open is not None and next_open_start < close_start:
            depth += 1
            i = next_open.end()
        else:
            depth -= 1
            if depth == 0:
                return (start, next_close.end())
            i = next_close.end()
    return None


def _extract_first_judge_root_xml_fragment(cleaned: str) -> Optional[str]:
    """
    Extract the first judge root XML document from ``cleaned`` text.

    Prefer the earliest-starting root among ``evaluation``, ``judge_output``, ``tgj``, and
    ``toolgen_judge_output``. Unlike a single backreference regex, open/close tags may differ in
    case (e.g. ``<Evaluation>...</evaluation>``).
    """
    spans: List[Tuple[int, int]] = []
    for local in ("evaluation", "judge_output", "tgj", "toolgen_judge_output"):
        sp = _extract_balanced_root_fragment(cleaned, local)
        if sp is not None:
            spans.append(sp)
    if not spans:
        return None
    spans.sort(key=lambda item: item[0])
    start, end = spans[0]
    return cleaned[start:end]


def _preprocess_judge_xml_text_for_parse(text: str) -> str:
    """
    Strip code fences and apply :func:`_repair_common_judge_xml_typos` before root extraction.

    Keeps behavior predictable for well-formed ``<evaluation>`` output; does not apply ad-hoc
    repairs for malformed model-specific corner cases.
    """
    cleaned = _strip_assistant_code_fences(text)
    cleaned = _repair_whitespace_in_xml_close_tags(cleaned)
    cleaned = _repair_common_judge_xml_typos(cleaned)
    return cleaned


def _repair_malformed_score_closing_tags(text: str) -> str:
    """
    Fix frontier XML where ``</score>`` is truncated to ``</5`` before a real ``</score>``, or to ``</5>``.

    Observed shapes: ``<score>2</5</score>``, ``<score>2</5></score>``, and ``<score>1</5>`` only.
    """
    out = text
    out = re.sub(r"(<score>[^<]*)</5(?=</score>)", r"\1", out)
    out = re.sub(r"(<score>[^<]*)</5>\s*</score>", r"\1</score>", out)
    out = re.sub(r"(<score>[^<]*)</5>", r"\1</score>", out)
    return out


def _repair_broken_evaluation_criterion_tags(text: str) -> str:
    """
    Fix ``<criterion>`` rows where ``</criterion>`` is missing before ``<reason>``, or mangled as ``?</␠…>``.

    Frontier sometimes drops ``</criterion>`` so the line becomes ``...?</`` + spaces + ``<reason>``
    (a truncated ``</...>`` before ``<reason>``), or only ``?`` + spaces + ``<reason>``.
    Another failure mode is ``...</?`` + spaces + ``>`` without a proper closing tag name.
    """
    out = text
    out = re.sub(
        r"(<criterion>[^<]*\?)</\s*<reason>",
        r"\1</criterion>\n      <reason>",
        out,
    )
    out = re.sub(
        r"(<criterion>[^<]*\?)\s+<reason>",
        r"\1</criterion>\n      <reason>",
        out,
    )
    out = re.sub(r"\?</\s+>", r"?</criterion>", out)
    return out


def _repair_common_judge_xml_typos(text: str) -> str:
    """
    Fix tag spellings that otherwise make ElementTree reject otherwise well-formed judge XML.

    Frontier models occasionally emit ``large_text`` instead of ``criterion`` (``<evaluation>`` schema)
    or legacy ``line_text`` (``<judge_output>`` schema).
    """
    out = text
    out = _repair_malformed_score_closing_tags(out)
    out = _repair_broken_evaluation_criterion_tags(out)
    if "large_text" not in out:
        return out
    # Prefer criterion when the document uses the default <evaluation> root (new template).
    if re.search(r"<evaluation\b", out, flags=re.IGNORECASE):
        return out.replace("<large_text>", "<criterion>").replace("</large_text>", "</criterion>")
    return out.replace("<large_text>", "<line_text>").replace("</large_text>", "</line_text>")


def _repair_bare_xml_ampersands_in_fragment(xml_fragment: str) -> str:
    """
    Escape ``&`` that are not the start of a well-formed XML entity or numeric character reference.

    Models often emit natural-language ``A & B`` inside element text or attributes; a bare ``&``
    is invalid XML and makes :func:`xml.etree.ElementTree.fromstring` raise :exc:`ParseError`.
    Known references (``&amp;``, ``&lt;``, ``&#...;``, ``&#x...;``, ``&copy;``, …) are left intact.
    """
    return re.sub(
        r"&(?!([0-9a-zA-Z]+|#[0-9]+|#x[0-9a-fA-F]+);)",
        "&amp;",
        xml_fragment,
    )


def _element_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _parse_applicable_bool(raw: str) -> bool:
    t = str(raw or "").strip().lower()
    if t in ("true", "1", "yes"):
        return True
    if t in ("false", "0", "no", ""):
        return False
    return False


def _container_child_texts(container: Optional[ET.Element]) -> List[str]:
    if container is None:
        return []
    return [_element_text(ch) for ch in list(container)]


def _linked_link_texts(container: Optional[ET.Element], *, line_tag: str) -> List[str]:
    if container is None:
        return []
    return [_element_text(ch) for ch in list(container) if ch.tag == line_tag]


def _links_container_phrases(container: Optional[ET.Element], *, link_tag: str = JUDGE_XML_LINK) -> List[str]:
    """
    Collect checklist/rubric link phrases from ``<checklist_links>`` / ``<rubric_links>``.

    Accepts either multiple ``<link>`` children (legacy style) or plain text / newline-separated
    phrases inside the container element.
    """
    if container is None:
        return []
    link_texts = _linked_link_texts(container, line_tag=link_tag)
    if link_texts:
        return [t for t in (x.strip() for x in link_texts) if t]
    raw = _element_text(container)
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        return lines
    return [raw.strip()]


def _checklist_item_criterion_text(ent: ET.Element) -> str:
    crit_el = ent.find(JUDGE_XML_CRITERION)
    if crit_el is None:
        crit_el = ent.find(LEGACY_JUDGE_XML_LINE_TEXT)
    return _element_text(crit_el)


def _visual_ref_block_from_evaluation_root(root: ET.Element) -> Optional[ET.Element]:
    vre = root.find(JUDGE_XML_VISUAL_REFERENCE)
    if vre is not None:
        return vre
    return root.find(LEGACY_JUDGE_XML_VISUAL_REF)


def _visual_ref_applicable_element(vre_root: ET.Element) -> Optional[ET.Element]:
    for tag in (JUDGE_XML_APPLICABLE, LEGACY_JUDGE_XML_APPLIES):
        el = vre_root.find(tag)
        if el is not None:
            return el
    return None


def _xml_attr_i_sort_key(el: ET.Element) -> int:
    raw = el.get(JUDGE_XML_I_ATTR)
    if raw is None or str(raw).strip() == "":
        return 0
    try:
        return int(str(raw).strip())
    except ValueError:
        return 0


def _xml_row_children_ordered(parent: Optional[ET.Element], row_tag: str) -> List[ET.Element]:
    if parent is None:
        return []
    rows = [c for c in list(parent) if c.tag == row_tag]
    if not rows:
        return []
    if all(
        c.get(JUDGE_XML_I_ATTR) is not None and str(c.get(JUDGE_XML_I_ATTR) or "").strip() != ""
        for c in rows
    ):
        return sorted(rows, key=_xml_attr_i_sort_key)
    return rows


def _root_local_tag(root: ET.Element) -> str:
    if "}" in root.tag:
        return str(root.tag).split("}", 1)[-1]
    return str(root.tag)


def _parse_evaluation_format_to_dict(root: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse ``<evaluation>`` (reference_alignment / checklist / rubric / visual_reference / text_reference)."""
    out: Dict[str, Any] = {
        "reference_criterion_alignment": [],
        "checklist_scores": [],
        "rubric_scores": {},
    }

    ral = root.find(JUDGE_XML_REFERENCE_ALIGNMENT)
    if ral is not None:
        for ent in _xml_row_children_ordered(ral, JUDGE_XML_ALIGNMENT_ITEM):
            ref_attr = ent.get(JUDGE_XML_REF_ATTR)
            ref_label = str(ref_attr).strip() if ref_attr else ""
            if not ref_label:
                ref_label_el = ent.find(LEGACY_JUDGE_XML_REF_LABEL)
                ref_label = _element_text(ref_label_el)
            cl_el = ent.find(JUDGE_XML_CHECKLIST_LINKS)
            rb_el = ent.find(JUDGE_XML_RUBRIC_LINKS)
            lcheck_legacy = ent.find(LEGACY_JUDGE_XML_LINKED_CL)
            lrub_legacy = ent.find(LEGACY_JUDGE_XML_LINKED_RUB)
            if cl_el is None and lcheck_legacy is not None:
                cl_el = lcheck_legacy
            if rb_el is None and lrub_legacy is not None:
                rb_el = lrub_legacy
            w_el = ent.find(JUDGE_XML_REASON)
            if w_el is None:
                w_el = ent.find(LEGACY_JUDGE_XML_REF_REASON)
            linked_cl = _links_container_phrases(cl_el)
            if not linked_cl and lcheck_legacy is not None:
                linked_cl = _linked_link_texts(lcheck_legacy, line_tag=JUDGE_XML_LINK)
            linked_rb = _links_container_phrases(rb_el)
            if not linked_rb and lrub_legacy is not None:
                linked_rb = _linked_link_texts(lrub_legacy, line_tag=JUDGE_XML_LINK)
            out["reference_criterion_alignment"].append(
                {
                    "reference_image": ref_label,
                    "linked_checklist_items": linked_cl,
                    "linked_rubric_dimensions": linked_rb,
                    "reason": _element_text(w_el),
                }
            )

    cs = root.find(JUDGE_XML_CHECKLIST)
    if cs is not None:
        for ent in _xml_row_children_ordered(cs, JUDGE_XML_CHECKLIST_ITEM):
            w_el = ent.find(JUDGE_XML_REASON)
            s_el = ent.find(JUDGE_XML_SCORE)
            out["checklist_scores"].append(
                {
                    "item": _checklist_item_criterion_text(ent),
                    "score": _element_text(s_el),
                    "reason": _element_text(w_el),
                }
            )
        if not out["checklist_scores"]:
            for ent in _xml_row_children_ordered(cs, LEGACY_JUDGE_XML_CL_ROW):
                text_el = ent.find(LEGACY_JUDGE_XML_LINE_TEXT)
                w_el = ent.find(JUDGE_XML_REASON)
                s_el = ent.find(JUDGE_XML_SCORE)
                out["checklist_scores"].append(
                    {
                        "item": _element_text(text_el),
                        "score": _element_text(s_el),
                        "reason": _element_text(w_el),
                    }
                )

    rub = root.find(JUDGE_XML_RUBRIC)
    if rub is not None:
        for ch in rub:
            if ch.tag not in (JUDGE_XML_DIMENSION, LEGACY_JUDGE_XML_RUBRIC_LINE):
                continue
            dim = ch.get(JUDGE_XML_NAME_ATTR)
            if not isinstance(dim, str) or not dim.strip():
                continue
            w_el = ch.find(JUDGE_XML_REASON)
            s_el = ch.find(JUDGE_XML_SCORE)
            out["rubric_scores"][dim.strip()] = {
                "score": _element_text(s_el),
                "reason": _element_text(w_el),
            }

    vre_root = _visual_ref_block_from_evaluation_root(root)
    if vre_root is not None:
        app_el = _visual_ref_applicable_element(vre_root)
        w_el = vre_root.find(JUDGE_XML_REASON)
        s_el = vre_root.find(JUDGE_XML_SCORE)
        out["visual_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(app_el)),
            "score": _element_text(s_el),
            "reason": _element_text(w_el),
        }

    tre_root = root.find(JUDGE_XML_TEXT_REFERENCE)
    if tre_root is not None:
        app_el = _visual_ref_applicable_element(tre_root)
        w_el = tre_root.find(JUDGE_XML_REASON)
        s_el = tre_root.find(JUDGE_XML_SCORE)
        out["text_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(app_el)),
            "score": _element_text(s_el),
            "reason": _element_text(w_el),
        }

    if not out["rubric_scores"] and not out["checklist_scores"]:
        return None
    return out


def _parse_judge_output_legacy_to_dict(root: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse legacy ``<judge_output>`` (ref_align / cl_row / rubric_line / visual_ref)."""
    out: Dict[str, Any] = {
        "reference_criterion_alignment": [],
        "checklist_scores": [],
        "rubric_scores": {},
    }

    al = root.find(LEGACY_JUDGE_XML_REF_ALIGN)
    if al is not None:
        for ent in _xml_row_children_ordered(al, LEGACY_JUDGE_XML_REF_ROW):
            lcheck = ent.find(LEGACY_JUDGE_XML_LINKED_CL)
            lrub = ent.find(LEGACY_JUDGE_XML_LINKED_RUB)
            ref_label = ent.find(LEGACY_JUDGE_XML_REF_LABEL)
            w_el = ent.find(LEGACY_JUDGE_XML_REF_REASON)
            out["reference_criterion_alignment"].append(
                {
                    "reference_image": _element_text(ref_label),
                    "linked_checklist_items": _linked_link_texts(lcheck, line_tag=JUDGE_XML_LINK),
                    "linked_rubric_dimensions": _linked_link_texts(lrub, line_tag=JUDGE_XML_LINK),
                    "reason": _element_text(w_el),
                }
            )

    cs = root.find(JUDGE_XML_CHECKLIST)
    if cs is not None:
        for ent in _xml_row_children_ordered(cs, LEGACY_JUDGE_XML_CL_ROW):
            text_el = ent.find(LEGACY_JUDGE_XML_LINE_TEXT)
            w_el = ent.find(JUDGE_XML_REASON)
            s_el = ent.find(JUDGE_XML_SCORE)
            out["checklist_scores"].append(
                {
                    "item": _element_text(text_el),
                    "score": _element_text(s_el),
                    "reason": _element_text(w_el),
                }
            )

    rub = root.find(JUDGE_XML_RUBRIC)
    if rub is not None:
        for ch in rub:
            if ch.tag != LEGACY_JUDGE_XML_RUBRIC_LINE:
                continue
            dim = ch.get(JUDGE_XML_NAME_ATTR)
            if not isinstance(dim, str) or not dim.strip():
                continue
            w_el = ch.find(JUDGE_XML_REASON)
            s_el = ch.find(JUDGE_XML_SCORE)
            out["rubric_scores"][dim.strip()] = {
                "score": _element_text(s_el),
                "reason": _element_text(w_el),
            }

    vre_root = root.find(LEGACY_JUDGE_XML_VISUAL_REF)
    if vre_root is not None:
        app_el = vre_root.find(LEGACY_JUDGE_XML_APPLIES)
        w_el = vre_root.find(JUDGE_XML_REASON)
        s_el = vre_root.find(JUDGE_XML_SCORE)
        out["visual_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(app_el)),
            "score": _element_text(s_el),
            "reason": _element_text(w_el),
        }

    if not out["rubric_scores"] and not out["checklist_scores"]:
        return None
    return out


def _parse_tgj_abbrev_to_dict(root: ET.Element) -> Optional[Dict[str, Any]]:
    """Legacy short-tag ``tgj`` block (al/rf/…/p) for older in-flight model outputs."""
    al = "al"
    rf = "rf"
    m = "m"
    lci = "lci"
    lrd = "lrd"
    q = "q"
    cs = "cs"
    ck = "ck"
    t = "t"
    w = "w"
    s = "s"
    rs = "rs"
    b = "b"
    n = "n"
    ve = "ve"
    p = "p"

    out: Dict[str, Any] = {
        "reference_criterion_alignment": [],
        "checklist_scores": [],
        "rubric_scores": {},
    }

    al_el = root.find(al)
    if al_el is not None:
        for ent in _xml_row_children_ordered(al_el, rf):
            lcheck = ent.find(lci)
            lrub = ent.find(lrd)
            m_el = ent.find(m)
            w_el = ent.find(w)
            out["reference_criterion_alignment"].append(
                {
                    "reference_image": _element_text(m_el),
                    "linked_checklist_items": _linked_link_texts(lcheck, line_tag=q),
                    "linked_rubric_dimensions": _linked_link_texts(lrub, line_tag=q),
                    "reason": _element_text(w_el),
                }
            )

    cs_el = root.find(cs)
    if cs_el is not None:
        for ent in _xml_row_children_ordered(cs_el, ck):
            t_el = ent.find(t)
            w_el = ent.find(w)
            s_el = ent.find(s)
            out["checklist_scores"].append(
                {
                    "item": _element_text(t_el),
                    "score": _element_text(s_el),
                    "reason": _element_text(w_el),
                }
            )

    rub = root.find(rs)
    if rub is not None:
        for ch in rub:
            if ch.tag != b:
                continue
            dim = ch.get(n)
            if not isinstance(dim, str) or not dim.strip():
                continue
            w_el = ch.find(w)
            s_el = ch.find(s)
            out["rubric_scores"][dim.strip()] = {
                "score": _element_text(s_el),
                "reason": _element_text(w_el),
            }

    vre_root = root.find(ve)
    if vre_root is not None:
        p_el = vre_root.find(p)
        w_el = vre_root.find(w)
        s_el = vre_root.find(s)
        out["visual_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(p_el)),
            "score": _element_text(s_el),
            "reason": _element_text(w_el),
        }

    if not out["rubric_scores"] and not out["checklist_scores"]:
        return None
    return out


def _parse_judge_xml_verbose_to_dict(root: ET.Element) -> Optional[Dict[str, Any]]:
    """Long-tag XML: ``toolgen_judge_output`` with ``ref_entry`` / ``cl_entry`` (optional ``i`` on ``cl_entry``)."""
    out: Dict[str, Any] = {
        "reference_criterion_alignment": [],
        "checklist_scores": [],
        "rubric_scores": {},
    }

    rca = root.find("reference_criterion_alignment")
    if rca is not None:
        for ent in rca:
            if ent.tag != "ref_entry":
                continue
            ref_img_el = ent.find("reference_image")
            reason_el = ent.find("reason")
            lcheck = ent.find("linked_checklist_items")
            lrub = ent.find("linked_rubric_dimensions")
            out["reference_criterion_alignment"].append(
                {
                    "reference_image": _element_text(ref_img_el),
                    "linked_checklist_items": _container_child_texts(lcheck),
                    "linked_rubric_dimensions": _container_child_texts(lrub),
                    "reason": _element_text(reason_el),
                }
            )

    cs = root.find("checklist_scores")
    if cs is not None:
        for ent in _xml_row_children_ordered(cs, "cl_entry"):
            text_el = ent.find("checklist_text")
            score_el = ent.find("score")
            reason_el = ent.find("reason")
            out["checklist_scores"].append(
                {
                    "item": _element_text(text_el),
                    "score": _element_text(score_el),
                    "reason": _element_text(reason_el),
                }
            )

    rub = root.find("rubric_scores")
    if rub is not None:
        for ch in rub:
            if ch.tag != "rubric":
                continue
            dim = ch.get("name")
            if not isinstance(dim, str) or not dim.strip():
                continue
            score_el = ch.find("score")
            reason_el = ch.find("reason")
            out["rubric_scores"][dim.strip()] = {
                "score": _element_text(score_el),
                "reason": _element_text(reason_el),
            }

    vre_root = root.find("visual_reference_evaluation")
    if vre_root is not None:
        app_el = vre_root.find("applicable")
        score_el = vre_root.find("score")
        reason_el = vre_root.find("reason")
        out["visual_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(app_el)),
            "score": _element_text(score_el),
            "reason": _element_text(reason_el),
        }

    tre_root = root.find("text_reference_evaluation")
    if tre_root is not None:
        app_el = tre_root.find("applicable")
        score_el = tre_root.find("score")
        reason_el = tre_root.find("reason")
        out["text_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(app_el)),
            "score": _element_text(score_el),
            "reason": _element_text(reason_el),
        }

    if not out["rubric_scores"] and not out["checklist_scores"]:
        return None
    return out


def parse_judge_xml_to_dict(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse judge XML (``evaluation`` default, legacy ``judge_output``, ``tgj`` abbrev, or long
    ``toolgen_judge_output``) into the same dict shape as legacy JSON (``reference_criterion_alignment``,
    ``checklist_scores``, ``rubric_scores``, ``visual_reference_evaluation``, ``text_reference_evaluation``) for
    :func:`parsed_judge_labeled_scores_03`.

    The root element is extracted with case-insensitive matching on the closing tag so a
    well-formed document such as ``<Evaluation>...</evaluation>`` still parses (the previous
    single-regex path required identical casing on the close tag).
    """
    cleaned = _preprocess_judge_xml_text_for_parse(text)
    fragment = _extract_first_judge_root_xml_fragment(cleaned)
    if not fragment:
        return None
    fragment = _repair_bare_xml_ampersands_in_fragment(fragment)
    try:
        root = ET.fromstring(fragment)
    except ET.ParseError:
        return None

    tag = _root_local_tag(root).casefold()
    if tag == JUDGE_XML_ROOT_TAG.casefold():
        return _parse_evaluation_format_to_dict(root)
    if tag == LEGACY_JUDGE_XML_ROOT.casefold():
        return _parse_judge_output_legacy_to_dict(root)
    if tag == "tgj":
        return _parse_tgj_abbrev_to_dict(root)
    if tag == "toolgen_judge_output":
        return _parse_judge_xml_verbose_to_dict(root)
    return None


def try_parse_judge_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort parse of a judge message: try strict XML (``<evaluation>``, legacy ``<judge_output>``,
    ``<tgj>``, or ``<toolgen_judge_output>``) first, then legacy JSON (including fenced blocks and
    brace-bounded JSON).
    """
    xml_parsed = parse_judge_xml_to_dict(text)
    if xml_parsed is not None:
        return xml_parsed
    return try_extract_json_obj(text)


def try_extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    candidate = text[start:end]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _score_to_float_03(x: Any, *, ctx: str) -> float:
    if isinstance(x, bool):
        raise TypeError(f"{ctx}: expected numeric score, got bool")
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)", x.strip())
        if not m:
            raise ValueError(f"{ctx}: could not parse score from string {x!r}")
        v = float(m.group(1))
    else:
        raise TypeError(f"{ctx}: expected int/float/str score, got {type(x).__name__}: {x!r}")
    if v < 0.0 or v > 3.0:
        raise ValueError(f"{ctx}: score must be in [0, 3], got {v}")
    return v


def toolgen_dpo_decompose_major_rest_native03(
    labeled: List[Tuple[str, float]],
) -> Tuple[float, float]:
    """
    Decompose ToolGen judge labeled scores for DPO pair **selection** (not reward replacement).

    **Major total** (native ``[0, 3]`` per component, same shaping as training logs):
    ``prompt_faithfulness`` + mean(task rubric / adaptive rows) + mean(checklist items).

    **Rest total**: sum of all other native scores — generic rubric dims except
    ``prompt_faithfulness``, plus ``visual_reference_evaluation`` and
    ``text_reference_evaluation``, and any unrecognized keys.

    Returns:
        (major_total, rest_sum) both in cumulative native scale (major up to ~9 if all max;
        rest scales with number of minor dimensions).
    """
    if not isinstance(labeled, list) or not labeled:
        raise ValueError(f"labeled must be non-empty list, got {labeled!r}")

    pf = 0.0
    pf_seen = False
    adaptive_vals: List[float] = []
    checklist_vals: List[float] = []
    rest_vals: List[float] = []

    for k, v in labeled:
        if not isinstance(k, str):
            raise TypeError(f"labeled key must be str, got {type(k).__name__}: {k!r}")
        if k.startswith("checklist:"):
            checklist_vals.append(float(v))
            continue
        if k.startswith("rubric:"):
            dim = k[len("rubric:") :]
            if dim == "prompt_faithfulness":
                pf = float(v)
                pf_seen = True
            elif dim in REQUIRED_GENERIC_RUBRIC_DIMS:
                rest_vals.append(float(v))
            else:
                adaptive_vals.append(float(v))
            continue
        if k in ("visual_reference_evaluation", "text_reference_evaluation"):
            rest_vals.append(float(v))
            continue
        rest_vals.append(float(v))

    if not pf_seen:
        pf = 0.0
    ad_m = sum(adaptive_vals) / len(adaptive_vals) if adaptive_vals else 0.0
    ch_m = sum(checklist_vals) / len(checklist_vals) if checklist_vals else 0.0
    major_total = float(pf) + float(ad_m) + float(ch_m)
    rest_sum = float(sum(rest_vals)) if rest_vals else 0.0
    return major_total, rest_sum


def parsed_judge_labeled_scores_03(parsed: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Extract ordered (key, score) pairs from judge JSON or XML-parsed dicts.

    Keys are stable across samples that share the same rubric/checklist layout:
    ``checklist:{i}``, ``rubric:{dim}``, and optionally ``visual_reference_evaluation`` / ``text_reference_evaluation``.
    Scores are on the model's native ``[0, 3]`` scale. A rubric dimension is **skipped** when ``score`` is
    missing, empty, or legacy ``N/A``/``NA``; checklist scores must be present and non-empty; when
    ``visual_reference_evaluation.applicable`` or ``text_reference_evaluation.applicable`` is true, an empty score is skipped.
    """
    out: List[Tuple[str, float]] = []

    checklist = parsed.get("checklist_scores")
    if isinstance(checklist, list):
        for idx, item in enumerate(checklist):
            if not isinstance(item, dict):
                raise TypeError(
                    f"checklist_scores[{idx}] expected dict, got {type(item).__name__}: {item!r}"
                )
            if "score" not in item:
                raise KeyError(f"checklist_scores[{idx}] missing 'score' key: {item!r}")
            raw_chk = item["score"]
            if raw_chk is None or (isinstance(raw_chk, str) and raw_chk.strip() == ""):
                raise ValueError(
                    f"checklist_scores[{idx}].score must be a numeric value in [0, 3] (empty score not allowed for checklist); "
                    f"got {raw_chk!r}"
                )
            v = _score_to_float_03(raw_chk, ctx=f"checklist_scores[{idx}].score")
            out.append((f"checklist:{idx}", v))

    rubric_scores = parsed.get("rubric_scores")
    if rubric_scores is None:
        rubric_scores = {}
    if not isinstance(rubric_scores, dict):
        raise TypeError(
            f"rubric_scores expected dict, got {type(rubric_scores).__name__}: {rubric_scores!r}"
        )
    for dim, payload in rubric_scores.items():
        if not isinstance(payload, dict):
            raise TypeError(
                f"rubric_scores[{dim!r}] expected dict, got {type(payload).__name__}: {payload!r}"
            )
        if "score" not in payload:
            continue
        raw_score = payload["score"]
        if raw_score is None or (isinstance(raw_score, str) and raw_score.strip() == ""):
            continue
        if isinstance(raw_score, str) and raw_score.strip().upper() in {"N/A", "NA"}:
            continue
        v = _score_to_float_03(raw_score, ctx=f"rubric_scores[{dim}].score")
        out.append((f"rubric:{dim}", v))

    vre = parsed.get("visual_reference_evaluation")
    if isinstance(vre, dict) and vre.get("applicable") is True:
        if "score" in vre:
            raw_v = vre["score"]
            if raw_v is not None and not (isinstance(raw_v, str) and raw_v.strip() == ""):
                v = _score_to_float_03(raw_v, ctx="visual_reference_evaluation.score")
                out.append(("visual_reference_evaluation", v))

    tre = parsed.get("text_reference_evaluation")
    if isinstance(tre, dict) and tre.get("applicable") is True:
        if "score" in tre:
            raw_t = tre["score"]
            if raw_t is not None and not (isinstance(raw_t, str) and raw_t.strip() == ""):
                tsc = _score_to_float_03(raw_t, ctx="text_reference_evaluation.score")
                out.append(("text_reference_evaluation", tsc))

    if not out:
        raise ValueError(
            "parsed judge JSON produced no scores (empty checklist_scores and rubric_scores); "
            f"keys present: {list(parsed.keys())}"
        )
    return out


def _renormalize_sub_weights(raw: Sequence[float]) -> List[float]:
    """
    Per checklist or per rubric group: any non-positive weight is replaced with
    :data:`FALLBACK_NONPOSITIVE_SUB_WEIGHT`, then weights are scaled to sum to ``1.0``.
    If the sum is still non-positive, fall back to equal weights.
    """
    if not raw:
        return []
    w: List[float] = []
    for x in raw:
        xf = float(x)
        if xf <= 0.0:
            w.append(float(FALLBACK_NONPOSITIVE_SUB_WEIGHT))
        else:
            w.append(xf)
    s = float(sum(w))
    n = len(w)
    if s <= 0.0 and n:
        u = 1.0 / float(n)
        return [u] * n
    if s <= 0.0:
        return []
    return [x / s for x in w]


def _adaptive_rubric_subaspect_raw_weight(dim: str, rubric_norm: Dict[str, Dict[str, Any]]) -> float:
    """
    Weights for **task (adaptive)** rubric rows only. ``dim`` must not be in
    :data:`REQUIRED_GENERIC_RUBRIC_DIMS`.
    """
    if dim in rubric_norm:
        return float(rubric_norm[dim].get("weight", 1.0))
    return 1.0


def parse_toolgen_judge_major_aspect_weights_cfg(raw: Any) -> Optional[Dict[str, float]]:
    """
    Parse ``toolgen_judge_major_aspect_weights`` from reward ``extra_kwargs``.
    ``None`` or empty dict -> ``None`` (equal weighting across major terms).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            f"toolgen_judge_major_aspect_weights must be a dict or null, got {type(raw).__name__}: {raw!r}"
        )
    if not raw:
        return None
    out: Dict[str, float] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            raise ValueError(f"toolgen_judge_major_aspect_weights has empty key: {raw!r}")
        try:
            out[key] = float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"toolgen_judge_major_aspect_weights[{k!r}] must be numeric, got {v!r}"
            ) from e
    return out


TOOLGEN_JUDGE_AUTO_REWARD_WEIGHT_DEFAULT_TEMPERATURE = 1.0
TOOLGEN_JUDGE_AUTO_REWARD_WEIGHT_DEFAULT_MIN_FLOOR = 0.02


def parse_toolgen_judge_auto_reward_weight_cfg(
    extra_kwargs: Optional[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """
    Parse the ``auto_reward_weight*`` block from a reward's ``extra_kwargs``.

    Returns ``None`` when ``auto_reward_weight`` is missing / falsy. Otherwise returns a small dict
    ``{temperature, min_floor}`` of validated floats. Temperature is unit-free (the implementation
    normalizes the variance vector by its mean before applying the softmax).
    """
    if not extra_kwargs:
        return None
    raw_enable = extra_kwargs.get("auto_reward_weight", False)
    if not bool(raw_enable):
        return None

    temperature = float(
        extra_kwargs.get(
            "auto_reward_weight_softmax_temperature",
            TOOLGEN_JUDGE_AUTO_REWARD_WEIGHT_DEFAULT_TEMPERATURE,
        )
    )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"auto_reward_weight_softmax_temperature must be a positive finite float, got {temperature!r}"
        )

    min_floor = float(
        extra_kwargs.get(
            "auto_reward_weight_min_floor",
            TOOLGEN_JUDGE_AUTO_REWARD_WEIGHT_DEFAULT_MIN_FLOOR,
        )
    )
    if not math.isfinite(min_floor) or min_floor < 0.0 or min_floor >= 1.0:
        raise ValueError(
            f"auto_reward_weight_min_floor must be in [0.0, 1.0), got {min_floor!r}"
        )

    return {"temperature": temperature, "min_floor": min_floor}


def softmax_over_variances(
    variances: Sequence[float],
    *,
    temperature: float,
    min_floor: float = 0.0,
) -> List[float]:
    """
    Softmax of per-aspect variances, with the unit-free convention used by the auto reward weighter.

    Steps: divide ``variances`` by ``mean(variances)`` (skipped if the mean is zero), divide by
    ``temperature``, subtract the per-row max for numerical stability, exponentiate, normalize to
    sum 1. If ``min_floor > 0``, clamp each weight up to ``min_floor`` and renormalize; if the
    floor is infeasible (``min_floor * K >= 1``), every aspect receives ``1/K``.
    """
    k = len(variances)
    if k == 0:
        return []
    if k == 1:
        return [1.0]

    v = [float(x) for x in variances]
    if any(not math.isfinite(x) for x in v):
        return [1.0 / k] * k

    if all(x <= 0.0 for x in v):
        out = [1.0 / k] * k
    else:
        mean_v = sum(v) / float(k)
        if mean_v > 0.0:
            v = [x / mean_v for x in v]
        scaled = [x / float(temperature) for x in v]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        s = sum(exps)
        if s <= 0.0 or not math.isfinite(s):
            out = [1.0 / k] * k
        else:
            out = [e / s for e in exps]

    if min_floor > 0.0:
        if min_floor * k >= 1.0:
            return [1.0 / k] * k
        min_w = min(out)
        if min_w < min_floor:
            # Convex-combine with uniform 1/k until min(out) == min_floor exactly. Both inputs
            # sum to 1, so the result sums to 1 by construction; no second renormalization.
            uniform = 1.0 / k
            alpha = (min_floor - min_w) / (uniform - min_w)
            alpha = max(0.0, min(1.0, alpha))
            out = [(1.0 - alpha) * w + alpha * uniform for w in out]
    return out


def compute_auto_reward_weights_for_group(
    major_terms_list: Sequence[Sequence[Tuple[str, float]]],
    *,
    temperature: float,
    min_floor: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Per-group softmax-over-variance weights for ToolGen judge major aspects.

    ``major_terms_list``: one ``[(aspect_name, score_03), ...]`` per successful sample in the group.

    Computes per-aspect (across-sample) variances over the **intersection** of aspects present in
    every sample (so the same weights apply consistently to every sample's reward). Variances are
    biased ``E[(x - mean)^2]`` (matches the population std used elsewhere in this module).

    Returns ``(weights, variances)`` keyed by aspect name. If fewer than 2 samples are provided or
    the intersection is empty, returns ``({}, {})`` to signal "fall back to manual weights".
    """
    n = len(major_terms_list)
    if n < 2:
        return {}, {}
    first_keys = [k for k, _ in major_terms_list[0]]
    intersect: List[str] = []
    for k in first_keys:
        if all(any(kk == k for kk, _ in mt) for mt in major_terms_list):
            intersect.append(k)
    if not intersect:
        return {}, {}

    variances: Dict[str, float] = {}
    for k in intersect:
        col = [float(next(v for kk, v in mt if kk == k)) for mt in major_terms_list]
        mean = sum(col) / float(n)
        variances[k] = sum((x - mean) * (x - mean) for x in col) / float(n)

    weights_vec = softmax_over_variances(
        [variances[k] for k in intersect],
        temperature=temperature,
        min_floor=min_floor,
    )
    weights = {k: float(w) for k, w in zip(intersect, weights_vec)}
    return weights, variances


def _raw_weight_for_major_term(term_key: str, weights: Dict[str, float]) -> float:
    """
    Resolve configured weight for one **major** aggregate term. Missing keys default to ``1.0``.

    Aliases (optional, for YAML ergonomics):
    ``adaptive_rubric`` -> ``rubric_adaptive``; ``visual_reference`` -> ``visual_reference_evaluation``;
    ``text_reference`` -> ``text_reference_evaluation``.
    """
    if term_key in weights:
        return float(weights[term_key])
    if term_key == "rubric_adaptive" and "adaptive_rubric" in weights:
        return float(weights["adaptive_rubric"])
    if term_key == "visual_reference_evaluation" and "visual_reference" in weights:
        return float(weights["visual_reference"])
    if term_key == "text_reference_evaluation" and "text_reference" in weights:
        return float(weights["text_reference"])
    return float(DEFAULT_MAJOR_ASPECT_WEIGHTS.get(term_key, 1.0))


def labeled_to_major_terms(
    labeled: Sequence[Tuple[str, float]],
    evaluation_rubric: Optional[Dict[str, Any]] = None,
    *,
    checklist_item_weight: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    Reduce a labeled score list (``parsed_judge_labeled_scores_03`` output) to the per-major-aspect
    scalar list on ``[0, 3]``. The adaptive rubric block and checklist block are each collapsed to
    one scalar via the same sub-aspect renormalization used by
    :func:`parsed_judge_json_to_weighted_reward`. Generic rubric dims, visual_reference_evaluation,
    and text_reference_evaluation each become one major term.
    """
    rubric_norm = normalize_rubric(evaluation_rubric or {})
    by_key: Dict[str, float] = {}
    rubric_adaptive: List[Tuple[str, float]] = []
    checklist_items: List[Tuple[str, float]] = []
    vre_val: Optional[float] = None
    tre_val: Optional[float] = None
    for k, v in labeled:
        if k == "visual_reference_evaluation":
            vre_val = float(v)
        elif k == "text_reference_evaluation":
            tre_val = float(v)
        elif k.startswith("rubric:"):
            dim = k[len("rubric:") :]
            by_key[k] = float(v)
            if dim in REQUIRED_GENERIC_RUBRIC_DIMS:
                continue
            rubric_adaptive.append((k, float(v)))
        elif k.startswith("checklist:"):
            checklist_items.append((k, float(v)))
        else:
            raise ValueError(f"unexpected labeled score key: {k!r}")

    major_terms: List[Tuple[str, float]] = []
    if rubric_adaptive:
        raws = [
            _adaptive_rubric_subaspect_raw_weight(k[len("rubric:") :], rubric_norm)
            for k, _ in rubric_adaptive
        ]
        if any(float(x) <= 0.0 for x in raws):
            logger.warning(
                "ToolGen judge reward: at least one adaptive rubric sub-aspect weight is non-positive; "
                "using %s and renormalizing to sum 1.0",
                FALLBACK_NONPOSITIVE_SUB_WEIGHT,
            )
        wn = _renormalize_sub_weights(raws)
        scores_r = [v for _, v in rubric_adaptive]
        major_terms.append(("rubric_adaptive", float(sum(wi * si for wi, si in zip(wn, scores_r)))))
    if checklist_items:
        raws = [float(checklist_item_weight) for _ in checklist_items]
        if any(float(x) <= 0.0 for x in raws):
            logger.warning(
                "ToolGen judge reward: checklist_item_weight is non-positive; using %s and "
                "renormalizing to sum 1.0",
                FALLBACK_NONPOSITIVE_SUB_WEIGHT,
            )
        wn = _renormalize_sub_weights(raws)
        scores_c = [v for _, v in checklist_items]
        major_terms.append(("checklist", float(sum(wi * si for wi, si in zip(wn, scores_c)))))
    for dim in sorted(REQUIRED_GENERIC_RUBRIC_DIMS):
        rk = f"rubric:{dim}"
        if rk in by_key:
            major_terms.append((dim, float(by_key[rk])))
    if vre_val is not None:
        major_terms.append(("visual_reference_evaluation", float(vre_val)))
    if tre_val is not None:
        major_terms.append(("text_reference_evaluation", float(tre_val)))
    return major_terms


def weighted_reward_from_major_terms(
    major_terms: Sequence[Tuple[str, float]],
    major_aspect_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Final aggregate from per-major-aspect scalars on ``[0, 3]`` to a reward on ``[0, 1]``.

    With ``major_aspect_weights=None``: unweighted mean of the present terms, divided by 3.
    With a dict: per-term raw weight (default ``1.0`` for omitted keys) via
    :func:`_raw_weight_for_major_term`, renormalized over **present** terms with the
    non-positive fallback from :func:`_renormalize_sub_weights`, then weighted sum divided by 3.
    """
    if not major_terms:
        raise ValueError("ToolGen judge reward: no score components in labeled output")
    if major_aspect_weights:
        raws_maj = [
            _raw_weight_for_major_term(tk, major_aspect_weights) for tk, _ in major_terms
        ]
        if any(float(x) <= 0.0 for x in raws_maj):
            logger.warning(
                "ToolGen judge reward: at least one major_aspect_weights entry is non-positive; "
                "using %s and renormalizing to sum 1.0",
                FALLBACK_NONPOSITIVE_SUB_WEIGHT,
            )
        wn_maj = _renormalize_sub_weights(raws_maj)
        mean_03 = sum(wi * float(val) for wi, (_, val) in zip(wn_maj, major_terms))
    else:
        mean_03 = sum(val for _, val in major_terms) / float(len(major_terms))
    return mean_03 / 3.0


def parsed_judge_json_to_weighted_reward(
    parsed: Dict[str, Any],
    evaluation_rubric: Optional[Dict[str, Any]] = None,
    *,
    checklist_item_weight: float = 1.0,
    major_aspect_weights: Optional[Dict[str, float]] = None,
    **legacy_kwargs: Any,
) -> float:
    """
    Map judge JSON to a reward in ``[0, 1]`` (mean on ``[0, 3]`` divided by ``3``).

    **Adaptive (per-query) sub-aspects — normalize weights to sum to 1, then one score each:**

    - **Rubric, task dimensions:** all ``rubric:<name>`` where ``name`` is **not** in
      :data:`REQUIRED_GENERIC_RUBRIC_DIMS`. Weights from ``evaluation_rubric``; missing keys use
      ``1.0``. Non-positive raw weights are replaced with :data:`FALLBACK_NONPOSITIVE_SUB_WEIGHT`
      and the vector is re-normalized. One scalar: weighted mean on ``[0, 3]``.

    - **Checklist:** all ``checklist:*`` items share the same raw ``checklist_item_weight`` per
      line, then the same re-normalize rule, yielding one scalar in ``[0, 3]``.

    **Generic** (fixed across tasks): each of the five generically requested rubric dimensions
    that the judge **returns** (``[0, 3]``) counts as a **separate** major term with implicit
    weight 1. ``text_rendering`` may be absent (empty/omitted score). **Visual reference** (if present) is a
    separate major term the same way.

    **Final:** unweighted mean of the major terms that exist, then **divide by 3** for ``[0, 1]``,
    unless ``major_aspect_weights`` is a non-empty dict: then each present major term :math:`i`
    uses raw weight :math:`w_i` (default ``1.0`` when a key is omitted), weights are renormalized
    to sum to ``1.0`` (with the same non-positive fallback as :func:`_renormalize_sub_weights`), and
    the aggregate is :math:`\\sum_i (w_i/\\sum_j w_j) \\cdot s_i` on ``[0, 3]``, then **divide by 3**.

    **Major-term keys** for ``major_aspect_weights``: ``rubric_adaptive`` (task rubric block),
    ``checklist``, each name in :data:`REQUIRED_GENERIC_RUBRIC_DIMS` (e.g. ``text_rendering``,
    ``prompt_faithfulness``), ``visual_reference_evaluation``, ``text_reference_evaluation``.
    Optional aliases: ``adaptive_rubric``, ``visual_reference``, ``text_reference``.

    Extra keyword arguments are **ignored** (for backward compatibility with old configs that
    passed e.g. ``min_worst_dimension_blend``).
    """
    if legacy_kwargs:
        logger.warning(
            "ToolGen judge reward: ignoring deprecated kwargs %s",
            sorted(legacy_kwargs.keys()),
        )
    labeled = parsed_judge_labeled_scores_03(parsed)
    major_terms = labeled_to_major_terms(
        labeled,
        evaluation_rubric,
        checklist_item_weight=checklist_item_weight,
    )
    return weighted_reward_from_major_terms(major_terms, major_aspect_weights)


def parsed_judge_json_to_reward(
    parsed: Dict[str, Any], evaluation_rubric: Optional[Dict[str, Any]] = None, **weight_kwargs: Any
) -> float:
    """
    Default reward in ``[0, 1]`` from full judge output.

    With only ``parsed`` (and default keyword arguments), this matches the original behavior: **mean**
    of all scalar scores on ``[0, 3]``, then divide by 3. Pass ``evaluation_rubric`` and optional
    weighting kwargs to use :func:`parsed_judge_json_to_weighted_reward` instead.
    """
    if not weight_kwargs:
        if evaluation_rubric is None:
            labeled = parsed_judge_labeled_scores_03(parsed)
            mean_03 = sum(v for _, v in labeled) / float(len(labeled))
            return mean_03 / 3.0
    return parsed_judge_json_to_weighted_reward(parsed, evaluation_rubric, **weight_kwargs)


def _population_std(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / float(n)
    var = sum((float(x) - mean) ** 2 for x in values) / float(n)
    return math.sqrt(var)


def toolgen_mean_reward_drop_constant_dimensions(
    member_labeled: List[List[Tuple[str, float]]],
    *,
    eps: float = 1e-6,
) -> List[float]:
    """
    For K generations of the same prompt, compute each member's reward in ``[0, 1]`` by
    averaging only score dimensions whose values differ across members (population std ``> eps``).

    Dimensions with identical values for every member are dropped so they do not dominate a
    tied group or mask relative differences on other axes.

    With a single member, falls back to the usual mean over all dimensions (same as
    :func:`parsed_judge_json_to_reward`).

    Args:
        member_labeled: Length K; each entry is the list from :func:`parsed_judge_labeled_scores_03`.
            Keys need not match across members: only the **intersection** of keys is used (for example,
            rubric dimensions with empty/omitted scores are omitted on that member and drop out of the intersection).
        eps: Minimum standard deviation (on ``[0,3]`` scores) for a dimension to be kept.

    Returns:
        Length-K list of rewards in ``[0, 1]``.
    """
    if not member_labeled:
        return []
    if len(member_labeled) == 1:
        vec = member_labeled[0]
        mean_03 = sum(v for _, v in vec) / float(len(vec))
        return [mean_03 / 3.0]

    key_sets = [frozenset(k for k, _ in vec) for vec in member_labeled]
    common_keys = key_sets[0].intersection(*key_sets[1:])
    keys0 = [k for k, _ in member_labeled[0]]
    ordered_common = [k for k in keys0 if k in common_keys]
    trailing = sorted(common_keys.difference(ordered_common))
    ordered_common = ordered_common + trailing
    if not ordered_common:
        return [sum(v for _, v in vec) / float(len(vec)) / 3.0 for vec in member_labeled]

    vec_maps = [dict(vec) for vec in member_labeled]
    d = len(ordered_common)
    columns: List[List[float]] = [[] for _ in range(d)]
    for vm in vec_maps:
        for j, k in enumerate(ordered_common):
            columns[j].append(vm[k])

    stds = [_population_std(col) for col in columns]
    active = [j for j in range(d) if stds[j] > eps]
    if not active:
        return [sum(v for _, v in vec) / float(len(vec)) / 3.0 for vec in member_labeled]

    out: List[float] = []
    for vm in vec_maps:
        vals = [vm[ordered_common[j]] for j in active]
        mean_03 = sum(vals) / float(len(vals))
        out.append(mean_03 / 3.0)
    return out


def toolgen_labeled_dict_rows_to_train_metrics(
    rows: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Train metrics for ToolGen judge logging. Aligns with :func:`parsed_judge_json_to_weighted_reward`
    semantics: **adaptive** rubric (``rubric:`` keys not in :data:`REQUIRED_GENERIC_RUBRIC_DIMS`)
    and **checklist** are aggregated (equal weight per sub-aspect here; YML weights are not
    available in this path). **Generic** dimensions are logged per ``generic_<dim>``.

    - **rubric_adaptive_…**: unweighted mean of non-generic ``rubric:*`` scores in each row.
    - **checklist_overall_…**: unweighted mean of all ``checklist:*`` in each row.
    - **generic_***, **generic_visual_reference_***, **generic_text_reference_***: as before.
    """
    if not rows:
        return {}
    rubric_adaptive_per_sample: List[float] = []
    checklist_per_sample: List[float] = []
    for d in rows:
        ad_vals = [
            float(v)
            for k, v in d.items()
            if str(k).startswith("rubric:") and str(k)[len("rubric:") :] not in REQUIRED_GENERIC_RUBRIC_DIMS
        ]
        checklist_vals = [float(v) for k, v in d.items() if str(k).startswith("checklist:")]
        if ad_vals:
            rubric_adaptive_per_sample.append(sum(ad_vals) / float(len(ad_vals)))
        if checklist_vals:
            checklist_per_sample.append(sum(checklist_vals) / float(len(checklist_vals)))

    out: Dict[str, float] = {}
    if rubric_adaptive_per_sample:
        m = sum(rubric_adaptive_per_sample) / float(len(rubric_adaptive_per_sample))
        out["train/toolgen_subscore/rubric_adaptive_overall_mean"] = m
        out["train/toolgen_subscore/rubric_adaptive_overall_std"] = _population_std(
            rubric_adaptive_per_sample
        )
        out["train/toolgen_subscore/rubric_adaptive_overall_mean_01"] = m / 3.0
    if checklist_per_sample:
        m = sum(checklist_per_sample) / float(len(checklist_per_sample))
        out["train/toolgen_subscore/checklist_overall_mean"] = m
        out["train/toolgen_subscore/checklist_overall_std"] = _population_std(checklist_per_sample)
        out["train/toolgen_subscore/checklist_overall_mean_01"] = m / 3.0

    for dim in sorted(REQUIRED_GENERIC_RUBRIC_DIMS):
        rubric_key = f"rubric:{dim}"
        vals = [float(d[rubric_key]) for d in rows if rubric_key in d]
        if not vals:
            continue
        mean_03 = sum(vals) / float(len(vals))
        out[f"train/toolgen_subscore/generic_{dim}_mean"] = mean_03
        out[f"train/toolgen_subscore/generic_{dim}_std"] = _population_std(vals)
        out[f"train/toolgen_subscore/generic_{dim}_mean_01"] = mean_03 / 3.0

    vre_key = "visual_reference_evaluation"
    vre_vals = [float(d[vre_key]) for d in rows if vre_key in d]
    if vre_vals:
        mean_03 = sum(vre_vals) / float(len(vre_vals))
        out["train/toolgen_subscore/generic_visual_reference_mean"] = mean_03
        out["train/toolgen_subscore/generic_visual_reference_std"] = _population_std(vre_vals)
        out["train/toolgen_subscore/generic_visual_reference_mean_01"] = mean_03 / 3.0

    tre_key = "text_reference_evaluation"
    tre_vals = [float(d[tre_key]) for d in rows if tre_key in d]
    if tre_vals:
        mean_03 = sum(tre_vals) / float(len(tre_vals))
        out["train/toolgen_subscore/generic_text_reference_mean"] = mean_03
        out["train/toolgen_subscore/generic_text_reference_std"] = _population_std(tre_vals)
        out["train/toolgen_subscore/generic_text_reference_mean_01"] = mean_03 / 3.0

    return out


def reference_slot_placeholders(n: int) -> List[str]:
    return [f"flowfactory://reference-slot/{i}" for i in range(n)]


def coerce_ref_image_list(condition_images_entry: Any) -> List[Image.Image]:
    """Normalize one batch element to a list of PIL images. Empty list = text-to-image (no reference slots)."""
    if condition_images_entry is None:
        return []
    if isinstance(condition_images_entry, Image.Image):
        return [condition_images_entry]
    if isinstance(condition_images_entry, list):
        out: List[Image.Image] = []
        for j, x in enumerate(condition_images_entry):
            if not isinstance(x, Image.Image):
                raise TypeError(
                    f"expected PIL.Image.Image inside condition_images list, "
                    f"got {type(x).__name__} at index {j}"
                )
            out.append(x)
        return out
    raise TypeError(
        f"expected PIL.Image.Image or list of PIL images for condition_images element, "
        f"got {type(condition_images_entry).__name__}"
    )


def validate_meta_list(name: str, values: Sequence[Any], batch_len: int) -> None:
    if not isinstance(values, list):
        raise TypeError(f"expected list for {name}, got {type(values).__name__}")
    if len(values) != batch_len:
        raise ValueError(
            f"expected len({name})==batch_size ({batch_len}), got {len(values)}"
        )


def as_str_list(checklist: Any, *, sample_index: int) -> List[str]:
    if checklist is None:
        return []
    if not isinstance(checklist, list):
        raise TypeError(
            f"sample {sample_index}: verification_checklist must be list or None, "
            f"got {type(checklist).__name__}: {checklist!r}"
        )
    return [str(item) for item in checklist]


def as_rubric_dict(rubric: Any, *, sample_index: int) -> Dict[str, Any]:
    if rubric is None:
        return {}
    if isinstance(rubric, str):
        s = rubric.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"sample {sample_index}: evaluation_rubric is not valid JSON: {e}; "
                f"first 200 chars: {s[:200]!r}"
            ) from e
        if not isinstance(parsed, dict):
            raise TypeError(
                f"sample {sample_index}: evaluation_rubric JSON must decode to a dict, "
                f"got {type(parsed).__name__}: {parsed!r}"
            )
        return parsed
    if isinstance(rubric, dict):
        return rubric
    raise TypeError(
        f"sample {sample_index}: evaluation_rubric must be dict, str, or None, "
        f"got {type(rubric).__name__}: {rubric!r}"
    )


def import_toolgen_frontier_model(phase4_agent_dir: str) -> Any:
    import sys
    from pathlib import Path

    p = Path(phase4_agent_dir).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            f"toolgen_phase4_agent_dir must be an existing directory (ToolGen phase4_agent root), got {p}"
        )
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    from frontier_model import FrontierModel  # type: ignore[import-not-found]

    return FrontierModel
