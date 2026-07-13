from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import judge_common as judge
from .api_client import APIClient
from .image_utils import local_image_to_data_url

RESULT = "augmented_parsed_result_evaluation_protocol.json"
PROMPT = "augmented_prompt_context_evaluation_protocol.txt"
RAW = "augmented_raw_api_output_evaluation_protocol.txt"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def result_valid(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("success") is True
    except Exception:
        return False


def build_request(row: dict, image_path: Path) -> tuple[str, list[dict]]:
    refs = [local_image_to_data_url(Path(slot["absolute_path"])) for slot in row["reference_slots"]]
    candidates = []
    for index, (slot, url) in enumerate(zip(row["reference_slots"], refs)):
        candidates.append({"candidate_index": index, "entity": slot.get("entity"), "severity": slot.get("severity"),
                           "reasoning": slot.get("reasoning"), "suggested_search": None, "search_type": None,
                           "matched_search_results": [{"query": slot.get("query"), "image_url": url,
                           "local_path": slot["absolute_path"], "selected_image_title": slot.get("title"), "source_file": None}]})
    context = {"has_critical_references": bool(refs), "critical_candidate_indices": list(range(len(refs))),
               "critical_candidates": candidates, "eval_text_knowledge_slots": row.get("text_knowledge_slots") or [],
               "has_text_knowledge_gaps_for_eval": bool(row.get("text_knowledge_slots"))}
    judge_row = {"user_prompt": row["user_prompt"], "sample_id": row["sample_id"],
                 "augmented_generation_details": {"used_reference_images_count": len(refs),
                 "selected_reference_images_count": len(refs), "expected_reference_images_count": len(refs)}}
    prompt = judge.build_eval_prompt_text(row=judge_row, verification_checklist=row["verification_checklist"],
        evaluation_rubric=row["evaluation_rubric"], visual_context=context, variant="augmented",
        reference_slot_urls=refs)
    content = judge.build_judge_interleaved_user_content(user_text_prompt=prompt, reference_slot_urls=refs,
        reference_image_urls=list(refs), visual_context=context, assess_image_data_url=local_image_to_data_url(image_path),
        max_pixels=1_500_000)
    return prompt, content


def evaluate(row: dict, prediction: dict, output_root: Path, client: APIClient) -> dict:
    out_dir = output_root / prediction["generator"] / row["sample_id"]
    result_path = out_dir / RESULT
    if result_valid(result_path):
        return {"success": True, "resumed": True, "bench_id": row["sample_id"]}
    prompt, content = build_request(row, Path(prediction["image_path"]))
    atomic_text(out_dir / PROMPT, prompt)
    try:
        response, raw = client.chat(judge.JUDGE_SYSTEM_PROMPT, content)
        atomic_text(out_dir / RAW, response)
        parsed = judge.try_parse_judge_output(response)
        if parsed is None:
            raise ValueError("judge response could not be parsed")
        scores = dict(judge.parsed_judge_labeled_scores_03(parsed))
        result = {"success": True, "skipped": False, "error": None, "parsed": parsed, "scores": scores,
                  "bench_id": row["sample_id"], "sample_id": row["sample_id"], "generator_name": prediction["generator"],
                  "attached_visual_reference_image_count": len(row["reference_slots"])}
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}", "bench_id": row["sample_id"]}
    atomic_text(result_path, json.dumps(result, ensure_ascii=False, indent=2))
    return result
