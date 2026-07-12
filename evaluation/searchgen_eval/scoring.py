from __future__ import annotations

import json
from pathlib import Path

COMPONENTS = ["checklist", "adaptive_rubric", "prompt_faithfulness", "image_quality", "text_rendering",
              "ai_naturalness", "composition_and_aesthetics", "physical_plausibility", "visual_reference", "text_reference"]


def aggregate(root: Path, policy: str = "skip") -> dict:
    files = list(root.rglob("augmented_parsed_result_evaluation_protocol.json"))
    values = {key: [] for key in COMPONENTS}
    invalid = parse_failures = 0
    for path in files:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not result.get("success"):
                invalid += 1
                if "pars" in str(result.get("error", "")).lower():
                    parse_failures += 1
                continue
            scores = result.get("scores") or {}
            mapping = {
                "checklist": lambda k: k.startswith("checklist:"),
                "adaptive_rubric": lambda k: k.startswith("rubric:") and k.split(":", 1)[1] not in {
                    "prompt_faithfulness", "image_quality", "text_rendering", "ai_naturalness",
                    "composition_and_aesthetics", "physical_plausibility"},
                "visual_reference": lambda k: k == "visual_reference_evaluation",
                "text_reference": lambda k: k == "text_reference_evaluation",
            }
            for component in COMPONENTS:
                predicate = mapping.get(component, lambda k, c=component: k == f"rubric:{c}")
                matched = [float(v) for k, v in scores.items() if predicate(k)]
                if matched:
                    values[component].append(sum(matched) / len(matched))
                elif policy == "zero":
                    values[component].append(0.0)
        except Exception:
            invalid += 1
    means = {key: (sum(items) / len(items) if items else None) for key, items in values.items()}
    present = [v for v in means.values() if v is not None]
    return {"policy": policy, "components": means, "overall_mean": sum(present) / len(present) if present else None,
            "evaluated_count": sum(1 for p in files if _success(p)), "result_file_count": len(files),
            "missing_count": 0, "invalid_count": invalid, "parse_failure_count": parse_failures}


def _success(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("success") is True
    except Exception:
        return False
