import base64
import json
from pathlib import Path

from searchgen_eval import judge_common as judge


URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


def build_fixture():
    row = {"user_prompt": "A red apple on a wooden table.", "sample_id": "fixture_en",
           "augmented_generation_details": {"used_reference_images_count": 1,
           "selected_reference_images_count": 1, "expected_reference_images_count": 1}}
    checklist = ["A red apple is visible.", "The apple rests on a wooden table."]
    rubric = {"Object fidelity": {"weight": 1.0, "description": "Correct apple color, shape, and placement."}}
    context = {"has_critical_references": True, "critical_candidate_indices": [0],
        "critical_candidates": [{"candidate_index": 0, "entity": "red apple", "severity": "critical",
        "reasoning": "Identity and color reference.", "matched_search_results": [{"query": "red apple",
        "image_url": URL, "local_path": "fixture.png", "selected_image_title": "Apple"}]}],
        "eval_text_knowledge_slots": [], "has_text_knowledge_gaps_for_eval": False}
    prompt = judge.build_eval_prompt_text(row=row, verification_checklist=checklist, evaluation_rubric=rubric,
        visual_context=context, variant="augmented", reference_slot_urls=[URL])
    content = judge.build_judge_interleaved_user_content(user_text_prompt=prompt, reference_slot_urls=[URL],
        reference_image_urls=[URL], visual_context=context, assess_image_data_url=URL, max_pixels=1_500_000)
    return {"system_prompt": judge.JUDGE_SYSTEM_PROMPT, "user_prompt": prompt, "interleaved": content}


def test_immutable_golden_fixture():
    encoded = (Path(__file__).parent / "fixtures" / "prompt_interleaved_golden.json.b64").read_bytes()
    golden = json.loads(base64.b64decode(encoded))
    assert build_fixture() == golden
