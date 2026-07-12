import json

from PIL import Image

from searchgen_eval.evaluator import RESULT, evaluate


class FakeClient:
    def chat(self, *_args, **_kwargs):
        parsed = {"checklist_scores": [{"score": 3}], "rubric_scores": {
            "prompt_faithfulness": {"score": 3}, "image_quality": {"score": 3},
            "text_rendering": {"score": ""}, "ai_naturalness": {"score": 2.5},
            "composition_and_aesthetics": {"score": 3}, "physical_plausibility": {"score": 3}},
            "visual_reference_evaluation": {"applicable": False, "score": 0},
            "text_reference_evaluation": {"applicable": False, "score": 0}}
        return json.dumps(parsed), {"mock": True}


def test_offline_evaluation_and_resume(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "red").save(image)
    row = {"sample_id": "bench_test", "user_prompt": "red square", "verification_checklist": ["red"],
           "evaluation_rubric": {}, "reference_slots": [], "text_knowledge_slots": []}
    pred = {"bench_id": "bench_test", "generator": "mock", "image_path": str(image)}
    result = evaluate(row, pred, tmp_path / "results", FakeClient())
    assert result["success"]
    assert (tmp_path / "results" / "mock" / "bench_test" / RESULT).is_file()
    assert evaluate(row, pred, tmp_path / "results", FakeClient())["resumed"]
