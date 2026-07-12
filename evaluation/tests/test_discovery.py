import json

from searchgen_eval.discovery import load_predictions


def test_relative_manifest_path(tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(b"x")
    manifest = tmp_path / "predictions.jsonl"
    manifest.write_text(json.dumps({"bench_id": "bench_0001", "image_path": "x.png"}) + "\n")
    got = load_predictions(manifest, None)[0]
    assert got["image_path"] == str(image.resolve())
    assert got["generator"] == "predictions"
