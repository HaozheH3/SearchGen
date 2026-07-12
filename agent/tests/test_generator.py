from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from searchgen_agent.generator.api import mock_image_generator
from searchgen_agent.generator.runner import generate_row, resolve_references, row_complete


class ImageGeneratorTest(unittest.TestCase):
    def test_mock_generator_consumes_s4_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            row = Path(temporary) / "eval_row_001"
            artifacts = row / "artifacts_files"
            artifacts.mkdir(parents=True)
            manifest = {
                "schema_version": 2,
                "stage": "s4_generation_manifest",
                "prompts": {
                    "user_prompt_raw": "A red apple",
                    "refined_prompt": "A detailed red apple on a wooden table",
                },
                "augmented_mode": "t2i",
                "reference_images": {"ordered_references": []},
                "row_identity": {"dataset": "demo", "lane": "agentic_reasoner", "row_index": 0},
            }
            (artifacts / "s4_generation_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            ok = generate_row(
                row,
                generator_name="mock",
                generator=mock_image_generator,
                model="offline",
                width=384,
                height=384,
            )
            self.assertTrue(ok)
            self.assertTrue(row_complete(row, "mock"))
            self.assertGreater((row / "mock_generator" / "generated_image.png").stat().st_size, 256)

    def test_reference_order_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            row = Path(temporary) / "eval_row_001"
            local = row / "reference_images" / "local.png"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"test fixture")
            manifest = {
                "reference_images": {
                    "ordered_references": [
                        {"url": "https://images.example.com/first.png"},
                        {"local_path_relative": "reference_images/local.png"},
                    ]
                }
            }
            resolved = resolve_references(row, manifest)
            self.assertEqual(resolved[0], "https://images.example.com/first.png")
            self.assertEqual(Path(resolved[1]), local.resolve())


if __name__ == "__main__":
    unittest.main()
