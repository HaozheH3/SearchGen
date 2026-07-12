from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from searchgen_agent.io import load_rows


class DatasetLoadingTest(unittest.TestCase):
    def test_jsonl_prompt_aliases_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(json.dumps({"query": "A red apple"}) + "\n", encoding="utf-8")
            rows = load_rows(path)
        self.assertEqual(rows[0]["user_request"], "A red apple")
        self.assertEqual(rows[0]["user_prompt"], "A red apple")


if __name__ == "__main__":
    unittest.main()
