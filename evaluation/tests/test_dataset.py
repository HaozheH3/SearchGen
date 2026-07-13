import os
from pathlib import Path

import pytest

from searchgen_eval.dataset import load_metadata


def test_public_dataset_counts():
    configured = os.environ.get("SEARCHGEN_BENCH_ROOT")
    if not configured:
        pytest.skip("set SEARCHGEN_BENCH_ROOT to run the full public-dataset check")
    root = Path(configured).expanduser().resolve()
    rows = load_metadata(root / "eval_metadata.jsonl", root)
    assert len(rows) == len({row["sample_id"] for row in rows}) == 751
    assert sum(len(row["reference_slots"]) for row in rows) == 1099
    assert all("release_row" not in row for row in rows)
