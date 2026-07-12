from pathlib import Path

from searchgen_eval.dataset import load_metadata


def test_public_dataset_counts():
    root = Path(__file__).resolve().parents[3] / "data_release" / "searchgen-bench"
    rows = load_metadata(root / "eval_metadata.jsonl", root)
    assert len(rows) == len({row["sample_id"] for row in rows}) == 751
    assert sum(len(row["reference_slots"]) for row in rows) == 1099
    assert all("release_row" not in row for row in rows)
