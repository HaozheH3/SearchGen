"""Public, repository-independent row finalization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .refined_prompt import sync_refined_prompt_txt


def finalize_row(
    row_dir: Path,
    *,
    dataset_name: str,
    lane_name: str,
    row_index: int,
    row: dict[str, Any],
    model_name: str,
    model_name_s1: str,
) -> None:
    """Write portable final metadata after a row reaches a terminal state."""

    artifacts = row_dir / "artifacts_files"
    sync_refined_prompt_txt(artifacts)
    prompt = str(row.get("user_request") or row.get("user_prompt") or "").strip()
    manifest = {
        "schema_version": 1,
        "eval_row_index": row_index - 1,
        "dataset": dataset_name,
        "lane": lane_name,
        "user_prompt_raw": prompt,
    }
    (row_dir / "ROW_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    inference = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_s1": model_name_s1,
        "model_s2_s3": model_name,
    }
    (artifacts / "reasoner_inference_meta.json").write_text(
        json.dumps(inference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
