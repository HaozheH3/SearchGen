"""Generic input dataset loading for the public pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_prompt(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    prompt = next(
        (
            str(result[key]).strip()
            for key in ("user_request", "user_prompt", "prompt", "query")
            if result.get(key) is not None and str(result[key]).strip()
        ),
        "",
    )
    if not prompt:
        raise ValueError("row has no non-empty user_request, user_prompt, prompt, or query")
    result["user_request"] = prompt
    result.setdefault("user_prompt", prompt)
    return result


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array/wrapper or JSONL file and normalize prompt fields."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".jsonl":
        raw_rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif source.suffix.lower() == ".json":
        loaded = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            raw_rows = loaded
        elif isinstance(loaded, dict):
            raw_rows = next(
                (loaded[key] for key in ("items", "data", "requests") if isinstance(loaded.get(key), list)),
                None,
            )
            if raw_rows is None:
                raise ValueError("JSON object must contain a list under items, data, or requests")
        else:
            raise ValueError("JSON input must be an array or supported wrapper object")
    else:
        raise ValueError("input must use .json or .jsonl")
    if not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError("every dataset row must be a JSON object")
    rows = [_normalize_prompt(row) for row in raw_rows]
    if not rows:
        raise ValueError("dataset contains no rows")
    return rows
