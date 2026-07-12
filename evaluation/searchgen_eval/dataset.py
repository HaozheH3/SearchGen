from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_metadata(metadata: Path, benchmark_root: Path, validate_files: bool = True) -> list[dict[str, Any]]:
    root = benchmark_root.resolve()
    rows = []
    seen: set[str] = set()
    with metadata.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            bench_id = row.get("bench_id")
            if not isinstance(bench_id, str) or not bench_id:
                raise ValueError(f"line {line_no}: missing bench_id")
            if bench_id in seen:
                raise ValueError(f"line {line_no}: duplicate bench_id {bench_id}")
            for key in ("user_prompt", "verification_checklist", "evaluation_rubric"):
                if not row.get(key):
                    raise ValueError(f"line {line_no}: missing {key}")
            slots = []
            for slot in row.get("visual_reference_slots") or []:
                rel = slot.get("relative_path")
                if not isinstance(rel, str) or not rel:
                    raise ValueError(f"line {line_no}: invalid reference relative_path")
                resolved = (root / rel).resolve()
                if not _within(resolved, root):
                    raise ValueError(f"line {line_no}: reference escapes benchmark root: {rel}")
                if validate_files and not resolved.is_file():
                    raise FileNotFoundError(f"line {line_no}: missing reference: {rel}")
                slots.append({**slot, "absolute_path": str(resolved)})
            rows.append({**row, "sample_id": bench_id, "reference_slots": slots})
            seen.add(bench_id)
    return rows
