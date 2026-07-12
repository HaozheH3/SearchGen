from __future__ import annotations

import json
from pathlib import Path


def load_predictions(manifest: Path | None, images_dir: Path | None) -> list[dict]:
    if manifest:
        base = manifest.resolve().parent
        records = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
        for record in records:
            path = Path(record["image_path"]).expanduser()
            record["image_path"] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
            record.setdefault("generator", "predictions")
        return records
    if images_dir:
        return [{"bench_id": p.stem, "generator": "predictions", "image_path": str(p.resolve())}
                for p in sorted(images_dir.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    raise ValueError("provide --predictions-manifest or --images-dir")
