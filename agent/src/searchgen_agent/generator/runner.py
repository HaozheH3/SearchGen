"""Image Generator runner using the S4 manifest as its sole input contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..signing import load_signer
from .api import (
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerator,
    mock_image_generator,
)


MIN_IMAGE_BYTES = 256


def load_s4_manifest(row_dir: Path) -> dict[str, Any]:
    path = row_dir / "artifacts_files" / "s4_generation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing S4 manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("S4 manifest must be a JSON object")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError("S4 manifest is missing prompts")
    if not str(prompts.get("refined_prompt") or prompts.get("user_prompt_raw") or "").strip():
        raise ValueError("S4 manifest has no usable prompt")
    return manifest


def _ordered_reference_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    block = manifest.get("reference_images")
    if isinstance(block, dict) and isinstance(block.get("ordered_references"), list):
        return [entry for entry in block["ordered_references"] if isinstance(entry, dict)]
    legacy = manifest.get("reference_images_ordered")
    if isinstance(legacy, list):
        return [entry for entry in legacy if isinstance(entry, dict)]
    return []


def resolve_references(
    row_dir: Path,
    manifest: dict[str, Any],
    *,
    allow_external_paths: bool = False,
) -> list[str]:
    """Resolve ordered reference URLs or local files without serializing private paths."""

    root = row_dir.resolve()
    references: list[str] = []
    for entry in _ordered_reference_entries(manifest):
        url = str(entry.get("url") or entry.get("imageUrl") or "").strip()
        if url.startswith(("http://", "https://")):
            references.append(url)
            continue
        candidates: list[Path] = []
        relative = str(entry.get("local_path_relative") or "").strip()
        if relative:
            candidates.append(root / relative)
        local = str(entry.get("local_path") or "").strip()
        if local:
            candidates.append(Path(local).expanduser())
        for candidate in candidates:
            resolved = candidate.resolve()
            inside = resolved == root or root in resolved.parents
            if resolved.is_file() and (inside or allow_external_paths):
                references.append(str(resolved))
                break
    return references


def load_generator(backend: str, plugin: str | None) -> ImageGenerator:
    if backend == "mock":
        return mock_image_generator
    if backend == "plugin":
        if not plugin:
            raise ValueError("--generator-plugin is required for backend=plugin")
        return load_signer(plugin)  # type: ignore[return-value]
    raise ValueError(f"unsupported backend: {backend}")


def row_complete(row_dir: Path, generator_name: str) -> bool:
    image = row_dir / f"{generator_name}_generator" / "generated_image.png"
    try:
        return image.is_file() and image.stat().st_size >= MIN_IMAGE_BYTES
    except OSError:
        return False


def generate_row(
    row_dir: Path,
    *,
    generator_name: str,
    generator: ImageGenerator,
    model: str,
    width: int = 1024,
    height: int = 1024,
    allow_external_paths: bool = False,
    overwrite: bool = False,
) -> bool:
    output_dir = row_dir / f"{generator_name}_generator"
    image_path = output_dir / "generated_image.png"
    if not overwrite and row_complete(row_dir, generator_name):
        return True
    try:
        manifest = load_s4_manifest(row_dir)
        prompts = manifest["prompts"]
        prompt = str(prompts.get("refined_prompt") or prompts.get("user_prompt_raw") or "").strip()
        references = resolve_references(
            row_dir,
            manifest,
            allow_external_paths=allow_external_paths,
        )
        result = generator(
            ImageGenerationRequest(
                prompt=prompt,
                reference_images=tuple(references),
                model=model,
                width=width,
                height=height,
            )
        )
        if not isinstance(result, ImageGenerationResult):
            raise TypeError("image generator must return ImageGenerationResult")
        if len(result.image_bytes) < MIN_IMAGE_BYTES:
            raise ValueError("generator returned an empty or implausibly small image")
        if not result.image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("image generator must return PNG bytes")
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = image_path.with_suffix(".tmp")
        temporary.write_bytes(result.image_bytes)
        temporary.replace(image_path)
        metadata = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": generator_name,
            "model": model,
            "reference_count": len(references),
            "provider_metadata": dict(result.metadata),
        }
        (output_dir / "generation_meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        failure = output_dir / "last_generation_failure.json"
        if failure.is_file():
            failure.unlink()
        return True
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": generator_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (output_dir / "last_generation_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return False


def iter_row_dirs(lane_dir: Path) -> list[Path]:
    if not lane_dir.is_dir():
        raise FileNotFoundError(lane_dir)
    return sorted(
        path
        for path in lane_dir.iterdir()
        if path.is_dir() and path.name.startswith("eval_row_")
    )
