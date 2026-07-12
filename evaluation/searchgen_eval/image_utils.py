from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from PIL import Image


def pil_image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt, quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def local_image_to_data_url(path: Path, max_total_pixels: int = 1_500_000, max_edge: int = 1568) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = min(1.0, max_edge / max(width, height), (max_total_pixels / (width * height)) ** 0.5)
        if scale < 1.0:
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
        data = pil_image_to_base64(image)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{data}"
