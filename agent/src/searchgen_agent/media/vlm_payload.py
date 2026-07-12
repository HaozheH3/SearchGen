"""
VLM image parts for OpenAI-style chat (``image_url`` with ``data:...`` or http(s)).

Each *slot* is either:

- A **Pillow** ``Image`` / **raw bytes** / a **local filesystem path** → we normalize and
  emit a JPEG ``data:`` URL (optional **max-edge** shrink, then area resize to
  ``max_total_pixels``; same idea as hpdv3 / MagicData ``max_pixels`` on the part).
- An **http(s) URL string** → that URL is sent as-is in the API (no client-side
  download; the gateway may fetch it).

**Never** use ``requests`` here. If the pipeline already materialized a remote image
to ``local_path``, that path is read from disk. Hotlink with URL only when there
is no local file.

Set ``TOOLGEN_VLM_MAX_EDGE`` (e.g. ``512``) to shrink **local** decoded images so
``max(width,height)`` never exceeds that value (before the ``max_total_pixels`` area cap).
Plain ``http(s)`` slot strings are unchanged (no client-side fetch).
"""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Total pixel budget (w*h), same semantics as hpdv3 ``max_pixels`` for resize.
DEFAULT_MAX_TOTAL_PIXELS = 768 * 768
VLM_JPEG_QUALITY = 86

# Some multimodal gateways reject decoded rasters with extreme aspect ratios.
VLM_MAX_ASPECT_RATIO = 8.0

# str = http(s) to pass through, or filesystem path, or "path" / Path; Pillow Image is accepted.
ImageItem = Union[str, "Path", Any]

# Pillow ``Image.format`` values we accept for search-downloaded / on-disk reference rasters
# (VLM path converts to RGB JPEG). Everything else (SVG, PDF-as-image, HTML masquerading, …)
# is rejected so ``select_reference_image`` never sees undecodable bytes.
REFERENCE_RASTER_PIL_FORMATS = frozenset(
    {"JPEG", "PNG", "WEBP", "GIF", "BMP", "MPO", "TIFF", "TIF", "AVIF"}
)


def coalesce_vlm_max_edge(explicit: int = 0) -> int:
    """
    Longest-side cap before the area-based ``max_total_pixels`` resize.

    When ``explicit > 0``, use it. Otherwise read ``TOOLGEN_VLM_MAX_EDGE`` (positive int).
    Used to keep **local** decoded condition/reference rasters bounded (plain ``http(s)`` URLs are
    still passed through unchanged).
    """

    if explicit and explicit > 0:
        return int(explicit)
    raw = os.environ.get("TOOLGEN_VLM_MAX_EDGE", "").strip()
    if not raw:
        return 0
    try:
        v = int(raw)
        return v if v > 0 else 0
    except ValueError:
        return 0


def reference_raster_pil_format_supported(fmt: Optional[str]) -> bool:
    if not fmt or not isinstance(fmt, str):
        return False
    return fmt.strip().upper() in REFERENCE_RASTER_PIL_FORMATS


def reference_raster_bytes_are_supported(raw: bytes) -> bool:
    """True if ``raw`` decodes as a Pillow image whose format is in ``REFERENCE_RASTER_PIL_FORMATS``."""
    if not raw or len(raw) < 8:
        return False
    head = raw[:512].lstrip()
    if head.startswith((b"<?xml", b"<!DOCTYPE", b"<!doctype")):
        return False
    if head.startswith(b"<") and (b"<svg" in raw[:2048] or b"<html" in raw[:2048].lower()):
        return False
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            return reference_raster_pil_format_supported(im.format)
    except Exception:
        return False


@dataclass
class VlmImagePayloadMeta:
    strategy: str  # "data_url_jpeg" | "http_url"
    source: str  # "path" | "http" | "bytes" | "pillow" | "none"
    byte_length: int = 0
    had_resize: bool = False


@dataclass
class BuildContentResult:
    user_content: List[Dict[str, Any]]
    image_metas: List[VlmImagePayloadMeta] = field(default_factory=list)


def _is_http(s: str) -> bool:
    t = s.strip().lower()
    return t.startswith("http://") or t.startswith("https://")


def local_image_file_is_openable(path: Union[str, Path]) -> bool:
    """
    True if ``path`` exists, Pillow can decode it, and ``Image.format`` is in the
    reference-raster allowlist (guards SVG/HTML/error pages saved with a raster suffix).
    """
    from PIL import Image

    p = Path(str(path)).expanduser()
    try:
        if not p.is_file() or p.stat().st_size < 8:
            return False
        with Image.open(p) as im:
            im.load()
            return reference_raster_pil_format_supported(im.format)
    except Exception:
        return False


def _is_pil_image(x: Any) -> bool:
    if x is None or isinstance(x, (str, bytes, Path)):
        return False
    try:
        from PIL import Image
        return isinstance(x, Image.Image)
    except Exception:
        return False


def _data_url_from_bytes(raw: bytes, mime: str) -> str:
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def pad_rgb_image_to_max_aspect_ratio(
    im: Any,
    *,
    max_ratio: float = VLM_MAX_ASPECT_RATIO,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Any:
    """
    Letterbox or pillarbox an RGB image so max(w/h, h/w) <= max_ratio.

    Upstream VLMs reject extreme panoramas (e.g. 2043×216); padding preserves full raster
    before the separate area resize in ``_pil_rgb_image_to_vlm_jpeg``.
    """
    from PIL import Image

    if max_ratio <= 1.0:
        return im
    w, h = im.size
    if w < 1 or h < 1:
        return im
    rw = w / float(h)
    rh = h / float(w)
    if rw <= max_ratio and rh <= max_ratio:
        return im
    if rw > max_ratio:
        new_h = max(h, int(math.ceil(w / max_ratio)))
        canvas = Image.new("RGB", (w, new_h), fill)
        y0 = (new_h - h) // 2
        canvas.paste(im, (0, y0))
        return canvas
    new_w = max(w, int(math.ceil(h / max_ratio)))
    canvas = Image.new("RGB", (new_w, h), fill)
    x0 = (new_w - w) // 2
    canvas.paste(im, (x0, 0))
    return canvas


def _resize_to_max_edge(img: Any, max_edge: int) -> Any:
    """Uniformly scale down so ``max(width,height) <= max_edge`` (never upscale)."""
    from PIL import Image

    if max_edge <= 0:
        return img
    w, h = img.size
    if w < 1 or h < 1:
        return img
    m = max(w, h)
    if m <= max_edge:
        return img
    scale = max_edge / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return img.resize((new_w, new_h), resample=resample)


def _resize_to_max_total_pixels(img: Any, max_total_pixels: int) -> Any:
    from PIL import Image

    if max_total_pixels <= 0:
        return img
    w, h = img.size
    cur = w * h
    if cur <= max_total_pixels:
        return img
    scale = (max_total_pixels / float(cur)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return img.resize((new_w, new_h), resample=resample)


def _pil_rgb_image_to_vlm_jpeg(
    im: Any,
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> Tuple[bytes, bool]:
    had_resize = False
    if max_edge > 0:
        w0, h0 = im.size
        if max(w0, h0) > max_edge:
            im = _resize_to_max_edge(im, max_edge)
            had_resize = True
    w0, h0 = im.size
    if max_total_pixels > 0 and w0 * h0 > max_total_pixels:
        im = _resize_to_max_total_pixels(im, max_total_pixels)
        had_resize = True
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=VLM_JPEG_QUALITY, optimize=True, progressive=True)
    return out.getvalue(), had_resize


def pil_image_to_vlm_data_url(
    im: Any,
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> Tuple[str, VlmImagePayloadMeta]:
    max_edge = coalesce_vlm_max_edge(max_edge)
    if im.mode == "RGBA":  # type: ignore[union-attr]
        from PIL import Image
        background = Image.new("RGB", im.size, (255, 255, 255))  # type: ignore[union-attr]
        background.paste(im, mask=im.split()[3])  # type: ignore[union-attr]
        im = background
    else:
        im = im.convert("RGB")  # type: ignore[union-attr]
    im = pad_rgb_image_to_max_aspect_ratio(im)
    jpg, need_resize = _pil_rgb_image_to_vlm_jpeg(
        im, max_total_pixels=max_total_pixels, max_edge=max_edge
    )
    meta = VlmImagePayloadMeta(
        strategy="data_url_jpeg",
        source="pillow",
        byte_length=len(jpg),
        had_resize=need_resize,
    )
    return _data_url_from_bytes(jpg, "image/jpeg"), meta


def image_bytes_to_vlm_data_url(
    raw: bytes,
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> Tuple[str, VlmImagePayloadMeta]:
    max_edge = coalesce_vlm_max_edge(max_edge)
    from PIL import Image

    im = Image.open(io.BytesIO(raw))
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    if im.mode == "RGBA":
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])
        im = background
    else:
        im = im.convert("RGB")
    im = pad_rgb_image_to_max_aspect_ratio(im)
    jpg, need_resize = _pil_rgb_image_to_vlm_jpeg(
        im, max_total_pixels=max_total_pixels, max_edge=max_edge
    )
    meta = VlmImagePayloadMeta(
        strategy="data_url_jpeg",
        source="bytes",
        byte_length=len(jpg),
        had_resize=need_resize,
    )
    return _data_url_from_bytes(jpg, "image/jpeg"), meta


def _ref_urls_and_path(ref: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    u = (ref or {}).get("url") or (ref or {}).get("imageUrl") or ""
    u = str(u).strip() or None
    loc = (ref or {}).get("local_path") or (ref or {}).get("path") or ""
    loc = str(loc).strip() or None
    return u, loc


def _image_part(
    url: str,
    *,
    max_total_pixels: int,
) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": url},
    }
    p["max_pixels"] = int(max_total_pixels)
    return p


def ref_dict_to_image_part(
    ref: Dict[str, Any],
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> Optional[Tuple[Dict[str, Any], VlmImagePayloadMeta]]:
    max_edge = coalesce_vlm_max_edge(max_edge)
    if not isinstance(ref, dict):
        return None
    if _is_pil_image(ref.get("image")):
        data_url, meta = pil_image_to_vlm_data_url(  # type: ignore[arg-type]
            ref["image"], max_total_pixels=max_total_pixels, max_edge=max_edge
        )
        return _image_part(data_url, max_total_pixels=max_total_pixels), meta
    if isinstance(ref.get("raw_bytes") or ref.get("image_bytes"), (bytes, bytearray)):
        b = ref.get("raw_bytes") or ref.get("image_bytes")
        b = bytes(b)  # type: ignore[arg-type]
        data_url, meta = image_bytes_to_vlm_data_url(
            b, max_total_pixels=max_total_pixels, max_edge=max_edge
        )
        return _image_part(data_url, max_total_pixels=max_total_pixels), meta
    http_url, local = _ref_urls_and_path(ref)
    if local:
        p = Path(str(local).strip()).expanduser()
        if p.is_file():
            try:
                raw = p.read_bytes()
            except OSError:
                raw = b""
            if raw:
                try:
                    data_url, meta = image_bytes_to_vlm_data_url(
                        raw, max_total_pixels=max_total_pixels, max_edge=max_edge
                    )
                except Exception:
                    return None
                meta.source = "path"
                return _image_part(data_url, max_total_pixels=max_total_pixels), meta
    if http_url and _is_http(str(http_url).strip()):
        m = VlmImagePayloadMeta(
            strategy="http_url",
            source="http",
            byte_length=0,
            had_resize=False,
        )
        return _image_part(str(http_url).strip(), max_total_pixels=max_total_pixels), m
    return None


def image_item_to_image_part(
    item: Any,
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> Optional[Tuple[Dict[str, Any], VlmImagePayloadMeta]]:
    """One Task-B slot: Pillow image, file path, raw bytes, or http(s) string."""
    max_edge = coalesce_vlm_max_edge(max_edge)
    if item is None:
        return None
    if isinstance(item, Path):
        item = str(item.expanduser())
    if _is_pil_image(item):
        durl, meta = pil_image_to_vlm_data_url(  # type: ignore[arg-type]
            item, max_total_pixels=max_total_pixels, max_edge=max_edge
        )
        return _image_part(durl, max_total_pixels=max_total_pixels), meta
    if isinstance(item, (bytes, bytearray)):
        b = bytes(item)
        try:
            durl, meta = image_bytes_to_vlm_data_url(
                b, max_total_pixels=max_total_pixels, max_edge=max_edge
            )
        except Exception:
            return None
        return _image_part(durl, max_total_pixels=max_total_pixels), meta
    s = str(item).strip()
    if not s:
        return None
    if _is_http(s):
        m = VlmImagePayloadMeta(
            strategy="http_url", source="http", byte_length=0, had_resize=False
        )
        return _image_part(s, max_total_pixels=max_total_pixels), m
    p = Path(s).expanduser()
    if p.is_file():
        try:
            raw = p.read_bytes()
        except OSError:
            return None
        try:
            durl, meta = image_bytes_to_vlm_data_url(
                raw, max_total_pixels=max_total_pixels, max_edge=max_edge
            )
        except Exception:
            return None
        meta.source = "path"
        return _image_part(durl, max_total_pixels=max_total_pixels), meta
    return None


def build_refinement_user_content(
    text_prompt: str,
    ref_dicts: List[Dict[str, Any]],
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> BuildContentResult:
    """``[text, image, image, ...]`` — same order as ``LLMClient.build_user_content_with_images``."""
    out: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    metas: List[VlmImagePayloadMeta] = []
    for ref in ref_dicts:
        if not isinstance(ref, dict):
            continue
        got = ref_dict_to_image_part(ref, max_total_pixels=max_total_pixels, max_edge=max_edge)
        if not got:
            continue
        part, m = got
        out.append(part)
        metas.append(m)
    return BuildContentResult(user_content=out, image_metas=metas)


def build_refinement_user_content_interleaved(
    preamble: str,
    reference_chunks: List[Tuple[str, Dict[str, Any]]],
    epilogue: str,
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> BuildContentResult:
    """``[text, text, image, text, image, …, text]`` — raster immediately follows its ``<image>`` caption block."""
    out: List[Dict[str, Any]] = [{"type": "text", "text": preamble}]
    metas: List[VlmImagePayloadMeta] = []
    for chunk_text, ref in reference_chunks:
        if chunk_text:
            out.append({"type": "text", "text": chunk_text})
        if not isinstance(ref, dict):
            continue
        got = ref_dict_to_image_part(ref, max_total_pixels=max_total_pixels, max_edge=max_edge)
        if not got:
            continue
        part, m = got
        out.append(part)
        metas.append(m)
    out.append({"type": "text", "text": epilogue})
    return BuildContentResult(user_content=out, image_metas=metas)


def local_file_to_vlm_data_url(
    local_path: str,
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> Optional[str]:
    part = image_item_to_image_part(
        local_path, max_total_pixels=max_total_pixels, max_edge=max_edge
    )
    if not part:
        return None
    return str((part[0].get("image_url") or {}).get("url") or "")


# Backwards name used by batch_sft
def local_file_to_data_url_vlm(
    local_path: str, *, request_timeout: float = 0.0, max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS
) -> Optional[str]:
    _ = request_timeout
    return local_file_to_vlm_data_url(
        local_path, max_total_pixels=max_total_pixels, max_edge=0
    )


def build_user_content_text_then_image_parts(
    image_items: List[ImageItem],
    *,
    max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    max_edge: int = 0,
) -> List[Dict[str, Any]]:
    """
    Build only the image *parts* (no text). One ordered slot per list entry:
    Pillow, bytes, path, or http URL string. No HTTP download.
    """
    parts: List[Dict[str, Any]] = []
    for item in image_items:
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        got = image_item_to_image_part(
            item, max_total_pixels=max_total_pixels, max_edge=max_edge
        )
        if not got:
            continue
        parts.append(got[0])
    return parts
