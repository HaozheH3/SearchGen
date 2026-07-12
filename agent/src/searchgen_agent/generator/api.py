"""Provider-neutral image-generation interface."""

from __future__ import annotations

import hashlib
import struct
import textwrap
import zlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    prompt: str
    reference_images: Sequence[str] = ()
    model: str = ""
    width: int = 1024
    height: int = 1024
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    image_bytes: bytes = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImageGenerator(Protocol):
    def __call__(self, request: ImageGenerationRequest, /) -> ImageGenerationResult:
        """Generate one image; provider credentials remain inside this callback."""


def mock_image_generator(request: ImageGenerationRequest) -> ImageGenerationResult:
    """Offline generator used by installation tests and the documented demo."""

    width = max(256, min(2048, int(request.width)))
    height = max(256, min(2048, int(request.height)))
    seed = hashlib.sha256(request.prompt.encode("utf-8")).digest()

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows: list[bytes] = []
    for y in range(height):
        pixels = bytearray()
        for x in range(width):
            pixels.extend(
                (
                    (seed[0] + x // 3 + y // 7) % 256,
                    (seed[1] + x // 9 + y // 2) % 256,
                    (seed[2] + x // 5 + y // 5) % 256,
                )
            )
        rows.append(b"\x00" + bytes(pixels))
    description = ("OFFLINE MOCK: " + " ".join(textwrap.wrap(request.prompt, width=80)))[:1800]
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"tEXt", b"Description\x00" + description.encode("utf-8", errors="replace"))
    png += chunk(b"IDAT", zlib.compress(b"".join(rows), level=6))
    png += chunk(b"IEND", b"")
    return ImageGenerationResult(
        image_bytes=png,
        metadata={"backend": "mock", "reference_count": len(request.reference_images)},
    )
