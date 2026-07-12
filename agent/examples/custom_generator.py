"""Template plugin for the SearchGen Image Generator."""

from __future__ import annotations

from searchgen_agent.generator.api import ImageGenerationRequest, ImageGenerationResult


def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    """Call your provider and return decoded image bytes.

    Load provider credentials from an environment variable, workload identity,
    or secret manager. Do not put credentials in this file or result metadata.
    """

    raise NotImplementedError(
        "Call your image provider, decode its image response, and return "
        "ImageGenerationResult(image_bytes=image_bytes, metadata={...})"
    )
