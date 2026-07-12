# SearchGen Agent

SearchGen Agent provides two adaptable components:

- **Agentic Reasoner with Search Tools** — analyzes an image request, decides whether external grounding is needed, performs web or image search, selects references when useful, and writes a refined generation manifest.
- **Image Generator** — reads the refined manifest and calls a user-defined image-generation function.

```text
JSON/JSONL requests
        │
        ▼
Agentic Reasoner with Search Tools
  analysis → optional search/download → optional reference selection → refinement
        │
        └── artifacts_files/s4_generation_manifest.json
                                      │
                                      ▼
                               Image Generator
                                      │
                                      └── generated_image.png
```

The package contains no service credentials or private endpoints. Users connect their own OpenAI-compatible chat model, search API, and image-generation API by following the contracts below.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For a fully offline example, see [QUICKSTART.md](QUICKSTART.md).

## Commands

```text
searchgen-reason    Run or resume the Agentic Reasoner with Search Tools
searchgen-generate  Run the Image Generator on one row or a complete lane
searchgen-consume   Continuously consume rows that are ready for image generation
```

## 1. Agentic Reasoner input protocol

The input is either:

- a `.jsonl` file containing one JSON object per line;
- a `.json` array; or
- a `.json` object containing an array under `items`, `data`, or `requests`.

Every row must contain a non-empty request under one of these keys:

```text
user_request → user_prompt → prompt → query
```

The first available key is normalized to `user_request` and `user_prompt`. Other row fields are preserved, so applications may attach their own IDs or metadata.

Minimal JSONL example:

```json
{"prompt": "A red apple on a wooden table in soft morning light."}
```

Run it with:

```bash
searchgen-reason \
  --input prompts.jsonl \
  --output-dir runs/example \
  --lane-name agentic_reasoner \
  --model your-multimodal-model \
  --chat-base-url https://chat.example.com/v1 \
  --yes
```

## 2. Chat-model API protocol

The Agentic Reasoner requires an OpenAI-compatible chat-completions endpoint. The resolved request URL is:

```text
<SEARCHGEN_CHAT_BASE_URL>/chat/completions
```

If the configured base does not end in `/v1`, the client inserts `/v1` automatically. The request follows the standard shape:

```json
{
  "model": "your-multimodal-model",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.2,
  "max_tokens": 8192
}
```

The endpoint must return assistant content under:

```text
choices[0].message.content
```

For reference-image stages, user content uses OpenAI-compatible text and `image_url` parts. Local images are encoded as data URLs before transmission.

Configuration:

```bash
export SEARCHGEN_CHAT_BASE_URL=https://chat.example.com/v1
export SEARCHGEN_CHAT_API_FORMAT=openai   # openai or vllm
export SEARCHGEN_CHAT_API_KEY="your-key" # optional for unauthenticated local servers
```

## 3. Custom search API

Search is called only when the Agentic Reasoner decides that web facts or visual references are necessary. One HTTP endpoint handles both web and image search.

Configure it with either:

```bash
searchgen-reason ... --search-api-url https://search.example.com/search
```

or:

```bash
export SEARCHGEN_SEARCH_API_URL=https://search.example.com/search
```

### Web-search request

The pipeline sends `POST` with `Content-Type: application/json`:

```json
{
  "search_type": "web",
  "query": "current Widget X appearance",
  "country": "us",
  "locale": "en",
  "page": 1,
  "num": 10
}
```

### Web-search response

Return HTTP 2xx and:

```json
{
  "success": true,
  "results": [
    {
      "title": "Widget X product page",
      "link": "https://example.com/widget-x",
      "snippet": "Widget X currently has a blue enclosure.",
      "position": 1
    }
  ]
}
```

Required result fields are `title`, `link`, and `snippet`. `position` is optional.

### Image-search request

```json
{
  "search_type": "image",
  "query": "current Widget X product",
  "page": 1,
  "num": 5
}
```

### Image-search response

```json
{
  "success": true,
  "results": [
    {
      "title": "Widget X front view",
      "imageUrl": "https://images.example.com/widget-x.png",
      "thumbnailUrl": "https://images.example.com/widget-x-thumb.png",
      "source": "Example source",
      "link": "https://example.com/widget-x",
      "position": 1
    }
  ]
}
```

`imageUrl` is the full-resolution candidate used for download. It must be reachable from the machine running the pipeline. `thumbnailUrl`, `source`, `link`, and `position` are optional.

### Search failure response

Return a non-2xx HTTP status or:

```json
{
  "success": false,
  "message": "provider-specific error summary",
  "results": []
}
```

### Search authentication/signing hook

If the search service needs custom authentication, implement:

```python
from dataclasses import replace
import os

from searchgen_agent.signing import SearchRequest


def sign_search_request(request: SearchRequest) -> SearchRequest:
    token = os.environ["MY_SEARCH_API_KEY"]
    headers = dict(request.headers)
    headers["Authorization"] = f"Bearer {token}"
    return replace(request, headers=headers)
```

`SearchRequest` contains:

| Field | Type | Meaning |
|---|---|---|
| `method` | `str` | HTTP method, normally `POST` |
| `url` | `str` | Configured search endpoint |
| `headers` | `Mapping[str, str]` | Headers before authentication |
| `query_params` | `Mapping[str, str]` | URL query parameters |
| `json_body` | `Mapping[str, Any]` | Web/image request body shown above |

The signer returns a new `SearchRequest` and may modify the URL, headers, query parameters, or JSON body. It is called immediately before the HTTP request.

Load it with:

```bash
searchgen-reason ... \
  --search-api-url https://search.example.com/search \
  --search-signer my_project.signers:sign_search_request
```

or:

```bash
export SEARCHGEN_SEARCH_SIGNER=my_project.signers:sign_search_request
```

The module path must be importable in the active Python environment. A template is available in `examples/custom_signers.py`.

## 4. Agentic Reasoner output protocol

Each input row becomes:

```text
<output-dir>/<lane-name>/eval_row_NNN/
├── ROW_MANIFEST.json
├── artifacts_files/
    ├── s1_analysis.json
    ├── s1d1_serp_pending.json             # only when search runs
    ├── s1d1_search_results.json            # only when search runs
    ├── s1d1_materialization_log.json
    ├── s2_reference_selection.json         # only when image references are selected
    ├── s3_refined_prompt.json
    ├── s4_generation_manifest.json
    └── refined_prompt.txt
└── reference_images/                       # downloaded image candidates
```

The stable handoff to the Image Generator is `s4_generation_manifest.json`. The Image Generator requires `prompts` and accepts ordered references from `reference_images.ordered_references`.

Example:

```json
{
  "schema_version": 2,
  "stage": "s4_generation_manifest",
  "row_identity": {
    "eval_row_index": 0,
    "evalset": "example",
    "reasoner": "agentic_reasoner"
  },
  "prompts": {
    "user_prompt_raw": "Show the current Widget X product",
    "refined_prompt": "A studio product photograph of the current Widget X...",
    "augmented_mode": "i2i"
  },
  "reference_images": {
    "ordered_references": [
      {
        "index": 0,
        "query": "current Widget X product",
        "title": "Widget X front view",
        "url": "https://images.example.com/widget-x.png",
        "local_path_relative": "reference_images/query_0_ref_0.png",
        "selection_reasoning": "Clear current front view",
        "used_in_refinement": true,
        "borrowed_traits": ["blue enclosure", "front control layout"]
      }
    ],
    "total_count": 1
  }
}
```

Reference order is significant. `local_path_relative` is resolved relative to the row directory. When a usable HTTP URL is present, the Image Generator passes the URL to the custom generator. For a text-only request, `ordered_references` is empty and `augmented_mode` is `t2i`.

Reference selection is optional. A row is valid without `s2_reference_selection.json` when analysis produces no image-reference plan.

## 5. Custom Image Generator API

The Image Generator uses a Python plugin so any HTTP API, SDK, local model, or job system can be adapted without changing pipeline code.

Implement this callable:

```python
from searchgen_agent.generator.api import (
    ImageGenerationRequest,
    ImageGenerationResult,
)


def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    image_bytes = call_your_api(request)
    return ImageGenerationResult(
        image_bytes=image_bytes,
        metadata={"request_id": "provider-request-id"},
    )
```

### Image-generator input

`ImageGenerationRequest` contains:

| Field | Type | Meaning |
|---|---|---|
| `prompt` | `str` | `prompts.refined_prompt`, falling back to `user_prompt_raw` |
| `reference_images` | `Sequence[str]` | Ordered HTTP URLs or resolved local paths; empty for T2I |
| `model` | `str` | Value passed through from `--model` |
| `width` | `int` | Requested width, default `1024` |
| `height` | `int` | Requested height, default `1024` |
| `parameters` | `Mapping[str, Any]` | Reserved provider-specific parameters |

The plugin must preserve `reference_images` order when calling an image-to-image API.

### Image-generator output

Return `ImageGenerationResult`:

| Field | Type | Requirement |
|---|---|---|
| `image_bytes` | `bytes` | Decoded PNG file bytes; must be at least 256 bytes |
| `metadata` | `Mapping[str, Any]` | Optional JSON-serializable provider metadata; never include credentials |

If an HTTP image API returns base64, decode it before returning. If it returns a URL, download the image inside the plugin and convert it to PNG bytes before returning. This keeps the pipeline-facing output contract identical for every provider.

Example HTTP adapter:

```python
import base64
import os
import requests

from searchgen_agent.generator.api import ImageGenerationRequest, ImageGenerationResult


def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    response = requests.post(
        os.environ["MY_IMAGE_API_URL"],
        headers={"Authorization": f"Bearer {os.environ['MY_IMAGE_API_KEY']}"},
        json={
            "model": request.model,
            "prompt": request.prompt,
            "reference_images": list(request.reference_images),
            "width": request.width,
            "height": request.height,
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    return ImageGenerationResult(
        image_bytes=base64.b64decode(payload["image_base64"]),
        metadata={"request_id": payload.get("request_id")},
    )
```

Run the adapter:

```bash
searchgen-generate \
  --lane-dir runs/example/agentic_reasoner \
  --generator-name my_image_api \
  --backend plugin \
  --generator-plugin my_project.image_api:generate_image \
  --model my-image-model \
  --all-rows \
  --workers 2
```

The plugin module must be importable in the active Python environment. A template is available in `examples/custom_generator.py`.

## 6. Image Generator output protocol

For each row, successful generation writes:

```text
eval_row_NNN/<generator-name>_generator/
├── generated_image.png
└── generation_meta.json
```

`generation_meta.json` contains:

```json
{
  "schema_version": 1,
  "created_at": "2026-01-01T00:00:00+00:00",
  "generator": "my_image_api",
  "model": "my-image-model",
  "reference_count": 1,
  "provider_metadata": {
    "request_id": "provider-request-id"
  }
}
```

On failure, the Image Generator writes `last_generation_failure.json` containing `error_type` and an error summary. It does not write a successful image.

An existing `generated_image.png` of at least 256 bytes is treated as complete. Use `--overwrite` to regenerate it.

## 7. Resume and batch consumption

The Agentic Reasoner classifies existing artifacts and queues only missing stages. Re-running the same input/output/lane resumes without repeating completed model or search calls.

To generate rows as soon as their manifests appear:

```bash
searchgen-consume \
  --lane-dir runs/example/agentic_reasoner \
  --generator-name my_image_api \
  --backend plugin \
  --generator-plugin my_project.image_api:generate_image \
  --model my-image-model \
  --workers 2 \
  --follow
```

## Testing

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The offline suite covers text-only reasoning, web search, resume behavior, the search signing hook, ordered references, manifest-to-image generation, and package interfaces. It makes no external network requests.

## License

Licensed under the [Apache License 2.0](LICENSE).
