# Quick Start

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Offline end-to-end demo

Start the included fake OpenAI-compatible server in terminal 1:

```bash
python examples/fake_openai_server.py --port 8765
```

Run the Agentic Reasoner with Search Tools in terminal 2:

```bash
export NO_PROXY=127.0.0.1,localhost

searchgen-reason \
  --input examples/prompts.jsonl \
  --output-dir runs/demo \
  --lane-name agentic_reasoner \
  --model offline-demo \
  --chat-base-url http://127.0.0.1:8765/v1 \
  --workers-s1 1 \
  --workers-s1d1-search 1 \
  --workers-s1d1-download 1 \
  --workers-s2 1 \
  --workers-s3 1 \
  --yes
```

Generate a test image from the resulting S4 manifest:

```bash
searchgen-generate \
  --lane-dir runs/demo/agentic_reasoner \
  --generator-name mock \
  --backend mock \
  --model offline-demo \
  --row-index 1
```

Expected files:

```text
runs/demo/agentic_reasoner/eval_row_001/artifacts_files/s1_analysis.json
runs/demo/agentic_reasoner/eval_row_001/artifacts_files/s3_refined_prompt.json
runs/demo/agentic_reasoner/eval_row_001/artifacts_files/s4_generation_manifest.json
runs/demo/agentic_reasoner/eval_row_001/mock_generator/generated_image.png
```

This demo intentionally produces no S2 file: the fake S1 response has an empty reference plan, so S2 is legitimately skipped.

## Real providers

For a real run, configure an OpenAI-compatible text/multimodal endpoint. Configure search only if prompts may require external grounding. Search credentials are supplied by a user signer documented in `README.md`.

For image generation, implement the callback in `examples/custom_generator.py`, then run:

```bash
searchgen-generate \
  --lane-dir runs/demo/agentic_reasoner \
  --generator-name my_generator \
  --backend plugin \
  --generator-plugin my_project.generator:generate_image \
  --model my-image-model \
  --all-rows \
  --workers 2
```
