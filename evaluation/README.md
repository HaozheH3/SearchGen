# SearchGen evaluation

This directory is a standalone release of the canonical ten-component evaluation protocol. Its vendored evaluation template and parser are immutable and preserve the production behavior.

## Install and run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp examples/env.example .env
# Edit .env with your own API URL and key, then export its values:
set -a; . ./.env; set +a
python evaluate.py \
  --metadata ../../data_release/searchgen-bench/eval_metadata.jsonl \
  --benchmark-root ../../data_release/searchgen-bench \
  --predictions-manifest predictions.jsonl \
  --output-dir results --include-pp --workers 16 \
  --model your-judge-model
```

Use `--preflight` to validate all metadata, references, and predictions without API calls. `--dry-run` additionally reports pending versus resumable jobs. Filter with repeatable `--bench-id`, `--generator`, and `--limit`. A valid result with `success=true` is resumed automatically.

The API must expose an OpenAI-compatible chat-completions endpoint. Configure `SEARCHGEN_EVAL_API_URL` and `SEARCHGEN_EVAL_API_KEY`, or pass `--endpoint` and `--api-key`. Replace `your-judge-model` with the model name accepted by your API. All paths shown above are release-relative or user-supplied; the package contains no private path, endpoint, credential, quota ID, or credential default.

Each job writes under `results/{generator}/{bench_id}/`:

- `augmented_parsed_result_evaluation_protocol.json`
- `augmented_prompt_context_evaluation_protocol.txt`
- `augmented_raw_api_output_evaluation_protocol.txt`

Aggregate with `python aggregate_scores.py results --missing-policy skip`. Choose `zero` only when missing components should explicitly count as zero.

See `docs/` for protocols and schemas. Live smoke tests require user-supplied API configuration and are intentionally not run during packaging.
