# AgentGen-Bench evaluation

This directory is a standalone release of the canonical ten-component evaluation protocol. Its vendored evaluation template and parser are immutable and preserve the production behavior.

AgentGen-Bench was previously named **SearchGen-Bench**. The new name reflects
its broader use for evaluating agentic generation systems.

## Install and run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp examples/env.example .env
# Edit .env with your own API URL and key, then export its values:
set -a; . ./.env; set +a
python evaluate.py \
  --metadata /path/to/AgentGen-Bench/eval_metadata.jsonl \
  --benchmark-root /path/to/AgentGen-Bench \
  --predictions-manifest predictions.jsonl \
  --output-dir results --workers 16 \
  --model your-judge-model
```

Use `--preflight` to validate all metadata, references, and predictions without API calls. `--dry-run` additionally reports pending versus resumable jobs. Filter with repeatable `--bench-id`, `--generator`, and `--limit`. A valid result with `success=true` is resumed automatically.

Download [AgentGen-Bench](https://huggingface.co/datasets/JasperHaozhe/AgentGen-Bench) and replace `/path/to/AgentGen-Bench` with its local directory. The API must expose an OpenAI-compatible chat-completions endpoint. Configure `SEARCHGEN_EVAL_API_URL` and `SEARCHGEN_EVAL_API_KEY`, or pass `--endpoint` and `--api-key`. Replace `your-judge-model` with the model name accepted by your API.

For full reproducibility, use
[AgentGen-Bench-Evaluator](https://huggingface.co/JasperHaozhe/AgentGen-Bench-Evaluator),
trained on top of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B):

```bash
pip install vllm
bash serve_evaluator.sh
# Multi-GPU serving uses data parallelism:
bash serve_evaluator.sh --gpus 0,1,2,3
```

The launcher enforces `--chat-template-content-format openai`; do not omit this
flag, because the evaluator expects multimodal OpenAI-style message content.
It derives `--data-parallel-size` from the GPU list and keeps tensor parallel
size at one by default.

Physical-plausibility evaluation is an unconditional part of the SearchGen protocol. There is no option to disable it or select a different protocol.

Each job writes under `results/{generator}/{bench_id}/`:

- `augmented_parsed_result_evaluation_protocol.json`
- `augmented_prompt_context_evaluation_protocol.txt`
- `augmented_raw_api_output_evaluation_protocol.txt`

Aggregate with `python aggregate_scores.py results --missing-policy skip`. Choose `zero` only when missing components should explicitly count as zero.

See `docs/` for protocols and schemas.
