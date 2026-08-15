#!/usr/bin/env bash
# Serve AgentGen-Bench-Evaluator with an OpenAI-compatible vLLM endpoint.

set -euo pipefail

MODEL_PATH="JasperHaozhe/AgentGen-Bench-Evaluator"
PORT=7070
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
DP_SIZE=""
TP_SIZE=1
MAX_MODEL_LEN=20480
GPU_MEM_UTIL=0.75
SERVED_NAME="AgentGen-Bench-Evaluator"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --dp) DP_SIZE="$2"; shift 2 ;;
    --tp) TP_SIZE="$2"; shift 2 ;;
    --name) SERVED_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model <HF-ID-or-path>] [--port <port>] [--gpus <ids>] [--dp <n>] [--tp <n>] [--name <name>]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPUS}"
if [[ -z "${DP_SIZE}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
  DP_SIZE="${#GPU_LIST[@]}"
fi

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --chat-template-content-format openai \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --tensor-parallel-size "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --port "${PORT}"
