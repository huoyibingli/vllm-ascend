#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
export HCCL_IF_IP=127.0.0.1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=2048
export VLLM_ENGINE_READY_TIMEOUT_S=3600
exec vllm serve "${MODEL_PATH:-/data/Qwen3.5-35B-A3B}" \
  --host 0.0.0.0 --port "${PORT:-8899}" \
  --data-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel \
  --seed 1024 --served-model-name qwen3.5-35b \
  --max-num-seqs 32 --max-model-len 32768 --max-num-batched-tokens 16384 \
  --trust-remote-code --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill --enable-prefix-caching --dtype bfloat16 \
  --async-scheduling --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}' \
  --additional-config "$(cat "$script_dir/single.json")"
