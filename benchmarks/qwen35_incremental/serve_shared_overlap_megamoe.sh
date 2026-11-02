#!/usr/bin/env bash
# shared_overlap_megamoe | native default GDN selection; no source patches
set -euo pipefail
REPO="${REPO:-/vllm-workspace/vllm-ascend}"
MODEL_PATH="${MODEL_PATH:-/home/bzcai/weights}"
LOG_ROOT="${LOG_ROOT:-/home/bzcai/mhd/perf}"
mkdir -p "$LOG_ROOT/shared_overlap_megamoe/profile"
unset BENCH_GDN_BACKEND BENCH_CASE_DIR
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
export HCCL_IF_IP=127.0.0.1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=2048
export VLLM_ENGINE_READY_TIMEOUT_S=3600

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "${PORT:-8899}" \
  --data-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel \
  --seed 1024 --served-model-name "${MODEL_NAME:-qwen3.5-35b}" \
  --max-num-seqs 32 --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill --enable-prefix-caching --dtype bfloat16 \
  --async-scheduling --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$LOG_ROOT/shared_overlap_megamoe/profile\",\"torch_profiler_with_stack\":false}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}' \
  --additional-config '{
    "weight_nz_mode": 1,
    "enable_flashcomm1": false,
    "multistream_overlap_shared_expert": true,
    "allow_megamoe_shared_expert_overlap": true,
    "enable_fused_mc2": 2,
    "enable_cpu_binding": true,
    "mega_moe_prefill_token_threshold": 64,
    "mega_moe_max_tokens": 131072
}'
