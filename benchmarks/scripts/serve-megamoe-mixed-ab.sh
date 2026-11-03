#!/usr/bin/env bash
set -euo pipefail

scenario=${1:?Usage: bash serve-megamoe-mixed-ab.sh baseline|megamoe}
case "$scenario" in
  baseline)
    extra='{"weight_nz_mode":1,"enable_flashcomm1":false,"multistream_overlap_shared_expert":false,"enable_fused_mc2":0,"enable_cpu_binding":true}'
    ;;
  megamoe)
    extra='{"weight_nz_mode":1,"enable_flashcomm1":false,"multistream_overlap_shared_expert":false,"enable_fused_mc2":2,"enable_cpu_binding":true,"mega_moe_prefill_token_threshold":64,"mega_moe_max_tokens":131072,"mega_moe_allow_mixed_prefill":true}'
    ;;
  *) printf 'Unknown scenario: %s\n' "$scenario" >&2; exit 2 ;;
esac

model_path=${MODEL_PATH:-/data/Qwen3.5-35B-A3B}
case_dir=${LOG_ROOT:-$PWD/perf_runs/megamoe-mixed}/$scenario
mkdir -p "$case_dir"
# Preserve the CANN environment; do not add a source-tree PYTHONPATH override.
export ASCEND_RT_VISIBLE_DEVICES=2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
export HCCL_IF_IP=127.0.0.1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=2048
export VLLM_ENGINE_READY_TIMEOUT_S=3600

exec vllm serve "$model_path" \
  --host 0.0.0.0 --port 8899 \
  --data-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel \
  --seed 1024 --served-model-name qwen3.5-35b \
  --max-num-seqs 32 --max-model-len 32768 --max-num-batched-tokens 16384 \
  --trust-remote-code --gpu-memory-utilization 0.8 \
  --enable-chunked-prefill --enable-prefix-caching --dtype bfloat16 \
  --async-scheduling --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}' \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$case_dir/profile\",\"torch_profiler_with_stack\":false}" \
  --additional-config "$extra"
