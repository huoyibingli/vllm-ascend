# A2 MegaMoe mixed prefill experiment

`mega_moe_allow_mixed_prefill` is an opt-in `additional_config` boolean, default
`false`. On the A2 expert-density policy it extends the existing pure-prefill
MegaMoe route to eager batches containing both prefill and decode tokens.
The fused operator processes the entire batch's MoE stage. Attention scheduling
and the distinction between prefill and decode requests remain intact.

The mixed route requires BF16 nonquantized weights, DP1, PP1, no context or
MoE sequence parallelism, no EPLB, no speculative model, and the existing
MegaMoe model/hardware/package support. LoRA remains excluded by that support
check. The batch must exceed `mega_moe_prefill_token_threshold` (default 64) and
fit `get_mc2_tokens_capacity()`. Graph capture/replay and pure decode retain
their ordinary backend. Missing prefill metadata falls back conservatively.

For Qwen3.5-35B-A3B on TP2/EP2 with scheduler budget 16384, the maximum input
is 8192 tokens per rank. `mega_moe_max_tokens=131072` covers the worst-case
received expert-token load for 16384 global tokens and top-k 8. Input capacity
and received expert-token capacity are separate limits.

This experiment retains the single feature's existing BF16 ND expert lists.
It introduces no shared-expert overlap, weight prefetch or weight-storage
sharing change. `weight_nz_mode=1` matches the earlier baseline configuration;
it does not imply BF16 weights are converted to NZ.

## Validation

Unit tests cover mixed selection, unsupported modes, graph exclusion, capacity
edges, and TP2 odd-token padding/mask/finalize order:

```bash
pytest -q tests/ut/test_qwen35_megamoe.py tests/ut/test_megamoe_mixed_prefill.py
```

After verifying that both physical cards 2 and 3 are idle, run the two-card
operator integration test. It checks exact identity-expert outputs and expert
counts at 65, 1055, 15361, 15376 and 16384 global tokens, using production
prepare/kernel/finalize functions. It does not measure model accuracy.

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 \
  HCCL_IF_IP=127.0.0.1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=2048 \
  torchrun --standalone --nproc-per-node=2 \
  tests/e2e/nightly/single_node/ops/test_mega_moe_mixed_bf16.py
```

## Controlled serving comparison

Copy the two shell scripts outside the checkout before switching branches.
Use the installed editable checkout; do not add a source `PYTHONPATH` or force a
GDN backend. Preserve the installed CANN environment. Stop the preceding owned
service and verify cards and port 8899 are free before every branch switch.

```bash
export BENCH_TOOLS="$PWD/../megamoe-mixed-tools"
mkdir -p "$BENCH_TOOLS"
cp benchmarks/scripts/{serve,bench}-megamoe-mixed-ab.sh "$BENCH_TOOLS/"
export MODEL_PATH=/data/Qwen3.5-35B-A3B
export LOG_ROOT="$PWD/../perf_runs/megamoe-mixed-$(date +%Y%m%d_%H%M%S)"

# Baseline: rc2 runtime, no feature patch and default GDN selection.
git switch --detach de0bbc48fd5a9ebb2f10bfac36dca46b08f501b5
npu-smi info
bash "$BENCH_TOOLS/serve-megamoe-mixed-ab.sh" baseline
# Separate shell with the same MODEL_PATH and LOG_ROOT:
bash "$BENCH_TOOLS/bench-megamoe-mixed-ab.sh" baseline

# After stopping that service and confirming both cards are idle:
git switch test/single-megamoe-mixed
npu-smi info
bash "$BENCH_TOOLS/serve-megamoe-mixed-ab.sh" megamoe
# Separate shell with the same MODEL_PATH and LOG_ROOT:
bash "$BENCH_TOOLS/bench-megamoe-mixed-ab.sh" megamoe
```

Run five rounds per case, keep prefix caching enabled without resetting caches,
and average rounds 3–5. Require 64 completed requests, zero failures and 65536
input/output tokens each round. Report output throughput and mean TTFT. Use
population CV limits of 3% for throughput and 5% for TTFT; if unstable, preserve
all records and repeat the complete five-round group, at most twice. Avoid
selecting individual favorable rounds.

Throughput gain is `new/base - 1`; TTFT improvement is `1 - new/base`.
The server log must show `batch_kind=mixed_prefill_decode` during requests to
confirm actual mixed routing. An `info_once` line proves a hit, not the fraction
of layers or requests using the backend. Profile/initialization hits alone do
not establish workload coverage. Performance remains unverified until this
comparison has run on idle hardware.
