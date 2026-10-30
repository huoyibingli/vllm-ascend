# Qwen3.5 Decode Weight Prefetch

## Overview

This experimental feature overlaps bounded weight prefetches with decode work for Qwen3.5-35B-A3B. It is disabled by default and should only be enabled after workload-specific correctness and performance testing.

The supported targets are:

- `next_input`: prefetch the next layer's fused GDN input or full-attention QKV weight while tensor-parallel all-reduce is queued.
- `gdn_out`: prefetch the next GDN layer's output projection across the preceding MoE collective and attention work; layer zero falls back to same-layer prefetch.
- `routed_w2`: warm a budgeted portion of active experts' `w2` weights before the down projection. This target requires the optional Ascend C extension described below.

## Compatibility

The current implementation is limited to Ascend A2/910B, Qwen3.5-35B-A3B's 40-layer text model, BF16, TP=2, DP=1, PP=1, with quantization, LoRA, and speculative decoding disabled. EP2 and dense ND/NZ weights can be explicitly enabled using `allow_ep_nz_dense=true`; routed_w2 remains excluded in that composition. Without that opt-in, unsupported model or topology combinations leave the regular model path in use. The dense EP/NZ opt-in rejects unsupported topology or missing targets instead of silently reporting an active experiment. Sleep mode, weight transfer, and weight reload are not supported while the feature is enabled because graph capture is tied to the loaded weight storage.

Model runner v1 is required. Use `weight_nz_mode=0` for the supported ND storage and `enable_flashcomm1=false` for the ALLGATHER path. NZ weights require the explicit dense EP/NZ opt-in above.

Configuration is read at model startup. Restart the service and use a fresh compilation cache when changing it.

## Configuration

Pass `qwen35_weight_prefetch` through `--additional-config`. For example:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --additional-config '{"weight_nz_mode":0,"enable_flashcomm1":false,"qwen35_weight_prefetch":{"enabled":true,"targets":["next_input"],"next_input_bytes":2097152,"min_decode_concurrency":8}}'
```

The default is equivalent to `{"enabled": false}`. The supported fields are:

| Field | Default | Description |
|---|---:|---|
| `allow_ep_nz_dense` | `false` | Opt in to validated TP2/EP2 dense ND/NZ projection weights; excludes routed_w2. |
| `enabled` | `false` | Master switch. |
| `mode` | `"full"` | `"full"` issues prefetch transfers; `"control"` keeps stream/wait plumbing without data movement. |
| `targets` | `["next_input"]` | Any non-empty, duplicate-free combination of `next_input`, `gdn_out`, and `routed_w2`. |
| `next_input_bytes` | `2097152` | Bytes to prefetch for `next_input`, or `"full"` for the entire weight. |
| `gdn_out_bytes` | `2097152` | Byte budget for `gdn_out`. |
| `routed_w2_bytes` | `4194304` | Byte budget for `routed_w2`. |
| `layers` | `null` | Optional list of zero-based layer indices; unset selects all eligible layers. |
| `next_layer_types` | `["linear_attention", "full_attention"]` | Eligible next-layer attention types. |
| `routed_start` | `"after_dispatch"` | Start `routed_w2` prefetch after dispatch or after the first grouped matmul. |
| `vector_cores` | `2` | Ascend C vector cores used by the routed target; supports 1, 2, or 4. |
| `next_input_join` | `"immediate"` | Join after the dependency collective, or defer the join to the next linear-attention layer. |
| `min_decode_concurrency` | `8` | Skip prefetch below this pure-decode request count. |

Byte budgets must be positive and 32-byte aligned. `next_input_bytes` must not exceed the selected weight's storage; the same bound applies to the routed and GDN budgets.

Deferred `next_input_join` is applied only to linear-attention targets; full-attention targets use the immediate join.

## Building the routed extension

`next_input` and `gdn_out` use `torch_npu.npu_prefetch` directly. To select `routed_w2`, build and install its optional extension on the target Ascend A2 host before starting vLLM:

```bash
VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
ASCEND_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
bash scripts/build_qwen35_routed_prefetch.sh
```

Stop inference services before running the script; it checks for running vLLM processes and installs the resulting extension under `vllm_ascend/ops/`.

## Validation guidance

The feature is opt-in and experimental. Compare prefetch-off and prefetch-on runs with the same model, prompt set, output length, graph mode, parallel settings, and cache state. Check output consistency, target-operator and consumer-wait timing, throughput, median and tail latency, and whether the dependency window regresses. Aggregate benchmark summaries alone do not establish that prefetch caused a change. Keep the feature disabled if the result is neutral, unstable, or slower.

## Cumulative TP2/EP2/NZ benchmark composition

`test/text-06-weight-prefetch-v2` retains the opt-in `allow_ep_nz_dense=true` option.
It permits EP with TP2 and dense BF16 weights in ND or NZ for `next_input`
and `gdn_out`. Budgets remain bounded by the logical weight byte count,
which does not require interpreting NZ expert strides. No conversion or
weight replacement is performed. The routed_w2 target is excluded from this
composition because its kernel expects contiguous ND expert storage.
Stage05 (`test/text-05-weight-prefetch`) keeps NZ mode2 and shared overlap,
with fused MC2 disabled. Stage06 (`test/text-06-megamoe`) adds MegaMoe last
and optionally binds per-layer NZ caches for ordinary GMM. Both stages use
the same v2 dense prefetch implementation on pure-decode ALLGATHER, with
only next_input and gdn_out enabled. Routed expert caches do not replace
the dense projections targeted by prefetch.
NPU capture/replay, correctness and performance validation remain required.

## Independent input/output prefetch timelines

When both dense targets are enabled, input and GDN output weights use separate
side streams. The existing immediate/deferred input join only waits for the
input stream. Output prefetch snapshots the dependency before the preceding
collective, submits after the collective is enqueued, and waits immediately
before the target out_proj consumes the weight. The first GDN layer retains
the same-layer fallback. Single-target configurations keep one side stream.

A pending output prefetch suppresses duplicate same-layer submissions. The
routed completion buffer and extension are allocated only for routed_w2.
Control mode retains the same waits without data movement. Reloaded/replaced
weights remain guarded by identity, storage address and shape checks.

This scheduling change needs NPU graph capture/replay, output consistency and
performance validation. Earlier output-weight fetches may be evicted or contend
for bandwidth; no performance improvement is claimed without measurement.
Keep byte budgets, minimum decode concurrency and all serving settings fixed
when comparing the original and candidate implementation.

For the 40-layer model with both dense targets bound, the startup log should
report `next_inputs=39`, `gdn_out=30` and `gdn_lookahead=29`. This confirms
binding only; inspect the device timeline to confirm overlap and cache effects.
