# Qwen3.5 A2 MegaMoe BF16 NZ experiment

`additional_config.mega_moe_bf16_nz=true` selects a single persistent set of
batched 3D BF16 FRACTAL_NZ parameters for both MegaMoe and ordinary GMM. It defaults
to false and requires `enable_fused_mc2=2`. Conversion happens after loading,
before profile/graph capture, only when the source is ND; already-NZ tensors are
reused. The loader retains `w13_weight` and `w2_weight` as registered parameters.
No per-expert split, forward-time ND conversion or weight swap is
introduced. Activations and final outputs remain ND.

This port targets static Qwen3.5-35B-A3B, BF16, A2, TP2/EP2, DP1/PP1 and runner
v1. FlashComm, SP/CP, quantization, LoRA, speculative execution, EPLB, sleep and
weight/KV transfer are rejected. Pure-prefill MegaMoe selection is retained;
this branch does not include the separate mixed-prefill experiment.

## Operator requirement

Install matching new operator binaries and the Torch adapter containing the
A2 **batched** BF16 NZ support from `feature/mega-moe-bf16-nz-910b`
(commit `be5f766ef` or a compatible successor).
The actual `cann_ops_transformer.ops` imported by vLLM must
refer to that build. Installing only a separate custom namespace is insufficient.
The installed package cannot be verified from Python function signatures alone;
run the two-card regression before serving. There is no silent fallback to ND.

The new package must accept single-item TensorLists containing complete 3D
format-29 tensors with zero storage offset: W13 `[128,2048,1024]` and W2
`[128,512,2048]`, with physical NZ shapes `[128,64,128,16,16]` and
`[128,128,32,16,16]`. Expert strides are 2,097,152 and 1,048,576 BF16 elements.
The operator must set `expertPerRank=128` and `listLen=1`; the old split-only
BF16 NZ package is insufficient. Both backends read the same two allocations.
It must take I=512 in `get_symm_buffer_for_mega_moe(intermediate_hidden=...)`;
its allocator internally uses 2I. The flag selects that contract while legacy
flag-off execution preserves the earlier 2I call convention. Do not enable this
flag against an old ND-only operator or update legacy package semantics blindly.

## Single-feature branch

```bash
git fetch origin test/single-megamoe-bf16-nz
git switch test/single-megamoe-bf16-nz
ASCEND_RT_VISIBLE_DEVICES=2,3 MODEL_PATH=/data/Qwen3.5-35B-A3B \
  bash examples/qwen35_megamoe_bf16_nz/serve_single.sh
```

When `origin` points upstream, fetch the branch from the user's fork instead.
The single-feature configuration keeps other optimizations disabled and GDN on
its default path. `weight_nz_mode=1` leaves unrelated dense weights under the
baseline policy; routed experts use NZ because of the explicit new flag.
Preserve CANN's environment without adding a source-tree PYTHONPATH override.

## Validation

```bash
pytest -q tests/ut/ops/test_megamoe_bf16_nz.py tests/ut/test_qwen35_megamoe.py
# Only after confirming both selected physical cards are idle:
ASCEND_RT_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 \
  HCCL_IF_IP=127.0.0.1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=2048 \
  torchrun --standalone --nproc-per-node=2 \
  tests/e2e/nightly/single_node/ops/test_mega_moe_bf16_nz.py
```

The two-card test checks ordinary GMM and MegaMoe against identity-expert
references using the same NZ tensors, odd-token masks/finalize at 65, 1055,
15361, 15376 and 16384 global tokens, expert counts and stable addresses.
It does not substitute for random-weight/operator or full-model accuracy tests.

Performance is unverified. Run five pure-text rounds per case, retain prefix
cache, and compare the mean output throughput and mean TTFT of rounds 3-5.
Keep the baseline and operator package versions fixed within a comparison.
