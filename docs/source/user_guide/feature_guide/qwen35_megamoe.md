# Qwen3.5 A2 MegaMoe prefill

Port of [PR #18821](https://github.com/vllm-project/vllm-ascend/pull/18821)
to `v0.30.0rc2`. Install the compatible custom `cann_ops_transformer` package
before starting workers, then use TP4, EP enabled, DP1:

```bash
vllm serve <Qwen3.5-35B-A3B-model> -tp 4 --enable-expert-parallel \
  --additional-config '{"enable_fused_mc2":2,"mega_moe_prefill_token_threshold":64}'
```

`enable_fused_mc2=2` retains MegaMoe; `1` explicitly rolls back to
`dispatch_ffn_combine`. The internal normalized value is `1` in both cases,
so check `use_cann_megamoe=True` and the selected `FUSED_MC2` route in logs.
`mega_moe_max_tokens` is the received-token capacity per rank, not a batch
selection threshold (default 65536). The example retains that default; size it for the actual routing workload to avoid token
truncation. `mega_moe_prefill_token_threshold` is a non-negative token count
(default 64); batches must be strictly larger to use the new route.

Only pure prefill or profiling uses the new A2 route. Decode, mixed batches,
unknown phase metadata, DP>1, and speculative decoding retain the ordinary
selection. DP>1 needs a global phase agreement before a phase-dependent
collective can be used safely. A3/A5 policies are unchanged.

The default fused-MC2/shared-overlap mutex remains. A2 BF16 TP2/EP2 DP1/PP1
can explicitly compose them with `allow_megamoe_shared_expert_overlap=true`.
Pure-decode AllGather keeps its four-stage shared scheduling. Fused prefill
publishes only its real enqueue boundary: shared Gate-Up is submitted after
the router, and the remaining shared MLP waits for fused start on the side
stream. Internal dispatch/GMM/combine events are not fabricated.

## Optional static NZ decode cache

A2 CANN MegaMoe requires per-expert **ND** weights for BF16. Its weight loader
therefore bypasses NZ conversion, including for later ordinary decode. This
branch offers a default-off second layout for selected decode layers:

```json
{
  "weight_nz_mode": 2,
  "enable_fused_mc2": 2,
  "multistream_overlap_shared_expert": true,
  "allow_megamoe_shared_expert_overlap": true,
  "qwen35_megamoe_decode_nz": {
    "enabled": true,
    "layers": [0, 1, 2, 3],
    "max_cache_bytes": 4294967296,
    "min_free_bytes": 2147483648
  }
}
```

The 40-layer, hidden-2048, 256-expert, intermediate-512 BF16 model with EP2
needs **0.75 GiB per selected layer per rank**: 3 GiB for four layers, 30 GiB
for all 40. `layers=null` selects all layers and requires an explicit budget
of at least 32212254720 bytes. The budget is an upper limit, not a reservation.
The remaining layers keep their previous ND-list decode path. Full-layer NZ
and MegaMoe cannot share one storage layout with this CANN BF16 interface.
Do not assume a two-card 64 GiB deployment has room for both full layouts,
KV cache and MegaMoe workspace.

After the ordinary model load, the binder validates shapes, per-expert ND
formats, selected layers and the byte budget. It checks free NPU memory for
the entire selected cache, one layer of temporary stacking storage and the
configured reserve. This reserve does not guarantee capacity for all later
workspace/KV allocations; normal vLLM memory profiling remains required.
Caches are nonpersistent model buffers with stable addresses, bound before
graph capture. Their actual allocator-byte delta is added to model_memory_usage
so KV memory profiling accounts for the second weight layout. Failure rolls back every cache created by that attempt.
There are no decode-time format casts or graph-address swaps. MegaMoe
continues to receive the original ND lists, while ordinary GMM reads the NZ
stack only for selected layers. The cache is never silently disabled.

Supported scope: static Qwen3.5-35B BF16 A2 TP2/EP2, DP1/PP1, runner v1,
NZ mode2, no FlashComm/SP/CP, quantization, LoRA, speculation, dynamic/forced
EPLB, sleep or weight transfer. Online reload is rejected while enabled;
restart the worker to change weights or the layer selection.

## Validation and production comparison

60 CPU dependency-isolated tests cover cache budgets, original ND-list retention,
NZ getter selection, failure rollback, lifecycle configuration and the real
fused/shared enqueue boundary. They do not validate NPU numerical results.
The opt-in two-card test is:

```bash
MODEL_PATH=/path/to/Qwen3.5-35B pytest -sv \
  tests/e2e/pull_request/two_card/test_qwen35_megamoe_decode_nz.py
```

It uses the repository's existing prefill/decode logprob checks for both
eager and FULL_DECODE_ONLY execution. Run this before performance acceptance.
NPU tests have not been run on this branch, and no speedup is claimed.

Compare cumulative stage04, the original stage05, and this branch with the
same 32768 context limit, two cards, model, dependencies, random workload and
prefix-cache policy. Record the actual selected cache layers and bytes.
Four cached layers are a partial-layout experiment, not full-model NZ parity.
For isolating the new cache itself, compare this branch with cache disabled
and enabled so the fused/shared enqueue policy stays identical.

The BF16 ND/2D contract was checked against the local ops-transformer
`mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp`,
`CheckWeight1Input`/`CheckWeight2Input`, checkout a7ffb1d4536b2f6ab2c7c1b9db6f14495d87afd4.
Confirm the installed CANN/MegaMoe package matches the production contract.
