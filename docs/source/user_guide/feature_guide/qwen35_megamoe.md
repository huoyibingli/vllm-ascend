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

Shared expert multistream overlap is automatically disabled by the existing
fused-MC2 configuration mutex. The decode weight-prefetch feature requires
EP disabled and therefore uses a different topology.

Validation: added selection boundary and topology regression tests plus the
updated A2 capability expectation. Full UT and NPU service validation require
the matching vLLM, torch_npu, CANN, and MegaMoe extension environment.
