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

## Shared-expert overlap comparison branch

`test/shared-overlap-megamoe` starts from `test/00-baseline` and combines
the four-phase shared-expert overlap port with A2 prefill MegaMoe. Enable
`multistream_overlap_shared_expert`, `allow_megamoe_shared_expert_overlap`,
and `enable_fused_mc2=2` together. The guard requires A2, EP, DP1/PP1,
and an unquantized supported model without LoRA or speculative decoding.

Pure prefill above the threshold uses MegaMoe. Decode retains ordinary GMM
and the four-phase shared-expert schedule. The fused path publishes only
its real enqueue boundary; it does not expose internal GMM stage events.

Run `bash benchmarks/qwen35_incremental/serve_shared_overlap_megamoe.sh`
with `MODEL_PATH`, `LOG_ROOT`, and optionally `ASCEND_RT_VISIBLE_DEVICES`
set for the machine. The script uses TP2/EP2, BF16, max length 32768,
and the default GDN path. No extra NZ mode, fusion, FLA override, prefetch,
or decode NZ cache is enabled. MegaMoe's existing BF16 loader keeps
per-expert ND weights for both its fused path and ordinary GMM fallback;
this branch does not implement shared BF16 NZ storage. Hardware accuracy
and performance validation are still required.

## BF16 ND weight sharing experiment

`test/shared-overlap-megamoe-nd-shared` isolates the ordinary GMM weight
representation from the prefill backend. For BF16, `weight_nz_mode=1`,
non-MXFP MegaMoe, and static expert placement, ordinary GMM retains the
contiguous three-dimensional ND parameters. MegaMoe receives cached
per-expert two-dimensional views of the same storage. The views are created
once during weight processing, including their expert offsets. They do not
allocate another full set of weights or pack weights during decode.

The original `test/shared-overlap-megamoe` branch remains the comparison
control. Use the same `serve_shared_overlap_megamoe.sh` launcher and model
settings for both branches; this experiment adds no serving option. Dynamic
EPLB, other dtypes, mode 2 and the A5 unquantized fallback retain their
existing behavior. The change does not add BF16 NZ input support to MegaMoe
and does not change its phase selection or shared-expert synchronization.

CPU contract tests cover storage aliases, stable cached views, in-place
updates, loader metadata, GMM arguments and the excluded layouts. The
two-card test compares stacked ND and independent-list GMM results, and
MegaMoe views against cloned expert tensors, using the Qwen3.5 expert shapes.
It covers 32/65/512 tokens, all expert offsets and inactive mask entries.

Run the hardware check with two idle A2 devices assigned by the caller:

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node=2 \
  tests/e2e/pull_request/two_card/test_megamoe_nd_weight_sharing_two_card.py \
  --output-dir /tmp/megamoe-nd-views
```

The standalone check requires the installed CANN MegaMoe extension and the
normal CANN environment. Its results do not replace full-model accuracy
and end-to-end performance validation. Compare the serving branches with
the same input workload and keep the original five-round measurement
policy. The previous four-scenario results do not measure this change.

The 2026-11-02 paired serving run tested code commit `34d739b6f` against
the original combined commit `2381e3ca9`. Both used five rounds of the same
1024-input/1024-output random workload, 64 requests and concurrency 32;
the final three rounds were averaged. Output throughput was 914.83 versus
915.55 tokens/s (+0.08%), with throughput CVs of 0.43% and 0.50%.
Mean TTFT was 1374.98 versus 1364.82 ms. This run did not establish a
discernible throughput improvement, and the experimental branch is retained
for isolating layout effects rather than claiming a resolved regression.
The six native unit tests and the committed two-card exact-output test
passed. Full `format.sh ci` dependency setup was blocked by Go proxy
connection timeouts; relevant Ruff and syntax checks passed.
