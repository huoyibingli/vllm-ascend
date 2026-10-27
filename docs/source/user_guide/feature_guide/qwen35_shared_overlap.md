# Qwen3.5 four-phase shared-expert overlap

Adapts [PR #16386](https://github.com/vllm-project/vllm-ascend/pull/16386)
to the `v0.30.0rc2` typed MoE MLP and `AscendSharedExperts` executor.
The historical PR's tuple-return and shared-expert split implementation
cannot be applied directly to this version.

```bash
vllm serve <Qwen3.5-35B-A3B-model> -tp 4 --enable-expert-parallel \
  --additional-config '{"multistream_overlap_shared_expert":true,"enable_fused_mc2":0}'
```

The existing switch remains disabled by default. For unquantized, no-LoRA,
pure-decode ALLGATHER calls with all stage events available, the schedule is:

| Phase | Shared expert | Routed expert |
|---|---|---|
| 1 | gate/up projection and expert gate linear | routing/top-k/dispatch |
| 2 | activation and gate sigmoid | GMM1 |
| 3 | down projection | activation |
| 4 | gate multiply | GMM2 |

Other communication methods, prefill/mixed batches, quantization and LoRA
retain the current executor's scheduling. Backends with no internal events
use the existing auxiliary-stream fallback; no wait is introduced for an
event that is never produced. The legacy 310P MLP tuple ABI remains unchanged.

`enable_fused_mc2`/MegaMoe and shared-expert overlap are mutually exclusive in
the baseline configuration validator, which disables shared-expert overlap.
Decode prefetch may share device bandwidth and streams; benchmark that
combination separately even when its TP2/no-EP topology is selected.

CPU regression tests check stage ordering, gated/ungated values and fallback
when stage events are absent. Real stream ordering, ACL graph replay,
correctness and performance still require target NPU validation.
