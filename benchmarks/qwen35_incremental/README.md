# Qwen3.5 cumulative feature benchmarks

Each `test/NN-*` branch contains its predecessor and exactly one additional
feature group. PRs #16335 and #16336 enter together at stage 02. Production
sources on `test/00-baseline` match tag `v0.30.0rc2`; only this benchmark harness
is added. `stage.json` records the active switches and fixed serving profile.

The default experiment uses Ascend A2, BF16, TP2 with EP enabled, DP1/PP1,
runner v1, no quantization/LoRA/speculation/SP, and ViT weight sharding. Keep
the same checkpoint, dependencies (including FLA and MegaMoe packages), graph
settings, HCCL settings, dataset, sampling and concurrency on every branch.
If another topology is required, change it consistently for every stage.

```bash
python3 benchmarks/qwen35_incremental/run_server.py /path/to/model --dry-run
python3 benchmarks/qwen35_incremental/run_server.py /path/to/model
```

The launcher does not install dependencies, download models, or run benchmarks.
Restart between stages and use separate compilation caches. Hold all non-feature
configuration constant. Stage 01 tests enabling `weight_nz_mode=2`; the singleton
NZ bug fix is already in the baseline. The additional shared-storage conv optimization is excluded from this chain;
baseline convolution packing remains unchanged. Stage 04 tests configurable
FLA/chunk128 against the native default FLA/chunk64 when available.

Use at least three paired repeats of image-heavy, prefill-heavy and decode-heavy
workloads. Decode concurrency should include 1, 8, 32 and 64; prefill sizes must
cross the MegaMoe threshold. Check correctness before accepting timings. Confirm
actual feature hits: ViT fusion needs eight local heads and D72; FLA must import
successfully; shared four-phase needs pure decode ALLGATHER; MegaMoe needs its
extension and a pure prefill; prefetch needs nonzero bound targets. A fallback
run cannot establish the intended feature's performance.

Copy `results.template.csv` outside the checkout and fill it only with actual
measurements. `case_fingerprint` must identify the fixed checkpoint, software,
topology, workload, sampling and graph settings (excluding the feature switches).
`valid=true` asserts correctness and intended feature-hit checks passed. Store
the exact tested commit in `git_sha` and use the same repeat IDs across stages.

```bash
python3 benchmarks/qwen35_incremental/compare_results.py /path/to/results.csv
```

The report gives per-stage medians, paired-repeat gain over the preceding stage,
and paired-repeat gain over baseline. Positive means improvement: throughput
uses `(current/reference - 1) * 100`; latency uses `(1 - current/reference) * 100`.
Regressions remain negative. Cumulative gains must not be added arithmetically.
No NPU performance measurements are supplied with these branches.

Stage 07 opts into `allow_megamoe_shared_expert_overlap=true` while retaining
`multistream_overlap_shared_expert=true`. This exception is restricted to A2
MegaMoe, EP and DP1/PP1 without quantization/LoRA/speculation. Pure prefill may
select MegaMoe, whose shared work uses the existing no-internal-event fallback;
pure decode ALLGATHER uses the four-phase schedule. Fused backend execution
never waits for nonexistent GMM-stage events. Original mutex behavior remains
unchanged when the opt-in flag is absent. Validate both paths and graph replay
on NPU before accepting performance numbers.

This no-conv test branch retains baseline convolution computation and packed
weights, but removes the additional source/packed weight storage aliasing.
