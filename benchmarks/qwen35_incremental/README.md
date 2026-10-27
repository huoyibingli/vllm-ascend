# Qwen3.5 cumulative feature benchmarks

Each `test/NN-*` branch contains its predecessor and exactly one additional
feature group. PRs #16335 and #16336 enter together at stage 03. Production
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
NZ bug fix is already in the baseline. Stage 02 tests the new single-storage
conv optimization, since baseline already packs convolution weights. Stage 05
tests configurable FLA/chunk128 against the baseline fused prefill/chunk64.

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

Stage 08 retains NZ and EP. Its prefetch config explicitly opts into
`allow_ep_nz_dense` for `next_input` and `gdn_out`: the torch_npu prefetch API
accepts weight data independently of layout. It passes the loaded NZ tensor
directly, uses bounded byte prefixes, and never converts it to ND or replaces
its storage. These dense attention projections remain TP2-sharded with EP2.
The existing ALLGATHER pure-decode and minimum-concurrency gates still apply;
MegaMoe prefill does not prefetch. Missing bound targets fail startup instead
of silently measuring an inactive feature. The routed_w2 Ascend C target remains
an independent ND/no-EP experiment and is rejected by the EP/NZ composition flag.
See the official API contract:
https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_prefetch.md
