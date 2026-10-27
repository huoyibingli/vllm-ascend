# NZ guard audit for Qwen3.5 on v0.30.0rc2

[PR #16058](https://github.com/vllm-project/vllm-ascend/pull/16058) is a bug
fix for weights with `k=1` or `n=1`, not the introduction of NZ support.
Its merged commit `d43f2475ab429373c647618a86a1b1602650fb72` is already an
ancestor of tag `v0.30.0rc2` (`8fde6d4e3`). No duplicate production patch is
needed on this independent feature branch.

`vllm_ascend/utils.py::_should_trans_nz` excludes dimension-one weights
before hardware layout policy, including 310P's FORCE_NZ policy. Existing
`tests/ut/test_utils.py::TestUtils::test_maybe_trans_nz` cases cover `(32,1)`,
`(1,64)` and batched `(2,32,1)` weights on configurable and forced policies.
The linear dispatch also keeps the relevant compatibility weights in ND.

Weight layout is controlled by `additional_config.weight_nz_mode`:

- `0`: disable configurable NZ conversion.
- `1` (default): select NZ automatically according to dtype/device policy.
- `2`: prefer NZ on supported configurable weights regardless of model dtype.

```bash
vllm serve <model> --additional-config '{"weight_nz_mode":2}'
```

This flag does not override unsupported shapes or 310P's hardware policy.
The bug fix is automatic and has no independent switch. Convolution weights
retain their separate layout path. The experimental Qwen3.5 weight-prefetch
feature accepts only matching BF16 ND storage; set `weight_nz_mode=0` for
that experiment and confirm its bind logs. Attention KV-cache NZ/C8 layout
is a separate configuration from linear weight NZ.
