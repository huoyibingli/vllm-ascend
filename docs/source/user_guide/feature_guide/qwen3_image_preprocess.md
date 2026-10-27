# Qwen3 image patchify on NPU

Port of [PR #15881](https://github.com/vllm-project/vllm-ascend/pull/15881)
to `v0.30.0rc2` and its Transformers 5.14.1 processor contract.

```bash
vllm serve <Qwen3.5-35B-A3B-model> \
  --additional-config '{"enable_qwen3_preprocess_on_npu":true}'
```

Disabled by default. Applies to Qwen3-VL and Qwen3.5 image-only requests.
Resize, rescale and normalization remain on CPU. Flat normalized FP32 CHW
pixels pass through multimodal transport; workers perform patchification and
the upstream runtime-selected dtype conversion on NPU. Consecutive images
with equal resized dimensions are batched without changing request order.

Video or mixed image/video requests and other models retain upstream
preprocessing. This changes the internal pixel transport shape when enabled,
so frontend and worker versions must match. It increases pixel transport
dtype from the normal model dtype to FP32; benchmark transport overhead as
well as CPU preprocessing time.

Vision QKV/RoPE/pad fusion operates after this preprocessing and can be
combined with it. This feature has no EP/TP topology dependency of its own.
CPU tests cover patch ordering, multiple image shapes and invalid transport;
full image/video fallback, graph, accuracy and throughput tests still require
the target inference environment.
