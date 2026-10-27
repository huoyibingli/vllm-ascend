# Qwen3-family fusion on v0.30.0rc2

Ports [PR #16335](https://github.com/vllm-project/vllm-ascend/pull/16335)
and [PR #16336](https://github.com/vllm-project/vllm-ascend/pull/16336).

The existing split-QKV/RMSNorm/MRoPE operator now computes cosine and sine
inline from positions and a registered FP32 inverse-frequency buffer. Supported
multi-axis Qwen3-VL/Qwen3.5 calls enable it automatically, with the cached
cos/sin interface retained for other calls. No extra switch is required.

Vision attention automatically fuses QKV extraction, Q/K RoPE, and padding
from 72 to 128 for contiguous BF16 inputs with eight local heads, batch size
one, and packed sequence metadata. Other shapes/dtypes use the original path.
With head sharding, the local head count must still be eight. Encoder ACL
graph capture retains the original graph buffer and update machinery.

Enable `VLLM_ASCEND_VIT_FUSED_QKV_ROPE_PAD_STATS=1` before worker startup for
periodic fusion/fallback counters. `VLLM_ASCEND_VIT_FUSED_QKV_ROPE_PAD_CORE_MODE`,
`VLLM_ASCEND_VIT_FUSED_QKV_ROPE_PAD_BLOCK_T`,
`VLLM_ASCEND_VIT_FUSED_QKV_ROPE_PAD_STORE_MODE`, and
`VLLM_ASCEND_VIT_FUSED_QKV_ROPE_PAD_OUT_MODE` are launch tuning options; defaults
are defined in `vllm_ascend/envs.py`. There is no master fusion enable flag.

The #16336 snapshot also includes zero-copy FIA read aliases for compatible
interleaved BF16/FP16 KV caches. Writers, cache capacity and physical page
ownership remain unchanged. C8/NZ, BNSD, sinks, sliding window and context
parallel cases retain their original paths.

The image patchify feature is complementary: patchify runs before ViT;
these fusions run inside ViT and the language decoder. Validate fused and
fallback inputs and encoder graph capture separately on the target NPU.
