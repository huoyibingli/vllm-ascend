# Qwen3.5 text fusion on v0.30.0rc2

Retains inline split-QKV/RMSNorm/MRoPE with the cached cos/sin fallback.
Also retains compatible interleaved KV-cache zero-copy FIA read aliases.
The extra ViT QKV/RoPE/padding fusion and its tuning/diagnostic variables
are excluded from this text-only experiment. Image preprocessing and the
extra conv source/packed storage aliasing are not part of this test chain.
Baseline packed convolution weights and convolution computation remain.
No new service flag is needed for the retained text fusion paths.
