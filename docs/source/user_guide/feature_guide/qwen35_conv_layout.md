# GDN convolution weight layout

Adapts [PR #16406](https://github.com/vllm-project/vllm-ascend/pull/16406)
to the registered packed-weight lifecycle already present in `v0.30.0rc2`.

The baseline already packs `[channels,1,width]` convolution weights into
`[width,channels]` after loading, removing repeated forward-time transpose
copies. This branch additionally aliases the source and registered packed
Parameters onto one allocation for supported Qwen GDN floating-point weights.
The source Parameter, checkpoint shape and loader metadata remain intact;
the kernel view stays contiguous. Reload copies into captured packed storage
before restoring the source view, preserving the packed Parameter and pointer.

Enabled automatically after weight loading for exact `.linear_attn.conv1d`
prefixes, matching source/packed dtype and device, and non-310P hardware.
Other layers and unsupported layouts retain the baseline packing path.
No new CLI flag is required. This does not remove convolution computation.

The GDN prefill backend and chunk-size feature changes a later computation
stage and can be combined with this layout change. CPU regression tests cover
values, metadata, shared storage, idempotency and source-storage replacement;
NPU loading, sleep/wake, LoRA, graph replay and reload integration still need
validation on the matching runtime.
