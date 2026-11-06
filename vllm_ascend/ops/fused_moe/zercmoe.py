# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Weight preparation and wheel ABI for the A2 ZercMoE integration."""

from functools import lru_cache
from importlib import import_module

import torch
from vllm.logger import logger

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights

_ZN_BLOCK = 16
_W8A8_DYNAMIC_QUANT_MODE = 2


def load_zercmoe():
    return import_module("zercmoe")


@lru_cache(maxsize=2)
def zercmoe_available(w8a8: bool) -> bool:
    try:
        wheel = load_zercmoe()
        operator = "zerc_moe_int8" if w8a8 else "zerc_moe"
        if not callable(getattr(wheel, "get_symm_buffer", None)) or not hasattr(torch.ops.npu, operator):
            raise RuntimeError(f"get_symm_buffer and torch.ops.npu.{operator} are required")
    except Exception as exc:
        logger.warning_once("ZercMoE wheel is unavailable for this route: %s", exc)
        return False
    return True


def validate_zercmoe_layer(layer) -> None:
    for config in (layer, layer.moe_config):
        activation = getattr(config, "activation", "silu")
        if (
            getattr(activation, "value", activation) != "silu"
            or getattr(config, "swigluoai_uninterleave", False)
            or getattr(config, "has_bias", False)
            or getattr(config, "apply_router_weight_on_input", False)
            or getattr(config, "swiglu_limit", None) not in (None, 0.0)
            or getattr(config, "swiglu_alpha", None) not in (None, 1.0)
            or getattr(config, "swiglu_beta", None) not in (None, 0.0)
        ):
            raise ValueError("ZercMoE requires bias-free standard SiLU with output router weighting")
    if getattr(layer, "w13_bias", None) is not None or getattr(layer, "w2_bias", None) is not None:
        raise ValueError("ZercMoE requires bias-free expert weights")


def validate_w8a8_zercmoe_weights(layer) -> None:
    """Validate symmetric, per-output-channel W8A8 at weight-load time."""
    validate_zercmoe_layer(layer)
    moe = layer.moe_config
    experts, hidden, intermediate = moe.num_local_experts, moe.hidden_dim, moe.intermediate_size_per_partition
    for weight, scale, offset, rows, cols in (
        (layer.w13_weight, layer.w13_weight_scale, layer.w13_weight_offset, hidden, 2 * intermediate),
        (layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_offset, intermediate, hidden),
    ):
        if weight.dtype != torch.int8 or weight.shape != (experts, rows, cols):
            raise ValueError("ZercMoE requires native-width INT8 expert weights")
        if any(param.shape[0] != experts or param.numel() != experts * cols for param in (scale, offset)):
            raise ValueError("ZercMoE W8A8 requires per-expert, per-output-channel quantization")
        if torch.count_nonzero(offset).item():
            raise ValueError("ZercMoE W8A8 requires symmetric INT8 weights (zero offsets)")
        if not torch.isfinite(scale).all().item() or not (scale > 0).all().item():
            raise ValueError("ZercMoE W8A8 scales must be finite and positive")


def build_w8a8_zercmoe_weights(layer, encode_scale) -> MoEWeights:
    """Pack stacked ND int8 weights into per-expert zN lists (kernel zN layout).

    The zerc_moe_int8 kernel expects the same layout gen_data.py's
    convert_nd_to_nz_int8 produces: per expert (cols/32, rows, 32) -- i.e.
    32-column inner blocks over 16-aligned rows. This is NOT the 16-wide bf16
    zN of ``pack_bf16_zercmoe_weights`` and NOT vLLM ACL FRACTAL_NZ
    (verified against Task-1 gen_data test bins). Runs AFTER the grouped-
    matmul transpose in process_weights_after_loading but BEFORE
    npu_format_cast, so ``w13_weight`` is the POST-transpose [E, hidden,
    2*intermediate] and ``w2_weight`` the [E, intermediate, hidden] ND.
    """
    experts = layer.moe_config.num_local_experts
    w1, w2 = layer.w13_weight.detach(), layer.w2_weight.detach()

    def to_zn(w):
        # [E, a, b] ND -> per-expert (b/32, a, 32) flattened int8 zN blocks
        # (gen_data convert_nd_to_nz_int8 layout), cat into one contiguous
        # tensor, then unbind back to per-expert 1D slices. The wheel's
        # FlattenInt8WeightList detects the contiguous storage and passes it
        # to the kernel zero-copy.
        e, a, b = w.shape
        if b % 32 != 0 or a % 16 != 0:
            raise ValueError(f"to_zn: shape ({a},{b}) not alignable to int8 zN (rows%16, cols%32)")
        packed = w.view(e, a, b // 32, 32).permute(0, 2, 1, 3).contiguous().view(e, -1)
        return list(packed.unbind(0))

    return MoEWeights(
        w1=to_zn(w1),
        w2=to_zn(w2),
        w1_scale=list(encode_scale(layer.w13_weight_scale.detach()).view(experts, -1).unbind(0)),
        w2_scale=list(encode_scale(layer.w2_weight_scale.detach()).view(experts, -1).unbind(0)),
    )


def pack_bf16_zercmoe_weights(layer, scratch: MoEWeights | None) -> MoEWeights:
    """Pack ND weights into zN, reusing the communication instance's scratch."""
    w1, w2 = layer.w13_weight.detach(), layer.w2_weight.detach()
    if scratch is None:
        scratch = MoEWeights(w1=w1.new_empty(w1.numel()), w2=w2.new_empty(w2.numel()))
    for weight, packed in ((w1, scratch.w1), (w2, scratch.w2)):
        experts, rows, cols = weight.shape
        blocks = cols // _ZN_BLOCK
        packed.view(experts, blocks, rows, _ZN_BLOCK).copy_(
            weight.view(experts, rows, blocks, _ZN_BLOCK).permute(0, 2, 1, 3)
        )
    return scratch


def run_zercmoe(inputs, weights: MoEWeights, buffer, *, w8a8: bool):
    """Keep the provisional INT8 ABI isolated from MoE scheduling and loading."""
    x = inputs.hidden_states
    expert_idx = inputs.topk_ids.to(torch.int32)
    probs = inputs.topk_weights.to(torch.float32)
    if w8a8:
        # Proposed ABI follows MegaMoE dynamic W8A8: BF16 x, INT8 NZ weights,
        # per-expert packed scales, and activation quantization inside the op.
        mask = inputs.routing.mc2_mask
        return torch.ops.npu.zerc_moe_int8(
            x=x,
            expert_idx=expert_idx,
            probs=probs,
            weight1=weights.w1,
            weight2=weights.w2,
            scale1=weights.w1_scale,
            scale2=weights.w2_scale,
            x_active_mask=mask.to(torch.int8) if mask is not None else None,
            moe_expert_num=buffer.moe_expert_num,
            ep_world_size=buffer.ep_world_size,
            ccl_buffer_size=buffer.ccl_buffer_size,
            quant_mode=_W8A8_DYNAMIC_QUANT_MODE,
            return_expert_tokens=0,
        )
    return torch.ops.npu.zerc_moe(
        x,
        expert_idx,
        probs,
        [weights.w1],
        [weights.w2],
        buffer.moe_expert_num,
        buffer.ep_world_size,
        buffer.ccl_buffer_size,
        0,  # topk_weights_type: fp32
        0,  # return_expert_tokens
    )
