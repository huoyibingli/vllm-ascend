# SPDX-License-Identifier: Apache-2.0
"""Bind optional NZ decode weights once, preserving MegaMoe's BF16 ND lists."""

import re

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.qwen35_megamoe_config import GIB, QWEN35_NUM_LAYERS, Qwen35MegaMoeDecodeNZConfig
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

_W13_CACHE = "_qwen35_decode_w13_nz"
_W2_CACHE = "_qwen35_decode_w2_nz"
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp(?:\.|$)")


def get_decode_nz_weights(layer):
    w13 = getattr(layer, _W13_CACHE, None)
    w2 = getattr(layer, _W2_CACHE, None)
    if (w13 is None) != (w2 is None):
        raise RuntimeError("Incomplete Qwen3.5 MegaMoe decode NZ cache; restart the worker")
    return None if w13 is None else (w13, w2)


def _validate_expert_lists(layer, index):
    groups = (getattr(layer, "w13_weight_list", None), getattr(layer, "w2_weight_list", None))
    shapes = ((2048, 1024), (512, 2048))
    for weights, shape in zip(groups, shapes):
        if not isinstance(weights, list) or len(weights) != 128:
            raise ValueError(f"Layer {index}: decode NZ cache requires 128 local BF16 ND experts")
        if any(
            weight.dtype != torch.bfloat16
            or tuple(weight.shape) != shape
            or torch_npu.get_npu_format(weight) not in (0, 2)
            for weight in weights
        ):
            raise ValueError(f"Layer {index}: unexpected expert dtype, shape or ND format")
    device = groups[0][0].device
    if any(weight.device != device for weights in groups for weight in weights):
        raise ValueError(f"Layer {index}: expert weights span multiple devices")
    return groups


def bind_qwen35_megamoe_decode_nz(model, config: Qwen35MegaMoeDecodeNZConfig):
    """Allocate persistent-address caches after loading, before graph capture.

    Only selected layers get a second layout. Unselected layers keep the
    existing ND-list decode behavior. No hot-path cast or weight swap occurs.
    """
    if not config.enabled:
        return
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

    candidates = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, AscendRoutedExperts):
            continue
        match = _LAYER_INDEX.search(name)
        if match is None or name.startswith("draft."):
            raise ValueError(f"Unexpected Qwen3.5 routed layer name: {name}")
        index = int(match.group(1))
        if index in candidates:
            raise ValueError(f"Duplicate Qwen3.5 routed layer {index}")
        candidates[index] = layer
    if sorted(candidates) != list(range(QWEN35_NUM_LAYERS)):
        raise ValueError("Decode NZ cache requires all 40 local Qwen3.5 MoE layers")
    indices = tuple(range(QWEN35_NUM_LAYERS)) if config.layers is None else config.layers
    plan = []
    for index in indices:
        layer = candidates[index]
        if get_decode_nz_weights(layer) is not None:
            raise RuntimeError("Decode NZ cache is already bound; restart before rebinding")
        groups = _validate_expert_lists(layer, index)
        size = sum(weight.numel() * weight.element_size() for weights in groups for weight in weights)
        plan.append((index, layer, groups, size))
    required = sum(entry[3] for entry in plan)
    logger.info(
        "Qwen3.5 MegaMoe decode NZ plan: layers=%s extra_bytes=%d (%.2f GiB), budget_bytes=%d",
        indices,
        required,
        required / GIB,
        config.max_cache_bytes,
    )
    if required > config.max_cache_bytes:
        raise ValueError(f"Decode NZ cache needs {required} bytes, exceeding max_cache_bytes={config.max_cache_bytes}")
    device = plan[0][2][0][0].device
    torch.npu.empty_cache()
    free, _total = torch.npu.mem_get_info(device)
    # One layer's temporary ND stack is released before binding the next layer.
    peak = required + max(entry[3] for entry in plan) + config.min_free_bytes
    if free < peak:
        raise RuntimeError(
            f"Insufficient free NPU memory for decode NZ cache: free={free}, "
            f"need cache+temporary+reserve={peak}. Select fewer layers or disable the cache."
        )
    bound = []
    try:
        for index, layer, groups, _size in plan:
            bound.append(layer)
            for name, weights in zip((_W13_CACHE, _W2_CACHE), groups):
                stacked = torch.stack(weights)
                cache = torch_npu.npu_format_cast(stacked, ACL_FORMAT_FRACTAL_NZ)
                del stacked
                if torch_npu.get_npu_format(cache) != ACL_FORMAT_FRACTAL_NZ:
                    raise RuntimeError(f"Layer {index}: NZ conversion did not produce format 29")
                layer.register_buffer(name, cache, persistent=False)
                del cache
        torch.npu.synchronize(device)
    except Exception:
        for layer in bound:
            for name in (_W13_CACHE, _W2_CACHE):
                if hasattr(layer, name):
                    delattr(layer, name)
        raise
    logger.info(
        "Qwen3.5 MegaMoe decode NZ bound: layers=%d extra_bytes=%d; ND MegaMoe weights retained", len(plan), required
    )
