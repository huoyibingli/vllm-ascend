# SPDX-License-Identifier: Apache-2.0
"""One persistent batched BF16 NZ layout for MegaMoe and ordinary GMM."""

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

NZ_BLOCK_SIZE = 16
MAX_LOCAL_EXPERTS = 128
ND_FORMATS = (0, 2)


def make_bf16_nz_expert_weights(w13: torch.Tensor, w2: torch.Tensor):
    """Return single-item lists containing complete [E,K,N] NZ tensors.

    Pack ND weights once before graph capture; reuse already-NZ parameters.
    Requires the A2 operator/adapter with batched BF16 NZ support.
    """
    if (
        w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or w13.ndim != 3
        or w2.ndim != 3
        or not 1 <= w13.shape[0] <= MAX_LOCAL_EXPERTS
        or w13.shape[0] != w2.shape[0]
        or w13.shape[1] != w2.shape[2]
        or w13.shape[2] != 2 * w2.shape[1]
        or w13.device != w2.device
        or any(dim <= 0 or dim % NZ_BLOCK_SIZE for dim in (*w13.shape[1:], *w2.shape[1:]))
        or any(torch_npu.get_npu_format(weight) not in (*ND_FORMATS, ACL_FORMAT_FRACTAL_NZ) for weight in (w13, w2))
    ):
        raise ValueError("MegaMoe BF16 NZ requires matching aligned 3D ND/NZ W13/W2 source weights (1-128 experts)")

    def convert(weight):
        packed = (
            weight
            if torch_npu.get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ
            else torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
        )
        if (
            packed.dtype != torch.bfloat16
            or packed.shape != weight.shape
            or packed.device != weight.device
            or packed.storage_offset() != 0
            or not packed.is_contiguous()
            or torch_npu.get_npu_format(packed) != ACL_FORMAT_FRACTAL_NZ
        ):
            raise RuntimeError("MegaMoe BF16 NZ requires a complete format-29 batched tensor")
        return [packed]

    return convert(w13), convert(w2)


def mega_moe_buffer_intermediate_size(intermediate: int, bf16_nz: bool) -> int:
    """New BF16-NZ packages take I; retain the legacy package's 2I contract."""
    return intermediate if bf16_nz else 2 * intermediate
