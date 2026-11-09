# SPDX-License-Identifier: Apache-2.0
"""One persistent BF16 NZ expert layout for MegaMoe and ordinary GMM."""

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def make_bf16_nz_expert_weights(w13: torch.Tensor, w2: torch.Tensor):
    """Convert complete ND expert matrices once, before graph capture.

    Slicing a stacked NZ tensor does not produce the standalone storage
    descriptors required by the A2 MegaMoe Torch adapter.
    """
    if (
        w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or w13.ndim != 3
        or w2.ndim != 3
        or w13.shape[0] == 0
        or w13.shape[0] != w2.shape[0]
        or w13.shape[1] != w2.shape[2]
        or w13.shape[2] != 2 * w2.shape[1]
        or w13.device != w2.device
        or any(dim % 16 for dim in (*w13.shape[1:], *w2.shape[1:]))
        or any(torch_npu.get_npu_format(weight) not in (0, 2) for weight in (w13, w2))
    ):
        raise ValueError("MegaMoe BF16 NZ requires matching aligned 3D ND W13/W2 source weights")

    def convert(weight):
        experts = []
        for source in weight.unbind(0):
            # Clone the ND slice before packing; no nonzero storage offset or
            # stacked-NZ alias may escape into the fused operator.
            expert = torch_npu.npu_format_cast(source.clone(), ACL_FORMAT_FRACTAL_NZ)
            if (
                expert.dtype != torch.bfloat16
                or expert.shape != source.shape
                or expert.device != source.device
                or expert.storage_offset() != 0
                or not expert.is_contiguous()
                or torch_npu.get_npu_format(expert) != ACL_FORMAT_FRACTAL_NZ
            ):
                raise RuntimeError("MegaMoe BF16 NZ conversion did not produce a complete format-29 matrix")
            experts.append(expert)
        return experts

    return convert(w13), convert(w2)


def mega_moe_buffer_intermediate_size(intermediate: int, bf16_nz: bool) -> int:
    """New BF16-NZ packages take I; retain the legacy package's 2I contract."""
    return intermediate if bf16_nz else 2 * intermediate
