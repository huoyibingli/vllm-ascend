# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.vision_qkv_rope_pad import vision_qkv_rope_pad


@pytest.mark.parametrize("tokens", [1, 4, 41, 257, 8160])
def test_vision_qkv_rope_pad(tokens):
    torch.manual_seed(0)
    qkv = torch.randn(tokens, 3 * 8 * 72, device="npu", dtype=torch.bfloat16)
    angles = torch.randn(tokens, 36, device="npu")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    actual = vision_qkv_rope_pad(qkv, cos, sin)
    planes = qkv.view(tokens, 3, 8, 72).unbind(1)
    expected = []
    for plane in planes[:2]:
        first, second = plane.float().chunk(2, dim=-1)
        left = first * cos[:, None].float() - second * sin[:, None].float()
        right = second * cos[:, None].float() + first * sin[:, None].float()
        expected.append(F.pad(torch.cat((left, right), dim=-1).bfloat16(), (0, 56)))
    expected.append(F.pad(planes[2], (0, 56)))
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0.02, rtol=0.02)
        assert torch.count_nonzero(result[..., 72:]) == 0
