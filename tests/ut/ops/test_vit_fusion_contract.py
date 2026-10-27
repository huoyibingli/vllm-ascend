# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops import mm_encoder_attention as mm


def test_packed_vision_lengths_reuse_and_invalid_boundaries():
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    assert mm._get_or_convert_cu_seqlens_host_lengths(cu, 8) == (3, 8)
    assert mm.peek_cu_seqlens_host_lengths(cu) == (3, 8)
    for values in ([0, 3, 7], [0, 5, 4, 8]):
        assert mm._get_or_convert_cu_seqlens_host_lengths(torch.tensor(values), 8) is None


def test_capture_keeps_existing_encoder_graph_path():
    layer = object.__new__(mm.AscendMMEncoderAttention)
    with (
        patch.object(mm, "HAS_TRITON", True),
        patch.object(mm, "get_encoder_forward_context", return_value=SimpleNamespace(capturing=True)),
    ):
        assert layer._check_fused_qkv_rope_pad_fia(None, None, None, None) == "capturing"


def test_unsupported_head_configuration_falls_back():
    layer = object.__new__(mm.AscendMMEncoderAttention)
    layer.head_size, layer.num_heads, layer.num_kv_heads = 128, 8, 8
    with (
        patch.object(mm, "HAS_TRITON", True),
        patch.object(mm, "get_encoder_forward_context", return_value=SimpleNamespace(capturing=False)),
    ):
        assert layer._check_fused_qkv_rope_pad_fia(None, None, None, None) == "head_config"
