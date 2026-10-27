# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import gdn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_conv_packed_alias_and_reload_preserve_captured_storage(monkeypatch, dtype):
    monkeypatch.setattr(gdn, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.linear_attn.conv1d"
    expected = torch.arange(128).reshape(32, 1, 4).to(dtype)
    source = torch.nn.Parameter(expected.clone(), requires_grad=False)
    source.weight_loader = object()
    source.input_dim, source.output_dim = 1, 0
    layer.register_parameter("weight", source)
    packed = torch.nn.Parameter(torch.empty(4, 32, dtype=dtype), requires_grad=False)
    layer.register_parameter("ascend_conv1d_weight", packed)
    loader = source.weight_loader
    gdn._pack_conv_weights(layer)
    pointer = packed.data_ptr()
    assert source is layer.weight and packed is layer.ascend_conv1d_weight
    assert source.weight_loader is loader and (source.input_dim, source.output_dim) == (1, 0)
    assert source.data_ptr() == pointer
    assert packed.is_contiguous()
    assert source.view(32, 4).transpose(0, 1).is_contiguous()
    torch.testing.assert_close(source, expected)
    torch.testing.assert_close(packed, expected[:, 0, :].T.contiguous())
    gdn._pack_conv_weights(layer)
    assert packed.data_ptr() == pointer
    # Simulate a loader replacing the original weight storage on reload.
    source.data = expected + 1
    gdn._pack_conv_weights(layer)
    assert packed.data_ptr() == pointer == source.data_ptr()
    torch.testing.assert_close(source, expected + 1)
    torch.testing.assert_close(packed, (expected[:, 0, :] + 1).T.contiguous())
