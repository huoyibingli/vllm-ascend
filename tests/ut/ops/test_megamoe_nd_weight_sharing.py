# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import routed_experts as routed_module
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod


@pytest.fixture
def loaded_weights(monkeypatch):
    method = object.__new__(AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = False
    method._maybe_pad_weight = lambda tensor: tensor
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(3, 8, 16, dtype=torch.bfloat16), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(3, 16, 4, dtype=torch.bfloat16), requires_grad=False)
    config = SimpleNamespace(enable_fused_mc2=1, weight_nz_mode=1)
    monkeypatch.setattr(routed_module, "get_ascend_config", lambda: config)
    monkeypatch.setattr(routed_module, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed_module, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        routed_module, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False)
    )
    upstream = AscendUnquantizedFusedMoEMethod.__mro__[2]
    monkeypatch.setattr(upstream, "process_weights_after_loading", lambda *_: None, raising=False)
    monkeypatch.setattr(torch.npu, "empty_cache", lambda: None)
    return method, layer, config


def test_gmm_keeps_stacked_parameters_and_megamoe_uses_aliases(loaded_weights, monkeypatch):
    method, layer, _ = loaded_weights
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()
    w13_loader, w2_loader = object(), object()
    layer.w13_weight.weight_loader = w13_loader
    layer.w2_weight.weight_loader = w2_loader
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.w13_weight, original_w13.transpose(1, 2))
    torch.testing.assert_close(layer.w2_weight, original_w2.transpose(1, 2))
    assert not hasattr(layer, "w13_weight_list")
    assert not hasattr(layer, "w2_weight_list")
    assert set(dict(layer.named_parameters())) == {"w13_weight", "w2_weight"}
    assert layer.w13_weight.weight_loader is w13_loader
    assert layer.w2_weight.weight_loader is w2_loader
    gmm_w13, gmm_w2 = method.get_mlp_weights(layer)
    assert gmm_w13 is layer.w13_weight and gmm_w2 is layer.w2_weight
    for stacked, views in (
        (layer.w13_weight, layer.cann_mega_moe_w13_weight_list),
        (layer.w2_weight, layer.cann_mega_moe_w2_weight_list),
    ):
        for index, view in enumerate(views):
            assert view.is_contiguous()
            assert view.untyped_storage().data_ptr() == stacked.untyped_storage().data_ptr()
            assert view.storage_offset() == index * stacked[0].numel()
            torch.testing.assert_close(view, stacked[index])
        # In-place weight updates remain visible to the cached expert views.
        stacked.data.add_(1)
        torch.testing.assert_close(views[-1], stacked[-1])
    monkeypatch.setattr(routed_module, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True))
    weights = method.get_fused_mc2_weights(layer)
    assert weights.w1 is layer.cann_mega_moe_w13_weight_list
    assert weights.w2 is layer.cann_mega_moe_w2_weight_list
    assert method.get_fused_mc2_weights(layer).w1 is weights.w1
    assert weights.w1_scale is None and weights.w2_scale is None


@pytest.mark.parametrize("excluded", ["dynamic_eplb", "nz_mode_2", "float16"])
def test_other_layouts_keep_legacy_expert_lists(loaded_weights, excluded):
    method, layer, config = loaded_weights
    if excluded == "dynamic_eplb":
        method.dynamic_eplb = True
    elif excluded == "nz_mode_2":
        config.weight_nz_mode = 2
    else:
        layer.to(dtype=torch.float16)
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")
    assert not hasattr(layer, "cann_mega_moe_w13_weight_list")
    assert isinstance(method.get_mlp_weights(layer)[0], list)
    assert all(view.storage_offset() == 0 for view in layer.w13_weight_list)


def test_a5_preserves_normal_gmm_layout(loaded_weights, monkeypatch):
    method, layer, _ = loaded_weights
    monkeypatch.setattr(routed_module, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    monkeypatch.setattr(routed_module, "maybe_trans_nz", lambda tensor: tensor)
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "cann_mega_moe_w13_weight_list")
    assert method.get_mlp_weights(layer)[0] is layer.w13_weight


def test_decode_gmm_receives_one_stacked_tensor_per_projection(loaded_weights, monkeypatch):
    method, layer, _ = loaded_weights
    method.process_weights_after_loading(layer)
    method.moe = SimpleNamespace(has_bias=False)
    calls = []

    def grouped_matmul(**kwargs):
        calls.append(kwargs)
        return [torch.zeros(3, kwargs["weight"][0].shape[-1], dtype=torch.bfloat16)]

    monkeypatch.setattr(routed_module.torch_npu, "npu_grouped_matmul", grouped_matmul, raising=False)
    inputs = SimpleNamespace(
        layer=layer,
        hidden_states=torch.ones(3, 16, dtype=torch.bfloat16),
        need_trans=False,
        group_list=torch.ones(3, dtype=torch.int64),
        group_list_type=1,
        lora_context=None,
    )
    method.apply_gmm1(inputs)
    method.apply_gmm2(inputs, torch.ones(3, 4, dtype=torch.bfloat16), None)
    assert len(calls) == 2
    assert len(calls[0]["weight"]) == len(calls[1]["weight"]) == 1
    assert calls[0]["weight"][0] is layer.w13_weight
    assert calls[1]["weight"][0] is layer.w2_weight
    assert all(call["split_item"] == 2 and call["group_list_type"] == 1 for call in calls)
