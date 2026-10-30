# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import megamoe_decode_nz as cache_module
from vllm_ascend.ops.fused_moe import routed_experts as routed_module
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, AscendUnquantizedFusedMoEMethod
from vllm_ascend.qwen35_megamoe_config import GIB, Qwen35MegaMoeDecodeNZConfig


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "true"},
        {"enabled": True},
        {"layers": []},
        {"layers": [0, 0]},
        {"layers": [True]},
        {"layers": [40]},
        {"max_cache_bytes": -1},
        {"min_free_bytes": False},
        {"unknown": 1},
        {"layers": "all"},
    ],
)
def test_invalid_cache_config(options):
    with pytest.raises(ValueError):
        Qwen35MegaMoeDecodeNZConfig.parse(options)


def test_cache_default_and_explicit_budget():
    assert not Qwen35MegaMoeDecodeNZConfig.parse({}).enabled
    config = Qwen35MegaMoeDecodeNZConfig.parse({"enabled": True, "layers": [0, 3], "max_cache_bytes": 2 * GIB})
    assert config.layers == (0, 3)


@pytest.fixture
def model_and_runtime(monkeypatch):
    # Meta tensors retain the real model shapes without allocating 30 GiB.
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for _index in range(40):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        experts = object.__new__(AscendRoutedExperts)
        torch.nn.Module.__init__(experts)
        experts.w13_weight_list = [torch.empty(2048, 1024, dtype=torch.bfloat16, device="meta")] * 128
        experts.w2_weight_list = [torch.empty(512, 2048, dtype=torch.bfloat16, device="meta")] * 128
        layer.mlp.experts = experts
        model.layers.append(layer)
    nz_ids = set()
    calls = []

    def cast(weight, fmt):
        calls.append(fmt)
        converted = weight.clone()
        nz_ids.add(id(converted))
        return converted

    monkeypatch.setattr(cache_module.torch_npu, "npu_format_cast", cast, raising=False)
    monkeypatch.setattr(cache_module.torch_npu, "get_npu_format", lambda w: 29 if id(w) in nz_ids else 2, raising=False)
    monkeypatch.setattr(torch.npu, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.npu, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda _device: (80 * GIB, 80 * GIB), raising=False)
    return model, calls


def settings(**kwargs):
    return Qwen35MegaMoeDecodeNZConfig.parse({"enabled": True, "layers": [0, 3], "max_cache_bytes": 2 * GIB, **kwargs})


def test_selected_layers_keep_nd_lists_and_bind_static_nz(model_and_runtime, monkeypatch):
    model, calls = model_and_runtime
    owner = model.layers[0].mlp.experts
    original = owner.w13_weight_list
    cache_module.bind_qwen35_megamoe_decode_nz(model, settings())
    assert calls == [29] * 4
    assert owner.w13_weight_list is original
    assert cache_module.get_decode_nz_weights(model.layers[1].mlp.experts) is None
    w13, w2 = cache_module.get_decode_nz_weights(owner)
    assert w13.shape == (128, 2048, 1024) and w2.shape == (128, 512, 2048)
    assert cache_module.torch_npu.get_npu_format(w13) == 29
    assert not model.state_dict()  # The second layout is never checkpointed.
    method = object.__new__(AscendUnquantizedFusedMoEMethod)
    monkeypatch.setattr(routed_module, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True))
    assert method.get_mlp_weights(owner)[0] is w13
    assert method.get_fused_mc2_weights(owner).w1 is original
    assert method.get_mlp_weights(model.layers[1].mlp.experts)[0] is model.layers[1].mlp.experts.w13_weight_list
    assert calls == [29] * 4  # Weight selection never performs a format cast.
    with pytest.raises(RuntimeError, match="already bound"):
        cache_module.bind_qwen35_megamoe_decode_nz(model, settings())


@pytest.mark.parametrize("failure", ["budget", "memory", "missing_layer", "shape"])
def test_preflight_failure_allocates_no_cache(model_and_runtime, monkeypatch, failure):
    model, calls = model_and_runtime
    config = settings()
    if failure == "budget":
        config = settings(max_cache_bytes=GIB)
    elif failure == "memory":
        monkeypatch.setattr(torch.npu, "mem_get_info", lambda _device: (GIB, 80 * GIB))
    elif failure == "missing_layer":
        del model.layers[-1]
    else:
        model.layers[3].mlp.experts.w2_weight_list = []
    with pytest.raises((ValueError, RuntimeError)):
        cache_module.bind_qwen35_megamoe_decode_nz(model, config)
    assert not calls
    assert cache_module.get_decode_nz_weights(model.layers[0].mlp.experts) is None


def test_conversion_failure_rolls_back_all_selected_layers(model_and_runtime, monkeypatch):
    model, calls = model_and_runtime
    cast = cache_module.torch_npu.npu_format_cast

    def fail_second_layer(weight, fmt):
        if len(calls) == 2:
            raise RuntimeError("conversion failed")
        return cast(weight, fmt)

    monkeypatch.setattr(cache_module.torch_npu, "npu_format_cast", fail_second_layer)
    with pytest.raises(RuntimeError, match="conversion failed"):
        cache_module.bind_qwen35_megamoe_decode_nz(model, settings())
    assert all(cache_module.get_decode_nz_weights(layer.mlp.experts) is None for layer in model.layers)


def test_full_model_cache_requires_thirty_gib(model_and_runtime):
    model, calls = model_and_runtime
    cache_module.bind_qwen35_megamoe_decode_nz(model, settings(layers=None, max_cache_bytes=30 * GIB))
    assert calls == [29] * 80
    assert all(cache_module.get_decode_nz_weights(layer.mlp.experts) is not None for layer in model.layers)


def test_wrong_conversion_format_is_rejected(model_and_runtime, monkeypatch):
    model, _calls = model_and_runtime
    monkeypatch.setattr(cache_module.torch_npu, "npu_format_cast", lambda w, _fmt: w.clone())
    with pytest.raises(RuntimeError, match="did not produce"):
        cache_module.bind_qwen35_megamoe_decode_nz(model, settings())
    assert cache_module.get_decode_nz_weights(model.layers[0].mlp.experts) is None


def test_incomplete_cache_is_rejected():
    with pytest.raises(RuntimeError, match="Incomplete"):
        cache_module.get_decode_nz_weights(SimpleNamespace(_qwen35_decode_w13_nz=object()))


def test_disabled_cache_does_not_inspect_model():
    cache_module.bind_qwen35_megamoe_decode_nz(object(), Qwen35MegaMoeDecodeNZConfig())
