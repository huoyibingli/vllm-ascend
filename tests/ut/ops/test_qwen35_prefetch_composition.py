# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import qwen3_5 as qwen_model_module

from vllm_ascend import utils as utils_module
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops import weight_prefetch_qwen35 as prefetch_module
from vllm_ascend.ops.fused_moe import fused_moe as fused_module
from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.weight_prefetch_config import Qwen35WeightPrefetchConfig


@pytest.mark.parametrize("format_code", [0, 2, 29, 3])
@pytest.mark.parametrize("allow_nz", [False, True])
def test_dense_layout_gate(monkeypatch, format_code, allow_nz):
    class Weight:
        device = SimpleNamespace(type="npu")
        dtype = "bf16"
        shape = (2048, 2048)

        def is_contiguous(self):
            return True

        def storage_offset(self):
            return 0

    monkeypatch.setattr(prefetch_module, "torch", SimpleNamespace(Tensor=Weight, bfloat16="bf16"))
    monkeypatch.setattr(prefetch_module.torch_npu, "get_npu_format", lambda _: format_code, raising=False)
    assert prefetch_module.supported_weight(Weight(), (2048, 2048), allow_nz=allow_nz) == (
        format_code in (0, 2) or (allow_nz and format_code == 29)
    )


def test_ep_nz_config_excludes_routed_kernel():
    config = Qwen35WeightPrefetchConfig.parse({"allow_ep_nz_dense": True, "targets": ["next_input", "gdn_out"]})
    assert config.allow_ep_nz_dense
    with pytest.raises(ValueError, match="only next_input and gdn_out"):
        Qwen35WeightPrefetchConfig.parse({"allow_ep_nz_dense": True, "targets": ["routed_w2"]})


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.3.linear_attn",
        "language_model.model.layers.3.linear_attn",
        "model.language_model.model.layers.3.linear_attn",
    ],
)
def test_main_model_prefixes(prefix):
    assert prefetch_module.gdn_layer_index(prefix) == 3
    assert prefetch_module.gdn_layer_index("draft." + prefix) is None


def test_prefetch_join_and_shared_milestone_order(monkeypatch):
    order = []

    def begin(_weight, _groups, _dependency, position, _group_type):
        order.append(position)
        return position == "after_gmm1"

    controller = SimpleNamespace(begin_routed=begin, join=lambda: order.append("join"))
    monkeypatch.setattr(prefetch_module, "current_controller", lambda: controller)
    monkeypatch.setattr(
        torch.npu, "current_stream", lambda: SimpleNamespace(record_event=lambda: order.append("event"))
    )
    monkeypatch.setattr(
        moe_mlp, "_unified_apply_activation", lambda _input, x, _method: (order.append("activation"), x)[1]
    )
    value = torch.ones(2, 4)
    method = SimpleNamespace(
        supports_fused_activation=lambda _: False,
        apply_gmm1=lambda _: (order.append("gmm1"), value)[1],
        apply_act_quant=lambda _input, x: (x, None),
        apply_gmm2=lambda _input, x, _scale: (order.append("gmm2"), x)[1],
    )
    compute_input = SimpleNamespace(
        quant=SimpleNamespace(is_quant=False),
        lora_context=None,
        weights=SimpleNamespace(w2=object()),
        group_list=object(),
        group_list_type=1,
        hidden_states=value,
        activation="silu",
    )
    stages = {}
    result, _event = moe_mlp.apply_moe_mlp(compute_input, method, stage_events=stages)
    assert result is value
    assert "before_swiglu" in stages
    assert order == ["after_dispatch", "gmm1", "event", "after_gmm1", "activation", "join", "event", "gmm2"]


@pytest.mark.parametrize("usable_weights", [True, False])
@pytest.mark.parametrize("expert_lists", [True, False])
def test_tp2_ep2_nz_bind_with_expert_storage(monkeypatch, usable_weights, expert_lists):
    class Layer:
        pass

    class Owner:
        pass

    class Weight:
        device = SimpleNamespace(type="npu")
        dtype = "bf16" if usable_weights else "fp16"

        def __init__(self, shape):
            self.shape = shape

        def is_contiguous(self):
            return True

        def storage_offset(self):
            return 0

        def numel(self):
            return self.shape[0] * self.shape[1]

        def element_size(self):
            return 2

        def data_ptr(self):
            return id(self)

    monkeypatch.setattr(qwen_model_module, "Qwen3_5DecoderLayer", Layer)
    monkeypatch.setattr(fused_module, "AscendMoERunner", Owner)
    monkeypatch.setattr(utils_module, "get_ascend_device_type", lambda: AscendDeviceType.A2)
    monkeypatch.setattr(prefetch_module, "torch", SimpleNamespace(Tensor=Weight, bfloat16="bf16"))
    monkeypatch.setattr(prefetch_module.torch_npu, "get_npu_format", lambda _: 29, raising=False)
    controller = SimpleNamespace(
        next_inputs={}, pending_next_input=set(), gdn_out_weights={}, next_gdn_out={}, routed_weights={}
    )
    monkeypatch.setattr(prefetch_module, "Qwen35WeightPrefetch", lambda *_args: controller)
    layers = []
    for index in range(40):
        layer = Layer()
        layer.layer_idx = index
        layer.layer_type = "full_attention" if index % 4 == 3 else "linear_attention"
        layer.linear_attn = SimpleNamespace(
            prefix=f"model.language_model.model.layers.{index}.linear_attn",
            out_proj=SimpleNamespace(weight=Weight((2048, 2048))),
            in_proj_qkvz=SimpleNamespace(weight=Weight((6144, 2048))),
        )
        layer.self_attn = SimpleNamespace(qkv_proj=SimpleNamespace(weight=Weight((4608, 2048))))
        owner = Owner()
        owner.layer_name = f"model.language_model.model.layers.{index}.mlp"
        # Dense prefetch must work with either native NZ or expert-list storage.
        owner.routed_experts = (
            SimpleNamespace(w2_weight_list=[]) if expert_lists else SimpleNamespace(w2_weight=Weight((128, 2048, 512)))
        )
        layer.mlp = SimpleNamespace(modules=lambda owner=owner: iter([owner]))
        layers.append(layer)
    model = SimpleNamespace(modules=lambda: iter(layers), parameters=lambda: iter([Weight((2, 2))]))
    runtime = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            use_sequence_parallel_moe=False,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text", num_hidden_layers=40)
        ),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
    )
    config = Qwen35WeightPrefetchConfig.parse(
        {"enabled": True, "allow_ep_nz_dense": True, "targets": ["next_input", "gdn_out"]}
    )
    if not usable_weights:
        with pytest.raises(RuntimeError, match="no bound storage"):
            prefetch_module.bind_qwen35_weight_prefetch(model, runtime, config)
        return
    prefetch_module.bind_qwen35_weight_prefetch(model, runtime, config)
    assert len(controller.next_inputs) == 39
    assert len(controller.gdn_out_weights) == 30
    assert len(controller.next_gdn_out) == 29
    assert not controller.routed_weights
    assert model._qwen35_weight_prefetch is controller
