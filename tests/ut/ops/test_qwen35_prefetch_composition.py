# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import weight_prefetch_qwen35 as prefetch_module
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
