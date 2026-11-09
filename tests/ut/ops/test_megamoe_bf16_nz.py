# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_config as ac
from vllm_ascend.ops.fused_moe import megamoe_bf16_nz as nz


@pytest.fixture
def formats(monkeypatch):
    values = {}
    calls = []

    def cast(value, fmt):
        result = value.clone()
        values[id(result)] = fmt
        calls.append((value, result))
        return result

    monkeypatch.setattr(nz.torch_npu, "get_npu_format", lambda value: values.get(id(value), 2))
    monkeypatch.setattr(nz.torch_npu, "npu_format_cast", cast)
    return values, calls


@pytest.mark.parametrize("experts,hidden,intermediate", [(3, 32, 16), (1, 2048, 512)])
def test_experts_are_complete_stable_nz_matrices(formats, experts, hidden, intermediate):
    values, calls = formats
    w13 = torch.randn(experts, hidden, 2 * intermediate, dtype=torch.bfloat16)
    w2 = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16)
    a, b = nz.make_bf16_nz_expert_weights(w13, w2)
    assert len(calls) == 2 * experts
    for source, output in ((w13, a), (w2, b)):
        for i, weight in enumerate(output):
            torch.testing.assert_close(weight, source[i], rtol=0, atol=0)
            assert weight.ndim == 2 and weight.storage_offset() == 0 and weight.is_contiguous()
            assert values[id(weight)] == 29
            assert weight.data_ptr() != source[i].data_ptr()
    assert len({weight.data_ptr() for weight in a + b}) == 2 * experts


@pytest.mark.parametrize("invalid", ["dtype", "shape", "experts", "unaligned", "nz_source", "empty", "dimensions"])
def test_reject_invalid_sources(formats, invalid):
    values, calls = formats
    a = torch.zeros(2, 32, 32, dtype=torch.bfloat16)
    b = torch.zeros(2, 16, 32, dtype=torch.bfloat16)
    if invalid == "dtype":
        b = b.float()
    elif invalid == "shape":
        b = torch.zeros(2, 32, 32, dtype=torch.bfloat16)
    elif invalid == "experts":
        b = b[:1]
    elif invalid == "unaligned":
        a, b = a[:, :31], b[:, :, :31]
    elif invalid == "nz_source":
        values[id(a)] = 29
    elif invalid == "empty":
        a, b = a[:0], b[:0]
    elif invalid == "dimensions":
        a = a[0]
    with pytest.raises(ValueError, match="3D ND"):
        nz.make_bf16_nz_expert_weights(a, b)
    assert not calls


@pytest.mark.parametrize("broken", ["format", "offset"])
def test_reject_incomplete_cast_result(monkeypatch, broken):
    a = torch.zeros(2, 32, 32, dtype=torch.bfloat16)
    b = torch.zeros(2, 16, 32, dtype=torch.bfloat16)
    monkeypatch.setattr(
        nz.torch_npu, "get_npu_format", lambda v: 2 if v is a or v is b else (2 if broken == "format" else 29)
    )
    monkeypatch.setattr(
        nz.torch_npu, "npu_format_cast", lambda v, _: torch.zeros(v.shape[0] + 1, v.shape[1], dtype=v.dtype)[1:]
    )
    with pytest.raises(RuntimeError, match="complete format-29"):
        nz.make_bf16_nz_expert_weights(a, b)


@pytest.mark.parametrize("enabled,expected", [(False, 1024), (True, 512)])
def test_new_package_buffer_contract(enabled, expected):
    assert nz.mega_moe_buffer_intermediate_size(512, enabled) == expected


def runtime():
    hf = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=40,
        hidden_size=2048,
        num_experts=256,
        moe_intermediate_size=512,
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            tensor_parallel_size=2,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_text_config=hf, enable_sleep_mode=False),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        use_v2_model_runner=False,
        weight_transfer_config=None,
        kv_transfer_config=None,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "fused",
        "package",
        "hardware",
        "tp",
        "ep",
        "dp",
        "pp",
        "pcp",
        "dcp",
        "sp",
        "dtype",
        "model",
        "quant",
        "lora",
        "spec",
        "runner",
        "sleep",
        "transfer",
        "kv",
        "eplb",
        "force_eplb",
        "flashcomm",
    ],
)
def test_runtime_gate(monkeypatch, invalid):
    c = SimpleNamespace(
        mega_moe_bf16_nz=True,
        enable_fused_mc2=1,
        enable_force_eplb=invalid == "force_eplb",
        eplb_config=SimpleNamespace(dynamic_eplb=invalid == "eplb"),
    )
    v = runtime()
    monkeypatch.setattr(ac, "is_mega_moe_supported", lambda: invalid != "package")
    monkeypatch.setattr(
        ac,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(
            supports=lambda _: invalid != "hardware", moe_comm_policy=ac.MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY
        ),
    )
    attributes = {
        "tp": "tensor_parallel_size",
        "dp": "data_parallel_size",
        "pp": "pipeline_parallel_size",
        "pcp": "prefill_context_parallel_size",
        "dcp": "decode_context_parallel_size",
    }
    if invalid in attributes:
        setattr(v.parallel_config, attributes[invalid], 3)
    if invalid == "ep":
        v.parallel_config.enable_expert_parallel = False
    if invalid == "fused":
        c.enable_fused_mc2 = 0
    if invalid == "sp":
        v.parallel_config.use_sequence_parallel_moe = True
    if invalid == "dtype":
        v.model_config.dtype = torch.float16
    if invalid == "model":
        v.model_config.hf_text_config.model_type = "other"
    for tag, attr in (
        ("quant", "quant_config"),
        ("lora", "lora_config"),
        ("spec", "speculative_config"),
        ("transfer", "weight_transfer_config"),
        ("kv", "kv_transfer_config"),
    ):
        if invalid == tag:
            setattr(v, attr, object())
    if invalid == "runner":
        v.use_v2_model_runner = True
    if invalid == "sleep":
        v.model_config.enable_sleep_mode = True
    if invalid is None:
        ac.AscendConfig._validate_megamoe_bf16_nz(c, v, False)
    else:
        with pytest.raises(ValueError, match="mega_moe_bf16_nz requires"):
            ac.AscendConfig._validate_megamoe_bf16_nz(c, v, invalid == "flashcomm")


def test_switch_defaults_off():
    assert not ac.AscendConfig(sparse_kv_offload_config=None).mega_moe_bf16_nz
    ac.AscendConfig._validate_megamoe_bf16_nz(SimpleNamespace(mega_moe_bf16_nz=False), None, True)


@pytest.mark.parametrize("enabled,expected", [(False, 1024), (True, 512)])
def test_buffer_allocator_receives_correct_width(monkeypatch, enabled, expected):
    from vllm_ascend.ops.fused_moe import moe_comm_method as comm_module
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2

    comm = object.__new__(comm_module.FusedMC2CommImpl)
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.global_bs = 0
    comm.token_dispatcher.max_num_tokens_per_rank = 8192
    comm.token_dispatcher.ep_world_size = 2
    comm.moe_config = SimpleNamespace(
        experts_per_token=8, num_experts=256, hidden_dim=2048, intermediate_size_per_partition=512
    )
    monkeypatch.setattr(comm_module, "get_mc2_group", lambda: SimpleNamespace(device_group=object()))
    monkeypatch.setattr(
        comm_module, "get_ascend_config", lambda: SimpleNamespace(mega_moe_bf16_nz=enabled, mega_moe_max_tokens=131072)
    )
    captured = {}

    def allocate(*args, **kwargs):
        captured.update(kwargs)
        return object()

    comm.get_symm_buffer_for_mega_moe = allocate
    comm._init_mega_moe_symm_buffer(is_decode_only_node=False)
    assert captured["intermediate_hidden"] == expected
    assert captured["max_recv_token_num"] == 131072


def test_old_dual_cache_cannot_be_enabled_with_shared_layout():
    config = SimpleNamespace(mega_moe_bf16_nz=True, qwen35_megamoe_decode_nz=SimpleNamespace(enabled=True))
    with pytest.raises(ValueError, match="Disable qwen35_megamoe_decode_nz"):
        ac.AscendConfig._validate_megamoe_bf16_nz(config, None, False)


@pytest.mark.parametrize("enabled", [False, True])
def test_loader_and_both_backends_share_expert_lists(monkeypatch, formats, enabled):
    from vllm_ascend.ops.fused_moe import routed_experts as routed

    base = routed.UnquantizedFusedMoEMethod.__mro__[1]
    monkeypatch.setattr(base, "process_weights_after_loading", lambda *args: None, raising=False)
    monkeypatch.setattr(
        routed, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1, mega_moe_bf16_nz=enabled)
    )
    monkeypatch.setattr(routed, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(routed, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    monkeypatch.setattr(routed, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True))
    monkeypatch.setattr(torch.npu, "empty_cache", lambda: None)
    monkeypatch.setattr(
        routed,
        "replace_parameter",
        lambda layer, name, value: setattr(layer, name, torch.nn.Parameter(value, requires_grad=False)),
    )
    layer = torch.nn.Module()
    original1 = torch.randn(3, 32, 32, dtype=torch.bfloat16)
    original2 = torch.randn(3, 32, 16, dtype=torch.bfloat16)
    layer.w13_weight = torch.nn.Parameter(original1, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(original2, requires_grad=False)
    method = object.__new__(routed.AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = False
    method._maybe_pad_weight = lambda value: value
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
    gmm1, gmm2 = method.get_mlp_weights(layer)
    fused = method.get_fused_mc2_weights(layer)
    assert gmm1 is fused.w1 is layer.w13_weight_list
    assert gmm2 is fused.w2 is layer.w2_weight_list
    for weights, source in ((gmm1, original1), (gmm2, original2)):
        for index, weight in enumerate(weights):
            torch.testing.assert_close(weight, source[index].T, rtol=0, atol=0)
            assert formats[0].get(id(weight), 2) == (29 if enabled else 2)
