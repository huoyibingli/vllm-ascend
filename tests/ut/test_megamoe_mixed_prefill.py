# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device.hardware_profile import MoECommPolicy
from vllm_ascend.ops.fused_moe import prepare_finalize as pf


@pytest.mark.parametrize(
    "tokens,prefills,decodes,mixed,graph,profile,capacity,expected_fused",
    [
        (144, 1, 16, False, False, False, 16384, False),
        (144, 1, 16, True, False, False, 16384, True),
        (1055, 1, 31, True, False, False, 16384, True),
        (15361, 15, 1, True, False, False, 16384, True),
        (15376, 15, 16, True, False, False, 16384, True),
        (16384, 15, 32, True, False, False, 16384, True),
        (16385, 15, 32, True, False, False, 16384, False),
        (16385, 1, 0, False, False, False, 16384, False),
        (64, 1, 1, True, False, False, 16384, False),
        (65, 1, 1, True, False, False, 16384, True),
        (32, 0, 32, True, False, False, 16384, False),
        (128, 0, 128, True, False, False, 16384, False),
        (144, 1, 16, True, True, False, 16384, False),
        (1024, 1, 0, True, True, False, 16384, False),
        (32, 0, 32, True, True, False, 16384, False),
        (64, 0, None, True, False, True, 16384, True),
        (64, 0, None, True, True, True, 16384, False),
        (0, 0, 0, True, False, True, 16384, False),
        (144, 1, 16, True, False, False, None, False),
        (None, 0, None, True, False, True, 16384, True),
        (None, 0, None, True, False, True, 0, False),
    ],
)
@pytest.mark.parametrize("as_dict", [False, True])
def test_mixed_prefill_selection(
    monkeypatch, tokens, prefills, decodes, mixed, graph, profile, capacity, expected_fused, as_dict
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, is_quantized=False, get_num_experts=lambda: 256),
        parallel_config=SimpleNamespace(world_size_across_dp=2, pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None,
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(mega_moe_allow_mixed_prefill=mixed, mega_moe_prefill_token_threshold=64),
    )
    metadata = SimpleNamespace(num_prefills=prefills, num_decode_tokens=decodes)
    if as_dict:
        metadata = {"layer0": metadata}
    result = afc._select_capacity_and_expert_density_moe_comm_method(
        tokens, config, capacity, attn_metadata=metadata, in_profile_run=profile, is_graph_mode=graph
    )
    assert result == (afc.MoECommType.FUSED_MC2 if expected_fused else afc.MoECommType.ALLGATHER)


@pytest.mark.parametrize("unsupported", ["fp16", "quant", "dp", "pp", "pcp", "dcp", "sp", "eplb", "draft", "spec"])
def test_mixed_prefill_keeps_unsupported_modes_on_original_backend(monkeypatch, unsupported):
    parallel = SimpleNamespace(
        world_size_across_dp=2,
        pipeline_parallel_size=2 if unsupported == "pp" else 1,
        data_parallel_size=2 if unsupported == "dp" else 1,
        prefill_context_parallel_size=2 if unsupported == "pcp" else 1,
        decode_context_parallel_size=2 if unsupported == "dcp" else 1,
        use_sequence_parallel_moe=unsupported == "sp",
        enable_eplb=unsupported == "eplb",
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float16 if unsupported == "fp16" else torch.bfloat16,
            is_quantized=unsupported == "quant",
            get_num_experts=lambda: 256,
        ),
        parallel_config=parallel,
        speculative_config=object() if unsupported == "spec" else None,
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(mega_moe_allow_mixed_prefill=True, mega_moe_prefill_token_threshold=64),
    )
    result = afc._select_capacity_and_expert_density_moe_comm_method(
        1055,
        config,
        16384,
        is_draft_model=unsupported == "draft",
        attn_metadata={"layer0": SimpleNamespace(num_prefills=1, num_decode_tokens=31)},
    )
    assert result == afc.MoECommType.ALLGATHER


def test_mixed_prefill_flag_is_opt_in():
    assert not AscendConfig(sparse_kv_offload_config=None).mega_moe_allow_mixed_prefill
    assert AscendConfig(sparse_kv_offload_config=None, mega_moe_allow_mixed_prefill=True).mega_moe_allow_mixed_prefill


@pytest.mark.parametrize("graph", [False, True])
def test_public_selector_propagates_graph_mode(monkeypatch, graph):
    policy = MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 16384)
    monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: SimpleNamespace(moe_comm_policy=policy))
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=2))
    captured = []

    def selector(*args):
        captured.append(args)
        return afc.MoECommType.ALLGATHER

    monkeypatch.setitem(afc._MOE_COMM_SELECTORS, policy, selector)
    config = SimpleNamespace(parallel_config=SimpleNamespace(enable_expert_parallel=True))
    metadata = {"layer0": SimpleNamespace(num_prefills=1, num_decode_tokens=31)}
    assert (
        afc.select_moe_comm_method(1055, config, attn_metadata=metadata, is_graph_mode=graph)
        == afc.MoECommType.ALLGATHER
    )
    assert captured[0][-1] is graph
    assert captured[0][-3] is metadata


def test_missing_metadata_does_not_enable_mixed_prefill(monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, get_num_experts=lambda: 256),
        parallel_config=SimpleNamespace(world_size_across_dp=2, pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None,
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(mega_moe_allow_mixed_prefill=True, mega_moe_prefill_token_threshold=64),
    )
    assert afc._select_capacity_and_expert_density_moe_comm_method(1024, config, 16384) == afc.MoECommType.ALLGATHER


@pytest.mark.parametrize("tokens", [65, 1055, 15361, 15376, 16384])
def test_mixed_prefill_tp2_padding_mask_and_finalize(monkeypatch, tokens):
    padded_tokens = ((tokens + 1) // 2) * 2
    monkeypatch.setattr(
        pf,
        "_EXTRA_CTX",
        SimpleNamespace(mc2_mask=torch.arange(padded_tokens) < tokens, padded_num_tokens=padded_tokens),
    )
    # Exercise the production prepare/finalize methods without initializing HCCL.
    # The pad helper's NPU allocation service is replaced by CPU zero-padding.
    monkeypatch.setattr(
        pf, "_pad_tokens_with_cat", lambda value, length: torch.nn.functional.pad(value, (0, 0, 0, length - len(value)))
    )
    hidden = torch.arange(tokens * 8, dtype=torch.float32).reshape(tokens, 8)
    logits = hidden[:, :4].clone()
    layers, prepared = [], []
    for rank in range(2):
        layer = object.__new__(pf.PrepareAndFinalizeWithMC2)
        layer.tp_size, layer.tp_rank = 2, rank
        layer.moe_config = SimpleNamespace(tp_group=SimpleNamespace(device_group=object()))
        output = layer.prepare(hidden, logits)
        start = rank * (padded_tokens // 2)
        end = start + padded_tokens // 2
        assert output.hidden_states.shape == (padded_tokens // 2, 8)
        assert output.router_logits.shape == (padded_tokens // 2, 4)
        torch.testing.assert_close(output.mc2_mask, torch.arange(start, end) < tokens)
        valid = min(end, tokens) - start
        torch.testing.assert_close(output.hidden_states[:valid], hidden[start : start + valid])
        torch.testing.assert_close(output.router_logits[:valid], logits[start : start + valid])
        if valid < padded_tokens // 2:
            assert torch.count_nonzero(output.hidden_states[valid:]) == 0
            assert torch.count_nonzero(output.router_logits[valid:]) == 0
        layers.append(layer)
        prepared.append(output)

    def all_gather(destinations, local, group):
        assert local.shape == prepared[0].hidden_states.shape
        for destination, source in zip(destinations, prepared):
            destination.copy_(source.hidden_states)

    monkeypatch.setattr(pf.dist, "all_gather", all_gather)
    for layer, output in zip(layers, prepared):
        actual = layer.finalize(output.hidden_states, False, output.padded_hidden_states_shape)
        torch.testing.assert_close(actual, hidden)
