# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend import ascend_forward_context as afc


@pytest.mark.parametrize(
    "tokens,prefills,decodes,profile,dp,draft,threshold,expected_fused",
    [
        (65, 1, 0, False, 1, False, 64, True),
        (64, 1, 0, False, 1, False, 64, False),
        (512, 1, 1, False, 1, False, 64, False),
        (512, 0, 512, False, 1, False, 64, False),
        (512, 1, 0, False, 2, False, 64, False),
        (512, 1, 0, False, 1, True, 64, False),
        (64, 0, None, True, 1, False, 64, True),
        (512, 1, None, False, 1, False, 64, False),
        (128, 1, 0, False, 1, False, 128, False),
        (129, 1, 0, False, 1, False, 128, True),
    ],
)
@pytest.mark.parametrize("as_dict", [False, True])
def test_a2_prefill_selection(
    monkeypatch, tokens, prefills, decodes, profile, dp, draft, threshold, expected_fused, as_dict
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(get_num_experts=lambda: 256),
        parallel_config=SimpleNamespace(world_size_across_dp=4, pipeline_parallel_size=1, data_parallel_size=dp),
        speculative_config=None,
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(mega_moe_prefill_token_threshold=threshold))
    metadata = SimpleNamespace(num_prefills=prefills, num_decode_tokens=decodes)
    if as_dict:
        metadata = {"layer0": metadata}
    actual = afc._select_capacity_and_expert_density_moe_comm_method(
        tokens, config, 512, is_draft_model=draft, attn_metadata=metadata, in_profile_run=profile
    )
    expected = afc.MoECommType.FUSED_MC2 if expected_fused else afc.MoECommType.ALLGATHER
    assert actual == expected


def test_negative_prefill_threshold():
    from vllm_ascend.ascend_config import AscendConfig

    with pytest.raises(ValueError, match="mega_moe_prefill_token_threshold"):
        AscendConfig(sparse_kv_offload_config=None, mega_moe_prefill_token_threshold=-1)
