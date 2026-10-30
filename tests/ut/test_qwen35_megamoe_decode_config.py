# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig


def setup_config():
    config = object.__new__(AscendConfig)
    config.allow_megamoe_shared_expert_overlap = True
    config._supports_megamoe_shared_expert_overlap = lambda _: True
    config.weight_nz_mode = 2
    config.enable_force_eplb = False
    config.eplb_config = SimpleNamespace(dynamic_eplb=False)
    runtime = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2, prefill_context_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="qwen3_5_moe_text",
                num_hidden_layers=40,
                hidden_size=2048,
                num_experts=256,
                moe_intermediate_size=512,
            ),
            dtype=torch.bfloat16,
            enable_sleep_mode=False,
        ),
        use_v2_model_runner=False,
        weight_transfer_config=None,
    )
    return config, runtime


def test_supported_static_configuration():
    config, runtime = setup_config()
    # FlashComm is deliberately not an AscendConfig field. The production
    # derivation computes it from additional_config, the environment and DSA-CP.
    assert not hasattr(config, "enable_flashcomm1")
    config._validate_qwen35_megamoe_decode_nz(runtime, effective_flashcomm=False)


@pytest.mark.parametrize(
    "target,name,value",
    [
        ("config", "weight_nz_mode", 1),
        ("config", "allow_megamoe_shared_expert_overlap", False),
        ("config", "enable_force_eplb", True),
        ("runtime", "use_v2_model_runner", True),
        ("runtime", "weight_transfer_config", object()),
        ("parallel", "tensor_parallel_size", 4),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "use_sequence_parallel_moe", True),
        ("model", "dtype", torch.float16),
        ("model", "enable_sleep_mode", True),
        ("hf", "hidden_size", 4096),
        ("hf", "num_hidden_layers", 48),
        ("hf", "model_type", "other"),
        ("eplb", "dynamic_eplb", True),
    ],
)
def test_unsupported_configuration_is_rejected(target, name, value):
    config, runtime = setup_config()
    owners = dict(
        config=config,
        runtime=runtime,
        parallel=runtime.parallel_config,
        model=runtime.model_config,
        hf=runtime.model_config.hf_text_config,
        eplb=config.eplb_config,
    )
    setattr(owners[target], name, value)
    with pytest.raises(ValueError, match="static A2"):
        config._validate_qwen35_megamoe_decode_nz(runtime, effective_flashcomm=False)


def test_missing_megamoe_or_shared_support_is_rejected():
    config, runtime = setup_config()
    config._supports_megamoe_shared_expert_overlap = lambda _: False
    with pytest.raises(ValueError, match="static A2"):
        config._validate_qwen35_megamoe_decode_nz(runtime, effective_flashcomm=False)


def test_effective_flashcomm_is_rejected_without_reading_a_config_field():
    config, runtime = setup_config()
    assert not hasattr(config, "enable_flashcomm1")
    with pytest.raises(ValueError, match="static A2"):
        config._validate_qwen35_megamoe_decode_nz(runtime, effective_flashcomm=True)
