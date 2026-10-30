# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend import ascend_config as config_module
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device.hardware_profile import MoECommPolicy
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts


@pytest.mark.parametrize(
    "unsupported", [None, "package", "hardware", "policy", "ep", "dp", "pp", "quant", "lora", "spec", "model"]
)
def test_a2_megamoe_shared_overlap_support(monkeypatch, unsupported):
    config = object.__new__(AscendConfig)
    config.enable_fused_mc2 = 1
    config.multistream_overlap_shared_expert = True
    config._is_megamoe_supported_by_config = lambda _: unsupported != "model"
    monkeypatch.setattr(config_module, "is_mega_moe_supported", lambda: unsupported != "package")
    policy = MoECommPolicy.FUSED_OR_CAPACITY if unsupported == "policy" else MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY
    monkeypatch.setattr(
        config_module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda _: unsupported != "hardware", moe_comm_policy=policy),
    )
    model = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=unsupported != "ep",
            data_parallel_size=2 if unsupported == "dp" else 1,
            pipeline_parallel_size=2 if unsupported == "pp" else 1,
        ),
        quant_config=object() if unsupported == "quant" else None,
        lora_config=object() if unsupported == "lora" else None,
        speculative_config=object() if unsupported == "spec" else None,
    )
    assert config._supports_megamoe_shared_expert_overlap(model) is (unsupported is None)


def test_fused_shared_fallback_never_waits_for_internal_events():
    shared = object.__new__(AscendSharedExperts)
    shared.multistream_overlap = True

    def forbidden_wait(*_args):
        pytest.fail("Fused MegaMoe must not wait for an internal event it cannot produce")

    shared._wait_for_milestone = forbidden_wait
    shared._wait_for_routed_stage(RoutedMoEMilestones(), None, "shared_activation_overlap_start")
