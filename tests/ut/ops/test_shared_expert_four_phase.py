# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts


@pytest.mark.parametrize("with_gate", [False, True])
def test_four_phase_order_and_values(with_gate):
    executor = object.__new__(AscendSharedExperts)
    executor.multistream_overlap = True
    order = []
    executor._uses_four_phase_overlap = lambda: True
    executor._wait_for_milestone = lambda _event, name: order.append(name)
    executor.apply_activation = lambda x: torch.nn.functional.silu(x)
    executor._prepare_linear_input_for_overlap = lambda _layer, x: x

    def down(x):
        order.append("down")
        return x * 2, None

    executor.layer = SimpleNamespace(down_proj=down)
    x = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    gate = torch.tensor([[0.2], [-0.7]]) if with_gate else None
    submitted = SimpleNamespace(gate_up=x, expert_gate=gate)
    milestones = RoutedMoEMilestones(
        routed_mlp_start=object(), routed_activation_start=object(), routed_gmm2_start=object()
    )
    actual = executor._finish_linear_wrapped_mlp(submitted, milestones)
    expected = torch.nn.functional.silu(x) * 2
    if gate is not None:
        expected *= torch.sigmoid(gate)
    torch.testing.assert_close(actual, expected)
    assert order == ["routed_mlp_start", "routed_activation_start", "down", "routed_gmm2_start"]


def test_missing_stage_keeps_original_fallback():
    executor = object.__new__(AscendSharedExperts)
    executor._uses_four_phase_overlap = lambda: True
    executor._wait_for_routed_stage = lambda *_args: None
    executor.apply_activation = lambda x: x + 1
    executor._prepare_linear_input_for_overlap = lambda _layer, x: x
    executor.layer = SimpleNamespace(down_proj=object())
    executor.part2 = lambda _hidden, x: x * 3
    submitted = SimpleNamespace(gate_up=torch.ones(2, 3), hidden_states=torch.zeros(2, 3))
    actual = executor._finish_linear_wrapped_mlp(submitted, RoutedMoEMilestones())
    torch.testing.assert_close(actual, torch.full((2, 3), 6.0))
