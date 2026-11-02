# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import moe_comm_method as comm_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, SharedExpertMLPPath
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2


@pytest.mark.parametrize("enabled", [True, False])
def test_megamoe_publishes_only_real_enqueue_boundary(monkeypatch, enabled):
    comm = object.__new__(FusedMC2CommImpl)
    comm.enable_fused_mc2 = 1
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm._apply_cann_mega_moe = lambda *_args, **_kwargs: ("output", "counts")
    event = object()
    recorded = []
    monkeypatch.setattr(
        torch.npu, "current_stream", lambda: SimpleNamespace(record_event=lambda: (recorded.append(event), event)[1])
    )
    monkeypatch.setattr(comm_module, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True, is_decode_only_node=False))
    monkeypatch.setattr(
        comm_module,
        "get_ascend_config",
        lambda: SimpleNamespace(allow_megamoe_shared_expert_overlap=enabled, multistream_overlap_shared_expert=enabled),
    )
    result = comm.fused_experts(SimpleNamespace(layer=None), SimpleNamespace(get_fused_mc2_weights=lambda _: None))
    assert result.routed_out == "output"
    assert result.fused_moe_start_evt is (event if enabled else None)
    assert recorded == ([event] if enabled else [])
    assert result.before_moe_up_evt is None and result.before_gmm2_evt is None
    milestones = RoutedMoEMilestones(routed_fused_start=result.fused_moe_start_evt)
    assert not milestones.has_fine_grained_stage_events


def test_shared_tail_waits_for_fused_start_not_internal_stages():
    shared = object.__new__(AscendSharedExperts)
    order = []
    shared._wait_for_milestone = lambda _event, name: order.append(name)
    shared._finish_linear_wrapped_mlp = lambda *_args: (order.append("shared_tail"), "output")[1]
    submitted = SimpleNamespace(path=SharedExpertMLPPath.LINEAR_WRAPPER)
    result = shared._finish_shared_mlp(submitted, RoutedMoEMilestones(routed_fused_start=object()))
    assert result == "output"
    assert order == ["routed_fused_start", "shared_tail"]
