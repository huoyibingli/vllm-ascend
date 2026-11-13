# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of production overlap scheduling without vLLM/torch_npu."""

import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.ops.test_a2_mega_moe import ROOT, Settings


def load_classes(path, names, namespace, methods=None):
    source = ast.parse((ROOT / path).read_text())
    nodes = []
    for node in source.body:
        if not isinstance(node, ast.ClassDef) or node.name not in names:
            continue
        node.bases = []
        if methods is not None:
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        nodes.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)


@pytest.mark.parametrize("scope", ["all", "prefill", "prefill_mixed"])
@pytest.mark.parametrize("a2", [False, True])
@pytest.mark.parametrize("backend", [0, 1, 2])
@pytest.mark.parametrize("overlap", [False, True])
def test_only_explicit_a2_cann_backend_preserves_shared_overlap(scope, a2, backend, overlap):
    source = ast.parse((ROOT / "vllm_ascend/ascend_config.py").read_text())
    owner = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
    validate = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_user_input_ranges")
    validate.decorator_list = []
    derive = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "derive_and_validate")
    guard = next(
        n
        for n in derive.body
        if isinstance(n, ast.If)
        and any(
            isinstance(a, ast.Attribute) and a.attr == "multistream_overlap_shared_expert" for a in ast.walk(n.test)
        )
    )
    config = SimpleNamespace(
        weight_nz_mode=1,
        enable_mc2_hierarchy_comm=False,
        enable_fused_mc2=backend,
        a2_mega_moe_prefill=Settings(enabled=a2, scope=scope),
        multistream_overlap_shared_expert=overlap,
    )
    namespace = dict(
        importlib=SimpleNamespace(util=SimpleNamespace(find_spec=lambda _: object())),
        logger=Mock(),
        self=config,
    )
    exec(compile(ast.Module(body=[validate], type_ignores=[]), "ascend_config.py", "exec"), namespace)
    if a2 and backend != 2:
        with pytest.raises(ValueError, match="enable_fused_mc2=2"):
            namespace[validate.name](config)
        return
    namespace[validate.name](config)
    namespace["is_mega_moe_supported"] = lambda: namespace["_MEGA_MOE_SUPPORTED"]
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "ascend_config.py", "exec"), namespace)
    assert config.multistream_overlap_shared_expert is (overlap and (backend == 0 or a2))


@pytest.fixture
def execution(monkeypatch):
    order = []
    current = ["main"]

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            event = f"{self.name}:{len(order)}"
            order.append((self.name, "record", event))
            return event

        def wait_event(self, event):
            order.append((self.name, "wait", event))

        def wait_stream(self, stream):
            order.append((self.name, "join", stream.name))

    streams = {name: Stream(name) for name in ("main", "shared")}

    @contextmanager
    def switch(stream, *, enabled=True):
        previous = current[0]
        if enabled:
            current[0] = stream.name
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: streams[current[0]]), raising=False)
    namespace = dict(
        __name__=__name__,
        torch=torch,
        F=torch.nn.functional,
        dataclass=dataclass,
        Enum=Enum,
        auto=auto,
        npu_stream_switch=switch,
        shared_experts_calculation_stream=lambda: streams["shared"],
        has_lora=lambda _: False,
        QuantType=SimpleNamespace(NONE=0, W8A8=1, W4A8=2, W4A8MXFP=3, W8A8MXFP=4),
    )
    load_classes(
        "vllm_ascend/ops/fused_moe/dataclass/shared_experts.py",
        {"RoutedMoEMilestones", "PreparedSharedExpertInput"},
        namespace,
    )
    load_classes(
        "vllm_ascend/ops/fused_moe/shared_experts.py",
        {
            "SharedExpertParallelMode",
            "SharedExpertMLPPath",
            "PreparedSharedExpertExecution",
            "SubmittedSharedExpertGateUp",
        },
        namespace,
    )
    load_classes(
        "vllm_ascend/ops/fused_moe/shared_experts.py",
        {"AscendSharedExperts"},
        namespace,
        {
            "parallel_mode",
            "prepare_input_before_routed",
            "_gather_sp_input",
            "_pad_and_reduce_scatter",
            "_wait_for_milestone",
            "_wait_for_routed_stage",
            "_prepare_execution_input",
            "prepare_for_router_overlap",
            "_prepare_gate_up_input",
            "_prepare_linear_input_for_overlap",
            "enqueue_gate_up_after_router",
            "_finish_linear_wrapped_mlp",
            "_finish_shared_mlp",
            "_run_a8_int_mlp",
            "_select_mlp_path",
            "wait_for_output",
            "forward",
            "part1",
            "apply_activation",
            "part2",
            "_apply_output_gate",
        },
    )
    load_classes(
        "vllm_ascend/ops/fused_moe/fused_moe.py",
        {"AscendMoERunner"},
        namespace,
        {
            "_forward_impl",
            "_prepare_router_and_milestones",
            "_prepare_shared_for_router_overlap",
            "_can_overlap_sp_shared_with",
        },
    )
    return namespace, order, current, streams


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("backend", ["megamoe", "allgather"])
@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("quantized_shared", [False, True])
def test_real_runner_preserves_outputs_and_orders_sp_collectives(
    execution, monkeypatch, sp, rank, backend, with_gate, quantized_shared
):
    ns, order, current, streams = execution
    torch.manual_seed(17)
    full = torch.randn(16, 4)
    context = SimpleNamespace(num_tokens=9)
    ns["_EXTRA_CTX"] = context
    shared = ns["AscendSharedExperts"]()
    shared.multistream_overlap = False
    shared.weights_replicated = False
    shared.quant_type = 0
    shared.lora_context = None
    shared.situ_activation = None
    shared.swiglu_limit = 0.0
    shared.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=8), is_sequence_parallel=sp)
    w1, w2, gate = torch.randn(4, 6), torch.randn(3, 4), torch.randn(4, 1)

    def gate_up(x):
        order.append((current[0], "gate_up"))
        return x @ w1, None

    def down(x):
        order.append((current[0], "down"))
        return x @ w2, None

    shared.layer = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=down,
        act_fn=lambda x: torch.nn.functional.silu(x[:, :3]) * x[:, 3:],
        expert_gate=(lambda x: (x @ gate, None)) if with_gate else None,
    )
    if quantized_shared:
        # Simulate only the device primitives; run the production W8A8 split
        # pipeline and stream/event scheduling. NPU precision is tested on board.
        shared.quant_type = ns["QuantType"].W8A8
        gate_up.weight = torch.randint(-8, 9, w1.shape, dtype=torch.int32)
        gate_up.weight_scale = gate_up.weight_scale_fp32 = torch.full((6,), 0.05)
        down.weight = torch.randint(-8, 9, w2.shape, dtype=torch.int32)
        down.weight_scale = torch.full((4,), 0.05)

        def quantize(x):
            scale = x.abs().amax(-1).clamp_min(1e-8) / 127
            return (x / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale

        def quant_matmul(x, weight, scale, *, pertoken_scale, output_dtype, **kwargs):
            order.append((current[0], "gate_up" if weight is gate_up.weight else "down"))
            result = x.int() @ weight
            if output_dtype != torch.int32:
                result = result.float() * scale * pertoken_scale[:, None]
            return result.to(output_dtype)

        def dequant_swiglu_quant(*, x, weight_scale, activation_scale, **kwargs):
            dequantized = x.float() * weight_scale * activation_scale[:, None]
            return quantize(shared.layer.act_fn(dequantized))

        ns["torch_npu"] = SimpleNamespace(npu_dynamic_quant=quantize, npu_quant_matmul=quant_matmul)
        ns["get_current_hardware_profile"] = lambda: SimpleNamespace(supports=lambda _: False)
        ns["HardwareCapability"] = SimpleNamespace(FUSED_SWIGLU_TUNING_ARGS=0)
        monkeypatch.setattr(torch.ops._C_ascend, "npu_dequant_swiglu_quant", dequant_swiglu_quant, raising=False)

    def gather(x, dim):
        order.append((current[0], "gather"))
        return full

    def scatter(x, dim):
        order.append((current[0], "scatter"))
        return x.chunk(8)[rank] * 8

    ns["tensor_model_parallel_all_gather"] = gather
    ns["tensor_model_parallel_reduce_scatter"] = scatter
    ns["maybe_record_event"] = lambda: streams[current[0]].record_event() if shared.multistream_overlap else None

    def routed(*, hidden_states, **kwargs):
        order.append((current[0], backend))
        milestones = ns["RoutedMoEMilestones"]()
        if backend == "allgather" and shared.multistream_overlap:
            milestones.routed_dispatch_start = streams["main"].record_event()
            milestones.shared_activation_overlap_start = streams["main"].record_event()
            milestones.routed_combine_start = streams["main"].record_event()
        return hidden_states * 0.5, milestones

    runner = ns["AscendMoERunner"]()
    runner.ascend_shared_experts = shared
    runner.is_internal_router = False
    runner.routed_input_transform = runner.routed_output_transform = None
    runner.routed_experts = SimpleNamespace(forward_impl=routed)
    runner._sequence_parallel_context = nullcontext
    x = full.chunk(8)[rank] if sp else full[:9]
    serial = runner._forward_impl(x, torch.zeros(x.shape[0], 512), None)
    order.clear()
    shared.multistream_overlap = True
    parallel = runner._forward_impl(x, torch.zeros(x.shape[0], 512), None)
    for actual, expected in zip(parallel, serial):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert order.index(("shared", "gate_up")) < order.index(("main", backend))
    assert order[-1] == ("main", "join", "shared")
    if sp:
        assert order[0] == ("main", "gather")
        assert order.count(("main", "gather")) == 1
        scatter_index = order.index(("shared", "scatter"))
        assert order[scatter_index - 1][:2] == ("shared", "wait")
        assert order.index(("shared", "down")) < scatter_index
    else:
        assert not any(item[1] in ("gather", "scatter") for item in order)


def test_partial_fused_milestones_still_fail_fast(execution):
    ns, _, _, _ = execution
    shared = ns["AscendSharedExperts"]()
    shared.multistream_overlap = True
    milestones = ns["RoutedMoEMilestones"](routed_dispatch_start="dispatch")
    with pytest.raises(RuntimeError, match="Missing shared_activation_overlap_start"):
        shared._wait_for_routed_stage(milestones, None, "shared_activation_overlap_start")
