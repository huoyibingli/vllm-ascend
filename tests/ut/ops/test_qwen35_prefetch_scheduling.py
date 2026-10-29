# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.ops import weight_prefetch_qwen35 as prefetch
from vllm_ascend.weight_prefetch_config import Qwen35WeightPrefetchConfig


class Weight:
    shape = (2048, 2048)

    def data_ptr(self):
        return id(self)


@pytest.fixture
def runtime(monkeypatch):
    order = []
    streams = []
    active = ["main"]

    class Stream:
        def __init__(self, device=None):
            self.name = f"prefetch-{len(streams)}"
            streams.append(self)

        def wait_stream(self, other):
            name = other.name if hasattr(other, "name") else other
            order.append((self.name, "wait", name))

    main = SimpleNamespace(name="main", wait_stream=lambda stream: order.append(("main", "wait", stream.name)))

    @contextmanager
    def use(stream):
        previous = active[0]
        active[0] = stream.name
        try:
            yield
        finally:
            active[0] = previous

    monkeypatch.setattr(prefetch.torch.npu, "Stream", Stream)
    monkeypatch.setattr(prefetch.torch.npu, "current_stream", lambda: main)
    monkeypatch.setattr(prefetch.torch.npu, "stream", use)
    monkeypatch.setattr(
        prefetch.torch_npu,
        "npu_prefetch",
        lambda weight, dependency, size, offset: order.append((active[0], "prefetch", weight, dependency, size)),
        raising=False,
    )
    monkeypatch.setattr(prefetch.Qwen35WeightPrefetch, "can_run", lambda _: True)
    scratch = MagicMock(side_effect=AssertionError("Dense targets must not allocate routed scratch"))
    monkeypatch.setattr(prefetch.torch, "empty", scratch)
    return SimpleNamespace(order=order, streams=streams, scratch=scratch)


def controller(*targets, mode="full", immediate=True):
    config = Qwen35WeightPrefetchConfig.parse({"enabled": True, "targets": list(targets), "mode": mode})
    result = prefetch.Qwen35WeightPrefetch(config, device=0)
    input_weight, output_weight = Weight(), Weight()
    result.next_inputs["source"] = (
        SimpleNamespace(weight=input_weight),
        input_weight,
        input_weight.data_ptr(),
        input_weight.shape,
        2048,
        immediate,
    )
    result.gdn_out_weights["target"] = (
        SimpleNamespace(weight=output_weight),
        output_weight,
        output_weight.data_ptr(),
        output_weight.shape,
        2048,
    )
    result.next_gdn_out["source"] = "target"
    return result


def test_input_join_does_not_drain_gdn_lookahead(runtime):
    target = controller("next_input", "gdn_out")
    dependency = object()
    assert target.stream is not target.gdn_out_stream
    assert target.begin_next("source", dependency)
    prepared = target.prepare_next_gdn_out("source", dependency)
    runtime.order.append(("main", "collective"))
    target.commit_next_gdn_out(prepared)
    target.settle_next_input()
    # Same-layer fallback must not enqueue the output weight a second time.
    assert not target.begin_gdn_out("target", dependency)
    assert target.prepare_next_gdn_out("source", dependency) is None
    target.join_gdn_out("target")
    assert [(x[0], x[1], x[2]) for x in runtime.order if x[1] == "wait"] == [
        ("prefetch-0", "wait", "main"),
        ("prefetch-1", "wait", "main"),
        ("main", "wait", "prefetch-0"),
        ("main", "wait", "prefetch-1"),
    ]
    collective = runtime.order.index(("main", "collective"))
    transfers = [(index, item) for index, item in enumerate(runtime.order) if item[1] == "prefetch"]
    assert len(transfers) == 2
    assert transfers[0][0] < collective < transfers[1][0]
    assert transfers[1][1][3] is dependency
    assert not target.pending_gdn_out
    assert target.done is None
    runtime.scratch.assert_not_called()


@pytest.mark.parametrize("mode", ["full", "control"])
def test_deferred_input_join_keeps_output_wait_independent(runtime, mode):
    target = controller("next_input", "gdn_out", mode=mode, immediate=False)
    assert target.begin_next("source", object())
    assert not target.begin_next("source", object())
    prepared = target.prepare_next_gdn_out("source", object())
    target.commit_next_gdn_out(prepared)
    target.settle_next_input()
    assert not any(item[0] == "main" for item in runtime.order)
    target.join_next_input("source")
    assert ("main", "wait", target.stream.name) == runtime.order[-1]
    assert "target" in target.pending_gdn_out
    target.join_gdn_out("target")
    assert ("main", "wait", target.gdn_out_stream.name) == runtime.order[-1]
    assert len([item for item in runtime.order if item[1] == "prefetch"]) == (2 if mode == "full" else 0)


def test_first_layer_fallback_and_single_target_keep_one_stream(runtime):
    target = controller("gdn_out")
    assert target.stream is target.gdn_out_stream
    assert len(runtime.streams) == 1
    assert target.begin_gdn_out("target", object())
    assert not target.begin_gdn_out("target", object())
    target.join_gdn_out("target")
    assert len([item for item in runtime.order if item[1] == "prefetch"]) == 1


def test_replaced_weight_and_skipped_batches_do_not_submit(runtime, monkeypatch):
    target = controller("next_input", "gdn_out")
    target.gdn_out_weights["target"][0].weight = Weight()
    assert target.prepare_next_gdn_out("source", object()) is None
    assert not target.begin_gdn_out("target", object())
    monkeypatch.setattr(target, "can_run", lambda: False)
    assert not target.begin_next("source", object())
    assert not runtime.order
    assert not target.pending_next_input
    assert not target.pending_gdn_out


@pytest.mark.parametrize("combined", [False, True])
def test_binding_connects_each_following_gdn_even_with_input_target(monkeypatch, combined):
    import sys
    from types import ModuleType

    class Layer:
        pass

    class Owner:
        pass

    def stub_module(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    stub_module("vllm.model_executor.models.qwen3_5", Qwen3_5DecoderLayer=Layer)
    stub_module("vllm_ascend.ops.fused_moe.fused_moe", AscendMoERunner=Owner)
    stub_module("vllm_ascend.utils", AscendDeviceType=SimpleNamespace(A2="A2"), get_ascend_device_type=lambda: "A2")
    monkeypatch.setattr(prefetch, "supported_weight", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(prefetch.torch_npu, "get_npu_format", lambda _: 29, raising=False)
    target = SimpleNamespace(next_inputs={}, gdn_out_weights={}, next_gdn_out={}, routed_weights={})
    monkeypatch.setattr(prefetch, "Qwen35WeightPrefetch", lambda *_args: target)
    layers = []
    weight = SimpleNamespace(device="npu", numel=lambda: 16 * 1024 * 1024, element_size=lambda: 2, data_ptr=lambda: 42)
    for index in range(40):
        layer = Layer()
        layer.layer_idx = index
        layer.layer_type = "full_attention" if index % 4 == 3 else "linear_attention"
        projection = SimpleNamespace(weight=weight)
        layer.linear_attn = SimpleNamespace(
            prefix=f"model.layers.{index}.linear_attn", out_proj=projection, in_proj_qkvz=projection
        )
        layer.self_attn = SimpleNamespace(qkv_proj=projection)
        owner = Owner()
        owner.layer_name = f"model.layers.{index}.mlp"
        layer.mlp = SimpleNamespace(modules=lambda owner=owner: iter([owner]))
        layers.append(layer)
    model = SimpleNamespace(modules=lambda: iter(layers), parameters=lambda: iter([weight]))
    runtime_config = SimpleNamespace(
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
        {
            "enabled": True,
            "allow_ep_nz_dense": True,
            "targets": ["next_input", "gdn_out"] if combined else ["gdn_out"],
        }
    )
    prefetch.bind_qwen35_weight_prefetch(model, runtime_config, config)
    assert len(target.gdn_out_weights) == 30
    assert len(target.next_gdn_out) == 29
    assert len(target.next_inputs) == (39 if combined else 0)
    assert "model.layers.0.linear_attn" not in target.next_gdn_out.values()
    assert target.next_gdn_out["model.layers.0.mlp"] == "model.layers.1.linear_attn"


@pytest.mark.parametrize(
    "comm,concurrency,allowed",
    [
        ("ALLGATHER", 0, False),
        ("ALLGATHER", 7, False),
        ("ALLGATHER", 8, True),
        ("ALLGATHER", 32, True),
        ("FUSED_MC2", 32, False),
    ],
)
def test_original_decode_gate_is_preserved(monkeypatch, comm, concurrency, allowed):
    target = object.__new__(prefetch.Qwen35WeightPrefetch)
    target.config = SimpleNamespace(min_decode_concurrency=8)
    monkeypatch.setattr(prefetch, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=getattr(prefetch.MoECommType, comm)))
    monkeypatch.setattr(prefetch, "decode_concurrency", lambda: concurrency)
    assert target.can_run() is allowed
