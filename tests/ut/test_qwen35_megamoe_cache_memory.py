# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import megamoe_decode_nz as cache_module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize("failure", [False, True])
def test_cache_allocations_are_accounted_as_weights(monkeypatch, failure):
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(qwen35_megamoe_decode_nz=SimpleNamespace(enabled=True)),
        model=object(),
        device="npu:0",
        model_memory_usage=100,
    )
    samples = iter([1000, 1400])
    monkeypatch.setattr(torch.npu, "memory_allocated", lambda _: next(samples))

    def bind(*_args):
        if failure:
            raise RuntimeError("cache failed")

    monkeypatch.setattr(cache_module, "bind_qwen35_megamoe_decode_nz", bind)
    if failure:
        with pytest.raises(RuntimeError, match="cache failed"):
            NPUModelRunner._bind_qwen35_megamoe_decode_nz(runner)
        assert runner.model_memory_usage == 100
    else:
        NPUModelRunner._bind_qwen35_megamoe_decode_nz(runner)
        assert runner.model_memory_usage == 500


def test_disabled_cache_has_no_memory_query():
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(qwen35_megamoe_decode_nz=SimpleNamespace(enabled=False)),
    )
    NPUModelRunner._bind_qwen35_megamoe_decode_nz(runner)


@pytest.mark.parametrize("prefetch,nz", [(True, False), (False, True), (True, True)])
def test_static_layouts_reject_reload(prefetch, nz):
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(
            qwen35_weight_prefetch=SimpleNamespace(enabled=prefetch),
            qwen35_megamoe_decode_nz=SimpleNamespace(enabled=nz),
        )
    )
    with pytest.raises(RuntimeError, match="before reloading"):
        NPUModelRunner.reload_weights(runner)
