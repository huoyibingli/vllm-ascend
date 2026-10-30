# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-model A2 TP2/EP2 correctness check; requires MODEL_PATH.

Run on two idle cards after sourcing CANN:
    MODEL_PATH=/path/to/Qwen3.5-35B pytest -sv \
      tests/e2e/pull_request/two_card/test_qwen35_megamoe_decode_nz.py

Uses the repository's existing prefill/decode logprob comparison and tests
both eager execution and FULL_DECODE_ONLY graph capture. This is correctness
validation, not the production throughput measurement workload.
"""

import os

import pytest

from tests.e2e.pull_request.utils import compare_logprobs

MODEL_PATH = os.getenv("MODEL_PATH")
PROMPTS = [
    "Read the following context and continue the last sentence. "
    + "The city library opens in the morning and closes in the evening. " * 32
    + ending
    for ending in (
        "The library is",
        "In the morning,",
        "People read",
        "The books are",
        "Reading is",
        "Visitors come",
        "At closing time,",
        "Every shelf",
    )
]


@pytest.mark.skipif(not MODEL_PATH, reason="Set MODEL_PATH to local Qwen3.5-35B BF16 weights")
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_qwen35_megamoe_nz_shared_logprobs(enforce_eager):
    compare_logprobs(
        runner_kwargs={
            "model_name": MODEL_PATH,
            "dtype": "bfloat16",
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.8,
            "enforce_eager": enforce_eager,
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8],
            },
            "additional_config": {
                "weight_nz_mode": 2,
                "enable_flashcomm1": False,
                "multistream_overlap_shared_expert": True,
                "enable_fused_mc2": 2,
                "enable_fla_gdn": True,
                "fla_chunk_size": 128,
                "allow_megamoe_shared_expert_overlap": True,
                "mega_moe_prefill_token_threshold": 64,
                "mega_moe_max_tokens": 65536,
                "qwen35_weight_prefetch": {
                    "enabled": True,
                    "allow_ep_nz_dense": True,
                    "targets": ["next_input", "gdn_out"],
                    "mode": "full",
                    "next_input_bytes": 2097152,
                    "gdn_out_bytes": 2097152,
                    "min_decode_concurrency": 8,
                },
                "qwen35_megamoe_decode_nz": {
                    "enabled": True,
                    "layers": [0, 1, 2, 3],
                    "max_cache_bytes": 4 * 1024**3,
                },
            },
        },
        prompts=PROMPTS,
    )
