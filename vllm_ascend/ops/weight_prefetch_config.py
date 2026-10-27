# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports; config itself must not initialize the ops package."""

from vllm_ascend.weight_prefetch_config import (
    NUM_HIDDEN_LAYERS as NUM_HIDDEN_LAYERS,
)
from vllm_ascend.weight_prefetch_config import (
    Qwen35WeightPrefetchConfig as Qwen35WeightPrefetchConfig,
)
