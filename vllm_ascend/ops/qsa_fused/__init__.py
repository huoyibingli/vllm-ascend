# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration of the cloud_qsa_indexer_fused_v2 aclnn operator.

The operator fuses the QSA indexer's int8 quant matmul (W8A8_DYNAMIC,
AIC cube) with per-group K compression, Gemma RMSNorm and interleaved
mRoPE (AIV vector), replacing the Triton path used when the layer weights
stay in bf16. The aclnn entry points are resolved at runtime (dlsym) from
libopapi.so / libcust_opapi.so by the vendored pytorch_npu_helper.hpp, so
no vendor headers are required at build time.

Gated by ``VLLM_ASCEND_ENABLE_QSA_INDEXER_FUSED``; anything other than
"1"/"true" keeps the existing Triton implementation.
"""

import importlib
import importlib.util
import os

from vllm.logger import logger

_CPP_DIR = os.path.dirname(os.path.realpath(__file__))

# Cached extension module; False marks "tried and unavailable".
_extension = None


def is_qsa_indexer_fused_enabled() -> bool:
    """Return True when the env gate is on and the extension loads."""
    global _extension
    value = os.getenv("VLLM_ASCEND_ENABLE_QSA_INDEXER_FUSED", None)
    if value is None or value.strip().lower() in ("", "0", "false"):
        return False
    if _extension is None:
        _extension = _load_extension()
    return _extension is not False


def _ascend_home() -> str:
    return os.getenv(
        "ASCEND_TOOLKIT_HOME",
        os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"),
    )


def _torch_npu_include_dirs() -> list[str]:
    dirs = []
    try:
        import torch_npu

        base = os.path.dirname(torch_npu.__file__)
        for cand in (os.path.join(base, "include"), base):
            if os.path.isdir(os.path.join(cand, "torch_npu")):
                dirs.append(cand)
    except ImportError:
        pass
    return dirs


def _import_prebuilt() -> object | None:
    """Import a pre-built extension module (setup.py deployment path)."""
    try:
        return importlib.import_module("qsa_fused_ops_lib")
    except ImportError:
        pass
    for name in os.listdir(_CPP_DIR):
        if name.startswith("qsa_fused_ops_lib") and name.endswith((".so", ".pyd")):
            spec = importlib.util.spec_from_file_location(
                "qsa_fused_ops_lib", os.path.join(_CPP_DIR, name)
            )
            if spec is not None and spec.loader is not None:
                return _exec(spec)
    return None


def _exec(spec):
    import sys

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extension():
    prebuilt = _import_prebuilt()
    if prebuilt is not None:
        return prebuilt
    from torch.utils.cpp_extension import load

    try:
        return load(
            name="qsa_fused_ops_lib",
            sources=[os.path.join(_CPP_DIR, "qsa_fused_bind.cpp")],
            extra_include_paths=[
                _CPP_DIR,
                os.path.join(_ascend_home(), "include"),
                os.path.join(_ascend_home(), "runtime", "include"),
                *_torch_npu_include_dirs(),
            ],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - build environment specific
        logger.warning_once(
            "cloud_qsa_indexer_fused_v2 extension failed to build, "
            "falling back to the Triton QSA path: %s",
            exc,
        )
        return False


def cloud_qsa_indexer_fused_v2(
    activation,
    weight,
    weight_scale,
    token_scale,
    positions,
    q_gamma,
    k_gamma,
    cos_sin,
    compressor_state_cache,
    compressor_state_block_table,
    token_to_req,
    query_start_loc,
    logical_positions,
    compressed_slots,
    rope_cache,
    q_head_num,
    k_head_num,
    head_dim,
    eps,
    rotary_dim,
    compress_ratio,
    pos_offset,
):
    """Run the fused QSA indexer op; returns (q_out, k_out, raw_k_out).

    ``rope_cache`` may be a zero-numel int64 tensor when the raw key cache
    has no rope-position ring attached.
    """
    assert _extension is not None and _extension is not False
    return _extension.cloud_qsa_indexer_fused_v2(
        activation,
        weight,
        weight_scale,
        token_scale,
        positions,
        q_gamma,
        k_gamma,
        cos_sin,
        compressor_state_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        rope_cache,
        q_head_num,
        k_head_num,
        head_dim,
        eps,
        rotary_dim,
        compress_ratio,
        pos_offset,
    )
