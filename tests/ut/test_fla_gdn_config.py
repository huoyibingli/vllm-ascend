# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend import ascend_config
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
def test_fla_gdn_chunk_size_defaults_to_64(monkeypatch, device_type):
    monkeypatch.setattr(ascend_config, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    assert AscendConfig(sparse_kv_offload_config=None).fla_gdn_chunk_size == 64


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_fla_gdn_chunk128_supported_profiles(monkeypatch, device_type):
    monkeypatch.setattr(ascend_config, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    assert AscendConfig(sparse_kv_offload_config=None, fla_gdn_chunk_size=128).fla_gdn_chunk_size == 128


@pytest.mark.parametrize("device_type", [AscendDeviceType.A5, AscendDeviceType._310P])
def test_fla_gdn_chunk128_rejects_unsupported_profiles(monkeypatch, device_type):
    monkeypatch.setattr(ascend_config, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    with pytest.raises(ValueError, match="only supported by the A2/A3 FLA GDN prefill path"):
        AscendConfig(sparse_kv_offload_config=None, fla_gdn_chunk_size=128)


@pytest.mark.parametrize("size", [0, 32, 256, None, True, "128", "invalid"])
def test_fla_gdn_chunk_size_rejects_invalid_values(size):
    with pytest.raises(ValueError, match="fla_gdn_chunk_size"):
        AscendConfig(sparse_kv_offload_config=None, fla_gdn_chunk_size=size)
