# SPDX-License-Identifier: Apache-2.0
"""Explicit memory budget for static BF16 MegaMoe/decode weight layouts."""

from dataclasses import dataclass

QWEN35_NUM_LAYERS = 40
GIB = 1024**3


@dataclass(frozen=True)
class Qwen35MegaMoeDecodeNZConfig:
    enabled: bool = False
    layers: tuple[int, ...] | None = None
    max_cache_bytes: int = 0
    min_free_bytes: int = 2 * GIB

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("qwen35_megamoe_decode_nz must be an object")
        if set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown qwen35_megamoe_decode_nz option")
        options = dict(value)
        layers = options.get("layers")
        if layers is not None:
            if not isinstance(layers, (list, tuple)) or not layers:
                raise ValueError("layers must be null or a nonempty array")
            if any(type(index) is not int or not 0 <= index < QWEN35_NUM_LAYERS for index in layers):
                raise ValueError("layers must contain integer indices in [0, 40)")
            if len(set(layers)) != len(layers):
                raise ValueError("duplicate decode NZ layers")
            options["layers"] = tuple(layers)
        result = cls(**options)
        if type(result.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("max_cache_bytes", "min_free_bytes"):
            size = getattr(result, name)
            if type(size) is not int or size < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if result.enabled and result.max_cache_bytes == 0:
            raise ValueError("enabled decode NZ cache requires an explicit positive max_cache_bytes")
        return result
