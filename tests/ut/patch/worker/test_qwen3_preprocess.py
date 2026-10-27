# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.worker.patch_qwen3_preprocess import _patchify_on_npu


@pytest.mark.parametrize("grids", [[(2, 4)], [(2, 4), (2, 4), (4, 2)], [(4, 4), (2, 2)]])
@pytest.mark.parametrize("temporal", [1, 2])
def test_patchify_matches_pixel_order(grids, temporal):
    channels, patch, merge = 3, 2, 2
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(
                in_channels=channels, patch_size=patch, spatial_merge_size=merge, temporal_patch_size=temporal
            )
        )
    )
    images = [
        torch.arange(channels * h * w * patch * patch).reshape(channels, h * patch, w * patch).float() for h, w in grids
    ]
    expected = []
    for image, (height, width) in zip(images, grids):
        for row in range(0, height, merge):
            for col in range(0, width, merge):
                for dr in range(merge):
                    for dc in range(merge):
                        token = []
                        for channel in range(channels):
                            pixels = (
                                image[
                                    channel,
                                    (row + dr) * patch : (row + dr + 1) * patch,
                                    (col + dc) * patch : (col + dc + 1) * patch,
                                ]
                                .flatten()
                                .tolist()
                            )
                            token.extend(pixels * temporal)
                        expected.append(token)
    result = _patchify_on_npu(
        model, torch.cat([image.flatten() for image in images]), torch.tensor([(1, *g) for g in grids])
    )
    torch.testing.assert_close(result, torch.tensor(expected))


@pytest.mark.parametrize(
    "grid,values",
    [([[1, 0, 2]], 0), ([[2, 2, 2]], 48), ([[1, 2]], 48), ([[1, 3, 2]], 72), ([[1, 2, 2]], 47), ([[1, 2, 2]], 49)],
)
def test_patchify_rejects_invalid_transport(grid, values):
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(in_channels=3, patch_size=2, spatial_merge_size=2, temporal_patch_size=2)
        )
    )
    with pytest.raises(ValueError):
        _patchify_on_npu(model, torch.zeros(values), torch.tensor(grid))
