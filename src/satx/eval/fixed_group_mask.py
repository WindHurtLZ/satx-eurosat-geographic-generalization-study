from __future__ import annotations

import torch
from torch import Tensor

from satx.data.band_info import (
    MS_BAND_ORDER,
    SPECTRAL_GROUP_INDICES,
    SPECTRAL_GROUP_NAMES,
)


def apply_fixed_group_mask(inputs: Tensor, group_name: str | None) -> Tensor:
    """
    Fixed masking for robustness evaluation

    Args:
        inputs shape: [B, 13, H, W]
        group_name: One of SPECTRAL_GROUP_NAMES. None or "CLEAN" means no masking

    Outputs: same shape but masked
    """
    if inputs.ndim != 4:
        raise ValueError(
            "Expected inputs with shape [B, 13, H, W], "
            f"but received {tuple(inputs.shape)}."
        )

    expected_channels = len(MS_BAND_ORDER)
    if inputs.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, "
            f"but received {inputs.shape[1]}."
        )

    if group_name is None or group_name == "CLEAN":
        return inputs

    if group_name not in SPECTRAL_GROUP_NAMES:
        raise ValueError(
            f"Unknown spectral group {group_name!r}. "
            f"Expected one of {SPECTRAL_GROUP_NAMES}."
        )

    output = inputs.clone()
    channel_indices = torch.tensor(
        SPECTRAL_GROUP_INDICES[group_name],
        dtype=torch.long,
        device=inputs.device,
    )
    output[:, channel_indices, :, :] = 0.0
    return output
