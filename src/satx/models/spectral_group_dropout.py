"""
MS Spatial Group Dropout Experiment - Robustness Test
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from satx.data.band_info import MS_BAND_ORDER, SPECTRAL_GROUP_INDICES, SPECTRAL_GROUP_NAMES

class SpectralGroupDropout(nn.Module):
    """
    With probability p, one of the four groups is selected uniformly and all
    channels in that group will be set to zero

    Example:
        p = 0.2, 20% of images in every batch will be selected randomly,
        and these images will drop a random group uniformly
    """

    def __init__(self, p: float, seed: int) -> None:
        super().__init__()

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0 and 1.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")

        self.p = float(p)
        self.seed = int(seed)

        # Keep dropout randomness separate from model/DataLoader randomness
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

        self.reset_epoch_counts()

    def reset_epoch_counts(self) -> None:
        self._epoch_mask_counts = {
            "NONE": 0,
            **{group_name: 0 for group_name in SPECTRAL_GROUP_NAMES},
        }

    @property
    def epoch_mask_counts(self) -> dict[str, int]:
        return dict(self._epoch_mask_counts)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Apply per-sample dropout

        Input shape: [batch, 13, height, width]
        Output shape: [batch, 13, height, width]
        """
        # Validation and test should not drop anything.
        if not self.training or self.p == 0.0:
            return inputs

        if inputs.ndim != 4:
            raise ValueError(
                "SpectralGroupDropout expects [B, C, H, W], "
                f"but received shape {tuple(inputs.shape)}"
            )

        expected_channels = len(MS_BAND_ORDER)
        if inputs.shape[1] != expected_channels:
            raise ValueError(
                f"SpectralGroupDropout requires {expected_channels} channels, "
                f"but received {inputs.shape[1]}"
            )

        batch_size = inputs.shape[0]

        should_mask = (
            torch.rand(batch_size, generator=self._generator) < self.p
        )

        # uniform over the four groups
        chosen_groups = torch.randint(
            low=0,
            high=len(SPECTRAL_GROUP_NAMES),
            size=(batch_size,),
            generator=self._generator,
        )

        masked_count = int(should_mask.sum().item())
        self._epoch_mask_counts["NONE"] += batch_size - masked_count

        if masked_count == 0:
            return inputs

        # Clone so the original DataLoader batch is not modified in place.
        output = inputs.clone()

        for group_number, group_name in enumerate(SPECTRAL_GROUP_NAMES):
            selected_rows = torch.nonzero(
                should_mask & (chosen_groups == group_number),
                as_tuple=True,
            )[0]

            count = selected_rows.numel()
            if count == 0:
                continue

            self._epoch_mask_counts[group_name] += count

            selected_rows = selected_rows.to(inputs.device)
            channel_indices = torch.tensor(
                SPECTRAL_GROUP_INDICES[group_name],
                device=inputs.device,
                dtype=torch.long,
            )

            output[
                selected_rows[:, None],
                channel_indices[None, :],
                :,
                :,
            ] = 0.0

        return output