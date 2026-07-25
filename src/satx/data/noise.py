"""Input corruption transforms for EuroSAT robustness experiments.

These live in the package rather than in a notebook cell because DataLoader
workers pickle the transform. On macOS workers are started with ``spawn``, and
a class defined interactively in a notebook is not importable from the spawned
process, so ``num_workers > 0`` fails with a PicklingError.

Nothing here is wired into the default training path: ``build_noise`` returns
``None`` unless a corruption is explicitly requested.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from satx.data.transforms_stats import get_normalization_stats

NoiseKind = str


class GaussianNoise:
    """Additive ``N(0, sigma * band_std)`` noise, independent of the signal.

    ``sigma`` is expressed in units of each band's standard deviation rather
    than raw reflectance. EuroSAT band stds span 4.9 (B09) to 1157 (B12), so a
    single scalar sigma in reflectance units would perturb the dim bands
    hundreds of times harder than the bright ones.
    """

    def __init__(self, sigma: float, band_std: Sequence[float]) -> None:
        self.sigma = float(sigma)
        self.scale = torch.as_tensor(band_std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: Tensor) -> Tensor:
        return image + torch.randn_like(image) * (self.sigma * self.scale)

    def __repr__(self) -> str:
        return f"GaussianNoise(sigma={self.sigma:g})"


class ShotNoise:
    """Poisson shot noise: ``x' = Poisson(x / gain) * gain``.

    Variance tracks the signal, so no per-band scaling is required: relative
    noise is ``sqrt(gain / x)``, which hits the dim bands hardest by
    construction. This is the physically correct detector-noise model for
    optical sensors, where the dominant term is photon counting.
    """

    def __init__(self, gain: float) -> None:
        self.gain = float(gain)

    def __call__(self, image: Tensor) -> Tensor:
        # Poisson rates must be non-negative; L1C reflectance already is, but
        # clamp defensively so an odd tile cannot raise mid-epoch.
        return torch.poisson(image.clamp(min=0) / self.gain) * self.gain

    def __repr__(self) -> str:
        return f"ShotNoise(gain={self.gain:g})"


def build_noise(
    kind: NoiseKind,
    level: float,
    modality: str = "ms",
    split_type: str = "spatial",
):
    """Return a noise transform, or ``None`` when noise is off.

    ``None`` is meaningful: it is exactly the transform the clean baseline runs
    with, so ``kind="none"`` reproduces the default model rather than
    approximating it.

    Args:
        kind: ``"none"``, ``"gaussian"``, or ``"shot"``.
        level: sigma for Gaussian (in per-band std units), gain for shot.
        modality: ``"ms"`` or ``"rgb"``, used to look up per-band statistics.
        split_type: split whose statistics to use.
    """
    if kind == "none" or level <= 0:
        return None
    if kind == "gaussian":
        statistics = get_normalization_stats(split_type=split_type, modality=modality)
        return GaussianNoise(level, statistics.std)
    if kind == "shot":
        return ShotNoise(level)
    raise ValueError(
        f"unknown noise kind {kind!r}; expected 'none', 'gaussian', or 'shot'."
    )


def perturbation_ratio(
    kind: NoiseKind,
    level: float,
    modality: str = "ms",
    split_type: str = "spatial",
    n_samples: int = 64,
    seed: int = 42,
) -> float:
    """Mean ``|dx| / |x|`` over a few validation samples."""
    noise = build_noise(kind, level, modality, split_type)
    if noise is None:
        return 0.0
    torch.manual_seed(seed)
    from satx.data import EuroSATDataset

    dataset = EuroSATDataset(modality=modality, split_type=split_type, split="val")
    delta = signal = 0.0
    for index in range(n_samples):
        x, _ = dataset[index]
        delta += (noise(x) - x).abs().mean().item()
        signal += x.abs().mean().item()
    return delta / signal
