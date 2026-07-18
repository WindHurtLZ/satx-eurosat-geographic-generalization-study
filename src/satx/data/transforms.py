"""Build training and evaluation transform pipelines for EuroSAT."""

from __future__ import annotations
from collections.abc import Callable
from torch import Tensor
from satx.data.transforms_stats import get_normalization_stats
from torchvision.transforms import Compose

TensorTransform = Callable[[Tensor], Tensor]

def _build_normalization(
    *,
    modality: str,
    split_type: str,
    normalization: str,
) -> TensorTransform | None:

    if normalization == "none":
        return None

    if normalization != "zscore":
        raise ValueError(
            "normalization must be either 'none' or 'zscore'."
        )

    try:
        from torchvision.transforms import Normalize
    except ImportError as exc:
        raise ImportError(
            "Normalization requires torchvision."
        ) from exc

    statistics = get_normalization_stats(
        split_type=split_type,
        modality=modality,
    )

    return Normalize(
        mean=statistics.mean,
        std=statistics.std,
    )


def build_train_transform(
    *,
    modality: str,
    split_type: str,
    normalization: str,
) -> TensorTransform | None:

    operations: list[TensorTransform] = []

    # Future train augmentation belongs here.

    normalize = _build_normalization(
        modality=modality,
        split_type=split_type,
        normalization=normalization,
    )
    if normalize is not None:
        operations.append(normalize)

    return Compose(operations) if operations else None


def build_eval_transform(
    *,
    modality: str,
    split_type: str,
    normalization: str,
) -> TensorTransform | None:
    """Build deterministic sample-level transforms for validation/test."""
    operations: list[TensorTransform] = []

    # Future resize belongs here and in the train pipeline.

    normalize = _build_normalization(
        modality=modality,
        split_type=split_type,
        normalization=normalization,
    )
    if normalize is not None:
        operations.append(normalize)

    return Compose(operations) if operations else None