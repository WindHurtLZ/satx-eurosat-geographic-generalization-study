"""Data import package: EuroSAT dataset reader and split management."""

from .dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    EuroSATDataset,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "EuroSATDataset",
]
