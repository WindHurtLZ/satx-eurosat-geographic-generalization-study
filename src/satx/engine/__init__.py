"""Training engine exports for SatX experiments."""

from .config import TrainingConfig, load_training_config

__all__ = [
    "TrainingConfig",
    "build_dataloaders",
    "fit",
    "load_training_config",
    "resolve_device",
    "save_checkpoint",
    "set_seed",
    "train_one_epoch",
    "validate",
]


def __getattr__(name):
    if name == "build_dataloaders":
        from .data import build_dataloaders

        return build_dataloaders
    if name in {
        "fit",
        "resolve_device",
        "save_checkpoint",
        "set_seed",
        "train_one_epoch",
        "validate",
    }:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
