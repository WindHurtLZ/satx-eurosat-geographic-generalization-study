"""Training engine exports for SatX experiments."""

from .config import TrainingConfig, load_training_config

__all__ = [
    "TrainingConfig",
    "build_dataloaders",
    "build_noisy_dataloaders",
    "fit",
    "fit_noisy",
    "load_run_model",
    "load_training_config",
    "resolve_device",
    "save_checkpoint",
    "set_seed",
    "train_one_epoch",
    "validate",
]


def __getattr__(name):
    if name in {"build_dataloaders", "build_noisy_dataloaders"}:
        from . import data

        return getattr(data, name)
    if name in {
        "fit",
        "fit_noisy",
        "load_run_model",
        "resolve_device",
        "save_checkpoint",
        "set_seed",
        "train_one_epoch",
        "validate",
    }:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
