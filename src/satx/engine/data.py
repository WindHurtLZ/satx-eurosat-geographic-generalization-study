"""DataLoader construction helpers for EuroSAT training runs."""

from __future__ import annotations

from .config import TrainingConfig


def build_dataloaders(config: TrainingConfig):
    """Build train and validation DataLoaders from a training config."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Building SatX DataLoaders requires torch. "
            "Install it with `python -m pip install torch`."
        ) from exc

    from satx.data import EuroSATDataset

    train_dataset = EuroSATDataset(
        modality=config.modality,
        split_type=config.split_type,
        split="train",
        data_dir=config.data_dir,
        splits_dir=config.splits_dir,
    )
    val_dataset = EuroSATDataset(
        modality=config.modality,
        split_type=config.split_type,
        split="val",
        data_dir=config.data_dir,
        splits_dir=config.splits_dir,
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
