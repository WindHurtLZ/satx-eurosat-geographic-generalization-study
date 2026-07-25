"""DataLoader construction helpers for EuroSAT training runs."""

from __future__ import annotations

import random
import numpy as np

from .config import TrainingConfig


def seed_worker(worker_id: int) -> None:
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloaders(config: TrainingConfig):
    """Build train and validation DataLoaders from a training config."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Building SatX DataLoaders requires torch. "
            "Install it with `python -m pip install torch`."
        ) from exc

    from satx.data import EuroSATDataset
    from satx.data.transforms import build_eval_transform, build_train_transform

    train_transform = build_train_transform(
        modality=config.modality,
        split_type=config.split_type,
        normalization=config.normalization,
    )
    evaluation_transform = build_eval_transform(
        modality=config.modality,
        split_type=config.split_type,
        normalization=config.normalization,
    )

    train_dataset = EuroSATDataset(
        modality=config.modality,
        split_type=config.split_type,
        split="train",
        transform=train_transform,
        data_dir=config.data_dir,
        splits_dir=config.splits_dir,
    )
    val_dataset = EuroSATDataset(
        modality=config.modality,
        split_type=config.split_type,
        split="val",
        transform=evaluation_transform,
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

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    val_generator = torch.Generator()
    val_generator.manual_seed(config.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_worker,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=val_generator,
        worker_init_fn=seed_worker,
        **loader_kwargs
    )

    return train_loader, val_loader


def build_noisy_dataloaders(config: TrainingConfig, kind: str, level: float):
    """Like ``build_dataloaders``, but composes noise onto the training transform only."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose

    from satx.data import EuroSATDataset
    from satx.data.noise import build_noise
    from satx.data.transforms import build_eval_transform, build_train_transform

    base_train = build_train_transform(
        modality=config.modality,
        split_type=config.split_type,
        normalization=config.normalization,
    )
    eval_transform = build_eval_transform(
        modality=config.modality,
        split_type=config.split_type,
        normalization=config.normalization,
    )
    noise = build_noise(kind, level, config.modality, config.split_type)

    if base_train is None:
        train_transform = noise
    elif noise is None:
        train_transform = base_train
    elif kind == "shot":
        train_transform = Compose([noise, base_train])
    else:
        train_transform = Compose([base_train, noise])

    datasets = {
        split: EuroSATDataset(
            modality=config.modality,
            split_type=config.split_type,
            split=split,
            transform=transform,
            data_dir=config.data_dir,
            splits_dir=config.splits_dir,
        )
        for split, transform in (("train", train_transform), ("val", eval_transform))
    }

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(config.seed + 1)

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_worker,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        datasets["val"],
        shuffle=False,
        generator=val_generator,
        worker_init_fn=seed_worker,
        **loader_kwargs,
    )
    return train_loader, val_loader
