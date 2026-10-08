"""Noise-robustness evaluation helpers for EuroSAT models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from satx.data import EuroSATDataset
from satx.data.noise import build_noise, perturbation_ratio
from satx.engine.config import TrainingConfig
from satx.engine.data import seed_worker
from satx.engine.train import load_run_model, resolve_device, validate


def evaluate_noisy(
    model,
    config: TrainingConfig,
    kind: str,
    level: float,
    device,
    split: str = "val",
    seed: int = 42,
    num_workers: int = 4,
) -> dict:
    """One forward pass over *split* with the given corruption applied."""
    dataset = EuroSATDataset(
        modality=config.modality,
        split_type=config.split_type,
        split=split,
        transform=build_noise(kind, level, config.modality, config.split_type),
        data_dir=config.data_dir,
        splits_dir=config.splits_dir,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    return validate(
        model, loader, torch.nn.CrossEntropyLoss(), device, config.num_classes,
    )


def sweep_noise(
    run_dir,
    settings: Sequence[tuple[str, float]],
    split: str = "val",
    seed: int = 42,
    num_workers: int = 4,
) -> list[dict]:
    """Evaluate one checkpoint across every noise setting in *settings*."""
    device = resolve_device("auto")
    model, config = load_run_model(run_dir, device)
    rows: list[dict] = []
    for kind, level in settings:
        metrics = evaluate_noisy(
            model, config, kind, level, device,
            split=split, seed=seed, num_workers=num_workers,
        )
        rows.append({
            "noise": kind,
            "level": level,
            "perturbation": perturbation_ratio(
                kind, level, config.modality, config.split_type,
            ),
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
        })
        label = "clean" if kind == "none" else f"{kind} {level:g}"
        print(
            f"{label:<18} macro_f1={metrics['macro_f1']:.4f}"
            f"  acc={metrics['accuracy']:.4f}"
        )
    return rows
