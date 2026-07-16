"""Reusable training loop for SatX EuroSAT classifiers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .data import build_dataloaders
from satx.models import build_resnet50


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Training SatX models requires torch. "
            "Install it with `python -m pip install torch torchvision`."
        ) from exc
    return torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch random generators."""
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto"):
    """Resolve a device string to a torch.device."""
    torch = _require_torch()
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_accuracy(logits, labels) -> tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.numel()


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> dict:
    """Train for one epoch and return loss/accuracy metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        correct, seen = _batch_accuracy(logits.detach(), labels)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_seen += seen

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


def validate(model, dataloader, criterion, device, return_predictions: bool = False) -> dict:
    """Evaluate on a validation DataLoader and return loss/accuracy metrics."""
    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            logits = model(images)
            preds = logits.argmax(dim=1)
            loss = criterion(logits, labels)

            batch_size = labels.numel()
            correct = (preds == labels).sum().item()
            seen = batch_size

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_seen += seen

            if return_predictions:
                all_y_true.extend(labels.detach().cpu().tolist())
                all_y_pred.extend(preds.detach().cpu().tolist())

    result = {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }

    if return_predictions:
        result["y_true"] = all_y_true
        result["y_pred"] = all_y_pred

    return result


def save_checkpoint(path: str | Path, model, optimizer, config: TrainingConfig, epoch: int, metrics: dict) -> None:
    """Save a training checkpoint with enough context to resume or audit a run."""
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.as_dict(),
            "metrics": metrics,
        },
        path,
    )


def _print_run_start(config: TrainingConfig, device) -> None:
    """Print a concise training run summary."""
    print(
        f"Starting run: {config.run_name}\n"
        f"Device: {device}\n"
        f"Epochs: {config.epochs}\n"
        f"Batch size: {config.batch_size}\n",
        flush=True,
    )


def _print_epoch_summary(
    epoch: int,
    config: TrainingConfig,
    train_metrics: dict,
    val_metrics: dict,
    best_val_accuracy: float,
) -> None:
    """Print train/validation metrics after one epoch."""
    print(
        f"Epoch {epoch:02d}/{config.epochs}\n"
        f"  train loss={train_metrics['loss']:.4f} "
        f"acc={train_metrics['accuracy']:.4f}\n"
        f"  val   loss={val_metrics['loss']:.4f} "
        f"acc={val_metrics['accuracy']:.4f}\n"
        f"  best val acc={best_val_accuracy:.4f}\n",
        flush=True,
    )


def fit(config: TrainingConfig) -> dict:
    """Run a complete train/validation loop from a TrainingConfig."""
    torch = _require_torch()
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(config)
    model = build_resnet50(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        pretrained=config.pretrained,
        input_mode=config.model_input_mode,
    ).to(device)

    _print_run_start(config, device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {
        "config": config.as_dict(),
        "device": str(device),
        "epochs": [],
        "best_epoch": None,
        "best_val_accuracy": None,
    }
    best_val_accuracy = -1.0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            return_predictions=(epoch == config.epochs),
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history["epochs"].append(epoch_record)

        save_checkpoint(run_dir / "last.pt", model, optimizer, config, epoch, epoch_record)
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            history["best_epoch"] = epoch
            history["best_val_accuracy"] = best_val_accuracy
            save_checkpoint(run_dir / "best.pt", model, optimizer, config, epoch, epoch_record)

        _print_epoch_summary(
            epoch,
            config,
            train_metrics,
            val_metrics,
            best_val_accuracy,
        )

        with open(run_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return history
