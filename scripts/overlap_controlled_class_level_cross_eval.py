"""Run overlap-controlled class-level cross-evaluation.

Examples:
    python scripts/overlap_controlled_class_level_cross_eval.py --manifest outputs/summary/runs_manifest.csv
    python scripts/overlap_controlled_class_level_cross_eval.py --manifest outputs/summary/runs_manifest.csv --test-splits standard spatial

The manifest must contain:
    run_name,modality,train_split,input_mode,pretrained,seed,checkpoint_path
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satx.engine import TrainingConfig, resolve_device
from satx.models import build_resnet50
from satx.utils.paths import resolve_project_path


CLASS_NAMES: list[str] = []
EuroSATDataset = None

REQUIRED_MANIFEST_COLUMNS = {
    "run_name",
    "modality",
    "train_split",
    "input_mode",
    "pretrained",
    "seed",
    "checkpoint_path",
}

SAME_TEST_TABLE_FIELDS = [
    "class_name",
    "rgb_standard_f1",
    "rgb_spatial_f1",
    "ms_standard_f1",
    "ms_spatial_f1",
]

RUN_ORDER = ["rgb_standard", "rgb_spatial", "ms_standard", "ms_spatial"]
RUN_LABELS = {
    "rgb_standard": "RGB, standard-trained",
    "rgb_spatial": "RGB, spatial-trained",
    "ms_standard": "MS, standard-trained",
    "ms_spatial": "MS, spatial-trained",
}
PLOT_CLASS_LABELS = {
    "AnnualCrop": "Annual Crop",
    "HerbaceousVegetation": "Herb. Veg.",
    "PermanentCrop": "Perm. Crop",
    "SeaLake": "Sea/Lake",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained checkpoints on an overlap-controlled clean common "
            "test set and write a class-level F1 table and plot."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV file listing checkpoints to evaluate.",
    )
    parser.add_argument(
        "--test-splits",
        nargs="+",
        choices=["standard", "spatial"],
        default=["standard", "spatial"],
        help="Test split types to evaluate each checkpoint on.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/overlap_controlled_class_level_cross_eval",
        help="Directory for class-level CSV/PNG outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


def _bool_from_text(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: str | Path) -> list[dict]:
    manifest_path = resolve_project_path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    missing = REQUIRED_MANIFEST_COLUMNS - set(rows[0])
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required column(s): {missing_text}")

    return rows


def load_checkpoint_config(row: dict, checkpoint: dict) -> TrainingConfig:
    if isinstance(checkpoint.get("config"), dict):
        base = TrainingConfig.from_dict(checkpoint["config"])
    else:
        base = TrainingConfig(
            modality=row["modality"],
            split_type=row["train_split"],
            model_input_mode=row["input_mode"],
            pretrained=_bool_from_text(row["pretrained"]),
            seed=int(row["seed"]),
        )

    return replace(
        base,
        modality=row["modality"],
        split_type=row["train_split"],
        model_input_mode=row["input_mode"],
        pretrained=_bool_from_text(row["pretrained"]),
        seed=int(row["seed"]),
    )


def split_file_path(config: TrainingConfig, split_type: str, split: str) -> Path:
    data_dir = resolve_project_path(config.data_dir)
    splits_dir = Path(config.splits_dir)
    if not splits_dir.is_absolute():
        splits_dir = data_dir / splits_dir

    if split_type == "spatial":
        filename = f"eurosat-spatial-{split}.txt"
    elif split_type == "standard":
        filename = f"eurosat-{split}.txt"
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    return splits_dir / filename


def read_split_names(config: TrainingConfig, split_type: str, split: str) -> set[str]:
    path = split_file_path(config, split_type, split)
    with path.open("r", encoding="utf-8") as f:
        return {Path(line.strip()).stem for line in f if line.strip()}


def sample_name_from_path(sample_path: str) -> str:
    return Path(sample_path).stem


def overlap_control_excluded_names(config: TrainingConfig, test_split: str) -> set[str]:
    if test_split == "standard":
        return read_split_names(config, "spatial", "train") | read_split_names(config, "spatial", "val")
    if test_split == "spatial":
        return read_split_names(config, "standard", "train") | read_split_names(config, "standard", "val")
    return set()


def apply_overlap_control_filter(dataset, config: TrainingConfig, test_split: str) -> None:
    excluded_names = overlap_control_excluded_names(config, test_split)
    if not excluded_names:
        print(f"No overlap-control filter defined for {test_split} test; using full test set.")
        return

    original_count = len(dataset.samples)
    dataset.samples = [
        sample for sample in dataset.samples if sample_name_from_path(sample[0]) not in excluded_names
    ]
    kept_count = len(dataset.samples)
    print(
        f"Overlap-controlled {test_split} test: kept {kept_count}/{original_count} "
        f"samples; removed {original_count - kept_count} overlapping samples."
    )


def build_test_loader(
    config: TrainingConfig,
    test_split: str,
    batch_size: int,
    num_workers: int,
    *,
    overlap_control: bool = False,
):
    import torch
    from torch.utils.data import DataLoader

    if EuroSATDataset is None:
        raise RuntimeError("EuroSATDataset was not initialized.")

    from satx.data.transforms import build_eval_transform
    from satx.engine.data import seed_worker

    evaluation_transform = build_eval_transform(
        modality=config.modality,
        split_type=config.split_type,
        normalization=config.normalization,
    )

    dataset = EuroSATDataset(
        modality=config.modality,
        split_type=test_split,
        split="test",
        transform=evaluation_transform,
        data_dir=config.data_dir,
        splits_dir=config.splits_dir,
    )
    if overlap_control:
        apply_overlap_control_filter(dataset, config, test_split)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": config.pin_memory,
        "shuffle": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    test_generator = torch.Generator()
    test_generator.manual_seed(config.seed + 2)
    loader_kwargs["generator"] = test_generator
    loader_kwargs["worker_init_fn"] = seed_worker
    return DataLoader(dataset, **loader_kwargs)


def per_class_f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    scores = []
    for class_idx in range(matrix.shape[0]):
        tp = matrix[class_idx, class_idx]
        fp = matrix[:, class_idx].sum() - tp
        fn = matrix[class_idx, :].sum() - tp
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else float((2 * tp) / denom))
    return np.asarray(scores, dtype=np.float64)


def evaluate_model(model, dataloader, device, num_classes: int) -> dict:
    import torch

    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True).float()
            logits = model(images)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            true_labels = labels.detach().cpu().long().numpy()
            np.add.at(matrix, (true_labels, preds), 1)

    per_class_f1 = per_class_f1_from_confusion(matrix)

    return {
        "per_class_f1": per_class_f1,
    }


def load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_test_class_level_table(per_class_rows: list[dict], test_split: str) -> list[dict]:
    """Build one test split's class-level F1 table across all checkpoints plus macro-F1."""
    by_class_run: dict[str, dict[str, float]] = {}
    for row in per_class_rows:
        if row["test_split"] != test_split:
            continue
        by_class_run.setdefault(row["class_name"], {})[row["run_name"]] = float(row["f1"])

    table_rows = []
    for class_name in CLASS_NAMES:
        scores = by_class_run.get(class_name, {})
        if not scores:
            continue
        table_rows.append(
            {
                "class_name": class_name,
                "rgb_standard_f1": _format_optional_score(scores.get("rgb_standard")),
                "rgb_spatial_f1": _format_optional_score(scores.get("rgb_spatial")),
                "ms_standard_f1": _format_optional_score(scores.get("ms_standard")),
                "ms_spatial_f1": _format_optional_score(scores.get("ms_spatial")),
            }
        )

    if table_rows:
        total_row = {"class_name": "Macro-F1"}
        for field in SAME_TEST_TABLE_FIELDS[1:]:
            values = [float(row[field]) for row in table_rows if row[field]]
            total_row[field] = "" if not values else f"{float(np.mean(values)):.6f}"
        table_rows.append(total_row)

    return table_rows


def _format_optional_score(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def plot_clean_common_test_class_level_f1(table_rows: list[dict], output_dir: Path) -> None:
    """Plot class-level F1 across checkpoints on the clean common test set."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping clean common test plot.")
        return

    if not table_rows:
        print("No class-level rows found for clean common test; skipping plot.")
        return

    labels = [PLOT_CLASS_LABELS.get(row["class_name"], row["class_name"]) for row in table_rows]
    series = {
        "rgb_standard": np.array(
            [float(row["rgb_standard_f1"]) if row["rgb_standard_f1"] else np.nan for row in table_rows],
            dtype=np.float64,
        ),
        "rgb_spatial": np.array(
            [float(row["rgb_spatial_f1"]) if row["rgb_spatial_f1"] else np.nan for row in table_rows],
            dtype=np.float64,
        ),
        "ms_standard": np.array(
            [float(row["ms_standard_f1"]) if row["ms_standard_f1"] else np.nan for row in table_rows],
            dtype=np.float64,
        ),
        "ms_spatial": np.array(
            [float(row["ms_spatial_f1"]) if row["ms_spatial_f1"] else np.nan for row in table_rows],
            dtype=np.float64,
        ),
    }

    x = np.arange(len(table_rows))
    width = 0.2
    offsets = {
        "rgb_standard": -1.5 * width,
        "rgb_spatial": -0.5 * width,
        "ms_standard": 0.5 * width,
        "ms_spatial": 1.5 * width,
    }

    fig, ax = plt.subplots(figsize=(13, 6.2))
    for run_name in RUN_ORDER:
        ax.bar(x + offsets[run_name], series[run_name], width, label=RUN_LABELS[run_name])

    ax.axvline(len(labels) - 1.5, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Per-class F1")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_title("Class-level F1 on the overlap-controlled test set", pad=30)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "class_level_f1_overlap_controlled_test_set.png", dpi=300)
    plt.close(fig)


def evaluate_checkpoint_across_splits(
    *,
    row: dict,
    test_splits: list[str],
    device,
    batch_size: int,
    num_workers: int,
    overlap_control: bool = False,
) -> list[dict]:
    """Evaluate one checkpoint on all requested test splits."""
    checkpoint_path = resolve_project_path(row["checkpoint_path"])
    checkpoint = load_checkpoint(checkpoint_path)
    config = load_checkpoint_config(row, checkpoint)

    model = build_resnet50(
        num_classes=config.num_classes,
        in_channels=config.in_channels,
        pretrained=False,
        input_mode=config.model_input_mode,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    per_class_rows: list[dict] = []

    for test_split in test_splits:
        print(f"Evaluating {row['run_name']} on {test_split} test...")
        loader = build_test_loader(
            config,
            test_split,
            batch_size,
            num_workers,
            overlap_control=overlap_control,
        )
        result = evaluate_model(model, loader, device, config.num_classes)

        for class_name, f1_score in zip(CLASS_NAMES, result["per_class_f1"]):
            per_class_rows.append(
                {
                    "run_name": row["run_name"],
                    "modality": row["modality"],
                    "train_split": row["train_split"],
                    "test_split": test_split,
                    "class_name": class_name,
                    "f1": f"{float(f1_score):.6f}",
                }
            )

    return per_class_rows


def run_cross_evaluation(
    *,
    manifest_rows: list[dict],
    test_splits: list[str],
    device,
    batch_size: int,
    num_workers: int,
    overlap_control: bool = False,
) -> list[dict]:
    """Run checkpoint x test-split evaluation for the full manifest."""
    per_class_rows: list[dict] = []

    for row in manifest_rows:
        checkpoint_per_class = evaluate_checkpoint_across_splits(
            row=row,
            test_splits=test_splits,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            overlap_control=overlap_control,
        )
        per_class_rows.extend(checkpoint_per_class)

    return per_class_rows


def write_clean_common_test_outputs(
    *,
    per_class_rows: list[dict],
    output_dir: Path,
) -> None:
    """Write the overlap-controlled clean common test table and plot."""
    test_splits = sorted({row["test_split"] for row in per_class_rows})
    if not test_splits:
        print("No overlap-controlled rows found; skipping clean common test outputs.")
        return

    reference_split = "standard" if "standard" in test_splits else test_splits[0]
    table_rows = build_test_class_level_table(per_class_rows, reference_split)
    write_csv(
        output_dir / "class_level_f1_overlap_controlled_test_set.csv",
        table_rows,
        SAME_TEST_TABLE_FIELDS,
    )
    plot_clean_common_test_class_level_f1(table_rows, output_dir)


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global CLASS_NAMES, EuroSATDataset
    from satx.data import CLASS_NAMES as dataset_class_names
    from satx.data import EuroSATDataset as dataset_cls

    CLASS_NAMES = dataset_class_names
    EuroSATDataset = dataset_cls

    rows = read_manifest(args.manifest)
    device = resolve_device(args.device)

    print(f"Loaded {len(rows)} checkpoint(s) from {resolve_project_path(args.manifest)}")
    print(f"Device: {device}")
    print(f"Test splits: {', '.join(args.test_splits)}")

    print("\nRunning overlap-controlled class-level cross-evaluation...")
    per_class_rows = run_cross_evaluation(
        manifest_rows=rows,
        test_splits=args.test_splits,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overlap_control=True,
    )

    write_clean_common_test_outputs(
        per_class_rows=per_class_rows,
        output_dir=output_dir,
    )
    print(f"Overlap-controlled class-level cross-evaluation outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
