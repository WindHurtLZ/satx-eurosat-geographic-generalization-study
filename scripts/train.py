"""Run a SatX training job from a config file or grid-search best config.

Examples:
    python scripts/train.py configs/rgb_spatial.yaml
    python scripts/train.py --modality rgb --split-type spatial
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satx.engine import TrainingConfig, fit, load_training_config
from satx.utils.paths import resolve_project_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SatX EuroSAT model.")
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to a .yaml, .yml, or .json training config.",
    )
    parser.add_argument("--modality", choices=["rgb", "ms"], default="rgb")
    parser.add_argument(
        "--split-type",
        choices=["standard", "spatial", "random"],
        default="spatial",
        help="Used to find outputs/grid/<modality>_<split_type>/best_config.json.",
    )
    parser.add_argument(
        "--best-config",
        help=(
            "Optional explicit path to best_config.json. Defaults to "
            "outputs/grid/<modality>_<split_type>/best_config.json when no "
            "positional config is provided."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Final training epochs when using best_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional final output directory when using best_config.json. "
            "Defaults to outputs/runs."
        ),
    )
    return parser.parse_args()


def load_best_config(args) -> TrainingConfig:
    grid_name = f"{args.modality}_{args.split_type}"
    best_config_path = resolve_project_path(
        args.best_config or f"outputs/grid/{grid_name}/best_config.json"
    )

    if not best_config_path.exists():
        raise FileNotFoundError(
            f"Could not find {best_config_path}. Run scripts/grid_search.py first "
            "for this modality/split, or pass a positional config file."
        )

    best = json.loads(best_config_path.read_text())
    output_dir = args.output_dir or "./outputs/runs"

    config = TrainingConfig(
        modality=best["modality"],
        split_type=best["split_type"],
        model_input_mode=best["model_input_mode"],
        pretrained=best["pretrained"],
        epochs=args.epochs,
        batch_size=best["batch_size"],
        num_workers=best["num_workers"],
        pin_memory=best["pin_memory"],
        learning_rate=best["learning_rate"],
        weight_decay=best["weight_decay"],
        seed=best["seed"],
        output_dir=output_dir,
    )

    print(
        "Using grid-search best config:\n"
        f"  source: {best_config_path}\n"
        f"  modality={config.modality}, split_type={config.split_type}\n"
        f"  learning_rate={config.learning_rate:g}, weight_decay={config.weight_decay:g}\n"
        f"  screening best macro_f1={best['best_val_macro_f1']:.4f} "
        f"at epoch {best['best_epoch']}\n"
    )
    return config


def main():
    args = parse_args()
    if args.config:
        config = load_training_config(args.config)
    else:
        config = load_best_config(args)

    history = fit(config)
    print(
        "Training complete. "
        f"Best epoch: {history['best_epoch']}, "
        f"best val macro_f1: {history['best_val_macro_f1']:.4f}, "
        f"val accuracy at best epoch: {history['best_val_accuracy']:.4f}"
    )
    print(f"Outputs: {config.run_dir()}")


if __name__ == "__main__":
    main()
