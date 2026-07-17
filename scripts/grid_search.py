"""Run a small SatX hyperparameter grid search and save the best config."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satx.engine import TrainingConfig, fit
from satx.utils.paths import resolve_project_path


SCREEN_EPOCHS = 5
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
WEIGHT_DECAYS = [1e-5, 1e-4, 1e-3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SatX learning-rate/weight-decay grid search."
    )
    parser.add_argument("--modality", choices=["rgb", "ms"], default="rgb")
    parser.add_argument(
        "--split-type",
        choices=["standard", "spatial", "random"],
        default="spatial",
        help="Dataset split to use for train/validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=SCREEN_EPOCHS,
        help="Screening epochs per run.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-input-mode", choices=["direct", "adapter"], default="direct")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=LEARNING_RATES)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=WEIGHT_DECAYS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the grid search even when screen_summary.json already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    grid_name = f"{args.modality}_{args.split_type}"
    summary_path = resolve_project_path(f"outputs/grid/{grid_name}/screen_summary.json")
    best_config_path = resolve_project_path(f"outputs/grid/{grid_name}/best_config.json")

    if summary_path.exists() and not args.force:
        results = json.loads(summary_path.read_text())
    else:
        results = []
        for lr, wd in itertools.product(args.learning_rates, args.weight_decays):
            tag = f"lr{lr:g}_wd{wd:g}"

            config = TrainingConfig(
                modality=args.modality,
                split_type=args.split_type,
                model_input_mode=args.model_input_mode,
                pretrained=args.pretrained,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                learning_rate=lr,
                weight_decay=wd,
                seed=args.seed,
                output_dir=f"./outputs/grid/{grid_name}/{tag}",
            )

            started = time.time()
            history = fit(config)

            final_val = history["epochs"][-1]["val"]
            y_true, y_pred = final_val["y_true"], final_val["y_pred"]

            results.append(
                {
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "epochs": args.epochs,
                    "best_epoch": history["best_epoch"],
                    "selection_metric": history["selection_metric"],
                    "best_val_score": history["best_val_score"],
                    "best_val_macro_f1": history["best_val_macro_f1"],
                    "best_val_accuracy": history["best_val_accuracy"],
                    "final_accuracy": accuracy_score(y_true, y_pred),
                    "final_macro_f1": f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "minutes": round((time.time() - started) / 60, 1),
                    "run_dir": str(config.run_dir()),
                }
            )

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2))

    for r in results:
        tag = f"lr{r['learning_rate']:g}_wd{r['weight_decay']:g}"
        print(
            f"{tag:20s} best_macro_f1={r['best_val_macro_f1']:.4f}  "
            f"best_acc={r['best_val_accuracy']:.4f}  "
            f"final_macro_f1={r['final_macro_f1']:.4f}  "
            f"({r['minutes']:.1f} min)"
        )

    lookup = {
        (r["learning_rate"], r["weight_decay"]): r["best_val_macro_f1"]
        for r in results
    }
    print(f"\nbest macro-F1 within {results[0]['epochs']} screening epochs")
    print("lr \\ wd    " + "".join(f"{wd:<10g}" for wd in args.weight_decays))
    for lr in args.learning_rates:
        print(
            f"{lr:<10g} "
            + "".join(f"{lookup[(lr, wd)]:<10.4f}" for wd in args.weight_decays)
        )

    best = max(results, key=lambda r: r["best_val_macro_f1"])
    print(
        f"\nBest by validation macro-F1: lr={best['learning_rate']:g}, "
        f"wd={best['weight_decay']:g} -> F1={best['best_val_macro_f1']:.4f} "
        f"(acc={best['best_val_accuracy']:.4f}, epoch={best['best_epoch']})"
    )

    best_config = {
        "modality": args.modality,
        "split_type": args.split_type,
        "model_input_mode": args.model_input_mode,
        "pretrained": args.pretrained,
        "learning_rate": best["learning_rate"],
        "weight_decay": best["weight_decay"],
        "screen_epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "seed": args.seed,
        "selection_metric": "macro_f1",
        "best_epoch": best["best_epoch"],
        "best_val_macro_f1": best["best_val_macro_f1"],
        "best_val_accuracy": best["best_val_accuracy"],
        "summary_path": str(summary_path),
        "selected_run_dir": best["run_dir"],
    }
    best_config_path.write_text(json.dumps(best_config, indent=2))
    print(f"Best config: {best_config_path}")


if __name__ == "__main__":
    main()
