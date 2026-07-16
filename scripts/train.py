"""Run a SatX training job from a YAML or JSON config file.

Example:
    python scripts/train.py configs/rgb_spatial.yaml
"""

import argparse

from satx.engine import fit, load_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SatX EuroSAT model.")
    parser.add_argument("config", help="Path to a .yaml, .yml, or .json training config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_training_config(args.config)
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
