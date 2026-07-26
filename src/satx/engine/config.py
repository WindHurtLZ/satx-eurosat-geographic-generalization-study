"""Configuration objects for SatX training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from satx.utils.paths import resolve_project_path

Modality = Literal["rgb", "ms"]
SplitType = Literal["random", "spatial", "standard"]
ModelInputMode = Literal["direct", "adapter"]
NormalizationMode = Literal["none", "zscore"]


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable training configuration for a EuroSAT ResNet run."""

    modality: Modality = "rgb"
    split_type: SplitType = "spatial"
    data_dir: str = "./data"
    splits_dir: str = "./splits_data"
    output_dir: str = "./outputs/runs"
    model_input_mode: ModelInputMode = "direct"
    num_classes: int = 10
    pretrained: bool = False
    normalization: NormalizationMode = "none"
    group_dropout_p: float = 0.0
    dropout_seed: int = 42
    epochs: int = 5
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    pin_memory: bool = True

    def __post_init__(self) -> None:
        valid_modalities = {"rgb", "ms"}
        valid_split_types = {"random", "spatial", "standard"}
        valid_input_modes = {"direct", "adapter"}
        valid_normalization_modes = {"none", "zscore"}

        if self.modality not in valid_modalities:
            raise ValueError(f"modality must be one of {sorted(valid_modalities)}.")
        if self.split_type not in valid_split_types:
            raise ValueError(f"split_type must be one of {sorted(valid_split_types)}.")
        if self.model_input_mode not in valid_input_modes:
            raise ValueError(
                f"model_input_mode must be one of {sorted(valid_input_modes)}."
            )
        if self.normalization not in valid_normalization_modes:
            raise ValueError(
                f"normalization must be one of {sorted(valid_normalization_modes)}."
            )
        if not 0.0 <= self.group_dropout_p <= 1.0:
            raise ValueError("group_dropout_p must be between 0 and 1.")
        if self.group_dropout_p > 0.0 and self.modality != "ms":
            raise ValueError(
                "Spectral-group dropout is only supported for modality='ms'."
            )
        if self.group_dropout_p > 0.0 and self.normalization != "zscore":
            raise ValueError(
                "Spectral-group dropout requires normalization='zscore' "
            )
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

    @property
    def in_channels(self) -> int:
        """Return the channel count implied by the selected modality."""
        if self.modality == "rgb":
            return 3
        if self.modality == "ms":
            return 13
        raise ValueError("modality must be either 'rgb' or 'ms'.")

    @property
    def run_name(self) -> str:
        """Stable default run name used for outputs and checkpoints."""
        weights = "pretrained" if self.pretrained else "scratch"
        preprocessing = "" if self.normalization == "none" else f"_{self.normalization}"
        dropout = ""
        if self.group_dropout_p > 0.0:
            probability_tag = round(self.group_dropout_p * 100)
            dropout = (
                f"_drop_p{probability_tag}"
                f"_dseed{self.dropout_seed}"
            )
        return (
            f"resnet50_{self.modality}_{self.split_type}_"
            f"{self.model_input_mode}_{weights}"
            f"{preprocessing}_seed{self.seed}"
            f"{dropout}"
        )

    def as_dict(self) -> dict:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "TrainingConfig":
        """Create a config from a dict, rejecting unknown keys."""
        valid_names = {field.name for field in fields(cls)}
        unknown = set(values) - valid_names
        if unknown:
            unknown_text = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown TrainingConfig field(s): {unknown_text}")
        return cls(**values)

    def run_dir(self) -> Path:
        """Return this run's output directory."""
        output_root = resolve_project_path(self.output_dir)
        return output_root / self.run_name

    def evaluation_dir(self, split: str) -> Path:
        """
        Return the directory for one evaluation split
        """
        if split not in {"validation", "test"}:
            raise ValueError("split must be either 'validation' or 'test'.")

        return self.run_dir() / "evaluation" / split


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from a JSON or YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            values = json.load(f)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "Loading YAML training configs requires PyYAML. "
                    "Install it with `python -m pip install pyyaml`."
                ) from exc
            values = yaml.safe_load(f) or {}
        else:
            raise ValueError("Training config files must use .json, .yaml, or .yml.")

    if not isinstance(values, dict):
        raise ValueError("Training config file must contain a mapping/object.")
    return TrainingConfig.from_dict(values)
