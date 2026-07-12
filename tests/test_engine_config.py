import json

import pytest

from satx.engine import TrainingConfig, load_training_config


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.modality == "rgb"
    assert config.in_channels == 3
    assert "resnet50_rgb_spatial_direct" in config.run_name
    assert config.run_dir().is_absolute()


def test_training_config_multispectral_channels():
    config = TrainingConfig(modality="ms")

    assert config.in_channels == 13
    assert config.as_dict()["modality"] == "ms"


def test_training_config_from_dict_rejects_unknown_fields():
    values = {
        "modality": "rgb",
        "split_type": "spatial",
        "not_a_field": True,
    }

    with pytest.raises(ValueError, match="not_a_field"):
        TrainingConfig.from_dict(values)


def test_training_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="modality"):
        TrainingConfig(modality="sar")

    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=0)


def test_load_training_config_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"modality": "ms", "epochs": 1}), encoding="utf-8")

    config = load_training_config(path)

    assert config.modality == "ms"
    assert config.epochs == 1
