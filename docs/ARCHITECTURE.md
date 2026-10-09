# Architecture

This page shows how the code is organized and what happens in one training run. To run an experiment, see [experiments.md](experiments.md).

## Repository layout

```text
configs/        One YAML file per training run
data/
  splits_data/  Train, validation and test lists for both splits
  EuroSAT_MS/   The dataset (download it yourself, not in Git)
docs/           Project docs and the README figures
notebooks/      The original experiment notebooks, with their outputs
outputs/        Checkpoints and training logs (not in Git)
report/         The final report (PDF)
results/        Result tables used in the report
scripts/        Command-line tools
src/satx/       The Python package
tests/          Unit tests
```

## The `satx` package

| Module | What it does |
| --- | --- |
| [`data/dataset.py`](../src/satx/data/dataset.py) | Reads a split list and loads each image. It returns all 13 bands, or only B04, B03 and B02 for RGB. |
| [`data/band_info.py`](../src/satx/data/band_info.py) | Defines the band order and the four band groups. It checks that the groups use all 13 bands exactly once. |
| [`data/transforms.py`](../src/satx/data/transforms.py), [`transforms_stats.py`](../src/satx/data/transforms_stats.py) | Normalizes each band with a fixed mean and standard deviation. Each split has its own numbers. |
| [`data/noise.py`](../src/satx/data/noise.py) | Adds Gaussian or shot noise to an image. Used in the noise study. |
| [`models/resnet.py`](../src/satx/models/resnet.py) | Builds a ResNet-50 with ImageNet weights. For 13 bands, it copies the red, green and blue filters to B04, B03 and B02. |
| [`models/spectral_group_dropout.py`](../src/satx/models/spectral_group_dropout.py) | During training, sets one random band group to zero for some samples. |
| [`engine/config.py`](../src/satx/engine/config.py) | Holds the settings of a run and checks that they are valid. It also builds the run name. |
| [`engine/data.py`](../src/satx/engine/data.py) | Builds the train, validation and test data loaders, with fixed seeds. |
| [`engine/train.py`](../src/satx/engine/train.py) | Runs training and validation. It keeps the checkpoint with the best validation macro-F1. |
| [`eval/evaluator.py`](../src/satx/eval/evaluator.py) | Saves test results as JSON and plots: accuracy, macro-F1, F1 per class and the confusion matrix. |
| [`eval/fixed_group_mask.py`](../src/satx/eval/fixed_group_mask.py) | Sets one chosen band group to zero for every test image. |
| [`eval/noise.py`](../src/satx/eval/noise.py) | Tests one checkpoint at many noise levels. |
| [`utils/paths.py`](../src/satx/utils/paths.py) | Finds paths from the repository root, so scripts and notebooks work from any folder. |
| [`visualization/eurosat_viz.py`](../src/satx/visualization/eurosat_viz.py) | Shows one RGB sample per class. |

## One training run

1. **Config:** The run starts from a YAML file in [`configs/`](../configs/). The run name comes from the settings, for example `resnet50_ms_spatial_direct_pretrained_zscore_seed42_drop_p50_dseed42`.
2. **Data:** The dataset reads the split list and normalizes each band. The data loaders use the seed from the config, so the same seed gives the same batches.
3. **Model:** The code builds a ResNet-50 with 3 or 13 input channels and 10 output classes.
4. **Training:** Training uses AdamW and cross-entropy loss. If `group_dropout_p` is above 0, every batch goes through spectral-group dropout before the model sees it. The dropout has its own seed and counts how often it drops each group.
5. **Selection:** After each epoch, the model runs on the validation set. The code keeps the checkpoint with the best macro-F1.
6. **Outputs:** Each run gets a folder `outputs/runs/<run name>/` with `best.pt`, `last.pt` and `metrics_history.json`.
7. **Testing:** Notebooks and scripts load `best.pt` and score it on the test set. They can test clean images, images with one band group set to zero, or noisy images. Final tables go to [`results/`](../results/).

## Same seed, same result

Training sets the seeds for Python, NumPy and PyTorch. It also turns on PyTorch's deterministic mode, seeds every data-loader worker and gives the dropout its own seed. All reported runs use seed 42.

## Tests

Run `pytest -q`. The tests check band groups, configs, transforms, the model builder, spectral-group dropout and path handling. They do not need the dataset. The model and dropout tests need PyTorch.
