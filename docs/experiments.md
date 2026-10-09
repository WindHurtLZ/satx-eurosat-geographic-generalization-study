# Experiments

This page lists every experiment in the report. For each one, it shows the configs, the result files and the command to run it again. To see how the code works, read [ARCHITECTURE.md](ARCHITECTURE.md).

Every run uses seed 42. Training saves checkpoints in `outputs/runs/<run name>/`, which is not in Git. The final result tables are in [`results/`](../results/). The four notebooks in [`notebooks/`](../notebooks/) are the original experiment logs, saved with their outputs. Keep them as a record. To rerun a step, the commands below are faster.

| Experiment | In the report | Configs | Results | Notebook |
| --- | --- | --- | --- | --- |
| [Baselines](#1-baselines) | Table 1 | `rgb_*`, `ms_standard`, `ms_spatial` | Printed in the notebook. Normalization check in [`ms_standard_normalization.csv`](../results/ms_standard_normalization.csv) | [`SatX_Baseline_Exp`](../notebooks/SatX_Baseline_Exp.ipynb) |
| [Spectral-group dropout](#2-spectral-group-dropout) | Figure 1, Table 2 | `ms_spatial_dropout_*` | [`results/drop_out_experiment/`](../results/drop_out_experiment/) | [`Group_Dropout_Exp`](../notebooks/Group_Dropout_Exp.ipynb) |
| [Overlap-controlled test](#3-overlap-controlled-test) | Figure 2 | The four baselines | [`results/overlap_controlled_class_level_cross_eval/`](../results/overlap_controlled_class_level_cross_eval/) | [`Overlap_Controlled_Cross_Eval_Exp`](../notebooks/Overlap_Controlled_Cross_Eval_Exp.ipynb) |
| [Noise robustness](#4-noise-robustness) | Table 3, Figure 3 | No config in Git | Printed in the notebook | [`Noise_Exp`](../notebooks/Noise_Exp.ipynb) |

## Before you start

1. Download [`EuroSAT_MS.zip`](https://zenodo.org/records/7711810/files/EuroSAT_MS.zip) from Zenodo.
2. Unzip it so the images sit at `data/EuroSAT_MS/<ClassName>/<ClassName>_<n>.tif`.
3. Run `python scripts/check_imports.py` to check your environment.

The split lists are already in [`data/splits_data/`](../data/splits_data/). They come from TorchGeo, through [`scripts/download_splits.py`](../scripts/download_splits.py).

## 1. Baselines

There are four ResNet-50 models: RGB and 13-band, each trained on the standard split and on the spatial split. All four train for 10 epochs with batch size 32, AdamW, ImageNet weights and per-band normalization.

| Config | Input | Split | Learning rate | Weight decay |
| --- | --- | --- | --- | --- |
| [`rgb_standard.yaml`](../configs/rgb_standard.yaml) | RGB | Standard | 1e-3 | 1e-4 |
| [`rgb_spatial.yaml`](../configs/rgb_spatial.yaml) | RGB | Spatial | 1e-3 | 1e-4 |
| [`ms_standard.yaml`](../configs/ms_standard.yaml) | 13 bands | Standard | 1e-4 | 1e-3 |
| [`ms_spatial.yaml`](../configs/ms_spatial.yaml) | 13 bands | Spatial | 1e-4 | 1e-3 |

```bash
python scripts/train.py configs/ms_spatial.yaml
```

The notebook then tests each model on the test set of its own split.

**Grid search (optional):** [`scripts/grid_search.py`](../scripts/grid_search.py) tries learning rates 1e-4, 3e-4 and 1e-3 with weight decays 1e-5, 1e-4 and 1e-3. Each try runs for 5 epochs. The best setting goes to `outputs/grid/<modality>_<split>/best_config.json`. Then `python scripts/train.py --modality ms --split-type spatial` trains with it.

**Normalization check:** [`ms_standard_normalization.csv`](../results/ms_standard_normalization.csv) tests the 13-band model on the standard split with and without normalization, and with and without ImageNet weights. With ImageNet weights, the best validation macro-F1 is 0.978 without normalization and 0.981 with it. Without ImageNet weights, it is 0.894 and 0.912.

## 2. Spectral-group dropout

There are seven 13-band runs on the spatial split. Each run trains for 20 epochs. The only difference is the dropout probability *p*: 0, 0.1, 0.3, 0.5, 0.7, 0.9 or 1.0. The configs go from [`ms_spatial_dropout_baseline.yaml`](../configs/ms_spatial_dropout_baseline.yaml) (*p* = 0) through [`ms_spatial_dropout_p10.yaml`](../configs/ms_spatial_dropout_p10.yaml) to [`ms_spatial_dropout_p100.yaml`](../configs/ms_spatial_dropout_p100.yaml). The dropout uses its own seed, `dropout_seed: 42`.

```bash
python scripts/train.py configs/ms_spatial_dropout_p50.yaml
```

Steps:

1. Train all seven runs. Keep the best validation macro-F1 of each run.
2. Pick *p* with the validation set only. *p* = 0.5 wins with 0.9498 at epoch 19.
3. Test the *p* = 0 model and the *p* = 0.5 model on the spatial test set. Test them on clean images, then with each band group set to zero by [`apply_fixed_group_mask`](../src/satx/eval/fixed_group_mask.py).

The *p* = 0 config and `ms_spatial.yaml` have the same run name, so they write to the same folder. The *p* = 0 run trains for 20 epochs instead of 10. In our run, the best epoch is the same for both: epoch 3.

| File | What is inside |
| --- | --- |
| [`w3_dropout_validation_summary.csv`](../results/drop_out_experiment/w3_dropout_validation_summary.csv) | Best epoch and validation scores for each *p* (Figure 1) |
| [`w3_dropout_epoch_history.csv`](../results/drop_out_experiment/w3_dropout_epoch_history.csv) | Loss and scores for every epoch, and how many samples lost each band group |
| [`masked_group_metrics.csv`](../results/drop_out_experiment/masked_group_metrics.csv) | Test accuracy and macro-F1 for *p* = 0 and *p* = 0.5, clean and with each group set to zero (Table 2) |
| [`masked_group_per_class_f1.csv`](../results/drop_out_experiment/masked_group_per_class_f1.csv) | The same test, with F1 for each class |

## 3. Overlap-controlled test

The standard split and the spatial split use the same 27,000 images. So an image in one split's test set can be in the other split's training set. That would not be a fair test.

To fix this, the script removes those images. It drops every standard test image that is in the spatial training or validation set. It drops every spatial test image that is in the standard training or validation set. Both test sets end up as the same clean set of images. All four baselines are tested on this set, with F1 for each class and macro-F1.

First, list the checkpoints in a CSV file, for example `outputs/summary/runs_manifest.csv`:

```csv
run_name,modality,train_split,input_mode,pretrained,seed,checkpoint_path
rgb_standard,rgb,standard,direct,true,42,outputs/runs/resnet50_rgb_standard_direct_pretrained_zscore_seed42/best.pt
rgb_spatial,rgb,spatial,direct,true,42,outputs/runs/resnet50_rgb_spatial_direct_pretrained_zscore_seed42/best.pt
ms_standard,ms,standard,direct,true,42,outputs/runs/resnet50_ms_standard_direct_pretrained_zscore_seed42/best.pt
ms_spatial,ms,spatial,direct,true,42,outputs/runs/resnet50_ms_spatial_direct_pretrained_zscore_seed42/best.pt
```

Then run:

```bash
python scripts/overlap_controlled_class_level_cross_eval.py \
  --manifest outputs/summary/runs_manifest.csv \
  --test-splits standard spatial
```

The results go to [`results/overlap_controlled_class_level_cross_eval/`](../results/overlap_controlled_class_level_cross_eval/). `class_level_f1_overlap_controlled_test_set.csv` has one row per class plus macro-F1. The `.png` file with the same name is Figure 2.

## 4. Noise robustness

A teammate ran this study. The code is in [`data/noise.py`](../src/satx/data/noise.py), [`eval/noise.py`](../src/satx/eval/noise.py) and `fit_noisy` in [`engine/train.py`](../src/satx/engine/train.py).

- **Model:** A 13-band model trained on the spatial split without normalization (`resnet50_ms_spatial_direct_pretrained_seed42`). Its clean test macro-F1 is 0.898. Its config is not in Git.
- **Noise:** Gaussian noise, scaled by each band's standard deviation (σ from 0.005 to 0.25). Shot noise, `Poisson(x / g) · g` (gain *g* from 0.01 to 10).
- **Phase 1:** Test the clean-trained model at every noise level.
- **Phase 2:** Train new models with noise on the training images only: Gaussian σ 0.05 and 0.1, shot *g* 0.1 and 1. Then test each one at every noise level.
- **Results:** Printed in the notebook (section 4.3, spatial test set) and in Table 3 of the report.

Two notes. Phase 2 in the notebook needs a `final_config` for the model above, and an older notebook defined it. Also, Gaussian noise during training is added after normalization but sized for raw values, so it only matches the reported runs when `normalization` is `none`.
