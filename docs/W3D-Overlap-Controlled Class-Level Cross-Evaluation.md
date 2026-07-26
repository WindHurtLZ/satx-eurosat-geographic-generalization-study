# Overlap-Controlled Class-Level Cross-Evaluation

This note describes the workflow for evaluating the four final checkpoints on a
clean common test subset.

## 1. Create A Run Manifest

Create a CSV file at `outputs/summary/runs_manifest.csv` with one row per
checkpoint:

```csv
run_name,modality,train_split,input_mode,pretrained,seed,checkpoint_path
rgb_standard,rgb,standard,direct,true,42,outputs/runs/resnet50_rgb_standard_direct_pretrained_seed42/best.pt
rgb_spatial,rgb,spatial,direct,true,42,outputs/runs/resnet50_rgb_spatial_direct_pretrained_seed42/best.pt
ms_standard,ms,standard,direct,true,42,outputs/runs/resnet50_ms_standard_direct_pretrained_zscore_seed42/best.pt
ms_spatial,ms,spatial,direct,true,42,outputs/runs/resnet50_ms_spatial_direct_pretrained_zscore_seed42/best.pt
```

Use the actual checkpoint paths from the final training runs.

## 2. Run The Evaluation

```bash
python scripts/overlap_controlled_class_level_cross_eval.py \
  --manifest outputs/summary/runs_manifest.csv \
  --test-splits standard spatial
```

## 3. Overlap Control

The original standard and spatial split protocols partition the same full
EuroSAT sample pool, so cross-protocol train/test overlap can occur. To avoid
using test samples that appeared in another protocol's training or validation
set, the script filters evaluation samples in memory.

For the standard test set, it removes samples that also appear in the spatial
training or validation sets. For the spatial test set, it removes samples that
also appear in the standard training or validation sets. Since both protocols
partition the same full dataset, these two filtered test sets reduce to the same
clean common test subset. The final output therefore contains one class-level
checkpoint comparison table and one plot.

## 4. Outputs

- `results/overlap_controlled_class_level_cross_eval/class_level_f1_overlap_controlled_test_set.csv`
- `results/overlap_controlled_class_level_cross_eval/class_level_f1_overlap_controlled_test_set.png`

The table compares the four checkpoints on the clean common test subset:

| Class | RGB, standard-trained | RGB, spatial-trained | MS, standard-trained | MS, spatial-trained |
|---|---:|---:|---:|---:|
| AnnualCrop | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| Macro-F1 | ... | ... | ... | ... |

The final `Macro-F1` row averages F1 across classes.
