# SatX EuroSAT Geographic Generalization

SatX studies geographic generalization in EuroSAT land-cover classification. The project compares standard and spatial splits using RGB and 13-band multispectral inputs, with additional spectral-group dropout and robustness experiments.

## Setup

Create the environment:

```
conda env create -f environment.yml
conda activate satx
```

Install PyTorch, project dependencies, and JupyterLab:

```
python -m pip install torch torchvision torchgeo rasterio jupyterlab
python -m pip install -e .
```

Register the environment as a Jupyter kernel:

```
python -m ipykernel install --user --name satx --display-name "Python (satx)"
```

Check the installation:

```
python scripts/check_imports.py
```

### Dataset

Download `EuroSAT_MS.zip` from:

https://zenodo.org/records/7711810/files/EuroSAT_MS.zip

Extract it to:

```
data/
└── EuroSAT_MS/
    ├── AnnualCrop/
    ├── Forest/
    └── ...
```

The standard and spatial split files are already stored in `data/splits_data/`.

## Experiments

Start JupyterLab from the repository root:

```
jupyter lab
```

Open the notebooks under `notebooks/`:

- `SatX_Baseline_Exp.ipynb` — RGB and multispectral baselines
- `Overlap_Controlled_Cross_Eval_Exp.ipynb` — overlap-controlled cross-evaluation
- `Group_Dropout_Exp.ipynb` — spectral-group dropout and masked-band evaluation
- `Noise_Exp.ipynb` — noise robustness experiments

Run cells from top to bottom. Check the notebook run switches before enabling training or evaluation. Generated artifacts are written to `outputs/` and `results/`.

## PR Rule

Do not push directly to the **main** branch. Pull and create new branch:

```
git checkout main
git pull
git checkout -b w1-b/data-protocol
```

Commit and push:

```
git add .
git commit -m "what"
git push -u origin w1-b/data-protocol
```

Open a PR from branch into `main`

After the PR is merged, update `main` and delete your this task branch, create a new branch for new task:

```
git checkout main
git pull
git branch -d w1-b/new-devs
```
