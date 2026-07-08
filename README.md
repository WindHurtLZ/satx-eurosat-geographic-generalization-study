# SatX EuroSAT Geographic Generalization Study

This repository contains the SatX team project for studying geographic generalization in EuroSAT land-cover classification with RGB and multispectral inputs.

## Setup

Clone repo and create the conda environment:

```bash
conda env create -f environment.yml
conda activate satx
```

Install project as a Python package:

```bash
python -m pip install -e .
```

Run the check:

```bash
python scripts/check_imports.py
pytest -q
```

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
