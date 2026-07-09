import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def random_splits(data_dir="./data", splits_dir="./splits_data"):
    """Generates stratified random split files if they do not exist yet."""
    random_train_file = os.path.join(splits_dir, "eurosat-random-train.txt")
    if os.path.exists(random_train_file):
        return

    os.makedirs(splits_dir, exist_ok=True)
    class_groups = {name: [] for name in CLASS_NAMES}

    search_base = os.path.join(data_dir, "EuroSAT_MS")

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(search_base, class_name)
        if os.path.exists(class_dir):
            for f in os.listdir(class_dir):
                if f.endswith((".jpg", ".tif")):
                    base = os.path.splitext(f)[0]
                    class_groups[class_name].append(f"{base}.jpg")

    train_list, val_list, test_list = [], [], []
    for class_name, files in class_groups.items():
        files = sorted(
            files, key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x else x
        )
        rng = random.Random(42 + CLASS_TO_IDX[class_name])
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)

        train_list.extend(files[:n_train])
        val_list.extend(files[n_train : n_train + n_val])
        test_list.extend(files[n_train + n_val :])

    for split_name, lst in [
        ("train", train_list),
        ("val", val_list),
        ("test", test_list),
    ]:
        out_path = os.path.join(splits_dir, f"eurosat-random-{split_name}.txt")
        with open(out_path, "w") as f:
            for item in sorted(lst):
                f.write(f"{item}\n")


class EuroSATDataset(Dataset):
    """
    Simplified PyTorch Dataset reader for EuroSAT supporting 13-band multispectral and RGB modalities.
    """

    def __init__(
        self,
        modality="ms",
        split_type="spatial",
        split="train",
        transform=None,
        data_dir="./data",
        splits_dir="./splits_data",
    ):
        # This module lives in src/satx/data/, so the project root is three
        # directories up and the dataset lives in <root>/data.
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        if data_dir == "./data":
            data_dir = os.path.join(base_dir, "data")
        if splits_dir == "./splits_data":
            splits_dir = os.path.join(data_dir, "splits_data")

        self.modality = modality
        self.transform = transform
        self.data_dir = data_dir

        if split_type == "spatial":
            split_file = os.path.join(splits_dir, f"eurosat-spatial-{split}.txt")
        elif split_type == "standard":
            split_file = os.path.join(splits_dir, f"eurosat-{split}.txt")
        elif split_type == "random":
            random_splits(data_dir, splits_dir)
            split_file = os.path.join(splits_dir, f"eurosat-random-{split}.txt")
        else:
            raise ValueError(
                f"Unknown split_type '{split_type}'. "
                "Expected one of: 'spatial', 'standard', 'random'."
            )

        with open(split_file, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]

        self.samples = []
        for fname in filenames:
            class_name = fname.split("_")[0]
            base_name = os.path.splitext(fname)[0]
            label = CLASS_TO_IDX[class_name]

            # Both modalities read from the same multispectral .tif; RGB is
            # extracted from the relevant bands in __getitem__.
            img_path = os.path.join(
                self.data_dir, "EuroSAT_MS", class_name, f"{base_name}.tif"
            )

            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        with rasterio.open(img_path) as src:
            if self.modality == "ms":
                # All 13 Sentinel-2 bands -> (13, 64, 64)
                img = src.read().astype(np.float32)
            else:
                # RGB from B04 (red), B03 (green), B02 (blue) -> (3, 64, 64).
                # rasterio band indices are 1-based.
                img = src.read([4, 3, 2]).astype(np.float32)
        img_tensor = torch.from_numpy(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label
