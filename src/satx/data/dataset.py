import os

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
        else:
            raise ValueError(
                f"Unknown split_type '{split_type}'. "
                "Expected one of: 'spatial', 'standard'."
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
