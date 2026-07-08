"""Visualization helpers for the EuroSAT dataset."""

import matplotlib.pyplot as plt
import numpy as np

from ..data import CLASS_NAMES


def rgb_to_display(img):
    """(3, H, W) tensor of raw reflectance -> (H, W, 3) array in [0, 1] for imshow.

    Reproduces the official EuroSAT RGB export (phelber/EuroSAT FAQ):
        gdal_translate -ot Byte -a_nodata 0 -scale 0 2750 1 255 -b 4 -b 3 -b 2 ...
    i.e. a linear stretch from reflectance [0, 2750] -> [0, 1], no gamma.
    """
    x = img.numpy().transpose(1, 2, 0)
    return np.clip(x / 2750.0, 0, 1)


def show_RGB(dataset, title):
    """Plot the first RGB sample of each land-cover class in a 2x5 grid."""
    # Labels come from dataset.samples, so we avoid decoding every image.
    first_idx = {}
    for idx, (_, lbl) in enumerate(dataset.samples):
        first_idx.setdefault(lbl, idx)

    fig, axes = plt.subplots(2, 5, figsize=(6, 3))
    for ax, lbl in zip(axes.ravel(), sorted(first_idx)):
        img, _ = dataset[first_idx[lbl]]
        ax.imshow(rgb_to_display(img))
        ax.set_title(CLASS_NAMES[lbl], fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()
