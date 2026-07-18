import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from satx.data.band_info import RGB_CHANNEL_INDICES
from satx.data.transforms_stats import (
    SPATIAL_MS_STATS,
    STANDARD_MS_STATS,
    get_normalization_stats,
)
from satx.data.transforms import (
    build_eval_transform,
    build_train_transform,
)


def test_standard_rgb_stats_match_rgb_channel_order():
    rgb_stats = get_normalization_stats(
        split_type="standard",
        modality="rgb",
    )

    expected_mean = tuple(STANDARD_MS_STATS.mean[index] for index in RGB_CHANNEL_INDICES)
    expected_std = tuple(STANDARD_MS_STATS.std[index] for index in RGB_CHANNEL_INDICES)

    assert rgb_stats.mean == expected_mean
    assert rgb_stats.std == expected_std


def test_none_normalization_returns_no_transforms():
    train_transform = build_train_transform(
        modality="ms",
        split_type="standard",
        normalization="none",
    )
    evaluation_transform = build_eval_transform(
        modality="ms",
        split_type="standard",
        normalization="none",
    )

    assert train_transform is None
    assert evaluation_transform is None


@pytest.mark.parametrize(
    ("split_type", "statistics"),
    [
        ("standard", STANDARD_MS_STATS),
        ("spatial", SPATIAL_MS_STATS),
    ],
)
def test_ms_zscore_normalization(split_type, statistics):
    train_transform = build_train_transform(
        split_type=split_type,
        modality="ms",
        normalization="zscore",
    )
    eval_transform = build_eval_transform(
        split_type=split_type,
        modality="ms",
        normalization="zscore",
    )

    mean = torch.tensor(statistics.mean).reshape(13, 1, 1)
    std = torch.tensor(statistics.std).reshape(13, 1, 1)
    image = mean + std
    expected = torch.ones_like(image)

    train_result = train_transform(image)
    eval_result = eval_transform(image)

    assert train_result.shape == image.shape
    assert eval_result.shape == image.shape
    assert torch.allclose(train_result, expected, atol=1e-5)
    assert torch.allclose(eval_result, expected, atol=1e-5)