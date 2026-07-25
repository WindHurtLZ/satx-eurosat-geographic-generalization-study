import pytest
torch = pytest.importorskip("torch")
from satx.data.band_info import SPECTRAL_GROUP_INDICES
from satx.models.spectral_group_dropout import SpectralGroupDropout


def test_zero_changes_nothing():
    inputs = torch.ones(8, 13, 4, 4)
    dropout = SpectralGroupDropout(p=0.0, seed=42)
    result = dropout(inputs)

    assert torch.equal(result, inputs)


def test_eval_mode_changes_nothing():
    inputs = torch.ones(8, 13, 4, 4)
    dropout = SpectralGroupDropout(p=1.0, seed=42)
    dropout.eval()
    result = dropout(inputs)

    assert torch.equal(result, inputs)


def test_same_seed_produces_same_masks():
    inputs = torch.ones(64, 13, 4, 4)
    first = SpectralGroupDropout(p=0.3, seed=42)
    second = SpectralGroupDropout(p=0.3, seed=42)

    assert torch.equal(first(inputs), second(inputs))
