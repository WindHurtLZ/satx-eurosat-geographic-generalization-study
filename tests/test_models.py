import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from satx.models import build_resnet50


def test_build_rgb_resnet50_forward_shape():
    model = build_resnet50(in_channels=3, num_classes=10, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 64, 64))

    assert logits.shape == (2, 10)


def test_build_multispectral_resnet50_forward_shape():
    model = build_resnet50(in_channels=13, num_classes=10, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(2, 13, 64, 64))

    assert logits.shape == (2, 10)


def test_build_multispectral_adapter_forward_shape():
    model = build_resnet50(
        in_channels=13,
        num_classes=10,
        pretrained=False,
        input_mode="adapter",
    )
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(2, 13, 64, 64))

    assert logits.shape == (2, 10)
