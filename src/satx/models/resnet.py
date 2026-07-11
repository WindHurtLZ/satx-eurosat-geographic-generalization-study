"""ResNet model factories for EuroSAT RGB and multispectral experiments."""

from __future__ import annotations

from typing import Literal

ResNetInputMode = Literal["direct", "adapter"]


def _require_torchvision():
    try:
        import torch
        from torch import nn
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:
        raise ImportError(
            "Building SatX ResNet models requires torch and torchvision. "
            "Install them with `python -m pip install torch torchvision`."
        ) from exc
    return torch, nn, ResNet50_Weights, resnet50


def _copy_rgb_weights_to_extra_channels(new_conv, old_conv, in_channels):
    """Initialize non-RGB first-conv weights from pretrained RGB filters."""
    old_weight = old_conv.weight.data
    new_conv.weight.data[:, :3, :, :] = old_weight
    if in_channels > 3:
        mean_weight = old_weight.mean(dim=1, keepdim=True)
        new_conv.weight.data[:, 3:, :, :] = mean_weight.repeat(1, in_channels - 3, 1, 1)


def build_resnet50(
    *,
    num_classes: int = 10,
    in_channels: int = 3,
    pretrained: bool = False,
    input_mode: ResNetInputMode = "direct",
):
    """Build a ResNet-50 classifier for EuroSAT.

    Args:
        num_classes: Number of output land-cover classes.
        in_channels: Number of input image channels, usually 3 or 13.
        pretrained: If true, initialize the ResNet backbone with ImageNet weights.
        input_mode: ``"direct"`` replaces the first convolution to accept
            ``in_channels`` directly. ``"adapter"`` prepends a 1x1 projection
            from ``in_channels`` to RGB before a standard ResNet-50.
    """
    if in_channels < 1:
        raise ValueError("in_channels must be a positive integer.")
    if input_mode not in {"direct", "adapter"}:
        raise ValueError("input_mode must be either 'direct' or 'adapter'.")

    _torch, nn, ResNet50_Weights, resnet50 = _require_torchvision()
    weights = ResNet50_Weights.DEFAULT if pretrained else None

    if input_mode == "adapter":
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        if in_channels == 3:
            return backbone
        adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(adapter.weight, mode="fan_out", nonlinearity="relu")
        return nn.Sequential(adapter, backbone)

    model = resnet50(weights=weights)
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            _copy_rgb_weights_to_extra_channels(model.conv1, old_conv, in_channels)
        else:
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        if model.conv1.bias is not None:
            nn.init.zeros_(model.conv1.bias)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
