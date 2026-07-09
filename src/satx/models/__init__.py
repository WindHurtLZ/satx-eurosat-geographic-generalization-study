"""Model factories for SatX experiments."""

__all__ = ["build_resnet50"]


def __getattr__(name):
    if name == "build_resnet50":
        from .resnet import build_resnet50

        return build_resnet50
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
