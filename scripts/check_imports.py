"""Check that the SatX package and its key dependencies are importable.

Run after setting up the environment:

    python scripts/check_imports.py
"""

import importlib


def check_satx_modules():
    """Import the SatX subpackages; raises on failure."""
    for name in (
        "satx",
        "satx.data",
        "satx.models",
        "satx.engine",
        "satx.eval",
        "satx.analysis",
        "satx.utils",
    ):
        importlib.import_module(name)
    print("All SatX imports passed.")


def check_dependencies():
    """Report installation status of key third-party dependencies."""
    print("\nDependency status:")
    all_ok = True
    for name in ("numpy", "pandas", "sklearn", "torch", "torchvision", "torchgeo"):
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [ OK ] {name:<12} {version}")
        except ImportError:
            all_ok = False
            print(f"  [MISS] {name:<12} not installed")

    # Report accelerator availability (CUDA on Linux/Windows, MPS on Apple Silicon).
    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif torch.backends.mps.is_available():
            device = "mps (Apple GPU)"
        else:
            device = "cpu"
        print(f"  [INFO] torch device  {device}")
    except ImportError:
        pass

    return all_ok


if __name__ == "__main__":
    check_satx_modules()
    ok = check_dependencies()
    if not ok:
        raise SystemExit("Some dependencies are missing; see status above.")
