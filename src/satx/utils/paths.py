"""
Project path resolution utils
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Locate the repository root

    Returns:
        Absolute path to the repository root.
    """
    module_path = Path(__file__).resolve()

    for candidate in module_path.parents:
        has_project_file = (candidate / "pyproject.toml").is_file()
        has_package_source = (candidate / "src" / "satx").is_dir()

        if has_project_file and has_package_source:
            return candidate

    raise RuntimeError(
        "Could not locate the SatX project root. "
        "Install the project in editable mode with `python -m pip install -e .`."
    )


def resolve_project_path(path: str | Path) -> Path:
    """
    Interpreted Relative path from the repository root, not from the current working directory.
    Absolute paths are preserved.

    Args:
        path: Absolute path or repository-relative path.

    Returns:
        Absolute, normalized path.

    Examples:
        resolve_project_path("data")
        = <repo-root>/data

        resolve_project_path("/tmp/runs")
        = /tmp/runs
    """
    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    return (get_project_root() / candidate).resolve()