from satx.engine import TrainingConfig
from satx.utils.paths import get_project_root, resolve_project_path


def test_project_root_contains_repository_markers():
    project_root = get_project_root()

    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "src" / "satx").is_dir()

def test_relative_path_uses_project_root():
    assert resolve_project_path("outputs/runs") == (
        get_project_root() / "outputs" / "runs"
    )

def test_training_run_directory_uses_project_outputs():
    """Training artifacts must always be stored under project outputs."""
    config = TrainingConfig()

    assert config.run_dir() == (
        get_project_root()
        / "outputs"
        / "runs"
        / config.run_name
    )

def test_validation_evaluation_belongs_to_run():
    """Evaluation artifacts must stay with their associated run."""
    config = TrainingConfig()

    assert config.evaluation_dir("validation") == (
        config.run_dir() / "evaluation" / "validation"
    )

def test_absolute_path_is_preserved(tmp_path):
    """Absolute paths must remain supported."""
    assert resolve_project_path(tmp_path) == tmp_path.resolve()