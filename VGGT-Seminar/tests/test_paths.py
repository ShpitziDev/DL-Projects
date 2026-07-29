from pathlib import Path

from vggt_seminar.config import load_config
from vggt_seminar.paths import ensure_project_directories, find_project_root


def test_root_detection() -> None:
    assert (find_project_root() / "pyproject.toml").is_file()


def test_path_creation(tmp_path: Path) -> None:
    created = ensure_project_directories(tmp_path)
    assert all(path.is_dir() for path in created.values())


def test_config_loading() -> None:
    config = load_config()
    assert config["device"] in {"auto", "cuda", "cpu"}
    assert config["seed"] == 42
