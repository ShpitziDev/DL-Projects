from __future__ import annotations

from pathlib import Path


MARKERS = ("pyproject.toml", "configs/default.yaml")


def find_project_root(start: Path | str | None = None) -> Path:
    """Find the nearest ancestor containing the project markers."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in MARKERS):
            return candidate
    raise FileNotFoundError(f"Could not find project root from {current}")


def resolve_project_path(value: str | Path, root: Path | None = None) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root or find_project_root()) / path


def ensure_project_directories(root: Path | None = None) -> dict[str, Path]:
    base = root or find_project_root()
    names = ["data/sample_inputs", "data/custom_inputs", "outputs/environment",
             "outputs/logs", "outputs/predictions", "outputs/visualizations",
             "outputs/metrics", "outputs/reports", "report/assets"]
    result = {}
    for name in names:
        result[name] = base / name
        result[name].mkdir(parents=True, exist_ok=True)
    return result
