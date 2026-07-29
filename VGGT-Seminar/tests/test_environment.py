from pathlib import Path
from types import SimpleNamespace

from vggt_seminar.environment import collect_environment


class NoCuda:
    __version__ = "test"
    version = SimpleNamespace(cuda=None)
    cuda = SimpleNamespace(is_available=lambda: False)


def test_report_without_torch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("vggt_seminar.environment.importlib.util.find_spec", lambda name: None)
    report = collect_environment(tmp_path)
    assert report["torch"]["installed"] is False
    assert report["warnings"]


def test_report_with_unavailable_cuda(tmp_path: Path) -> None:
    report = collect_environment(tmp_path, NoCuda())
    assert report["torch"]["cuda_available"] is False
    assert "GPU inference" in report["warnings"][0]


def test_import_safety() -> None:
    import vggt_seminar
    assert vggt_seminar.__version__


def test_cuda_validation_schema(tmp_path: Path) -> None:
    required = {"schema_version", "status", "torch_version", "cuda_available", "gpu_name",
                "compute_capability", "matrix_multiply_passed"}
    sample = {key: True for key in required}
    sample.update({"schema_version": 1, "status": "passed", "torch_version": "test",
                   "gpu_name": "test", "compute_capability": [12, 0]})
    path = tmp_path / "validation.json"
    import json
    path.write_text(json.dumps(sample), encoding="utf-8")
    assert required <= json.loads(path.read_text(encoding="utf-8")).keys()
