import json
from pathlib import Path

from vggt_seminar.external import checkout_status, load_vggt_pin


def _root(tmp_path: Path, commit: str) -> Path:
    (tmp_path / "external").mkdir()
    (tmp_path / "external/VGGT_PIN.json").write_text(json.dumps({
        "commit": commit, "checkout_path": "external/vggt"
    }), encoding="utf-8")
    return tmp_path


def test_parse_pin(tmp_path: Path) -> None:
    commit = "a" * 40
    assert load_vggt_pin(_root(tmp_path, commit))["commit"] == commit


def test_missing_checkout(tmp_path: Path) -> None:
    status = checkout_status(_root(tmp_path, "a" * 40))
    assert status == {"exists": False, "matches_pin": False, "clean": None, "commit": None}


def test_wrong_commit_detection(monkeypatch, tmp_path: Path) -> None:
    root = _root(tmp_path, "a" * 40)
    (root / "external/vggt/.git").mkdir(parents=True)
    results = iter([type("R", (), {"stdout": "b" * 40 + "\n"})(), type("R", (), {"stdout": ""})()])
    monkeypatch.setattr("vggt_seminar.external.subprocess.run", lambda *a, **k: next(results))
    status = checkout_status(root)
    assert status["exists"] and not status["matches_pin"] and status["clean"]


def test_dependency_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    inference = (root / "requirements/vggt-inference.txt").read_text(encoding="utf-8")
    pytorch = (root / "requirements/pytorch-cu130.txt").read_text(encoding="utf-8")
    assert "numpy==1.26.4" in inference
    assert "torch==2.13.0" in pytorch and "torchvision==0.28.0" in pytorch


def test_verifier_forces_offline_import() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/verify_installation.py").read_text(encoding="utf-8")
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert "from_pretrained(" not in source
