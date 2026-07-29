import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_manifest_contract_without_checkpoint() -> None:
    path = ROOT / "local_assets/checkpoints/checkpoint_manifest.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["size_bytes"] > 5_000_000_000
    assert len(data["sha256"]) == 64
    assert data["filename"] == "model.pt"


def test_phase3_runner_is_offline_and_single_forward() -> None:
    source = (ROOT / "scripts/run_phase3_first_inference.py").read_text(encoding="utf-8")
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert "from_pretrained(" not in source
    assert source.count("predictions = model(") == 1


def test_output_schema_contract() -> None:
    expected = {"pose_enc", "pose_enc_list", "depth", "depth_conf", "world_points",
                "world_points_conf", "track", "vis", "conf", "images"}
    model_source = ROOT / "external/vggt/vggt/models/vggt.py"
    if not model_source.is_file():
        pytest.skip("optional pinned VGGT checkout is intentionally not committed")
    source = model_source.read_text(encoding="utf-8")
    assert all(f'predictions["{key}"]' in source for key in expected)


def test_runtime_metadata_contract() -> None:
    source = (ROOT / "scripts/run_phase3_first_inference.py").read_text(encoding="utf-8")
    for key in ("checkpoint_load_seconds", "inference_seconds", "peak_gpu_memory_bytes",
                "cuda_verified", "image_count"):
        assert f'"{key}"' in source


def test_live_notebook_structure_has_one_forward_cell() -> None:
    notebook = json.loads((ROOT / "notebooks/01_live_vggt_inference_demo.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    # This teaching notebook may intentionally retain a user's interactive execution state.
    # Cleanliness is enforced only for the main experiment source notebook below.
    assert code.count("predictions = model(") == 1
    assert "from_pretrained(" not in code
    assert code.index('os.environ["HF_HUB_OFFLINE"]') < code.index("from vggt.models.vggt import VGGT")


def test_phase4_notebook_is_clean_gated_and_supports_phase5_eth3d() -> None:
    notebook = json.loads((ROOT / "notebooks/02_vggt_multi_input_experiments.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code = "\n".join("".join(cell.get("source", [])) for cell in cells if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None and cell.get("outputs", []) == [] for cell in cells if cell["cell_type"] == "code")
    assert "single_cartoon" not in code and "anime" not in code.lower()
    assert "APPROVE_INFERENCE = False" in code
    assert code.index("if not APPROVE_INFERENCE") < code.index("model=VGGT()")
    assert 'INPUT_SOURCE = "eth3d"' in code
    assert 'SCENE_NAME = "delivery_area"' in code
    assert "load_eth3d_scene(ETH3D_ROOT, SCENE_NAME)" in code
    assert "load_scene_manifest(scene_dir)" in code
    assert "build_experiment_configurations" in code
    assert "display(sheet)" in code
