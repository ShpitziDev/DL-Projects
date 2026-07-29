from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase6_configuration_is_exactly_one_approved_condition() -> None:
    config = yaml.safe_load((ROOT / "configs/experiments/phase6_eth3d_smoke.yaml").read_text(encoding="utf-8"))
    assert config["input"]["scene"] == "delivery_area"
    assert config["input"]["frame_count"] == 2
    assert config["input"]["selection_strategy"] == "evenly_spaced"
    assert config["input"]["order"] == "original"
    assert config["constraints"]["expected_filenames"] == ["DSC_0675.JPG", "DSC_0718.JPG"]
    assert config["constraints"]["expected_forward_passes"] == 1
    assert not config["model"]["flash_sdp_enabled"]


def test_phase6_runner_is_offline_local_and_one_forward() -> None:
    source = (ROOT / "scripts/run_phase6_eth3d_smoke.py").read_text(encoding="utf-8")
    assert source.count("predictions = model(") == 1
    assert "from_pretrained(" not in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert "enable_flash_sdp(False)" in source
    assert "torch.cuda.is_available()" in source
    assert "load_scene(ETH3D_ROOT, \"delivery_area\")" in source
    assert "courtyard" not in source

