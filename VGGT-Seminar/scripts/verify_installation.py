from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vggt_seminar.config import load_config  # noqa: E402
from vggt_seminar.external import checkout_status, load_vggt_pin  # noqa: E402
from vggt_seminar.paths import find_project_root  # noqa: E402


def main() -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    assert find_project_root() == ROOT
    config = load_config()
    print(f"Project root: {ROOT}")
    print(f"Configuration: device={config['device']}, precision={config['precision']}")
    installed = importlib.util.find_spec("vggt") is not None
    if not installed:
        raise RuntimeError("Official VGGT package is not importable")
    import torch
    import vggt
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    pin = load_vggt_pin(ROOT)
    status = checkout_status(ROOT)
    if not status["matches_pin"] or not status["clean"]:
        raise RuntimeError(f"External checkout does not match clean pin: {status}")
    checkpoint_dir = ROOT / config["checkpoint"]["directory"]
    checkpoint_files = [] if not checkpoint_dir.exists() else [
        str(path.relative_to(ROOT)) for path in checkpoint_dir.rglob("*") if path.is_file()
    ]
    print(f"Project package import: OK ({__import__('vggt_seminar').__version__})")
    print(f"Official VGGT package import: OK ({Path(vggt.__path__[0]).resolve()})")
    print(f"VGGT class import: OK ({VGGT.__module__})")
    print(f"Preprocessor import: OK ({load_and_preprocess_images.__module__})")
    print(f"Torch: {torch.__version__}; CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pinned checkout: {pin['commit']} (clean={status['clean']})")
    print(f"Checkpoint directory files: {len(checkpoint_files)} (absence is expected in Phase 2)")
    print("Hugging Face offline mode was forced during verification; no download could occur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
