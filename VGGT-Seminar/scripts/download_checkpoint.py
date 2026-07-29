from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "local_assets/checkpoints"
REPO_ID = "facebook/VGGT-1B"
REPO_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
FILENAME = "model.pt"
EXPECTED_SIZE = 5_026_874_952
EXPECTED_SHA256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
CODE_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / FILENAME
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {target}")
    downloaded = Path(hf_hub_download(
        repo_id=REPO_ID, filename=FILENAME, revision=REPO_REVISION,
        local_dir=DESTINATION,
    )).resolve()
    size = downloaded.stat().st_size
    checksum = sha256_file(downloaded)
    if size != EXPECTED_SIZE or checksum != EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint verification failed: size={size}, sha256={checksum}")
    manifest = {
        "schema_version": 1,
        "source": f"https://huggingface.co/{REPO_ID}/blob/{REPO_REVISION}/{FILENAME}",
        "repository": REPO_ID,
        "repository_revision": REPO_REVISION,
        "filename": FILENAME,
        "relative_path": str(downloaded.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": size,
        "sha256": checksum,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_code_commit": CODE_COMMIT,
        "authentication_required": False,
        "license": "CC-BY-NC-4.0 (original research checkpoint; non-commercial)",
    }
    (DESTINATION / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
