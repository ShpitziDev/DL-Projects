"""Validate a future input directory without copying or modifying source images."""
from __future__ import annotations

import argparse
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    images = sorted(p for p in args.directory.iterdir() if p.suffix.lower() in EXTENSIONS)
    print(f"Found {len(images)} supported images in {args.directory}")
    for image in images:
        print(image.name)
    return 0 if images else 1


if __name__ == "__main__":
    raise SystemExit(main())
