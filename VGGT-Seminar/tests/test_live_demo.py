from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vggt_seminar.live_demo import (
    center_or_explicit_query,
    discover_images,
    heatmap_rgb,
    point_cloud_preview,
    prediction_schema,
)


def test_discover_images_filters_and_limits(tmp_path: Path) -> None:
    for name in ("b.png", "a.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [path.name for path in discover_images(tmp_path, "*", 1)] == ["a.jpg"]


def test_visual_helpers_need_no_optional_plotting_package() -> None:
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert heatmap_rgb(tensor).shape == (4, 4, 3)
    points = np.stack(np.meshgrid(np.arange(4), np.arange(4), [1.0]), axis=-1).reshape(-1, 3)
    colors = np.ones_like(points, dtype=np.float32) * 0.5
    preview = point_cloud_preview(points, colors, size=64)
    assert isinstance(preview, Image.Image) and preview.size == (64, 64)


def test_prediction_schema_and_query_validation() -> None:
    schema = prediction_schema({"depth": torch.ones(1, 2, 3)})
    assert schema["outputs"]["depth"]["finite"] is True
    query = center_or_explicit_query(20, 10, None, torch.device("cpu"))
    assert query.tolist() == [[[10.0, 5.0]]]
    with pytest.raises(ValueError):
        center_or_explicit_query(20, 10, (25, 5), torch.device("cpu"))
