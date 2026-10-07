import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import primitive_controls


def test_uniform_control_never_calls_physical_extractor(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Physics extractor was called")
    monkeypatch.setattr(primitive_controls, "extract_primitives", forbidden)
    pixels = np.random.default_rng(0).integers(0, 256, (59, 83, 3), dtype=np.uint8)
    result = primitive_controls.extract_for_mode(Image.fromarray(pixels), mode="uniform_grid")
    assert len(result["regions"]) == 6
    assert len(result["scatterers"]) == 8
    for key, columns in (("regions", 3), ("scatterers", 4)):
        boxes = primitive_controls.grid_boxes(83, 59, 2, columns)
        covered = np.zeros((59, 83), dtype=int)
        for (crop, features), (x0, y0, x1, y1) in zip(result[key], boxes):
            np.testing.assert_array_equal(crop, pixels[y0:y1, x0:x1])
            np.testing.assert_array_equal(features["typefeat"], np.zeros(4))
            covered[y0:y1, x0:x1] += 1
        assert np.all(covered == 1)
    assert result["L"] == result["nu"] == 0


def test_uniform_control_rejects_undefined_grid_or_mode():
    image = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError):
        primitive_controls.extract_for_mode(image, max_regions=5, mode="uniform_grid")
    with pytest.raises(ValueError):
        primitive_controls.extract_for_mode(image, mode="typo")
