"""Regression tests for float64 statistics and explicitly specified spatial ties."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from saqre_phys import _cfar_background, ca_cfar, local_mean_std, scatterer_patches, to_intensity


def centers_from_boxes(boxes, half=12):
    return [(x1 - half if x0 == 0 else x0 + half,
             y1 - half if y0 == 0 else y0 + half) for x0, y0, x1, y1 in boxes]


def test_statistics_promote_before_square_and_match_direct_float64():
    gray = np.random.default_rng(11).random((61, 73)).astype(np.float32)
    mean, std = local_mean_std(gray, 9)
    windows = np.lib.stride_tricks.sliding_window_view(gray.astype(np.float64), (9, 9))
    assert mean.dtype == std.dtype == np.float64
    np.testing.assert_allclose(mean, windows.mean(axis=(-2, -1)), rtol=0, atol=1e-13)
    np.testing.assert_allclose(std, windows.std(axis=(-2, -1)), rtol=0, atol=1e-12)


def test_ring_background_matches_direct_float64_windows():
    gray = np.random.default_rng(17).random((47, 51)).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(gray.astype(np.float64), (17, 17))
    direct = (windows.sum(axis=(-2, -1)) - windows[..., 6:11, 6:11].sum(axis=(-2, -1))) / 264
    background = _cfar_background(gray)
    assert background.dtype == np.float64
    assert np.all(background >= 0)
    np.testing.assert_allclose(background, direct, rtol=0, atol=1e-13)


def test_large_dark_boundary_has_nonnegative_background_and_monotonic_cfar():
    gray = np.full((1024, 1024), np.float32(117 / 255), dtype=np.float32)
    gray[700:, 800:] = 0
    background = _cfar_background(gray)
    assert np.all(background >= 0)
    detections = [ca_cfar(gray, pfa_factor=factor) for factor in (2.0, 2.5, 3.0)]
    assert not (detections[1] & ~detections[0]).any()
    assert not (detections[2] & ~detections[1]).any()
    assert not detections[0][gray == 0].any()


def test_equal_intensity_center_then_maximin_ties():
    gray = np.zeros((65, 65), dtype=np.float32)
    for y in (8, 32, 56):
        for x in (8, 32, 56):
            gray[y, x] = 1
    detection = gray > 0
    centers = centers_from_boxes(scatterer_patches(gray, detection, max_sc=5))
    assert centers == [(32, 32), (8, 8), (56, 8), (8, 56), (56, 56)]
    assert centers == centers_from_boxes(scatterer_patches(gray, detection, max_sc=5))


def test_intensity_priority_remains_strict_and_suppression_is_unchanged():
    gray = np.zeros((81, 81), dtype=np.float32)
    for x, y, value in ((12, 12, 1.0), (40, 40, 0.9), (45, 45, 0.8), (60, 40, 0.8)):
        gray[y, x] = value
    centers = centers_from_boxes(scatterer_patches(gray, gray > 0))
    assert centers == [(12, 12), (40, 40), (60, 40)]
    for i, (x, y) in enumerate(centers):
        assert all(abs(x - cx) > 12 or abs(y - cy) > 12 for cx, cy in centers[:i])


def test_empty_and_clipped_fine_crops_preserve_contract():
    gray = np.zeros((48, 48), dtype=np.float32)
    assert scatterer_patches(gray, gray > 0) == []
    gray[8, 8] = 1
    assert scatterer_patches(gray, gray > 0) == [(0, 0, 20, 20)]


@pytest.mark.parametrize("image_id", ["USA_Utah_Centerfield_142.png", "USA_Texas_Garden-City_180.png"])
def test_real_negative_background_witness_images(image_id):
    path = ROOT / "data/SARimages_pre" / image_id
    if not path.exists():
        pytest.skip("Real-image regression requires the prepared dataset")
    with Image.open(path) as source:
        gray = to_intensity(source.convert("RGB"))
    assert np.all(_cfar_background(gray) >= 0)
    low, default, high = [ca_cfar(gray, pfa_factor=factor) for factor in (2.0, 2.5, 3.0)]
    assert not (default & ~low).any()
    assert not (high & ~default).any()
    assert not low[gray == 0].any()


def test_real_figure_one_spatial_ties_are_not_confined_to_top_strip():
    path = ROOT / "data/SARimages_pre/SN6_Train_AOI_11_Rotterdam_20190804111224_20190804111453_tile_8679.png"
    if not path.exists():
        pytest.skip("Figure regression requires the prepared dataset")
    with Image.open(path) as source:
        gray = to_intensity(source.convert("RGB"))
    detection = ca_cfar(gray)
    centers = centers_from_boxes(scatterer_patches(gray, detection))
    assert len(centers) == 8
    assert all(gray[y, x] == 1 for x, y in centers)
    assert max(y for _, y in centers) - min(y for _, y in centers) > gray.shape[0] / 2
    assert max(x for x, _ in centers) - min(x for x, _ in centers) > gray.shape[1] / 2
    ys, xs = np.where(detection & (gray == 1))
    nearest = np.argmin((xs - (gray.shape[1] - 1) / 2) ** 2 + (ys - (gray.shape[0] - 1) / 2) ** 2)
    assert centers[0] == (xs[nearest], ys[nearest])
