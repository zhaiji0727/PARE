import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from saqre_phys import ca_cfar, estimate_enl, extract_primitives, local_mean_std, to_intensity


class PhysicsTests(unittest.TestCase):
    def test_local_statistics_match_direct_windows(self):
        values = np.random.default_rng(3).random((32, 32)).astype(np.float32)
        mean, std = local_mean_std(values, 9)
        windows = np.lib.stride_tricks.sliding_window_view(values, (9, 9))
        np.testing.assert_allclose(mean, windows.mean(axis=(-2, -1)), atol=2e-6)
        np.testing.assert_allclose(std, windows.std(axis=(-2, -1)), atol=5e-6)

    def test_uniform_tile_and_isolated_scatterer(self):
        values = np.full((64, 64), 64, dtype=np.uint8)
        g = to_intensity(Image.fromarray(values))
        self.assertFalse(ca_cfar(g).any())
        self.assertAlmostEqual(estimate_enl(g), 64.0)
        values[32, 32] = 255
        g = to_intensity(Image.fromarray(values))
        self.assertTrue(ca_cfar(g)[32, 32])
        primitives = extract_primitives(Image.fromarray(values))
        self.assertTrue(primitives["scatterers"])
        for crop, features in primitives["regions"] + primitives["scatterers"]:
            self.assertGreater(min(crop.size), 0)
            self.assertTrue(np.isfinite(features["typefeat"]).all())
        self.assertGreaterEqual(primitives["L"], 1)
        self.assertLessEqual(primitives["L"], 64)


if __name__ == "__main__":
    unittest.main()
