"""Cross-check the header parser against TIFFs independently written by Pillow."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image, PngImagePlugin, TiffImagePlugin

from inspect_raster_spacing import read_header


class RasterSpacingTests(unittest.TestCase):
    def write_tiff(self, path, model=1, unit=9001, tie=True, matrix=False):
        info = TiffImagePlugin.ImageFileDirectory_v2()
        info[33550] = (0.5, 0.25, 0.0)
        info.tagtype[33550] = 12
        if tie:
            info[33922] = (0.0, 0.0, 0.0, 300000.0, 4000000.0, 0.0)
            info.tagtype[33922] = 12
        if matrix:
            info[34264] = tuple(float(i % 5 == 0) for i in range(16))
            info.tagtype[34264] = 12
        info[34735] = (1, 1, 0, 3, 1024, 0, 1, model, 3072, 0, 1, 32653,
                       3076, 0, 1, unit)
        info.tagtype[34735] = 3
        Image.new("RGB", (20, 24)).save(path, format="TIFF", tiffinfo=info)

    def test_projected_metre_tiff_with_png_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actually_tiff.png"
            self.write_tiff(path)
            record = read_header(path)
            self.assertEqual(record["format"], "TIFF")
            self.assertEqual(record["size_wh"], [20, 24])
            self.assertEqual(record["raster_spacing_m_xy"], [0.5, 0.25])

    def test_geographic_or_nonmetre_coordinates_are_not_assumed_metres(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.tif"
            for model, unit in ((2, 9001), (1, 9002)):
                self.write_tiff(path, model=model, unit=unit)
                self.assertIsNone(read_header(path)["raster_spacing_m_xy"])

    def test_missing_tie_or_conflicting_matrix_is_not_silently_interpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.tif"
            for kwargs in ({"tie": False}, {"matrix": True}):
                self.write_tiff(path, **kwargs)
                self.assertIsNone(read_header(path)["raster_spacing_m_xy"])

    def test_png_dimensions_and_text_are_not_treated_as_gsd(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.png"
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("GSD", "0.5")
            Image.new("RGB", (20, 24)).save(path, pnginfo=metadata)
            record = read_header(path)
            self.assertEqual(record["size_wh"], [20, 24])
            self.assertIsNone(record["raster_spacing_m_xy"])
            self.assertIn("tEXt", record["other_chunk_types"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
