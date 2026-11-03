"""CPU checks for geometry, antialiasing, identity and stale-cache rejection."""
from copy import deepcopy
import unittest

from PIL import Image

from scale_contract import (EXTENSION, bind_signature, object_hash, target_size,
                            transform_image, verify_and_unwrap_signature,
                            verify_image_records)


class ScaleContractTests(unittest.TestCase):
    def setUp(self):
        self.base = {"schema_version": 2, "model": "ViT-B-32", "primitive_mode": "physics",
                     "source_sha256": {"extractor.py": "a" * 64}}
        self.helpers = {"scale_contract.py": "b" * 64}

    def test_rounding_and_actual_spacing_are_explicit(self):
        self.assertEqual(target_size(257, 259, "2"), (129, 130))
        self.assertEqual(target_size(256, 257, "1.5"), (171, 171))
        image, row = transform_image(Image.new("RGB", (257, 259)), "2")
        self.assertEqual(image.size, (129, 130))
        self.assertEqual(row["effective_spacing_ratio_xy"], [257 / 129, 259 / 130])
        self.assertNotEqual(row["effective_spacing_ratio_xy"], [2, 2])

    def test_identity_preserves_pixels_and_owns_its_storage(self):
        source = Image.new("RGB", (31, 47), (90, 20, 200))
        output, row = transform_image(source, "1")
        self.assertEqual(source.tobytes(), output.tobytes())
        self.assertEqual(row["source_decoded_rgb_sha256"], row["transformed_decoded_rgb_sha256"])
        output.putpixel((0, 0), (0, 0, 0))
        self.assertEqual(source.getpixel((0, 0)), (90, 20, 200))

    def test_downsample_reduces_grid_and_filters_checkerboard(self):
        image = Image.new("L", (64, 64))
        image.putdata([255 * ((x + y) % 2) for y in range(64) for x in range(64)])
        result, _ = transform_image(image, "2")
        self.assertEqual(result.size, (32, 32))
        center = list(result.crop((4, 4, 28, 28)).getdata())
        self.assertTrue(all(100 < rgb[0] < 155 for rgb in center))

    def test_undersized_image_rejected_not_padded_or_upsampled(self):
        with self.assertRaises(ValueError):
            transform_image(Image.new("RGB", (30, 30)), "2")

    def test_scale_signatures_are_distinct_and_reject_relabeling(self):
        signatures = [bind_signature(self.base, factor, self.helpers) for factor in ("1", "1.5", "2")]
        self.assertEqual(len({object_hash(x) for x in signatures}), 3)
        self.assertEqual(verify_and_unwrap_signature(signatures[1], "1.5", self.helpers), self.base)
        for wrong in ("1", "2"):
            with self.assertRaises(ValueError):
                verify_and_unwrap_signature(signatures[1], wrong, self.helpers)
        with self.assertRaises(ValueError):
            verify_and_unwrap_signature(self.base, "1", self.helpers)

    def test_base_or_helper_or_filter_mutation_rejected(self):
        signature = bind_signature(self.base, "1.5", self.helpers)
        mutated = deepcopy(signature)
        mutated["source_sha256"]["extractor.py"] = "c" * 64
        with self.assertRaises(ValueError):
            verify_and_unwrap_signature(mutated, "1.5", self.helpers)
        with self.assertRaises(ValueError):
            verify_and_unwrap_signature(signature, "1.5", {"scale_contract.py": "d" * 64})
        mutated = deepcopy(signature)
        mutated[EXTENSION]["transform"]["filter"] = "nearest"
        with self.assertRaises(ValueError):
            verify_and_unwrap_signature(mutated, "1.5", self.helpers)

    def test_manifest_preserves_ids_source_hashes_and_dimensions(self):
        _, row = transform_image(Image.new("RGB", (256, 256)), "1.5")
        row["source_file_sha256"] = "a" * 64
        records = {"example.png": row}
        hashes = {"example.png": "a" * 64}
        verify_image_records(["example.png"], records, "1.5", hashes)
        for wrong_ids in ([], ["example.png", "example.png"], ["different.png"]):
            with self.assertRaises(ValueError):
                verify_image_records(wrong_ids, records, "1.5", hashes)
        wrong = deepcopy(records)
        wrong["example.png"]["transformed_size_wh"] = [256, 256]
        with self.assertRaises(ValueError):
            verify_image_records(["example.png"], wrong, "1.5", hashes)
        wrong = deepcopy(records)
        wrong["example.png"]["physical_gsd_m"] = 0.5
        with self.assertRaises(ValueError):
            verify_image_records(["example.png"], wrong, "1.5", hashes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
