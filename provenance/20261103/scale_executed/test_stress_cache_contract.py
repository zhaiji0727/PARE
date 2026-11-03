"""Exercise adapter validation against the release's real cache validator, on CPU."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

from scale_contract import bind_signature, object_hash, transform_image
from stress_cache import helper_hashes, set_device_guard, validate_scale_cache


DEFAULT_CODE = Path(__file__).resolve().parents[1] / "pare_code_archive_20261008/PARE_Code_910d74d/code"
CODE = Path(os.environ.get("TEST_RELEASE_CODE_ROOT", DEFAULT_CODE))


@unittest.skipUnless((CODE / "artifacts.py").is_file(), "Set TEST_RELEASE_CODE_ROOT to a frozen release code directory")
class ActualValidatorAdapterTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("tested_formal_artifacts", CODE / "artifacts.py")
        self.artifacts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.artifacts)
        self.base = {"schema_version": self.artifacts.SCHEMA_VERSION, "model": "ViT-B-32",
                     "pretrained": "test", "primitive_mode": "physics", "max_regions": 6,
                     "max_sc": 8, "source_sha256": {"extractor": "e" * 64}}
        ids = ["example.png"]
        provenance = {"schema_version": self.artifacts.SCHEMA_VERSION,
                      "signature": self.base, "signature_sha256": object_hash(self.base),
                      "ids_sha256": object_hash(ids), "image_file_sha256": {ids[0]: "a" * 64}}
        self.reference = {"ids": ids, "D": 512, "Mc": 6, "Mf": 8, "provenance": provenance}
        self.transformed = deepcopy(self.reference)
        signature = bind_signature(self.base, "1.5", helper_hashes())
        self.transformed["provenance"].update(signature=signature, signature_sha256=object_hash(signature))
        _, row = transform_image(Image.new("RGB", (256, 256)), "1.5")
        row["source_file_sha256"] = "a" * 64
        records = {ids[0]: row}
        self.transformed["scale_test"] = {"factor": "1.5", "reference_cache_sha256": "b" * 64,
                                          "image_records": records, "image_records_sha256": object_hash(records)}
        self.module_patch = patch.dict(sys.modules, {"artifacts": self.artifacts})
        self.extract_patch = patch.object(self.artifacts, "extraction_signature", return_value=self.base)
        self.module_patch.start()
        self.extract_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(self.extract_patch.stop)

    def validate(self, transformed=None, factor="1.5", reference_hash="b" * 64):
        return validate_scale_cache(transformed or self.transformed, self.reference,
                                    reference_hash, factor, "ViT-B-32", "test")

    def test_generic_evaluator_rejects_scale_cache(self):
        with self.assertRaisesRegex(ValueError, "Stale/incompatible"):
            self.artifacts.validate_cache(self.transformed, "ViT-B-32", "test")

    def test_explicit_adapter_accepts_and_does_not_mutate_original_signature(self):
        before = deepcopy(self.transformed)
        view = self.validate()
        self.assertEqual(view["provenance"]["signature"], self.base)
        self.assertEqual(before, self.transformed)

    def test_wrong_scale_reference_or_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(factor="2")
        with self.assertRaises(ValueError):
            self.validate(reference_hash="c" * 64)
        wrong = deepcopy(self.transformed)
        wrong["scale_test"]["image_records"]["example.png"]["transformed_size_wh"] = [256, 256]
        with self.assertRaises(ValueError):
            self.validate(wrong)


class DeviceGuardTests(unittest.TestCase):
    def test_wrong_gpu_selection_is_rejected(self):
        for value in ("", "0", "0,1,2,3,4,5", "4", "6"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                with self.assertRaises(ValueError):
                    set_device_guard("cuda")

    def test_only_gpu5_is_allowed(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}):
            set_device_guard("cuda")
            self.assertEqual(os.environ["PARE_DEVICE"], "cuda:0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
