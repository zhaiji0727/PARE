"""No-GPU regression for the completion/hash handoff gate."""
import json
from pathlib import Path
import tempfile
import unittest

import evaluation_controller as control


class HandoffTests(unittest.TestCase):
    def test_explicit_five_tag_mapping(self):
        self.assertEqual(control.FORMAL_CONFIG_TAGS, {
            "vitb32": "vitb32_formal_20261103", "vitb16": "vitb16_formal_20261103",
            "remoteclip": "remoteclip_formal_20261103", "siglip": "siglip_formal_20261103",
            "georsclip": "georsclip_formal_20261103"})

    def test_gate_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "results/vitb32"
            (root / "logs").mkdir()
            run.mkdir(parents=True)
            self.assertFalse(control.assess_training(root, "vitb32")[0])
            state_path = root / "logs/training_vitb32_state.json"
            state_path.write_text(json.dumps({"status": "running_train", "ready_for_evaluation": False}))
            self.assertFalse(control.assess_training(root, "vitb32")[0])
            (root / "code").mkdir()
            (root / "scripts").mkdir()
            runner = root / "scripts/run_repaired_experiment.py"
            runner.write_text("test runner")
            model = root / "code/saqre_phys.py"
            model.write_text("test source")
            freeze = root / "TRAINING_SOURCE_FROZEN.json"
            freeze.write_text(json.dumps({"file_sha256": {"code/saqre_phys.py": control.sha(model)}}))
            real_expected = control.TRAINING_FREEZE_SHA
            control.TRAINING_FREEZE_SHA = control.sha(freeze)
            try:
                split = root / "data/clean"
                split.mkdir(parents=True)
                captions = split / "Caption_test.csv"
                captions.write_text("image,caption\n")
                sm = split / "manifest.json"
                sm.write_text(json.dumps({"content_disjoint_train_test": True,
                                          "content_disjoint_fit_val": True,
                                          "output_sha256": {"Caption_test.csv": control.sha(captions)}}))
                steps = []
                for name in control.VARIANTS:
                    path = run / f"head_{name}.pt"
                    path.write_bytes(name.encode())
                    steps.append({"name": "train_" + name, "status": "complete",
                                  "artifact_sha256": {str(path): control.sha(path)}})
                for mode in ("physics", "uniform_grid"):
                    path = run / f"cache_{mode}_test.pt"
                    path.write_bytes(mode.encode())
                    steps.append({"name": f"cache_{mode}_test", "status": "imported_verified_cache",
                                  "artifact_sha256": {str(path): control.sha(path)}})
                config = {"workspace": str(root), "output": str(run), "tag": "vitb32_formal_20261103",
                          "model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "remoteclip": None,
                          "source_sha256": {"saqre_phys.py": control.sha(model)},
                          "runner_sha256": control.sha(runner), "split": str(split),
                          "split_manifest_sha256": control.sha(sm)}
                (run / "run_manifest.json").write_text(json.dumps({"config": config, "steps": steps}))
                state = {"status": "complete_sweep_and_train", "ready_for_evaluation": True,
                         "current_child_pid": None, "frozen_manifest_sha256": control.TRAINING_FREEZE_SHA}
                state_path.write_text(json.dumps(state))
                ready, _, evidence = control.assess_training(root, "vitb32")
                self.assertTrue(ready)
                self.assertEqual(len(evidence["verified_artifacts"]), 6)
                manifest_path = run / "run_manifest.json"
                for bad_tag in ("vitb32", "vitb32_formal_20261007", "georsclip_formal_20261103", "vitb32_formal_20261103_extra"):
                    config["tag"] = bad_tag
                    manifest_path.write_text(json.dumps({"config": config, "steps": steps}))
                    with self.assertRaisesRegex(ValueError, "Wrong/non-formal"):
                        control.assess_training(root, "vitb32")
                config["tag"] = "vitb32_formal_20261103"
                manifest_path.write_text(json.dumps({"config": config, "steps": steps}))
                lock = run / ".runner.lock"
                lock.write_text("123")
                self.assertFalse(control.assess_training(root, "vitb32")[0])
                lock.unlink()
                state["current_child_pid"] = 123
                state_path.write_text(json.dumps(state))
                self.assertFalse(control.assess_training(root, "vitb32")[0])
                state["current_child_pid"] = None
                state_path.write_text(json.dumps(state))
                (run / "head_full.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                    control.assess_training(root, "vitb32")
            finally:
                control.TRAINING_FREEZE_SHA = real_expected

    def test_metric_failure_handling(self):
        self.assertIsNone(control.metric([]))
        control.compare_metrics(control.metric([1, None, 5]), control.metric([1, None, 5]))
        with self.assertRaises(ValueError):
            control.compare_metrics({"R@1": float("nan")}, {"R@1": 0})


if __name__ == "__main__":
    unittest.main()
