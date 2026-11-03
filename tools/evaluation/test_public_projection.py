"""Temporary synthetic originals verify redaction, bindings and completion gates."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import evaluation_controller as control
import export_paper_results as exports
import public_projection as public
from test_export_paper_results import make_fixture, make_stress, dump, bind_outputs

SECRET = "DO_NOT_PUBLISH_CAPTION_BODY /Users/private_researcher/raw_data"


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def complete_fixture(root):
    evaluation, sweep_path, manifest = make_fixture(root)
    training = sweep_path.parent
    sweep = exports.read(sweep_path)
    split = root / "split"
    split.mkdir()
    ids = {"fit": sweep["provenance"]["train_ids"], "val": sweep["provenance"]["validation_ids"],
           "test": exports.read(evaluation / "queries.json")["gallery_ids"]}
    ids["train"] = ids["fit"] + ids["val"]
    for name, values in ids.items():
        dump(split / (name + "_ids.json"), values)
        (split / ("Caption_" + name + ".csv")).write_text("image,caption\nfixture," + SECRET + "\n")
    split_manifest = {"status": "complete", "protocol": "content_disjoint_rgb_v1_20261007", "seed": 0,
        "content_disjoint_train_test": True, "content_disjoint_fit_val": True,
        "official_test_duplicate_content_group_count": 0, "split_counts": public.SPLIT_SIZES,
        "output_sha256": {path.name: control.sha(path) for path in split.iterdir()}}
    dump(split / "manifest.json", split_manifest)
    signature = {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "primitive_mode": "physics",
                 "pretrained_weight_sha256": text_hash("pretrained"),
                 "source_sha256": {"/Users/private_researcher/source/saqre_phys.py": text_hash("head")}}
    grid_signature = {**signature, "primitive_mode": "uniform_grid"}
    for mode in ("physics", "uniform_grid"):
        for population in ("train", "test"):
            (training / f"cache_{mode}_{population}.pt").write_bytes(f"{mode}/{population}".encode())
    common_hashes = {"train_cache_sha256": control.sha(training / "cache_physics_train.pt"),
                     "captions_sha256": control.sha(split / "Caption_train.csv"),
                     "head_source_sha256": text_hash("head"), "training_source_sha256": text_hash("training"),
                     "batch_scoring_sha256": control.HELPER_SHA}
    sweep["provenance"].update(common_hashes, cache_signature=signature, selection_source_sha256=text_hash("selection"))
    sweep["config"]["out"] = str(sweep_path)
    dump(sweep_path, sweep)
    manifest["artifact_validation"]["cache_signatures"] = {"physics": signature, "uniform_grid": grid_signature}
    manifest["inputs"]["captions"] = {"path": str(split / "Caption_test.csv"), "sha256": control.sha(split / "Caption_test.csv")}
    for mode in ("physics", "uniform_grid"):
        path = training / f"cache_{mode}_test.pt"
        manifest["inputs"][mode + "_cache"] = {"path": str(path), "sha256": control.sha(path)}
    for variant in exports.VARIANTS:
        path = training / f"head_{variant}.pt"
        path.write_bytes(("actual fixture " + variant).encode())
        manifest["inputs"]["head_" + variant] = {"path": str(path), "sha256": control.sha(path)}
        head = manifest["artifact_validation"]["heads"][variant]
        head["ablate"] = {"full": [], "stats_off": ["sr", "ty"], "cfar_strict": ["cfar_strict"], "uniform_grid": ["sr", "ty"]}[variant]
        head["config"]["cache"] = "/Users/private_researcher/cache.pt"
        head["config"]["captions"] = SECRET
        head["training_provenance"].update(common_hashes, cache_signature=grid_signature if variant == "uniform_grid" else signature)
        if variant == "uniform_grid":
            head["training_provenance"]["train_cache_sha256"] = control.sha(training / "cache_uniform_grid_train.pt")
    for filename in [f"eval_{v}_k{k}.json" for v in exports.VARIANTS for k in exports.KS] + ["eval_baselines.json", "queries.json"]:
        path = evaluation / filename
        document = exports.read(path)
        for query in document["queries"]:
            query["caption"] = SECRET
            query["caption_sha256"] = text_hash(SECRET)
            query["private_caption_note"] = SECRET
        document["private_machine_path"] = "/Users/private_researcher/experiment"
        document["private_free_text"] = SECRET
        if "provenance" in document:
            p = document["provenance"]
            p["captions_sha256"] = control.sha(split / "Caption_test.csv")
            if filename != "eval_baselines.json":
                variant = filename[5:filename.rindex("_k")]
                mode = "uniform_grid" if variant == "uniform_grid" else "physics"
                p["head_sha256"] = manifest["inputs"]["head_" + variant]["sha256"]
                p["cache_sha256"] = manifest["inputs"][mode + "_cache"]["sha256"]
                p["cache_signature"] = grid_signature if mode == "uniform_grid" else signature
            p["private_caption_note"] = SECRET
            p["independent_manifest"] = str(evaluation / "independent_manifest.json")
        dump(path, document)
    bind_outputs(evaluation, manifest)
    run_path = training / "run_manifest.json"
    run = exports.read(run_path)
    run["config"].update(output=str(training), split=str(split), split_manifest_sha256=control.sha(split / "manifest.json"))
    run["private_free_text"] = SECRET
    for step in run["steps"]:
        if step["name"] == "sweep":
            step["artifact_sha256"] = {name: control.sha(name) for name in step["artifact_sha256"]}
        else:
            variant = step["name"][6:]
            path = training / f"head_{variant}.pt"
            step["artifact_sha256"] = {str(path): control.sha(path)}
    dump(run_path, run)
    return evaluation, training, split


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        cls.evaluation, cls.training, cls.split = complete_fixture(cls.root)
        evaluation = exports.load_evaluation("vitb32", cls.evaluation)
        cls.stress, manifest = make_stress(cls.root, evaluation)
        tool_dir = Path(__file__).resolve().parent
        helper_candidates = [tool_dir / "batch_scoring.py",
                             tool_dir.parent.parent / "code/batch_scoring.py",
                             tool_dir.parent / "pare_optimization_20261103/batch_scoring.py"]
        helper = next(path for path in helper_candidates if path.is_file() and control.sha(path) == control.HELPER_SHA)
        evaluator = Path(__file__).resolve().with_name("evaluate_multik.py")
        for name, path in (("evaluator", evaluator), ("scoring_helper", helper)):
            manifest["inputs"][name] = {"path": str(path), "sha256": control.sha(path)}
        dump(cls.stress / "manifest.json", manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_projection_removes_text_and_paths_but_preserves_metrics(self):
        out = self.root / "public"
        result = public.create_projection(self.evaluation, self.training, self.split, out)
        self.assertEqual(len(result["projection_files"]), 26)
        self.assertEqual(len(list(out.iterdir())), 28)
        for path in out.iterdir():
            content = path.read_text()
            self.assertNotIn(SECRET, content)
            self.assertNotIn("/Users/", content)
            self.assertNotIn(str(self.root), content)
            self.assertNotIn('"caption":', content)
        original = exports.read(self.evaluation / "eval_full_k100.json")
        projected = exports.read(out / "eval_full_k100.json")
        self.assertEqual(projected["natural_scene_shortlist"], original["natural_scene_shortlist"])
        self.assertEqual(projected["queries"][0]["natural_PARE_rank"], original["queries"][0]["natural_PARE_rank"])
        self.assertEqual(projected["queries"][0]["caption_sha256"], text_hash(SECRET))
        self.assertTrue(projected["_projection"]["not_original_json"])
        for name, entry in result["projection_files"].items():
            self.assertEqual(entry["source_sha256"], control.sha(self.evaluation / name))
            self.assertEqual(entry["projection_sha256"], control.sha(out / name))
            self.assertNotEqual(entry["source_sha256"], entry["projection_sha256"])
        self.assertEqual(result["bindings"]["split"]["ids"]["train"]["count"], 9181)
        self.assertEqual(len(result["bindings"]["selection"]["trials"]), 9)
        self.assertEqual(len(result["bindings"]["final_heads"]), 4)
        self.assertEqual(result["original_training_manifest_status"], "partial")
        self.assertFalse((out / "independent_manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "overwrite"):
            public.create_projection(self.evaluation, self.training, self.split, out)

    def test_incomplete_evaluation_rejected_without_output(self):
        path = self.evaluation / "independent_manifest.json"
        old = path.read_bytes()
        try:
            document = exports.read(path)
            document["status"] = "running"
            dump(path, document)
            out = self.root / "refused_running"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                public.create_projection(self.evaluation, self.training, self.split, out)
            self.assertFalse(out.exists())
        finally:
            path.write_bytes(old)

    def test_incomplete_training_rejected(self):
        path = self.training / "run_manifest.json"
        old = path.read_bytes()
        try:
            document = exports.read(path)
            document["steps"][0]["status"] = "running"
            dump(path, document)
            out = self.root / "refused_training"
            with self.assertRaisesRegex(ValueError, "Sweep is still"):
                public.create_projection(self.evaluation, self.training, self.split, out)
            self.assertFalse(out.exists())
        finally:
            path.write_bytes(old)

    def test_current_head_hash_mismatch_rejected(self):
        path = self.training / "head_full.pt"
        old = path.read_bytes()
        try:
            path.write_bytes(b"mutated after completed evaluation")
            out = self.root / "refused_head"
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                public.create_projection(self.evaluation, self.training, self.split, out)
            self.assertFalse(out.exists())
        finally:
            path.write_bytes(old)

    def test_paths_and_raw_caption_keys_fail_closed(self):
        for value in ("/mnt/data/file", "C:\\Users\\name\\file", "file:///private/file", "prefix /Users/name/file"):
            with self.assertRaisesRegex(ValueError, "Absolute"):
                public.safe_tree({"value": value})
        with self.assertRaisesRegex(ValueError, "Unredacted"):
            public.safe_tree({"caption": "private"})
        query = public.project_query({"query_id": "image#caption=1", "image_id": "image", "caption_sha256": text_hash(SECRET),
                                      "caption": SECRET, "unknown_private_field": SECRET, "natural_PARE_rank": 5})
        self.assertNotIn("caption", query)
        self.assertNotIn("unknown_private_field", query)
        self.assertEqual(query["natural_PARE_rank"], 5)

    def test_projection_rejects_unknown_formal_config_tag(self):
        path = self.training / "run_manifest.json"
        original = path.read_bytes()
        try:
            run = json.loads(original)
            run["config"]["tag"] = "vitb32_formal_20261103_extra"
            dump(path, run)
            with self.assertRaisesRegex(ValueError, "path/tag binding"):
                public.create_projection(self.evaluation, self.training, self.split, self.root / "bad_tag")
        finally:
            path.write_bytes(original)

    def test_optional_scale_projection_preserves_ranks_and_redacts_queries(self):
        out = self.root / "public_with_stress"
        manifest = public.create_projection(self.evaluation, self.training, self.split, out, self.stress)
        self.assertEqual(len(manifest["projection_files"]), 31)
        self.assertTrue(manifest["bindings"]["synthetic_scale"]["original_scale_ranks_equal_to_primary_evaluation"])
        raw = exports.read(self.stress / "eval_spacing_2.json")
        projected = exports.read(out / "stress_eval_spacing_2.json")
        self.assertEqual(raw["metrics_percent"], projected["metrics_percent"])
        self.assertEqual(raw["queries"][0]["ranks"], projected["queries"][0]["ranks"])
        for path in out.iterdir():
            self.assertNotIn(SECRET, path.read_text())
            self.assertNotIn(str(self.root), path.read_text())
        self.assertNotIn("caption", exports.read(out / "stress_queries.json")["queries"][0])

    def test_optional_unfinished_scale_rejects_entire_projection(self):
        path = self.stress / "manifest.json"
        old = path.read_bytes()
        try:
            manifest = exports.read(path)
            manifest["status"] = "running"
            dump(path, manifest)
            out = self.root / "refused_stress"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                public.create_projection(self.evaluation, self.training, self.split, out, self.stress)
            self.assertFalse(out.exists())
        finally:
            path.write_bytes(old)


if __name__ == "__main__":
    unittest.main()
