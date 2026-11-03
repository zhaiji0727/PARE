"""CPU-only exporter regression; synthetic data are never paper output."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import export_paper_results as export
import evaluation_controller as control


def dump(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))


def bind_outputs(directory, manifest):
    manifest["outputs"] = {p.name: {"sha256": control.sha(p), "bytes": p.stat().st_size}
                           for p in directory.iterdir() if p.name != "independent_manifest.json"}
    dump(directory / "independent_manifest.json", manifest)


def make_fixture(root):
    directory = root / "evaluation"
    directory.mkdir()
    n = 3688
    ids = [f"test_{i}" for i in range(n)]
    records = [{"query_id": name + "#caption=1", "image_id": name, "caption_index_one_based": 1,
                "caption": "fixture", "caption_sha256": "fixture", "gold_gallery_row": i}
               for i, name in enumerate(ids)]
    qp = {"mode": "first-caption", "max_queries": 0, "truncated": False,
          "gallery_images_without_evaluated_query": []}
    dump(directory / "queries.json", {"queries": records, "gallery_ids": ids, "query_protocol": qp})
    config = {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "remoteclip": None, "smoke": False}
    signature = {"model": config["model"], "primitive_mode": "physics"}
    fit, val = [f"train_{i}" for i in range(8263)], [f"val_{i}" for i in range(918)]
    manifest = {"schema": "pare.independent-multik.v1", "status": "complete", "input_code_recheck_passed": True,
                "config": config, "expected_eval_json_count": 25,
                "code": {"independent_evaluator": {"sha256": control.EVALUATOR_SHA},
                         "vmap_helper": {"sha256": control.HELPER_SHA}},
                "inputs": {"captions": {"sha256": "captions"}, "physics_cache": {"sha256": "physics"},
                           "uniform_grid_cache": {"sha256": "uniform"}},
                "artifact_validation": {"cache_signatures": {"physics": signature}, "heads": {}}}
    base = {"model": config["model"], "n_queries": n, "n_gallery": n, "query_protocol": qp,
            "gallery_ids": ids, "seed": 0, "provenance": {"captions_sha256": "captions",
            "evaluator_sha256": control.EVALUATOR_SHA, "allow_legacy_artifacts": False}}
    scene = [[10, 11, 50, n][i % 4] for i in range(n)]
    for vi, variant in enumerate(export.VARIANTS):
        manifest["inputs"]["head_" + variant] = {"sha256": "head_" + variant}
        manifest["artifact_validation"]["heads"][variant] = {"n_images": 9181, "n_pairs": 31945,
            "config": {"lr": 0.0005, "kappa": 0.03, "epochs": 10}, "history": [0] * 10,
            "training_provenance": {"train_ids": fit + val}}
        for k in export.KS:
            ranks = [[1 + vi, 10, 11, 50 if k == n else None][i % 4] for i in range(n)]
            rows = [{**row, "scene_full_gallery_rank": scene[i], "natural_PARE_rank": ranks[i],
                     "natural_candidate_included": ranks[i] is not None} for i, row in enumerate(records)]
            report = {**base, "provenance": {**base["provenance"], "head_sha256": "head_" + variant,
                      "cache_sha256": "uniform" if variant == "uniform_grid" else "physics"},
                      "queries": rows, "scene_full_gallery": control.metric(scene),
                      "natural_scene_shortlist": {"K": k, "candidate_recall_percent": 100 if k == n else 75,
                          "n_natural_inclusions": n if k == n else n * 3 // 4,
                          "PARE_end_to_end": control.metric(ranks),
                          "PARE_conditional_on_inclusion": control.metric([r for r in ranks if r is not None])}}
            if k == 100:
                report["controlled_random_pools"] = {"K": 100, "pools_per_query": 5,
                    "scene": control.metric([1, 2, 3, 4, 5]), "PARE": control.metric([1, 2, 3, 4, 5])}
                report["appearance_adversarial_pool"] = {"M": 50, "scene": control.metric([10]), "PARE": control.metric([10])}
                for row in rows:
                    row["controlled_random_pool_ranks"] = {"scene": [1, 2, 3, 4, 5], "PARE": [1, 2, 3, 4, 5]}
                    row["appearance_adversarial_ranks"] = {"scene": 10, "PARE": 10}
            dump(directory / f"eval_{variant}_k{k}.json", report)
        (directory / f"scores_{variant}.npy").write_bytes(b"synthetic-hash-fixture")
    methods = ("scene", "mean_pool", "MaxSim")
    baseline = {**base, "queries": [{**row, "full_gallery_ranks": {m: scene[i] for m in methods},
        "controlled_random_pool_ranks": {m: [1, 2, 3, 4, 5] for m in methods},
        "appearance_adversarial_ranks": {m: 10 for m in methods}} for i, row in enumerate(records)],
        "full_gallery": {m: control.metric(scene) for m in methods},
        "controlled_random_pools": {"K": 100, "pools_per_query": 5, **{m: control.metric([1, 2, 3, 4, 5]) for m in methods}},
        "appearance_adversarial_pool": {"M": 50, **{m: control.metric([10]) for m in methods}}}
    dump(directory / "eval_baselines.json", baseline)
    for filename in ["query_embeddings.npy"] + [f"baseline_scores_{m}.npy" for m in methods] + [f"{kind}_{mode}.{ext}" for mode in ("physics", "uniform_grid") for kind, ext in (("scene_scores", "npy"), ("candidates", "npz"))]:
        (directory / filename).write_bytes(b"synthetic-hash-fixture")
    bind_outputs(directory, manifest)
    training = root / "training"
    training.mkdir()
    trial_dir = training / "hparam_best_trials"
    trial_dir.mkdir()
    grid = []
    for lr in (0.0005, 0.001, 0.002):
        for kappa in (0.03, 0.05, 0.07):
            stem = f"lr_{lr:g}_kappa_{kappa:g}"
            head = trial_dir / (stem + ".pt")
            head.write_bytes(stem.encode())
            ranks = [10] * 918
            dump(trial_dir / (stem + ".json"), {"query_ids": val, "gallery_ids": val, "ranks": ranks,
                                              "checkpoint_sha256": control.sha(head)})
            grid.append({"lr": lr, "kappa": kappa, **{"val_" + key: value for key, value in control.metric(ranks).items()}})
    sweep = {"epochs": 10, "n_val_q": 918, "n_val_gallery": 918, "grid": grid, "best": grid[0],
             "config": {**config, "epochs": 10, "seed": 0, "val_q": 0}, "trial_directory": str(trial_dir),
             "provenance": {"train_ids": fit, "validation_ids": val, "cache_signature": signature,
                            "batch_scoring_sha256": control.HELPER_SHA}}
    sweep_path = training / "hparam_best.json"
    dump(sweep_path, sweep)
    step = {"name": "sweep", "status": "complete", "artifact_sha256": {str(p): control.sha(p) for p in [sweep_path] + list(trial_dir.iterdir())}}
    steps = [step] + [{"name": "train_" + variant, "status": "complete",
                      "artifact_sha256": {str(training / f"head_{variant}.pt"): "head_" + variant}}
                     for variant in export.VARIANTS]
    dump(training / "run_manifest.json", {"status": "partial", "config": {**config, "tag": "vitb32_formal_20261103"}, "steps": steps})
    return directory, sweep_path, manifest


def make_stress(root, evaluation):
    directory = root / "stress"
    directory.mkdir()
    manifest = {"schema": "pare.synthetic-spacing-evaluation.v1", "status": "complete",
                "input_code_recheck_passed": True, "is_real_gsd_validation": False,
                "config": evaluation["manifest"]["config"], "inputs": {}}
    for key, reference in (("head", "head_full"), ("stats_off_head", "head_stats_off"),
                           ("captions", "captions"), ("reference_cache", "physics_cache")):
        manifest["inputs"][key] = evaluation["manifest"]["inputs"][reference]
    manifest["inputs"]["evaluator"] = {"sha256": control.EVALUATOR_SHA}
    manifest["inputs"]["scoring_helper"] = {"sha256": control.HELPER_SHA}
    document = export.read(evaluation["directory"] / "queries.json")
    dump(directory / "queries.json", document)
    baseline = evaluation["reports"]["eval_baselines.json"]["queries"]
    full = evaluation["reports"]["eval_full_k3688.json"]["queries"]
    stats = evaluation["reports"]["eval_stats_off_k3688.json"]["queries"]
    reports = {}
    for factor, shift in (("1", 0), ("1.5", 20), ("2", 40)):
        rows = []
        for i, row in enumerate(document["queries"]):
            ranks = {"Scene-CLIP": baseline[i]["full_gallery_ranks"]["scene"],
                     "PARE": full[i]["natural_PARE_rank"], "PARE-stats-off": stats[i]["natural_PARE_rank"]}
            rows.append({**{key: row[key] for key in ("image_id", "query_id", "gold_gallery_row")},
                         "ranks": {key: min(export.N, value + shift) for key, value in ranks.items()}})
        report = {"n_queries": export.N, "n_gallery": export.N, "queries": rows,
                  "is_real_gsd_validation": False, "spacing_factor": factor,
                  "query_protocol": document["query_protocol"],
                  "metrics_percent": {key: control.metric([row["ranks"][key] for row in rows]) for key in ranks}}
        reports[factor] = report
        dump(directory / ("eval_spacing_" + factor.replace(".", "p") + ".json"), report)
    comparison = {"n_queries": export.N, "n_gallery": export.N, "is_real_gsd_validation": False,
                  "metrics_percent": {factor: report["metrics_percent"] for factor, report in reports.items()},
                  "delta_percentage_points_from_original": {factor: {
                      method: {metric: value - reports["1"]["metrics_percent"][method][metric]
                               for metric, value in values.items()}
                      for method, values in report["metrics_percent"].items()}
                      for factor, report in reports.items() if factor != "1"}}
    dump(directory / "comparison.json", comparison)
    manifest["outputs"] = {path.name: {"sha256": control.sha(path), "bytes": path.stat().st_size}
                           for path in directory.iterdir()}
    dump(directory / "manifest.json", manifest)
    return directory, manifest


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.directory, cls.sweep_path, cls.manifest = make_fixture(cls.root)
        cls.loaded = export.load_evaluation("vitb32", cls.directory)
        cls.sweep, _ = export.load_sweep("vitb32", cls.sweep_path, cls.loaded)
        cls.stress_directory, cls.stress_manifest = make_stress(cls.root, cls.loaded)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_known_metrics_and_protocol_separation(self):
        tables = export.build_tables({tag: self.loaded for tag in export.TAGS}, {tag: self.sweep for tag in export.TAGS}, {})
        self.assertEqual(len(tables["table_i_full_gallery"]), 25)
        self.assertEqual(len(tables["appendix_natural_k"]), 120)
        self.assertEqual(len(tables["appendix_validation_grid"]), 45)
        self.assertEqual(len(tables["appendix_hard50"]), 35)
        control_row = tables["table_ii_vitb32_controls"][0]
        self.assertEqual(control_row["controlled100_R@10"], 100)
        self.assertEqual(control_row["natural100_R@10"], 50)
        self.assertEqual(control_row["natural100_candidate_recall_percent"], 75)
        self.assertAlmostEqual(control_row["full_gallery_MRR@10"], 27.5)
        self.assertNotIn("synthetic_scale_sensitivity", tables)
        out = self.root / "export"
        export.write_exports(out, tables, {"test_fixture": True})
        self.assertIn("not end-to-end", (out / "appendix_controlled100.tex").read_text())
        self.assertIn("27.50", (out / "table_i_full_gallery.tex").read_text())
        self.assertEqual(export.read(out / "export_manifest.json")["status"], "complete")

    def test_noncomplete_smoke_and_hash_rejected(self):
        for status in ("running", "partial", "failed"):
            manifest = {**self.manifest, "status": status}
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export.verify_outputs(self.directory, manifest, [])
        with self.assertRaisesRegex(ValueError, "Smoke"):
            export.verify_outputs(self.directory, {**self.manifest, "config": {"smoke": True}}, [])
        manifest = copy.deepcopy(self.manifest)
        manifest["outputs"]["queries.json"]["sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            export.verify_outputs(self.directory, manifest, [])

    def test_query_population_and_model_binding(self):
        document = export.read(self.directory / "queries.json")
        document["queries"].pop()
        with self.assertRaisesRegex(ValueError, "3688"):
            export.query_contract(document)
        with self.assertRaisesRegex(ValueError, "External"):
            export.check_model(self.manifest["config"], "remoteclip")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            export.bindings(["vitb32=a", "vitb32=b"])
        with self.assertRaisesRegex(ValueError, "five"):
            export.bindings(["vitb32=a"], all_five=True)

    def test_grid_and_final_head_selection(self):
        self.assertIs(export.sweep_best(self.sweep["grid"]), self.sweep["grid"][0])
        grid = copy.deepcopy(self.sweep["grid"])
        grid[-1]["val_MRR@10"] += 1
        self.assertIs(export.sweep_best(grid), grid[-1])
        grid[0] = grid[1]
        with self.assertRaisesRegex(ValueError, "nine-point"):
            export.sweep_best(grid)
        loaded = {**self.loaded, "manifest": copy.deepcopy(self.loaded["manifest"])}
        loaded["manifest"]["artifact_validation"]["heads"]["full"]["config"]["lr"] = 0.002
        with self.assertRaisesRegex(ValueError, "validation-selected"):
            export.load_sweep("vitb32", self.sweep_path, loaded)

    def test_sweep_rejects_unmapped_or_other_backbone_tag(self):
        path = self.sweep_path.parent / "run_manifest.json"
        original = path.read_bytes()
        try:
            for tag in ("vitb32", "vitb32_formal_20261007", "georsclip_formal_20261103"):
                run = json.loads(original)
                run["config"]["tag"] = tag
                dump(path, run)
                with self.assertRaisesRegex(ValueError, "Wrong training run"):
                    export.load_sweep("vitb32", self.sweep_path, self.loaded)
        finally:
            path.write_bytes(original)

    def test_scale_bound_to_original_and_negative_delta_preserved(self):
        result = export.load_stress("vitb32", self.stress_directory, self.loaded)
        tables = export.build_tables({tag: self.loaded for tag in export.TAGS}, {tag: self.sweep for tag in export.TAGS}, {"vitb32": result})
        rows = tables["synthetic_scale_sensitivity"]
        self.assertEqual(len(rows), 9)
        changed = next(row for row in rows if row["spacing_factor"] == 1.5 and row["method"] == "PARE")
        self.assertEqual(changed["delta_R10_pp"], -50)
        self.assertFalse(changed["is_real_gsd_validation"])
        loaded = {**self.loaded, "manifest": copy.deepcopy(self.loaded["manifest"])}
        loaded["manifest"]["inputs"]["head_full"]["sha256"] = "wrong_head"
        with self.assertRaisesRegex(ValueError, "corresponding original"):
            export.load_stress("vitb32", self.stress_directory, loaded)

    def test_scale_original_rank_mismatch_rejected(self):
        loaded = {**self.loaded, "reports": copy.deepcopy(self.loaded["reports"])}
        loaded["reports"]["eval_full_k3688.json"]["queries"][0]["natural_PARE_rank"] = 2
        with self.assertRaisesRegex(ValueError, "Original-scale ranks"):
            export.load_stress("vitb32", self.stress_directory, loaded)


if __name__ == "__main__":
    unittest.main()
