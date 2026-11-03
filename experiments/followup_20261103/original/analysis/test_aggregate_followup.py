"""Synthetic, temporary-only checks of completeness gates and fixed contrasts."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest

import aggregate_followup as aggregate

WORK = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(root):
    source = WORK / "followup_ablation_20261103"
    public = WORK / "pare_public_review_20261103/PARE_6e4ed9a6/data/splits_content_disjoint_20261007"
    plan_dir = WORK / "pare_factorial_followup_20261103"
    for name in ("protocol.json", "protocol.md"):
        target = root / "plans" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plan_dir / name, target)
    shutil.copyfile(source / "FOLLOWUP_SOURCE_FROZEN.json", root / "FOLLOWUP_SOURCE_FROZEN.json")
    for name in ("fit_ids.json", "val_ids.json"):
        target = root / "data/ret_clean_20261007" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(public / name, target)
    plan = aggregate.read(root / "plans/protocol.json")
    freeze = aggregate.read(root / "FOLLOWUP_SOURCE_FROZEN.json")
    fit = aggregate.read(public / "fit_ids.json")
    val = aggregate.read(public / "val_ids.json")
    bindings = plan["input_bindings"]
    inputs = {"training_cache": bindings["training_cache_sha256"], "training_captions": bindings["clean_training_captions_sha256"],
              **bindings["split_file_sha256"], "original_split_manifest": bindings["original_split_manifest_sha256"],
              "pretrained_weight": plan["backbone"]["pretrained_weight_sha256"]}
    sources = {k: v for k, v in freeze["file_sha256"].items() if k.startswith(("base_code/", "code/")) or k == "scripts/run_factorial_fitval.py"}
    audit = {"all_passed": True, "n_runs": 42, "original_cache_unchanged": True,
             "protocol_sha256": aggregate.PROTOCOL_SHA, "source_freeze_sha256": aggregate.FREEZE_SHA,
             "verifier_sha256": aggregate.VERIFIER_SHA, "rank_replay_device": "cuda:0", "seed_manifests": {},
             "initial_state_sha256_by_seed": {}, "frozen_text_embeddings_sha256_by_seed": {}, "runs": [],
             "synthetic_test_fixture_only": True}
    caption_hashes = [digest("synthetic caption " + str(i)) for i in range(918)]
    query_hash = aggregate.object_sha([[image, text] for image, text in zip(val, caption_hashes)])
    for seed in (0, 1, 2):
        initial = digest("initial seed " + str(seed))
        order, orders, rng = list(range(28759)), [], random.Random(seed)
        for _ in range(10):
            rng.shuffle(order)
            orders.append(aggregate.object_sha(order))
        ident = {"protocol_sha256": aggregate.PROTOCOL_SHA, "seed": seed, "source_sha256": sources, "input_sha256": inputs,
                 "initial_pairs_sha256": digest("same pairs"), "validation_queries_sha256": query_hash,
                 "epoch_order_sha256": orders, "configuration_ids": [c["id"] for c in plan["configurations"]]}
        manifest = {"schema": "pare.factorial-fit-validation.v1", "status": "complete_all_14_configurations", "active_configuration": None,
                    "identity": ident, "settings": plan["training"], "posthoc_exploratory": True, "uses_official_test": False,
                    "n_fit_images": 8263, "n_fit_pairs": 28759, "n_validation_queries": 918, "n_validation_gallery": 918,
                    "cache_contains_images": 9181, "train_ids": fit, "validation_ids": val,
                    "initial_state_sha256": initial, "text_embeddings_sha256": digest("frozen embeddings"), "runs": {}}
        for index, config in enumerate(plan["configurations"]):
            name = config["id"]
            directory = root / "results" / f"seed_{seed}" / name
            directory.mkdir(parents=True)
            # These are intentionally not torch objects. The fixture represents
            # the separate audit boundary; no real evaluation data are used.
            (directory / "head.pt").write_bytes(f"synthetic head {seed} {name}".encode())
            (directory / "validation_scores.pt").write_bytes(f"synthetic scores {seed} {name}".encode())
            ranks = [(1, 5, 10, 11)[(i + index + seed) % 4] for i in range(918)]
            c1, c5, c10 = (ranks.count(i) for i in (1, 5, 10))
            metrics = {"R@1": 100 * c1 / 918, "R@5": 100 * (c1 + c5) / 918,
                       "R@10": 100 * (c1 + c5 + c10) / 918, "MRR@10": 100 * (c1 + c5 / 5 + c10 / 10) / 918}
            result = {"configuration": config, "seed": seed, "protocol_sha256": aggregate.PROTOCOL_SHA,
                      "n_queries": 918, "n_gallery": 918, "posthoc_exploratory": True, "official_test_accessed": False,
                      "checkpoint_sha256": aggregate.sha(directory / "head.pt"), "query_ids": val, "gallery_ids": val,
                      "query_caption_sha256": caption_hashes, "gold_gallery_index": list(range(918)), "ranks": ranks,
                      "metrics_percent": metrics}
            write(directory / "validation.json", result)
            history = [{"epoch": i + 1, "pairs": 28759, "batches": 450, "tail_batch_pairs": 23,
                        "pair_order_sha256": orders[i], "mean_batch_loss": 1 / (i + 1)} for i in range(10)]
            write(directory / "training_progress.json", {"history": history})
            artifacts = {p.name: aggregate.sha(p) for p in directory.iterdir()}
            manifest["runs"][name] = {"status": "complete", "configuration": config, "seed": seed,
                "initial_state_sha256": initial, "text_embeddings_sha256": digest("frozen embeddings"),
                "initial_pairs_sha256": ident["initial_pairs_sha256"], "epoch_order_sha256": orders,
                "epochs": 10, "batches": 4500, "artifacts_sha256": artifacts, "metrics_percent": metrics}
            audit["runs"].append({"seed": seed, "configuration": name, "relative_directory": str(directory.relative_to(root)),
                "head_sha256": artifacts["head.pt"], "matrix_sha256": artifacts["validation_scores.pt"],
                "validation_sha256": artifacts["validation.json"], "initial_state_sha256": initial,
                "all_checks_passed": True, "metrics_percent": metrics})
        path = root / "results" / f"seed_{seed}" / "manifest.json"
        write(path, manifest)
        audit["seed_manifests"][str(seed)] = {"relative_path": str(path.relative_to(root)), "sha256": aggregate.sha(path)}
        audit["initial_state_sha256_by_seed"][str(seed)] = initial
        audit["frozen_text_embeddings_sha256_by_seed"][str(seed)] = digest("frozen embeddings")
    write(root / "audit.json", audit)
    return root


class AggregatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original = fixture(Path(cls.temp.name) / "original")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.case = tempfile.TemporaryDirectory()
        self.root = Path(self.case.name) / "case"
        shutil.copytree(self.original, self.root)
        self.audit = self.root / "audit.json"

    def tearDown(self):
        self.case.cleanup()

    def mutate_manifest(self, mutate, seed=0):
        path = self.root / f"results/seed_{seed}/manifest.json"
        value = aggregate.read(path)
        mutate(value)
        write(path, value)
        audit = aggregate.read(self.audit)
        audit["seed_manifests"][str(seed)]["sha256"] = aggregate.sha(path)
        write(self.audit, audit)

    def test_complete_exports_all_rows_and_bound_sources(self):
        result = aggregate.run(self.root, self.audit, self.root / "export")
        self.assertEqual(result["row_counts"], {"run_metrics": 42, "configuration_summary": 14,
            "paired_contrasts_by_seed": 108, "contrast_summary": 36, "primary_contrast_summary": 9})
        summary = aggregate.read(self.root / "export/summary.json")
        self.assertEqual(len(summary["contrast_definitions"]), 36)
        for file, binding in result["outputs"].items():
            self.assertEqual(aggregate.sha(self.root / "export" / file), binding["sha256"])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            aggregate.run(self.root, self.audit, self.root / "export")

    def test_reject_partial_without_creating_output(self):
        self.mutate_manifest(lambda m: m.update(status="running"))
        with self.assertRaisesRegex(ValueError, "running, partial"):
            aggregate.run(self.root, self.audit, self.root / "export")
        self.assertFalse((self.root / "export").exists())

    def test_missing_audit_cell_is_rejected(self):
        audit = aggregate.read(self.audit)
        audit["runs"].pop()
        write(self.audit, audit)
        with self.assertRaisesRegex(ValueError, "42 run records"):
            aggregate.load_complete(self.root, self.audit)

    def test_hash_changes_are_rejected(self):
        path = self.root / "results/seed_0/ABC000_R1T1/validation.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            aggregate.load_complete(self.root, self.audit)

    def test_initialization_and_shuffle_mismatches_are_rejected(self):
        self.mutate_manifest(lambda m: m["runs"]["ABC000_R1T1"].update(initial_state_sha256=digest("different")))
        with self.assertRaisesRegex(ValueError, "initialization/order"):
            aggregate.load_complete(self.root, self.audit)

    def test_wrong_population_is_rejected(self):
        self.mutate_manifest(lambda m: m.update(n_fit_pairs=31945))
        with self.assertRaisesRegex(ValueError, "population"):
            aggregate.load_complete(self.root, self.audit)

    def test_metrics_must_match_ranks_even_after_rebinding_hash(self):
        path = self.root / "results/seed_0/ABC000_R1T1/validation.json"
        result = aggregate.read(path)
        result["metrics_percent"]["R@10"] += 1
        write(path, result)
        new_sha = aggregate.sha(path)
        self.mutate_manifest(lambda m: m["runs"]["ABC000_R1T1"]["artifacts_sha256"].update({"validation.json": new_sha}))
        audit = aggregate.read(self.audit)
        audit["runs"][0]["validation_sha256"] = new_sha
        write(self.audit, audit)
        with self.assertRaisesRegex(ValueError, "Rank/metric disagreement"):
            aggregate.load_complete(self.root, self.audit)

    def test_frozen_contrast_coefficients_are_not_rescaled_or_selected(self):
        plan = aggregate.read(self.root / "plans/protocol.json")
        complete = {}
        for seed in plan["seeds"]:
            for config in plan["configurations"]:
                a, b, c, r, t = (config["factors"][k] for k in "ABCRT")
                value = 20 + (seed + 1) * (2*a + 3*b - 4*c + 5*a*b + 7*a*b*c + 11*r - 13*t + 17*r*t)
                complete[seed, config["id"]] = {metric: float(value) for metric in aggregate.METRICS}
        tables = aggregate.summarize(plan, complete)
        by = {r["contrast"]: r for r in tables["contrast_summary"]}
        self.assertEqual(by["interaction_ABC"]["R@10_mean_delta_pp"], 14)
        self.assertEqual(by["interaction_ABC"]["R@10_sample_sd_pp"], 7)
        self.assertEqual(by["interaction_AB"]["R@10_mean_delta_pp"], 17)
        self.assertEqual(by["ABC000_interaction_RT"]["R@10_mean_delta_pp"], 34)
        self.assertEqual(by["C_given_A0B0"]["R@10_mean_delta_pp"], -8)
        tex = aggregate.render_tables(tables)
        self.assertIn("-8.000", tex["all_36_contrasts.tex"])
        self.assertEqual(len(tables["primary_contrast_summary"]), 9)


if __name__ == "__main__":
    unittest.main()
