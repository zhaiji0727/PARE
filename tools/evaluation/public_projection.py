#!/usr/bin/env python3
"""Create explicitly labeled public projections of completed original results.

This is not an evaluator and does not publish anything. It validates original
results, selection, heads and split bindings before writing a fresh directory.
Captions, absolute machine paths, weights, matrices and caption CSVs are never
copied. Internal paper exporters must continue to use the original results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

import export_paper_results as exports
import evaluation_controller as validator

EXPORTER_SHA = "fef49d50b5fe1e91ba7dd01bf099a566033421c1e3ed68f6e74556dbbf87591c"
VALIDATOR_SHA = "322dbfda59ca259f5e64e2d4804ac7a49eb678a089e09ea94f942131dafde984"
SPLIT_SIZES = {"train": 9181, "fit": 8263, "val": 918, "test": 3688}
QUERY_FIELDS = ("query_id", "image_id", "caption_index_one_based", "caption_sha256", "gold_gallery_row",
                "scene_full_gallery_rank", "natural_candidate_included", "natural_PARE_rank",
                "controlled_random_pool_ranks", "appearance_adversarial_ranks", "full_gallery_ranks", "ranks")
QUERY_PROTOCOL_FIELDS = ("mode", "max_queries", "available_queries", "evaluated_queries", "truncated",
                         "evaluated_unique_images", "query_prefix_counts", "query_image_prefix_counts",
                         "gallery_prefix_counts", "gallery_images_without_evaluated_query")
REPORT_FIELDS = ("model", "remoteclip", "n_queries", "n_gallery", "seed", "gallery_ids", "ablate",
                 "scene_full_gallery", "natural_scene_shortlist", "controlled_random_pools",
                 "appearance_adversarial_pool", "full_gallery")
SIGNATURE_FIELDS = ("schema_version", "model", "pretrained", "pretrained_weight_sha256",
                    "external_checkpoint_sha256", "primitive_mode", "max_regions", "max_sc", "packages")
TRAINING_CONFIG_FIELDS = ("model", "pretrained", "epochs", "bs", "text_bs", "max_pairs", "lr", "kappa",
                          "weight_decay", "seed", "ablate", "allow_legacy_artifacts")
HEAD_HASH_FIELDS = ("train_cache_sha256", "captions_sha256", "head_source_sha256",
                    "training_source_sha256", "batch_scoring_sha256")
REPORT_HASH_FIELDS = ("cache_sha256", "captions_sha256", "evaluator_sha256", "query_selection_sha256",
                      "batch_scoring_sha256", "head_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def picked(value, fields):
    return {key: value[key] for key in fields if key in value}


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Expected a SHA-256 digest")
    return value


def safe_tree(value):
    """Fail closed if an allowlisted structure nevertheless contains a path/text key."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(key not in {"caption", "caption_text", "raw_caption", "text", "command", "argv"},
                    "Unredacted text or command field")
            safe_tree(key)
            safe_tree(child)
    elif isinstance(value, list):
        for child in value:
            safe_tree(child)
    elif isinstance(value, str):
        require(not value.startswith(("/", "~", "file://", "\\\\")) and not PureWindowsPath(value).is_absolute()
                and re.search(r"(?:^|\s)(?:/[A-Za-z0-9_.-]|[A-Za-z]:[\\/])", value) is None,
                "Absolute machine path found in public projection")


def hash_map(values):
    result = {}
    for name, value in values.items():
        basename = Path(name).name
        require(basename not in result, "Source basename collision")
        result[basename] = digest(value)
    return result


def signature(value):
    result = picked(value, SIGNATURE_FIELDS)
    result["source_sha256"] = hash_map(value.get("source_sha256", {}))
    result["original_signature_object_sha256"] = object_hash(value)
    safe_tree(result)
    return result


def ids_binding(ids, source_sha=None):
    result = {"count": len(ids), "ordered_ids_sha256": object_hash(ids), "sorted_ids_sha256": object_hash(sorted(ids))}
    if source_sha:
        result["source_json_sha256"] = digest(source_sha)
    return result


def validate_originals(evaluation_dir, training_run, split_dir):
    validator.assert_hash(exports.__file__, EXPORTER_SHA)
    validator.assert_hash(validator.__file__, VALIDATOR_SHA)
    run_path = training_run / "run_manifest.json"
    run = exports.read(run_path)
    require(run.get("status") in ("partial", "complete") and not (training_run / ".runner.lock").exists(),
            "Training is active, failed, or incomplete")
    config = run["config"]
    tags_by_config = {formal: short for short, formal in validator.FORMAL_CONFIG_TAGS.items()}
    tag = tags_by_config.get(config["tag"])
    require(tag in exports.TAGS and Path(config["output"]).resolve() == training_run,
            "Training-run path/tag binding differs")
    require(Path(config["split"]).resolve() == split_dir, "Training split path differs")
    evaluation = exports.load_evaluation(tag, evaluation_dir)
    sweep, _ = exports.load_sweep(tag, training_run / "hparam_best.json", evaluation)
    manifest = evaluation["manifest"]
    checked = {run_path: validator.sha(run_path), evaluation_dir / "independent_manifest.json": validator.sha(evaluation_dir / "independent_manifest.json"),
               training_run / "hparam_best.json": validator.sha(training_run / "hparam_best.json")}
    # Existing validators bind result and selection files. Also verify the actual
    # evaluator inputs and final training artifacts have not changed afterwards.
    for entry in manifest["inputs"].values():
        path = Path(entry["path"])
        expected = digest(entry["sha256"])
        validator.assert_hash(path, expected)
        checked[path] = expected
    steps = {step["name"]: step for step in run["steps"]}
    for name in ["sweep"] + ["train_" + variant for variant in exports.VARIANTS]:
        require(steps[name]["status"] == "complete", "Incomplete training or selection step")
        for path, expected in steps[name]["artifact_sha256"].items():
            validator.assert_hash(path, expected)
            checked[Path(path)] = expected
    split_path = split_dir / "manifest.json"
    split = exports.read(split_path)
    require(split.get("status") == "complete" and split.get("content_disjoint_train_test") is True
            and split.get("content_disjoint_fit_val") is True
            and split.get("official_test_duplicate_content_group_count") == 0,
            "Expected completed content-disjoint split")
    validator.assert_hash(split_path, digest(config["split_manifest_sha256"]))
    checked[split_path] = config["split_manifest_sha256"]
    require(split["split_counts"] == SPLIT_SIZES, "Unexpected split populations")
    for name, expected in split["output_sha256"].items():
        require(Path(name).name == name, "Split output must be a direct child")
        validator.assert_hash(split_dir / name, digest(expected))
        checked[split_dir / name] = expected
    ids = {name: exports.read(split_dir / (name + "_ids.json")) for name in SPLIT_SIZES}
    for name, values in ids.items():
        require(len(values) == len(set(values)) == SPLIT_SIZES[name], "Invalid split ID list")
    require(set(ids["fit"]).isdisjoint(ids["val"]) and set(ids["train"]) == set(ids["fit"]) | set(ids["val"])
            and set(ids["train"]).isdisjoint(ids["test"]), "Split ID sets are not disjoint partitions")
    require(evaluation["gallery"] == ids["test"], "Gallery differs from exact ordered test-ID list")
    require(set(sweep["provenance"]["train_ids"]) == set(ids["fit"])
            and set(sweep["provenance"]["validation_ids"]) == set(ids["val"]), "Sweep split differs")
    require(manifest["inputs"]["captions"]["sha256"] == split["output_sha256"]["Caption_test.csv"], "Test-caption input differs")
    require(sweep["provenance"]["captions_sha256"] == split["output_sha256"]["Caption_train.csv"], "Sweep-caption input differs")
    for variant, head in manifest["artifact_validation"]["heads"].items():
        provenance = head["training_provenance"]
        require(set(provenance["train_ids"]) == set(ids["train"])
                and provenance["captions_sha256"] == split["output_sha256"]["Caption_train.csv"], "Final head training population differs")
        for key in HEAD_HASH_FIELDS:
            digest(provenance[key])
    for record in evaluation["records"]:
        require(hashlib.sha256(record["caption"].encode()).hexdigest() == record["caption_sha256"], "Caption digest mismatch")
    return tag, run, evaluation, sweep, split, ids, checked


def project_query(record):
    result = picked(record, QUERY_FIELDS)
    if "caption_sha256" in result:
        digest(result["caption_sha256"])
    safe_tree(result)
    return result


def project_document(original, name, source_sha, evaluation_manifest_sha, namespace="evaluation"):
    result = {"_projection": {"schema": "pare.public-result-projection.v1", "not_original_json": True,
              "source_artifact": namespace + "/" + name, "source_json_sha256": digest(source_sha),
              "source_evaluation_manifest_sha256": digest(evaluation_manifest_sha),
              "bindings": "projection_manifest.json", "caption_body_removed": True,
              "machine_paths_removed": True}, **picked(original, REPORT_FIELDS)}
    result["query_protocol"] = {**picked(original["query_protocol"], QUERY_PROTOCOL_FIELDS),
        "selection": "First caption of every test image, in original CSV encounter order",
        "positive_definition": "One official image ID per query",
        "weighting": "One equally weighted query per gallery image"}
    result["queries"] = [project_query(row) for row in original["queries"]]
    if "provenance" in original:
        provenance = original["provenance"]
        result["provenance"] = picked(provenance, REPORT_HASH_FIELDS)
        for value in result["provenance"].values():
            digest(value)
        if "cache_signature" in provenance:
            result["provenance"]["cache_signature"] = signature(provenance["cache_signature"])
        result["provenance"]["public_bindings"] = "projection_manifest.json"
    safe_tree(result)
    return result


def project_stress_document(original, name, source_sha, manifest_sha):
    fields = ("spacing_factor", "is_real_gsd_validation", "n_queries", "n_gallery", "metrics_percent",
              "delta_percentage_points_from_original", "observational_original_raster_spacing_subgroups")
    if "query_protocol" in original:
        result = project_document(original, name, source_sha, manifest_sha, "stress")
    else:
        result = {"_projection": {"schema": "pare.public-result-projection.v1", "not_original_json": True,
            "source_artifact": "stress/" + name, "source_json_sha256": digest(source_sha),
            "source_evaluation_manifest_sha256": digest(manifest_sha), "bindings": "projection_manifest.json",
            "caption_body_removed": True, "machine_paths_removed": True}}
    result.update(picked(original, fields))
    safe_tree(result)
    return result


def provenance_bindings(tag, run, evaluation, sweep, split, ids, split_dir, training_run):
    manifest = evaluation["manifest"]
    inputs = {name: {"artifact_alias": "inputs/" + name,
                     "basename": Path(entry["path"]).name, "sha256": digest(entry["sha256"])}
              for name, entry in manifest["inputs"].items()}
    sources = {Path(name).name: digest(entry["sha256"]) for name, entry in manifest["code"].items()}
    require(len(sources) == len(manifest["code"]), "Evaluator source basename collision")
    heads = {}
    for variant in exports.VARIANTS:
        head = manifest["artifact_validation"]["heads"][variant]
        provenance = head["training_provenance"]
        heads[variant] = {"artifact": inputs["head_" + variant], "ablate": head["ablate"],
                          "n_images": head["n_images"], "n_pairs": head["n_pairs"],
                          "config": picked(head["config"], TRAINING_CONFIG_FIELDS), "completed_epochs": len(head["history"]),
                          "training_ids": ids_binding(provenance["train_ids"]),
                          "training_hashes": {key: digest(provenance[key]) for key in HEAD_HASH_FIELDS},
                          "cache_signature": signature(provenance["cache_signature"])}
    trials = []
    trial_dir = Path(sweep["trial_directory"])
    for row in sweep["grid"]:
        filename = f"lr_{row['lr']:g}_kappa_{row['kappa']:g}.json"
        path = trial_dir / filename
        raw = exports.read(path)
        trials.append({"validation_metrics": row, "source_artifact": "selection_trials/" + filename,
                       "source_json_sha256": validator.sha(path), "checkpoint_sha256": digest(raw["checkpoint_sha256"]),
                       "query_ids": ids_binding(raw["query_ids"]), "gallery_ids": ids_binding(raw["gallery_ids"]),
                       "ranks_sha256": object_hash(raw["ranks"])})
    return {"model": {"tag": tag, "formal_config_tag": run["config"]["tag"], **picked(run["config"], ("model", "pretrained", "remoteclip_sha256"))},
            "evaluator_code_sha256": sources, "evaluator_inputs": inputs, "final_heads": heads,
            "selection": {"source_artifact": "training/hparam_best.json",
                          "source_json_sha256": validator.sha(training_run / "hparam_best.json"),
                          "rule": "Maximum validation R@10; tie-break MRR@10; first original grid row on exact tie",
                          "best": sweep["best"], "epochs_per_trial": sweep["epochs"],
                          "n_validation_queries": sweep["n_val_q"], "n_validation_gallery": sweep["n_val_gallery"],
                          "fit_ids": ids_binding(sweep["provenance"]["train_ids"]),
                          "validation_ids": ids_binding(sweep["provenance"]["validation_ids"]),
                          "provenance_hashes": {key: digest(sweep["provenance"][key]) for key in HEAD_HASH_FIELDS},
                          "selection_source_sha256": digest(sweep["provenance"]["selection_source_sha256"]),
                          "trials": trials},
            "split": {"source_artifact": "split/manifest.json", "source_json_sha256": validator.sha(split_dir / "manifest.json"),
                      "protocol": split["protocol"], "seed": split["seed"], "split_counts": split["split_counts"],
                      "content_disjoint_train_test": True, "content_disjoint_fit_val": True,
                      "official_test_duplicate_content_group_count": 0,
                      "files": hash_map(split["output_sha256"]),
                      "ids": {name: {**ids_binding(values, split["output_sha256"][name + "_ids.json"]), "values": values}
                              for name, values in ids.items()}},
            "id_hash_encoding": "SHA256 of JSON: sorted object keys, compact separators, ensure_ascii=True; list order retained"}


def create_projection(evaluation_dir, training_run, split_dir, out, stress_dir=None):
    evaluation_dir, training_run, split_dir, out = [Path(path).resolve() for path in (evaluation_dir, training_run, split_dir, out)]
    require(not out.exists(), "Refusing to overwrite an existing projection")
    tag, run, evaluation, sweep, split, ids, checked = validate_originals(evaluation_dir, training_run, split_dir)
    evaluation_sha = checked[evaluation_dir / "independent_manifest.json"]
    names = [f"eval_{variant}_k{k}.json" for variant in exports.VARIANTS for k in exports.KS]
    names += ["eval_baselines.json", "queries.json"]
    projections = {}
    projected_sources = {}
    for name in names:
        path = evaluation_dir / name
        source_sha = evaluation["manifest"]["outputs"][name]["sha256"]
        validator.assert_hash(path, source_sha)
        checked[path] = source_sha
        projections[name] = project_document(exports.read(path), name, source_sha, evaluation_sha)
        projected_sources[name] = (path, "evaluation/" + name)
    bindings = provenance_bindings(tag, run, evaluation, sweep, split, ids, split_dir, training_run)
    stress_manifest_sha = None
    if stress_dir is not None:
        stress_dir = Path(stress_dir).resolve()
        exports.load_stress(tag, stress_dir, evaluation)
        stress_manifest_path = stress_dir / "manifest.json"
        stress_manifest = exports.read(stress_manifest_path)
        stress_manifest_sha = validator.sha(stress_manifest_path)
        checked[stress_manifest_path] = stress_manifest_sha
        stress_inputs = {}
        for name, entry in stress_manifest["inputs"].items():
            path, expected = Path(entry["path"]), digest(entry["sha256"])
            validator.assert_hash(path, expected)
            checked[path] = expected
            stress_inputs[name] = {"artifact_alias": "stress_inputs/" + name, "basename": path.name, "sha256": expected}
        for name in ("eval_spacing_1.json", "eval_spacing_1p5.json", "eval_spacing_2.json", "comparison.json", "queries.json"):
            path = stress_dir / name
            expected = stress_manifest["outputs"][name]["sha256"]
            validator.assert_hash(path, expected)
            checked[path] = expected
            output_name = "stress_" + name
            projections[output_name] = project_stress_document(exports.read(path), name, expected, stress_manifest_sha)
            projected_sources[output_name] = (path, "stress/" + name)
        bindings["synthetic_scale"] = {
            "source_manifest_sha256": stress_manifest_sha, "is_real_gsd_validation": False,
            "conditions": [1, 1.5, 2], "n_queries": 3688, "n_gallery": 3688,
            "original_scale_ranks_equal_to_primary_evaluation": True,
            "inputs": stress_inputs, "source_sha256": hash_map(stress_manifest.get("source_sha256", {})),
            "scope": "Same fixed original-scale heads; one Lanczos downsample without manual upsample; no claim of acquisition-GSD invariance"}
    manifest = {"schema": "pare.public-projection-manifest.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "tag": tag, "not_original_result_files": True,
        "purpose": "Caption-redacted and machine-path-redacted projections, with original-source hash bindings",
        "internal_exporter_compatibility": "Use original complete results for internal validation and paper export, not these projections",
        "source_manifests": {
            "evaluation/independent_manifest.json": evaluation_sha,
            "training/run_manifest.json": checked[training_run / "run_manifest.json"],
            "split/manifest.json": checked[split_dir / "manifest.json"]},
        "original_training_manifest_status": run["status"],
        "training_status_note": "The original runner may remain partial because the completed evaluation was executed independently; the selection and all four final-head steps were verified complete",
        "validation": {"original_eval_json_count": 25, "projected_json_count": len(projections), "n_queries": 3688, "n_gallery": 3688,
                       "selection_trials": 9, "validation_queries": 918, "final_heads": 4,
                       "optional_stress_evaluation_json_count": 3 if stress_dir else 0,
                       "all_original_result_output_hashes_checked": True, "actual_evaluator_input_hashes_checked": True,
                       "actual_final_head_hashes_checked": True, "split_file_hashes_checked": True,
                       "query_metrics_recomputed_from_ranks": True},
        "generator_sha256": {"public_projection.py": validator.sha(__file__),
                              "export_paper_results.py": EXPORTER_SHA, "evaluation_controller.py": VALIDATOR_SHA},
        "aliases": {"evaluation": "Supplied completed independent-evaluation directory",
                    "training": "Supplied corresponding training-run directory",
                    "split": "Supplied corresponding content-disjoint split directory",
                    "inputs": "Original evaluated inputs; no weights, caches, or caption CSVs are copied",
                    "selection_trials": "Original validation trial records; only hashes and bindings are published here",
                    "stress": "Optional completed fixed-head synthetic-scale evaluation directory",
                    "stress_inputs": "Original scale-test inputs; not copied"},
        "bindings": bindings, "projection_files": {}}
    if stress_manifest_sha:
        manifest["source_manifests"]["stress/manifest.json"] = stress_manifest_sha
    safe_tree(manifest)
    for path, expected in checked.items():
        validator.assert_hash(path, expected)
    out.mkdir(parents=True, exist_ok=False)
    for name, projected in projections.items():
        path = out / name
        path.write_text(json.dumps(projected, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
        source_path, source_alias = projected_sources[name]
        manifest["projection_files"][name] = {"source_artifact": source_alias,
            "source_sha256": checked[source_path], "projection_sha256": validator.sha(path),
            "projection_bytes": path.stat().st_size}
    (out / "projection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    (out / "README.txt").write_text(
        "These files are sanitized projections, NOT the original evaluator JSON files.\n"
        "Caption bodies and absolute machine paths are omitted. Caption SHA-256 digests, image/query IDs, ranks and metrics are retained.\n"
        "projection_manifest.json maps every original JSON SHA-256 to its distinct projection SHA-256 and binds model, final heads, selection and split IDs.\n"
        "No model weights, cache/score matrices, source caption CSVs or original machine-path manifests are copied.\n"
        "Internal paper-result exporters must read the original full result directories. These projections are for public inspection only.\n"
        "Optional stress_* files are fixed-head synthetic-scale projections, not new acquisition-GSD measurements.\n"
        "A partial original training-run manifest is permissible only because evaluation ran independently; the sweep and all final heads were verified complete.\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stress", type=Path, help="Optional corresponding completed scale-evaluation directory")
    args = parser.parse_args()
    manifest = create_projection(args.evaluation, args.training_run, args.split_dir, args.out, args.stress)
    print(json.dumps({"status": "complete", "tag": manifest["tag"], "projection_json_count": len(manifest["projection_files"]),
                      "projection_manifest_sha256": validator.sha(args.out / "projection_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
