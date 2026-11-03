#!/usr/bin/env python3
"""Export verified, completed PARE experiments; never score or select test runs.

All five tag=directory and tag=sweep.json assignments are explicit. This script
uses only the standard library and the adjacent read-only result validator.
It refuses incomplete evaluations and never creates placeholder paper values.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import evaluation_controller as validator

TAGS = tuple(validator.BACKBONES)
VARIANTS = validator.VARIANTS
KS = validator.KS
METRICS = ("R@1", "R@5", "R@10", "MRR@10")
DISPLAY = {"vitb32": "CLIP ViT-B/32", "vitb16": "CLIP ViT-B/16",
           "remoteclip": "RemoteCLIP", "siglip": "SigLIP", "georsclip": "GeoRSCLIP"}
METHODS = {"full": "PARE", "stats_off": "PARE-stats-off",
           "cfar_strict": "PARE-CFAR-strict", "uniform_grid": "Uniform-grid"}
N = 3688


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    # Python otherwise accepts NaN/Infinity extensions in JSON.
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def bindings(values, all_five=False):
    result = {}
    for value in values:
        tag, sep, path = value.partition("=")
        require(sep and tag in TAGS and path, "Use a known tag=path assignment")
        require(tag not in result, "Duplicate tag assignment: " + tag)
        result[tag] = Path(path).resolve()
    if all_five:
        require(set(result) == set(TAGS), "Exactly the five prespecified backbones are required")
    require(len(set(result.values())) == len(result), "A result path cannot represent multiple backbones")
    return result


def verify_outputs(directory, manifest, required):
    require(manifest.get("status") == "complete" and manifest.get("input_code_recheck_passed") is True,
            f"Refusing incomplete/unverified evaluation: {directory}")
    require(manifest.get("config", {}).get("smoke") is False, "Smoke results are not paper results")
    outputs = manifest.get("outputs", {})
    require(set(required) <= set(outputs), "Required outputs are not bound by the completed manifest")
    for relative, entry in outputs.items():
        path = directory / relative
        require(Path(relative).name == relative, "Output manifest must use direct child filenames")
        validator.assert_hash(path, entry["sha256"])
        require(path.stat().st_size == entry["bytes"], "Output length mismatch")


def check_model(config, tag):
    _, model, pretrained = validator.BACKBONES[tag]
    require((config.get("model"), config.get("pretrained")) == (model, pretrained),
            "Model/pretrained differs from explicit backbone assignment: " + tag)
    require(bool(config.get("remoteclip")) == (tag in ("remoteclip", "georsclip")),
            "External checkpoint configuration differs: " + tag)
    if config.get("remoteclip"):
        expected = "RemoteCLIP-ViT-B-32.pt" if tag == "remoteclip" else "RS5M_ViT-B-32.pt"
        require(Path(config["remoteclip"]).name == expected, "External checkpoint identity differs")


def query_contract(document):
    protocol = document["query_protocol"]
    records, gallery = document["queries"], document["gallery_ids"]
    require(len(gallery) == len(set(gallery)) == len(records) == N, "Expected 3688 unique gallery/query images")
    require(protocol.get("mode") == "first-caption" and protocol.get("truncated") is False
            and protocol.get("max_queries") == 0 and not protocol.get("gallery_images_without_evaluated_query"),
            "Only complete first-caption test results are accepted")
    require(len({row["query_id"] for row in records}) == N, "Duplicate query IDs")
    require({row["image_id"] for row in records} == set(gallery), "Queries do not cover the gallery once")
    for row in records:
        require(row["caption_index_one_based"] == 1 and row["query_id"] == row["image_id"] + "#caption=1",
                "Unexpected first-caption identity")
        require(type(row["gold_gallery_row"]) is int and 0 <= row["gold_gallery_row"] < N
                and gallery[row["gold_gallery_row"]] == row["image_id"], "Gold/gallery mismatch")
    return records, gallery


def load_evaluation(tag, directory):
    manifest = read(directory / "independent_manifest.json")
    required = [f"eval_{variant}_k{k}.json" for variant in VARIANTS for k in KS]
    required += ["eval_baselines.json", "queries.json"]
    expected_files = required + [f"scores_{variant}.npy" for variant in VARIANTS]
    expected_files += ["query_embeddings.npy"]
    expected_files += [f"baseline_scores_{method}.npy" for method in ("scene", "mean_pool", "MaxSim")]
    expected_files += [f"{kind}_{mode}.{extension}" for mode in ("physics", "uniform_grid")
                       for kind, extension in (("scene_scores", "npy"), ("candidates", "npz"))]
    verify_outputs(directory, manifest, expected_files)
    require(manifest.get("schema") == "pare.independent-multik.v1", "Unknown evaluation schema")
    require(manifest.get("expected_eval_json_count") == 25, "Incomplete planned result matrix")
    check_model(manifest["config"], tag)
    require(manifest["code"]["vmap_helper"]["sha256"] == validator.HELPER_SHA, "Unreviewed scoring helper")
    # Recomputes every reported metric from ranks, including natural misses.
    summary = validator.validate_results(directory)
    document = read(directory / "queries.json")
    records, gallery = query_contract(document)
    identity_keys = tuple(records[0])
    identity = [{key: row[key] for key in identity_keys} for row in records]
    reports = {}
    for filename in required:
        if filename == "queries.json":
            continue
        report = read(directory / filename)
        require(report["seed"] == 0 and report["gallery_ids"] == gallery, "Evaluation seed/gallery differs")
        require(report["query_protocol"] == document["query_protocol"], "Query protocol differs across outputs")
        require([{key: row[key] for key in identity_keys} for row in report["queries"]] == identity,
                "Query identity/order differs across outputs")
        provenance = report["provenance"]
        require(provenance["captions_sha256"] == manifest["inputs"]["captions"]["sha256"]
                and provenance["evaluator_sha256"] == validator.EVALUATOR_SHA
                and provenance["allow_legacy_artifacts"] is False, "Report/input provenance mismatch")
        if filename != "eval_baselines.json":
            variant = filename[5:filename.rindex("_k")]
            mode = "uniform_grid" if variant == "uniform_grid" else "physics"
            require(provenance["head_sha256"] == manifest["inputs"]["head_" + variant]["sha256"]
                    and provenance["cache_sha256"] == manifest["inputs"][mode + "_cache"]["sha256"],
                    "Head/cache binding differs across outputs")
        if filename in ("eval_baselines.json", "eval_full_k3688.json", "eval_stats_off_k3688.json"):
            reports[filename] = report
    return {"tag": tag, "directory": directory, "manifest": manifest, "summary": summary,
            "records": records, "gallery": gallery, "reports": reports}


def sweep_best(grid):
    expected = {(lr, kappa) for lr in (0.0005, 0.001, 0.002) for kappa in (0.03, 0.05, 0.07)}
    require(len(grid) == 9 and {(row["lr"], row["kappa"]) for row in grid} == expected,
            "The prespecified complete nine-point validation grid is required")
    for row in grid:
        for name in METRICS:
            value = row["val_" + name]
            require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 100,
                    "Invalid validation metric")
    return max(grid, key=lambda row: (row["val_R@10"], row["val_MRR@10"]))


def load_sweep(tag, path, evaluation):
    report = read(path)
    run_path = path.parent / "run_manifest.json"
    run = read(run_path)
    steps = {step["name"]: step for step in run["steps"]}
    # The legacy runner's overall status stays 'partial' because evaluation was
    # performed externally. Only its completed sweep and train steps are used.
    require(steps["sweep"].get("status") == "complete", "Sweep is still running/partial")
    require(not run["config"].get("smoke") and run["config"]["tag"] == validator.FORMAL_CONFIG_TAGS[tag], "Wrong training run")
    check_model(run["config"], tag)
    check_model(report["config"], tag)
    recorded_files = steps["sweep"].get("artifact_sha256", {})
    require(str(path) in recorded_files, "Sweep JSON is not bound by completed training step")
    for filename, digest in recorded_files.items():
        validator.assert_hash(filename, digest)
    require(report["epochs"] == report["config"]["epochs"] == 10
            and report["n_val_q"] == report["n_val_gallery"] == 918,
            "Expected 10 epochs and all 918 validation queries/gallery images")
    require(report["config"]["seed"] == 0 and report["config"]["val_q"] == 0,
            "Validation seed/query truncation differs")
    best = sweep_best(report["grid"])
    require(report["best"] == best, "Stored selection violates max validation R@10, then MRR@10")
    provenance = report["provenance"]
    fit, val = provenance["train_ids"], provenance["validation_ids"]
    require(len(fit) == len(set(fit)) == 8263 and len(val) == len(set(val)) == 918
            and not set(fit) & set(val) and not (set(fit) | set(val)) & set(evaluation["gallery"]),
            "Invalid fit/validation/test partition")
    require(provenance["batch_scoring_sha256"] == validator.HELPER_SHA, "Sweep scoring helper differs")
    require(provenance["cache_signature"] == evaluation["manifest"]["artifact_validation"]["cache_signatures"]["physics"],
            "Sweep and evaluation backbone/primitive signatures differ")
    trial_dir = Path(report["trial_directory"])
    require(len(list(trial_dir.glob("*.json"))) == len(list(trial_dir.glob("*.pt"))) == 9,
            "Missing validation trial ranks/checkpoints")
    for row in report["grid"]:
        stem = f"lr_{row['lr']:g}_kappa_{row['kappa']:g}"
        raw_path, head_path = trial_dir / (stem + ".json"), trial_dir / (stem + ".pt")
        require(str(raw_path) in recorded_files and str(head_path) in recorded_files,
                "Trial is not bound by completed sweep")
        raw = read(raw_path)
        require(raw["gallery_ids"] == val and len(raw["query_ids"]) == 918
                and set(raw["query_ids"]) == set(val) and len(raw["ranks"]) == 918,
                "Validation trial query/gallery population differs")
        require(all(type(rank) is int and 1 <= rank <= 918 for rank in raw["ranks"]), "Invalid validation rank")
        require(raw["checkpoint_sha256"] == recorded_files[str(head_path)], "Trial checkpoint binding differs")
        validator.compare_metrics({name: row["val_" + name] for name in METRICS}, validator.metric(raw["ranks"]))
    for variant in VARIANTS:
        step = steps["train_" + variant]
        require(step["status"] == "complete", "Final training is incomplete")
        expected = str(path.parent / f"head_{variant}.pt")
        digest = step["artifact_sha256"].get(expected)
        require(digest and digest == evaluation["manifest"]["inputs"]["head_" + variant]["sha256"],
                "Evaluation does not use this sweep's final trained head")
        head = evaluation["manifest"]["artifact_validation"]["heads"][variant]
        require(head["config"]["lr"] == best["lr"] and head["config"]["kappa"] == best["kappa"],
                "Final head does not use the validation-selected settings")
        require(set(head["training_provenance"]["train_ids"]) == set(fit) | set(val),
                "Final head was not trained on the declared retained training set")
        require(head["n_images"] == 9181 and head["n_pairs"] == 31945
                and head["config"]["epochs"] == len(head["history"]) == 10, "Incomplete final head")
    return report, {"path": str(path), "sha256": validator.sha(path),
                    "training_manifest": str(run_path), "training_manifest_sha256": validator.sha(run_path)}


def load_stress(tag, directory, evaluation):
    manifest = read(directory / "manifest.json")
    required = ["queries.json", "comparison.json", "eval_spacing_1.json", "eval_spacing_1p5.json", "eval_spacing_2.json"]
    verify_outputs(directory, manifest, required)
    require(manifest["schema"] == "pare.synthetic-spacing-evaluation.v1"
            and manifest["is_real_gsd_validation"] is False, "Unknown scale experiment contract")
    check_model(manifest["config"], tag)
    reference = evaluation["manifest"]["inputs"]
    for stress_key, reference_key in (("head", "head_full"), ("stats_off_head", "head_stats_off"),
                                      ("captions", "captions"), ("reference_cache", "physics_cache")):
        require(manifest["inputs"][stress_key]["sha256"] == reference[reference_key]["sha256"],
                "Scale experiment does not use the corresponding original experiment inputs")
    require(manifest["inputs"]["evaluator"]["sha256"] == validator.EVALUATOR_SHA
            and manifest["inputs"]["scoring_helper"]["sha256"] == validator.HELPER_SHA,
            "Scale experiment numerical core differs")
    document = read(directory / "queries.json")
    records, gallery = query_contract(document)
    require(records == evaluation["records"] and gallery == evaluation["gallery"],
            "Scale experiment query/gallery identity or order differs")
    result = {}
    methods = ("Scene-CLIP", "PARE", "PARE-stats-off")
    for factor in ("1", "1.5", "2"):
        report = read(directory / ("eval_spacing_" + factor.replace(".", "p") + ".json"))
        require(report["n_queries"] == report["n_gallery"] == len(report["queries"]) == N
                and report["spacing_factor"] == factor and report["is_real_gsd_validation"] is False,
                "Wrong scale-test population/condition")
        require(report["query_protocol"] == document["query_protocol"], "Scale query protocol differs")
        require(set(report["metrics_percent"]) == set(methods), "All three prespecified scale methods are required")
        for raw, query in zip(report["queries"], records):
            require(all(raw[key] == query[key] for key in ("query_id", "image_id", "gold_gallery_row")),
                    "Scale result query order differs")
            require(set(raw["ranks"]) == set(methods)
                    and all(type(rank) is int and 1 <= rank <= N for rank in raw["ranks"].values()),
                    "Invalid scale full-gallery ranks")
        for method in methods:
            ranks = [raw["ranks"][method] for raw in report["queries"]]
            validator.compare_metrics(report["metrics_percent"][method], validator.metric(ranks))
            if factor == "1":
                if method == "Scene-CLIP":
                    original = [raw["full_gallery_ranks"]["scene"] for raw in evaluation["reports"]["eval_baselines.json"]["queries"]]
                else:
                    variant = "full" if method == "PARE" else "stats_off"
                    original = [raw["natural_PARE_rank"] for raw in evaluation["reports"][f"eval_{variant}_k3688.json"]["queries"]]
                require(ranks == original, "Original-scale ranks differ from formal evaluation")
        result[factor] = report
    comparison = read(directory / "comparison.json")
    require(comparison["n_queries"] == comparison["n_gallery"] == N
            and comparison["is_real_gsd_validation"] is False, "Scale comparison dimensions differ")
    for factor, report in result.items():
        require(comparison["metrics_percent"][factor] == report["metrics_percent"], "Scale comparison metrics differ")
        if factor != "1":
            for method in methods:
                for metric in METRICS:
                    expected = report["metrics_percent"][method][metric] - result["1"]["metrics_percent"][method][metric]
                    require(abs(comparison["delta_percentage_points_from_original"][factor][method][metric] - expected) < 1e-9,
                            "Scale delta differs")
    return result


def metric_row(tag, method, protocol, values, **extra):
    return {"backbone": tag, "method": method, "protocol": protocol, "n_gallery": N,
            "n_queries": N, **values, **extra}


def build_tables(evaluations, sweeps, stresses):
    tables = {key: [] for key in ("table_i_full_gallery", "table_ii_vitb32_controls", "appendix_natural_k",
                                "appendix_hard50", "appendix_controlled100", "appendix_validation_grid",
                                "validation_selected", "pare_differences")}
    for tag in TAGS:
        evaluation = evaluations[tag]
        summary = evaluation["summary"]
        baseline = summary["baselines"]
        full = {METHODS[v]: summary["variants"][v]["3688"]["natural"]["PARE_end_to_end"] for v in VARIANTS}
        for name, key in (("Scene-CLIP", "scene"), ("Mean-pool", "mean_pool"), ("MaxSim", "MaxSim")):
            tables["table_i_full_gallery"].append(metric_row(tag, name, "full_gallery", baseline["full_gallery"][key]))
        for name in ("PARE-stats-off", "PARE"):
            tables["table_i_full_gallery"].append(metric_row(tag, name, "full_gallery", full[name]))
        for variant in VARIANTS:
            name = METHODS[variant]
            reports = summary["variants"][variant]
            controlled, hard = reports["100"]["controlled"], reports["100"]["hard"]
            tables["appendix_controlled100"].append(metric_row(tag, name, "controlled100_positive_inserted", controlled["PARE"],
                                                              candidate_count=100, pools_per_query=5, seed=0))
            tables["appendix_hard50"].append(metric_row(tag, name, "hard50_positive_inserted", hard["PARE"], candidate_count=50))
            for k in KS:
                natural = reports[str(k)]["natural"]
                row = metric_row(tag, name, "full_gallery" if k == N else "natural_shortlist_end_to_end",
                                 natural["PARE_end_to_end"], K=k, candidate_recall_percent=natural["candidate_recall_percent"],
                                 n_included=natural["n_natural_inclusions"],
                                 **{"conditional_" + key: (natural["PARE_conditional_on_inclusion"] or {}).get(key) for key in METRICS})
                tables["appendix_natural_k"].append(row)
            if tag == "vitb32":
                natural = reports["100"]["natural"]
                tables["table_ii_vitb32_controls"].append({"backbone": tag, "method": name,
                    "controlled100_R@10": controlled["PARE"]["R@10"], "controlled100_MRR@10": controlled["PARE"]["MRR@10"],
                    "natural100_candidate_recall_percent": natural["candidate_recall_percent"],
                    "natural100_R@10": natural["PARE_end_to_end"]["R@10"], "natural100_MRR@10": natural["PARE_end_to_end"]["MRR@10"],
                    "full_gallery_R@10": full[name]["R@10"], "full_gallery_MRR@10": full[name]["MRR@10"]})
        for name, key in (("Scene-CLIP", "scene"), ("Mean-pool", "mean_pool"), ("MaxSim", "MaxSim")):
            tables["appendix_controlled100"].append(metric_row(tag, name, "controlled100_positive_inserted", baseline["controlled_random_pools"][key],
                                                              candidate_count=100, pools_per_query=5, seed=0))
            tables["appendix_hard50"].append(metric_row(tag, name, "hard50_positive_inserted", baseline["appearance_adversarial_pool"][key], candidate_count=50))
        sweep = sweeps[tag]
        for index, row in enumerate(sweep["grid"]):
            tables["appendix_validation_grid"].append({"backbone": tag, "trial_index_one_based": index + 1, **row,
                "selected": row == sweep["best"], "epochs": 10, "n_fit_images": 8263, "n_val_queries": 918, "n_val_gallery": 918})
        tables["validation_selected"].append({"backbone": tag, **sweep["best"], "selection": "maximum val_R@10; tie-break val_MRR@10; first in grid on exact tie"})
        for name, values in (("PARE-stats-off", full["PARE-stats-off"]), ("Scene-CLIP", baseline["full_gallery"]["scene"])):
            tables["pare_differences"].append({"backbone": tag, "comparison": "PARE minus " + name, "protocol": "full_gallery",
                **{"delta_" + key + "_pp": full["PARE"][key] - values[key] for key in METRICS}})
    if stresses:
        tables["synthetic_scale_sensitivity"] = []
        for tag in TAGS:
            if tag not in stresses:
                continue
            for factor in ("1", "1.5", "2"):
                for method in ("Scene-CLIP", "PARE", "PARE-stats-off"):
                    values = stresses[tag][factor]["metrics_percent"][method]
                    tables["synthetic_scale_sensitivity"].append(metric_row(tag, method, "synthetic_scale_full_gallery", values,
                        spacing_factor=float(factor), is_real_gsd_validation=False,
                        delta_R10_pp=values["R@10"] - stresses[tag]["1"]["metrics_percent"][method]["R@10"]))
    return tables


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(character, character) for character in str(value))


def latex_table(rows, columns, note):
    # No boldface or winners are selected; the root can arrange the layout.
    lines = ["% " + note, "% All retrieval metrics are percentages; differences are percentage points.",
             r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
             " & ".join(tex_escape(label) for _, label in columns) + r" \\", r"\hline"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                value = "--"
            elif key == "backbone":
                value = DISPLAY[value]
            elif type(value) is float:
                value = f"{value:g}" if key in ("lr", "kappa", "spacing_factor") else f"{value:.2f}"
            values.append(tex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines + [r"\hline", r"\end{tabular}", ""])


def write_exports(directory, tables, provenance):
    directory.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        with (directory / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory / "paper_results.json").write_text(json.dumps({"units": "percent; deltas in percentage points", "tables": tables}, indent=2, allow_nan=False) + "\n")
    common = [("backbone", "Backbone"), ("method", "Method")]
    four = [(key, key) for key in METRICS]
    specs = {
        "table_i_full_gallery": (common + four, "Complete 3688-image gallery; 3688 first-caption queries."),
        "table_ii_vitb32_controls": ([("method", "Method")] + [(key, key) for key in tables["table_ii_vitb32_controls"][0] if key not in ("method", "backbone")], "B/32 controls. Controlled100 explicitly inserts the positive; natural100 does not."),
        "appendix_natural_k": (common + [("K", "K"), ("candidate_recall_percent", "Candidate recall")] + four + [("conditional_R@10", "Conditional R@10")], "Natural candidate shortlist; misses count as failures in the main metrics. K=3688 is full gallery."),
        "appendix_hard50": (common + four, "Positive plus 49 strongest Scene-CLIP negatives; conditional diagnostic, not end-to-end retrieval."),
        "appendix_controlled100": (common + four, "Positive plus 99 random negatives; five seed-0 pools per query; conditional diagnostic, not end-to-end retrieval."),
        "appendix_validation_grid": ([("backbone", "Backbone"), ("lr", "Learning rate"), ("kappa", "Kappa")] + [("val_" + key, "Val " + key) for key in METRICS] + [("selected", "Selected")], "All nine 10-epoch grid trials on 918 validation images; no test-based selection."),
        "validation_selected": ([("backbone", "Backbone"), ("lr", "Learning rate"), ("kappa", "Kappa"), ("val_R@10", "Val R@10"), ("val_MRR@10", "Val MRR@10")], "Prespecified maximum validation R@10, then MRR@10; first grid row on exact ties."),
        "pare_differences": ([("backbone", "Backbone"), ("comparison", "Comparison")] + [("delta_" + key + "_pp", "Delta " + key) for key in METRICS], "Signed full-gallery PARE differences; negative values retained."),
    }
    if "synthetic_scale_sensitivity" in tables:
        specs["synthetic_scale_sensitivity"] = (common + [("spacing_factor", "Spacing factor"), ("R@1", "R@1"), ("R@10", "R@10"), ("MRR@10", "MRR@10"), ("delta_R10_pp", "Delta R@10")], "Synthetic single-downsample scale sensitivity, not real acquisition-GSD validation; fixed original-scale heads.")
    for name, (columns, note) in specs.items():
        (directory / (name + ".tex")).write_text(latex_table(tables[name], columns, note))
    (directory / "README.txt").write_text(
        "Verified numeric exports for layout, not a compiled manuscript.\n"
        "No best test run or backbone is selected. The five explicit input assignments are exported in fixed order.\n"
        "CSV/JSON retain source precision. TeX rounds metrics to two decimals only.\n"
        "R@1, R@5, R@10 and MRR@10 are percentages; signed deltas are percentage points.\n"
        "Controlled100 and hard50 explicitly include the positive. They are not end-to-end natural retrieval.\n"
        "Natural shortlist metrics count candidate misses as failures; conditional columns are labeled separately.\n"
        "Synthetic scale tests, when supplied, use fixed original-scale heads and do not establish acquisition-GSD robustness.\n"
        "No scale table is emitted when scale results are not supplied. No placeholder values are emitted.\n")
    provenance.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                      outputs={path.name: {"sha256": validator.sha(path), "bytes": path.stat().st_size}
                               for path in sorted(directory.iterdir()) if path.is_file()})
    (directory / "export_manifest.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, metavar="TAG=DIR")
    parser.add_argument("--sweep", action="append", required=True, metavar="TAG=JSON")
    parser.add_argument("--stress", action="append", default=[], metavar="TAG=DIR")
    parser.add_argument("--out", type=Path, required=True, help="New directory only")
    args = parser.parse_args()
    evaluations = bindings(args.evaluation, all_five=True)
    sweeps = bindings(args.sweep, all_five=True)
    stresses = bindings(args.stress)
    require(not args.out.exists(), "Refusing to overwrite an existing export directory")
    loaded, sweep_reports, stress_reports = {}, {}, {}
    provenance = {"schema": "pare.paper-numeric-export.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "command": sys.argv, "selection_policy": "All five explicit backbones; no test-result selection",
                  "source_sha256": {Path(__file__).name: validator.sha(__file__),
                                    "evaluation_controller.py": validator.sha(validator.__file__)},
                  "evaluations": {}, "sweeps": {}, "stresses": {}}
    for tag in TAGS:
        print("Verifying completed evaluation and validation sweep: " + tag, flush=True)
        loaded[tag] = load_evaluation(tag, evaluations[tag])
        if tag != TAGS[0]:
            require(loaded[tag]["records"] == loaded[TAGS[0]]["records"]
                    and loaded[tag]["gallery"] == loaded[TAGS[0]]["gallery"], "Backbones use different test populations")
        sweep_reports[tag], provenance["sweeps"][tag] = load_sweep(tag, sweeps[tag], loaded[tag])
        provenance["evaluations"][tag] = {"path": str(evaluations[tag]),
            "manifest_sha256": validator.sha(evaluations[tag] / "independent_manifest.json")}
        if tag in stresses:
            stress_reports[tag] = load_stress(tag, stresses[tag], loaded[tag])
            provenance["stresses"][tag] = {"path": str(stresses[tag]), "manifest_sha256": validator.sha(stresses[tag] / "manifest.json")}
    tables = build_tables(loaded, sweep_reports, stress_reports)
    # Export files are created only after every supplied result passes validation.
    write_exports(args.out.resolve(), tables, provenance)
    print(json.dumps({"status": "complete", "out": str(args.out.resolve()),
                      "table_rows": {key: len(value) for key, value in tables.items()}}, indent=2))


if __name__ == "__main__":
    main()
