#!/usr/bin/env python3
"""Summarize only the complete, audited 42-run exploratory validation study.

Standard library only. The separate pinned audit reads the torch checkpoints
and replays matrix ranks. This program binds its exact files, independently
checks identities/shuffles and rank-derived metrics, then applies the frozen
contrast coefficients without selecting a model or a favorable result.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

PROTOCOL_SHA = "42789ee8793b7c5bd59bf950dd8deefe8bf1c3caa2e9be2d54143630f63bf288"
FREEZE_SHA = "3ebc187c1f9b67256747bc46e5383d43a166fdb11cb5441aea2660a937aa011a"
VERIFIER_SHA = "117ddbccaaf2802951aa149a105f08e0ca595b1054b20c4b65515a2d308e0045"
METRICS = ("R@1", "R@5", "R@10", "MRR@10")
ARTIFACTS = {"head.pt", "validation_scores.pt", "validation.json", "training_progress.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def hash_bound(path, expected, label):
    require(sha(path) == expected, "Hash mismatch: " + label)


def digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def check_metrics(actual, expected, label):
    require(set(actual) == set(METRICS), "Unexpected metric set: " + label)
    for key in METRICS:
        value = actual[key]
        require(type(value) in (float, int) and math.isfinite(value)
                and abs(value - expected[key]) < 1e-9, "Rank/metric disagreement: " + label + "/" + key)


def from_ranks(ranks):
    require(len(ranks) == 918 and all(type(r) is int and 1 <= r <= 918 for r in ranks),
            "Expected 918 valid full-gallery ranks")
    result = {f"R@{k}": 100 * sum(r <= k for r in ranks) / 918 for k in (1, 5, 10)}
    result["MRR@10"] = 100 * sum(1 / r if r <= 10 else 0 for r in ranks) / 918
    return result


def shuffled_orders(seed):
    order, results, rng = list(range(28759)), [], random.Random(seed)
    for _ in range(10):
        rng.shuffle(order)
        results.append(object_sha(order))
    return results


def load_complete(workspace, audit_path):
    root, audit_path = Path(workspace).resolve(), Path(audit_path).resolve()
    protocol_path, freeze_path = root / "plans/protocol.json", root / "FOLLOWUP_SOURCE_FROZEN.json"
    hash_bound(protocol_path, PROTOCOL_SHA, "protocol revision 2")
    hash_bound(freeze_path, FREEZE_SHA, "source freeze")
    plan, freeze, audit = read(protocol_path), read(freeze_path), read(audit_path)
    hash_bound(root / "plans/protocol.md", plan["protocol_markdown_sha256"], "protocol markdown")
    require(audit.get("all_passed") is True and audit.get("n_runs") == 42
            and audit.get("original_cache_unchanged") is True, "A complete passed 42-run audit is required")
    require(audit.get("protocol_sha256") == PROTOCOL_SHA and audit.get("source_freeze_sha256") == FREEZE_SHA,
            "Audit protocol/source binding differs")
    require(audit.get("verifier_sha256") == VERIFIER_SHA, "Unreviewed torch verification tool")
    require(isinstance(audit.get("rank_replay_device"), str) and audit["rank_replay_device"].startswith("cuda"),
            "Reference CUDA matrix-rank replay is required")
    configurations = {item["id"]: item for item in plan["configurations"]}
    require(len(configurations) == 14 and plan["seeds"] == [0, 1, 2], "Unexpected fixed study matrix")
    expected_keys = {(seed, name) for seed in plan["seeds"] for name in configurations}
    require(len(audit["runs"]) == 42, "Audit must contain exactly 42 run records")
    audited = {}
    for row in audit["runs"]:
        key = (row["seed"], row["configuration"])
        require(key not in audited and row.get("all_checks_passed") is True, "Duplicate or failed audit run")
        audited[key] = row
    require(set(audited) == expected_keys, "Audit run matrix is incomplete")
    require(set(audit["seed_manifests"]) == {"0", "1", "2"}, "Audit seed manifests are incomplete")
    bindings = plan["input_bindings"]
    expected_inputs = {"training_cache": bindings["training_cache_sha256"],
                       "training_captions": bindings["clean_training_captions_sha256"],
                       **bindings["split_file_sha256"],
                       "original_split_manifest": bindings["original_split_manifest_sha256"],
                       "pretrained_weight": plan["backbone"]["pretrained_weight_sha256"]}
    source_keys = {name for name in freeze["file_sha256"] if name.startswith(("base_code/", "code/"))
                   or name == "scripts/run_factorial_fitval.py"}
    expected_sources = {name: freeze["file_sha256"][name] for name in source_keys}
    split_dir = root / "data/ret_clean_20261007"
    for name in ("fit_ids.json", "val_ids.json"):
        hash_bound(split_dir / name, bindings["split_file_sha256"][name], name)
    fit, val = set(read(split_dir / "fit_ids.json")), set(read(split_dir / "val_ids.json"))
    require(len(fit) == 8263 and len(val) == 918 and not fit & val, "Wrong fit/validation population")
    stable = {protocol_path: PROTOCOL_SHA, freeze_path: FREEZE_SHA, audit_path: sha(audit_path)}
    complete, provenance_rows, seed_states = {}, [], {}
    common_identity = None
    common_query_identity = None
    for seed in plan["seeds"]:
        rel_manifest = f"results/seed_{seed}/manifest.json"
        bound = audit["seed_manifests"][str(seed)]
        require(bound["relative_path"] == rel_manifest, "Unexpected audit manifest path")
        manifest_path = root / rel_manifest
        hash_bound(manifest_path, bound["sha256"], rel_manifest)
        stable[manifest_path] = bound["sha256"]
        manifest = read(manifest_path)
        require(manifest.get("schema") == "pare.factorial-fit-validation.v1"
                and manifest.get("status") == "complete_all_14_configurations"
                and manifest.get("active_configuration") is None, "Seed manifest is running, partial, or invalid")
        require(set(manifest["runs"]) == set(configurations), "Seed is missing a planned configuration")
        require(manifest["settings"] == plan["training"], "Training settings differ from the fixed plan")
        require(manifest.get("posthoc_exploratory") is True and manifest.get("uses_official_test") is False,
                "Study must remain exploratory validation without official test access")
        require((manifest["n_fit_images"], manifest["n_fit_pairs"], manifest["n_validation_queries"],
                 manifest["n_validation_gallery"], manifest["cache_contains_images"]) == (8263, 28759, 918, 918, 9181),
                "Incorrect training/evaluation population")
        require(len(manifest["train_ids"]) == len(set(manifest["train_ids"])) == 8263
                and set(manifest["train_ids"]) == fit and len(manifest["validation_ids"]) == 918
                and set(manifest["validation_ids"]) == val, "Seed split IDs differ")
        ident = manifest["identity"]
        require(ident["protocol_sha256"] == PROTOCOL_SHA and ident["seed"] == seed,
                "Seed/protocol identity differs")
        require(ident["configuration_ids"] == list(configurations), "Configuration order differs from plan")
        require(ident["source_sha256"] == expected_sources and ident["input_sha256"] == expected_inputs,
                "Seed inputs or executed sources differ")
        orders = shuffled_orders(seed)
        require(ident["epoch_order_sha256"] == orders, "Seed shuffle differs from the fixed protocol")
        initial, embeddings = manifest["initial_state_sha256"], manifest["text_embeddings_sha256"]
        require(all(digest_string(v) for v in (initial, embeddings, ident["initial_pairs_sha256"])), "Malformed identity hash")
        require(audit["initial_state_sha256_by_seed"][str(seed)] == initial
                and audit["frozen_text_embeddings_sha256_by_seed"][str(seed)] == embeddings, "Audit seed hashes differ")
        seed_states[str(seed)] = {"initial_state_sha256": initial, "text_embeddings_sha256": embeddings,
                                  "epoch_order_sha256": orders}
        shared = {"sources": ident["source_sha256"], "inputs": ident["input_sha256"],
                  "initial_pairs": ident["initial_pairs_sha256"], "queries": ident["validation_queries_sha256"],
                  "train_ids": manifest["train_ids"], "validation_ids": manifest["validation_ids"],
                  "text_embeddings": embeddings}
        if common_identity is None:
            common_identity = shared
        require(shared == common_identity, "Seeds use different data order, query population, or frozen embeddings")
        for name, configuration in configurations.items():
            run, proof = manifest["runs"][name], audited[seed, name]
            require(run["status"] == "complete" and run["configuration"] == configuration and run["seed"] == seed,
                    "Incomplete or mismatched run")
            require(run["initial_state_sha256"] == initial == proof["initial_state_sha256"]
                    and run["text_embeddings_sha256"] == embeddings
                    and run["initial_pairs_sha256"] == ident["initial_pairs_sha256"]
                    and run["epoch_order_sha256"] == orders, "Paired initialization/order differs")
            require(run["epochs"] == 10 and run["batches"] == 4500, "Wrong run training budget")
            directory = root / f"results/seed_{seed}/{name}"
            require(proof["relative_directory"] == str(directory.relative_to(root)), "Audit directory differs")
            require(set(run["artifacts_sha256"]) == ARTIFACTS, "Required run artifact set differs")
            for artifact, digest in run["artifacts_sha256"].items():
                hash_bound(directory / artifact, digest, f"{seed}/{name}/{artifact}")
            for artifact, key in (("head.pt", "head_sha256"), ("validation_scores.pt", "matrix_sha256"),
                                  ("validation.json", "validation_sha256")):
                require(run["artifacts_sha256"][artifact] == proof[key], "Artifact differs from exact audited bytes")
            progress = read(directory / "training_progress.json")["history"]
            require(len(progress) == 10, "Incomplete epoch history")
            for index, entry in enumerate(progress):
                require((entry["epoch"], entry["pairs"], entry["batches"], entry["tail_batch_pairs"], entry["pair_order_sha256"])
                        == (index + 1, 28759, 450, 23, orders[index]), "Epoch history differs from the planned budget/order")
                require(type(entry["mean_batch_loss"]) in (int, float) and math.isfinite(entry["mean_batch_loss"]), "Nonfinite training history")
            result = read(directory / "validation.json")
            require(result["configuration"] == configuration and result["seed"] == seed
                    and result["protocol_sha256"] == PROTOCOL_SHA, "Validation configuration differs")
            require(result["n_queries"] == result["n_gallery"] == 918
                    and result["posthoc_exploratory"] is True and result["official_test_accessed"] is False,
                    "Validation population/protocol differs")
            require(result["checkpoint_sha256"] == run["artifacts_sha256"]["head.pt"], "Validation/head binding differs")
            require(result["gallery_ids"] == manifest["validation_ids"]
                    and len(result["query_ids"]) == len(set(result["query_ids"])) == 918
                    and set(result["query_ids"]) == val, "Query/gallery ID mismatch")
            require(len(result["query_caption_sha256"]) == len(result["gold_gallery_index"]) == 918
                    and all(digest_string(v) for v in result["query_caption_sha256"]), "Invalid caption/target identity")
            for image_id, gold in zip(result["query_ids"], result["gold_gallery_index"]):
                require(type(gold) is int and 0 <= gold < 918 and result["gallery_ids"][gold] == image_id, "Incorrect gold target")
            query_identity = {key: result[key] for key in ("query_ids", "query_caption_sha256", "gallery_ids", "gold_gallery_index")}
            require(object_sha(list(map(list, zip(result["query_ids"], result["query_caption_sha256"]))))
                    == ident["validation_queries_sha256"], "Query identity hash differs")
            if common_query_identity is None:
                common_query_identity = query_identity
            require(query_identity == common_query_identity, "Runs use different query identities/order")
            values = from_ranks(result["ranks"])
            for label, reported in (("validation", result["metrics_percent"]), ("manifest", run["metrics_percent"]),
                                    ("audit", proof["metrics_percent"])):
                check_metrics(reported, values, f"{seed}/{name}/{label}")
            complete[seed, name] = values
            provenance_rows.append({"seed": seed, "configuration": name, "artifacts_sha256": run["artifacts_sha256"]})
    require(len(complete) == 42 and len({v["initial_state_sha256"] for v in seed_states.values()}) == 3,
            "Missing runs or seeds do not have distinct initial states")
    for path, digest in stable.items():
        hash_bound(path, digest, "Input changed during aggregation")
    provenance = {"protocol_sha256": PROTOCOL_SHA, "source_freeze_sha256": FREEZE_SHA,
                  "audit_sha256": stable[audit_path], "verifier_sha256": VERIFIER_SHA,
                  "seed_manifests": audit["seed_manifests"], "seed_identities": seed_states,
                  "audited_run_artifacts": provenance_rows, "workspace": str(root), "audit_path": str(audit_path)}
    return plan, complete, provenance


def summarize(plan, complete):
    runs, configuration_means, paired, contrast_means = [], [], [], []
    for config in plan["configurations"]:
        name = config["id"]
        row = {"configuration": name, **config["factors"], "n_seeds": 3}
        for seed in plan["seeds"]:
            runs.append({"configuration": name, **config["factors"], "seed": seed,
                         "n_fit_images": 8263, "n_fit_pairs": 28759, "n_queries": 918, "n_gallery": 918,
                         **complete[seed, name]})
        for metric in METRICS:
            values = [complete[seed, name][metric] for seed in plan["seeds"]]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sample_sd"] = statistics.stdev(values)
        configuration_means.append(row)
    require(len(plan["paired_contrasts"]) == 36, "Unexpected contrast set")
    for contrast in plan["paired_contrasts"]:
        terms = contrast["terms"]
        require(sum(t["coefficient"] for t in terms) == 0, "Contrast coefficients are not zero-sum")
        records = []
        for seed in plan["seeds"]:
            row = {"contrast": contrast["id"], "primary": contrast["primary_descriptive_contrast"], "seed": seed}
            for metric in METRICS:
                # The protocol contains the complete DID/DDD coefficients.
                # Do not divide again by the number of factors or levels.
                row[metric + "_delta_pp"] = math.fsum(t["coefficient"] * complete[seed, t["configuration_id"]][metric] for t in terms)
            paired.append(row)
            records.append(row)
        row = {"contrast": contrast["id"], "primary": contrast["primary_descriptive_contrast"], "n_seeds": 3}
        for metric in METRICS:
            values = [r[metric + "_delta_pp"] for r in records]
            row[metric + "_mean_delta_pp"] = statistics.mean(values)
            row[metric + "_sample_sd_pp"] = statistics.stdev(values)
        contrast_means.append(row)
    primary = [row for row in contrast_means if row["primary"]]
    require(len(primary) == 9, "Expected all nine predeclared primary descriptive contrasts")
    return {"run_metrics": runs, "configuration_summary": configuration_means,
            "paired_contrasts_by_seed": paired, "contrast_summary": contrast_means, "primary_contrast_summary": primary}


def esc(value):
    return str(value).replace("_", r"\_\allowbreak{}")


def number(value, signed=False):
    if round(value, 3) == 0:
        value = 0.0
    return f"{value:+.3f}" if signed and value != 0 else f"{value:.3f}"


def tex_table(caption, headers, rows):
    spec = r"@{}p{0.36\linewidth}" + "r" * (len(headers) - 1) + "@{}"
    line = lambda cells: " & ".join(cells) + r" \\"
    result = [r"\begingroup\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{" + spec + "}",
              r"\caption{" + caption + r"}\\", r"\toprule", line(headers), r"\midrule\endfirsthead",
              r"\toprule", line(headers), r"\midrule\endhead", r"\bottomrule\endfoot"]
    result += [line(row) for row in rows]
    return "\n".join(result + [r"\end{longtable}", r"\endgroup", ""])


def render_tables(tables):
    result = {}
    result["all_42_runs.tex"] = tex_table(
        "All 42 exploratory validation runs. Metrics are percentages; every query uses the full 918-image validation gallery.",
        ["Configuration", "Seed"] + list(METRICS),
        [[esc(r["configuration"]), str(r["seed"])] + [number(r[k]) for k in METRICS] for r in tables["run_metrics"]])
    result["all_14_configurations.tex"] = tex_table(
        "All 14 configurations: mean and sample standard deviation over the same three seeds (percent).",
        ["Configuration"] + list(METRICS),
        [[esc(r["configuration"])] + ["$" + number(r[k + "_mean"]) + r"\pm" + number(r[k + "_sample_sd"]) + "$" for k in METRICS]
         for r in tables["configuration_summary"]])
    result["all_108_paired_contrasts.tex"] = tex_table(
        "All predeclared contrasts, paired within seed, in percentage points. An asterisk denotes a primary descriptive contrast.",
        ["Contrast", "Seed"] + list(METRICS),
        [[esc(r["contrast"]) + ("*" if r["primary"] else ""), str(r["seed"])]
         + [number(r[k + "_delta_pp"], True) for k in METRICS] for r in tables["paired_contrasts_by_seed"]])
    for key, filename, caption in (("contrast_summary", "all_36_contrasts.tex", "All 36 predeclared contrasts"),
                                   ("primary_contrast_summary", "all_9_primary_contrasts.tex", "The nine predeclared primary descriptive contrasts")):
        result[filename] = tex_table(caption + ": paired-seed mean and sample standard deviation in percentage points. No significance test or winner selection is performed.",
            ["Contrast"] + list(METRICS),
            [[esc(r["contrast"])] + ["$" + number(r[k + "_mean_delta_pp"], True) + r"\pm" + number(r[k + "_sample_sd_pp"]) + "$" for k in METRICS]
             for r in tables[key]])
    result["all_tables.tex"] = "\n".join(result.values())
    return result


def write_outputs(directory, plan, tables, provenance):
    out = Path(directory).resolve()
    require(not out.exists(), "Refusing to overwrite an existing summary directory")
    tex = render_tables(tables)
    document = {"schema": "pare.exploratory-factorial-summary.v1", "status": "complete", "posthoc_exploratory": True,
                "validation_previously_used_for_selection": True, "official_test_used": False,
                "units": "metrics in percent; contrasts in percentage points; sample SD across 3 seeds",
                "n_runs": 42, "n_configurations": 14, "n_paired_contrasts": 36,
                "tables": tables, "contrast_definitions": plan["paired_contrasts"]}
    lines = ["Complete exploratory fit/validation follow-up: 42 runs, 14 configurations, seeds 0/1/2.",
             "The hypothesis was prompted by prior test results. This previously used validation population is not a new blinded test.",
             "All 36 predeclared contrasts are retained. Differences are paired within seed before mean/sample SD.",
             "No model winner, p-value, or replacement for the original test results is selected.",
             "Each row below lists R@1, R@5, R@10, MRR@10 as mean +/- sample SD.", "", "Configurations (percent):"]
    for r in tables["configuration_summary"]:
        lines.append(r["configuration"] + ": " + "; ".join(number(r[k + "_mean"]) + " +/- " + number(r[k + "_sample_sd"]) for k in METRICS))
    lines += ["", "All paired contrasts (percentage points; * = predeclared primary):"]
    for r in tables["contrast_summary"]:
        lines.append(("* " if r["primary"] else "  ") + r["contrast"] + ": "
                     + "; ".join(number(r[k + "_mean_delta_pp"], True) + " +/- " + number(r[k + "_sample_sd_pp"]) for k in METRICS))
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        with (out / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    (out / "README.txt").write_text(
        "All files derive from the complete audited 42-run protocol revision 2. No partial study is exported.\n"
        "CSV/JSON retain unrounded numbers. TeX display rounding is three decimals and never changes aggregation.\n"
        "TeX fragments require longtable, booktabs and amsmath; use all_tables.tex or individual fragments, not both.\n"
        "All contrasts use their exact frozen coefficients (including DID/DDD averaging); no additional rescaling.\n"
        "This is exploratory analysis on a previously used validation set. No p-values, winner selection or official test evaluation.\n"
        "The source manifest contains private machine paths; sanitize it before public distribution.\n")
    for name, content in tex.items():
        (out / name).write_text(content)
    manifest = {"schema": "pare.exploratory-factorial-export.v1", "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(), "source": provenance,
                "aggregator_sha256": sha(__file__), "command": sys.argv,
                "row_counts": {name: len(rows) for name, rows in tables.items()},
                "outputs": {f.name: {"sha256": sha(f), "bytes": f.stat().st_size} for f in sorted(out.iterdir())}}
    (out / "summary_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def run(workspace, audit, out):
    require(not Path(out).exists(), "Refusing to overwrite an existing summary directory")
    plan, complete, provenance = load_complete(workspace, audit)
    tables = summarize(plan, complete)
    return write_outputs(out, plan, tables, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.workspace, args.audit, args.out)
    print(json.dumps({"status": "complete", "out": str(args.out.resolve()), "row_counts": result["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
