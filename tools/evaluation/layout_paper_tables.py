#!/usr/bin/env python3
"""Lay out completed, hash-verified numeric exports without changing results.

Accepts an export directory or its paper_results.json/CSV. Unbound standalone
tables are not accepted. JSON and every CSV must agree before any TeX is made.
No model code, training data, or paper source is imported or modified.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

EXPORTER_SHA = "fef49d50b5fe1e91ba7dd01bf099a566033421c1e3ed68f6e74556dbbf87591c"
TAGS = ("vitb32", "vitb16", "remoteclip", "siglip", "georsclip")
BACKBONES = dict(zip(TAGS, ("OpenCLIP ViT-B/32", "OpenCLIP ViT-B/16", "RemoteCLIP", "SigLIP", "GeoRSCLIP")))
MAIN_METHODS = ("Scene-CLIP", "Mean-pool", "MaxSim", "PARE-stats-off", "PARE")
MAIN_DISPLAY_METHODS = MAIN_METHODS + ("PARE-CFAR-strict",)
CONTROLS = ("PARE", "PARE-stats-off", "PARE-CFAR-strict", "Uniform-grid")
POOL_METHODS = MAIN_METHODS[:3] + CONTROLS
SCALE_METHODS = ("Scene-CLIP", "PARE", "PARE-stats-off")
METHOD_LABELS = {"PARE-stats-off": "Stats-off", "PARE-CFAR-strict": "CFAR-strict", "Uniform-grid": "Uniform grid"}
METRICS = ("R@1", "R@5", "R@10", "MRR@10")
KS = (100, 200, 500, 1000, 2000, 3688)
BASE_TABLES = {"table_i_full_gallery", "table_ii_vitb32_controls", "appendix_natural_k",
               "appendix_hard50", "appendix_controlled100", "appendix_validation_grid",
               "validation_selected", "pare_differences"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def near(actual, expected, message):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) < 1e-9, message)


def keyed(rows, fields, expected):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, "Duplicate result row: " + str(key))
        result[key] = row
    require(set(result) == set(expected), "Missing or unexpected result rows for " + str(fields))
    return result


def metrics(row, prefix=""):
    values = [row[prefix + name] for name in METRICS]
    require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100 for value in values),
            "Invalid percentage metric")
    require(values[0] <= values[1] <= values[2] and values[3] <= values[2] + 1e-9,
            "Inconsistent recall/MRR ordering")


def population(row):
    require(row["n_gallery"] == row["n_queries"] == 3688, "Expected 3688 first-caption queries/gallery images")


def validate_tables(tables, stress_tags):
    expected_names = BASE_TABLES | ({"synthetic_scale_sensitivity"} if stress_tags else set())
    require(set(tables) == expected_names, "Unexpected/missing exported table set")
    full = keyed(tables["table_i_full_gallery"], ("backbone", "method"),
                 [(tag, method) for tag in TAGS for method in MAIN_METHODS])
    natural = keyed(tables["appendix_natural_k"], ("backbone", "method", "K"),
                    [(tag, method, k) for tag in TAGS for method in CONTROLS for k in KS])
    for row in full.values():
        population(row)
        metrics(row)
        require(row["protocol"] == "full_gallery", "Main table must be full-gallery retrieval")
    for (tag, method, k), row in natural.items():
        population(row)
        metrics(row)
        require(type(row["n_included"]) is int and 0 <= row["n_included"] <= 3688, "Invalid candidate inclusion count")
        near(row["candidate_recall_percent"], row["n_included"] * 100 / 3688, "Candidate count/recall mismatch")
        if row["n_included"]:
            metrics(row, "conditional_")
            for name in METRICS:
                near(row[name], row["candidate_recall_percent"] * row["conditional_" + name] / 100,
                     "Conditional/end-to-end metric mismatch")
        else:
            require(all(row["conditional_" + name] is None for name in METRICS), "Undefined conditional metrics expected")
            require(all(row[name] == 0 for name in METRICS), "All missed queries must fail")
        require(row["protocol"] == ("full_gallery" if k == 3688 else "natural_shortlist_end_to_end"),
                "Natural protocol label mismatch")
        if k == 3688:
            require(row["n_included"] == 3688, "Full-gallery candidate misses")
            if method in MAIN_METHODS:
                for name in METRICS:
                    near(row[name], full[tag, method][name], "Full-gallery tables differ")
    for tag in TAGS:
        for method in CONTROLS:
            counts = [natural[tag, method, k]["n_included"] for k in KS]
            require(counts == sorted(counts), "Nested shortlist candidate recall cannot decrease with K")
    pools = {}
    for name, count, protocol in (("appendix_controlled100", 100, "controlled100_positive_inserted"),
                                  ("appendix_hard50", 50, "hard50_positive_inserted")):
        pools[name] = keyed(tables[name], ("backbone", "method"), [(tag, method) for tag in TAGS for method in POOL_METHODS])
        for row in pools[name].values():
            population(row)
            metrics(row)
            require(row["protocol"] == protocol and row["candidate_count"] == count, "Positive-inserted pool protocol differs")
            if count == 100:
                require(row["pools_per_query"] == 5 and row["seed"] == 0, "Controlled100 sampling differs")
    controls = keyed(tables["table_ii_vitb32_controls"], ("backbone", "method"), [("vitb32", m) for m in CONTROLS])
    for (_, method), row in controls.items():
        for prefix, source in (("controlled100_", pools["appendix_controlled100"]["vitb32", method]),
                               ("natural100_", natural["vitb32", method, 100]),
                               ("full_gallery_", natural["vitb32", method, 3688])):
            for name in ("R@10", "MRR@10"):
                near(row[prefix + name], source[name], "Table II protocol/value mismatch")
        near(row["natural100_candidate_recall_percent"], natural["vitb32", method, 100]["candidate_recall_percent"],
             "Table II candidate recall differs")
    selected = keyed(tables["validation_selected"], ("backbone",), [(tag,) for tag in TAGS])
    grid = keyed(tables["appendix_validation_grid"], ("backbone", "trial_index_one_based"),
                 [(tag, index) for tag in TAGS for index in range(1, 10)])
    pairs = {(lr, kappa) for lr in (0.0005, 0.001, 0.002) for kappa in (0.03, 0.05, 0.07)}
    for tag in TAGS:
        rows = [grid[tag, index] for index in range(1, 10)]
        require({(row["lr"], row["kappa"]) for row in rows} == pairs, "Incomplete validation grid")
        for row in rows:
            metrics(row, "val_")
            require(row["epochs"] == 10 and row["n_fit_images"] == 8263
                    and row["n_val_queries"] == row["n_val_gallery"] == 918, "Validation population/training differs")
        best = max(rows, key=lambda row: (row["val_R@10"], row["val_MRR@10"]))
        require(sum(row["selected"] is True for row in rows) == 1 and best["selected"] is True,
                "Validation selection or tie order differs")
        require(all(selected[tag,][key] == best[key] for key in ("lr", "kappa") + tuple("val_" + m for m in METRICS)),
                "Selected table differs from complete grid")
    differences = keyed(tables["pare_differences"], ("backbone", "comparison"),
                        [(tag, "PARE minus " + method) for tag in TAGS for method in ("PARE-stats-off", "Scene-CLIP")])
    for (tag, comparison), row in differences.items():
        require(row["protocol"] == "full_gallery", "Difference protocol mismatch")
        other = comparison.removeprefix("PARE minus ")
        for name in METRICS:
            near(row["delta_" + name + "_pp"], full[tag, "PARE"][name] - full[tag, other][name], "Signed difference mismatch")
    if stress_tags:
        scale = keyed(tables["synthetic_scale_sensitivity"], ("backbone", "method", "spacing_factor"),
                      [(tag, method, factor) for tag in stress_tags for method in SCALE_METHODS for factor in (1, 1.5, 2)])
        for (tag, method, factor), row in scale.items():
            population(row)
            metrics(row)
            require(row["protocol"] == "synthetic_scale_full_gallery" and row["is_real_gsd_validation"] is False,
                    "Synthetic scale results must not be relabeled as acquisition-GSD validation")
            for name in METRICS:
                near(scale[tag, method, 1][name], full[tag, method][name], "Original-scale metrics differ")
            near(row["delta_R10_pp"], row["R@10"] - scale[tag, method, 1]["R@10"], "Scale delta mismatch")


def check_csv(path, rows):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(rows[0]), "CSV column order/schema differs: " + path.name)
        csv_rows = list(reader)
    require(len(csv_rows) == len(rows), "CSV/JSON row count differs: " + path.name)
    for encoded, original in zip(csv_rows, rows):
        for key, value in original.items():
            text = encoded[key]
            expected = "" if value is None else str(value)
            require(text == expected, "CSV/JSON value or order differs: " + path.name + " / " + key)


def load_verified_export(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    require(path.is_dir() or path.name == "paper_results.json" or path.suffix == ".csv", "Pass the export directory, JSON, or a bound CSV")
    manifest = read(directory / "export_manifest.json")
    require(manifest.get("schema") == "pare.paper-numeric-export.v1" and manifest.get("status") == "complete",
            "Only completed verified numeric exports can be laid out")
    require(manifest.get("source_sha256", {}).get("export_paper_results.py") == EXPORTER_SHA,
            "Export was not produced by the reviewed exporter version")
    require(set(manifest["evaluations"]) == set(manifest["sweeps"]) == set(TAGS), "All five backbone exports are required")
    require(set(manifest.get("stresses", {})) <= set(TAGS), "Unknown scale backbone")
    outputs = manifest["outputs"]
    require("paper_results.json" in outputs, "JSON is not bound by export manifest")
    if path.is_file():
        require(path.name in outputs, "Requested input is not bound by export manifest")
    for relative, entry in outputs.items():
        require(Path(relative).name == relative, "Export outputs must be direct child files")
        source = directory / relative
        require(sha(source) == entry["sha256"] and source.stat().st_size == entry["bytes"], "Export hash/length mismatch: " + relative)
    document = read(directory / "paper_results.json")
    require(document["units"] == "percent; deltas in percentage points", "Unknown metric units")
    tables = document["tables"]
    for name, rows in tables.items():
        require(rows and name + ".csv" in outputs, "Missing bound CSV: " + name)
        check_csv(directory / (name + ".csv"), rows)
    validate_tables(tables, set(manifest.get("stresses", {})))
    return directory, manifest, tables


def esc(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(character, character) for character in str(value))


def number(value, signed=False):
    if value is None:
        return "--"  # Undefined conditional metric, never an unmeasured result.
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite table cell")
    rounded = round(value, 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.2f}" if signed and rounded != 0 else f"{rounded:.2f}"


def method_label(method):
    return esc(METHOD_LABELS.get(method, method))


def line(cells):
    return " & ".join(cells) + r" \\"


def tabular(spec, headers, rows, midrules=(), width=r"\linewidth"):
    result = [r"\begingroup", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
              r"\renewcommand{\arraystretch}{1.05}", r"\setbox0=\hbox{%",
              r"\begin{tabular}{" + spec + "}", r"\toprule", *headers, r"\midrule"]
    for index, row in enumerate(rows):
        if index in midrules:
            result.append(r"\addlinespace[2pt]")
        result.append(line(row))
    return "\n".join(result + [r"\bottomrule", r"\end{tabular}%", "}",
                               r"\ifdim\wd0>" + width,
                               r"\resizebox{" + width + r"}{!}{\box0}%",
                               r"\else\box0\fi", r"\endgroup", ""])


def floating(content, caption, label, wide=False, placement="htbp"):
    env = "table*" if wide else "table"
    return "\n".join([r"\begin{" + env + "}[" + placement + "]", r"\centering", r"\caption{" + caption + "}",
                      r"\label{" + label + "}", content.rstrip(), r"\end{" + env + "}", ""])


def main_tables(tables):
    files = {}
    full = {(row["backbone"], row["method"]): row for row in tables["table_i_full_gallery"]}
    # The numeric export remains unchanged. The added display row is drawn only
    # from the verified, complete-gallery strict-control records, never a K=100
    # conditional score or a test-selected replacement for the full PARE method.
    for row in tables["appendix_natural_k"]:
        if row["method"] == "PARE-CFAR-strict" and row["K"] == 3688:
            full[row["backbone"], row["method"]] = row
    header = line([""] + [r"\multicolumn{4}{c}{" + esc(BACKBONES[tag]) + "}" for tag in TAGS])
    rules = "".join(r"\cmidrule(lr){" + str(2 + 4 * index) + "-" + str(5 + 4 * index) + "}" for index in range(5))
    headers = [header, rules, line(["Method"] + list(METRICS) * 5)]
    rows = [[method_label(method)] + [number(full[tag, method][metric]) for tag in TAGS for metric in METRICS] for method in MAIN_DISPLAY_METHODS]
    content = ("% CFAR-strict row: verified appendix_natural_k, method=PARE-CFAR-strict, K=3688, protocol=full_gallery.\n"
               + tabular("@{}l*{20}{r}@{}", headers, rows, width=r"\textwidth"))
    caption = (r"Full-gallery retrieval on 3,688 first-caption queries and the complete 3,688-image gallery. "
               r"All metrics are percentages; MRR@10 assigns zero beyond rank 10. Each backbone uses its own trained head. "
               r"CFAR-strict is a separately retrained control; PARE denotes the prespecified full configuration.")
    files["main_table_i_tabular.tex"] = content
    files["main_table_i_float.tex"] = floating(content, caption, "tab:retrieval", True, "!t")
    controls = {row["method"]: row for row in tables["table_ii_vitb32_controls"]}
    headers = [line(["", "Controlled100", r"\multicolumn{2}{c}{Natural $K=100$}", r"\multicolumn{2}{c}{Full gallery}"]),
               r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}", line(["Method", "R@10", "Cand. recall", "R@10", "R@10", "MRR@10"])]
    fields = ("controlled100_R@10", "natural100_candidate_recall_percent", "natural100_R@10", "full_gallery_R@10", "full_gallery_MRR@10")
    rows = [[method_label(method)] + [number(controls[method][field]) for field in fields] for method in CONTROLS]
    content = tabular("@{}lrrrrr@{}", headers, rows)
    caption = (r"ViT-B/32 controls (percent). Controlled100 inserts the positive and averages five seed-0 pools; "
               r"natural retrieval inserts no positive and counts candidate misses as failures. Full gallery uses all 3,688 images. "
               r"Stats-off disables reliability and tokens; CFAR-strict and uniform-grid are separately retrained controls.")
    files["main_table_ii_tabular.tex"] = content
    files["main_table_ii_float.tex"] = floating(content, caption, "tab:controls", placement="!t")
    if "synthetic_scale_sensitivity" in tables:
        scale = {(row["backbone"], row["method"], row["spacing_factor"]): row for row in tables["synthetic_scale_sensitivity"]}
        if any(key[0] == "vitb32" for key in scale):
            headers = [line(["", r"\multicolumn{3}{c}{R@10 (\%)}", r"$\Delta$ R@10 (pp)"]), r"\cmidrule(lr){2-4}",
                       line(["Method", "Original", r"$1.5\times$", r"$2\times$", r"$2\times-$original"])]
            rows = [[method_label(method)] + [number(scale["vitb32", method, factor]["R@10"]) for factor in (1, 1.5, 2)]
                    + [number(scale["vitb32", method, 2]["delta_R10_pp"], signed=True)] for method in SCALE_METHODS]
            content = tabular("@{}lrrrr@{}", headers, rows)
            caption = (r"ViT-B/32 synthetic spatial-scale sensitivity with fixed original-scale heads and the complete gallery. "
                       r"Factors denote nominal spacing after one Lanczos downsample, without manual upsampling. "
                       r"The signed difference compares $2\times$ with original. This is not real acquisition-GSD validation.")
            files["main_table_iii_tabular.tex"] = content
            files["main_table_iii_float.tex"] = floating(content, caption, "tab:scale_sensitivity", placement="!t")
    return files


def supplementary_tables(tables):
    files = {}
    natural = {(row["backbone"], row["method"], row["K"]): row for row in tables["appendix_natural_k"]}
    for tag in TAGS:
        headers = [line(["", "", "", "", r"\multicolumn{4}{c}{End-to-end (\%)}", r"\multicolumn{4}{c}{Conditional on inclusion (\%)}"]),
                   r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
                   line(["Method", "$K$", r"$n_{+}$", "Cand. recall"] + list(METRICS) * 2)]
        rows = []
        for method in CONTROLS:
            for index, k in enumerate(KS):
                row = natural[tag, method, k]
                rows.append([method_label(method) if index == 0 else "", str(k), str(row["n_included"]), number(row["candidate_recall_percent"])]
                            + [number(row[name]) for name in METRICS] + [number(row["conditional_" + name]) for name in METRICS])
        content = tabular("@{}lr*{10}{r}@{}", headers, rows, (6, 12, 18))
        caption = (esc(BACKBONES[tag]) + r": all prespecified natural candidate sizes. There are 3,688 first-caption queries; "
                   r"$n_{+}$ counts included positives. Misses count as failures in end-to-end metrics. "
                   r"At $K=3,688$, all scores are full-gallery metrics. A dash denotes an undefined conditional metric when no positive is included.")
        files[f"supp_natural_{tag}.tex"] = floating(content, caption, "tab:supp_natural_" + tag)
    for kind, key in (("controlled100", "appendix_controlled100"), ("hard50", "appendix_hard50")):
        indexed = {(row["backbone"], row["method"]): row for row in tables[key]}
        for tag in TAGS:
            rows = [[method_label(method)] + [number(indexed[tag, method][name]) for name in METRICS] for method in POOL_METHODS]
            content = tabular("@{}lrrrr@{}", [line(["Method"] + list(METRICS))], rows)
            detail = (r"Positive plus 99 uniformly sampled distractors; five seed-0 pools per query. " if kind == "controlled100"
                      else r"Positive plus the 49 strongest scene-level negatives. ")
            caption = esc(BACKBONES[tag]) + ": " + detail + r"All 3,688 queries are used. These positive-inserted diagnostics are not end-to-end natural retrieval. All metrics are percentages."
            files[f"supp_{kind}_{tag}.tex"] = floating(content, caption, "tab:supp_" + kind + "_" + tag)
    grid = {(row["backbone"], row["trial_index_one_based"]): row for row in tables["appendix_validation_grid"]}
    for tag in TAGS:
        headers = [line(["Trial", "Learning rate", r"$\kappa$"] + ["Val " + name for name in METRICS] + ["Selected"])]
        rows = []
        for index in range(1, 10):
            row = grid[tag, index]
            rows.append([str(index), f"{row['lr']:g}", f"{row['kappa']:g}"] + [number(row["val_" + name]) for name in METRICS] + ["Yes" if row["selected"] else ""])
        content = tabular("@{}lrrrrrrc@{}", headers, rows)
        caption = (esc(BACKBONES[tag]) + r": all nine 10-epoch validation trials, with 8,263 fit images and 918 first-caption validation queries/gallery images. "
                   r"Selection maximizes R@10, then MRR@10, then retains the first grid trial on an exact tie. Test results are not used for selection.")
        files[f"supp_validation_{tag}.tex"] = floating(content, caption, "tab:supp_validation_" + tag)
    selected = {row["backbone"]: row for row in tables["validation_selected"]}
    rows = [[esc(BACKBONES[tag]), f"{selected[tag]['lr']:g}", f"{selected[tag]['kappa']:g}"]
            + [number(selected[tag]["val_" + name]) for name in METRICS] for tag in TAGS]
    content = tabular("@{}lrrrrrr@{}", [line(["Backbone", "Learning rate", r"$\kappa$"] + ["Val " + name for name in METRICS])], rows)
    files["supp_selected_parameters.tex"] = floating(content, r"Validation-selected parameters and validation metrics (percent). Each backbone's full head and controls use the same selected settings for final retraining.", "tab:supp_selected_parameters")
    differences = {(row["backbone"], row["comparison"]): row for row in tables["pare_differences"]}
    rows = []
    for tag in TAGS:
        for method in ("PARE-stats-off", "Scene-CLIP"):
            row = differences[tag, "PARE minus " + method]
            rows.append([esc(BACKBONES[tag]), "PARE minus " + method_label(method)]
                        + [number(row["delta_" + name + "_pp"], signed=True) for name in METRICS])
    content = tabular("@{}llrrrr@{}", [line(["Backbone", "Comparison"] + [r"$\Delta$ " + name for name in METRICS])], rows, (2, 4, 6, 8))
    files["supp_signed_differences.tex"] = floating(content, r"Signed full-gallery differences in percentage points, computed from unrounded metrics. Negative differences are retained.", "tab:supp_signed_differences")
    if "synthetic_scale_sensitivity" in tables:
        scale = {(row["backbone"], row["method"], row["spacing_factor"]): row for row in tables["synthetic_scale_sensitivity"]}
        for tag in TAGS:
            if not any(key[0] == tag for key in scale):
                continue
            rows = []
            for method in SCALE_METHODS:
                for factor in (1, 1.5, 2):
                    row = scale[tag, method, factor]
                    rows.append([method_label(method) if factor == 1 else "", "Original" if factor == 1 else f"{factor:g}x"]
                                + [number(row[name]) for name in METRICS] + [number(row["delta_R10_pp"], signed=True)])
            content = tabular("@{}llrrrrr@{}", [line(["Method", "Spacing"] + list(METRICS) + [r"$\Delta$ R@10 (pp)"])], rows, (3, 6))
            caption = (esc(BACKBONES[tag]) + r": complete synthetic scale sensitivity, using the same original-scale heads and all 3,688 queries/gallery images. "
                       r"All four metrics are percentages; changes are relative to original spacing. No physical acquisition-GSD invariance is inferred.")
            files[f"supp_scale_{tag}.tex"] = floating(content, caption, "tab:supp_scale_" + tag)
    return files


def render(input_path, out):
    out = Path(out).resolve()
    require(not out.exists(), "Refusing to overwrite an existing layout directory")
    directory, export_manifest, tables = load_verified_export(input_path)
    source_manifest_sha = sha(directory / "export_manifest.json")
    main = main_tables(tables)
    supplementary = supplementary_tables(tables)
    files = {**main, **supplementary}
    files["main_tables_all.tex"] = "\n".join(main[name] for name in main if name.endswith("_float.tex"))
    files["supplementary_tables_all.tex"] = "\n".join(supplementary.values())
    files["README.txt"] = (
        "Numeric LaTeX fragments generated from a completed verified export. No manuscript source was edited.\n"
        "Required packages: booktabs and graphicx. Main Table I is a table* using textwidth; II and III use column linewidth.\n"
        "Use either *_tabular.tex inside an existing caption/label/float, or the matching *_float.tex, never both.\n"
        "main_tables_all.tex and supplementary_tables_all.tex are optional concatenations, not extra data.\n"
        "All values preserve input order by prespecified backbone/method keys, with metrics rounded only for display to two decimals.\n"
        "Main Table I displays six methods: its CFAR-strict row is drawn from the same verified export's appendix_natural_k records at K=3688, protocol=full_gallery. The original 25-row numeric Table I export is unchanged.\n"
        "No best-result bolding, backbone selection, statistical-significance claim, or performance interpretation is generated.\n"
        "A dash is only an undefined conditional metric when candidate inclusion is zero, never missing experiment data.\n"
        "Table III is generated only for completed ViT-B/32 scale data. Other completed scale backbones remain in the supplement.\n"
        "Tables use footnotesize at natural width; only over-wide tables are scaled down to the chosen width. Inspect actual pagination with real data.\n"
        "These files are table fragments, not a final five-page manuscript or a submission-ready PDF.\n")
    require(sha(directory / "export_manifest.json") == source_manifest_sha, "Input manifest changed while laying out results")
    out.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (out / name).write_text(content)
    manifest = {"schema": "pare.paper-table-layout.v1", "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_export": str(directory), "export_manifest_sha256": source_manifest_sha,
                "exporter_sha256": EXPORTER_SHA, "layout_script_sha256": sha(__file__), "command": sys.argv,
                "main_table_iii_included": "main_table_iii_tabular.tex" in main,
                "main_table_iii_omission_reason": None if "main_table_iii_tabular.tex" in main else "No completed verified ViT-B/32 scale results supplied",
                "table_row_counts": {name: len(rows) for name, rows in tables.items()},
                "main_table_i_display_methods": list(MAIN_DISPLAY_METHODS),
                "main_table_i_added_row_source": {"table": "appendix_natural_k", "method": "PARE-CFAR-strict",
                                                   "K": 3688, "protocol": "full_gallery", "metrics": list(METRICS)},
                "outputs": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in files}}
    (out / "layout_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--export-dir", required=True, type=Path, help="Verified export directory, paper_results.json, or any bound CSV")
    parser.add_argument("--out", required=True, type=Path, help="New directory for TeX fragments")
    args = parser.parse_args()
    manifest = render(args.input, args.out)
    print(json.dumps({"status": manifest["status"], "out": str(args.out.resolve()),
                      "main_table_iii_included": manifest["main_table_iii_included"], "file_count": len(manifest["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
