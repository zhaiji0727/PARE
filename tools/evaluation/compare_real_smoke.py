#!/usr/bin/env python3
"""Re-run the original evaluators on unchanged real 16-image smoke artifacts.

Writes only a fresh comparison directory. It does not train, rewrite inputs,
or change the code directory. Every protocol metric and query rank is compared.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code-dir", type=Path, required=True)
    p.add_argument("--smoke-artifacts", type=Path, required=True)
    p.add_argument("--captions", type=Path, required=True)
    p.add_argument("--new-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    common = ["--captions", str(args.captions), "--model", "ViT-B-32", "--pretrained", "laion2b_s34b_b79k",
              "--pools", "1", "--hard_M", "8", "--seed", "0"]
    cases = []
    for variant, mode, flags in (("full", "physics", ""), ("stats_off", "physics", "sr,ty"),
                                  ("cfar_strict", "physics", "cfar_strict"),
                                  ("uniform_grid", "uniform_grid", "sr,ty")):
        for k in (8, 16):
            name = f"eval_{variant}_k{k}"
            command = [sys.executable, str(args.code_dir / "eval_retrieval_phys.py"),
                       "--cache", str(args.smoke_artifacts / f"cache_{mode}_test.pt"),
                       "--head", str(args.smoke_artifacts / f"head_{variant}.pt"),
                       "--ablate", flags, "--candidate_K", str(k), *common,
                       "--out", str(args.out / (name + ".json"))]
            if k != 8:
                command += ["--skip_random", "--skip_hard"]
            cases.append((name, command))
    name = "eval_baselines"
    cases.append((name, [sys.executable, str(args.code_dir / "eval_retrieval_baselines.py"),
                         "--cache", str(args.smoke_artifacts / "cache_physics_test.pt"),
                         "--candidate_K", "8", *common, "--out", str(args.out / (name + ".json"))]))
    reports = []
    excluded = {"provenance", "elapsed_seconds"}
    for name, command in cases:
        print("Running original evaluator:", name, flush=True)
        with (args.out / (name + ".log")).open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        old = json.loads((args.out / (name + ".json")).read_text())
        new = json.loads((args.new_results / (name + ".json")).read_text())
        mismatches = [key for key in old if key not in excluded and old[key] != new.get(key)]
        reports.append({"case": name, "mismatching_keys": mismatches,
                        "n_queries": old["n_queries"], "all_metrics_and_query_ranks_equal": not mismatches})
        if mismatches:
            print("MISMATCH", name, mismatches, flush=True)
    result = {"purpose": "Real artifact regression only, not paper results", "cases": reports,
              "passed": all(item["all_metrics_and_query_ranks_equal"] for item in reports)}
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
