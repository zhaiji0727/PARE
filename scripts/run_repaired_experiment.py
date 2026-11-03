#!/usr/bin/env python3
"""Run a provenance-bound repaired protocol, with a separate CPU smoke mode."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = {"full": ("physics", ""), "stats_off": ("physics", "sr,ty"),
            "cfar_strict": ("physics", "cfar_strict"),
            "uniform_grid": ("uniform_grid", "sr,ty")}
STAGES = ("cache", "sweep", "train", "eval")
COMPLETED_STATUSES = {"complete", "imported_verified_cache"}
SOURCE_FILES = ("cache_primitives.py", "train_saqre.py", "sweep_hparams.py",
                "eval_retrieval_phys.py", "eval_retrieval_baselines.py",
                "retrieval_queries.py", "runtime.py", "artifacts.py",
                "primitive_controls.py", "saqre_phys.py", "batch_scoring.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_plan(workspace, output, model="ViT-B-32", pretrained="laion2b_s34b_b79k",
               remoteclip=None, smoke=False, gallery_size=3688, python=sys.executable):
    workspace, output = Path(workspace), Path(output)
    split = workspace / "data/ret_clean_20261007"
    backbone = ["--model", model, "--pretrained", pretrained]
    if remoteclip:
        backbone += ["--remoteclip", str(remoteclip)]
    steps = []

    def add(name, stage, script, args, outputs, requires=(), **details):
        steps.append({"name": name, "stage": stage,
                      "command": [python, str(workspace / "code" / script), *map(str, args)],
                      "outputs": [str(output / path) for path in outputs],
                      "requires": list(requires), **details})

    train_split = "fit" if smoke else "train"
    for mode in ("physics", "uniform_grid"):
        for side in ("train", "test"):
            ids = split / f"{train_split if side == 'train' else 'test'}_ids.json"
            target = f"cache_{mode}_{side}.pt"
            add(f"cache_{mode}_{side}", "cache", "cache_primitives.py",
                ["--ids", ids, "--img_root", workspace / "data/SARimages_pre", *backbone,
                 "--primitive_mode", mode, "--max_images", 16 if smoke else 0,
                 "--encode_bs", 16 if smoke else 128, "--out", output / target], [target],
                mode=mode, side=side, ids_path=str(ids))
    if not smoke:
        add("sweep", "sweep", "sweep_hparams.py",
            ["--cache", output / "cache_physics_train.pt", "--captions", split / "Caption_train.csv",
             "--fit_ids", split / "fit_ids.json", "--val_ids", split / "val_ids.json", *backbone,
             "--epochs", 10, "--bs", 64, "--val_q", 0, "--seed", 0,
             "--lrs", "5e-4,1e-3,2e-3", "--kappas", "0.03,0.05,0.07",
             "--csv", output / "hparam_sweep.csv", "--out", output / "hparam_best.json"],
            ["hparam_sweep.csv", "hparam_best.json", "hparam_best_trials"], ["cache_physics_train"])
    for variant, (mode, ablate) in VARIANTS.items():
        flags = ["--ablate", ablate] if ablate else []
        head = f"head_{variant}.pt"
        dependencies = [f"cache_{mode}_train"] + ([] if smoke else ["sweep"])
        add(f"train_{variant}", "train", "train_saqre.py",
            ["--cache", output / f"cache_{mode}_train.pt", "--captions", split / f"Caption_{train_split}.csv",
             *backbone, "--epochs", 1 if smoke else 10, "--bs", 4 if smoke else 64,
             "--max_pairs", 32 if smoke else 0, "--text_bs", 16 if smoke else 256,
             "--lr", "0.002" if smoke else "<selected_lr>",
             "--kappa", "0.03" if smoke else "<selected_kappa>",
             "--weight_decay", "0.0001", "--seed", 0, *flags, "--out", output / head],
            [head], dependencies, variant=variant, mode=mode, ablate=ablate)
    gallery_size = min(16, gallery_size) if smoke else gallery_size
    ks = sorted({min(8, gallery_size), gallery_size}) if smoke else [100, 200, 500, 1000, 2000, gallery_size]
    pools, hard = (1, min(8, gallery_size)) if smoke else (5, 50)
    for variant, (mode, ablate) in VARIANTS.items():
        for index, k in enumerate(ks):
            name = f"eval_{variant}_k{k}"
            add(name, "eval", "eval_retrieval_phys.py",
                ["--cache", output / f"cache_{mode}_test.pt", "--head", output / f"head_{variant}.pt",
                 "--captions", split / "Caption_test.csv", *backbone, "--candidate_K", k,
                 "--pools", pools, "--hard_M", hard, "--max_queries", 0,
                 "--query_mode", "first-caption", "--text_bs", 16 if smoke else 256, "--seed", 0,
                 *(["--ablate", ablate] if ablate else []),
                 *(["--skip_random", "--skip_hard"] if index else []),
                 "--out", output / f"{name}.json"], [f"{name}.json"],
                [f"cache_{mode}_test", f"train_{variant}"], variant=variant, mode=mode, k=k)
    add("eval_baselines", "eval", "eval_retrieval_baselines.py",
        ["--cache", output / "cache_physics_test.pt", "--captions", split / "Caption_test.csv",
         *backbone, "--candidate_K", ks[0], "--pools", pools, "--hard_M", hard,
         "--max_queries", 0, "--query_mode", "first-caption", "--text_bs", 16 if smoke else 256,
         "--seed", 0, "--out", output / "eval_baselines.json"], ["eval_baselines.json"], ["cache_physics_test"],
        mode="physics")
    return steps


def artifact_hashes(outputs):
    files = []
    for output in outputs:
        path = Path(output)
        if not path.exists():
            raise ValueError(f"Expected output missing: {path}")
        files.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
    return {str(path): sha256(path) for path in files if path.is_file()}


def verify_completed(step):
    if step.get("status") not in COMPLETED_STATUSES:
        raise ValueError(f"Prerequisite step is not complete: {step['name']}")
    if artifact_hashes(step["outputs"]) != step.get("artifact_sha256"):
        raise ValueError(f"Completed output changed: {step['name']}")
    if step.get("status") == "imported_verified_cache":
        if step.get("stage") != "cache":
            raise ValueError("Only cache steps may carry imported_verified_cache status")
        origin = step.get("import_provenance", {})
        source = origin.get("source_run_manifest")
        if not source or sha256(source) != origin.get("source_run_manifest_sha256"):
            raise ValueError("Imported cache source manifest changed or is missing")
        original = json.loads(Path(source).read_text())
        matches = [item for item in original.get("steps", [])
                   if item.get("name") == origin.get("source_step_name")]
        if len(matches) != 1 or matches[0].get("status") != "complete" or matches[0].get("stage") != "cache":
            raise ValueError("Imported cache is not bound to a completed original cache step")



def selected_parameters(output, steps):
    verify_completed(steps["sweep"])
    result = json.loads((Path(output) / "hparam_best.json").read_text())
    if result["epochs"] != 10 or len(result["grid"]) != 9:
        raise ValueError("Formal selection requires all nine 10-epoch validation trials")
    best = max(result["grid"], key=lambda row: (row["val_R@10"], row["val_MRR@10"]))
    if result["best"] != best:
        raise ValueError("Recorded best does not follow the predefined validation selection rule")
    return str(best["lr"]), str(best["kappa"])


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, dict):
        for nested in value.values():
            finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            finite_json(nested)


def verify_step(step, config):
    import torch

    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(config["workspace"]) / "code"))
    from artifacts import validate_cache, validate_head
    output = Path(config["output"])
    if step["stage"] == "cache":
        cache = torch.load(step["outputs"][0], map_location="cpu", weights_only=True)
        expected = json.loads(Path(step["ids_path"]).read_text())
        if config["smoke"]:
            expected = expected[:16]
        if cache["ids"] != expected:
            raise ValueError("Cache IDs differ from planned IDs")
        signature = validate_cache(cache, config["model"], config["pretrained"], config["remoteclip"])
        if signature["primitive_mode"] != step["mode"]:
            raise ValueError("Cache primitive mode mismatch")
        if any(not torch.isfinite(value).all() for value in cache.values() if torch.is_tensor(value)):
            raise ValueError("Nonfinite cached tensor")
        return {"images": len(expected), "signature": signature}
    if step["stage"] == "train":
        head = torch.load(step["outputs"][0], map_location="cpu", weights_only=True)
        cache = torch.load(output / f"cache_{step['mode']}_train.pt", map_location="cpu", weights_only=True)
        validate_head(head, cache)
        if set(head["ablate"]) != {name for name in step["ablate"].split(",") if name}:
            raise ValueError("Head ablation flags differ from plan")
        if any(not torch.isfinite(value).all() for value in head["state_dict"].values()):
            raise ValueError("Nonfinite trained head")
        epochs = 1 if config["smoke"] else 10
        if len(head["history"]) != epochs or (config["smoke"] and head["n_pairs"] != 32):
            raise ValueError("Head training scope does not match requested protocol")
        finite_json(head["history"])
        return {"variant": step["variant"], "n_images": head["n_images"], "n_pairs": head["n_pairs"],
                "epochs": epochs, "ablate": head["ablate"], "signature": head["provenance"]["cache_signature"]}
    if step["stage"] == "sweep":
        report = json.loads(Path(step["outputs"][1]).read_text())
        finite_json(report)
        expected_val = json.loads((Path(config["split"]) / "val_ids.json").read_text())
        if report["n_val_q"] != len(expected_val) or report["n_val_gallery"] != len(expected_val):
            raise ValueError("Validation sweep did not evaluate the whole clean validation set")
        if report["epochs"] != 10 or len(report["grid"]) != 9:
            raise ValueError("Incomplete formal validation grid")
        trial_files = list(Path(step["outputs"][2]).glob("*.pt"))
        if len(trial_files) != 9 or len(list(Path(step["outputs"][2]).glob("*.json"))) != 9:
            raise ValueError("Missing trial checkpoints or per-query ranks")
        return {"trials": 9, "validation_queries": len(expected_val), "best": report["best"]}
    report = json.loads(Path(step["outputs"][0]).read_text())
    finite_json(report)
    n = 16 if config["smoke"] else 3688
    if report["n_gallery"] != n or report["n_queries"] != n or len(report["queries"]) != n:
        raise ValueError("Evaluation does not cover every planned image's first caption")
    if report["query_protocol"]["truncated"] or report["provenance"]["allow_legacy_artifacts"]:
        raise ValueError("Truncated or unverified evaluation is not accepted")
    if report["provenance"]["cache_signature"]["primitive_mode"] != step["mode"]:
        raise ValueError("Evaluation signature uses wrong primitive mode")
    return {"queries": n, "gallery": n, "finite_numbers": True,
            "mode": step["mode"], "query_protocol": report["query_protocol"]}


def environment(workspace, output, device):
    env = os.environ.copy()
    env.update({"RS_RAG_ROOT": str(workspace), "RS_RAG_DATA_ROOT": str(workspace / "data"),
                "RS_RAG_MODEL_ROOT": str(workspace / "models"), "RS_RAG_RESULTS_ROOT": str(output),
                "HF_HOME": str(workspace / "models/huggingface"), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "PARE_DEVICE": device,
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1"})
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without creating output files")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--remoteclip", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"cpu|cuda:[0-9]+", args.device):
        parser.error("device must be cpu or cuda:N")
    tag = args.tag or ("vitb32_smoke_cpu" if args.smoke else "vitb32")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag) or tag in (".", ".."):
        parser.error("tag must be a single safe directory name")
    if args.smoke and args.stage == "sweep":
        parser.error("Smoke mode intentionally has no hyperparameter selection")
    workspace = args.workspace.resolve()
    output = (args.out or workspace / "results/repaired_20261007" / tag).resolve()
    split = workspace / "data/ret_clean_20261007"
    split_manifest = json.loads((split / "manifest.json").read_text())
    if not split_manifest.get("content_disjoint_train_test") or not split_manifest.get("content_disjoint_fit_val"):
        raise ValueError("Content-disjoint split manifest is required")
    if split_manifest["official_test_duplicate_content_group_count"]:
        raise ValueError("This one-positive runner requires a duplicate-content-free test gallery")
    for name, expected in split_manifest["output_sha256"].items():
        if sha256(split / name) != expected:
            raise ValueError(f"Clean split artifact changed: {name}")
    remoteclip = str(args.remoteclip.resolve()) if args.remoteclip else None
    config = {"workspace": str(workspace), "output": str(output), "split": str(split),
              "tag": tag, "smoke": args.smoke, "device": args.device, "model": args.model,
              "pretrained": args.pretrained, "remoteclip": remoteclip,
              "remoteclip_sha256": sha256(remoteclip) if remoteclip else None,
              "python": str(Path(sys.executable).absolute()), "seed": 0,
              "split_manifest_sha256": sha256(split / "manifest.json"),
              "source_sha256": {name: sha256(workspace / "code" / name) for name in SOURCE_FILES},
              "runner_sha256": sha256(__file__)}
    plan = build_plan(workspace, output, args.model, args.pretrained, remoteclip, args.smoke,
                      split_manifest["split_counts"]["test"])
    if args.dry_run:
        print(json.dumps({"config": config, "paper_results": False, "dry_run": True,
                          "selection": "Fixed smoke settings, no parameter selection" if args.smoke else
                          "Nine 10-epoch clean fit/val trials; maximum val R@10, tie-break MRR@10; no test selection",
                          "steps": [step for step in plan if args.stage in ("all", step["stage"])]}, indent=2))
        return
    manifest_path = output / "run_manifest.json"
    if output.exists() and not manifest_path.exists():
        raise ValueError("Refusing to reuse an existing directory without this runner's manifest")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".runner.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["config"] != config:
                raise ValueError("Configuration or code changed; use a fresh output directory")
        else:
            env = environment(workspace, output, args.device)
            manifest = {"status": "pending", "created_utc": now(), "config": config,
                        "purpose": "PIPELINE SMOKE ONLY; NOT PAPER RESULTS" if args.smoke else "Formal repaired-protocol experiment",
                        "environment": {"platform": platform.platform(), "python": sys.version,
                            "packages": {name: importlib.metadata.version(name) for name in
                                         ("torch", "open_clip_torch", "numpy", "Pillow", "scipy")},
                            "variables": {key: env[key] for key in ("PARE_DEVICE", "HF_HOME", "HF_HUB_OFFLINE",
                                          "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")}},
                        "steps": [{**step, "status": "pending"} for step in plan]}
        (output / "logs").mkdir(exist_ok=True)
        steps = {step["name"]: step for step in manifest["steps"]}
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        try:
            for step in manifest["steps"]:
                if args.stage not in ("all", step["stage"]):
                    continue
                if config["source_sha256"] != {name: sha256(workspace / "code" / name) for name in SOURCE_FILES}:
                    raise RuntimeError("Experiment source changed during execution; refusing mixed-version artifacts")
                if step["status"] in COMPLETED_STATUSES:
                    verify_completed(step)
                    print(f"Verified existing completed step {step['name']}; not overwritten", flush=True)
                    continue
                for name in step["requires"]:
                    verify_completed(steps[name])
                logfile = output / "logs" / f"{step['name']}.log"
                if logfile.exists() or any(Path(path).exists() for path in step["outputs"]):
                    raise ValueError(f"Refusing to overwrite prior/partial output for {step['name']}; choose fresh output")
                command = step["command"][:]
                if "<selected_lr>" in command:
                    lr, kappa = selected_parameters(output, steps)
                    command = [lr if token == "<selected_lr>" else kappa if token == "<selected_kappa>" else token for token in command]
                step.update(status="running", started_utc=now(), executed_command=command, log=str(logfile))
                write_json(manifest_path, manifest)
                print(f"Running {step['name']}; log={logfile}", flush=True)
                with logfile.open("x") as stream:
                    result = subprocess.run(command, cwd=workspace,
                                            env=environment(workspace, output, args.device),
                                            stdout=stream, stderr=subprocess.STDOUT)
                step.update(finished_utc=now(), returncode=result.returncode, log_sha256=sha256(logfile))
                if result.returncode:
                    step["status"] = "failed"
                    raise RuntimeError(f"{step['name']} failed with exit {result.returncode}; see {logfile}")
                step["verification"] = verify_step(step, config)
                step["artifact_sha256"] = artifact_hashes(step["outputs"])
                step["status"] = "complete"
                write_json(manifest_path, manifest)
                print(f"Verified {step['name']}", flush=True)
            complete = all(step["status"] in COMPLETED_STATUSES for step in manifest["steps"])
            manifest.update(status="complete" if complete else "partial", updated_utc=now(),
                            all_required_steps_complete=complete)
            write_json(manifest_path, manifest)
            print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path),
                              "purpose": manifest["purpose"], "completed_steps": sum(step["status"] in COMPLETED_STATUSES for step in manifest["steps"]),
                              "total_steps": len(manifest["steps"])}, indent=2), flush=True)
        except BaseException as error:
            for step in manifest["steps"]:
                if step["status"] == "running":
                    step["status"] = "failed"
            manifest.update(status="failed", updated_utc=now(), error=f"{type(error).__name__}: {error}")
            write_json(manifest_path, manifest)
            raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
