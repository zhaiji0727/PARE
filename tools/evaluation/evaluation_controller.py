#!/usr/bin/env python3
"""Wait for one frozen formal training run, then evaluate it independently."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

EVALUATOR_SHA = "9b08150a28035876307fc7ee225df7e87756c220005ed42f56beacfee69ea33d"
HELPER_SHA = "80661378e01809b17a46f9ec7844eed0e0940d7e208979eca2dcd81d0547a18c"
TRAINING_FREEZE_SHA = "9a36ebe8f794427f4273f6b559f2a28bd7f4bd6c7e2902577c7fe4448f7b644c"
VARIANTS = ("full", "stats_off", "cfar_strict", "uniform_grid")
KS = (100, 200, 500, 1000, 2000, 3688)
BACKBONES = {
    "vitb32": (0, "ViT-B-32", "laion2b_s34b_b79k"),
    "vitb16": (1, "ViT-B-16", "laion2b_s34b_b88k"),
    "remoteclip": (2, "ViT-B-32", "laion2b_s34b_b79k"),
    "siglip": (3, "ViT-B-16-SigLIP", "webli"),
    "georsclip": (4, "ViT-B-32", "openai"),
}
FORMAL_CONFIG_TAGS = {
    "vitb32": "vitb32_formal_20261103",
    "vitb16": "vitb16_formal_20261103",
    "remoteclip": "remoteclip_formal_20261103",
    "siglip": "siglip_formal_20261103",
    "georsclip": "georsclip_formal_20261103",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def assert_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: {actual} != {expected}")


def assess_training(root, tag):
    """Return (ready, reason, verified handoff). Never changes training files."""
    root = Path(root).resolve()
    state_path = root / "logs" / f"training_{tag}_state.json"
    try:
        state = read(state_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return False, "waiting_for_training_state", None
    if state.get("status") != "complete_sweep_and_train" or not state.get("ready_for_evaluation"):
        return False, "training_" + str(state.get("status", "unknown")), None
    if state.get("current_child_pid") is not None:
        return False, "training_child_still_present", None
    run = root / "results" / tag
    if (run / ".runner.lock").exists():
        return False, "training_runner_lock_present", None
    freeze_path = root / "TRAINING_SOURCE_FROZEN.json"
    assert_hash(freeze_path, TRAINING_FREEZE_SHA)
    if state.get("frozen_manifest_sha256") != TRAINING_FREEZE_SHA:
        raise ValueError("Training-state source-freeze binding differs")
    freeze = read(freeze_path)
    for relative, digest in freeze["file_sha256"].items():
        assert_hash(root / relative, digest)
    manifest_path = run / "run_manifest.json"
    manifest = read(manifest_path)
    config = manifest["config"]
    if config.get("smoke") or config.get("tag") != FORMAL_CONFIG_TAGS[tag]:
        raise ValueError("Wrong/non-formal training run")
    if Path(config["workspace"]).resolve() != root or Path(config["output"]).resolve() != run:
        raise ValueError("Training manifest workspace/output binding differs")
    _, model, pretrained = BACKBONES[tag]
    if (config["model"], config["pretrained"]) != (model, pretrained):
        raise ValueError("Training backbone differs from assigned GPU run")
    if bool(config.get("remoteclip")) != (tag in ("remoteclip", "georsclip")):
        raise ValueError("Unexpected external-checkpoint configuration")
    if config.get("remoteclip"):
        assert_hash(config["remoteclip"], config["remoteclip_sha256"])
    for relative, digest in config["source_sha256"].items():
        assert_hash(root / "code" / relative, digest)
    assert_hash(root / "scripts/run_repaired_experiment.py", config["runner_sha256"])
    steps = {step["name"]: step for step in manifest["steps"]}
    verified = {}
    required = ["train_" + name for name in VARIANTS] + ["cache_physics_test", "cache_uniform_grid_test"]
    for name in required:
        step = steps[name]
        allowed = {"complete"} if name.startswith("train_") else {"complete", "imported_verified_cache"}
        if step["status"] not in allowed:
            return False, "waiting_for_" + name, None
        if not step.get("artifact_sha256"):
            raise ValueError("Missing artifact hashes for " + name)
        for path, digest in step["artifact_sha256"].items():
            assert_hash(path, digest)
            verified[str(Path(path).resolve())] = digest
        expected_path = run / ("head_" + name[6:] + ".pt" if name.startswith("train_") else name + ".pt")
        if str(expected_path.resolve()) not in verified:
            raise ValueError("Expected artifact is not bound by step " + name)
    split = Path(config["split"])
    split_manifest_path = split / "manifest.json"
    assert_hash(split_manifest_path, config["split_manifest_sha256"])
    split_manifest = read(split_manifest_path)
    if not split_manifest.get("content_disjoint_train_test") or not split_manifest.get("content_disjoint_fit_val"):
        raise ValueError("Training split lacks content-disjoint verification")
    for relative, digest in split_manifest["output_sha256"].items():
        assert_hash(split / relative, digest)
    captions = split / "Caption_test.csv"
    return True, "verified_training_ready", {
        "training_state_path": str(state_path), "training_state_sha256": sha(state_path),
        "training_manifest_path": str(manifest_path), "training_manifest_sha256": sha(manifest_path),
        "training_freeze_sha256": TRAINING_FREEZE_SHA, "verified_artifacts": verified,
        "captions": str(captions), "captions_sha256": sha(captions), "config": config}


def gpu_busy(index):
    identities = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    uuid = next(line.split(",")[1].strip() for line in identities.splitlines() if int(line.split(",")[0]) == index)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    return [line.strip() for line in processes.splitlines() if line.split(",")[0].strip() == uuid]


def metric(ranks):
    ranks = [float("inf") if rank is None else rank for rank in ranks]
    if not ranks:
        return None
    n = len(ranks)
    return {"R@1": 100 * sum(r <= 1 for r in ranks) / n,
            "R@5": 100 * sum(r <= 5 for r in ranks) / n,
            "R@10": 100 * sum(r <= 10 for r in ranks) / n,
            "MRR@10": 100 * sum(1 / r if r <= 10 else 0 for r in ranks) / n}


def compare_metrics(actual, expected):
    if expected is None:
        if actual is not None:
            raise ValueError("Nonempty metrics for empty condition")
        return
    if set(actual) != set(expected):
        raise ValueError("Unexpected metric keys")
    for key, value in expected.items():
        if not math.isfinite(actual[key]) or not 0 <= actual[key] <= 100 or abs(actual[key] - value) > 1e-9:
            raise ValueError(f"Metric/rank inconsistency: {key}")


def validate_results(output, evaluator_sha=EVALUATOR_SHA):
    output = Path(output)
    manifest = read(output / "independent_manifest.json")
    if manifest.get("status") != "complete" or not manifest.get("input_code_recheck_passed"):
        raise ValueError("Independent evaluator did not complete verified execution")
    if manifest["code"]["independent_evaluator"]["sha256"] != evaluator_sha:
        raise ValueError("Unexpected evaluator version")
    for relative, entry in manifest["outputs"].items():
        assert_hash(output / relative, entry["sha256"])
    summary = {"output": str(output), "manifest_sha256": sha(output / "independent_manifest.json"), "variants": {}}
    for variant in VARIANTS:
        summary["variants"][variant] = {}
        for k in KS:
            report = read(output / f"eval_{variant}_k{k}.json")
            records = report["queries"]
            if report["n_gallery"] != 3688 or report["n_queries"] != 3688 or len(records) != 3688:
                raise ValueError("Wrong full-test dimensions")
            if len({record["query_id"] for record in records}) != 3688:
                raise ValueError("Duplicate/missing query IDs")
            qp = report["query_protocol"]
            if qp["mode"] != "first-caption" or qp["truncated"] or qp["gallery_images_without_evaluated_query"]:
                raise ValueError("Wrong query protocol")
            natural = report["natural_scene_shortlist"]
            if natural["K"] != k:
                raise ValueError("Wrong natural candidate count")
            ranks = [record["natural_PARE_rank"] for record in records]
            if any(rank is not None and (not isinstance(rank, int) or not 1 <= rank <= k) for rank in ranks):
                raise ValueError("Invalid natural rank")
            inclusion = [record["natural_candidate_included"] for record in records]
            if inclusion != [rank is not None for rank in ranks] or natural["n_natural_inclusions"] != sum(inclusion):
                raise ValueError("Candidate inclusion/rank mismatch")
            compare_metrics(natural["PARE_end_to_end"], metric(ranks))
            compare_metrics(natural["PARE_conditional_on_inclusion"], metric([rank for rank in ranks if rank is not None]))
            if abs(natural["candidate_recall_percent"] - 100 * sum(inclusion) / 3688) > 1e-9:
                raise ValueError("Candidate recall mismatch")
            if k == 3688 and not all(inclusion):
                raise ValueError("Full-gallery candidate miss")
            compare_metrics(report["scene_full_gallery"], metric([record["scene_full_gallery_rank"] for record in records]))
            if k == 100:
                if report["controlled_random_pools"]["K"] != 100 or report["controlled_random_pools"]["pools_per_query"] != 5:
                    raise ValueError("Wrong controlled pools")
                if report["appearance_adversarial_pool"]["M"] != 50:
                    raise ValueError("Wrong hard pool")
                for method in ("scene", "PARE"):
                    if any(len(record["controlled_random_pool_ranks"][method]) != 5 for record in records):
                        raise ValueError("Missing controlled pool ranks")
                    compare_metrics(report["controlled_random_pools"][method], metric([rank for record in records for rank in record["controlled_random_pool_ranks"][method]]))
                    compare_metrics(report["appearance_adversarial_pool"][method], metric([record["appearance_adversarial_ranks"][method] for record in records]))
            elif "controlled_random_pools" in report or "appearance_adversarial_pool" in report:
                raise ValueError("Unexpected additional pool protocol")
            summary["variants"][variant][str(k)] = {"natural": natural,
                **({"controlled": report["controlled_random_pools"], "hard": report["appearance_adversarial_pool"]} if k == 100 else {})}
    baseline = read(output / "eval_baselines.json")
    if baseline["n_gallery"] != 3688 or baseline["n_queries"] != 3688 or len(baseline["queries"]) != 3688:
        raise ValueError("Wrong baseline dimensions")
    for method in ("scene", "mean_pool", "MaxSim"):
        records = baseline["queries"]
        compare_metrics(baseline["full_gallery"][method], metric([record["full_gallery_ranks"][method] for record in records]))
        compare_metrics(baseline["controlled_random_pools"][method], metric([rank for record in records for rank in record["controlled_random_pool_ranks"][method]]))
        compare_metrics(baseline["appearance_adversarial_pool"][method], metric([record["appearance_adversarial_ranks"][method] for record in records]))
    summary["baselines"] = {key: baseline[key] for key in ("full_gallery", "controlled_random_pools", "appearance_adversarial_pool")}
    summary["verified_eval_json_count"] = 25
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True, help="Original workspace holding models and venv")
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--tag", choices=BACKBONES, required=True)
    p.add_argument("--evaluator", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--control-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--check-once", action="store_true")
    args = p.parse_args()
    if args.poll_seconds < 1 or args.poll_seconds > 60:
        p.error("poll-seconds must be between 1 and 60")
    assert_hash(args.evaluator, EVALUATOR_SHA)
    assert_hash(args.helper, HELPER_SHA)
    if args.check_once:
        ready, reason, handoff = assess_training(args.training_root, args.tag)
        print(json.dumps({"ready": ready, "reason": reason, "handoff": handoff}, indent=2))
        return
    args.control_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.control_dir / f"evaluation_{args.tag}_state.json"
    lock = args.control_dir / f"evaluation_{args.tag}.lock"
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.rename(lock.with_name(lock.name + f".stale_{int(time.time())}"))
        else:
            raise RuntimeError("Evaluation controller already running")
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    state = {"status": "waiting_for_verified_training", "tag": args.tag, "controller_pid": os.getpid(),
             "started_utc": now(), "gpu_physical_index": BACKBONES[args.tag][0],
             "controller_sha256": sha(__file__), "evaluator_sha256": EVALUATOR_SHA, "helper_sha256": HELPER_SHA,
             "training_freeze_sha256": TRAINING_FREEZE_SHA, "current_child_pid": None,
             "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    try:
        if state_path.exists():
            prior = read(state_path)
            if prior.get("status") == "complete_verified_evaluation":
                validate_results(prior["output"])
                print("Existing completed evaluation is verified; nothing to run", flush=True)
                return
            history = args.control_dir / f"evaluation_{args.tag}_previous_{int(time.time())}.json"
            atomic_json(history, prior)
        last_reason = None
        while True:
            ready, reason, handoff = assess_training(args.training_root, args.tag)
            if ready:
                busy = gpu_busy(BACKBONES[args.tag][0])
                if busy:
                    reason = "waiting_for_assigned_gpu_idle"
                    ready = False
                    state["assigned_gpu_processes"] = busy
            state.update(status=reason, updated_utc=now())
            atomic_json(state_path, state)
            if reason != last_reason:
                print(f"{now()} {args.tag}: {reason}", flush=True)
                last_reason = reason
            if ready:
                break
            time.sleep(args.poll_seconds)
        output = args.output_root / args.tag
        if output.exists():
            output = args.output_root / (args.tag + "__attempt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}")
        if output.exists():
            raise FileExistsError("Refusing to overwrite an evaluation attempt")
        assert_hash(args.evaluator, EVALUATOR_SHA)
        assert_hash(args.helper, HELPER_SHA)
        run = args.training_root / "results" / args.tag
        command = [str(args.workspace / ".venv/bin/python"), str(args.evaluator),
                   "--code-dir", str(args.training_root / "code"), "--scoring-helper", str(args.helper),
                   "--physics-cache", str(run / "cache_physics_test.pt"),
                   "--uniform-grid-cache", str(run / "cache_uniform_grid_test.pt"),
                   "--captions", handoff["captions"], "--model", handoff["config"]["model"],
                   "--pretrained", handoff["config"]["pretrained"], "--device", "cuda:0", "--out", str(output)]
        for variant in VARIANTS:
            command += ["--head-" + variant.replace("_", "-"), str(run / f"head_{variant}.pt")]
        if handoff["config"].get("remoteclip"):
            command += ["--remoteclip", handoff["config"]["remoteclip"]]
        env = os.environ.copy()
        explicit_env = {"RS_RAG_ROOT": str(args.workspace), "RS_RAG_MODEL_ROOT": str(args.workspace / "models"),
                        "HF_HOME": str(args.workspace / "models/huggingface"), "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": str(BACKBONES[args.tag][0]),
                        "PARE_DEVICE": "cuda:0", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
        env.update(explicit_env)
        log = args.control_dir / (output.name + ".evaluation.log")
        state.update(status="running_independent_evaluation", command=command, environment=explicit_env,
                     training_handoff=handoff, output=str(output), evaluation_log=str(log))
        atomic_json(state_path, state)
        with log.open("x") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            state.update(current_child_pid=child.pid, evaluation_started_utc=now())
            atomic_json(state_path, state)
            returncode = child.wait()
        state.update(current_child_pid=None, returncode=returncode, evaluation_finished_utc=now())
        if returncode:
            raise RuntimeError(f"Independent evaluator failed with status {returncode}; see {log}")
        ready, _, after = assess_training(args.training_root, args.tag)
        if not ready or after != handoff:
            raise ValueError("Training handoff changed during evaluation")
        summary = validate_results(output)
        summary_path = args.control_dir / (output.name + ".summary.json")
        atomic_json(summary_path, summary)
        state.update(status="complete_verified_evaluation", summary=str(summary_path), summary_sha256=sha(summary_path),
                     independent_manifest_sha256=summary["manifest_sha256"], updated_utc=now())
        atomic_json(state_path, state)
        print(f"{now()} {args.tag}: complete_verified_evaluation; summary={summary_path}", flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_utc=now())
        atomic_json(state_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
