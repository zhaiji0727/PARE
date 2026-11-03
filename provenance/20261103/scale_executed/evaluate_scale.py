#!/usr/bin/env python3
"""Frozen-head full-gallery evaluation of independently signed scale caches."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

from scale_contract import file_hash, object_hash
from spacing_groups import summarize_rank_rows
from stress_cache import load_frozen_code, set_device_guard, validate_scale_cache


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def full_gallery_head_ranks(scores, scene_scores, gold_rows, rank_of_gold):
    """Match the formal evaluator's K=N scene-order candidate tie convention."""
    import torch
    scene_order = torch.argsort(scene_scores, dim=1, descending=True)
    return [rank_of_gold(scores[row, scene_order[row]], scene_order[row].tolist(), gold)
            for row, gold in enumerate(gold_rows)]


def validate_complete_head(checkpoint, cache, flags, helper_hash, smoke=False):
    from artifacts import validate_head
    validate_head(checkpoint, cache)
    if set(checkpoint.get("ablate", [])) != set(flags):
        raise ValueError("Checkpoint has the wrong ablation flags")
    train_ids = checkpoint["provenance"].get("train_ids", [])
    if not train_ids or len(train_ids) != len(set(train_ids)) or set(train_ids) & set(cache["ids"]):
        raise ValueError("Missing, duplicate or overlapping training IDs")
    trained_helper = checkpoint["provenance"].get("batch_scoring_sha256")
    if trained_helper and trained_helper != helper_hash:
        raise ValueError("Training and evaluation batching helpers differ")
    if not smoke and (checkpoint.get("n_images") != 9181 or checkpoint.get("n_pairs") != 31945
                      or checkpoint.get("config", {}).get("epochs") != 10
                      or len(checkpoint.get("history", [])) != 10 or len(train_ids) != 9181):
        raise ValueError("Formal comparison needs the completed 10-epoch retained-train checkpoint")


def validate_extra_tensors(cache):
    import torch
    for name in ("L", "c_star"):
        value = cache[name]
        if tuple(value.shape) != (len(cache["ids"]),) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid float32 cache tensor: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--scoring-helper", required=True, type=Path)
    parser.add_argument("--reference-cache", required=True, type=Path)
    parser.add_argument("--cache-1p5", required=True, type=Path)
    parser.add_argument("--cache-2", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--stats-off-head", type=Path)
    parser.add_argument("--captions", required=True, type=Path)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--remoteclip", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--doc-chunk", type=int, default=512)
    parser.add_argument("--text-bs", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="16 images only; not publishable results")
    args = parser.parse_args()
    if min(args.query_chunk, args.doc_chunk, args.text_bs) < 1:
        parser.error("Batch sizes must be positive")
    set_device_guard(args.device)
    code = load_frozen_code(args.code_root)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import open_clip
    import artifacts
    import retrieval_queries
    import runtime
    import saqre_phys
    import train_saqre
    for module in (artifacts, retrieval_queries, runtime, saqre_phys, train_saqre):
        if Path(module.__file__).resolve().parent != code:
            raise RuntimeError("A scientific module came from outside the frozen code root")
    evaluator = load_module("scale_numerical_evaluator", args.evaluator)
    helper = load_module("scale_batch_helper", args.scoring_helper)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = runtime.get_device()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {key: value for key, value in vars(args).items()
              if isinstance(value, Path) and key not in ("out", "code_root")}
    sources = {path.name: path for path in code.glob("*.py")}
    sources.update({"scale/" + path.name: path for path in Path(__file__).parent.glob("*.py")
                    if not path.name.startswith("test_")})
    def hashes():
        return {"inputs": {key: {"path": str(path.resolve()), "sha256": file_hash(path)}
                           for key, path in inputs.items()},
                "source_sha256": {key: file_hash(path) for key, path in sources.items()}}
    before = hashes()
    manifest = {"schema": "pare.synthetic-spacing-evaluation.v1", "status": "running",
                "started_utc": utcnow(), "is_real_gsd_validation": False,
                "purpose": "SMOKE ONLY, NOT PAPER RESULTS" if args.smoke else "Synthetic spatial-scale sensitivity",
                "primary_methods": ["PARE", "Scene-CLIP"],
                "protocol": "Same frozen original-scale head, same first-caption queries, complete unchanged-ID gallery",
                "ranking": "Float32; PARE follows formal K=N scene-order tie convention; scene follows full-gallery 2D argsort",
                "conditions": {"1": "Unmodified formal reference cache",
                               "1.5": "Fresh features after one LANCZOS downsample; no manual upsample",
                               "2": "Fresh features after one LANCZOS downsample; no manual upsample"},
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "environment": {"torch": str(torch.__version__), "device": str(device),
                                "gpu_physical_index": 5 if args.device == "cuda" else None,
                                "tf32": False, "autocast": False}, **before}
    manifest_path = args.out / "manifest.json"
    write_json(manifest_path, manifest)
    try:
        reference = torch.load(args.reference_cache, map_location="cpu", weights_only=True)
        signature = artifacts.validate_cache(reference, args.model, args.pretrained, args.remoteclip)
        if signature["primitive_mode"] != "physics":
            raise ValueError("Expected original physics cache")
        evaluator.validate_tensor_layout(reference)
        validate_extra_tensors(reference)
        ids = reference["ids"]
        n = 16 if args.smoke else 3688
        if len(ids) != n:
            raise ValueError(f"Expected {n} gallery images")
        caches = {"1": reference}
        validation_views = {"1": reference}
        for factor, path in (("1.5", args.cache_1p5), ("2", args.cache_2)):
            caches[factor] = torch.load(path, map_location="cpu", weights_only=True)
            validation_views[factor] = validate_scale_cache(
                caches[factor], reference, before["inputs"]["reference_cache"]["sha256"],
                factor, args.model, args.pretrained, args.remoteclip)
            evaluator.validate_tensor_layout(caches[factor])
            validate_extra_tensors(caches[factor])
        groups = json.loads(args.groups.read_text()) if args.groups else None
        if groups and groups["test_ids"] != ids:
            raise ValueError("Observational spacing groups do not match this exact gallery")
        records, protocol = retrieval_queries.select_queries(
            train_saqre.load_captions(args.captions), ids, "first-caption", 0)
        if len(records) != n or protocol["truncated"] or protocol["gallery_images_without_evaluated_query"]:
            raise ValueError("Expected exactly one first-caption query for every gallery image")
        write_json(args.out / "queries.json", {"query_protocol": protocol, "gallery_ids": ids, "queries": records})
        head_paths = {"PARE": (args.head, set())}
        if args.stats_off_head:
            head_paths["PARE-stats-off"] = (args.stats_off_head, {"sr", "ty"})
        heads = {}
        for name, (path, flags) in head_paths.items():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            for view in validation_views.values():
                validate_complete_head(checkpoint, view, flags,
                                       before["inputs"]["scoring_helper"]["sha256"], args.smoke)
            head = saqre_phys.SAQREHead(reference["D"]).to(device)
            head.load_state_dict(checkpoint["state_dict"], strict=True)
            head.ablate = artifacts.parse_ablation_flags(",".join(sorted(flags)), reference)
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
            if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
                raise FloatingPointError("Nonfinite head weights")
            heads[name] = head
        model, _, _ = runtime.create_backbone(args.model, args.pretrained, args.remoteclip)
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(args.model)
        with torch.inference_mode():
            queries = torch.cat([F.normalize(model.encode_text(tokenizer(
                [record["caption"] for record in records[start:start + args.text_bs]]).to(device)), dim=-1).float()
                for start in range(0, n, args.text_bs)])
        del model
        if not torch.isfinite(queries).all():
            raise FloatingPointError("Nonfinite query embeddings")
        np.save(args.out / "query_embeddings.npy", queries.cpu().numpy(), allow_pickle=False)
        gold = [record["gold_gallery_row"] for record in records]
        reports = {}
        for factor, cache in caches.items():
            print(f"Evaluating fixed head at spacing factor {factor}", flush=True)
            started = time.perf_counter()
            scene = evaluator.score_baselines(queries, cache, device, args.doc_chunk)["scene"].to(device)
            scores = {"Scene-CLIP": scene}
            ranks = {"Scene-CLIP": evaluator.full_gallery_ranks(scene, gold)}
            for name, head in heads.items():
                scores[name] = evaluator.score_head_matrix(head, queries, cache, train_saqre.move_docs,
                    helper.score_matrix, device, args.query_chunk, args.doc_chunk).to(device)
                ranks[name] = full_gallery_head_ranks(scores[name], scene, gold, evaluator.rank_of_gold)
            key = factor.replace(".", "p")
            for name, score in scores.items():
                np.save(args.out / f"scores_spacing_{key}_{name}.npy", score.cpu().numpy(), allow_pickle=False)
            rows = [{"image_id": ids[record["gold_gallery_row"]], "query_id": record["query_id"],
                     "gold_gallery_row": record["gold_gallery_row"],
                     "ranks": {name: int(values[index]) for name, values in ranks.items()}}
                    for index, record in enumerate(records)]
            report = {"spacing_factor": factor, "is_real_gsd_validation": False,
                      "n_queries": n, "n_gallery": n, "query_protocol": protocol,
                      "metrics_percent": {name: evaluator.metrics(values) for name, values in ranks.items()},
                      "queries": rows, "elapsed_seconds": time.perf_counter() - started}
            if factor == "1" and groups:
                report["observational_original_raster_spacing_subgroups"] = summarize_rank_rows(rows, groups)
            reports[factor] = report
            write_json(args.out / f"eval_spacing_{key}.json", report)
            del scores, scene
        comparison = {"is_real_gsd_validation": False, "n_queries": n, "n_gallery": n,
                      "metrics_percent": {factor: result["metrics_percent"] for factor, result in reports.items()},
                      "delta_percentage_points_from_original": {
                          factor: {name: {metric: value - reports["1"]["metrics_percent"][name][metric]
                                          for metric, value in values.items()}
                                   for name, values in report["metrics_percent"].items()}
                          for factor, report in reports.items() if factor != "1"}}
        write_json(args.out / "comparison.json", comparison)
        if hashes() != before:
            raise RuntimeError("Input or source changed during evaluation")
        manifest.update(status="complete", finished_utc=utcnow(), input_code_recheck_passed=True,
                        query_protocol=protocol, query_records_sha256=object_hash(records),
                        outputs={path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
                                 for path in sorted(args.out.iterdir()) if path.is_file() and path != manifest_path})
        write_json(manifest_path, manifest)
        print(json.dumps(comparison, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status="failed", finished_utc=utcnow(), error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
