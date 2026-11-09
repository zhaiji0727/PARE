#!/usr/bin/env python3
"""Independent full-score PARE evaluator; never mutates the training runner.

Reuses the specified code directory's unmodified model, preprocessing, query
selection and artifact validation. All scores are float32. Candidate generation
and ranking follow eval_retrieval_phys.py; an independent manifest identifies
this evaluator, the vmap helper, code, inputs, and outputs explicitly.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


VARIANTS = {
    "full": ("physics", set()),
    "stats_off": ("physics", {"sr", "ty"}),
    "cfar_strict": ("physics", {"cfar_strict"}),
    "uniform_grid": ("uniform_grid", {"sr", "ty"}),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def metrics(ranks):
    values = np.asarray(ranks, dtype=np.float64)
    if not values.size:
        return None
    return {"R@1": float((values <= 1).mean() * 100),
            "R@5": float((values <= 5).mean() * 100),
            "R@10": float((values <= 10).mean() * 100),
            "MRR@10": float(np.mean([1 / r if r <= 10 else 0 for r in values]) * 100)}


def rank_of_gold(local_scores, candidates, gold):
    # Intentionally identical to the original one-dimensional argsort path.
    # Do not replace with greater-than counts: tied scores have different ranks.
    order = torch.argsort(local_scores, descending=True).tolist()
    return [candidates[index] for index in order].index(gold) + 1


def random_ranker_theory(k):
    top = min(10, k)
    return {"R@1": 100.0 / k, "R@5": 100.0 * min(5, k) / k,
            "R@10": 100.0 * top / k,
            "MRR@10": 100.0 * sum(1.0 / r for r in range(1, top + 1)) / k}


def make_random_pools(gold_rows, gallery_size, k=100, pools=5, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = np.empty((len(gold_rows), pools, k), dtype=np.int32)
    for qi, gold in enumerate(gold_rows):
        for pool in range(pools):
            sampled = torch.randperm(gallery_size - 1, generator=generator)[:k - 1]
            sampled = sampled + (sampled >= gold).long()
            result[qi, pool] = [gold] + sampled.tolist()
    return result


def make_hard_pools(scene_order, gold_rows, hard_m=50):
    result = np.empty((len(gold_rows), hard_m), dtype=np.int32)
    for qi, gold in enumerate(gold_rows):
        negatives = [int(row) for row in scene_order[qi] if row != gold][:hard_m - 1]
        result[qi] = [gold] + negatives
    return result


@torch.no_grad()
def score_head_matrix(head, queries, cache, move_docs, score_matrix, device,
                      query_chunk=32, doc_chunk=512):
    """Return CPU float32 Q x N scores using the supplied validated model/cache.

    This reusable numerical core performs no artifact validation. Callers such
    as a synthetic-spacing adapter must validate their own input contract.
    score_matrix is the supplied helper's (head, queries, docs) callable.
    """
    if min(query_chunk, doc_chunk) < 1:
        raise ValueError("Chunk sizes must be positive")
    queries = queries.to(device=device, dtype=torch.float32)
    n = len(cache["ids"])
    result = torch.empty((len(queries), n), dtype=torch.float32, device="cpu")
    for ds in range(0, n, doc_chunk):
        de = min(ds + doc_chunk, n)
        docs = move_docs(cache, torch.arange(ds, de), device)
        for qs in range(0, len(queries), query_chunk):
            qe = min(qs + query_chunk, len(queries))
            block = score_matrix(head, queries[qs:qe], docs)
            if block.dtype != torch.float32 or not torch.isfinite(block).all():
                raise FloatingPointError("Non-finite or non-float32 head score")
            result[qs:qe, ds:de] = block.cpu()
    return result


@torch.no_grad()
def score_baselines(queries, cache, device, doc_chunk=512):
    """Exact scene/mean-pool/MaxSim formulas from eval_retrieval_baselines.py."""
    queries = queries.to(device=device, dtype=torch.float32)
    scene = F.normalize(cache["scene"].to(device), dim=-1)
    regions = F.normalize(cache["c_emb"].to(device), dim=-1)
    fine = F.normalize(cache["f_emb"].to(device), dim=-1)
    cm = torch.arange(cache["Mc"], device=device)[None] < cache["c_n"].to(device)[:, None]
    fm = torch.arange(cache["Mf"], device=device)[None] < cache["f_n"].to(device)[:, None]
    scene_scores = queries @ scene.T
    counts = 1.0 + cm.sum(1, keepdim=True).float() + fm.sum(1, keepdim=True).float()
    mean = F.normalize((scene + (regions * cm[..., None]).sum(1)
                        + (fine * fm[..., None]).sum(1)) / counts, dim=-1)
    mean_scores = queries @ mean.T
    maximum = torch.empty_like(scene_scores)
    negative = torch.finfo(torch.float32).min
    for start in range(0, len(scene), doc_chunk):
        end = min(start + doc_chunk, len(scene))
        global_part = queries @ scene[start:end].T
        coarse_part = torch.einsum("qd,bmd->qbm", queries, regions[start:end])
        coarse_part = coarse_part.masked_fill(~cm[start:end][None], negative).max(-1).values
        fine_part = torch.einsum("qd,bmd->qbm", queries, fine[start:end])
        fine_part = fine_part.masked_fill(~fm[start:end][None], negative).max(-1).values
        maximum[:, start:end] = torch.maximum(torch.maximum(global_part, coarse_part), fine_part)
    result = {"scene": scene_scores.cpu(), "mean_pool": mean_scores.cpu(), "MaxSim": maximum.cpu()}
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Invalid baseline score")
    return result


def full_gallery_ranks(scores, gold_rows):
    # The original baseline uses a two-dimensional argsort for this protocol.
    order = torch.argsort(scores, dim=1, descending=True)
    return [(order[qi] == gold).nonzero(as_tuple=True)[0].item() + 1
            for qi, gold in enumerate(gold_rows)]


def pool_ranks(scores, pools, gold_rows):
    ranks = []
    for qi, gold in enumerate(gold_rows):
        row_pools = pools[qi] if pools.ndim == 3 else pools[qi:qi + 1]
        for pool in row_pools:
            candidates = pool.tolist()
            ranks.append(rank_of_gold(scores[qi, candidates], candidates, gold))
    return ranks


def derive_head_reports(scores, scene_scores, records, ks, random_pools,
                        hard_pools, scene_order, base_result):
    """Scores stay on the original evaluation device during ranking/ties."""
    gold = [record["gold_gallery_row"] for record in records]
    scene_ranks = full_gallery_ranks(scene_scores, gold)
    controlled_pare = pool_ranks(scores, random_pools, gold)
    controlled_scene = pool_ranks(scene_scores, random_pools, gold)
    hard_pare = pool_ranks(scores, hard_pools, gold)
    hard_scene = pool_ranks(scene_scores, hard_pools, gold)
    reports = {}
    for ki, k in enumerate(ks):
        started = time.perf_counter()
        query_records = copy.deepcopy(records)
        conditional, end_to_end, inclusions = [], [], []
        for qi, (record, target) in enumerate(zip(query_records, gold)):
            record["scene_full_gallery_rank"] = scene_ranks[qi]
            candidates = scene_order[qi, :k].tolist()
            included = target in candidates
            inclusions.append(int(included))
            rank = rank_of_gold(scores[qi, candidates], candidates, target) if included else None
            if included:
                conditional.append(rank)
            end_to_end.append(rank if rank is not None else float("inf"))
            record.update(natural_candidate_included=included, natural_PARE_rank=rank)
            if ki == 0:
                start, stop = qi * random_pools.shape[1], (qi + 1) * random_pools.shape[1]
                record["controlled_random_pool_ranks"] = {
                    "scene": controlled_scene[start:stop], "PARE": controlled_pare[start:stop]}
                record["appearance_adversarial_ranks"] = {"scene": hard_scene[qi], "PARE": hard_pare[qi]}
        candidate_recall = float(np.mean(inclusions) * 100)
        conditional_metrics = metrics(conditional)
        result = copy.deepcopy(base_result)
        result.update(scene_full_gallery=metrics(scene_ranks), queries=query_records,
                      natural_scene_shortlist={
                          "K": k, "candidate_recall_percent": candidate_recall,
                          "n_natural_inclusions": sum(inclusions),
                          "PARE_conditional_on_inclusion": conditional_metrics,
                          "PARE_end_to_end": metrics(end_to_end),
                          "candidate_recall_times_conditional_R@10_percent":
                              candidate_recall * conditional_metrics["R@10"] / 100
                              if conditional_metrics is not None else 0.0})
        if ki == 0:
            pool_k = random_pools.shape[-1]
            result["controlled_random_pools"] = {
                "protocol": "positive plus K-1 uniformly sampled gallery distractors",
                "K": pool_k, "pools_per_query": random_pools.shape[1],
                "random_ranker_theory": random_ranker_theory(pool_k),
                "scene": metrics(controlled_scene), "PARE": metrics(controlled_pare)}
            result["appearance_adversarial_pool"] = {
                "protocol": "positive plus M-1 highest-scoring scene-level negatives",
                "M": hard_pools.shape[-1], "scene": metrics(hard_scene), "PARE": metrics(hard_pare)}
        result["elapsed_seconds"] = time.perf_counter() - started
        result["elapsed_seconds_scope"] = "report derivation only; shared scoring is timed in independent_manifest.json"
        reports[k] = result
    return reports


def derive_baseline_report(score_sets, records, random_pools, hard_pools, base_result):
    gold = [record["gold_gallery_row"] for record in records]
    full = {name: full_gallery_ranks(scores, gold) for name, scores in score_sets.items()}
    random = {name: pool_ranks(scores, random_pools, gold) for name, scores in score_sets.items()}
    hard = {name: pool_ranks(scores, hard_pools, gold) for name, scores in score_sets.items()}
    result = copy.deepcopy(base_result)
    result["full_gallery"] = {name: metrics(values) for name, values in full.items()}
    result["controlled_random_pools"] = {
        "K": random_pools.shape[-1], "pools_per_query": random_pools.shape[1],
        **{name: metrics(values) for name, values in random.items()}}
    result["appearance_adversarial_pool"] = {
        "M": hard_pools.shape[-1], **{name: metrics(values) for name, values in hard.items()}}
    result["queries"] = copy.deepcopy(records)
    for qi, record in enumerate(result["queries"]):
        start, stop = qi * random_pools.shape[1], (qi + 1) * random_pools.shape[1]
        record["full_gallery_ranks"] = {name: values[qi] for name, values in full.items()}
        record["controlled_random_pool_ranks"] = {name: values[start:stop] for name, values in random.items()}
        record["appearance_adversarial_ranks"] = {name: values[qi] for name, values in hard.items()}
    return result


def validate_tensor_layout(cache):
    n, d, mc, mf = len(cache["ids"]), cache["D"], cache["Mc"], cache["Mf"]
    shapes = {"scene": (n, d), "agg": (n, d), "nu": (n,)}
    for pre, count in (("c", mc), ("f", mf)):
        shapes.update({pre + "_emb": (n, count, d), pre + "_mu": (n, count),
                       pre + "_t": (n, count), pre + "_tf": (n, count, 4)})
        counts = cache[pre + "_n"]
        if counts.dtype != torch.int64 or tuple(counts.shape) != (n,) or not ((counts >= 0) & (counts <= count)).all():
            raise ValueError("Invalid crop counts")
    for name, shape in shapes.items():
        value = cache[name]
        if tuple(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid float32 cache tensor: {name}")


def load_helper(path):
    spec = importlib.util.spec_from_file_location("independent_pare_batch_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--scoring-helper", type=Path, required=True)
    parser.add_argument("--physics-cache", type=Path, required=True)
    parser.add_argument("--uniform-grid-cache", type=Path, required=True)
    for name in VARIANTS:
        parser.add_argument("--head-" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--remoteclip", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--doc-chunk", type=int, default=512)
    parser.add_argument("--text-bs", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Explicit non-paper mode: 16 images, K=8/16, 1 pool, hard M=8")
    args = parser.parse_args()
    if min(args.query_chunk, args.doc_chunk, args.text_bs) < 1:
        parser.error("Batch sizes must be positive")
    args.code_dir = args.code_dir.resolve()
    args.scoring_helper = args.scoring_helper.resolve()
    sys.path.insert(0, str(args.code_dir))
    import open_clip
    import artifacts
    import retrieval_queries
    import runtime
    import saqre_phys
    import train_saqre
    modules = (artifacts, retrieval_queries, runtime, saqre_phys, train_saqre)
    if any(Path(module.__file__).resolve().parent != args.code_dir for module in modules):
        raise RuntimeError("Imported training module did not come from --code-dir")
    helper = load_helper(args.scoring_helper)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    device = torch.device(args.device)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {"captions": args.captions, "physics_cache": args.physics_cache,
              "uniform_grid_cache": args.uniform_grid_cache,
              **{"head_" + name: getattr(args, "head_" + name) for name in VARIANTS}}
    if args.remoteclip:
        inputs["external_checkpoint"] = args.remoteclip
    code = {path.name: path for path in args.code_dir.glob("*.py")}
    code.update({"independent_evaluator": Path(__file__).resolve(), "vmap_helper": args.scoring_helper})
    before = {"inputs": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in inputs.items()},
              "code": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in code.items()}}
    manifest = {"schema": "pare.independent-multik.v1", "status": "running", "started_utc": utcnow(),
                "purpose": "SMOKE ONLY, NOT PAPER RESULTS" if args.smoke else "Revised-model retrieval evaluation",
                "runner_attribution": "Independent external evaluator; original run_manifest is neither changed nor credited",
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "environment": {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                                "device": str(device), "tf32": False, "autocast": False},
                "score_matrix_layout": {"rows": "queries.json: queries order",
                                        "columns": "queries.json: gallery_ids order", "dtype": "float32"},
                "candidate_array_layout": "Rows map through gallery_ids stored inside each candidates_*.npz; natural K takes the first K columns of the full scene order",
                "timing_warning": "Shared full-gallery scoring and report derivation are timed separately; these are not independent per-K end-to-end latency benchmarks",
                **before, "timing_seconds": {}}
    manifest_path = args.out / "independent_manifest.json"
    write_json(manifest_path, manifest)
    try:
        cache_paths = {"physics": args.physics_cache, "uniform_grid": args.uniform_grid_cache}
        caches, signatures = {}, {}
        for mode, path in cache_paths.items():
            cache = torch.load(path, map_location="cpu", weights_only=True)
            signatures[mode] = artifacts.validate_cache(cache, args.model, args.pretrained, args.remoteclip)
            if signatures[mode]["primitive_mode"] != mode:
                raise ValueError(f"Wrong primitive mode for {mode}")
            validate_tensor_layout(cache)
            caches[mode] = cache
        ids = caches["physics"]["ids"]
        if ids != caches["uniform_grid"]["ids"] or caches["physics"]["D"] != caches["uniform_grid"]["D"]:
            raise ValueError("Physics and uniform-grid caches have different gallery order/dimension")
        if caches["physics"]["provenance"]["image_file_sha256"] != caches["uniform_grid"]["provenance"]["image_file_sha256"]:
            raise ValueError("Cache image provenance differs between primitive modes")
        expected_n = 16 if args.smoke else 3688
        if len(ids) != expected_n:
            raise ValueError(f"Expected exactly {expected_n} gallery images")
        records, query_protocol = retrieval_queries.select_queries(train_saqre.load_captions(args.captions), ids, "first-caption", 0)
        if len(records) != expected_n or query_protocol["truncated"] or query_protocol["gallery_images_without_evaluated_query"]:
            raise ValueError("Every gallery image must have exactly one first-caption query")
        heads = {}
        manifest["artifact_validation"] = {"cache_signatures": signatures, "heads": {}}
        for name, (mode, expected_flags) in VARIANTS.items():
            checkpoint = torch.load(getattr(args, "head_" + name), map_location="cpu", weights_only=True)
            artifacts.validate_head(checkpoint, caches[mode])
            if set(checkpoint.get("ablate", [])) != expected_flags:
                raise ValueError(f"Wrong trained ablation flags for {name}")
            train_ids = checkpoint["provenance"].get("train_ids", [])
            if not train_ids or len(train_ids) != len(set(train_ids)):
                raise ValueError("Missing or duplicate training-ID provenance")
            if set(train_ids) & set(ids):
                raise ValueError("Training/gallery image IDs overlap")
            trained_helper = checkpoint["provenance"].get("batch_scoring_sha256")
            if trained_helper and trained_helper != before["code"]["vmap_helper"]["sha256"]:
                raise ValueError("Training and evaluation batching helpers differ; an explicit equivalence review is required")
            if not args.smoke and (checkpoint.get("n_images") != 9181 or checkpoint.get("n_pairs") != 31945
                                   or checkpoint.get("config", {}).get("epochs") != 10
                                   or len(checkpoint.get("history", [])) != 10 or len(train_ids) != 9181):
                raise ValueError("Formal evaluation requires the declared complete retained-train checkpoint")
            manifest["artifact_validation"]["heads"][name] = {
                "ablate": sorted(expected_flags), "n_images": checkpoint.get("n_images"),
                "n_pairs": checkpoint.get("n_pairs"), "config": checkpoint.get("config"),
                "training_provenance": checkpoint["provenance"], "history": checkpoint.get("history")}
            head = saqre_phys.SAQREHead(caches[mode]["D"]).to(device)
            head.load_state_dict(checkpoint["state_dict"], strict=True)
            head.ablate = artifacts.parse_ablation_flags(",".join(sorted(expected_flags)), caches[mode])
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
            if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
                raise FloatingPointError("Non-finite head weights")
            heads[name] = head
        started = time.perf_counter()
        model, _, _ = runtime.create_backbone(args.model, args.pretrained, args.remoteclip)
        tokenizer = open_clip.get_tokenizer(args.model)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(records), args.text_bs):
                tokens = tokenizer([record["caption"] for record in records[start:start + args.text_bs]]).to(device)
                chunks.append(F.normalize(model.encode_text(tokens), dim=-1).float())
        queries = torch.cat(chunks)
        if not torch.isfinite(queries).all():
            raise FloatingPointError("Non-finite query embeddings")
        del model
        np.save(args.out / "query_embeddings.npy", queries.cpu().numpy(), allow_pickle=False)
        write_json(args.out / "queries.json", {"query_protocol": query_protocol, "gallery_ids": ids, "queries": records})
        manifest["timing_seconds"]["text_encoding"] = time.perf_counter() - started
        ks = [8, 16] if args.smoke else [100, 200, 500, 1000, 2000, 3688]
        pools, hard_m = (1, 8) if args.smoke else (5, 50)
        gold = [record["gold_gallery_row"] for record in records]
        random_pools = make_random_pools(gold, len(ids), ks[0], pools, 0)
        started = time.perf_counter()
        baselines = score_baselines(queries, caches["physics"], device, args.doc_chunk)
        manifest["timing_seconds"]["training_free_baselines"] = time.perf_counter() - started
        scenes = {"physics": baselines["scene"]}
        with torch.no_grad():
            scenes["uniform_grid"] = (queries @ F.normalize(caches["uniform_grid"]["scene"].to(device), dim=-1).T).cpu()
        manifest["scene_cache_max_abs_difference"] = float((scenes["physics"] - scenes["uniform_grid"]).abs().max())
        orders, hard = {}, {}
        for mode, scene in scenes.items():
            np.save(args.out / f"scene_scores_{mode}.npy", scene.numpy(), allow_pickle=False)
            orders[mode] = torch.argsort(scene.to(device), dim=1, descending=True).cpu().numpy().astype(np.int32)
            hard[mode] = make_hard_pools(orders[mode], gold, hard_m)
            np.savez_compressed(args.out / f"candidates_{mode}.npz", gallery_ids=np.asarray(ids),
                                query_ids=np.asarray([record["query_id"] for record in records]),
                                random_pool_gallery_rows=random_pools, hard_pool_gallery_rows=hard[mode],
                                natural_full_scene_order_gallery_rows=orders[mode], gold_gallery_rows=np.asarray(gold, dtype=np.int32))
        def base_for(mode, name=None):
            provenance = {"cache_signature": signatures[mode], "allow_legacy_artifacts": False,
                          "cache_sha256": before["inputs"][mode + "_cache"]["sha256"],
                          "captions_sha256": before["inputs"]["captions"]["sha256"],
                          "evaluator_sha256": before["code"]["independent_evaluator"]["sha256"],
                          "query_selection_sha256": before["code"]["retrieval_queries.py"]["sha256"],
                          "batch_scoring_sha256": before["code"]["vmap_helper"]["sha256"],
                          "independent_manifest": "independent_manifest.json", "evaluation_backend": "independent-full-float32-vmap-scores"}
            if name:
                provenance["head_sha256"] = before["inputs"]["head_" + name]["sha256"]
            return {"model": args.model, "remoteclip": bool(args.remoteclip), "n_queries": len(records),
                    "n_gallery": len(ids), "seed": 0, "query_protocol": query_protocol,
                    "gallery_ids": ids, "provenance": provenance}
        for name, (mode, _) in VARIANTS.items():
            print(f"Scoring {name}: {len(records)} queries x {len(ids)} images", flush=True)
            started = time.perf_counter()
            scores = score_head_matrix(heads[name], queries, caches[mode], train_saqre.move_docs,
                                       helper.score_matrix, device, args.query_chunk, args.doc_chunk)
            manifest["timing_seconds"]["head_scores_" + name] = time.perf_counter() - started
            np.save(args.out / f"scores_{name}.npy", scores.numpy(), allow_pickle=False)
            base = base_for(mode, name)
            base["ablate"] = sorted(heads[name].ablate)
            started = time.perf_counter()
            reports = derive_head_reports(scores.to(device), scenes[mode].to(device), records, ks,
                                          random_pools, hard[mode], orders[mode], base)
            for k, report in reports.items():
                write_json(args.out / f"eval_{name}_k{k}.json", report)
            manifest["timing_seconds"]["protocol_derivation_" + name] = time.perf_counter() - started
            write_json(manifest_path, manifest)
        for name, scores in baselines.items():
            np.save(args.out / f"baseline_scores_{name}.npy", scores.numpy(), allow_pickle=False)
        baseline_result = derive_baseline_report({name: scores.to(device) for name, scores in baselines.items()},
                                                records, random_pools, hard["physics"], base_for("physics"))
        write_json(args.out / "eval_baselines.json", baseline_result)
        after = {"inputs": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in inputs.items()},
                 "code": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in code.items()}}
        if after != before:
            raise RuntimeError("Inputs or source code changed during evaluation")
        manifest["outputs"] = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                               for path in sorted(args.out.iterdir()) if path.is_file() and path != manifest_path}
        manifest.update(status="complete", finished_utc=utcnow(), input_code_recheck_passed=True,
                        expected_eval_json_count=len(VARIANTS) * len(ks) + 1)
        write_json(manifest_path, manifest)
        print(json.dumps({"status": "complete", "manifest": str(manifest_path)}, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status="failed", finished_utc=utcnow(), error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
