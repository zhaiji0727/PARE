#!/usr/bin/env python3
"""Evaluate PARE on controlled pools and a natural first-stage shortlist.

The paper's main table is a controlled second-stage experiment: each query is
evaluated in several size-K pools containing the positive and K-1 uniformly
sampled distractors. This script additionally reports full-gallery scene
retrieval, candidate recall for the natural scene top-K shortlist, conditional
PARE recall on natural inclusions, and exact end-to-end recall, where a missed
first-stage positive is counted as a failure.

An appearance-adversarial pool, containing the positive and the M-1 most
similar scene-level negatives, is also reported. No result from a forced-
positive pool is labelled full-gallery retrieval.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from saqre_phys import SAQREHead
from train_saqre import load_captions, move_docs
from runtime import create_backbone, get_device
from retrieval_queries import add_query_arguments, select_queries
from artifacts import file_sha256, parse_ablation_flags, validate_cache, validate_head


def metrics(ranks):
    values = np.asarray(ranks, dtype=np.float64)
    if values.size == 0:
        return None
    return {
        "R@1": float((values <= 1).mean() * 100.0),
        "R@5": float((values <= 5).mean() * 100.0),
        "R@10": float((values <= 10).mean() * 100.0),
        "MRR@10": float(
            np.mean([1.0 / rank if rank <= 10 else 0.0 for rank in values]) * 100.0
        ),
    }


def rank_of_gold(scores, candidates, gold):
    sorted_local = torch.argsort(scores, descending=True).tolist()
    sorted_global = [candidates[index] for index in sorted_local]
    return sorted_global.index(gold) + 1


def random_ranker_theory(pool_size):
    top = min(10, pool_size)
    return {
        "R@1": 100.0 / pool_size,
        "R@5": 100.0 * min(5, pool_size) / pool_size,
        "R@10": 100.0 * top / pool_size,
        "MRR@10": 100.0 * sum(1.0 / rank for rank in range(1, top + 1)) / pool_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="test primitives cache")
    parser.add_argument("--captions", required=True, help="Caption_test.csv")
    parser.add_argument("--head", required=True, help="trained PARE head")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--remoteclip", default=None)
    parser.add_argument("--candidate_K", type=int, default=100)
    parser.add_argument("--pools", type=int, default=5)
    parser.add_argument("--hard_M", type=int, default=50)
    add_query_arguments(parser)
    parser.add_argument("--text_bs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablate", default="", help="comma-separated: sr,ty,cfar,cfar_strict")
    parser.add_argument("--skip_random", action="store_true")
    parser.add_argument("--skip_hard", action="store_true")
    parser.add_argument("--allow_legacy_artifacts", action="store_true",
                        help="Explicitly permit unsigned historical caches/checkpoints; never a new verified run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.candidate_K < 2 or args.hard_M < 2:
        raise ValueError("candidate_K and hard_M must be at least 2")
    if args.max_q < 0 or min(args.text_bs, args.pools) < 1:
        parser.error("max_q must be nonnegative; text_bs and pools must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device()

    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    cache_signature = validate_cache(cache, args.model, args.pretrained,
                                     checkpoint=args.remoteclip, allow_legacy=args.allow_legacy_artifacts)
    ids = cache["ids"]
    dimension = cache["D"]
    gallery_size = len(ids)
    if args.candidate_K > gallery_size or args.hard_M > gallery_size:
        raise ValueError("candidate pool cannot exceed gallery size")
    model, _, _ = create_backbone(args.model, args.pretrained, args.remoteclip)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    validate_head(checkpoint, cache, allow_legacy=args.allow_legacy_artifacts)
    head = SAQREHead(dimension).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.ablate = parse_ablation_flags(args.ablate, cache)
    if set(checkpoint.get("ablate", [])) != head.ablate:
        raise ValueError("Evaluation ablation flags do not match the trained head")
    head.eval()

    captions = load_captions(args.captions)
    query_records, query_protocol = select_queries(captions, ids, args.query_mode, args.max_q)
    queries = [(record["gold_gallery_row"], record["caption"]) for record in query_records]
    gold_rows = [row for row, _ in queries]
    if not queries:
        raise ValueError("No caption queries match the cached image IDs")
    print(f"queries={len(queries)} gallery={gallery_size} device={device}", flush=True)

    @torch.no_grad()
    def embed_texts(texts):
        chunks = []
        for start in range(0, len(texts), args.text_bs):
            tokens = tokenizer(texts[start : start + args.text_bs]).to(device)
            chunks.append(F.normalize(model.encode_text(tokens), dim=-1))
        return torch.cat(chunks, dim=0)

    query_embeddings = embed_texts([caption for _, caption in queries])
    scene = F.normalize(cache["scene"].to(device), dim=-1)
    scene_scores = query_embeddings @ scene.T
    full_order = torch.argsort(scene_scores, dim=1, descending=True)

    full_scene_ranks = []
    for query_index, gold in enumerate(gold_rows):
        full_scene_ranks.append(
            (full_order[query_index] == gold).nonzero(as_tuple=True)[0].item() + 1
        )

    result = {
        "model": args.model,
        "remoteclip": bool(args.remoteclip),
        "ablate": sorted(head.ablate),
        "n_queries": len(queries),
        "n_gallery": gallery_size,
        "seed": args.seed,
        "scene_full_gallery": metrics(full_scene_ranks),
        "query_protocol": query_protocol,
        "gallery_ids": ids,
        "provenance": {
            "cache_signature": cache_signature,
            "allow_legacy_artifacts": args.allow_legacy_artifacts,
            "cache_sha256": file_sha256(args.cache),
            "head_sha256": file_sha256(args.head),
            "captions_sha256": file_sha256(args.captions),
            "evaluator_sha256": file_sha256(__file__),
            "query_selection_sha256": file_sha256(Path(__file__).with_name("retrieval_queries.py")),
        },
    }
    for record, rank in zip(query_records, full_scene_ranks):
        record["scene_full_gallery_rank"] = rank

    started = time.time()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Main controlled K-candidate protocol: positive + K-1 random negatives.
    if not args.skip_random:
        controlled_scene_ranks = []
        controlled_pare_ranks = []
        with torch.no_grad():
            for query_index, gold in enumerate(gold_rows):
                for _ in range(args.pools):
                    sampled = torch.randperm(gallery_size - 1, generator=generator)[
                        : args.candidate_K - 1
                    ]
                    sampled = sampled + (sampled >= gold).long()
                    candidates = [gold] + sampled.tolist()
                    local_scene = scene_scores[query_index, candidates]
                    controlled_scene_ranks.append(rank_of_gold(local_scene, candidates, gold))
                    documents = move_docs(cache, torch.tensor(candidates), device)
                    fused = head.fuse_batch(query_embeddings[query_index], documents)
                    controlled_pare_ranks.append(
                        rank_of_gold(fused @ query_embeddings[query_index], candidates, gold)
                    )
        result["controlled_random_pools"] = {
            "protocol": "positive plus K-1 uniformly sampled gallery distractors",
            "K": args.candidate_K,
            "pools_per_query": args.pools,
            "random_ranker_theory": random_ranker_theory(args.candidate_K),
            "scene": metrics(controlled_scene_ranks),
            "PARE": metrics(controlled_pare_ranks),
        }
        for index, record in enumerate(query_records):
            start = index * args.pools
            record["controlled_random_pool_ranks"] = {
                "scene": controlled_scene_ranks[start:start + args.pools],
                "PARE": controlled_pare_ranks[start:start + args.pools],
            }

    # Natural scene top-K shortlist: no forced positive. This is the matching
    # distribution for candidate recall, conditional recall, and their product.
    natural_inclusions = []
    natural_pare_ranks = []
    end_to_end_ranks = []
    with torch.no_grad():
        for query_index, gold in enumerate(gold_rows):
            candidates = full_order[query_index, : args.candidate_K].tolist()
            included = gold in candidates
            natural_inclusions.append(int(included))
            query_records[query_index]["natural_candidate_included"] = included
            if included:
                documents = move_docs(cache, torch.tensor(candidates), device)
                fused = head.fuse_batch(query_embeddings[query_index], documents)
                rank = rank_of_gold(fused @ query_embeddings[query_index], candidates, gold)
                natural_pare_ranks.append(rank)
                end_to_end_ranks.append(rank)
                query_records[query_index]["natural_PARE_rank"] = rank
            else:
                end_to_end_ranks.append(float("inf"))
                query_records[query_index]["natural_PARE_rank"] = None
    candidate_recall = float(np.mean(natural_inclusions) * 100.0)
    conditional_metrics = metrics(natural_pare_ranks)
    product_r10 = (
        candidate_recall * conditional_metrics["R@10"] / 100.0
        if conditional_metrics is not None
        else 0.0
    )
    result["natural_scene_shortlist"] = {
        "K": args.candidate_K,
        "candidate_recall_percent": candidate_recall,
        "n_natural_inclusions": int(sum(natural_inclusions)),
        "PARE_conditional_on_inclusion": conditional_metrics,
        "PARE_end_to_end": metrics(end_to_end_ranks),
        "candidate_recall_times_conditional_R@10_percent": product_r10,
    }

    # Positive plus the M-1 strongest scene-level negatives. This controlled
    # stress test never receives the label "full-gallery retrieval".
    if not args.skip_hard:
        hard_scene_ranks = []
        hard_pare_ranks = []
        with torch.no_grad():
            for query_index, gold in enumerate(gold_rows):
                strongest_negatives = []
                for row in full_order[query_index].tolist():
                    if row != gold:
                        strongest_negatives.append(row)
                    if len(strongest_negatives) == args.hard_M - 1:
                        break
                candidates = [gold] + strongest_negatives
                hard_scene_ranks.append(
                    rank_of_gold(scene_scores[query_index, candidates], candidates, gold)
                )
                documents = move_docs(cache, torch.tensor(candidates), device)
                fused = head.fuse_batch(query_embeddings[query_index], documents)
                hard_pare_ranks.append(
                    rank_of_gold(fused @ query_embeddings[query_index], candidates, gold)
                )
        result["appearance_adversarial_pool"] = {
            "protocol": "positive plus M-1 highest-scoring scene-level negatives",
            "M": args.hard_M,
            "scene": metrics(hard_scene_ranks),
            "PARE": metrics(hard_pare_ranks),
        }
        for record, scene_rank, pare_rank in zip(query_records, hard_scene_ranks, hard_pare_ranks):
            record["appearance_adversarial_ranks"] = {"scene": scene_rank, "PARE": pare_rank}

    result["queries"] = query_records
    result["elapsed_seconds"] = time.time() - started
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("queries", "gallery_ids")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
