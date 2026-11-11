#!/usr/bin/env python3
"""Evaluate training-free scene, mean-pool, and MaxSim retrieval baselines.

The script reports full-gallery metrics, the paper's controlled random pools,
and the appearance-adversarial pools used in the hard-negative table. Candidate
sampling is identical to ``eval_retrieval_phys.py`` for the same seed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F

from train_saqre import load_captions
from runtime import create_backbone, get_device
from retrieval_queries import add_query_arguments, select_queries
from artifacts import file_sha256, validate_cache


def metrics(ranks):
    values = np.asarray(ranks, dtype=np.float64)
    return {
        "R@1": float((values <= 1).mean() * 100.0),
        "R@5": float((values <= 5).mean() * 100.0),
        "R@10": float((values <= 10).mean() * 100.0),
        "MRR@10": float(
            np.mean([1.0 / rank if rank <= 10 else 0.0 for rank in values]) * 100.0
        ),
    }


def ranks_from_scores(scores, gold_rows):
    order = torch.argsort(scores, dim=1, descending=True)
    return [
        (order[index] == gold).nonzero(as_tuple=True)[0].item() + 1
        for index, gold in enumerate(gold_rows)
    ]


def rank_in_pool(scores, candidates, gold):
    local_order = torch.argsort(scores[candidates], descending=True).tolist()
    return [candidates[index] for index in local_order].index(gold) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--remoteclip", default=None)
    add_query_arguments(parser)
    parser.add_argument("--allow_legacy_artifacts", action="store_true",
                        help="Explicitly permit unsigned historical caches; never a new verified run")
    parser.add_argument("--text_bs", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--candidate_K", type=int, default=100)
    parser.add_argument("--pools", type=int, default=5)
    parser.add_argument("--hard_M", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    cache_signature = validate_cache(cache, args.model, args.pretrained,
                                     checkpoint=args.remoteclip, allow_legacy=args.allow_legacy_artifacts)
    ids = cache["ids"]
    max_regions = cache["Mc"]
    max_scatterers = cache["Mf"]
    gallery_size = len(ids)
    if min(args.candidate_K, args.hard_M) < 2 or max(args.candidate_K, args.hard_M) > gallery_size:
        parser.error("candidate_K and hard_M must be between 2 and gallery size")
    if args.max_q < 0 or min(args.text_bs, args.pools, args.chunk) < 1:
        parser.error("max_q must be nonnegative; batch, pool and chunk counts must be positive")

    model, _, _ = create_backbone(args.model, args.pretrained, args.remoteclip)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    captions = load_captions(args.captions)
    query_records, query_protocol = select_queries(captions, ids, args.query_mode, args.max_q)
    queries = [(record["gold_gallery_row"], record["caption"]) for record in query_records]
    gold_rows = [row for row, _ in queries]
    if not queries:
        raise ValueError("No caption queries match the cached image IDs")
    print(f"queries={len(queries)} gallery={gallery_size}", flush=True)

    @torch.no_grad()
    def embed_texts(texts):
        chunks = []
        for start in range(0, len(texts), args.text_bs):
            tokens = tokenizer(texts[start : start + args.text_bs]).to(device)
            chunks.append(F.normalize(model.encode_text(tokens), dim=-1))
        return torch.cat(chunks, dim=0)

    query_embeddings = embed_texts([caption for _, caption in queries])
    query_count = query_embeddings.shape[0]
    scene = F.normalize(cache["scene"].to(device), dim=-1)
    region_embeddings = F.normalize(cache["c_emb"].to(device), dim=-1)
    scatterer_embeddings = F.normalize(cache["f_emb"].to(device), dim=-1)
    region_counts = cache["c_n"].to(device)
    scatterer_counts = cache["f_n"].to(device)
    region_mask = torch.arange(max_regions, device=device).unsqueeze(0) < region_counts.unsqueeze(1)
    scatterer_mask = (
        torch.arange(max_scatterers, device=device).unsqueeze(0)
        < scatterer_counts.unsqueeze(1)
    )

    scene_scores = query_embeddings @ scene.T

    # Query-agnostic mean pooling over all valid primitives.
    region_sum = (region_embeddings * region_mask.unsqueeze(-1)).sum(1)
    scatterer_sum = (scatterer_embeddings * scatterer_mask.unsqueeze(-1)).sum(1)
    counts = (
        1.0
        + region_mask.sum(1, keepdim=True).float()
        + scatterer_mask.sum(1, keepdim=True).float()
    )
    mean_documents = F.normalize((scene + region_sum + scatterer_sum) / counts, dim=-1)
    mean_scores = query_embeddings @ mean_documents.T

    # One-vector-query MaxSim over scene/region/scatterer primitives.
    negative = torch.finfo(torch.float32).min
    maxsim_scores = torch.empty(query_count, gallery_size, device=device)
    with torch.no_grad():
        for start in range(0, gallery_size, args.chunk):
            end = min(start + args.chunk, gallery_size)
            scene_part = query_embeddings @ scene[start:end].T
            region_part = torch.einsum(
                "qd,bmd->qbm", query_embeddings, region_embeddings[start:end]
            )
            region_part = region_part.masked_fill(
                ~region_mask[start:end].unsqueeze(0), negative
            ).max(-1).values
            scatterer_part = torch.einsum(
                "qd,bmd->qbm", query_embeddings, scatterer_embeddings[start:end]
            )
            scatterer_part = scatterer_part.masked_fill(
                ~scatterer_mask[start:end].unsqueeze(0), negative
            ).max(-1).values
            maxsim_scores[:, start:end] = torch.maximum(
                torch.maximum(scene_part, region_part), scatterer_part
            )

    score_sets = {"scene": scene_scores, "mean_pool": mean_scores, "MaxSim": maxsim_scores}
    full_ranks = {name: ranks_from_scores(scores, gold_rows) for name, scores in score_sets.items()}
    output = {
        "model": args.model,
        "remoteclip": bool(args.remoteclip),
        "n_queries": query_count,
        "n_gallery": gallery_size,
        "seed": args.seed,
        "full_gallery": {
            name: metrics(ranks) for name, ranks in full_ranks.items()
        },
        "query_protocol": query_protocol,
        "gallery_ids": ids,
        "provenance": {
            "cache_signature": cache_signature,
            "allow_legacy_artifacts": args.allow_legacy_artifacts,
            "cache_sha256": file_sha256(args.cache),
            "captions_sha256": file_sha256(args.captions),
            "evaluator_sha256": file_sha256(__file__),
            "query_selection_sha256": file_sha256(Path(__file__).with_name("retrieval_queries.py")),
        },
    }

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    random_ranks = {name: [] for name in score_sets}
    for query_index, gold in enumerate(gold_rows):
        for _ in range(args.pools):
            sampled = torch.randperm(gallery_size - 1, generator=generator)[
                : args.candidate_K - 1
            ]
            sampled = sampled + (sampled >= gold).long()
            candidates = [gold] + sampled.tolist()
            for name, scores in score_sets.items():
                random_ranks[name].append(
                    rank_in_pool(scores[query_index], candidates, gold)
                )
    output["controlled_random_pools"] = {
        "K": args.candidate_K,
        "pools_per_query": args.pools,
        **{name: metrics(ranks) for name, ranks in random_ranks.items()},
    }

    full_scene_order = torch.argsort(scene_scores, dim=1, descending=True)
    hard_ranks = {name: [] for name in score_sets}
    for query_index, gold in enumerate(gold_rows):
        negatives = []
        for row in full_scene_order[query_index].tolist():
            if row != gold:
                negatives.append(row)
            if len(negatives) == args.hard_M - 1:
                break
        candidates = [gold] + negatives
        for name, scores in score_sets.items():
            hard_ranks[name].append(rank_in_pool(scores[query_index], candidates, gold))
    output["appearance_adversarial_pool"] = {
        "M": args.hard_M,
        **{name: metrics(ranks) for name, ranks in hard_ranks.items()},
    }
    for index, record in enumerate(query_records):
        start = index * args.pools
        record["full_gallery_ranks"] = {name: ranks[index] for name, ranks in full_ranks.items()}
        record["controlled_random_pool_ranks"] = {
            name: ranks[start:start + args.pools] for name, ranks in random_ranks.items()
        }
        record["appearance_adversarial_ranks"] = {name: ranks[index] for name, ranks in hard_ranks.items()}
    output["queries"] = query_records

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ("queries", "gallery_ids")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
