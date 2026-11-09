#!/usr/bin/env python3
"""Efficiency benchmark for PARE (the analysis the paper promises but
never delivered: parameters, offline indexing cost, online rerank latency).

All numbers are measured end-to-end on the same hardware and the SAME cached
primitives / trained head used by the retrieval tables. Nothing is interpolated.

Reports, as JSON + a console table:
  (A) Parameter budget: trainable PARE-head params vs frozen CLIP backbone
      params (the "lightweight head on a frozen backbone" claim).
  (B) Offline indexing cost (one-time, amortized over the corpus): wall-clock
      to extract SAR primitives (ENL/CFAR/region) + encode them, per image,
      and the resulting cache footprint per image.
  (C) Online query cost: added latency of the PARE re-ranker over a candidate
      pool of M, vs a plain scene-CLIP cosine lookup. Reported as ms/query and
      queries/s, mean +/- std over repeated trials (GPU-synced).
  (D) Evidence-budget savings handled separately by spec_ablation.py (Avg.k).

Usage:
  python bench_efficiency.py --cache results/cache_vitb32_test.pt \
      --head results/head_vitb32_full_v2.pt --out results/efficiency.json
"""
import argparse, json, os, time, statistics
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from saqre_phys import SAQREHead
from primitive_controls import extract_for_mode
from artifacts import file_sha256, parse_ablation_flags, validate_cache, validate_head
from train_saqre import move_docs
from runtime import DATA_ROOT, RESULTS_ROOT, create_backbone, get_device


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="test primitives cache .pt")
    ap.add_argument("--head", required=True, help="trained PARE head .pt")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--img_root", default=str(DATA_ROOT / "SARimages_pre"))
    ap.add_argument("--remoteclip", default=None)
    ap.add_argument("--M", type=int, default=50, help="rerank candidate-pool size")
    ap.add_argument("--n_query_trials", type=int, default=500)
    ap.add_argument("--n_index_imgs", type=int, default=100, help="imgs to time offline indexing")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--allow_legacy_artifacts", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_ROOT / "efficiency.json"))
    args = ap.parse_args()

    dev = get_device()
    gpu_name = torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else str(dev)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    signature = validate_cache(cache, args.model, args.pretrained, args.remoteclip,
                               args.allow_legacy_artifacts)
    ids = cache["ids"]; D = cache["D"]; N = len(ids)
    if not 1 <= args.M <= N:
        ap.error("M must be between 1 and the cached gallery size")
    if min(args.n_query_trials, args.n_index_imgs) < 1 or args.warmup < 0:
        ap.error("trial/image counts must be positive and warmup nonnegative")
    scene = F.normalize(cache["scene"].to(dev), dim=-1)            # (N,D) frozen index

    # ---- backbone (frozen) + head (trainable) ----
    model, _, preprocess = create_backbone(args.model, args.pretrained, args.remoteclip)
    model = model.to(dev).eval()
    head = SAQREHead(D).to(dev)
    checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    validate_head(checkpoint, cache, args.allow_legacy_artifacts)
    head.load_state_dict(checkpoint["state_dict"])
    head.ablate = parse_ablation_flags(",".join(checkpoint.get("ablate", [])), cache)
    head.eval()

    # =========================================================
    # (A) PARAMETER BUDGET
    # =========================================================
    head_params = sum(p.numel() for p in head.parameters())
    head_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.parameters())
    vis_params = sum(p.numel() for p in model.visual.parameters())
    per_param = {n: int(p.numel()) for n, p in head.named_parameters()}
    paramA = {
        "head_total": int(head_params),
        "head_trainable": int(head_train),
        "backbone_total": int(backbone_params),
        "backbone_visual": int(vis_params),
        "head_pct_of_backbone": 100.0 * head_train / backbone_params,
        "head_pct_of_visual": 100.0 * head_train / vis_params,
        "head_breakdown": per_param,
    }

    # =========================================================
    # (B) OFFLINE INDEXING COST (per image, one-time)
    # =========================================================
    # time: primitive extraction (CPU: ENL/CFAR/region) + crop encoding (GPU)
    img_index = {}
    for dp, _, fns in os.walk(args.img_root):
        for fn in fns:
            if fn.endswith(".png"):
                img_index.setdefault(fn, os.path.join(dp, fn))
    sample = [b for b in ids if b in img_index][: args.n_index_imgs]
    if not sample:
        raise ValueError("No cached image IDs can be resolved under img_root")
    t_extract, t_preprocess, t_encode, t_total, n_crops_list = [], [], [], [], []
    for b in sample:
        im = Image.open(img_index[b]).convert("RGB")
        sync(dev); t0 = time.perf_counter()
        prim = extract_for_mode(im, max_regions=cache["Mc"], max_sc=cache["Mf"],
                                mode=signature["primitive_mode"] if signature else "physics")
        t1 = time.perf_counter()
        crops = [prim["scene"][0]] + [c for c, _ in prim["regions"]] + [c for c, _ in prim["scatterers"]]
        batch = torch.stack([preprocess(im2) for im2 in crops]).to(dev)
        sync(dev); t2 = time.perf_counter()
        _ = F.normalize(model.encode_image(batch), dim=-1)
        sync(dev); t3 = time.perf_counter()
        t_extract.append((t1 - t0) * 1e3)
        t_preprocess.append((t2 - t1) * 1e3)
        t_encode.append((t3 - t2) * 1e3)
        t_total.append((t3 - t0) * 1e3)
        n_crops_list.append(len(crops))
    # cache footprint per image: sum of per-image tensor bytes / N
    cache_bytes = 0
    for k, v in cache.items():
        if torch.is_tensor(v):
            cache_bytes += v.element_size() * v.nelement()
    indexB = {
        "extract_ms_mean": float(np.mean(t_extract)),
        "extract_ms_std": float(np.std(t_extract)),
        "encode_ms_mean": float(np.mean(t_encode)),
        "encode_ms_std": float(np.std(t_encode)),
        "preprocess_and_transfer_ms_mean": float(np.mean(t_preprocess)),
        "total_index_ms_per_img": float(np.mean(t_total)),
        "timing_scope": "Extraction + crop preprocessing + device transfer + encoding; excludes image-file decoding and cache serialization",
        "timed_image_ids": sample,
        "mean_crops_per_img": float(np.mean(n_crops_list)),
        "cache_bytes_per_img": float(cache_bytes / N),
        "n_timed": len(sample),
    }

    # =========================================================
    # (C) ONLINE QUERY COST: plain scene lookup vs PARE rerank
    # =========================================================
    M = args.M
    torch.manual_seed(0)
    # random query embeddings drawn on the unit sphere (query-cost is geometry-
    # independent; we measure compute, not accuracy here)
    Q = args.n_query_trials + args.warmup
    qs = F.normalize(torch.randn(Q, D, device=dev), dim=-1)
    # candidate pools: top-M by scene cosine (the real pipeline's pool)
    # precompute once outside timing
    pools = []
    for i in range(Q):
        sims = scene @ qs[i]
        pool = torch.topk(sims, M).indices
        pools.append(pool)

    # --- (C1) plain scene-CLIP full-gallery cosine + top-k (baseline) ---
    base_ms = []
    for i in range(Q):
        sync(dev); t0 = time.perf_counter()
        sims = scene @ qs[i]
        _ = torch.topk(sims, min(10, N))
        sync(dev); t1 = time.perf_counter()
        if i >= args.warmup:
            base_ms.append((t1 - t0) * 1e3)

    # --- (C2) PARE rerank of the M-candidate pool (added stage) ---
    rer_ms = []
    for i in range(Q):
        cand = pools[i]
        docs = move_docs(cache, cand.cpu(), dev)
        sync(dev); t0 = time.perf_counter()
        fused = head.fuse_batch(qs[i], docs)
        _ = torch.argsort(fused @ qs[i], descending=True)
        sync(dev); t1 = time.perf_counter()
        if i >= args.warmup:
            rer_ms.append((t1 - t0) * 1e3)

    # --- (C3) end-to-end PARE query = scene shortlist + rerank ---
    e2e_ms = []
    for i in range(Q):
        sync(dev); t0 = time.perf_counter()
        sims = scene @ qs[i]
        cand = torch.topk(sims, M).indices
        docs = move_docs(cache, cand.cpu(), dev)
        fused = head.fuse_batch(qs[i], docs)
        _ = torch.argsort(fused @ qs[i], descending=True)
        sync(dev); t1 = time.perf_counter()
        if i >= args.warmup:
            e2e_ms.append((t1 - t0) * 1e3)

    def stat(x):
        return {"ms_mean": float(np.mean(x)), "ms_std": float(np.std(x)),
                "ms_p50": float(np.percentile(x, 50)), "ms_p95": float(np.percentile(x, 95)),
                "qps": float(1000.0 / np.mean(x))}
    queryC = {
        "query_scope": "Synthetic normalized embeddings; no text-tokenization/encoding or retrieval accuracy is measured",
        "rerank_timing_scope": "Candidate document tensors already on device; excludes gathering and transfer",
        "pipeline_timing_scope": "Scene search + candidate gathering/transfer + reranking; excludes text encoding",
        "M": M, "gallery_N": N, "n_trials": len(base_ms),
        "scene_baseline": stat(base_ms),
        "pare_rerank_only": stat(rer_ms),
        "pare_end_to_end": stat(e2e_ms),
        "rerank_overhead_ms": float(np.mean(rer_ms)),
        "overhead_vs_baseline_x": float(np.mean(e2e_ms) / np.mean(base_ms)),
    }

    out = {
        "hardware": gpu_name, "torch": torch.__version__, "model": args.model,
        "provenance": {"cache_signature": signature, "cache_sha256": file_sha256(args.cache),
                       "head_sha256": file_sha256(args.head), "benchmark_sha256": file_sha256(__file__)},
        "A_parameters": paramA, "B_offline_index": indexB, "C_online_query": queryC,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    # ---- console summary ----
    print("\n=================  EFFICIENCY BENCHMARK  =================")
    print(f"hardware: {gpu_name} | backbone: {args.model}")
    print("\n(A) PARAMETERS")
    print(f"  PARE head (trainable): {head_train:,}  ({head_train/1e3:.1f}K)")
    print(f"  CLIP backbone (frozen): {backbone_params:,}  (visual {vis_params:,})")
    print(f"  head / backbone = {paramA['head_pct_of_backbone']:.3f}%   "
          f"head / visual = {paramA['head_pct_of_visual']:.3f}%")
    print("\n(B) OFFLINE INDEXING (one-time, per image)")
    print(f"  primitive extract: {indexB['extract_ms_mean']:.1f} +/- {indexB['extract_ms_std']:.1f} ms")
    print(f"  crop encode:       {indexB['encode_ms_mean']:.1f} +/- {indexB['encode_ms_std']:.1f} ms")
    print(f"  total:             {indexB['total_index_ms_per_img']:.1f} ms/img  "
          f"({indexB['mean_crops_per_img']:.1f} crops)  cache {indexB['cache_bytes_per_img']/1024:.1f} KB/img")
    print("\n(C) ONLINE QUERY (M=%d pool, gallery N=%d)" % (M, N))
    print(f"  scene-CLIP baseline: {queryC['scene_baseline']['ms_mean']:.3f} ms  "
          f"({queryC['scene_baseline']['qps']:.0f} qps)")
    print(f"  PARE rerank only:    {queryC['pare_rerank_only']['ms_mean']:.3f} ms  "
          f"({queryC['pare_rerank_only']['qps']:.0f} qps)")
    print(f"  PARE end-to-end:     {queryC['pare_end_to_end']['ms_mean']:.3f} ms  "
          f"({queryC['pare_end_to_end']['qps']:.0f} qps)  "
          f"= {queryC['overhead_vs_baseline_x']:.2f}x baseline")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
