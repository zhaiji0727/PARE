#!/usr/bin/env python3
"""Hyper-parameter search for the SA-QRE head (method-fix reliability).

Protocol (no test leakage):
  * Hold out 10%% of the TRAIN images as a validation gallery; the head is
    NEVER trained on those images.
  * Grid-search (lr, kappa) by training a fresh head per config and measuring
    caption->image R@10 / MRR@10 on the held-out validation split.
  * Every trial is appended to results/hparam_sweep.csv (exploration record).
  * The best (lr, kappa) by val R@10 (tie-break MRR@10) is printed; the FINAL
    models are retrained on the FULL train split at that config and reported
    on the TEST split by run_abl (separate step).

Loads cache + backbone + captions ONCE and reuses them across all trials.
"""
import argparse, csv, os, random, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from saqre_phys import SAQREHead
from train_saqre import load_captions, move_docs
from runtime import RESULTS_ROOT, create_backbone, get_device
from artifacts import file_sha256, head_provenance, validate_cache


def metrics(ranks):
    r = np.asarray(ranks, float)
    return (float((r <= 1).mean()*100), float((r <= 5).mean()*100),
            float((r <= 10).mean()*100),
            float(np.mean([1.0/x if x <= 10 else 0.0 for x in r])*100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="TRAIN primitives cache")
    ap.add_argument("--captions", required=True, help="Caption_train.csv")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--remoteclip", default=None)
    ap.add_argument("--epochs", type=int, default=10, help="epochs per trial")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--val_q", type=int, default=0, help="0 uses every validation image's first caption")
    ap.add_argument("--fit_ids", help="versioned fit IDs; must be supplied with val_ids")
    ap.add_argument("--val_ids", help="versioned validation IDs")
    ap.add_argument("--allow_legacy_artifacts", action="store_true")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lrs", default="5e-4,1e-3,2e-3")
    ap.add_argument("--kappas", default="0.03,0.05,0.07")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=str(RESULTS_ROOT / "hparam_sweep.csv"))
    ap.add_argument("--out", default=str(RESULTS_ROOT / "hparam_best.json"))
    args = ap.parse_args()
    if not 0 < args.val_frac < 1 or min(args.bs, args.epochs, args.chunk) < 1 or args.val_q < 0:
        ap.error("val_frac must be in (0, 1); batch/epoch/chunk must be positive and val_q nonnegative")
    if bool(args.fit_ids) != bool(args.val_ids):
        ap.error("Supply both --fit_ids and --val_ids")
    if os.path.exists(args.csv) or os.path.exists(args.out):
        raise FileExistsError("Use a new output directory for each independently traceable sweep")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = get_device()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    validate_cache(cache, args.model, args.pretrained, args.remoteclip, args.allow_legacy_artifacts)
    if cache.get("provenance", {}).get("signature", {}).get("primitive_mode") == "uniform_grid":
        raise ValueError("This selection sweep is for the full physical head, not the uniform-grid control")
    ids = cache["ids"]; D = cache["D"]; N = len(ids)
    id2row = {b: i for i, b in enumerate(ids)}

    # ---- backbone (frozen) ----
    model, _, _ = create_backbone(args.model, args.pretrained, args.remoteclip)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- train / val split BY IMAGE (val gallery never trained on) ----
    if args.fit_ids:
        fit_ids = json.loads(Path(args.fit_ids).read_text())
        val_ids = json.loads(Path(args.val_ids).read_text())
        if len(fit_ids) != len(set(fit_ids)) or len(val_ids) != len(set(val_ids)):
            raise ValueError("Duplicate IDs within the provided split")
        if set(fit_ids) & set(val_ids) or set(fit_ids) | set(val_ids) != set(ids):
            raise ValueError("fit/validation IDs must disjointly partition this cache")
        train_rows = {id2row[name] for name in fit_ids}
        val_rows = {id2row[name] for name in val_ids}
    else:
        rng = random.Random(args.seed)
        all_rows = list(range(N))
        rng.shuffle(all_rows)
        n_val = int(N * args.val_frac)
        val_rows = set(all_rows[:n_val])
        train_rows = set(all_rows[n_val:])
    print(f"split: {len(train_rows)} train imgs / {len(val_rows)} val imgs", flush=True)

    caps = load_captions(args.captions)
    pairs = []                      # (row, caption) only for train images
    val_queries = []                # (row, caption) for val images
    for b, cl in caps.items():
        if b not in id2row:
            continue
        r = id2row[b]
        if r in train_rows:
            for c in cl:
                pairs.append((r, c))
        elif r in val_rows:
            val_queries.append((r, cl[0]))
    if args.val_q:
        val_queries = val_queries[: args.val_q]
    if not pairs or not val_queries:
        raise ValueError("The split must contain captioned training and validation images")
    print(f"train pairs: {len(pairs)} | val queries: {len(val_queries)}", flush=True)

    # ---- pre-embed all texts ONCE ----
    uniq = sorted({c for _, c in pairs} | {c for _, c in val_queries})

    @torch.no_grad()
    def embed_texts(texts):
        out = []
        for i in range(0, len(texts), 256):
            toks = tokenizer(texts[i:i+256]).to(device)
            out.append(F.normalize(model.encode_text(toks), dim=-1).cpu())
        return torch.cat(out, 0)

    cap2vec = {c: v for c, v in zip(uniq, embed_texts(uniq))}
    del model
    print("texts embedded", flush=True)

    # validation gallery rows (sorted list) and query tensors
    val_gallery = sorted(val_rows)
    vg_index = {r: i for i, r in enumerate(val_gallery)}
    Nv = len(val_gallery)
    vq_emb = torch.stack([cap2vec[c] for _, c in val_queries]).to(device)
    vq_gold = [vg_index[r] for r, _ in val_queries]
    provenance = head_provenance(cache, args.cache, args.captions)
    provenance["train_ids"] = [ids[row] for row in sorted(train_rows)]
    provenance["validation_ids"] = [ids[row] for row in val_gallery]
    provenance["selection_source_sha256"] = file_sha256(__file__)
    immutable_inputs = {path: file_sha256(path) for path in
                        (args.cache, args.captions, __file__, str(Path(__file__).with_name("saqre_phys.py")))}
    for path in (args.fit_ids, args.val_ids):
        if path:
            immutable_inputs[path] = file_sha256(path)
    trial_dir = Path(args.out).with_suffix("").with_name(Path(args.out).stem + "_trials")
    trial_dir.mkdir(parents=True, exist_ok=False)

    def train_one(lr, kappa):
        # Every grid point receives the same initialization and batch order so
        # the comparison isolates (lr, kappa), rather than random-state drift.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        trial_rng = random.Random(args.seed)
        head = SAQREHead(D).to(device)
        head.ablate = set()
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        plist = pairs[:]
        t0 = time.time()
        for ep in range(args.epochs):
            trial_rng.shuffle(plist)
            head.train()
            for i in range(0, len(plist), args.bs):
                batch = plist[i:i+args.bs]
                if not batch:
                    continue
                rows = torch.tensor([r for r, _ in batch])
                q = torch.stack([cap2vec[c] for _, c in batch]).to(device)
                docs = move_docs(cache, rows, device)
                B = len(batch)
                S = torch.zeros(B, B, device=device)
                for qi in range(B):
                    S[qi] = head.fuse_batch(q[qi], docs) @ q[qi]
                loss = F.cross_entropy(S / kappa, torch.arange(B, device=device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite sweep loss")
                opt.zero_grad(); loss.backward(); opt.step()
        # ---- validate on held-out gallery ----
        if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
            raise FloatingPointError("Non-finite sweep weights; no trial checkpoint will be saved")
        head.eval()
        ranks = []
        with torch.no_grad():
            for i in range(len(val_queries)):
                qv = vq_emb[i]
                scores = torch.empty(Nv, device=device)
                for s in range(0, Nv, args.chunk):
                    e = min(s + args.chunk, Nv)
                    docs = move_docs(cache, torch.tensor(val_gallery[s:e]), device)
                    scores[s:e] = head.fuse_batch(qv, docs) @ qv
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Non-finite validation scores; ranking is invalid")
                r = (torch.argsort(scores, descending=True) == vq_gold[i]).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(r)
        trial_name = f"lr_{lr:g}_kappa_{kappa:g}"
        if any(file_sha256(path) != digest for path, digest in immutable_inputs.items()):
            raise RuntimeError("Sweep inputs or code changed during this run")
        checkpoint_path = trial_dir / (trial_name + ".pt")
        torch.save({"state_dict": head.state_dict(), "D": D, "ablate": [],
                    "config": {**vars(args), "lr": lr, "kappa": kappa},
                    "provenance": provenance}, checkpoint_path)
        raw = {"query_ids": [ids[row] for row, _ in val_queries],
               "gallery_ids": [ids[row] for row in val_gallery], "ranks": ranks,
               "checkpoint_sha256": file_sha256(checkpoint_path)}
        (trial_dir / (trial_name + ".json")).write_text(json.dumps(raw, indent=2) + "\n")
        return metrics(ranks), time.time() - t0

    lrs = [float(x) for x in args.lrs.split(",")]
    kappas = [float(x) for x in args.kappas.split(",")]

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.csv)
    fcsv = open(args.csv, "a", newline="")
    w = csv.writer(fcsv)
    if new:
        w.writerow(["model", "epochs", "bs", "lr", "kappa",
                    "val_R@1", "val_R@5", "val_R@10", "val_MRR@10",
                    "n_val_q", "n_val_gallery", "train_s"])

    trials = []
    for lr in lrs:
        for kappa in kappas:
            (r1, r5, r10, mrr), dt = train_one(lr, kappa)
            row = [args.model, args.epochs, args.bs, lr, kappa,
                   round(r1, 3), round(r5, 3), round(r10, 3), round(mrr, 3),
                   len(val_queries), Nv, round(dt, 1)]
            w.writerow(row); fcsv.flush()
            print(f"lr={lr:g} kappa={kappa:g} -> valR@10={r10:.2f} MRR={mrr:.2f} ({dt:.0f}s)", flush=True)
            trials.append({"lr": lr, "kappa": kappa, "val_R@1": r1, "val_R@5": r5,
                           "val_R@10": r10, "val_MRR@10": mrr})
    fcsv.close()

    best = max(trials, key=lambda d: (d["val_R@10"], d["val_MRR@10"]))
    json.dump({"best": best, "grid": trials, "epochs": args.epochs,
               "n_val_q": len(val_queries), "n_val_gallery": Nv,
               "config": vars(args), "provenance": provenance,
               "trial_directory": str(trial_dir.resolve())},
              open(args.out, "w"), indent=2)
    print("BEST:", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
