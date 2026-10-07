#!/usr/bin/env python3
"""Train the SA-QRE head with the contrastive retrieval objective (Eq. contrastive).

Caption -> image retrieval on SARLANG-1M. The CLIP backbone is FROZEN; we
train only SAQREHead (type head, prototypes, reliability scale beta,
attention temperature tau, fusion gate). Operates on cached primitives
(cache_primitives.py) for speed.
"""
import argparse, csv, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from saqre_phys import SAQREHead
from runtime import create_backbone, get_device
from artifacts import head_provenance, parse_ablation_flags, validate_cache


def load_captions(csv_path):
    """Return dict image_name -> list[caption]."""
    cap = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        keys = r.fieldnames
        kimg = keys[0]
        kcap = [k for k in keys if "caption" in k.lower() and "type" not in k.lower()][0]
        for row in r:
            img = row[kimg].strip()
            c = row[kcap].strip()
            if c:
                cap.setdefault(img, []).append(c)
    return cap


def move_docs(cache, idxs, device):
    sel = lambda key: cache[key][idxs].to(device)
    Mc, Mf = cache["Mc"], cache["Mf"]
    c_n = cache["c_n"][idxs]; f_n = cache["f_n"][idxs]
    c_mask = (torch.arange(Mc).unsqueeze(0) < c_n.unsqueeze(1)).to(device)
    f_mask = (torch.arange(Mf).unsqueeze(0) < f_n.unsqueeze(1)).to(device)
    return {"g": sel("scene"), "agg": sel("agg"), "nu": sel("nu"),
            "c_emb": sel("c_emb"), "c_mu": sel("c_mu"), "c_t": sel("c_t"),
            "c_tf": sel("c_tf"), "c_mask": c_mask,
            "f_emb": sel("f_emb"), "f_mu": sel("f_mu"), "f_t": sel("f_t"),
            "f_tf": sel("f_tf"), "f_mask": f_mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="train primitives cache .pt")
    ap.add_argument("--captions", required=True, help="Caption_train.csv")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--remoteclip", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--text_bs", type=int, default=256)
    ap.add_argument("--max_pairs", type=int, default=0, help="smoke subset; 0 keeps all caption pairs")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kappa", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="", help="comma-separated: sr,ty,cfar (legacy),cfar_strict")
    ap.add_argument("--allow_legacy_artifacts", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {args.out}")
    if min(args.bs, args.text_bs, args.epochs) < 1 or args.max_pairs < 0:
        ap.error("batch sizes/epochs must be positive and max_pairs nonnegative")

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
    provenance = head_provenance(cache, args.cache, args.captions)
    ablation_flags = parse_ablation_flags(args.ablate, cache)
    ids = cache["ids"]; D = cache["D"]
    id2row = {b: i for i, b in enumerate(ids)}

    # text encoder (frozen) to embed captions
    model, _, _ = create_backbone(args.model, args.pretrained, args.remoteclip)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    caps = load_captions(args.captions)
    # build training pairs: (row, caption) for images present in cache
    pairs = []
    for b, cl in caps.items():
        if b in id2row:
            for c in cl:
                pairs.append((id2row[b], c))
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if len(pairs) < 2:
        raise ValueError("At least two matching caption-image pairs are required")
    print(f"train pairs: {len(pairs)} over {len(ids)} cached images, D={D}")

    # pre-embed all unique captions (frozen) for speed
    uniq_caps = sorted({c for _, c in pairs})
    cap2vec = {}

    @torch.no_grad()
    def embed_texts(texts):
        out = []
        for i in range(0, len(texts), args.text_bs):
            toks = tokenizer(texts[i:i+args.text_bs]).to(device)
            f = model.encode_text(toks)
            out.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(out, 0)

    cv = embed_texts(uniq_caps)
    for c, v in zip(uniq_caps, cv):
        cap2vec[c] = v
    print("captions embedded")

    del model
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    pair_rng = random.Random(args.seed)
    head = SAQREHead(D).to(device)
    head.ablate = ablation_flags
    if head.ablate:
        print("ablation:", sorted(head.ablate))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    t0 = time.time()
    history = []
    for ep in range(args.epochs):
        pair_rng.shuffle(pairs)
        head.train()
        tot, nb = 0.0, 0
        for i in range(0, len(pairs), args.bs):
            batch = pairs[i:i+args.bs]
            if not batch:
                continue
            rows = torch.tensor([r for r, _ in batch])
            qtext = [c for _, c in batch]
            q = torch.stack([cap2vec[c] for c in qtext]).to(device)  # (B,D)
            docs = move_docs(cache, rows, device)
            B = len(batch)
            # score matrix S[i,j] = q_i . fuse(q_i, doc_j)
            S = torch.zeros(B, B, device=device)
            for qi in range(B):
                fused = head.fuse_batch(q[qi], docs)        # (B,D)
                S[qi] = fused @ q[qi]
            target = torch.arange(B, device=device)
            loss = F.cross_entropy(S / args.kappa, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no checkpoint will be saved")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
              f"beta={torch.exp(head.log_beta).item():.3f} "
              f"tau={torch.exp(head.log_tau).item():.4f} t={time.time()-t0:.0f}s")
        history.append({"epoch": ep + 1, "mean_batch_loss": tot / max(nb, 1),
                        "batches": nb, "pairs": len(pairs), "elapsed_seconds": time.time() - t0})

    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise FloatingPointError("Non-finite head weights; no checkpoint will be saved")
    current = head_provenance(cache, args.cache, args.captions)
    if any(provenance[key] != current[key] for key in provenance if key != "created_utc"):
        raise RuntimeError("Training inputs or code changed during this run")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "D": D, "config": vars(args),
                "ablate": sorted(head.ablate), "provenance": provenance,
                "history": history, "n_pairs": len(pairs), "n_images": len(ids)}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
