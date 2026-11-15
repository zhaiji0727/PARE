#!/usr/bin/env python3
"""Pre-compute SA-QRE primitives (multi-granular CLIP embeddings + SAR
physical features) for a list of images and cache to a .pt file.

Backbone is FROZEN. We cache, per image:
  scene emb, region embs (regime-segmented), scatterer embs (CFAR), and the
  physical features (mu, excess-structure t, type-features) needed by
  SAQREHead. Training then operates on the cache (fast, backbone-free).
"""
import argparse, json, os, time
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from primitive_controls import extract_for_mode
from artifacts import extraction_signature, file_sha256, make_cache_provenance
from runtime import DATA_ROOT, create_backbone, get_device


def build_image_index(img_root):
    idx = {}
    for dp, _, fns in os.walk(img_root):
        for fn in fns:
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                continue
            if fn in idx:
                raise ValueError(f"Duplicate image basename: {fn}")
            idx[fn] = os.path.join(dp, fn)
    return idx


@torch.no_grad()
def encode(model, preprocess, device, pil_list, bs=128):
    out = []
    for i in range(0, len(pil_list), bs):
        batch = torch.stack([preprocess(im) for im in pil_list[i:i+bs]]).to(device)
        f = model.encode_image(batch)
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(out, 0) if out else torch.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="json list of image basenames")
    ap.add_argument("--img_root", default=str(DATA_ROOT / "SARimages_pre"))
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--remoteclip", default=None)
    ap.add_argument("--max_regions", type=int, default=6)
    ap.add_argument("--max_sc", type=int, default=8)
    ap.add_argument("--primitive_mode", choices=("physics", "uniform_grid"), default="physics")
    ap.add_argument("--max_images", type=int, default=0, help="smoke subset; 0 keeps all requested IDs")
    ap.add_argument("--encode_bs", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError(f"Refusing to overwrite an existing cache: {args.out}")

    if args.max_images < 0 or min(args.encode_bs, args.max_regions, args.max_sc) < 1:
        ap.error("max_images must be nonnegative and batch/primitive sizes positive")
    ids = json.load(open(args.ids))
    if args.max_images:
        ids = ids[:args.max_images]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected a nonempty list of unique image IDs")
    img_index = build_image_index(args.img_root)
    missing = [name for name in ids if name not in img_index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} requested images missing under {args.img_root}: {missing[:5]}")
    device = get_device()
    signature = extraction_signature(args.model, args.pretrained, args.remoteclip,
                                     args.primitive_mode, args.max_regions, args.max_sc)
    model, _, preprocess = create_backbone(args.model, args.pretrained, args.remoteclip)
    model = model.to(device).eval()
    D = model.visual.output_dim if hasattr(model.visual, "output_dim") else \
        model.encode_image(torch.zeros(1, 3, 224, 224).to(device)).shape[-1]

    print(f"caching {len(ids)} images, D={D}, model={args.model}")

    Mc, Mf = args.max_regions, args.max_sc
    N = len(ids)
    scene = torch.zeros(N, D); agg = torch.zeros(N, D); nu = torch.zeros(N)
    c_emb = torch.zeros(N, Mc, D); c_mu = torch.zeros(N, Mc); c_t = torch.zeros(N, Mc)
    c_tf = torch.zeros(N, Mc, 4); c_n = torch.zeros(N, dtype=torch.long)
    f_emb = torch.zeros(N, Mf, D); f_mu = torch.zeros(N, Mf); f_t = torch.zeros(N, Mf)
    f_tf = torch.zeros(N, Mf, 4); f_n = torch.zeros(N, dtype=torch.long)
    Lv = torch.zeros(N); cstar = torch.zeros(N)

    t0 = time.time()
    image_hashes = {}
    for k, b in enumerate(tqdm(ids, desc="cache")):
        try:
            im = Image.open(img_index[b]).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load image {b!r} at {img_index[b]!r}") from exc
        image_hashes[b] = file_sha256(img_index[b])
        prim = extract_for_mode(im, max_regions=Mc, max_sc=Mf, mode=args.primitive_mode)
        crops = [prim["scene"][0]] + [c for c, _ in prim["regions"]] + [c for c, _ in prim["scatterers"]]
        embs = encode(model, preprocess, device, crops, bs=args.encode_bs)
        nr = len(prim["regions"]); ns = len(prim["scatterers"])
        scene[k] = embs[0]
        ridx = 1
        for i, (_, fe) in enumerate(prim["regions"][:Mc]):
            c_emb[k, i] = embs[ridx + i]
            c_mu[k, i] = fe["mu"]; c_t[k, i] = fe["t"]; c_tf[k, i] = torch.tensor(fe["typefeat"])
        c_n[k] = min(nr, Mc)
        sidx = 1 + nr
        for i, (_, fe) in enumerate(prim["scatterers"][:Mf]):
            f_emb[k, i] = embs[sidx + i]
            f_mu[k, i] = fe["mu"]; f_t[k, i] = fe["t"]; f_tf[k, i] = torch.tensor(fe["typefeat"])
        f_n[k] = min(ns, Mf)
        # aggregate = mean of all valid primitive embeddings
        allv = [embs[0]]
        allv += [embs[ridx + i] for i in range(min(nr, Mc))]
        allv += [embs[sidx + i] for i in range(min(ns, Mf))]
        agg[k] = torch.stack(allv).mean(0)
        nu[k] = prim["nu"]; Lv[k] = prim["L"]; cstar[k] = prim["c_star"]

    cache = {"ids": ids, "D": D, "Mc": Mc, "Mf": Mf,
             "scene": scene, "agg": agg, "nu": nu,
             "c_emb": c_emb, "c_mu": c_mu, "c_t": c_t, "c_tf": c_tf, "c_n": c_n,
             "f_emb": f_emb, "f_mu": f_mu, "f_t": f_t, "f_tf": f_tf, "f_n": f_n,
             "L": Lv, "c_star": cstar,
             "config": vars(args),
             "provenance": make_cache_provenance(signature, ids, image_hashes, args.ids, preprocess)}
    if signature != extraction_signature(args.model, args.pretrained, args.remoteclip,
                                         args.primitive_mode, Mc, Mf):
        raise RuntimeError("Extraction code or environment changed during cache generation")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(cache, args.out)
    print(f"saved {args.out} in {time.time()-t0:.1f}s ; mean ENL={Lv.mean():.2f}")


if __name__ == "__main__":
    main()
