#!/usr/bin/env python3
"""Build two NEW train-only B32 caches, changing only fine tie selection.

Never imports the pre-repair extractor. Float64 CFAR/statistics come from the
hash-verified current extractor. Run --help first; no job is started on import.
"""
import argparse
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image

SCHEMA = "pare.fine-tie-paired-cache.v1"
BASE_SHA256 = "d04278829bad17a6de28e3ae39ad5f59a81feebb480ef9ed2190d028bfa8aa83"
LEGACY_REFERENCE = {
    "path": "diagnostics/reference/saqre_phys_before_20261007.py",
    "file_sha256": "38482c06c852f1d45b985fe6b862b0ae009ec4d0d880432e1ddca2f5b655acbc",
    "function": "scatterer_patches",
    "scope": "stable raster tie selection only; NOT legacy numerical extraction",
}
FINE = ("f_emb", "f_mu", "f_t", "f_tf", "f_n")
PRESERVED = ("scene", "nu", "c_emb", "c_mu", "c_t", "c_tf", "c_n", "L", "c_star")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def object_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_sha256(t):
    a = t.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(a.dtype), tuple(a.shape))).encode())
    h.update(a.numpy().tobytes())
    return h.hexdigest()


def normalized_transform(value):
    # Python function repr embeds a process-specific address, not a transform.
    return re.sub(r" at 0x[0-9a-fA-F]+", "", str(value))


def legacy_boxes(g, det, max_sc=8, half=12):
    """Exact pre-20261007 stable sort/NMS; np.where starts in raster order."""
    H, W = g.shape
    ys, xs = np.where(det)
    if len(xs) == 0:
        return []
    pts = list(zip(xs.tolist(), ys.tolist(), g[ys, xs].tolist()))
    pts.sort(key=lambda p: p[2], reverse=True)
    chosen = []
    for x, y, _ in pts:
        if all(abs(x-cx) > half or abs(y-cy) > half for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) >= max_sc:
            break
    return [(max(0, x-half), max(0, y-half), min(W, x+half), min(H, y+half))
            for x, y in chosen]


def validate_population(base_ids, train_ids, fit_ids, val_ids):
    for name, ids, size in (("train", train_ids, 9181), ("fit", fit_ids, 8263), ("val", val_ids, 918)):
        if len(ids) != size or len(set(ids)) != size:
            raise ValueError(f"Wrong {name} population: expected {size} distinct IDs")
        if any(Path(i).name != i for i in ids):
            raise ValueError("Expected image basenames, not relative/absolute paths")
    if base_ids != train_ids:
        raise ValueError("Parent cache must exactly match the 9181 train IDs in order")
    if set(fit_ids) & set(val_ids) or set(fit_ids) | set(val_ids) != set(train_ids):
        raise ValueError("fit/val must be a disjoint, complete partition of train")


def image_index(root, wanted):
    index = {}
    for dp, _, names in os.walk(root):
        for name in names:
            if name in wanted:
                if name in index:
                    raise ValueError(f"Duplicate requested basename: {name}")
                index[name] = Path(dp) / name
    if set(index) != set(wanted):
        raise FileNotFoundError(f"Missing requested images: {sorted(set(wanted)-set(index))[:5]}")
    return index


def fine_features(g, det, c_star, boxes, extractor):
    features = []
    for x0, y0, x1, y1 in boxes:
        sub = g[y0:y1, x0:x1]
        if sub.size == 0:
            sub = g
        mu, sig = float(sub.mean()), float(sub.std())
        cv = sig / (mu + 1e-4)
        sub_det = det[y0:y1, x0:x1]
        cf = float(sub_det.mean()) if sub_det.size else 0.0
        features.append((mu, max(cv-c_star, 0.0), extractor.type_features(sub, cf)))
    return features


@torch.no_grad()
def encode_fine(model, preprocess, device, image, boxes, slots=8):
    """Identical float32/eval fixed-8 forward policy for BOTH selections.

    Padding is RGB black 24x24; padded embeddings never enter the output cache.
    Even a zero-detection image makes the same fixed-8 call.
    """
    if len(boxes) > slots:
        raise ValueError("Too many fine crops")
    crops = [image.crop(box) for box in boxes]
    crops += [Image.new("RGB", (24, 24), 0) for _ in range(slots-len(crops))]
    batch = torch.stack([preprocess(crop) for crop in crops]).to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        result = torch.nn.functional.normalize(model.encode_image(batch).float(), dim=-1).cpu()
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite fine embeddings")
    return result[:len(boxes)]


def empty_variant(base, n):
    result = {k: v for k, v in base.items() if k not in ("config", "provenance")}
    result["ids"] = list(base["ids"][:n])
    for k, value in list(result.items()):
        if torch.is_tensor(value):
            if k in FINE or k == "agg":
                result[k] = torch.zeros_like(value[:n])
            else:
                result[k] = value[:n].clone()  # avoids serializing a full parent storage in smoke files
    return result


def assign_fine(cache, row, embeddings, features):
    n = len(features)
    if len(embeddings) != n:
        raise ValueError("Feature/embedding count mismatch")
    cache["f_n"][row] = n
    cache["f_emb"][row, :n] = embeddings
    for j, (mu, t, tf) in enumerate(features):
        cache["f_mu"][row, j] = mu
        cache["f_t"][row, j] = t
        cache["f_tf"][row, j] = torch.as_tensor(tf)
    nc = int(cache["c_n"][row])
    allv = torch.cat((cache["scene"][row:row+1], cache["c_emb"][row, :nc], embeddings), dim=0)
    cache["agg"][row] = allv.mean(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-cache", "train-ids", "fit-ids", "val-ids", "image-root", "base-code", "model-root", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", default="cpu", help="Explicitly pass cuda:N for an approved GPU build")
    p.add_argument("--max-images", type=int, default=0, help="0=all train; positive=prefix SMOKE cache, never full train")
    args = p.parse_args()
    if args.max_images < 0 or args.max_images > 9181:
        p.error("max-images must be 0..9181")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; never overwrite a prior run")
    if file_sha256(args.base_cache) != BASE_SHA256:
        raise ValueError("This audit is bound to the original 9181-image B32 train cache")
    base = torch.load(args.base_cache, map_location="cpu", weights_only=True)
    id_files = {name: getattr(args, name+"_ids") for name in ("train", "fit", "val")}
    id_file_hashes = {name: file_sha256(path) for name, path in id_files.items()}
    ids = {name: json.loads(path.read_text()) for name, path in id_files.items()}
    validate_population(base["ids"], ids["train"], ids["fit"], ids["val"])
    if (base["D"], base["Mc"], base["Mf"]) != (512, 6, 8):
        raise ValueError("Expected B32 D=512, Mc=6, Mf=8")
    parent = base["provenance"]
    if parent["signature_sha256"] != object_sha256(parent["signature"]):
        raise ValueError("Invalid parent signature")
    if parent["ids_sha256"] != object_sha256(base["ids"]):
        raise ValueError("Invalid parent ID signature")
    if parent["ids_file_sha256"] != file_sha256(args.train_ids):
        raise ValueError("Train ID file hash differs from the parent provenance")
    if set(parent["image_file_sha256"]) != set(base["ids"]):
        raise ValueError("Parent lacks complete image provenance")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", RS_RAG_MODEL_ROOT=str(args.model_root.resolve()))
    sys.path.insert(0, str(args.base_code.resolve()))
    extractor = importlib.import_module("saqre_phys")
    artifacts = importlib.import_module("artifacts")
    runtime = importlib.import_module("runtime")
    for module in (extractor, artifacts, runtime):
        if Path(module.__file__).resolve().parent != args.base_code.resolve():
            raise ValueError("Unexpected imported source directory")
    current_signature = artifacts.extraction_signature("ViT-B-32", "laion2b_s34b_b79k", None, "physics", 6, 8)
    if current_signature != parent["signature"]:
        raise ValueError("Current source, weights or packages differ from parent extraction contract")
    if extractor.EXTRACTOR_VERSION != "2026-10-07-float64-spatial-ties":
        raise ValueError("Must retain repaired float64 numerical extraction")
    mu, sig = extractor.local_mean_std(np.zeros((32, 32), np.float32), 9)
    if mu.dtype != np.float64 or sig.dtype != np.float64:
        raise ValueError("Float64 local statistics were not retained")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if str(args.device).startswith("cuda"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    model, _, preprocess = runtime.create_backbone("ViT-B-32", "laion2b_s34b_b79k", None)
    model = model.float().to(args.device).eval()
    if normalized_transform(repr(preprocess)) != normalized_transform(parent["eval_transform"]):
        raise ValueError("Backbone evaluation preprocessing changed")

    n = args.max_images or len(base["ids"])
    run_ids = base["ids"][:n]
    index = image_index(args.image_root, set(run_ids))
    outputs = {name: empty_variant(base, n) for name in ("legacy_raster", "spatial")}
    preserved_hashes = {k: tensor_sha256(base[k][:n]) for k in PRESERVED}
    records = []
    own_source_hash = file_sha256(__file__)
    for row, name in enumerate(run_ids):
        raw = index[name].read_bytes()
        image_hash = hashlib.sha256(raw).hexdigest()
        if image_hash != parent["image_file_sha256"][name]:
            raise ValueError(f"Image bytes differ from parent provenance: {name}")
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert("RGB")
        g = extractor.to_intensity(image)
        L = extractor.estimate_enl(g)
        c_star = 1.0 / np.sqrt(L)
        nu = float(np.quantile(g, 0.05))
        for key, value in (("L", L), ("c_star", c_star), ("nu", nu)):
            if torch.tensor(value, dtype=base[key].dtype).item() != base[key][row].item():
                raise ValueError(f"Unchanged numerical field {key} differs for {name}")
        det = extractor.ca_cfar(g)
        boxes = {"legacy_raster": legacy_boxes(g, det, 8, 12),
                 "spatial": extractor.scatterer_patches(g, det, max_sc=8, half=12)}
        for selection, selected in boxes.items():
            features = fine_features(g, det, c_star, selected, extractor)
            embeddings = encode_fine(model, preprocess, args.device, image, selected, 8)
            assign_fine(outputs[selection], row, embeddings, features)
        # Repaired spatial extraction must reproduce ALL old non-embedding fine fields exactly.
        for key in ("f_mu", "f_t", "f_tf", "f_n"):
            if not torch.equal(outputs["spatial"][key][row], base[key][row]):
                raise ValueError(f"Spatial extraction {key} changed for {name}")
        if boxes["legacy_raster"] == boxes["spatial"] and not torch.equal(
                outputs["legacy_raster"]["f_emb"][row], outputs["spatial"]["f_emb"][row]):
            raise ValueError(f"Identical ordered crops encoded differently: {name}")
        values = g[det]
        _, counts = np.unique(values, return_counts=True)
        records.append({"id": name, "image_sha256": image_hash,
                        "detections": int(det.sum()), "tied_detection_count": int(counts[counts>1].sum()),
                        "boxes": boxes, "same_boxes": boxes["legacy_raster"] == boxes["spatial"],
                        "spatial_reencoding_max_abs": float((outputs["spatial"]["f_emb"][row]-base["f_emb"][row]).abs().max())})
        if (row+1) % 100 == 0 or row+1 == n:
            print(f"verified and encoded {row+1}/{n}", flush=True)

    if current_signature != artifacts.extraction_signature("ViT-B-32", "laion2b_s34b_b79k", None, "physics", 6, 8):
        raise ValueError("Input source/weights/environment changed during this run")
    if file_sha256(__file__) != own_source_hash or file_sha256(args.base_cache) != BASE_SHA256:
        raise ValueError("Builder or parent cache changed during this run")
    if id_file_hashes != {name: file_sha256(path) for name, path in id_files.items()}:
        raise ValueError("Population ID files changed during this run")
    shared = {"schema_version": SCHEMA, "parent_cache_sha256": BASE_SHA256,
              "parent_extraction_signature": current_signature,
              "builder_sha256": own_source_hash, "legacy_selection_reference": LEGACY_REFERENCE,
              "population": "train_9181" if n == 9181 else f"SMOKE_prefix_{n}_of_train_9181",
              "ids_sha256": object_sha256(run_ids), "population_id_files": {
                  name: {"path": str(path.resolve()), "sha256": id_file_hashes[name]} for name, path in id_files.items()},
              "fine_crop_half_pixels": 12, "fine_slots": 8,
              "encoding_policy": "each image, each selection: one float32 eval forward of 8 crops; RGB-black 24x24 right padding; padding discarded",
              "precision": "float32, no autocast, no TF32, deterministic algorithms, CUBLAS :4096:8",
              "device": str(args.device), "device_name": torch.cuda.get_device_name(torch.device(args.device)) if str(args.device).startswith("cuda") else "cpu",
              "preprocess": normalized_transform(repr(preprocess)), "preserved_tensor_sha256": preserved_hashes,
              "changed_fields": list(FINE) + ["agg"],
              "agg_policy": "mean of unchanged scene, unchanged valid coarse, and newly encoded valid fine"}
    args.output_dir.mkdir(parents=True)
    audit = {"schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
             "contract": shared, "base_cache_path": str(args.base_cache.resolve()),
             "image_root": str(args.image_root.resolve()), "base_code": str(args.base_code.resolve()),
             "model_weight_path": str(artifacts.resolve_pretrained_path("ViT-B-32", "laion2b_s34b_b79k")),
             "images_with_different_boxes": sum(not r["same_boxes"] for r in records), "outputs": {}, "images": records}
    audit["max_spatial_reencoding_abs"] = max(r["spatial_reencoding_max_abs"] for r in records)
    for selection, cache in outputs.items():
        if any(tensor_sha256(cache[k]) != preserved_hashes[k] for k in PRESERVED):
            raise ValueError("A preserved field was modified")
        signature = dict(shared, fine_tie_selection=selection)
        cache["config"] = {"schema_version": SCHEMA, "fine_tie_selection": selection,
                           "parent_config": copy.deepcopy(base["config"]), "build_args": {k: str(v) for k, v in vars(args).items()}}
        cache["provenance"] = {"schema_version": SCHEMA, "created_utc": audit["created_utc"],
                               "signature": signature, "signature_sha256": object_sha256(signature),
                               "ids_sha256": object_sha256(run_ids),
                               "image_file_sha256": {r["id"]: r["image_sha256"] for r in records},
                               "eval_transform": repr(preprocess)}
        filename = f"cache_fine_{selection}_{shared['population']}.pt"
        path = args.output_dir / filename
        torch.save(cache, str(path)+".partial")
        os.replace(str(path)+".partial", path)
        audit["outputs"][selection] = {"path": filename, "sha256": file_sha256(path),
            "signature_sha256": cache["provenance"]["signature_sha256"],
            "tensor_sha256": {k: tensor_sha256(v) for k, v in cache.items() if torch.is_tensor(v)}}
    manifest = args.output_dir / "PAIRED_CACHE_AUDIT.json"
    manifest.write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"manifest": str(manifest), "sha256": file_sha256(manifest),
                      "different_boxes": audit["images_with_different_boxes"], "count": n}), flush=True)


if __name__ == "__main__":
    main()
