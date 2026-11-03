#!/usr/bin/env python3
"""Build a separately signed scale-test cache from an unchanged formal snapshot.

This script never trains a head or rewrites the formal code/reference cache.
Generic release evaluators intentionally reject its extended signature.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from scale_contract import (bind_signature, factor_string, file_hash, object_hash,
                            transform_image, verify_and_unwrap_signature,
                            verify_image_records)


HERE = Path(__file__).resolve().parent


def helper_hashes():
    return {name: file_hash(HERE / name) for name in ("scale_contract.py", "stress_cache.py")}


def load_frozen_code(code_root):
    code_root = Path(code_root).resolve()
    required = ("saqre_phys.py", "artifacts.py", "cache_primitives.py", "runtime.py",
                "primitive_controls.py", "train_saqre.py", "retrieval_queries.py")
    if not all((code_root / name).is_file() for name in required):
        raise FileNotFoundError(f"Not a compatible frozen code directory: {code_root}")
    sys.path.insert(0, str(code_root))
    return code_root


def verify_frozen_imports(code_root):
    code_root = Path(code_root).resolve()
    for name in ("saqre_phys", "artifacts", "cache_primitives", "runtime",
                 "primitive_controls", "train_saqre", "retrieval_queries"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != code_root:
            raise RuntimeError(f"Module {name} came from outside the requested frozen code root")


def set_device_guard(requested):
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PARE_DEVICE"] = "cpu"
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
            raise ValueError("GPU jobs require explicit CUDA_VISIBLE_DEVICES=5; GPUs 0--4 are reserved")
        os.environ["PARE_DEVICE"] = "cuda:0"


def validate_scale_cache(cache, reference, reference_hash, factor, model, pretrained,
                         remoteclip=None):
    """Verify the permitted domain perturbation, then the unchanged base contract.

    The returned shallow view is ONLY for the original head validator. It must
    not be saved or passed off as a normal untransformed cache.
    """
    from artifacts import validate_cache

    base_reference = validate_cache(reference, model, pretrained, remoteclip)
    provenance = cache["provenance"]
    signature = provenance["signature"]
    if object_hash(signature) != provenance["signature_sha256"]:
        raise ValueError("Corrupted extended extraction signature")
    base = verify_and_unwrap_signature(signature, factor, helper_hashes())
    if base != base_reference:
        raise ValueError("Scale test changed more than the documented input transformation")
    if cache["ids"] != reference["ids"]:
        raise ValueError("Scale-test gallery or ordering differs from formal reference")
    if (cache["D"], cache["Mc"], cache["Mf"]) != (reference["D"], reference["Mc"], reference["Mf"]):
        raise ValueError("Scale-test embedding dimension or crop limits changed")
    details = cache["scale_test"]
    if details["reference_cache_sha256"] != reference_hash:
        raise ValueError("Wrong formal reference cache")
    if details["factor"] != factor_string(factor):
        raise ValueError("Wrong declared spacing condition")
    if details["image_records_sha256"] != object_hash(details["image_records"]):
        raise ValueError("Image transformation manifest checksum mismatch")
    hashes = reference["provenance"]["image_file_sha256"]
    if provenance["image_file_sha256"] != hashes:
        raise ValueError("Source images differ from formal evaluation")
    verify_image_records(cache["ids"], details["image_records"], factor, hashes)
    view = dict(cache)
    view["provenance"] = {**deepcopy(provenance), "signature": base,
                          "signature_sha256": object_hash(base)}
    validate_cache(view, model, pretrained, remoteclip)
    return view


def compare_identity_cache(cache, reference, atol=2e-5):
    """Check regenerated original-scale features against formal reference."""
    import torch

    if cache["scale_test"]["factor"] != "1":
        raise ValueError("Identity comparison requested for a non-identity condition")
    report = {}
    for key, actual in cache.items():
        if not torch.is_tensor(actual):
            continue
        expected = reference[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Identity tensor shape/dtype differs: {key}")
        if not actual.is_floating_point():
            if not torch.equal(actual, expected):
                raise ValueError(f"Identity tensor differs: {key}")
            report[key] = {"exact_equal": True}
            continue
        difference = (actual - expected).abs().max().item() if actual.numel() else 0.0
        # Feature extraction must be exact; only backbone-dependent embeddings
        # may exhibit small floating-point differences with GPU batch size.
        permitted = atol if key in ("scene", "agg", "c_emb", "f_emb") else 0.0
        if difference > permitted:
            raise ValueError(f"Identity tensor {key}: max error {difference} exceeds {permitted}")
        report[key] = {"max_abs_difference": difference, "allowed_abs_difference": permitted}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--factor", required=True, choices=("1", "1.5", "2"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--remoteclip")
    parser.add_argument("--expected-gallery", type=int, default=3688)
    parser.add_argument("--encode-bs", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.expected_gallery < 1 or args.encode_bs < 1:
        parser.error("Gallery count and encoder batch size must be positive")
    output = Path(args.out).resolve()
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite a scale-test cache or manifest")
    set_device_guard(args.device)
    load_frozen_code(args.code_root)
    import torch
    from PIL import Image
    from artifacts import extraction_signature, make_cache_provenance, validate_cache
    from cache_primitives import build_image_index, encode
    from primitive_controls import extract_for_mode
    from runtime import create_backbone, get_device
    verify_frozen_imports(args.code_root)

    ids_file_hash = file_hash(args.ids)
    ids = json.loads(Path(args.ids).read_text())
    if len(ids) != args.expected_gallery or len(set(ids)) != len(ids):
        raise ValueError("Unexpected or duplicate test-gallery IDs")
    reference_hash = file_hash(args.reference_cache)
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=True)
    base = validate_cache(reference, args.model, args.pretrained, args.remoteclip)
    if ids != reference["ids"]:
        raise ValueError("IDs/order do not match the formal reference cache")
    signature = bind_signature(base, args.factor, helper_hashes())
    original_hashes = reference["provenance"]["image_file_sha256"]
    if set(ids) != set(original_hashes):
        raise ValueError("Formal reference lacks exact source-file coverage")
    index = build_image_index(args.img_root)
    if any(image_id not in index for image_id in ids):
        raise FileNotFoundError("At least one required gallery image is missing")
    device = get_device()
    model, _, preprocess = create_backbone(args.model, args.pretrained, args.remoteclip)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dimension, mc, mf = reference["D"], reference["Mc"], reference["Mf"]
    tensor_keys = {key for key, value in reference.items() if torch.is_tensor(value)}
    expected_tensor_keys = {"scene", "agg", "nu", "c_emb", "c_mu", "c_t", "c_tf", "c_n",
                            "f_emb", "f_mu", "f_t", "f_tf", "f_n", "L", "c_star"}
    if tensor_keys != expected_tensor_keys:
        raise ValueError("Formal cache tensor interface changed; inspect before running scale extraction")
    # Keep the original cache's tensor layout; no scientific head code is copied.
    cache = {key: torch.zeros_like(value) for key, value in reference.items()
             if torch.is_tensor(value)}
    records = {}
    started = time.monotonic()
    with torch.inference_mode():
        for row, image_id in enumerate(ids):
            path = index[image_id]
            if file_hash(path) != original_hashes[image_id]:
                raise ValueError(f"Source image changed since formal cache: {image_id}")
            with Image.open(path) as image:
                perturbed, record = transform_image(image, args.factor)
            record["source_file_sha256"] = original_hashes[image_id]
            records[image_id] = record
            primitives = extract_for_mode(perturbed, max_regions=mc, max_sc=mf,
                                          mode=base["primitive_mode"])
            crops = ([primitives["scene"][0]] + [x for x, _ in primitives["regions"]]
                     + [x for x, _ in primitives["scatterers"]])
            embeddings = encode(model, preprocess, device, crops, bs=args.encode_bs)
            if embeddings.shape != (len(crops), dimension):
                raise ValueError("Unexpected backbone output dimension")
            cache["scene"][row] = embeddings[0]
            region_count, fine_count = len(primitives["regions"]), len(primitives["scatterers"])
            for prefix, values, offset, limit in (("c", primitives["regions"], 1, mc),
                                                  ("f", primitives["scatterers"], 1 + region_count, mf)):
                for position, (_, feature) in enumerate(values[:limit]):
                    cache[f"{prefix}_emb"][row, position] = embeddings[offset + position]
                    cache[f"{prefix}_mu"][row, position] = feature["mu"]
                    cache[f"{prefix}_t"][row, position] = feature["t"]
                    cache[f"{prefix}_tf"][row, position] = torch.tensor(feature["typefeat"])
                cache[f"{prefix}_n"][row] = min(len(values), limit)
            valid = ([embeddings[0]] + list(embeddings[1:1 + min(region_count, mc)])
                     + list(embeddings[1 + region_count:1 + region_count + min(fine_count, mf)]))
            cache["agg"][row] = torch.stack(valid).mean(0)
            for key, primitive_key in (("nu", "nu"), ("L", "L"), ("c_star", "c_star")):
                cache[key][row] = primitives[primitive_key]
            if row % 100 == 0 or row + 1 == len(ids):
                print(f"factor={args.factor} {row + 1}/{len(ids)} elapsed={time.monotonic() - started:.1f}s", flush=True)
    cache.update({"ids": ids, "D": dimension, "Mc": mc, "Mf": mf, "config": vars(args),
                  "provenance": make_cache_provenance(signature, ids, original_hashes, args.ids, preprocess),
                  "scale_test": {"factor": args.factor, "reference_cache_sha256": reference_hash,
                                 "image_records": records, "image_records_sha256": object_hash(records)}})
    if any(not torch.isfinite(value).all() for value in cache.values() if torch.is_tensor(value)):
        raise FloatingPointError("Nonfinite scale-cache values")
    current = extraction_signature(args.model, args.pretrained, args.remoteclip,
                                   base["primitive_mode"], mc, mf)
    if current != base or bind_signature(current, args.factor, helper_hashes()) != signature:
        raise ValueError("Frozen code, environment or scale helper changed during extraction")
    if file_hash(args.reference_cache) != reference_hash:
        raise ValueError("Formal reference changed during extraction")
    if file_hash(args.ids) != ids_file_hash:
        raise ValueError("Test ID file changed during extraction")
    validate_scale_cache(cache, reference, reference_hash, args.factor, args.model,
                         args.pretrained, args.remoteclip)
    identity_check = compare_identity_cache(cache, reference) if args.factor == "1" else None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Previous temporary file exists: {temporary}")
    torch.save(cache, temporary)
    temporary.rename(output)
    summary = {"factor": args.factor, "cache_sha256": file_hash(output),
               "signature": signature, "signature_sha256": object_hash(signature),
               "reference_cache_sha256": reference_hash, "n_gallery": len(ids),
               "identity_comparison": identity_check,
               "image_records_sha256": object_hash(records), "image_records": records,
               "elapsed_seconds": time.monotonic() - started,
               "is_real_gsd_validation": False, "gpu_physical_index": 5 if args.device == "cuda" else None}
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved independent factor={args.factor} cache: {output}", flush=True)


if __name__ == "__main__":
    main()
