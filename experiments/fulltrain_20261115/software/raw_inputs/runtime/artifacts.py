"""Bind cached features and trained heads to their actual extraction contract."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import warnings


CODE = Path(__file__).resolve().parent
SCHEMA_VERSION = 2
ABLATIONS = {"sr", "ty", "cfar", "cfar_strict"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_pretrained_path(model, pretrained):
    """Resolve the file used by installed OpenCLIP without downloading anything."""
    candidate = Path(pretrained).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    from open_clip.pretrained import get_pretrained_cfg
    from huggingface_hub import hf_hub_download
    from runtime import MODEL_ROOT

    config = get_pretrained_cfg(model, pretrained)
    if not config:
        raise ValueError(f"No known pretrained configuration: {model}/{pretrained}")
    if config.get("url"):
        candidate = MODEL_ROOT / os.path.basename(config["url"])
    elif config.get("hf_hub"):
        repository, filename = os.path.split(config["hf_hub"])
        candidate = Path(hf_hub_download(repository, filename or "open_clip_pytorch_model.bin",
                                        cache_dir=str(MODEL_ROOT), local_files_only=True))
    else:
        raise ValueError("Pretrained configuration has no resolvable weight file")
    if not candidate.is_file():
        raise FileNotFoundError(f"Prepare the offline backbone weights before caching: {candidate}")
    return candidate.resolve()


def extraction_signature(model, pretrained, checkpoint=None, primitive_mode="physics",
                         max_regions=6, max_sc=8):
    import importlib.metadata

    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "pretrained": pretrained,
        "pretrained_weight_sha256": file_sha256(resolve_pretrained_path(model, pretrained)),
        "external_checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "primitive_mode": primitive_mode,
        "max_regions": max_regions,
        "max_sc": max_sc,
        "source_sha256": {
            name: file_sha256(CODE / name)
            for name in ("saqre_phys.py", "primitive_controls.py", "cache_primitives.py", "runtime.py", "artifacts.py")
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "open_clip_torch", "numpy", "Pillow", "scipy")
        },
    }


def make_cache_provenance(signature, ids, image_hashes, ids_path, transform):
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "signature_sha256": object_sha256(signature),
        "ids_sha256": object_sha256(ids),
        "ids_file_sha256": file_sha256(ids_path),
        "image_file_sha256": image_hashes,
        "eval_transform": repr(transform),
    }


def validate_cache(cache, model, pretrained, checkpoint=None, allow_legacy=False):
    provenance = cache.get("provenance")
    if not provenance or provenance.get("schema_version") != SCHEMA_VERSION:
        if allow_legacy:
            warnings.warn("Explicit legacy artifact mode: extraction provenance is not verified.")
            return None
        raise ValueError("Legacy/unversioned cache rejected; regenerate features with the repaired extractor")
    signature = provenance["signature"]
    if object_sha256(signature) != provenance.get("signature_sha256"):
        raise ValueError("Cache signature hash mismatch")
    if object_sha256(cache["ids"]) != provenance.get("ids_sha256"):
        raise ValueError("Cached image IDs do not match their recorded provenance")
    expected = extraction_signature(model, pretrained, checkpoint,
                                    signature["primitive_mode"], cache["Mc"], cache["Mf"])
    if signature != expected:
        raise ValueError("Stale/incompatible cache: encoder, extraction code, settings or package versions differ")
    if len(cache["ids"]) != len(set(cache["ids"])):
        raise ValueError("Duplicate IDs in cache")
    return signature


def validate_head(head, cache, allow_legacy=False):
    recorded = head.get("provenance")
    cached = cache.get("provenance")
    if not recorded or not cached:
        if allow_legacy:
            warnings.warn("Explicit legacy artifact mode: head/cache correspondence is not verified.")
            return
        raise ValueError("Unversioned head/cache rejected; retrain from a versioned cache")
    if recorded.get("cache_signature") != cached.get("signature"):
        raise ValueError("Head was trained with a different encoder/extractor/control cache")
    if recorded.get("head_source_sha256") != file_sha256(CODE / "saqre_phys.py"):
        raise ValueError("Head implementation changed since training; retraining is required")
    if head["D"] != cache["D"]:
        raise ValueError("Head/cache embedding dimensions differ")


def parse_ablation_flags(value, cache=None):
    flags = {item.strip() for item in value.split(",") if item.strip()}
    unknown = flags - ABLATIONS
    if unknown:
        raise ValueError(f"Unknown ablation flags: {sorted(unknown)}")
    if {"cfar", "cfar_strict"} <= flags:
        raise ValueError("Select either legacy cfar or cfar_strict, not both")
    mode = (cache or {}).get("provenance", {}).get("signature", {}).get("primitive_mode")
    if mode == "uniform_grid" and not {"sr", "ty"} <= flags:
        raise ValueError("The uniform-grid physics-free control requires both sr and ty disabled")
    return flags


def head_provenance(cache, cache_path, captions_path):
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_signature": cache.get("provenance", {}).get("signature"),
        "train_cache_sha256": file_sha256(cache_path),
        "train_ids": cache["ids"],
        "captions_sha256": file_sha256(captions_path),
        "head_source_sha256": file_sha256(CODE / "saqre_phys.py"),
        "training_source_sha256": file_sha256(CODE / "train_saqre.py"),
    }
