#!/usr/bin/env python3
"""Download and fingerprint the five paper backbones and tokenizer assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models/huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import open_clip
from huggingface_hub import hf_hub_download, snapshot_download
from open_clip.pretrained import download_pretrained


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def prepare(tag, model, pretrained):
    config = open_clip.get_pretrained_cfg(model, pretrained)
    print(f"Downloading {tag}: {model} / {pretrained}", flush=True)
    path = download_pretrained(config, cache_dir=str(ROOT / "models"))
    result = {"tag": tag, "model": model, "pretrained": pretrained,
              "source": config.get("hf_hub") or config.get("url"),
              **fingerprint(path)}
    print(f"Ready: {tag}", flush=True)
    return result


def external(tag, repository, filename, expected_sha256):
    directory = ROOT / "models/external_checkpoints" / tag
    print(f"Downloading {tag}: {repository}/{filename}", flush=True)
    path = hf_hub_download(repository, filename, local_dir=str(directory))
    result = {"tag": tag, "source": repository, "filename": filename,
              **fingerprint(path)}
    if result["sha256"] != expected_sha256:
        raise RuntimeError(f"{tag}: checkpoint differs from the recorded paper SHA-256")
    print(f"Ready: {tag}, paper SHA-256 matched", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    assets = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, *item) for item in [
            ("vitb32", "ViT-B-32", "laion2b_s34b_b79k"),
            ("vitb16", "ViT-B-16", "laion2b_s34b_b88k"),
            ("siglip", "ViT-B-16-SigLIP", "webli"),
            ("georsclip_base", "ViT-B-32", "openai"),
        ]]
        futures.extend([
            pool.submit(external, "remoteclip", "chendelong/RemoteCLIP",
                        "RemoteCLIP-ViT-B-32.pt",
                        "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4"),
            pool.submit(external, "georsclip", "Zilun/GeoRSCLIP",
                        "ckpt/RS5M_ViT-B-32.pt",
                        "129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01"),
        ])
        for future in as_completed(futures):
            assets.append(future.result())
    tokenizer_path = snapshot_download(
        "timm/ViT-B-16-SigLIP", allow_patterns=["*.json", "*.model", "*.txt"])
    report = {"status": "downloaded_and_hashed", "assets": sorted(assets, key=lambda x: x["tag"]),
              "siglip_tokenizer": str(Path(tokenizer_path).relative_to(ROOT)),
              "note": "Pretrained encoders only. No recovered or paper-trained PARE head is included."}
    target = ROOT / "reports/models_manifest.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {target}", flush=True)


if __name__ == "__main__":
    main()
