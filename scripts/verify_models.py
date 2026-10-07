#!/usr/bin/env python3
"""Load all five frozen encoders offline and exercise image/text encoding on CPU."""
import gc
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import open_clip
from PIL import Image
import torch


def main():
    torch.set_num_threads(4)
    specs = [
        ("vitb32", "ViT-B-32", "laion2b_s34b_b79k", None),
        ("vitb16", "ViT-B-16", "laion2b_s34b_b88k", None),
        ("remoteclip", "ViT-B-32", "laion2b_s34b_b79k",
         "external_checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt"),
        ("siglip", "ViT-B-16-SigLIP", "webli", None),
        ("georsclip", "ViT-B-32", "openai",
         "external_checkpoints/georsclip/ckpt/RS5M_ViT-B-32.pt"),
    ]
    report = {"device": "cpu", "offline": True, "models": []}
    for tag, architecture, pretrained, external in specs:
        print(f"Checking {tag}", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            architecture, pretrained=pretrained, cache_dir=str(ROOT / "models"))
        if external:
            state = torch.load(ROOT / "models" / external, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            del state
        model.eval()
        tokenizer = open_clip.get_tokenizer(architecture)
        with torch.inference_mode():
            visual = model.encode_image(preprocess(Image.new("RGB", (224, 224), "gray")).unsqueeze(0))
            text = model.encode_text(tokenizer(["A SAR image of a harbor with ships."]))
        if visual.shape != text.shape or not torch.isfinite(visual).all() or not torch.isfinite(text).all():
            raise RuntimeError(f"Invalid embeddings for {tag}")
        report["models"].append({"tag": tag, "architecture": architecture,
                                 "embedding_dimension": visual.shape[-1],
                                 "parameters": sum(p.numel() for p in model.parameters()),
                                 "external_checkpoint_strict_match": bool(external), "status": "passed"})
        del model, tokenizer, preprocess, visual, text
        gc.collect()
    report["status"] = "passed"
    target = ROOT / "reports/model_loading.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
