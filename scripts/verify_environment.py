#!/usr/bin/env python3
"""Record the actual runtime and check a small CUDA operation, without training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": dict(sorted(versions.items())),
              "torch_cuda_runtime": torch.version.cuda,
              "compiled_cuda_architectures": torch.cuda.get_arch_list(),
              "cuda_available": torch.cuda.is_available(),
              "historical_runtime": "Python 3.10 / torch 2.3.1 cu121 / A100; not identical to this host"}
    inventory = subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader"], capture_output=True, text=True, check=True)
    report["gpu_inventory"] = inventory.stdout.strip().splitlines()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        a = torch.ones((32, 32), device=device)
        result = a @ a
        torch.cuda.synchronize(device)
        if not torch.allclose(result.cpu(), torch.full((32, 32), 32.0)):
            raise RuntimeError("CUDA matrix multiplication failed")
        report["cuda_smoke"] = {"status": "passed", "device": str(device),
                                "capability": torch.cuda.get_device_capability(device)}
    else:
        raise RuntimeError("CUDA is unavailable on this GPU host")
    (ROOT / "reports").mkdir(exist_ok=True)
    target = ROOT / "reports/environment.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", sys.executable],
                            capture_output=True, text=True, check=True)
    (ROOT / "requirements.lock.txt").write_text(freeze.stdout)
    print(json.dumps({"status": "passed", "torch": torch.__version__,
                      "cuda": torch.version.cuda, "report": str(target)}, indent=2))


if __name__ == "__main__":
    main()
