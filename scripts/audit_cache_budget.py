#!/usr/bin/env python3
"""Check exact head counts and tensor storage; do not measure retrieval or latency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from saqre_phys import SAQREHead


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "results/smoke/cache_vitb32_test.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results/cache_budget_20260916/summary.json")
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    n, dim, coarse, fine = len(cache["ids"]), cache["D"], cache["Mc"], cache["Mf"]
    expected = {"scene": (n, dim), "agg": (n, dim), "nu": (n,), "L": (n,), "c_star": (n,)}
    for prefix, count in (("c", coarse), ("f", fine)):
        expected.update({prefix + "_emb": (n, count, dim), prefix + "_mu": (n, count),
                         prefix + "_t": (n, count), prefix + "_tf": (n, count, 4),
                         prefix + "_n": (n,)})
    tensors = {key: value for key, value in cache.items() if torch.is_tensor(value)}
    assert set(tensors) == set(expected)
    fields = {}
    for key, value in tensors.items():
        assert tuple(value.shape) == expected[key], (key, value.shape, expected[key])
        assert value.dtype == (torch.int64 if key in {"c_n", "f_n"} else torch.float32)
        fields[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                       "bytes": value.numel() * value.element_size()}
    measured_per_image = sum(item["bytes"] for item in fields.values()) // n

    def storage(d):
        return 4 * ((coarse + fine + 2) * d + 6 * (coarse + fine) + 3) + 16

    assert measured_per_image == storage(dim)
    dimensions = {}
    for d in (512, 768):
        torch.manual_seed(0)
        head = SAQREHead(d)
        parameters = sum(parameter.numel() for parameter in head.parameters())
        assert parameters == 259 * d + 651
        dimensions[str(d)] = {
            "head_parameters_measured": parameters,
            "cache_tensor_bytes_per_image": storage(d),
            "cache_tensor_kib_per_image": storage(d) / 1024,
            "gallery_3688_tensor_mib": storage(d) * 3688 / (1024 ** 2),
            "storage_evidence": "measured and schema-checked" if d == dim else "derived from the checked cache schema, not a new encoded cache",
        }
    report = {
        "status": "passed", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "torch": torch.__version__, "cached_images": n,
        "max_coarse": coarse, "max_fine": fine, "fields": fields,
        "tensor_bytes_per_image_measured": measured_per_image,
        "formula_bytes": "4*((Mc+Mf+2)*d + 6*(Mc+Mf) + 3) + 16",
        "dimensions": dimensions,
        "provenance": {str(path.relative_to(ROOT)): digest(path) for path in
                       (Path(__file__), ROOT / "code/saqre_phys.py", ROOT / "code/cache_primitives.py", args.cache)},
        "limits": ["Tensor payload only; excludes Python metadata, filenames, serialization and allocator overhead.",
                   "The small cache checks allocation layout, not accuracy or full-dataset encoding.",
                   "The 3688-image totals are arithmetic storage projections, not runtime measurements.",
                   "No model training, GPU use, retrieval evaluation or latency benchmark was performed."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "dimensions": dimensions, "report": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
