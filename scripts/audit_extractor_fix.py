#!/usr/bin/env python3
"""CPU-only comparison of the frozen old extractor and the October revision."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import saqre_phys as revised

OLD_PATH = ROOT / "diagnostics/reference/saqre_phys_before_20261007.py"
spec = importlib.util.spec_from_file_location("extractor_before_fixes", OLD_PATH)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
FIGURE_ID = "SN6_Train_AOI_11_Rotterdam_20190804111224_20190804111453_tile_8679.png"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def exact_jaccard(left, right):
    a, b = set(map(tuple, left)), set(map(tuple, right))
    return len(a & b) / len(a | b) if a or b else 1.0


def centers(boxes):
    return [(x1 - 12 if x0 == 0 else x0 + 12,
             y1 - 12 if y0 == 0 else y0 + 12) for x0, y0, x1, y1 in boxes]


def raw_old_background(gray):
    outer, _ = original.local_mean_std(gray, 17)
    inner, _ = original.local_mean_std(gray, 5)
    return (outer * 289 - inner[6:6 + outer.shape[0], 6:6 + outer.shape[1]] * 25) / 264


def direct_checks(gray, background, boxes):
    max_mean_error = max_std_error = max_ring_error = 0.0
    mean_checks = ring_checks = 0
    for win in (5, 9, 16, 17):
        mean, std = revised.local_mean_std(gray, win)
        for y, x in ((0, 0), (mean.shape[0] // 2, mean.shape[1] // 2),
                     (mean.shape[0] - 1, mean.shape[1] - 1)):
            patch = gray[y:y + win, x:x + win].astype(np.float64)
            max_mean_error = max(max_mean_error, abs(float(mean[y, x] - patch.mean())))
            max_std_error = max(max_std_error, abs(float(std[y, x] - patch.std())))
            mean_checks += 1
    height, width = gray.shape
    probes = set(centers(boxes)) | {(8, 8), (width // 2, height // 2), (width - 9, height - 9)}
    for x, y in probes:
        patch = gray[y - 8:y + 9, x - 8:x + 9].astype(np.float64)
        direct = (patch.sum() - patch[6:11, 6:11].sum()) / 264
        max_ring_error = max(max_ring_error, abs(float(background[y - 8, x - 8] - direct)))
        ring_checks += 1
    return max_mean_error, max_std_error, max_ring_error, mean_checks, ring_checks


def inspect_image(image_id):
    image_path = ROOT / "data/SARimages_pre" / image_id
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    gray = revised.to_intensity(rgb)
    old_enl, new_enl = original.estimate_enl(gray), revised.estimate_enl(gray)
    old_coarse = original.region_proposals(gray, 1 / np.sqrt(old_enl))
    new_coarse = revised.region_proposals(gray, 1 / np.sqrt(new_enl))
    old_dets = [original.ca_cfar(gray, pfa_factor=factor) for factor in (2.0, 2.5, 3.0)]
    new_dets = [revised.ca_cfar(gray, pfa_factor=factor) for factor in (2.0, 2.5, 3.0)]
    old_fine = original.scatterer_patches(gray, old_dets[1])
    new_fine = revised.scatterer_patches(gray, new_dets[1])
    precision_only_fine = original.scatterer_patches(gray, new_dets[1])
    tie_only_fine = revised.scatterer_patches(gray, old_dets[1])
    old_background, new_background = raw_old_background(gray), revised._cfar_background(gray)
    old_inverse = (old_dets[1] & ~old_dets[0]) | (old_dets[2] & ~old_dets[1])
    new_inverse = (new_dets[1] & ~new_dets[0]) | (new_dets[2] & ~new_dets[1])
    assert not new_inverse.any() and np.all(new_background >= 0)
    mean_error, std_error, ring_error, stats_checks, ring_checks = direct_checks(gray, new_background, new_fine)
    assert mean_error < 1e-9 and std_error < 1e-6 and ring_error < 1e-9
    extracted = revised.extract_primitives(rgb)
    for boxes, key in ((new_coarse, "regions"), (new_fine, "scatterers")):
        assert len(boxes) == len(extracted[key])
        for box, (crop, _) in zip(boxes, extracted[key]):
            np.testing.assert_array_equal(np.asarray(crop), np.asarray(rgb.crop(box)))
    row = {
        "image_id": image_id, "image_sha256": sha256(image_path),
        "width": gray.shape[1], "height": gray.shape[0],
        "old_enl": old_enl, "new_enl": new_enl,
        "old_coarse_count": len(old_coarse), "new_coarse_count": len(new_coarse),
        "old_fine_count": len(old_fine), "new_fine_count": len(new_fine),
        "coarse_exact_box_set_changed": int(set(old_coarse) != set(new_coarse)),
        "fine_exact_box_set_changed": int(set(old_fine) != set(new_fine)),
        "coarse_exact_box_set_jaccard": exact_jaccard(old_coarse, new_coarse),
        "fine_exact_box_set_jaccard": exact_jaccard(old_fine, new_fine),
        "precision_only_fine_box_set_changed": int(set(old_fine) != set(precision_only_fine)),
        "tie_only_fine_box_set_changed": int(set(old_fine) != set(tie_only_fine)),
        "old_default_detection_count": int(old_dets[1].sum()),
        "new_default_detection_count": int(new_dets[1].sum()),
        "default_detection_membership_changes": int((old_dets[1] ^ new_dets[1]).sum()),
        "old_nonmonotonic_pixel_count": int(old_inverse.sum()),
        "new_nonmonotonic_pixel_count": int(new_inverse.sum()),
        "old_negative_background_count": int((old_background < 0).sum()),
        "new_negative_background_count": int((new_background < 0).sum()),
        "old_default_zero_intensity_detection_count": int((old_dets[1] & (gray == 0)).sum()),
        "new_default_zero_intensity_detection_count": int((new_dets[1] & (gray == 0)).sum()),
        "old_clipped_fine_count": sum(x1 - x0 < 24 or y1 - y0 < 24 for x0, y0, x1, y1 in old_fine),
        "new_clipped_fine_count": sum(x1 - x0 < 24 or y1 - y0 < 24 for x0, y0, x1, y1 in new_fine),
        "direct_mean_max_absolute_error": mean_error,
        "direct_std_max_absolute_error": std_error,
        "direct_ring_mean_max_absolute_error": ring_error,
        "direct_local_window_checks": stats_checks, "direct_ring_checks": ring_checks,
        "new_crop_pixel_verification_passed": True,
    }
    detail = {"image_id": image_id, "old_coarse_boxes": old_coarse, "new_coarse_boxes": new_coarse,
              "old_fine_boxes": old_fine, "new_fine_boxes": new_fine,
              "precision_only_fine_boxes": precision_only_fine, "tie_only_fine_boxes": tie_only_fine,
              "old_fine_centers_xy_intensity": [[x, y, float(gray[y, x])] for x, y in centers(old_fine)],
              "new_fine_centers_xy_intensity": [[x, y, float(gray[y, x])] for x, y in centers(new_fine)]}
    return row, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results/extractor_fix_20261007")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use 1..4 CPU workers")
    os.nice(10)
    split = ROOT / "data/ret/val_ids.json"
    ids = json.loads(split.read_text())
    assert len(ids) == len(set(ids)) == 919
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Use a fresh output directory")
    sources = {"old_extractor": OLD_PATH, "new_extractor": Path(revised.__file__),
               "audit_script": Path(__file__), "validation_ids": split}
    provenance = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in sources.items()}
    (output / "sources").mkdir(exist_ok=True)
    for key in ("old_extractor", "new_extractor", "audit_script"):
        shutil.copy2(sources[key], output / "sources" / (key + ".py"))
    started = time.monotonic()
    rows = []
    with (output / "per_image.csv").open("w", newline="") as table, (output / "boxes.jsonl").open("w") as details:
        writer = None
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                                 initializer=configure_worker) as executor:
            for index, (row, detail) in enumerate(executor.map(inspect_image, ids), 1):
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                details.write(json.dumps(detail) + "\n")
                rows.append(row)
                if index % 50 == 0 or index == len(ids):
                    table.flush()
                    details.flush()
                    print(json.dumps({"completed": index, "total": len(ids),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    figure_row, figure_detail = inspect_image(FIGURE_ID)
    for key, path in sources.items():
        assert sha256(path) == provenance[key]["sha256"], f"Source changed during audit: {key}"
    counts = ("old_coarse_count", "new_coarse_count", "old_fine_count", "new_fine_count",
              "coarse_exact_box_set_changed", "fine_exact_box_set_changed", "precision_only_fine_box_set_changed",
              "tie_only_fine_box_set_changed", "old_default_detection_count", "new_default_detection_count",
              "default_detection_membership_changes", "old_nonmonotonic_pixel_count", "new_nonmonotonic_pixel_count",
              "old_negative_background_count", "new_negative_background_count",
              "old_default_zero_intensity_detection_count", "new_default_zero_intensity_detection_count",
              "old_clipped_fine_count", "new_clipped_fine_count", "direct_local_window_checks", "direct_ring_checks")
    report = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All 919 original validation IDs for paired implementation comparison; not a leakage-clean training/evaluation run.",
        "image_count": len(rows), "device": "cpu", "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started, "extractor_version": revised.EXTRACTOR_VERSION,
        "provenance": provenance,
        "totals": {key: sum(row[key] for row in rows) for key in counts},
        "images": {key + "_positive": sum(row[key] > 0 for row in rows) for key in (
            "default_detection_membership_changes", "old_nonmonotonic_pixel_count", "new_nonmonotonic_pixel_count",
            "old_negative_background_count", "new_negative_background_count")},
        "maximum_errors": {key: max(row[key] for row in rows) for key in (
            "direct_mean_max_absolute_error", "direct_std_max_absolute_error", "direct_ring_mean_max_absolute_error")},
        "mean_exact_box_set_jaccard": {scale: sum(row[scale + "_exact_box_set_jaccard"] for row in rows) / len(rows)
                                      for scale in ("coarse", "fine")},
        "all_new_crop_pixel_verifications_passed": all(row["new_crop_pixel_verification_passed"] for row in rows),
        "figure_one": {"metrics": figure_row, "boxes_and_centers": figure_detail},
        "limitations": [
            "Exact box-set changes are not retrieval or detection accuracy and not semantic object coverage.",
            "Precision-only and tie-only fine comparisons isolate implementation changes at the fixed default detector threshold, not trained ablations.",
            "No cache embeddings, head training, retrieval accuracy, or GPU measurements were generated.",
            "The original validation list has separately documented exact-image duplicates across the original fit/validation split; this audit does not certify split independence.",
            "Changed extraction invalidates old embeddings and trained-head performance claims for the revised version.",
        ],
        "files": {name: {"path": str(output / name), "sha256": sha256(output / name)}
                  for name in ("per_image.csv", "boxes.jsonl")},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "summary": str(output / "summary.json"),
                      "totals": report["totals"], "maximum_errors": report["maximum_errors"]}, indent=2))


if __name__ == "__main__":
    main()
