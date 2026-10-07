#!/usr/bin/env python3
"""Create a new content-disjoint protocol without modifying official data/splits."""

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_fingerprint(path):
    path = Path(path)
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(b"RGB\0" + struct.pack(">QQ", *rgb.size) + rgb.tobytes())
        return {
            "image_id": path.name,
            "path": str(path.resolve()),
            "width": rgb.width,
            "height": rgb.height,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "rgb_content_sha256": digest.hexdigest(),
        }


def make_protocol(train_ids, test_ids, fingerprints, seed=0, val_fraction=0.1):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError("Input split IDs must be unique")
    if set(train_ids) & set(test_ids):
        raise ValueError("Official split IDs must be disjoint")
    content = {item["image_id"]: item["rgb_content_sha256"] for item in fingerprints}
    if set(content) != set(train_ids) | set(test_ids):
        raise ValueError("Fingerprints must cover exactly all input IDs")
    test_groups = defaultdict(list)
    train_groups = defaultdict(list)
    for name in test_ids:
        test_groups[content[name]].append(name)
    for name in train_ids:
        train_groups[content[name]].append(name)
    selected = set()
    excluded = []
    for key, names in sorted(train_groups.items()):
        if key in test_groups:
            excluded.extend({
                "image_id": name,
                "reason": "content_already_in_official_test",
                "rgb_content_sha256": key,
                "matching_test_ids": sorted(test_groups[key]),
            } for name in sorted(names))
        else:
            canonical = min(names)
            selected.add(canonical)
            excluded.extend({
                "image_id": name,
                "reason": "noncanonical_training_content_duplicate",
                "rgb_content_sha256": key,
                "canonical_train_id": canonical,
            } for name in sorted(names) if name != canonical)
    # Keep canonical IDs in official order, then use one isolated seed-0 shuffle.
    train = [name for name in train_ids if name in selected]
    shuffled = train[:]
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    if n_val == 0 or n_val == len(shuffled):
        raise ValueError("Insufficient unique training content for nonempty fit/val")
    val_set = set(shuffled[:n_val])
    splits = {
        "train": train,
        "fit": [name for name in train if name not in val_set],
        "val": [name for name in train if name in val_set],
        "test": list(test_ids),
    }
    hashes = {name: {content[image_id] for image_id in ids} for name, ids in splits.items()}
    assert not hashes["train"] & hashes["test"]
    assert not hashes["fit"] & hashes["val"]
    assert len(hashes["train"]) == len(train)
    return {
        "splits": splits,
        "excluded_training_ids": excluded,
        "official_test_duplicate_content_groups": [
            {"rgb_content_sha256": key, "image_ids": names}
            for key, names in sorted(test_groups.items()) if len(names) > 1
        ],
        "content_disjoint_train_test": True,
        "content_disjoint_fit_val": True,
        "clean_train_unique_content": True,
    }


def read_caption_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    image_key = fields[0]
    caption_key = next(key for key in fields if "caption" in key.lower() and "type" not in key.lower())
    return fields, rows, image_key, caption_key


def write_filtered_captions(source, destination, ids):
    fields, rows, image_key, caption_key = read_caption_rows(source)
    allowed = set(ids)
    selected = [row for row in rows if row[image_key].strip() in allowed]
    covered = {row[image_key].strip() for row in selected if row[caption_key].strip()}
    if covered != allowed:
        raise ValueError("Not every split ID has a nonempty caption")
    with Path(destination).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return {"rows": len(selected), "nonempty_rows": sum(bool(row[caption_key].strip()) for row in selected),
            "unique_captioned_images": len(covered)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    workspace = args.workspace.resolve()
    output = (args.out or workspace / "data/ret_clean_20261007").resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a fresh output to preserve provenance")
    original = workspace / "data/ret"
    train_path, test_path = original / "train_ids.json", original / "test_ids.json"
    caption_paths = {
        name: workspace / f"data/Text_unz/Text/Caption/{name}/Caption_{name}.csv"
        for name in ("train", "test")
    }
    input_paths = [train_path, test_path, *caption_paths.values(), Path(__file__).resolve()]
    before_hashes = {str(path): file_hash(path) for path in input_paths}
    train_ids, test_ids = json.loads(train_path.read_text()), json.loads(test_path.read_text())
    image_paths = [workspace / "data/SARimages_pre" / name for name in sorted(set(train_ids) | set(test_ids))]
    started = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fingerprints = []
        for index, value in enumerate(pool.map(image_fingerprint, image_paths), 1):
            fingerprints.append(value)
            if index % 1000 == 0:
                print(f"Decoded and hashed {index}/{len(image_paths)} images", flush=True)
    protocol = make_protocol(train_ids, test_ids, fingerprints, args.seed, args.val_fraction)
    if before_hashes != {str(path): file_hash(path) for path in input_paths}:
        raise RuntimeError("Input files changed during audit")
    output.mkdir(parents=True)
    for name, ids in protocol["splits"].items():
        write_json(output / f"{name}_ids.json", ids)
    write_json(output / "image_fingerprints.json", fingerprints)
    _, train_rows, image_key, caption_key = read_caption_rows(caption_paths["train"])
    caption_counts = defaultdict(int)
    for row in train_rows:
        caption_counts[row[image_key].strip()] += 1
    for entry in protocol["excluded_training_ids"]:
        entry["discarded_caption_rows"] = caption_counts[entry["image_id"]]
    write_json(output / "excluded_training_ids.json", protocol["excluded_training_ids"])
    write_json(output / "official_test_duplicate_content_groups.json", protocol["official_test_duplicate_content_groups"])
    caption_stats = {}
    for name, ids in protocol["splits"].items():
        source = caption_paths["test" if name == "test" else "train"]
        destination = output / f"Caption_{name}.csv"
        caption_stats[name] = write_filtered_captions(source, destination, ids)
        if name == "test":
            # Preserve the official test CSV bytes as well as its rows and IDs.
            shutil.copyfile(source, destination)
    reasons = defaultdict(lambda: {"images": 0, "caption_rows": 0})
    for entry in protocol["excluded_training_ids"]:
        reasons[entry["reason"]]["images"] += 1
        reasons[entry["reason"]]["caption_rows"] += entry["discarded_caption_rows"]
    manifest = {
        "status": "complete",
        "protocol": "content_disjoint_rgb_v1_20261007",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "workers": args.workers,
        "pixel_hash_definition": "SHA256(b'RGB\\0' + big-endian uint64 width,height + PIL convert('RGB').tobytes()); dimensions and pixels included; no resize/normalization/EXIF transpose",
        "policy": [
            "Keep official test IDs, order and caption CSV unchanged; test captions never influence filtering or splitting.",
            "Remove every training ID whose exact decoded RGB content occurs in official test.",
            "For remaining training duplicate content, retain lexicographically smallest ID and only that ID's original captions; do not merge captions from other IDs.",
            "Preserve selected IDs' official train order; random.Random(seed) shuffle, first floor(val_fraction*N) validation, remainder fit.",
            "No original file/split/annotation is modified. This is a new evaluation protocol, not a reproduction of archived results.",
            "Exact RGB grouping does not detect spatial overlap or near duplicates.",
        ],
        "original_images": {"train": len(train_ids), "test": len(test_ids)},
        "images_decoded": len(fingerprints),
        "split_counts": {name: len(ids) for name, ids in protocol["splits"].items()},
        "caption_counts": caption_stats,
        "training_exclusions": dict(reasons),
        "excluded_training_caption_rows": sum(entry["discarded_caption_rows"] for entry in protocol["excluded_training_ids"]),
        "content_disjoint_train_test": protocol["content_disjoint_train_test"],
        "content_disjoint_fit_val": protocol["content_disjoint_fit_val"],
        "official_test_duplicate_content_group_count": len(protocol["official_test_duplicate_content_groups"]),
        "test_relevance_policy": "Official image-ID positives retained. If test content duplicates exist, identical content under other test IDs is a potential false negative; report content-group relevance as an additional protocol before content-based claims.",
        "input_sha256": before_hashes,
        "output_sha256": {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "split_counts": manifest["split_counts"],
                      "caption_counts": caption_stats, "training_exclusions": dict(reasons),
                      "test_duplicate_groups": manifest["official_test_duplicate_content_group_count"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
