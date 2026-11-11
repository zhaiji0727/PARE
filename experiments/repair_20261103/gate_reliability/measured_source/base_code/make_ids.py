#!/usr/bin/env python3
"""Create deterministic SARLANG-1M image splits from the caption CSV files.

The official caption train/test files define the outer split.  We additionally
hold out a deterministic image-level validation subset from the official train
split.  Images, rather than captions, are split to prevent caption leakage.
"""
import argparse
import csv
import json
import os
import random
from runtime import DATA_ROOT


def build_index(root):
    index = set()
    for _, _, filenames in os.walk(root):
        index.update(filenames)
    return index


def ids_from_csv(csv_path, available):
    ids = []
    seen = set()
    with open(csv_path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        image_key = reader.fieldnames[0]
        for row in reader:
            image_id = row[image_key].strip()
            if image_id in available and image_id not in seen:
                seen.add(image_id)
                ids.append(image_id)
    return ids


def write_json(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default=str(DATA_ROOT / "SARimages_pre"))
    parser.add_argument(
        "--train_csv",
        default=str(DATA_ROOT / "Text_unz/Text/Caption/train/Caption_train.csv"),
    )
    parser.add_argument(
        "--test_csv",
        default=str(DATA_ROOT / "Text_unz/Text/Caption/test/Caption_test.csv"),
    )
    parser.add_argument("--out_dir", default=str(DATA_ROOT / "ret"))
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 <= args.val_frac < 1.0:
        raise ValueError("--val_frac must be in [0, 1)")

    available = build_index(args.img_root)
    official_train = ids_from_csv(args.train_csv, available)
    test_ids = ids_from_csv(args.test_csv, available)
    overlap = set(official_train) & set(test_ids)
    if overlap:
        raise RuntimeError(f"official train/test overlap: {len(overlap)} images")

    shuffled = official_train[:]
    random.Random(args.seed).shuffle(shuffled)
    n_val = int(len(shuffled) * args.val_frac)
    val_set = set(shuffled[:n_val])
    fit_ids = [image_id for image_id in official_train if image_id not in val_set]
    val_ids = [image_id for image_id in official_train if image_id in val_set]

    os.makedirs(args.out_dir, exist_ok=True)
    write_json(os.path.join(args.out_dir, "official_train_ids.json"), official_train)
    # ``train_ids`` retains the dataset's full official training split for the
    # final retraining run. ``fit_ids`` excludes the nested validation subset
    # and is useful when reproducing an individual sweep trial.
    write_json(os.path.join(args.out_dir, "train_ids.json"), official_train)
    write_json(os.path.join(args.out_dir, "fit_ids.json"), fit_ids)
    write_json(os.path.join(args.out_dir, "val_ids.json"), val_ids)
    write_json(os.path.join(args.out_dir, "test_ids.json"), test_ids)
    write_json(
        os.path.join(args.out_dir, "split_metadata.json"),
        {
            "seed": args.seed,
            "val_frac": args.val_frac,
            "official_train": len(official_train),
            "train": len(official_train),
            "fit_without_validation": len(fit_ids),
            "validation": len(val_ids),
            "test": len(test_ids),
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
        },
    )
    print(
        f"official_train={len(official_train)} fit={len(fit_ids)} "
        f"validation={len(val_ids)} test={len(test_ids)} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
