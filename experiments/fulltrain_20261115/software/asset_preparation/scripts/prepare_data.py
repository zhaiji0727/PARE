#!/usr/bin/env python3
"""Download and verify the official SARLANG-1M caption-track assets."""

import argparse
import csv
import hashlib
import json
import shutil
import stat
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests
from PIL import Image


REPOSITORY = "YiminJimmy/SARLANG-1M"
REVISION = "0271408f71bcd9afff69e4e7d324cae912461ebd"
ASSETS = {
    "Text.zip": {
        "size": 17074606,
        "sha256": "1447e79768d192341726ca2529e7dd94314ae9155f8e593637e2a3cebc2d4062",
    },
    "SARimages_preprocessed.zip": {
        "size": 6792425151,
        "sha256": "19b3f762a8eead69a6d015abe42c3f78272d667d60546f98e8c1e61a4ce3e63d",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(name, directory):
    expected = ASSETS[name]
    destination = directory / name
    partial = directory / (name + ".part")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
    if destination.exists():
        if destination.stat().st_size != expected["size"] or sha256(destination) != expected["sha256"]:
            raise ValueError(f"Existing archive fails verification: {destination}")
        print(f"Verified existing {name}", flush=True)
        return {**expected, "url": url, "path": str(destination), "verified": True}
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(10):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected["size"]:
            raise ValueError(f"Oversized partial archive: {partial}")
        if offset == expected["size"]:
            break
        print(f"Downloading {name}: {offset}/{expected['size']} bytes, attempt {attempt + 1}", flush=True)
        try:
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an unexpected byte range")
                    mode = "ab"
                elif response.status_code == 200:
                    mode = "wb"
                    offset = 0
                else:
                    raise ValueError(f"Unexpected download status {response.status_code}")
                started = last_print = time.monotonic()
                transferred = 0
                with partial.open(mode) as output:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not block:
                            continue
                        output.write(block)
                        transferred += len(block)
                        if offset + transferred > expected["size"]:
                            raise ValueError("Download exceeds the declared asset size")
                        now = time.monotonic()
                        if now - last_print >= 20:
                            rate = transferred / max(now - started, 1) / 1e6
                            print(f"{name}: {(offset + transferred) / expected['size']:.1%}, {rate:.1f} MB/s", flush=True)
                            last_print = now
            if partial.stat().st_size == expected["size"]:
                break
        except requests.RequestException as error:
            print(f"Retrying {name} after {type(error).__name__}; partial bytes retained", flush=True)
            time.sleep(min(2 ** attempt, 30))
    else:
        raise RuntimeError(f"Download incomplete after retries: {partial}")
    actual = sha256(partial)
    if actual != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
    partial.replace(destination)
    print(f"SHA-256 verified: {name}", flush=True)
    return {**expected, "url": url, "path": str(destination), "verified": True}


def safe_member(info):
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if path.is_absolute() or ".." in path.parts or "\\" in info.filename or stat.S_ISLNK(mode):
        raise ValueError(f"Unsafe ZIP member: {info.filename}")


def extract_member(archive, info, destination):
    safe_member(info)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with archive.open(info) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    if temporary.stat().st_size != info.file_size:
        raise ValueError(f"Extracted size mismatch: {info.filename}")
    temporary.replace(destination)


def read_splits(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ("train", "fit", "val", "test"):
        original = source / f"{name}_ids.json"
        target = destination / original.name
        shutil.copy2(original, target)
        values = json.loads(target.read_text(encoding="utf-8"))
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs in {name}")
        if any(Path(value).name != value or value in (".", "..") for value in values):
            raise ValueError(f"Unsafe image IDs in {name}")
        splits[name] = set(values)
    shutil.copy2(source / "split_metadata.json", destination / "split_metadata.json")
    if splits["fit"] & splits["val"] or splits["train"] & splits["test"]:
        raise ValueError("Split leakage detected")
    if splits["fit"] | splits["val"] != splits["train"]:
        raise ValueError("Fit/validation do not partition training")
    return splits


def extract_data(data, required):
    captions = {}
    with zipfile.ZipFile(data / "archives" / "Text.zip") as archive:
        for split in ("train", "test"):
            suffix = f"Text/Caption/{split}/Caption_{split}.csv"
            matches = [info for info in archive.infolist() if info.filename.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one {suffix}, found {len(matches)}")
            target = data / "Text_unz" / suffix
            extract_member(archive, matches[0], target)
            captions[split] = target
    with zipfile.ZipFile(data / "archives" / "SARimages_preprocessed.zip") as archive:
        selected = {}
        for info in archive.infolist():
            name = PurePosixPath(info.filename).name
            if not info.is_dir() and name in required:
                if name in selected:
                    raise ValueError(f"Duplicate required image name in archive: {name}")
                selected[name] = info
        missing = required - set(selected)
        if missing:
            raise ValueError(f"Required images absent from archive: {len(missing)}, examples: {sorted(missing)[:5]}")
        extracted_bytes = 0
        for index, (name, info) in enumerate(selected.items(), 1):
            extract_member(archive, info, data / "SARimages_pre" / name)
            extracted_bytes += info.file_size
            if index % 1000 == 0:
                print(f"Extracted {index}/{len(required)} required images", flush=True)
        print(f"Extracted {len(required)} images, {extracted_bytes} bytes", flush=True)
    return captions


def validate(data, splits):
    caption_report = {}
    for split in ("train", "test"):
        path = data / "Text_unz" / "Text" / "Caption" / split / f"Caption_{split}.csv"
        valid_ids = set()
        rows = nonempty = 0
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            image_key = columns[0]
            caption_key = next(key for key in columns if "caption" in key.lower() and "type" not in key.lower())
            for row in reader:
                rows += 1
                if row[caption_key].strip():
                    valid_ids.add(row[image_key].strip())
                    nonempty += 1
        missing = splits[split] - valid_ids
        caption_report[split] = {
            "path": str(path), "sha256": sha256(path), "columns": columns,
            "rows": rows, "nonempty_caption_rows": nonempty,
            "all_caption_image_ids": len(valid_ids), "required_image_ids": len(splits[split]),
            "required_ids_with_caption": len(splits[split] & valid_ids),
            "missing_caption_ids": sorted(missing),
            "extra_caption_ids_outside_preserved_split": len(valid_ids - splits[split]),
        }
        if missing:
            raise ValueError(f"Missing {split} captions: {len(missing)}")
    required = splits["train"] | splits["test"]
    inventory = []
    formats = {}
    with zipfile.ZipFile(data / "archives" / "SARimages_preprocessed.zip") as archive:
        expected = {PurePosixPath(info.filename).name: info for info in archive.infolist()
                    if not info.is_dir() and PurePosixPath(info.filename).name in required}
        for name in sorted(required):
            path = data / "SARimages_pre" / name
            info = expected[name]
            digest = hashlib.sha256()
            crc = size = 0
            with path.open("rb") as stream:
                signature = stream.read(8)
                if signature == b"\x89PNG\r\n\x1a\n":
                    image_format = "PNG"
                else:
                    # Some official .png members contain TIFF bytes; preserve them.
                    with Image.open(path) as image:
                        image_format = image.format
                        image.verify()
                    if image_format not in ("TIFF", "JPEG", "PNG"):
                        raise ValueError(f"Unexpected image format {image_format}: {name}")
                stream.seek(0)
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    crc = zlib.crc32(block, crc)
                    size += len(block)
            if size != info.file_size or crc != info.CRC:
                raise ValueError(f"Extracted image CRC/size mismatch: {name}")
            formats[image_format] = formats.get(image_format, 0) + 1
            inventory.append({"name": name, "bytes": size, "sha256": digest.hexdigest(),
                              "crc32": f"{crc:08x}", "format": image_format})
    return {"splits": {name: len(values) for name, values in splits.items()},
            "fit_val_disjoint": True, "train_test_disjoint": True,
            "fit_union_val_equals_train": True, "captions": caption_report,
            "required_images": len(required), "verified_images": len(inventory),
            "image_formats": formats,
            "png_extension_with_other_encoding": len(inventory) - formats.get("PNG", 0),
            "original_image_bytes_and_names_preserved": True,
            "image_bytes": sum(item["bytes"] for item in inventory)}, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--workspace", type=Path, default=workspace_default)
    parser.add_argument("--split-source", type=Path,
                        default=workspace_default / "data" / "splits_original")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    data = workspace / "data"
    report_path = workspace / "reports" / "data_preparation.json"
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "repository": REPOSITORY, "revision": REVISION,
              "source_url": f"https://huggingface.co/datasets/{REPOSITORY}",
              "split_source": str(args.split_source.resolve()), "archives": {}}
    write_json(report_path, report)
    try:
        splits = read_splits(args.split_source, data / "ret")
        for name, expected in ASSETS.items():
            if args.verify_only:
                path = data / "archives" / name
                if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
                    raise ValueError(f"Archive verification failed: {path}")
                report["archives"][name] = {**expected, "path": str(path), "verified": True}
            else:
                report["archives"][name] = download(name, data / "archives")
            write_json(report_path, report)
        if not args.verify_only:
            extract_data(data, splits["train"] | splits["test"])
        coverage, inventory = validate(data, splits)
        report.update(coverage)
        write_json(workspace / "reports" / "data_image_inventory.json", inventory)
        report["status"] = "complete"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        print(json.dumps({"status": report["status"], "verified_images": report["verified_images"],
                          "splits": report["splits"], "report": str(report_path)}, indent=2), flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
