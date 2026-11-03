#!/usr/bin/env python3
"""Read existing image headers only; distinguish projected raster spacing from GSD.

No pixel arrays are decoded and no image/archive is modified or extracted.
Only projected GeoTIFF coordinates with an explicit metre unit are interpreted.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys


TIFF_TYPES = {1: (1, "B"), 2: (1, None), 3: (2, "H"), 4: (4, "I"), 5: (8, None),
              12: (8, "d"), 16: (8, "Q")}
WANTED = {256, 257, 33550, 33922, 34264, 34735}


def read_header(path):
    size = path.stat().st_size
    result = {"file_bytes": size, "format": "unknown", "raster_spacing_m_xy": None,
              "status": "unrecognized_format"}
    with path.open("rb") as stream:
        header = stream.read(8)
        if header == b"\x89PNG\r\n\x1a\n":
            result.update(format="PNG", status="no_geotiff_metadata")
            tags = []
            while stream.tell() + 12 <= size:
                length, kind = struct.unpack(">I4s", stream.read(8))
                if stream.tell() + length + 4 > size:
                    raise ValueError("PNG chunk exceeds file size")
                if kind == b"IHDR":
                    data = stream.read(length)
                    result["size_wh"] = list(struct.unpack(">II", data[:8]))
                else:
                    stream.seek(length, 1)
                stream.read(4)
                if kind not in (b"IDAT", b"IEND", b"IHDR"):
                    tags.append(kind.decode("ascii", errors="replace"))
                if kind == b"IEND":
                    break
            result["other_chunk_types"] = sorted(set(tags))
            return result
        if header[:2] not in (b"II", b"MM"):
            return result
        endian = "<" if header[:2] == b"II" else ">"
        magic, offset = struct.unpack(endian + "HI", header[2:])
        result.update(format="TIFF", status="unsupported_tiff_variant")
        if magic != 42:
            return result
        if offset + 2 > size:
            raise ValueError("Invalid TIFF IFD offset")
        stream.seek(offset)
        count = struct.unpack(endian + "H", stream.read(2))[0]
        if count > 4096 or stream.tell() + 12 * count + 4 > size:
            raise ValueError("Invalid TIFF IFD length")
        entries = [stream.read(12) for _ in range(count)]
        values = {}
        result["tag_ids"] = []
        for entry in entries:
            tag, kind, nvalues, pointer = struct.unpack(endian + "HHII", entry)
            result["tag_ids"].append(tag)
            if tag not in WANTED or kind not in TIFF_TYPES:
                continue
            width, code = TIFF_TYPES[kind]
            length = width * nvalues
            if length > 65536 or not code:
                raise ValueError("Unexpected metadata value size/type")
            if length <= 4:
                raw = entry[8:8 + length]
            else:
                if pointer + length > size:
                    raise ValueError("TIFF metadata exceeds file size")
                stream.seek(pointer)
                raw = stream.read(length)
            values[tag] = list(struct.unpack(endian + str(nvalues) + code, raw))
        if 256 in values and 257 in values:
            result["size_wh"] = [values[256][0], values[257][0]]
        directory = values.get(34735, [])
        keys = {}
        if directory:
            if len(directory) < 4 or len(directory) != 4 + 4 * directory[3]:
                raise ValueError("Malformed GeoKey directory")
            for offset in range(4, len(directory), 4):
                key, location, count, value = directory[offset:offset + 4]
                if location == 0 and count == 1:
                    keys[str(key)] = value
        result["inline_geokeys"] = keys
        result["model_pixel_scale"] = values.get(33550)
        result["projected_crs_epsg"] = keys.get("3072")
        result["status"] = "unverified_coordinate_units_or_geometry"
        if keys.get("1024") != 1 or keys.get("3076") != 9001:
            return result
        if 34264 in values:
            # Avoid choosing one georeferencing representation when both occur.
            result["status"] = "matrix_transform_requires_review"
            return result
        scale = values.get(33550, [])
        tie = values.get(33922, [])
        if len(scale) >= 2 and len(tie) >= 6 and all(math.isfinite(x) and x > 0 for x in scale[:2]):
            result["raster_spacing_m_xy"] = scale[:2]
            result["status"] = "projected_metre_raster_spacing_available"
        return result


def summarize(ids, records):
    selected = [records[name] for name in ids]
    known = [row for row in selected if row["raster_spacing_m_xy"] is not None]
    return {
        "images": len(ids), "formats": dict(Counter(row["format"] for row in selected)),
        "statuses": dict(Counter(row["status"] for row in selected)),
        "with_projected_metre_raster_spacing": len(known),
        "without_verified_raster_spacing": len(selected) - len(known),
        "min_xy_m": [min(row["raster_spacing_m_xy"][axis] for row in known) for axis in (0, 1)] if known else None,
        "max_xy_m": [max(row["raster_spacing_m_xy"][axis] for row in known) for axis in (0, 1)] if known else None,
        "known_prefix_counts": dict(Counter(row["image_id"].split("_")[0] for row in known)),
        "unknown_prefix_counts": dict(Counter(row["image_id"].split("_")[0] for row in selected if row not in known)),
        "rounded_spacing_histogram_xy_6dp": dict(Counter(
            ",".join(f"{value:.6f}" for value in row["raster_spacing_m_xy"]) for row in known)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--split-root", required=True)
    args = parser.parse_args()
    splits = {name: json.loads((Path(args.split_root) / f"{name}_ids.json").read_text())
              for name in ("train", "fit", "val", "test")}
    ids = sorted(set(splits["train"]) | set(splits["test"]))
    root = Path(args.image_root)
    records, errors = {}, []
    for name in ids:
        if Path(name).name != name:
            raise ValueError("Only flat image basenames are supported")
        try:
            row = read_header(root / name)
        except Exception as exc:
            errors.append({"image_id": name, "error": str(exc)})
            row = {"format": "unknown", "status": "header_parse_error", "raster_spacing_m_xy": None}
        records[name] = {"image_id": name, **row}
    report = {
        "method": "Read primary IFD or PNG chunk headers; no image decoding, resampling, extraction or file modification",
        "known_spacing_criterion": "TIFF GeoKeys model=projected (1024=1), explicit metre unit (3076=9001), positive ModelPixelScale and Tiepoint",
        "scope_limitation": "Stored projected raster spacing; not a verified original-acquisition GSD or spatial resolution, and not extrapolated to unlabeled images",
        "summaries": {name: summarize(names, records) for name, names in splits.items()},
        "all_train_and_test": summarize(ids, records), "errors": errors, "images": records,
    }
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
