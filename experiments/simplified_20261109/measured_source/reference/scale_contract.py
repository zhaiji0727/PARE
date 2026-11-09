"""Pixel-grid perturbation contract; this never assigns an image a physical GSD."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

from PIL import Image, __version__ as pillow_version

EXTENSION = "synthetic_spacing_test"
SCHEMA = "pare-synthetic-spacing-v1"
ALLOWED_FACTORS = ("1", "1.5", "2")


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_string(value):
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError("Spacing factor must be finite")
    result = format(value.normalize(), "f")
    if result not in ALLOWED_FACTORS:
        raise ValueError(f"Prespecified factors are {ALLOWED_FACTORS}")
    return result


def target_size(width, height, factor):
    factor = Decimal(factor_string(factor))
    if min(width, height) < 1:
        raise ValueError("Invalid input image dimensions")
    size = tuple(int((Decimal(x) / factor).to_integral_value(rounding=ROUND_HALF_UP))
                 for x in (width, height))
    # The unchanged extractor requires a valid 17x17 outer CFAR window.
    if min(size) < 17:
        raise ValueError("Perturbed image is smaller than the unchanged 17x17 extractor support")
    return size


def transform_spec(factor):
    factor = factor_string(factor)
    return {
        "schema": SCHEMA,
        "nominal_spacing_factor": factor,
        "input_grid": "released_preprocessed_image_pixels",
        "physical_gsd_m": None,
        "color_conversion": "PIL RGB before resampling, as in original cache pipeline",
        "target_dimensions": "round_half_up(width/s), round_half_up(height/s)",
        "filter": "identity" if factor == "1" else "PIL.Image.Resampling.LANCZOS",
        "reducing_gap": None,
        "pillow_version": pillow_version,
        "resize_back_to_original_grid": False,
        "backbone_preprocessing": "unchanged, applied separately to each extracted crop",
        "interpretation": "synthetic spatial-scale sensitivity, not a real acquisition-GSD experiment",
    }


def decoded_hash(image):
    header = json.dumps({"mode": image.mode, "size": list(image.size)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + image.tobytes()).hexdigest()


def transform_image(image, factor):
    factor = factor_string(factor)
    source = image.convert("RGB")
    size = target_size(*source.size, factor)
    transformed = source.copy() if factor == "1" else source.resize(
        size, resample=Image.Resampling.LANCZOS, reducing_gap=None)
    record = {
        "source_size_wh": list(source.size),
        "transformed_size_wh": list(transformed.size),
        "effective_spacing_ratio_xy": [source.width / transformed.width,
                                       source.height / transformed.height],
        "source_decoded_rgb_sha256": decoded_hash(source),
        "transformed_decoded_rgb_sha256": decoded_hash(transformed),
        "physical_gsd_m": None,
    }
    return transformed, record


def bind_signature(base_signature, factor, helper_hashes):
    if EXTENSION in base_signature:
        raise ValueError("Cannot bind an already transformed signature")
    base = deepcopy(base_signature)
    return {
        **base,
        EXTENSION: {
            "transform": transform_spec(factor),
            "base_signature_sha256": object_hash(base),
            "helper_source_sha256": dict(sorted(helper_hashes.items())),
        },
    }


def verify_and_unwrap_signature(signature, factor, helper_hashes):
    if EXTENSION not in signature:
        raise ValueError("A normal cache cannot stand in for a bound scale-test cache")
    base = deepcopy(signature)
    base.pop(EXTENSION)
    expected = bind_signature(base, factor, helper_hashes)
    if signature != expected:
        raise ValueError("Scale, filter, base extraction, environment or helper signature mismatch")
    return base


def verify_image_records(ids, records, factor, original_file_hashes):
    if len(ids) != len(set(ids)) or set(ids) != set(records):
        raise ValueError("Image manifest does not exactly cover the gallery")
    if set(ids) != set(original_file_hashes):
        raise ValueError("Original source-file hashes do not exactly cover the gallery")
    for image_id in ids:
        row = records[image_id]
        width, height = row["source_size_wh"]
        target = target_size(width, height, factor)
        if tuple(row["transformed_size_wh"]) != target:
            raise ValueError(f"Wrong transformed dimensions for {image_id}")
        if row["effective_spacing_ratio_xy"] != [width / target[0], height / target[1]]:
            raise ValueError(f"Wrong effective spacing ratio for {image_id}")
        if row.get("physical_gsd_m") is not None:
            raise ValueError("A synthetic scale manifest must not invent a physical GSD")
        if row["source_file_sha256"] != original_file_hashes[image_id]:
            raise ValueError(f"Changed source file for {image_id}")
        for key in ("source_decoded_rgb_sha256", "transformed_decoded_rgb_sha256"):
            digest = row[key]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid decoded-pixel hash for {image_id}")
        if factor_string(factor) == "1" and row["source_decoded_rgb_sha256"] != row["transformed_decoded_rgb_sha256"]:
            raise ValueError("Identity condition changed decoded pixels")
