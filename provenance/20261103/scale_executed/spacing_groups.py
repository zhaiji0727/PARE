"""Prespecified observational bins for metadata-bearing test query images."""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path

from scale_contract import object_hash


BIN_LABELS = ("lt_0.30", "0.30_to_lt_0.45", "ge_0.45", "unavailable")
INTERPRETATION = (
    "Observational query subgroups by embedded projected-raster spacing, not a controlled "
    "acquisition-GSD experiment. Every rank retains the complete original test gallery. "
    "Geography, scene content and acquisition differences may confound subgroup comparisons."
)


def spacing_bin(row):
    spacing = row.get("raster_spacing_m_xy")
    if spacing is None:
        return "unavailable", None
    x, y = spacing
    if not all(math.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("Invalid positive raster spacing")
    nominal = Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    nominal_y = Decimal(str(y)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if nominal != nominal_y:
        raise ValueError("Axes differ after 0.01m rounding; a separate anisotropic rule is required")
    label = "lt_0.30" if nominal < Decimal("0.30") else (
        "0.30_to_lt_0.45" if nominal < Decimal("0.45") else "ge_0.45")
    return label, str(nominal)


def make_groups(audit, test_ids):
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate test image IDs")
    if len(test_ids) != audit["summaries"]["test"]["images"]:
        raise ValueError("Test population count differs from the header audit")
    records = audit["images"]
    groups = {label: [] for label in BIN_LABELS}
    nominal_by_id = {}
    for name in test_ids:
        label, nominal = spacing_bin(records[name])
        groups[label].append(name)
        nominal_by_id[name] = nominal
    known_count = len(test_ids) - len(groups["unavailable"])
    if known_count != audit["summaries"]["test"]["with_projected_metre_raster_spacing"]:
        raise ValueError("Test spacing coverage differs from the header audit")
    return {
        "interpretation": INTERPRETATION,
        "test_ids": test_ids,
        "test_ids_sha256": object_hash(test_ids),
        "n_gallery_retained_for_all_subgroups": len(test_ids),
        "rounding": "Round each stored axis to 0.01m using Decimal(str(value)).quantize(0.01, ROUND_HALF_UP); require equal rounded axes",
        "bin_edges_m_per_pixel_after_rounding": ["<0.30", "0.30<=s<0.45", "s>=0.45"],
        "groups": groups,
        "nominal_spacing_m_by_id": nominal_by_id,
        "counts": {label: len(names) for label, names in groups.items()},
        "filename_prefix_counts_by_bin": {label: dict(sorted(Counter(name.split("_")[0] for name in names).items()))
                                           for label, names in groups.items()},
        "filename_label_caveat": "Prefixes are filename labels, not independently verified SAR-sensor identities or GSD values",
    }


def rank_metrics(ranks):
    if not ranks:
        return None
    return {**{f"R@{k}": 100 * sum(rank <= k for rank in ranks) / len(ranks) for k in (1, 5, 10)},
            "MRR@10": 100 * sum(1 / rank if rank <= 10 else 0 for rank in ranks) / len(ranks)}


def summarize_rank_rows(rows, groups):
    """Rows: {image_id: str, ranks: {method: 1-based full-gallery rank}}."""
    by_id = {row["image_id"]: row for row in rows}
    expected = groups["test_ids"]
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError("Subgroup analysis requires exactly one unchanged query per complete test image ID")
    methods = set(rows[0]["ranks"])
    size = len(expected)
    for row in rows:
        if set(row["ranks"]) != methods:
            raise ValueError("Methods differ across query rows")
        for value in row["ranks"].values():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= size:
                raise ValueError("Expected finite integer full-gallery ranks")
    return {
        "interpretation": INTERPRETATION,
        "gallery_size": size,
        "groups_sha256": object_hash(groups),
        "subgroups": {label: {"n_queries": len(names),
                               "metrics": {method: rank_metrics([by_id[name]["ranks"][method] for name in names])
                                           for method in sorted(methods)},
                               "filename_prefix_counts": groups["filename_prefix_counts_by_bin"][label]}
                      for label, names in groups["groups"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--test-ids", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audit_path, ids_path = Path(args.audit), Path(args.test_ids)
    result = make_groups(json.loads(audit_path.read_text()), json.loads(ids_path.read_text()))
    result["audit_file_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    result["test_ids_file_sha256"] = hashlib.sha256(ids_path.read_bytes()).hexdigest()
    result["grouping_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "filename_prefix_counts_by_bin")}, indent=2))


if __name__ == "__main__":
    main()
