"""Shared, explicit query selection and coverage for retrieval evaluators."""

import hashlib
from collections import Counter


def add_query_arguments(parser):
    parser.add_argument("--max_q", "--max_queries", dest="max_q", type=int, default=0,
                        help="0 evaluates every eligible query; explicit 2000 restores legacy truncation")
    parser.add_argument("--query_mode", choices=("first-caption", "all-captions"), default="first-caption",
                        help="all-captions weights each caption row equally, including repeated rows")


def select_queries(captions, gallery_ids, mode="first-caption", max_queries=0):
    if max_queries < 0:
        raise ValueError("max_queries must be nonnegative")
    if mode not in ("first-caption", "all-captions"):
        raise ValueError("Unknown query mode")
    if len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError("Gallery IDs must be unique")
    id_to_row = {name: row for row, name in enumerate(gallery_ids)}
    available = []
    for name, values in captions.items():
        if name not in id_to_row:
            continue
        selected = values[:1] if mode == "first-caption" else values
        for index, caption in enumerate(selected):
            available.append({
                "query_id": f"{name}#caption={index + 1}",
                "image_id": name,
                "caption_index_one_based": index + 1,
                "caption": caption,
                "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
                "gold_gallery_row": id_to_row[name],
            })
    records = available[:max_queries] if max_queries else available
    if not records:
        raise ValueError("No caption queries match the cached image IDs")
    images = {record["image_id"] for record in records}
    return records, {
        "mode": mode,
        "max_queries": max_queries,
        "available_queries": len(available),
        "evaluated_queries": len(records),
        "truncated": len(records) < len(available),
        "order": "First image encounter in caption CSV; captions in original within-image row order",
        "weighting": "Each evaluated caption row has equal weight; no exact-text deduplication",
        "positive_definition": "One official image ID per query; equal text or other image content is not automatically a positive",
        "evaluated_unique_images": len(images),
        "query_prefix_counts": dict(sorted(Counter(record["image_id"].split("_")[0] for record in records).items())),
        "query_image_prefix_counts": dict(sorted(Counter(name.split("_")[0] for name in images).items())),
        "gallery_prefix_counts": dict(sorted(Counter(name.split("_")[0] for name in gallery_ids).items())),
        "gallery_images_without_evaluated_query": [name for name in gallery_ids if name not in images],
    }
