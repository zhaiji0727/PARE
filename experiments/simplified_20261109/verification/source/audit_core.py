"""Independent helpers: no experiment runner/model/validator is imported."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random

TAGS = ("vitb32", "vitb16", "remoteclip", "siglip", "georsclip")
VARIANTS = ("full", "strict", "strict_token_off")
SCALES = ("test", "spacing_1p5", "spacing_2")
ABLATIONS = {"full": [], "strict": ["cfar_strict"],
             "strict_token_off": ["cfar_strict", "ty"]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def textsha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def same_tensor_bytes(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            a.detach().cpu().contiguous().numpy().tobytes() ==
            b.detach().cpu().contiguous().numpy().tobytes())


def same_state_bytes(a, b):
    return set(a) == set(b) and all(same_tensor_bytes(a[key], b[key]) for key in a)


def epoch_order_hashes(seed, n_pairs=31945, epochs=10):
    order, rng, result = list(range(n_pairs)), random.Random(seed), []
    for _ in range(epochs):
        rng.shuffle(order)
        result.append(objsha(order))
    return result


def read_captions(path):
    """Keep image first encounter and all within-image caption row order."""
    result = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(bool(reader.fieldnames), "Empty caption CSV")
        image_key = reader.fieldnames[0]
        fields = [key for key in reader.fieldnames if "caption" in key.lower() and "type" not in key.lower()]
        require(bool(fields), "Missing caption column")
        for row in reader:
            image, caption = row[image_key].strip(), row[fields[0]].strip()
            if caption:
                result.setdefault(image, []).append(caption)
    return result


def training_identity(cache_ids, caption_path, expected_images=9181, expected_pairs=31945):
    require(len(cache_ids) == len(set(cache_ids)) == expected_images, "Wrong training cache IDs")
    captions, retained = read_captions(caption_path), set(cache_ids)
    pairs = [[image, textsha(caption)] for image, values in captions.items()
             if image in retained for caption in values]
    require(len(pairs) == expected_pairs and {p[0] for p in pairs} == retained,
            "Wrong training caption population")
    unique = sorted({caption for image, values in captions.items() if image in retained for caption in values})
    return {"pairs_sha256": objsha(pairs), "n_pairs": len(pairs),
            "unique_caption_sha256": [textsha(caption) for caption in unique],
            "n_unique_captions": len(unique)}


def evaluation_identity(cache_ids, caption_path, expected=3688):
    require(len(cache_ids) == len(set(cache_ids)) == expected, "Wrong evaluation cache IDs")
    captions, index = read_captions(caption_path), {image: row for row, image in enumerate(cache_ids)}
    queries = [image for image in captions if image in index]
    require(len(queries) == expected and set(queries) == set(index), "Wrong query population")
    return {"query_ids": [image + "#caption=1" for image in queries],
            "query_image_ids": queries, "gallery_ids": cache_ids,
            "query_caption_sha256": [textsha(captions[image][0]) for image in queries],
            "gold_gallery_index": [index[image] for image in queries]}


def metrics_from_ranks(ranks, n_gallery=3688):
    require(bool(ranks) and all(type(rank) is int and 1 <= rank <= n_gallery for rank in ranks), "Invalid ranks")
    return {**{f"R@{k}": 100 * sum(rank <= k for rank in ranks) / len(ranks) for k in (1, 5, 10)},
            "MRR@10": 100 * math.fsum(1 / rank if rank <= 10 else 0 for rank in ranks) / len(ranks)}


def replay_ranks(scores, gold, torch, device="cuda"):
    require(scores.ndim == 2 and len(gold) == scores.shape[0], "Matrix/gold dimensions differ")
    require(scores.dtype == torch.float32 and bool(torch.isfinite(scores).all()), "Invalid score values")
    require(all(type(row) is int and 0 <= row < scores.shape[1] for row in gold), "Invalid gold rows")
    order = torch.argsort(scores.to(device), dim=1, descending=True)
    target = torch.tensor(gold, device=device)[:, None]
    match = order.eq(target)
    require(bool(match.sum(1).eq(1).all()), "Gold row missing/duplicated")
    return match.to(torch.int64).argmax(1).add(1).cpu().tolist()


def replay_legacy_full_gallery_ranks(scores, scene_scores, gold, torch, device="cuda"):
    """Preserve07 ties: scene argsort first, then one-dimensional head sort.

    A direct 2D head argsort is not equivalent when scores tie. Every gallery
    image is retained; the first sort provides order only, never a shortlist.
    """
    require(scores.shape == scene_scores.shape and scores.ndim == 2 and
            scores.shape[0] == len(gold), "Head/scene matrix mismatch")
    require(scores.dtype == scene_scores.dtype == torch.float32 and
            bool(torch.isfinite(scores).all()) and bool(torch.isfinite(scene_scores).all()),
            "Invalid head/scene score values")
    require(all(type(row) is int and 0 <= row < scores.shape[1] for row in gold), "Invalid gold rows")
    order = torch.argsort(scene_scores.to(device), dim=1, descending=True)
    result = []
    for index, target in enumerate(gold):
        candidates = order[index]
        values = scores[index].to(device)[candidates]
        ranked = candidates[torch.argsort(values, descending=True)]
        location = torch.where(ranked == target)[0]
        require(location.numel() == 1, "Nonunique gold match")
        result.append(int(location.item()) + 1)
    return result


def expected_run_keys():
    return [(tag, seed, variant) for tag in TAGS for seed in range(5) for variant in VARIANTS]


def expected_evaluation_keys():
    return [(tag, seed, variant, label) for tag, seed, variant in expected_run_keys()
            for label in (SCALES if tag == "vitb32" else ("test",))]
