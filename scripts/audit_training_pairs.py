#!/usr/bin/env python3
"""Audit duplicate-target candidates under an isolated replay of the pair shuffle rule.

This is not a replay of observed historical batches: the trainer uses global
Python RNG state after external model/tokenizer calls. No model is trained.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
COUNT_KEYS = (
    "batches", "queries", "ordered_off_diagonal_candidates", "same_image_off_diagonal_candidates",
    "same_image_different_caption_candidates", "same_image_identical_caption_candidates",
    "batches_with_same_image", "batches_with_same_image_different_caption",
    "queries_with_same_image_off_diagonal", "queries_with_different_caption_same_image_off_diagonal",
)
HISTOGRAM_KEYS = ("query_image_multiplicity", "batch_image_group_multiplicity",
                  "query_exact_pair_multiplicity", "batch_exact_pair_group_multiplicity")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pairs(captions, ids):
    """Match the trainer's image-first insertion order and caption multiplicity."""
    if len(ids) != len(set(ids)):
        raise ValueError("Training IDs must be unique")
    included = set(ids)
    return [(image, caption) for image, texts in captions.items() if image in included for caption in texts]


def add_rates(record):
    ratio = lambda numerator, denominator: record[numerator] / record[denominator] if record[denominator] else 0.0
    record["same_image_off_diagonal_fraction"] = ratio("same_image_off_diagonal_candidates", "ordered_off_diagonal_candidates")
    record["different_caption_same_image_off_diagonal_fraction"] = ratio("same_image_different_caption_candidates", "ordered_off_diagonal_candidates")
    record["duplicate_image_batch_fraction"] = ratio("batches_with_same_image", "batches")
    record["different_caption_duplicate_image_batch_fraction"] = ratio("batches_with_same_image_different_caption", "batches")
    record["affected_query_fraction"] = ratio("queries_with_same_image_off_diagonal", "queries")
    record["different_caption_affected_query_fraction"] = ratio("queries_with_different_caption_same_image_off_diagonal", "queries")
    record["mean_query_ce_lower_bound_nats"] = ratio("query_log_image_multiplicity_sum", "queries")
    return record


def audit_batch(batch):
    images = Counter(image for image, _ in batch)
    exact_pairs = Counter(batch)
    same_image = sum(count * (count - 1) for count in images.values())
    identical = sum(count * (count - 1) for count in exact_pairs.values())
    different = same_image - identical
    image_query_hist = Counter()
    exact_query_hist = Counter()
    for count in images.values():
        image_query_hist[count] += count
    for count in exact_pairs.values():
        exact_query_hist[count] += count
    result = {
        "batches": 1, "queries": len(batch), "ordered_off_diagonal_candidates": len(batch) * (len(batch) - 1),
        "same_image_off_diagonal_candidates": same_image,
        "same_image_different_caption_candidates": different,
        "same_image_identical_caption_candidates": identical,
        "batches_with_same_image": int(same_image > 0),
        "batches_with_same_image_different_caption": int(different > 0),
        "queries_with_same_image_off_diagonal": sum(count for count in images.values() if count > 1),
        "queries_with_different_caption_same_image_off_diagonal": sum(
            count for (image, _), count in exact_pairs.items() if images[image] > count),
        "query_log_image_multiplicity_sum": sum(count * math.log(count) for count in images.values()),
        "histograms": {"query_image_multiplicity": dict(image_query_hist),
                       "batch_image_group_multiplicity": dict(Counter(images.values())),
                       "query_exact_pair_multiplicity": dict(exact_query_hist),
                       "batch_exact_pair_group_multiplicity": dict(Counter(exact_pairs.values()))},
    }
    return add_rates(result)


def summarize(records):
    result = {key: sum(record[key] for record in records) for key in COUNT_KEYS}
    result["query_log_image_multiplicity_sum"] = math.fsum(
        record["query_log_image_multiplicity_sum"] for record in records)
    result["histograms"] = {}
    for key in HISTOGRAM_KEYS:
        merged = Counter()
        for record in records:
            merged.update(record["histograms"][key])
        result["histograms"][key] = dict(sorted(merged.items()))
    return add_rates(result)


def simulate(pairs, batch_size=64, epochs=10, seed=0):
    if not pairs or min(batch_size, epochs) < 1:
        raise ValueError("Nonempty pairs and positive batch size/epochs are required")
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    batches = []
    epoch_records = []
    for epoch in range(1, epochs + 1):
        # Like training, shuffle the previous epoch's list in place, without reseeding.
        rng.shuffle(order)
        current = []
        for start in range(0, len(order), batch_size):
            batch = [pairs[index] for index in order[start:start + batch_size]]
            record = audit_batch(batch)
            record.update({"epoch": epoch, "batch_1based": len(current) + 1, "batch_size": len(batch)})
            current.append(record)
        epoch_summary = summarize(current)
        epoch_summary.update({"epoch": epoch, "last_batch_size": current[-1]["batch_size"],
                              "pair_order_sha256": hashlib.sha256(
                                  ",".join(map(str, order)).encode("ascii")).hexdigest()})
        epoch_records.append(epoch_summary)
        batches.extend(current)
    return {"aggregate": summarize(batches), "epochs": epoch_records, "batches": batches}


def write_csv(path, records):
    fields = [key for key in records[0] if key != "histograms"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=ROOT / "data/Text_unz/Text/Caption/train/Caption_train.csv")
    parser.add_argument("--ids", type=Path, default=ROOT / "data/ret/train_ids.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/training_pair_audit_20260916")
    args = parser.parse_args()
    trainer = ROOT / "code/train_saqre.py"
    sys.path.insert(0, str(ROOT / "code"))
    from train_saqre import load_captions

    captions = load_captions(args.captions)
    ids = json.loads(args.ids.read_text(encoding="utf-8"))
    pairs = build_pairs(captions, ids)
    simulation = simulate(pairs, args.batch_size, args.epochs, args.seed)
    image_counts = Counter(image for image, _ in pairs)
    pair_counts = Counter(pairs)
    random_calls = [{"call": f"random.{node.func.attr}", "line": node.lineno}
                    for node in ast.walk(ast.parse(trainer.read_text(encoding="utf-8")))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "random"]
    sources = {"captions": args.captions, "training_ids": args.ids, "trainer": trainer,
               "head": ROOT / "code/saqre_phys.py", "runtime": ROOT / "code/runtime.py",
               "launcher": ROOT / "code/run_ret.sh", "audit": Path(__file__),
               "focused_tests": ROOT / "tests/test_training_pair_audit.py",
               "python_random": Path(random.__file__)}
    denominator_notes = {
        "off_diagonal_fraction": "Ordered query-candidate positions: sum over retained batches of B*(B-1); diagonal positives excluded.",
        "same_image_different_caption": "Positions (a,b) with a != b, image_id[a] == image_id[b], caption_text[a] != caption_text[b]; exact stripped strings from the trainer parser.",
        "duplicate_batch_fraction": "Batches containing any repeated image ID / all retained batches, including the last partial batch.",
        "affected_query_fraction": "Query exposures with at least one off-diagonal candidate sharing the same image ID / all query exposures; each pair appears once per epoch.",
        "mean_query_ce_lower_bound_nats": "Sum over query exposures of natural log(m), divided by all query exposures; m is that query's image multiplicity in its batch.",
        "query_histograms": "Multiplicity -> number of query exposures (not unique dataset captions). Image multiplicity and exact (image,caption) pair multiplicity are reported separately.",
        "group_histograms": "Multiplicity -> number of distinct image or exact-pair groups counted separately within each batch.",
    }
    report = {
        "experiment": "Training-only caption-pair collision audit with isolated fixed-seed shuffle simulation",
        "sources": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in sources.items()},
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                    "rng": "stdlib random.Random(seed), one continuous stream; in-place shuffle of the previous order"},
        "configuration": {"seed": args.seed, "batch_size": args.batch_size, "epochs": args.epochs,
                          "drop_last": False, "max_pairs": 0, "caption_deduplication": False},
        "dataset": {"split_image_count": len(ids), "images_with_caption_pairs": len(image_counts),
                    "caption_pair_count": len(pairs), "distinct_exact_image_caption_pairs": len(pair_counts),
                    "repeated_exact_pair_rows": len(pairs) - len(pair_counts),
                    "caption_pairs_per_image_histogram": dict(sorted(Counter(image_counts.values()).items())),
                    "dataset_exact_pair_multiplicity_histogram": dict(sorted(Counter(pair_counts.values()).items())),
                    "initial_order": "Trainer load_captions insertion order: images first encountered in CSV, then all their nonempty captions in encounter order; restricted to training IDs.",
                    "missing_caption_image_ids": sorted(set(ids) - set(image_counts))},
        "trainer_python_random_calls": sorted(random_calls, key=lambda item: item["line"]),
        "denominators": denominator_notes,
        "aggregate": simulation["aggregate"], "epochs": simulation["epochs"],
        "ce_lower_bound": {"formula": "loss_a = -log(exp(S_aa/kappa)/sum_b exp(S_ab/kappa)) >= log(m_a)",
                           "reason": "With identical cached documents and the same query, all m_a same-image candidate scores are equal. The denominator therefore contains at least m_a copies of the positive numerator.",
                           "scope": "Single-positive caption-to-image cross entropy; query-conditioned scoring without stochastic per-document operations.",
                           "not_a_measurement": "This combinatorial lower bound is not an observed loss, a loss increase, or a retrieval-performance degradation estimate.",
                           "grouped_positive_identity": "L_diagonal - L_same_image_grouped_positive = log(m_a) exactly when the m_a same-document scores are identical for a fixed query.",
                           "gradient_implication": "The difference is parameter-independent, so both losses have identical gradients with respect to shared model parameters under this deterministic repeated-document condition. Merely summing duplicate-positive columns does not change optimization.",
                           "multiplicity_weighting": "The softmax denominator still weights each unique candidate image by its multiplicity; this differs from deduplicating candidate images or image-unique batching."},
        "limitations": ["No test data, model encoding, optimization, retrieval evaluation, or checkpoint inference is used.",
                        "These are independently simulated batches, not saved batches from any historical or current training run.",
                        "The trainer seeds global random before external backbone/tokenizer/text-encoding calls. Those calls may consume Python RNG state; an exact historical shuffle sequence cannot be certified without captured RNG state.",
                        "The simulation matches the trainer's pair construction, repeated in-place shuffle, batch slicing, and retained last batch under an isolated Random(0) stream (or the requested seed).",
                        "All official training IDs are used, matching the launcher default; this is not a fit-only hyperparameter trial.",
                        "Only same-image off-diagonal candidates are counted. Semantic equivalence between different image IDs is not assessed.",
                        "The audit quantifies duplicate-target exposure and an irreducible diagonal-loss floor, not contradictory gradients or a measured effect on model quality.",
                        "Same-image grouped-positive CE is gradient-equivalent to diagonal CE for deterministic repeated-document columns; no benefit from this relabeling is established. Deduplicated candidate weighting or image-unique sampling would require matched training experiments."],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output / "per_epoch.csv", simulation["epochs"])
    write_csv(args.output / "per_batch.csv", simulation["batches"])
    totals = report["aggregate"]
    readme = f"""# Training Pair Audit

Training-only, fixed-seed isolated shuffle simulation; not observed training batches.
Run: `.venv/bin/python scripts/audit_training_pairs.py`

Dataset: {len(pairs):,} caption-image rows, {len(image_counts):,} training images.
Configuration: {args.epochs} epochs, batch size {args.batch_size}, seed {args.seed}, last partial batch retained.
Total: {totals['batches']:,} batches and {totals['queries']:,} query exposures.

| Quantity | Numerator | Denominator | Fraction |
| --- | ---: | ---: | ---: |
| Same-image, different-caption off-diagonal candidates | {totals['same_image_different_caption_candidates']} | {totals['ordered_off_diagonal_candidates']} | {totals['different_caption_same_image_off_diagonal_fraction']:.10f} |
| Batches with a repeated image | {totals['batches_with_same_image']} | {totals['batches']} | {totals['duplicate_image_batch_fraction']:.10f} |
| Queries with a same-image off-diagonal candidate | {totals['queries_with_same_image_off_diagonal']} | {totals['queries']} | {totals['affected_query_fraction']:.10f} |

Mean query-level cross-entropy lower bound: {totals['mean_query_ce_lower_bound_nats']:.10f} nats, averaging log(m) over all queries.
This is a mathematical lower bound under identical cached candidates, not a measured loss or performance reduction.
For deterministic repeated-document columns, diagonal CE minus CE with all same-image positive columns grouped in the numerator is exactly log(m), a parameter-independent constant. The two losses therefore have identical gradients with respect to shared scoring parameters. This does not establish contradictory gradients or a benefit from merely grouping duplicate positives. Candidate-image multiplicities still weight the denominator, so image-unique batching or deduplication is a separate intervention.

The original trainer uses global `random.shuffle`, not a DataLoader. External model initialization/encoding runs after `random.seed` and before the first shuffle; its RNG consumption is not certified here. The audit uses local `random.Random(seed)` with the trainer's unchanged pair-construction and shuffle rules.

`summary.json` records source hashes, precise denominators, per-epoch results, multiplicity histograms, the bound, and limitations. `per_batch.csv` and `per_epoch.csv` expose the underlying counts. The source trainer and data are unchanged; no test set is read.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"dataset": report["dataset"], "aggregate": totals,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
