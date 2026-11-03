#!/usr/bin/env python3
"""Small independent regression against released sampling/ranking and model."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_multik as new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code-dir", required=True)
    p.add_argument("--scoring-helper", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sys.path.insert(0, args.code_dir)
    import eval_retrieval_phys as old
    import eval_retrieval_baselines as old_baselines
    from saqre_phys import SAQREHead
    from train_saqre import move_docs
    helper = new.load_helper(args.scoring_helper)
    torch.manual_seed(41)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    qn, n, d, mc, mf = 7, 13, 512, 6, 8
    queries = F.normalize(torch.randn(qn, d, device=device), dim=-1)
    cache = {"ids": [f"image-{i}" for i in range(n)], "D": d, "Mc": mc, "Mf": mf,
             "scene": F.normalize(torch.randn(n, d), dim=-1), "agg": torch.randn(n, d), "nu": torch.rand(n)}
    for prefix, count in (("c", mc), ("f", mf)):
        counts = torch.arange(n) % (count + 1)
        mask = torch.arange(count)[None] < counts[:, None]
        cache[prefix + "_n"] = counts
        for suffix, value in (("emb", F.normalize(torch.randn(n, count, d), dim=-1)),
                              ("mu", torch.rand(n, count)), ("t", torch.rand(n, count)),
                              ("tf", torch.rand(n, count, 4))):
            expanded = mask
            while expanded.ndim < value.ndim:
                expanded = expanded.unsqueeze(-1)
            cache[prefix + "_" + suffix] = value.masked_fill(~expanded, 0)
    gold = [0, 4, 2, 12, 3, 6, 9]
    records = [{"query_id": f"q{i}", "gold_gallery_row": g} for i, g in enumerate(gold)]
    report = {"device": str(device), "cases": []}

    # Independent old-code sampling loop; compare every candidate, not only metrics.
    generator = torch.Generator(device="cpu").manual_seed(0)
    expected = []
    for g in gold:
        rows = []
        for _ in range(5):
            sample = torch.randperm(n - 1, generator=generator)[:4]
            sample = sample + (sample >= g).long()
            rows.append([g] + sample.tolist())
        expected.append(rows)
    pools = new.make_random_pools(gold, n, 5, 5, 0)
    np.testing.assert_array_equal(pools, expected)

    # Ties, zero vectors, misses, and zero-inclusion conditional metrics.
    for tied in (False, True):
        scene = torch.randint(-1, 2, (qn, n), device=device).float() if tied else torch.randn(qn, n, device=device)
        values = torch.randint(-1, 2, (qn, n), device=device).float() if tied else torch.randn(qn, n, device=device)
        order = torch.argsort(scene, dim=1, descending=True).cpu().numpy()
        hard = new.make_hard_pools(order, gold, 7)
        reports = new.derive_head_reports(values, scene, records, [5, 8, 13], pools, hard, order, {})
        for ki, k in enumerate((5, 8, 13)):
            expected_cond, expected_end, incl = [], [], []
            for qi, g in enumerate(gold):
                candidates = order[qi, :k].tolist()
                incl.append(int(g in candidates))
                r = old.rank_of_gold(values[qi, candidates], candidates, g) if g in candidates else None
                expected_end.append(r if r is not None else float("inf"))
                if r is not None:
                    expected_cond.append(r)
                assert reports[k]["queries"][qi]["natural_PARE_rank"] == r
            actual = reports[k]["natural_scene_shortlist"]
            assert actual["PARE_end_to_end"] == old.metrics(expected_end)
            assert actual["PARE_conditional_on_inclusion"] == old.metrics(expected_cond)
            assert actual["n_natural_inclusions"] == sum(incl)
            assert ("controlled_random_pools" in reports[k]) == (ki == 0)
        for qi, g in enumerate(gold):
            expected_hard = [g] + [int(x) for x in order[qi] if x != g][:6]
            assert hard[qi].tolist() == expected_hard
            for pool in pools[qi]:
                candidates = pool.tolist()
                assert new.rank_of_gold(values[qi, candidates], candidates, g) == old.rank_of_gold(values[qi, candidates], candidates, g)
                assert new.rank_of_gold(values[qi, candidates], candidates, g) == old_baselines.rank_in_pool(values[qi], candidates, g)
        report["cases"].append({"case": "old_sampling_and_ranking", "ties": tied, "passed": True})
    zero_order = np.asarray([[i for i in range(n) if i != g] + [g] for g in gold], dtype=np.int32)
    no_inclusion = new.derive_head_reports(values, scene, records, [5], pools, hard, zero_order, {})[5]
    assert no_inclusion["natural_scene_shortlist"]["PARE_conditional_on_inclusion"] is None
    assert no_inclusion["natural_scene_shortlist"]["PARE_end_to_end"] == old.metrics([float("inf")] * qn)
    json.dumps(no_inclusion, allow_nan=False)

    for flags in (set(), {"sr", "ty"}, {"cfar_strict"}, {"sr", "ty", "cfar"}):
        head = SAQREHead(d).to(device).eval()
        head.ablate = flags
        with torch.no_grad():
            docs = move_docs(cache, torch.arange(n), device)
            expected_scores = torch.stack([head.fuse_batch(query, docs) @ query for query in queries]).cpu()
        scores = new.score_head_matrix(head, queries, cache, move_docs, helper.score_matrix, device, 3, 5)
        torch.testing.assert_close(scores, expected_scores, atol=2e-6, rtol=2e-5)
        report["cases"].append({"case": "blocked_vmap_model", "flags": sorted(flags),
                                 "max_error": float((scores - expected_scores).abs().max())})

    baseline = new.score_baselines(queries, cache, device, 5)
    explicit = {name: [] for name in baseline}
    with torch.no_grad():
        for index in range(n):
            vectors = torch.cat((F.normalize(cache["scene"][index:index + 1], dim=-1),
                                 F.normalize(cache["c_emb"][index, :cache["c_n"][index]], dim=-1),
                                 F.normalize(cache["f_emb"][index, :cache["f_n"][index]], dim=-1))).to(device)
            explicit["scene"].append(queries @ vectors[0])
            explicit["mean_pool"].append(queries @ F.normalize(vectors.mean(0), dim=-1))
            explicit["MaxSim"].append((queries @ vectors.T).max(1).values)
    for name in explicit:
        expected_scores = torch.stack(explicit[name], dim=1).cpu()
        torch.testing.assert_close(baseline[name], expected_scores, atol=2e-6, rtol=2e-5)
        report["cases"].append({"case": "explicit_valid_crop_baseline", "name": name,
                                 "max_error": float((baseline[name] - expected_scores).abs().max())})
    report["passed"] = True
    new.write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
