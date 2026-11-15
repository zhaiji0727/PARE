"""Pure in-memory scoring helpers. No input discovery, file IO or model selection.

An eventual final evaluator must independently validate and bind every tensor,
query identity, model, checkpoint and input before calling these helpers.
"""
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assemble_all_caption_text(first, extra, row_map):
    """Interleave first and extra caption vectors using the explicit query map.

    Does not concatenate populations, normalize vectors, deduplicate captions,
    or load prior model predictions. Every source row must occur exactly once.
    """
    import torch
    require(first.ndim == extra.ndim == 2 and first.shape[1] == extra.shape[1], 'Text dimensions')
    require(first.dtype == extra.dtype == torch.float32, 'Text dtype must be float32')
    require(first.device == extra.device, 'Text device differs')
    require(bool(torch.isfinite(first).all()) and bool(torch.isfinite(extra).all()), 'Nonfinite text')
    source = {'first': first, 'extra': extra}
    expected = {(key, i) for key, tensor in source.items() for i in range(len(tensor))}
    require(all(isinstance(r, (list, tuple)) and len(r) == 2 and r[0] in source
                and type(r[1]) is int for r in row_map), 'Invalid row map')
    keys = [tuple(r) for r in row_map]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Text map must cover each source row once')
    require(len(keys) > 0, 'Empty query set')
    return torch.stack([source[k][i] for k, i in keys])


def summarize_ranks(ranks, gallery_size):
    require(type(gallery_size) is int and gallery_size > 0, 'Gallery size')
    require(len(ranks) > 0 and all(type(r) is int and 1 <= r <= gallery_size for r in ranks), 'Invalid ranks')
    return {**{f'R@{k}': 100 * sum(r <= k for r in ranks) / len(ranks) for k in (1, 5, 10)},
            'MRR@10': 100 * math.fsum(1 / r if r <= 10 else 0 for r in ranks) / len(ranks)}


def stable_gallery_ranks(scores, gold):
    """Rank CPU float32 scores by descending score, then ascending gallery row.

    Two independent calculations (stable argsort and greater/tie counts) must
    agree. No tolerance is used to turn near-equal scores into exact ties.
    """
    import torch
    require(scores.ndim == 2 and scores.dtype == torch.float32 and scores.device.type == 'cpu',
            'Scores must be a CPU float32 matrix')
    require(scores.shape[0] > 0 and scores.shape[1] > 0 and bool(torch.isfinite(scores).all()), 'Score population')
    require(len(gold) == len(scores) and all(type(g) is int and 0 <= g < scores.shape[1] for g in gold), 'Gold rows')
    targets = torch.tensor(gold, dtype=torch.int64)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    sorted_ranks = (order == targets[:, None]).nonzero(as_tuple=True)[1] + 1
    correct = scores[torch.arange(len(scores)), targets][:, None]
    equal = scores == correct
    earlier = torch.arange(scores.shape[1])[None, :] < targets[:, None]
    count_ranks = 1 + (scores > correct).sum(1) + (equal & earlier).sum(1)
    require(torch.equal(sorted_ranks, count_ranks), 'Independent rank calculations disagree')
    return {'ranks': sorted_ranks.tolist(),
            'gold_score_tie_group_sizes': equal.sum(1).tolist(),
            'queries_with_tied_gold_score': int((equal.sum(1) > 1).sum()),
            'tie_rule': 'descending_float32_score_then_ascending_gallery_row; CPU stable argsort'}


def report_both_query_modes(scores, all_records, first_records, gallery_ids):
    """Summarize only scores from ONE caller-bound model/checkpoint.

    first_records is explicitly mapped by identity, not assumed to be a prefix.
    Records must agree on image, caption identity, and gold gallery row.
    """
    require(len(scores) == len(all_records) and scores.shape[1] == len(gallery_ids), 'Matrix/identity shape')
    require(len(gallery_ids) == len(set(gallery_ids)), 'Duplicate gallery IDs')
    index = {}
    for i, row in enumerate(all_records):
        require(row['query_id'] not in index, 'Duplicate query ID')
        require(type(row['gold_gallery_row']) is int and 0 <= row['gold_gallery_row'] < len(gallery_ids), 'Gold index')
        require(gallery_ids[row['gold_gallery_row']] == row['image_id'], 'Image/gold mismatch')
        index[row['query_id']] = i
    require(len(first_records) > 0, 'Missing first-caption records')
    first_idx = []
    for row in first_records:
        require(row['query_id'] in index and row['caption_index_one_based'] == 1, 'Invalid first-caption identity')
        i = index[row['query_id']]
        require(all(row[k] == all_records[i][k] for k in
                    ('image_id', 'caption_sha256', 'gold_gallery_row', 'caption_index_one_based')),
                'First/all caption identity mismatch')
        first_idx.append(i)
    require(len(first_idx) == len(set(first_idx)) and
            set(first_idx) == {i for i, row in enumerate(all_records) if row['caption_index_one_based'] == 1},
            'First-caption selection incomplete/duplicated')
    ranked = stable_gallery_ranks(scores, [r['gold_gallery_row'] for r in all_records])
    first_ranks = [ranked['ranks'][i] for i in first_idx]
    return {'schema': 'pare.same-checkpoint-first-and-all-caption-ranks.v1',
            'gallery_size': len(gallery_ids), 'tie_rule': ranked['tie_rule'],
            'all-captions': {'n_queries': len(all_records), 'ranks': ranked['ranks'],
                             'query_ids': [r['query_id'] for r in all_records],
                             'metrics_percent': summarize_ranks(ranked['ranks'], len(gallery_ids)),
                             'gold_score_tie_group_sizes': ranked['gold_score_tie_group_sizes']},
            'first-caption': {'n_queries': len(first_records), 'ranks': first_ranks,
                              'query_ids': [r['query_id'] for r in first_records],
                              'metrics_percent': summarize_ranks(first_ranks, len(gallery_ids)),
                              'source_all_caption_rows': first_idx},
            'weighting': 'Every caption row equal; no text deduplication; one image ID positive per query.'}
