"""Final-evaluation numerical core, with caller-validated in-memory inputs.

No input paths, model choices, seeds, budgets or actual datasets are selected.
The eventual bound runner must validate checkpoints/caches/text identities first.
"""
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def synchronize(device):
    import torch
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def prepare_queries(first_vectors, extra_vectors, query_definition, *, official=True):
    from scoring_contract import assemble_all_caption_text
    first, extra, allrows = (query_definition[k] for k in ('first', 'extra', 'all'))
    require((len(first_vectors), len(extra_vectors)) == (len(first), len(extra)), 'Text vector/record count differs')
    mapping = query_definition['all_caption_row_map']
    require(len(mapping) == len(allrows), 'All-caption mapping length differs')
    for (origin, index), row in zip(mapping, allrows):
        require(origin in ('first', 'extra') and type(index) is int, 'Unknown text row mapping')
        source = query_definition[origin]
        require(0 <= index < len(source), 'Text row index out of bounds')
        reference = source[index]
        require(all(row[key] == reference[key] for key in
                    ('query_id', 'image_id', 'caption_sha256', 'caption_index_one_based', 'gold_gallery_row')),
                'Mapped text/query identity differs')
    if official:
        require((len(first), len(extra), len(allrows)) == (3688, 2502, 6190), 'Official query population differs')
    return assemble_all_caption_text(first_vectors, extra_vectors, mapping)


def evaluate_model(model, cache, first_vectors, extra_vectors, query_definition, gallery_ids,
                   *, query_batch_size=8, official=True):
    """Fresh all-caption scores; derive both modes from this same checkpoint."""
    import torch
    from pare_portable.api import score_queries
    from pare_portable.io import documents_from_cache
    from scoring_contract import report_both_query_modes
    require(query_batch_size == 8, 'Inherited numerical evaluation uses query chunks of eight')
    if official:
        require(len(gallery_ids) == 3688, 'Official gallery must contain all 3688 images')
    device = next(model.parameters()).device
    query_vectors = prepare_queries(first_vectors, extra_vectors, query_definition, official=official).to(device)
    documents = documents_from_cache(cache, gallery_ids, device)
    model.eval().requires_grad_(False)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        scores = score_queries(model, query_vectors, documents, query_batch_size=query_batch_size)
    synchronize(device)
    seconds = time.perf_counter() - start
    require(scores.dtype == torch.float32 and tuple(scores.shape) == (len(query_vectors), len(gallery_ids)),
            'Unexpected model score shape/dtype')
    report = report_both_query_modes(scores, query_definition['all'], query_definition['first'], gallery_ids)
    report.update(single_scoring_call_wall_seconds=seconds,
                  timing_scope='One complete portable all-caption scoring API call, including internal tensor validation, chunk concatenation and CPU score transfer; excludes input loading, encoding, query/document transfer and ranking.',
                  timing_warmup_calls=0, timing_repetitions=1,
                  timing_use='Execution record only; not a matched or steady-state latency benchmark.',
                  query_batch_size=query_batch_size, device=str(device), official_population=official)
    return scores, report


def evaluate_frozen_scene(cache, first_vectors, extra_vectors, query_definition, gallery_ids,
                          *, device='cpu', official=True):
    """Frozen scene cosine reference; normalization matches the existing baseline."""
    import torch
    from torch.nn import functional as F
    from scoring_contract import report_both_query_modes
    query_vectors = prepare_queries(first_vectors, extra_vectors, query_definition, official=official)
    if official:
        require(len(gallery_ids) == 3688, 'Official gallery must contain all 3688 images')
    index = {name: i for i, name in enumerate(cache['ids'])}
    require(len(index) == len(cache['ids']) and len(set(gallery_ids)) == len(gallery_ids), 'Duplicate image IDs')
    require(all(name in index for name in gallery_ids), 'Gallery image missing from cache')
    rows = torch.tensor([index[name] for name in gallery_ids], dtype=torch.int64)
    scene = F.normalize(cache['scene'][rows].to(device), dim=-1)
    queries = query_vectors.to(device)
    require(scene.dtype == queries.dtype == torch.float32 and bool(torch.isfinite(scene).all()), 'Baseline embeddings')
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        scores = (queries @ scene.T).cpu()
    synchronize(device)
    seconds = time.perf_counter() - start
    report = report_both_query_modes(scores, query_definition['all'], query_definition['first'], gallery_ids)
    report.update(single_scoring_call_wall_seconds=seconds,
                  timing_scope='One complete all-caption scene matrix product plus CPU score transfer; excludes input loading, encoding, embedding transfer and ranking.',
                  timing_warmup_calls=0, timing_repetitions=1,
                  timing_use='Execution record only; not a matched or steady-state latency benchmark.',
                  model='frozen_scene', device=str(device), official_population=official)
    return scores, report


def first_caption_gsd_groups(report, query_definition, grouping):
    """Use existing bins and full-gallery ranks; never filter the gallery itself."""
    from spacing_groups import summarize_rank_rows
    first = report['first-caption']
    require(len(first['ranks']) == len(query_definition['first']), 'GSD rank count differs')
    require(first['query_ids'] == [r['query_id'] for r in query_definition['first']], 'GSD query ordering differs')
    require(report['gallery_size'] == grouping['n_gallery_retained_for_all_subgroups'], 'GSD gallery differs')
    rows = [{'image_id': row['image_id'], 'ranks': {'evaluated_model': rank}}
            for row, rank in zip(query_definition['first'], first['ranks'])]
    return summarize_rank_rows(rows, grouping)
