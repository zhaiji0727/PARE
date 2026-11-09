"""Exact original training/evaluation kernels; population counts are supplied in settings."""
import hashlib
import json
import random
import time

def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def train_one(head, cache, pairs, cap2vec, device, seed, settings, expected_orders, progress):
    import torch
    import torch.nn.functional as F
    from batch_scoring import score_matrix
    from train_saqre import move_docs
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    order = list(range(len(pairs)))
    rng = random.Random(seed)
    history = []
    start = time.monotonic()
    for epoch in range(settings['epochs']):
        rng.shuffle(order)
        digest = object_sha(order)
        if digest != expected_orders[epoch]:
            raise RuntimeError('Within-seed pair shuffle diverged')
        head.train()
        total_loss, batches = 0., 0
        for offset in range(0, len(order), settings['batch_size']):
            selected = [pairs[index] for index in order[offset:offset + settings['batch_size']]]
            rows = torch.tensor([row for row, _ in selected])
            q = torch.stack([cap2vec[caption] for _, caption in selected]).to(device)
            scores = score_matrix(head, q, move_docs(cache, rows, device))
            loss = F.cross_entropy(scores / settings['kappa'], torch.arange(len(selected), device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        if batches != settings['batches_per_epoch'] or len(selected) != settings['tail_batch_pairs']:
            raise RuntimeError('Training batch/tail count mismatch')
        row = {'epoch': epoch + 1, 'batches': batches, 'pairs': len(pairs),
               'tail_batch_pairs': len(selected), 'pair_order_sha256': digest,
               'mean_batch_loss': total_loss / batches, 'elapsed_seconds': time.monotonic() - start}
        history.append(row)
        progress(history)
        print(json.dumps({'epoch_progress': row}), flush=True)
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise FloatingPointError('Non-finite final head')
    return history


def validate_one(head, cache, gallery, queries, cap2vec, device):
    import torch
    from batch_scoring import score_matrix
    from train_saqre import move_docs
    head.eval()
    gallery_index = {row: index for index, row in enumerate(gallery)}
    gold = [gallery_index[row] for row, _ in queries]
    query_vectors = torch.stack([cap2vec[caption] for _, caption in queries]).to(device)
    matrix = torch.empty((len(queries), len(gallery)), dtype=torch.float32)
    ranks = []
    with torch.no_grad():
        for start in range(0, len(queries), 16):
            end = min(start + 16, len(queries))
            scores = torch.empty((end - start, len(gallery)), device=device)
            for gs in range(0, len(gallery), 1024):
                ge = min(gs + 1024, len(gallery))
                docs = move_docs(cache, torch.tensor(gallery[gs:ge]), device)
                scores[:, gs:ge] = score_matrix(head, query_vectors[start:end], docs)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Non-finite validation score')
            ordered = torch.argsort(scores, dim=1, descending=True)
            targets = torch.tensor(gold[start:end], device=device).unsqueeze(1)
            ranks.extend((ordered == targets).nonzero(as_tuple=True)[1].add(1).tolist())
            matrix[start:end] = scores.cpu()
    if len(ranks) != len(queries) or not all(1 <= rank <= len(gallery) for rank in ranks):
        raise RuntimeError('Invalid validation ranks')
    count = len(ranks)
    metrics = {f'R@{k}': 100 * sum(rank <= k for rank in ranks) / count for k in (1, 5, 10)}
    metrics['MRR@10'] = 100 * sum(1 / rank if rank <= 10 else 0 for rank in ranks) / count
    return matrix, ranks, metrics, gold

