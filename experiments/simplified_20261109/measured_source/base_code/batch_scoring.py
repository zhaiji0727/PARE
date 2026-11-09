"""Batch the unchanged PARE query-to-document score with torch.vmap.

This wrapper evaluates the existing SAQREHead.fuse_batch implementation for
each query. It does not change the model, loss, masks, candidate sets or data.
"""
import hashlib
from pathlib import Path
import torch


def score_matrix(head, queries, docs, chunk_size=None):
    """Return Q x N scores, equivalent to stacking the original query loop."""
    if queries.ndim != 2 or queries.shape[1] != docs['g'].shape[1]:
        raise ValueError('Expected Q x D queries and N x D documents')
    return torch.vmap(lambda query: head.fuse_batch(query, docs) @ query,
                      chunk_size=chunk_size)(queries)


def scoring_provenance():
    return {'batch_scoring_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'batch_scoring_backend': 'torch.vmap over unchanged SAQREHead.fuse_batch',
            'batch_scoring_torch': str(torch.__version__)}
