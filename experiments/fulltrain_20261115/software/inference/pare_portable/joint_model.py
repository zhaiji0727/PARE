"""Exploratory shared semantic adaptation around the original complete PARE head.

The original head mathematics is imported from a hash-bound source. This is a
new candidate, not a claim to recover the exact historical implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .head import SAQREHead
from .scoring import score_matrix
from .adapter import SceneResidualAdapter

VARIANTS = ['adapter', 'original_full', 'joint_full', 'joint_constant',
            'joint_no_token', 'joint_uniform', 'joint_paths_off', 'joint_scene_control']


class JointPARE(nn.Module):
    def __init__(self, dimension, variant, coarse_mean, fine_mean):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        # Always construct both in the same order to bind identical initial tensors.
        self.adapter = SceneResidualAdapter(dimension)
        self.physics = SAQREHead(dimension)
        self.register_buffer('coarse_mean', coarse_mean.clone())
        self.register_buffer('fine_mean', fine_mean.clone())
        self.physics.ablate = {
            'joint_no_token': {'ty'}, 'joint_uniform': {'sr'},
            'joint_paths_off': {'cfar_strict'}, 'joint_scene_control': {'sr'},
        }.get(variant, set())

    def adapt_docs(self, docs):
        out = dict(docs)
        out['g'] = self.adapter.encode(docs['g'])
        for branch in ['c', 'f']:
            mask = docs[branch+'_mask']
            emb = self.adapter.encode(docs[branch+'_emb'])
            # A biased adapter maps zero padding to nonzero vectors.
            out[branch+'_emb'] = emb.masked_fill(~mask[..., None], 0)
        if self.variant in ['joint_constant', 'joint_scene_control']:
            out['c_tf'] = self.coarse_mean.expand_as(docs['c_tf'])
            out['f_tf'] = self.fine_mean.expand_as(docs['f_tf'])
        if self.variant == 'joint_scene_control':
            for branch in ['c', 'f']:
                out[branch+'_emb'] = out['g'][:,None,:].expand_as(docs[branch+'_emb'])
                out[branch+'_mask'] = torch.ones_like(docs[branch+'_mask'])
                out[branch+'_mu'] = torch.zeros_like(docs[branch+'_mu'])
                out[branch+'_t'] = torch.zeros_like(docs[branch+'_t'])
            out['nu'] = torch.zeros_like(docs['nu'])
        count = 1 + out['c_mask'].sum(1,keepdim=True) + out['f_mask'].sum(1,keepdim=True)
        out['agg'] = (out['g'] + out['c_emb'].sum(1) + out['f_emb'].sum(1))/count
        return out

    def forward(self, queries, docs):
        if self.variant == 'adapter':
            return self.adapter(queries, docs['g'])
        if self.variant == 'original_full':
            return score_matrix(self.physics, queries, docs)
        return score_matrix(self.physics, queries, self.adapt_docs(docs))

    def active_parameters(self):
        if self.variant == 'adapter':
            return self.adapter.parameters()
        if self.variant == 'original_full':
            return self.physics.parameters()
        return self.parameters()
