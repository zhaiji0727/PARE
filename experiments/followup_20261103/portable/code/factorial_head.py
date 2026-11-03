"""Independent path ablations of the frozen PARE head; parameters are unchanged.

A: use the attended fine vector (1) or the scene vector (0), retaining g_f.
B: use the original scene/coarse/fine gate aggregate (1) or scene/coarse (0).
C: retain the coarse descriptor's detector fraction (1) or replace only it (0).
R and T correspond exactly to the frozen implementation's sr/ty switches.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from saqre_phys import SAQREHead


@dataclass(frozen=True)
class Factors:
    A: int = 1
    B: int = 1
    C: int = 1
    R: int = 1
    T: int = 1

    def __post_init__(self):
        if any(type(value) is not int or value not in (0, 1) for value in asdict(self).values()):
            raise ValueError('Every factor must be the integer 0 or 1')

    @property
    def key(self):
        return f'ABC{self.A}{self.B}{self.C}_R{self.R}T{self.T}'

    def to_dict(self):
        return asdict(self)


class FactorialHead(SAQREHead):
    """Same initialization/state_dict as SAQREHead, with explicit path controls."""

    def __init__(self, dim, factors=Factors(), hidden=128):
        super().__init__(dim, hidden)
        self.factors = factors
        self.ablate = set()

    def forward(self, *args, **kwargs):
        raise RuntimeError('FactorialHead uses fuse_batch; legacy single-document forward is disabled')

    def fuse_batch(self, q, docs):
        if self._ablation_set():
            raise ValueError('Use explicit A/B/C/R/T factors; do not combine legacy ablation flags')
        B = docs['g'].shape[0]
        factors = self.factors
        nu = docs['nu'].unsqueeze(-1)
        beta = torch.exp(self.log_beta)
        s = torch.exp(self.log_s).clamp(1e-3, 1.0)
        rho_min = torch.sigmoid(self.logit_rho_min)

        if not factors.T:
            c_aug = torch.nan_to_num(F.normalize(docs['c_emb'], dim=-1))
        else:
            c_tf = docs['c_tf']
            if not factors.C:
                c_tf = c_tf.clone()
                c_tf[..., 3] = 0.0
            c_aug, _ = self._augment_b(docs['c_emb'], c_tf)
        c_soft = torch.sigmoid((docs['c_mu'] - nu) / s)
        c_rho = rho_min + (1.0 - rho_min) * c_soft * (1.0 - torch.exp(-beta * docs['c_t'] ** 2))
        if not factors.R:
            c_rho = torch.ones_like(c_rho)
        c_rho = c_rho * docs['c_mask'].float()
        c_att = self._attend_b(q, c_aug, c_rho, docs['c_mask'])

        if not factors.A:
            f_att = docs['g']
        else:
            if not factors.T:
                f_aug = torch.nan_to_num(F.normalize(docs['f_emb'], dim=-1))
            else:
                f_aug, _ = self._augment_b(docs['f_emb'], docs['f_tf'])
            f_soft = torch.sigmoid((docs['f_mu'] - nu) / s)
            f_rho = rho_min + (1.0 - rho_min) * f_soft * (1.0 - torch.exp(-beta * docs['f_t'] ** 2))
            if not factors.R:
                f_rho = torch.ones_like(f_rho)
            f_rho = f_rho * docs['f_mask'].float()
            f_att = self._attend_b(q, f_aug, f_rho, docs['f_mask'])
            f_att = torch.where(docs['f_mask'].any(1, keepdim=True), f_att, docs['g'])

        c_att = torch.where(docs['c_mask'].any(1, keepdim=True), c_att, docs['g'])
        qrep = q.unsqueeze(0).expand(B, -1)
        if not factors.B:
            coarse = docs['c_emb'].masked_fill(~docs['c_mask'].unsqueeze(-1), 0.0)
            agg = (docs['g'] + coarse.sum(1)) / (1 + docs['c_mask'].sum(1, keepdim=True))
        else:
            agg = docs['agg']
        gates = torch.sigmoid(self.gate(torch.cat([qrep, agg], dim=-1)))
        g_c = gates[:, :1]
        g_f = gates[:, 1:2]
        rest = (1.0 - g_c - g_f).clamp(min=0.0)
        fused = g_c * c_att + g_f * f_att + rest * docs['g']
        return F.normalize(fused, dim=-1)
