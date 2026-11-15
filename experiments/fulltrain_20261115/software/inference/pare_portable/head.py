"""Verbatim trainable head only; image extraction is deliberately outside this package."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAQREHead(nn.Module):
    """Lightweight trainable head over cached primitive embeddings.

    Learnable params (Eqs. type/reliability/attn/gates):
      - h_psi: scattering-type head (4 -> 3 logits)
      - U:     3 type prototypes in R^D  (type token)
      - log_beta: ENL-reliability scale  (Eq. reliability)
      - log_tau:  attention temperature  (Eq. attn)
      - log_s, logit_rho_min, log_lambda_rel: soft reliability sharpness,
        floor, and attention strength (Eqs. reliability/attn)
      - gate MLP: [q; agg] -> [g_c, g_f]  (Eq. gates)
      - type_scale: scalar on the type token contribution
    The CLIP backbone is frozen; only this head is trained with Eq. contrastive.
    """
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.dim = dim
        self.h_psi = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 3))
        self.U = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.log_beta = nn.Parameter(torch.tensor(1.0))   # beta = exp(.) ~ e
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.05)).float())
        # Differentiable reliability: the initial values are s=0.05,
        # rho_min=sigmoid(-2)=0.119, and lambda_rel=1.
        self.log_s = nn.Parameter(torch.tensor(np.log(0.05)).float())
        self.logit_rho_min = nn.Parameter(torch.tensor(-2.0))
        self.log_lambda_rel = nn.Parameter(torch.tensor(0.0))
        self.type_scale = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def _ablation_set(self):
        """Return optional component-ablation flags used by the experiment runner."""
        return getattr(self, "ablate", set())

    def reliability(self, mu, t, nu):
        """Soft ENL-calibrated reliability.

        ``mu``, ``t``, and ``nu`` are measured on the [0,1] grayscale proxy;
        the soft gate and non-zero floor preserve shadow/low-return cues.
        """
        beta = torch.exp(self.log_beta)
        s = torch.exp(self.log_s).clamp(1e-3, 1.0)
        rho_min = torch.sigmoid(self.logit_rho_min)
        soft_gate = torch.sigmoid((mu - nu) / s)
        structure = 1.0 - torch.exp(-beta * t * t)
        return rho_min + (1.0 - rho_min) * soft_gate * structure

    def type_posterior(self, typefeat):
        """Eq. type: pi = softmax h_psi(feat)."""
        return torch.softmax(self.h_psi(typefeat), dim=-1)

    def _augment(self, emb, typefeat):
        """Add scattering-type token to region embeddings."""
        pi = self.type_posterior(typefeat)            # (M,3)
        etype = pi @ self.U                            # (M,D)
        out = emb + self.type_scale * etype
        return F.normalize(out, dim=-1), pi

    def attend(self, q, emb, rho):
        """Eq. attn: speckle-modulated query attention. q:(D), emb:(M,D), rho:(M)."""
        tau = torch.exp(self.log_tau).clamp(1e-3, 1.0)
        lam = torch.exp(self.log_lambda_rel)
        logits = (emb @ q) / tau + lam * torch.log(rho + 1e-6)
        alpha = torch.softmax(logits, dim=-1)
        return (alpha.unsqueeze(-1) * emb).sum(0)

    def forward(self, q, doc):
        """Query-aligned doc embedding (Eq. fusion).

        q: (D) normalized query embedding.
        doc: dict of cached tensors on device:
            g (D); c_emb (Mc,D), c_mu (Mc), c_t (Mc), c_tf (Mc,4);
            f_emb (Mf,D), f_mu (Mf), f_t (Mf), f_tf (Mf,4); nu (scalar)
        Returns (D) normalized fused embedding.
        """
        g = doc["g"]
        nu = doc["nu"]
        ablate = self._ablation_set()
        strict_cfar = "cfar_strict" in ablate
        # coarse (regions)
        if doc["c_emb"].shape[0] > 0:
            if "ty" in ablate:
                c_aug = torch.nan_to_num(F.normalize(doc["c_emb"], dim=-1))
            else:
                c_tf = doc["c_tf"]
                if strict_cfar:
                    c_tf = c_tf.clone()
                    c_tf[:, 3] = 0.0
                c_aug, _ = self._augment(doc["c_emb"], c_tf)
            c_rho = (torch.ones_like(doc["c_mu"]) if "sr" in ablate else
                     self.reliability(doc["c_mu"], doc["c_t"], nu))
            c_att = self.attend(q, c_aug, c_rho)
        else:
            c_att = g
        # fine (scatterers)
        if not strict_cfar and doc["f_emb"].shape[0] > 0:
            if "ty" in ablate:
                f_aug = torch.nan_to_num(F.normalize(doc["f_emb"], dim=-1))
            else:
                f_aug, _ = self._augment(doc["f_emb"], doc["f_tf"])
            f_rho = (torch.ones_like(doc["f_mu"]) if "sr" in ablate else
                     self.reliability(doc["f_mu"], doc["f_t"], nu))
            f_att = g if "cfar" in ablate else self.attend(q, f_aug, f_rho)
        else:
            f_att = g
        # gates (Eq. gates): from q and aggregated doc
        agg = (torch.cat([g.unsqueeze(0), doc["c_emb"]], dim=0).mean(0)
               if strict_cfar else doc["agg"])
        gate_in = torch.cat([q, agg], dim=-1)
        gates = torch.sigmoid(self.gate(gate_in))
        g_c, g_f = gates[0], gates[1]
        fused = g_c * c_att + g_f * f_att + (1.0 - g_c - g_f).clamp(min=0.0) * g
        return F.normalize(fused, dim=-1)

    # ----------------- vectorized batched fusion (training/eval) -----------------
    def _augment_b(self, emb, tf):
        """emb:(B,M,D) tf:(B,M,4) -> augmented (B,M,D), pi (B,M,3)."""
        pi = torch.softmax(self.h_psi(tf), dim=-1)        # (B,M,3)
        etype = pi @ self.U                               # (B,M,D)
        return F.normalize(emb + self.type_scale * etype, dim=-1), pi

    def _attend_b(self, q, emb, rho, mask):
        """q:(D), emb:(B,M,D), rho:(B,M), mask:(B,M) bool -> (B,D)."""
        tau = torch.exp(self.log_tau).clamp(1e-3, 1.0)
        lam = torch.exp(self.log_lambda_rel)
        logits = (emb @ q) / tau + lam * torch.log(rho + 1e-6)  # (B,M)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return (alpha.unsqueeze(-1) * emb).sum(1)         # (B,D)

    def fuse_batch(self, q, docs):
        """Query-aligned fused embeddings for a batch of docs against one query.

        q: (D). docs: dict with batched tensors on device:
          g (B,D), agg (B,D), nu (B,);
          c_emb (B,Mc,D), c_mu (B,Mc), c_t (B,Mc), c_tf (B,Mc,4), c_mask (B,Mc);
          f_emb (B,Mf,D), f_mu (B,Mf), f_t (B,Mf), f_tf (B,Mf,4), f_mask (B,Mf)
        Returns (B,D) normalized fused embeddings.
        """
        B = docs["g"].shape[0]
        ablate = self._ablation_set()
        strict_cfar = "cfar_strict" in ablate
        nu = docs["nu"].unsqueeze(-1)                     # (B,1)
        beta = torch.exp(self.log_beta)
        s = torch.exp(self.log_s).clamp(1e-3, 1.0)
        rho_min = torch.sigmoid(self.logit_rho_min)

        if "ty" in ablate:
            c_aug = torch.nan_to_num(F.normalize(docs["c_emb"], dim=-1))
        else:
            c_tf = docs["c_tf"]
            if strict_cfar:
                c_tf = c_tf.clone()
                c_tf[..., 3] = 0.0
            c_aug, _ = self._augment_b(docs["c_emb"], c_tf)
        c_soft = torch.sigmoid((docs["c_mu"] - nu) / s)
        c_rho = rho_min + (1.0 - rho_min) * c_soft * (1.0 - torch.exp(-beta * docs["c_t"] ** 2))
        if "sr" in ablate:
            c_rho = torch.ones_like(c_rho)
        c_rho = c_rho * docs["c_mask"].float()
        c_att = self._attend_b(q, c_aug, c_rho, docs["c_mask"])

        if strict_cfar:
            f_att = docs["g"]
        else:
            if "ty" in ablate:
                f_aug = torch.nan_to_num(F.normalize(docs["f_emb"], dim=-1))
            else:
                f_aug, _ = self._augment_b(docs["f_emb"], docs["f_tf"])
            f_soft = torch.sigmoid((docs["f_mu"] - nu) / s)
            f_rho = rho_min + (1.0 - rho_min) * f_soft * (1.0 - torch.exp(-beta * docs["f_t"] ** 2))
            if "sr" in ablate:
                f_rho = torch.ones_like(f_rho)
            f_rho = f_rho * docs["f_mask"].float()
            f_att = self._attend_b(q, f_aug, f_rho, docs["f_mask"])
            f_att = torch.where(docs["f_mask"].any(1, keepdim=True), f_att, docs["g"])

        # if a doc has no regions/scatterers, fall back to scene
        c_att = torch.where(docs["c_mask"].any(1, keepdim=True), c_att, docs["g"])
        if "cfar" in ablate:
            f_att = docs["g"]

        qrep = q.unsqueeze(0).expand(B, -1)
        if strict_cfar:
            # Remove the fine-to-gate path as well as the attended branch.
            coarse = docs["c_emb"].masked_fill(~docs["c_mask"].unsqueeze(-1), 0.0)
            agg = (docs["g"] + coarse.sum(1)) / (1 + docs["c_mask"].sum(1, keepdim=True))
        else:
            agg = docs["agg"]
        gates = torch.sigmoid(self.gate(torch.cat([qrep, agg], dim=-1)))
        g_c = gates[:, :1]; g_f = gates[:, 1:2]
        rest = (1.0 - g_c - g_f).clamp(min=0.0)
        fused = g_c * c_att + g_f * f_att + rest * docs["g"]
        return F.normalize(fused, dim=-1)
