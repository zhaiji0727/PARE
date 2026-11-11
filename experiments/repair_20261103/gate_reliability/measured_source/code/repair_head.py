"""Explicit head-only PARE repairs; frozen extraction inputs are unchanged.

Import SAQREHead from the separately copied, hash-bound original base_code.
Original modes preserve that head/factorial computation. New token/reliability
bounds and simplex fusion are explicit configuration choices, never defaults.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from saqre_phys import SAQREHead


@dataclass(frozen=True)
class RepairConfig:
    A: int = 1
    B: int = 1
    C: int = 1
    T: int = 1
    fusion: str = "original"
    reliability: str = "original"
    token_mode: str = "original"

    def __post_init__(self):
        for key in ("A", "B", "C"):
            value = getattr(self, key)
            if type(value) is not int or value not in (0, 1):
                raise ValueError(f"{key} must be the integer 0 or 1")
        if type(self.T) is not int or self.T not in (0, 1):
            raise ValueError("T must be the integer 0 or 1")
        if self.fusion not in ("original", "simplex"):
            raise ValueError("fusion must be original or simplex")
        if self.reliability not in ("original", "constant", "bounded"):
            raise ValueError("reliability must be original, constant, or bounded")
        if self.token_mode not in ("original", "bounded"):
            raise ValueError("token_mode must be original or bounded")

    def to_dict(self):
        return asdict(self)


class RepairHead(SAQREHead):
    """Same base initialization, with two deterministic prior-mixing scalars.

    The two extra scalars are present in every repair checkpoint. They consume
    no random draws and are unused unless reliability='bounded'. A separate
    checkpoint/schema must record RepairConfig; this is not an old full head.
    """
    NEW_PARAMETER_NAMES = ("prior_mix_coarse_logit", "prior_mix_fine_logit")
    TOKEN_DELTA_CAP = 0.5

    def __init__(self, dim, config=RepairConfig(), hidden=128):
        if not isinstance(config, RepairConfig):
            raise TypeError("config must be a RepairConfig")
        super().__init__(dim, hidden)
        self.config = config
        self.ablate = set()
        initial = math.log(1.0 / 9.0)
        self.prior_mix_coarse_logit = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.prior_mix_fine_logit = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    def load_base_state_dict(self, state):
        """Load exactly the old base tensors; preserve the two new initializers."""
        expected = set(self.state_dict()) - set(self.NEW_PARAMETER_NAMES)
        if set(state) != expected:
            raise ValueError("Base checkpoint keys differ from the original parameter schema")
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != set(self.NEW_PARAMETER_NAMES) or result.unexpected_keys:
            raise ValueError("Unexpected base checkpoint incompatibility")
        return result

    def forward(self, *args, **kwargs):
        raise RuntimeError("RepairHead uses fuse_batch; single-document forward is disabled")

    def prior_mix(self, branch):
        if branch == "coarse":
            logit = self.prior_mix_coarse_logit
        elif branch == "fine":
            logit = self.prior_mix_fine_logit
        else:
            raise ValueError("branch must be coarse or fine")
        return 0.5 * torch.sigmoid(logit)

    @staticmethod
    def _masked_softmax(logits, mask):
        # A genuinely empty crop row has zero weights and later uses the scene.
        # Avoid all-minus-infinity softmax so its backward pass stays finite.
        masked = logits.masked_fill(~mask, float("-inf"))
        safe = torch.where(mask.any(-1, keepdim=True), masked, torch.zeros_like(masked))
        return torch.softmax(safe, dim=-1).masked_fill(~mask, 0.0)

    def attention_diagnostics(self, q, emb, rho, mask, branch):
        """Return weights without changing state; also used by the bounded path.

        p is semantic attention; p_r includes the original lambda*log(rho+eps).
        For constant mode the original constant log term is retained exactly.
        The caller may detach diagnostics after computation when logging.
        """
        if branch not in ("coarse", "fine"):
            raise ValueError("branch must be coarse or fine")
        tau = torch.exp(self.log_tau).clamp(1e-3, 1.0)
        lam = torch.exp(self.log_lambda_rel)
        semantic_logits = (emb @ q) / tau
        if self.config.reliability == "constant":
            rho = torch.ones_like(rho) * mask.to(rho.dtype)
        prior_logits = semantic_logits + lam * torch.log(rho + 1e-6)
        p = self._masked_softmax(semantic_logits, mask)
        p_r = self._masked_softmax(prior_logits, mask)
        if self.config.reliability == "bounded":
            eta = self.prior_mix(branch)
            alpha = (1.0 - eta) * p + eta * p_r
        else:
            eta = None
            alpha = p_r
        return {"semantic": p, "prior": p_r, "alpha": alpha, "eta": eta}

    def _attend_variant(self, q, emb, rho, mask, branch):
        if self.config.reliability != "bounded":
            # Keep the original exact operation sequence, including the
            # constant log-attention term for reliability='constant'.
            return self._attend_b(q, emb, rho, mask)
        weights = self.attention_diagnostics(q, emb, rho, mask, branch)["alpha"]
        return (weights.unsqueeze(-1) * emb).sum(1)

    def token_diagnostics(self, emb, tf):
        """Describe raw and applied token deltas; no normalization of prototypes."""
        if not self.config.T:
            zero = torch.zeros_like(emb)
            return {"pi": None, "raw_delta": zero, "delta": zero,
                    "augmented": torch.nan_to_num(F.normalize(emb, dim=-1))}
        pi = torch.softmax(self.h_psi(tf), dim=-1)
        raw_delta = self.type_scale * (pi @ self.U)
        if self.config.token_mode == "bounded":
            delta = raw_delta / (1.0 + torch.linalg.vector_norm(raw_delta, dim=-1, keepdim=True)
                                 / self.TOKEN_DELTA_CAP)
        else:
            delta = raw_delta
        return {"pi": pi, "raw_delta": raw_delta, "delta": delta,
                "augmented": F.normalize(emb + delta, dim=-1)}

    def _augment_b(self, emb, tf):
        if not self.config.T:
            return torch.nan_to_num(F.normalize(emb, dim=-1)), None
        if self.config.token_mode == "original":
            return super()._augment_b(emb, tf)
        parts = self.token_diagnostics(emb, tf)
        return parts["augmented"], parts["pi"]

    def fusion_coefficients(self, logits):
        """Return [scene, coarse, fine]; simplex does not add an affine output."""
        if self.config.fusion == "simplex":
            return torch.softmax(torch.cat([torch.zeros_like(logits[..., :1]), logits], dim=-1), dim=-1)
        gates = torch.sigmoid(logits)
        g_c, g_f = gates[..., :1], gates[..., 1:2]
        rest = (1.0 - g_c - g_f).clamp(min=0.0)
        return torch.cat([rest, g_c, g_f], dim=-1)

    def fuse_batch(self, q, docs):
        if self._ablation_set():
            raise ValueError("Use RepairConfig; do not combine legacy ablation flags")
        batch = docs["g"].shape[0]
        config = self.config
        nu = docs["nu"].unsqueeze(-1)
        beta = torch.exp(self.log_beta)
        s = torch.exp(self.log_s).clamp(1e-3, 1.0)
        rho_min = torch.sigmoid(self.logit_rho_min)

        c_tf = docs["c_tf"]
        if not config.C:
            c_tf = c_tf.clone()
            c_tf[..., 3] = 0.0
        c_aug, _ = self._augment_b(docs["c_emb"], c_tf)
        c_soft = torch.sigmoid((docs["c_mu"] - nu) / s)
        c_rho = rho_min + (1.0 - rho_min) * c_soft * (1.0 - torch.exp(-beta * docs["c_t"] ** 2))
        if config.reliability == "constant":
            c_rho = torch.ones_like(c_rho)
        c_rho = c_rho * docs["c_mask"].float()
        c_att = self._attend_variant(q, c_aug, c_rho, docs["c_mask"], "coarse")

        if not config.A:
            f_att = docs["g"]
        else:
            f_aug, _ = self._augment_b(docs["f_emb"], docs["f_tf"])
            f_soft = torch.sigmoid((docs["f_mu"] - nu) / s)
            f_rho = rho_min + (1.0 - rho_min) * f_soft * (1.0 - torch.exp(-beta * docs["f_t"] ** 2))
            if config.reliability == "constant":
                f_rho = torch.ones_like(f_rho)
            f_rho = f_rho * docs["f_mask"].float()
            f_att = self._attend_variant(q, f_aug, f_rho, docs["f_mask"], "fine")
            f_att = torch.where(docs["f_mask"].any(1, keepdim=True), f_att, docs["g"])
        c_att = torch.where(docs["c_mask"].any(1, keepdim=True), c_att, docs["g"])

        if not config.B:
            coarse = docs["c_emb"].masked_fill(~docs["c_mask"].unsqueeze(-1), 0.0)
            agg = (docs["g"] + coarse.sum(1)) / (1 + docs["c_mask"].sum(1, keepdim=True))
        else:
            agg = docs["agg"]
        qrep = q.unsqueeze(0).expand(batch, -1)
        weights = self.fusion_coefficients(self.gate(torch.cat([qrep, agg], dim=-1)))
        rest, g_c, g_f = weights[:, :1], weights[:, 1:2], weights[:, 2:3]
        # Preserve the original order of the three additions.
        fused = g_c * c_att + g_f * f_att + rest * docs["g"]
        return F.normalize(fused, dim=-1)
