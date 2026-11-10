#!/usr/bin/env python3
"""PARE full head and image-relative grayscale feature extraction.

The frozen-backbone retrieval implementation provides:
  - a clipped local mean/standard-deviation proxy Lhat from low-CV windows;
  - an image-relative CV reference c_star = 1/sqrt(Lhat);
  - nonnegative excess CV t_r = max(cv_r - c_star, 0);
  - a learned soft reliability weight based on relative brightness and t_r;
  - connected-component proposals above 1.3*c_star, with a 2x2 fallback;
  - CFAR-inspired bright-center crops for the full-model comparison;
  - a learned three-prototype token mixed by softmax h_psi(feat_r);
  - query-conditioned region attention and scene/region/fine fusion.

These display-domain statistics are radiometry-inspired design features.
Lhat and the fifth-percentile brightness reference are not calibrated
sensor looks or a noise-equivalent floor. The learned token channels have
no assigned scattering classes. Function and checkpoint names containing
"physics", "type" or "scatterer" are retained for artifact compatibility.

Two parts:
  * grayscale statistics and primitive extraction (numpy / PIL);
  * SAQREHead (torch.nn.Module), trained on cached embeddings and features.
The strict simplified configuration is selected by the existing ablation
interface; removing fine crops alone is not its complete definition.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

EXTRACTOR_VERSION = "2026-10-07-float64-spatial-ties"


# =====================================================================
#  IMAGE-RELATIVE GRAYSCALE STATISTICS  (numpy)
# =====================================================================
def to_intensity(img: Image.Image) -> np.ndarray:
    """Pillow L-mode display values in [0,1]; not calibrated SAR amplitude."""
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return g


def _integral(x):
    x = np.asarray(x, dtype=np.float64)
    return np.pad(np.cumsum(np.cumsum(x, 0), 1), ((1, 0), (1, 0)))


def local_mean_std(g: np.ndarray, win: int):
    """Sliding-window local mean and std via integral images. Returns mu, sigma (same HxW, valid centers)."""
    # Promote before squaring as well as accumulation; float32 integral
    # subtraction can otherwise create negative means in dark image regions.
    values = np.asarray(g, dtype=np.float64)
    I1 = _integral(values)
    I2 = _integral(values * values)
    k = win
    # window sums over [i:i+k, j:j+k]
    def box(I):
        return (I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k])
    n = k * k
    s1 = box(I1)
    s2 = box(I2)
    mu = s1 / n
    var = np.clip(s2 / n - mu * mu, 0, None)
    sig = np.sqrt(var)
    return mu, sig  # shape (H-k+1, W-k+1)


def estimate_enl(g: np.ndarray, win: int = 16, homog_frac: float = 0.2) -> float:
    """Eq. enl: Lhat = median_{homogeneous w} (mu_w/sigma_w)^2.

    Eligible windows have mean > 0.02; select their lowest-CV quantile,
    including ties. This texture proxy is clipped to [1, 64], with fallback
    4 when fewer than eight windows are eligible; it is not sensor ENL.
    """
    mu, sig = local_mean_std(g, win)
    eps = 1e-4
    cv = sig / (mu + eps)
    valid = mu > 0.02  # exclude low display-domain means
    cvv = cv[valid]
    muv = mu[valid]
    sigv = sig[valid]
    if cvv.size < 8:
        return 4.0
    thr = np.quantile(cvv, homog_frac)
    sel = cvv <= thr
    ratio = (muv[sel] / (sigv[sel] + eps)) ** 2
    L = float(np.median(ratio))
    return float(np.clip(L, 1.0, 64.0))


def cv_map(g: np.ndarray, win: int = 9):
    mu, sig = local_mean_std(g, win)
    return sig / (mu + 1e-4), mu


def _cfar_background(g: np.ndarray, guard: int = 2, train: int = 6):
    """Float64 ring mean on valid centers of a nonnegative grayscale image."""
    k_out = train + guard
    win_out = 2 * k_out + 1
    win_in = 2 * guard + 1
    mu_out, _ = local_mean_std(g, win_out)   # (H-win_out+1, W-win_out+1)
    mu_in, _ = local_mean_std(g, win_in)
    # align centers: crop mu_in to mu_out's valid region
    off = k_out - guard
    mu_in_c = mu_in[off:off + mu_out.shape[0], off:off + mu_out.shape[1]]
    n_out = win_out ** 2
    n_in = win_in ** 2
    # background = (ring sum) / (n_out - n_in)
    bg = (mu_out * n_out - mu_in_c * n_in) / max(n_out - n_in, 1)
    return np.maximum(bg, 0.0)


def ca_cfar(g: np.ndarray, guard: int = 2, train: int = 6, pfa_factor: float = 2.5):
    """Cell-averaging CFAR-inspired detection with a nonnegative ring mean."""
    H, W = g.shape
    k_out = train + guard
    bg = _cfar_background(g, guard=guard, train=train)
    center_val = g[k_out:k_out + bg.shape[0], k_out:k_out + bg.shape[1]]
    det = center_val > (pfa_factor * bg + 1e-3)
    full = np.zeros((H, W), dtype=bool)
    full[k_out:k_out + det.shape[0], k_out:k_out + det.shape[1]] = det
    return full


def connected_components(mask: np.ndarray, min_area: int = 64):
    """4-connectivity CC labeling. Returns list of bbox (x0,y0,x1,y1)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    boxes = []
    slices = ndimage.find_objects(lab)
    for sl in slices:
        if sl is None:
            continue
        ys, xs = sl
        area = (ys.stop - ys.start) * (xs.stop - xs.start)
        if area >= min_area:
            boxes.append((xs.start, ys.start, xs.stop, ys.stop))
    return boxes


def type_features(patch_g: np.ndarray, cfar_frac: float) -> np.ndarray:
    """Display-domain descriptors for the learned token mixture.

    [peak-to-mean ratio, normalized 2nd moment, normalized 3rd moment, CFAR-positive fraction].
    """
    x = patch_g.ravel()
    mu = float(x.mean()) + 1e-4
    peak = float(x.max())
    m2 = float((x ** 2).mean()) / (mu ** 2)
    m3 = float((x ** 3).mean()) / (mu ** 3)
    p2m = peak / mu
    return np.array([np.log1p(p2m), np.log1p(m2), np.log1p(m3), cfar_frac], dtype=np.float32)


def region_proposals(g: np.ndarray, c_star: float, max_regions: int = 6,
                     win: int = 9):
    """Connected boxes where local CV exceeds 1.3*c_star.

    Keep the largest eligible boxes; if none survive, partition the image
    into four cells. No complementary homogeneous component is added.
    These boxes do not have semantic or scattering-class labels.
    """
    cvm, mu = cv_map(g, win)
    H, W = g.shape
    # pad cv map back to image size (valid region offset = win//2)
    pad = win // 2
    full_cv = np.zeros((H, W), dtype=cvm.dtype)
    full_cv[pad:pad + cvm.shape[0], pad:pad + cvm.shape[1]] = cvm
    hetero = full_cv > (1.3 * c_star)      # image-relative high-CV mask
    boxes = connected_components(hetero, min_area=max(64, (H * W) // 200))
    # rank by area, keep top
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    boxes = boxes[:max_regions]
    if not boxes:
        # fall back to a 2x2 coarse partition
        hh, ww = H // 2, W // 2
        boxes = [(0, 0, ww, hh), (ww, 0, W, hh), (0, hh, ww, H), (ww, hh, W, H)]
    return boxes


def scatterer_patches(g: np.ndarray, det: np.ndarray, max_sc: int = 8,
                      half: int = 12):
    """Select intensity-prioritized centers with deterministic spatial ties.

    Among equal-intensity candidates, first prefer the image center, then
    maximize the minimum squared distance to selected centers. Exact residual
    ties use (y, x) order. Keep the original square suppression and crop size.
    This version changes extraction and requires recaching and retraining.
    """
    H, W = g.shape
    ys, xs = np.where(det)
    if len(xs) == 0:
        return []
    values = g[ys, xs]
    eligible = np.ones(len(xs), dtype=bool)
    min_distance = np.full(len(xs), np.inf)
    center_distance = (xs - (W - 1) / 2.0) ** 2 + (ys - (H - 1) / 2.0) ** 2
    chosen = []
    while eligible.any() and len(chosen) < max_sc:
        ties = np.flatnonzero(eligible & (values == values[eligible].max()))
        # np.where enumerates (y,x), so argmin/argmax resolve only exact
        # remaining distance ties by that documented coordinate order.
        index = ties[np.argmax(min_distance[ties]) if chosen else np.argmin(center_distance[ties])]
        x, y = int(xs[index]), int(ys[index])
        chosen.append((x, y))
        eligible &= (np.abs(xs - x) > half) | (np.abs(ys - y) > half)
        distance = (xs.astype(np.float64) - x) ** 2 + (ys.astype(np.float64) - y) ** 2
        min_distance = np.minimum(min_distance, distance)
    boxes = []
    for x, y in chosen:
        x0 = max(0, x - half); y0 = max(0, y - half)
        x1 = min(W, x + half); y1 = min(H, y + half)
        boxes.append((x0, y0, x1, y1))
    return boxes


# =====================================================================
#  Primitive extraction (per image) -> dict of crops + grayscale features
# =====================================================================
def extract_primitives(img: Image.Image, max_regions=6, max_sc=8):
    """Return scene, region and CFAR-inspired crops with grayscale features.

    Output dict:
      scene: PIL, feature dict
      regions: list of (PIL crop, feat dict)
      scatterers: list of (PIL crop, feat dict)
    feat dict per primitive: mu, cv, t (excess CV), typefeat(4)
    image-level: L, c_star, nu
    """
    img = img.convert("RGB")
    g = to_intensity(img)
    L = estimate_enl(g)
    c_star = 1.0 / np.sqrt(L)
    nu = float(np.quantile(g, 0.05))         # image-relative brightness reference
    det = ca_cfar(g)
    cfar_frac_global = float(det.mean())

    def feat_for(box):
        x0, y0, x1, y1 = box
        sub = g[y0:y1, x0:x1]
        if sub.size == 0:
            sub = g
        mu = float(sub.mean())
        sig = float(sub.std())
        cv = sig / (mu + 1e-4)
        t = max(cv - c_star, 0.0)
        sub_det = det[y0:y1, x0:x1]
        cf = float(sub_det.mean()) if sub_det.size else 0.0
        tf = type_features(sub, cf)
        return {"mu": mu, "cv": cv, "t": t, "typefeat": tf}

    W, H = img.size
    scene_box = (0, 0, W, H)
    scene = (img, feat_for(scene_box))

    rboxes = region_proposals(g, c_star, max_regions=max_regions)
    regions = [(img.crop(b), feat_for(b)) for b in rboxes]

    sboxes = scatterer_patches(g, det, max_sc=max_sc)
    scatterers = [(img.crop(b), feat_for(b)) for b in sboxes]

    return {"scene": scene, "regions": regions, "scatterers": scatterers,
            "L": L, "c_star": float(c_star), "nu": nu,
            "cfar_frac": cfar_frac_global}


# =====================================================================
#  Trainable SA-QRE head  (torch)
# =====================================================================
class SAQREHead(nn.Module):
    """Lightweight trainable head over cached primitive embeddings.

    Learnable params (Eqs. type/reliability/attn/gates):
      - h_psi: token-mixture head (4 -> 32 -> 3 logits)
      - U:     3 learned prototypes in R^D, without scattering labels
      - log_beta: excess-CV reliability scale (Eq. reliability)
      - log_tau:  attention temperature  (Eq. attn)
      - log_s, logit_rho_min, log_lambda_rel: soft reliability sharpness,
        floor, and attention strength (Eqs. reliability/attn)
      - gate MLP: [q; agg] -> [g_c, g_f]  (Eq. gates)
      - type_scale: scalar on the learned-token contribution
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
        """Learned soft weight from image-relative grayscale statistics.

        ``mu``, ``t``, and ``nu`` use display-domain values. The positive
        floor avoids zero weights; no calibrated reliability is implied.
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
        """Add a learned prototype-mixture token to region embeddings."""
        pi = self.type_posterior(typefeat)            # (M,3)
        etype = pi @ self.U                            # (M,D)
        out = emb + self.type_scale * etype
        return F.normalize(out, dim=-1), pi

    def attend(self, q, emb, rho):
        """Eq. attn: reliability-weighted query attention. q:(D), emb:(M,D), rho:(M)."""
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
