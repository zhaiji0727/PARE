# PARE: fixed token-mechanism retraining diagnostic (9 November 2026)

This is a new, bounded diagnostic following the user's request for positive and solid core results. It does not promise positive outcomes. It does not change the completed 07 manuscript or 08 experimental records.

The only candidate is the 24-pixel model retaining original reliability and token augmentation with B0 (scene/coarse) gate context. Fifteen new heads cross seeds 0–4 with conditional descriptors, constant training descriptors, and token-off. Training uses the exact old 6,610-image/23,016-pair partition, original loss and ordering, ten epochs, lr0.002, temperature0.03, batch64, AdamW weight decay0.0001. No tuning or model selection is permitted.

The constant control replaces only the four inputs to h_psi by the per-branch arithmetic mean over valid training crops, accumulated in float64 and cast to float32. The original reliability calculation still receives original measurements. All learnable parameter shapes, random initialization, prototype vectors and fusion computations match the conditional head. Token-off bypasses token augmentation and keeps the remaining computation identical.

Both evaluation sets (918 and1,653 images) have been used before. They are exploratory development checks, not fresh blind tests. Use the complete within-split galleries, original first captions, the same ranking/tie rule and all R@1/5/10 and MRR@10 metrics. Keep every score matrix, rank, weight, initialization and epoch-order hash.

Before other training, the conditional seed0 run must reproduce the prior corresponding B0/original-reliability head and both rank vectors. Small synthetic/unit tests and independent source review precede the immutable code/input freeze. A failed anchor is investigated as an implementation compatibility issue, not treated as model improvement.

Four primary comparisons are conditional minus constant-descriptor and conditional minus token-off, each on the two evaluation sets. The statistical unit is five paired training seeds on fixed datasets. Report every seed delta, mean, sample SD, ordinary95% t interval and a Bonferroni family-wise95% interval across these four contrasts (individual98.75% interval, df4). These intervals rely on a small-sample distributional approximation and say nothing about unseen-dataset generalization.

For a strong within-dataset evidence flag, require mean R@10 delta at least0.5percentage points, all five seed deltas positive, the family-wise interval lower bound above zero, and nonnegative mean MRR@10 delta. A claim against a control requires both splits to meet this rule. Failure means evidence is insufficient under this predeclared rule; it does not prove equality or absence of an effect. No result may change the rule, metric, seeds, or splits.

Conditional versus token-off can support the module's aggregate effect. Conditional versus constant-descriptor addresses the contribution of regional conditioning within this training recipe. Neither identifies true scattering classes, independent fine-branch effects, or physical GSD invariance. No A0 experiment is included. All historical negative results remain available.

The JSON is the exact machine-readable specification. Source/input freeze follows peer review. No new manuscript claims or five-backbone expansion are authorized by the diagnostic flag alone.

The conditional heads for seeds0–2 have historical anchor results. This post-hoc design does not reset that history. Five samples cannot validate the normality assumption, and the t-based flags are not distribution-free or preregistered confirmatory proof.
