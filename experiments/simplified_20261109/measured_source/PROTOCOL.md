# Matched evaluation of a simplified PARE candidate

Protocol date: 2026-11-09, Asia/Shanghai. This protocol precedes the new runs but follows prior examination of the same benchmark. It is not preregistration of an unseen test.

## Fixed candidate and comparisons

Use the original SAQREHead implementation and preserve parameter shapes, initialization, reliability, token projection and original two-sigmoid fusion. The candidate is the existing `cfar_strict` configuration: exclude fine attention, exclude fine features from the gate aggregate, and fix the coarse descriptor's detector-fraction coordinate to zero. The fine fusion slot aliases the scene. Its algebra is `normalize(gc*coarse + [gf + max(1-gc-gf,0)]*scene)`, with the original operation order retained. No new one-gate, softmax, constant-token or reliability-off model is introduced.

Compare three fixed variants: original full (`[]`), simplified (`[cfar_strict]`), and simplified without token augmentation (`[cfar_strict,ty]`). Use all five existing backbones and seeds 0,1,2,3,4: 75 fresh heads. Each backbone retains the original validation-selected hyperparameters. Use 9181 training images, 31945 caption pairs, 10 epochs, batch size 64 (500 batches, tail 9), AdamW with weight decay 0.0001 and float32. No warm starts, new sweep, selected epoch, selected seed, or outcome-triggered extension.

## Pairing and reproducibility

Within each backbone/seed, all three variants share initial parameters and minibatch ordering. Text embeddings are computed from the sorted unique TRAIN captions alone with the original batching. Reset the head RNG after embedding. Ten fresh seed-0 full/strict heads are compatibility anchors and count toward the 75. Require agreement with the corresponding earlier weights and recorded rankings; investigate discrepancies rather than weakening criteria.

Freeze source, analysis, this protocol and exact input hashes before training. Keep the prior files immutable. Preserve every run and any failed attempt. Independent verification checks weights, all score-derived rankings, identities, metrics and provenance.

## Evaluation and Reviewer 2

Evaluate the original first caption for each of 3688 images against the entire 3688-image gallery. No ground-truth injection or selected candidate subset. Retain all 75 evaluations. Also evaluate the B32 heads (all three variants and five seeds) on the existing 1.5x and 2x image-spacing stress caches: 30 further evaluations, for 105 total. Reuse original-scale main results, recompute fixed metadata groups from the new rankings, and report all scales/groups. A scene-only baseline uses identical frozen representations and is not represented as five independently trained models.

The older scale and GSD results belong to the original full model. They must not be relabeled as simplified-model results. Dataset metadata coverage can be reused. Window sizes of the original full model remain 24x24 pixels; no 48x48 experiment is introduced.

## Statistics and interpretation

Primary metric: R@10 in percentage points. Secondary: R@1, R@5 and MRR@10. Compute within-seed differences first, then report all five differences, their mean and sample SD. Ten primary contrasts are fixed: simplified minus full and simplified minus simplified-token-off for each backbone. Report ordinary 95% and Bonferroni family-95% Student-t intervals (each 99.5%, df=4, quantile 0.9975).

A comparison supports the phrase “stable positive improvement under this protocol” only if all five R@10 differences are strictly positive, the Bonferroni interval lower endpoint is above zero, and the mean MRR@10 difference is nonnegative. A claim across all backbones requires that each backbone independently satisfy this rule. Failure means insufficient support for that claim, not no effect or equivalence. All results still appear. Scale and subgroup results are descriptive, not additional independent confirmatory tests.

Intervals describe training-seed variability on fixed data under a small-sample normality approximation. These test images have already been examined and helped motivate the candidate. New seeds do not turn them into unseen data. The manuscript must disclose this development history, distinguish the original full method from the simplified version, retain older negative results, and avoid attributing token gains specifically to regional statistical conditioning without direct supporting evidence.

## Manuscript and stopping rule

Complete the fixed experiment regardless of metric signs. Update method, tables, abstract, response letter, supplementary history and code to describe the exact evaluated model. Narrow unsupported claims. Do not claim end-to-end speedups merely because a cached input is unused. Preserve clean and marked <=5-page main manuscripts and do not submit to the journal without the author's inspection.

## Bound scale and subgroup inputs

`analysis_inputs/spacing_groups.json` fixes every test image's group under the original 0.01 m ROUND_HALF_UP rule, with 760 / 238 / 188 / 2502 images in the three known bins and unknown group. Its exact hash and both original scale-cache manifest hashes are in `protocol.json`. Scale inputs use PIL RGB and LANCZOS downsampling to round_half_up(width/s) by round_half_up(height/s), without resizing back to the original raster, followed by unchanged per-crop backbone preprocessing. Every evaluation result must carry its exact evaluation label and cache hash, matching the frozen bound inputs.
