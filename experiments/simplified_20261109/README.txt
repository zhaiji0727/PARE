PARE-S matched reproducibility study, 9 November 2026

Complete evidence: 75 trained heads (five backbones x five seeds x three
variants), 105 full-gallery model evaluations, seven frozen scene baselines,
all per-query rank vectors, training records, frozen execution-source copies,
independent artifact and statistical audits, and preserved execution history.
All prespecified outcomes are retained, regardless of sign.

The variants are prior full PARE-F ([]), simplified PARE-S ([cfar_strict]), and
PARE-S without token augmentation ([cfar_strict,ty]). Existing CFAR-strict jointly
removes the fine attended vector, fine-to-gate input, and coarse detector
fraction. This comparison does not identify the isolated effect of any one path.
The method remains radiometric-inspired; it does not claim physically verified
scattering labels or independently established scene-conditional token benefit.

PARE-S minus PARE-F has positive mean R@10 differences on all five backbones,
with positive differences in 23 of 25 paired seeds. Only SigLIP meets the fixed
stable-positive-evidence rule. PARE-S minus its token-off control has positive
R@10 differences in all 25 paired seeds; four backbones meet the rule, while
RemoteCLIP does not. These qualifications are part of the result, not omitted
exceptions. The rule requires all five paired R@10 differences to be positive,
a positive lower bound for the Bonferroni family-95% interval across ten primary
contrasts, and nonnegative mean MRR@10 difference. Five-seed t intervals rely on
a small-sample approximation. Means near one another do not establish equivalence.

All primary R@10 contrasts (percentage points):
Backbone   Comparison                    Mean    Family95 interval    Positive seeds  Rule
vitb32     strict_minus_full             +1.5293 [-1.3274, +4.3860] 5/5 no
vitb32     strict_minus_strict_token_off +7.8145 [+5.5616, +10.0675] 5/5 yes
vitb16     strict_minus_full             +0.8514 [-0.1164, +1.8192] 5/5 no
vitb16     strict_minus_strict_token_off +7.3753 [+6.5453, +8.2052] 5/5 yes
remoteclip strict_minus_full             +1.5835 [-5.6003, +8.7674] 4/5 no
remoteclip strict_minus_strict_token_off +4.0889 [-3.0886, +11.2665] 5/5 no
siglip     strict_minus_full             +1.5130 [+0.5407, +2.4853] 5/5 yes
siglip     strict_minus_strict_token_off +8.0098 [+6.7018, +9.3177] 5/5 yes
georsclip  strict_minus_full             +0.7755 [-4.9215, +6.4724] 4/5 no
georsclip  strict_minus_strict_token_off +7.3536 [+1.1105, +13.5967] 5/5 yes

The candidate was chosen after earlier test results, and the fixed test set was
previously used. This is matched numerical evidence on reused data, not a new
blind external test. The 30 additional ViT-B/32 fixed-head synthetic-spacing
evaluations and all nominal-spacing subgroups are reported in analysis/;
synthetic resampling does not establish invariance to real sensor/acquisition
changes. The separate 15-head diagnostic is ../token_mechanism_20261109/ and
uses a different full-branch recipe; its numbers are not substituted for this
simplified-study experiment.

measured_source/ holds actual frozen execution-source copies. The original
freeze SHA-256 is fb39609b2d5cdedff24ffcfe4db3c1f16a3d7b2159dc4c46195f942f8ab585f3.
PUBLIC_ARCHIVE_MANIFEST.json and MEASURED_SOURCE_MANIFEST.json give original and
public-copy hashes and disclose path redactions. Numerical model and scoring
files are unchanged; redacted historical orchestration paths are not runnable
replacements for readers' own paths. verification/source/ contains the exact
independent auditing code (or disclosed path-only copies), bound by its reports.

execution/ preserves the failed initial SigLIP tokenizer attempt and successful
recovery, including the original failed anchor-stage record. No head had been
trained in that failed attempt. Restoring the original offline HF_HOME resolved
the missing tokenizer without changing frozen numerical source, data, or the
training recipe. Subsequent train-all/evaluate/audit stages completed. Logs and
JSON are path-redacted; their original/public SHA values are recorded. Only
transient process-ID locator files are omitted from execution history.

All 75 measured heads are distributed at ../../checkpoints/simplified_20261109/.
Only metadata path/host strings were projected. Every parameter's dtype, shape
and raw bytes were compared to its independently audited original after reload;
HEAD_PROJECTION_MANIFEST.json maps original/public hashes and every change.
Images, caption bodies, embedding caches and third-party weights are excluded.
Full score matrix hashes and independently verified rank projections remain in
the run/audit records; matrices are not redistributed in this compact archive.

portable/ is newly authored reader orchestration, not the measured executor.
It calls unchanged original code/train_saqre.py with the fixed three variants,
five seeds, ten epochs, batch size64 including the final partial batch, learning
rate0.002, per-backbone temperature and deterministic float32 settings. Its
full-gallery scorer uses the original numerical kernel and tie convention.
It needs no private machine paths or legacy anchor matrices. See portable/README.txt
for data/backbone/tokenizer preparation and execution. Source, split and asset
identities are checked before running. New reproduction outputs are explicitly
labeled separately from these archived measurements.

GPU validation of the portable scoring adapter: all six audited original
seed0 heads (ViT-B/32 and SigLIP, three variants each) produced exactly the
same complete 3688x3688 float32 score matrices, query identities and ranks as
the measured executor; maximum score difference was zero. No training occurred
in this check. The 75 public projections separately passed parameter-byte
identity and reload checks; no additional six projected-head GPU evaluations
were performed. This does not validate end-to-end portable retraining or imply
GPU parity for untested backbones/seeds. See verification/portable_gpu_parity/.
