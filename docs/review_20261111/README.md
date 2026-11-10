# Historical supplement and revision context

[Historical_Supplement_20261110.pdf](Historical_Supplement_20261110.pdf) is the
unchanged 24-page supplement prepared before the 11 November review. Its
[provenance](HISTORICAL_PDF_PROVENANCE.json) identifies its exact bytes. It is an
extended historical record, not the short supplement for the current submission.

## Reading the historical record

- PARE-F denotes the full configuration. Historical `PARE` or `full` rows refer
  to that configuration, unless their local caption explicitly states otherwise.
- PARE-S is the `cfar_strict` control from the October submission. It disables
  fine attended features, removes fine embeddings from the gate input and zeros
  the coarse descriptor's detector-fraction coordinate together. Dropping only
  fine attention does not implement the strict configuration.
- The ten seed-0 F/S runs in the matched study reproduce results known before
  the recommendation of PARE-S. They are compatibility checks, not new evidence.
  Seeds 1–4 are the additional runs; test-set reuse still limits inference.
- Old future-tense statements and non-promotion statements describe the stage
  at which those sections were written. PARE-S was subsequently recommended.
  The historical text is retained for provenance rather than silently rewritten.
- Token-off, uniform-grid, reliability and constant-descriptor controls include
  unfavorable and inconclusive differences. The token module's aggregate benefit
  does not prove that region-specific physical conditioning causes that benefit.
- The five-seed scale study and its original metadata groups use OpenCLIP
  ViT-B/32. Group ordering must not be generalized to every backbone. The current
  paper distinguishes source specifications, stored pixel spacing and sensor
  resolution, and provides the Scene reference alongside scale results.

## Existing experiment archives

The source and evaluated artifacts remain pinned to experimental release
`b174e52fa987085f09b2c75ca173fa0ca042c484`, including the matched study in
`experiments/simplified_20261109/`, the token mechanism study in
`experiments/token_mechanism_20261109/`, and the factorial follow-up in
`benchmarks/followup_20261103/`. Historical measured-source trees and their
hash contracts are unchanged.

A separate development exploration considered gate inputs, reliability rules,
limits on token magnitude and 48-pixel crops: its recorded scope was 54 trainings,
108 development/internal-check evaluations and 42 fixed-model probes. These were
exploratory analyses on previously used data. The 48-pixel replacement was not
adopted for the final matched models. This pre-review supplement does not contain
the complete record of that exploration and should not be described as containing
all development outcomes. The current revision's experiment index must identify
any additional public record added for it.

## Documentation-only source correction

Current `code/saqre_phys.py` comments describe display-domain, image-relative
features and a learned prototype token without assigning scattering classes.
`code/bench_efficiency.py` describes what the supplied artifacts allow the timing
program to measure. The update changes comments and docstrings only; stripping
docstrings leaves identical Python ASTs.

Source hashes nevertheless change. To reproduce historical measured artifacts,
use their pinned commit or measured-source path and original contract. Do not
rewrite archived source hashes or disable compatibility checks to make a newer
commented source tree masquerade as the measured version.
