# Follow-up fit-only heads

These 42 files are the exact evaluated checkpoint bytes for 14 configurations
and seeds 0, 1 and 2. Every head was trained for ten epochs using only the
8,263-image fit subset (28,759 caption-image pairs). They are not the 20 original
9,181-image final-training heads and must not replace those test-evaluation heads.

This is a post-hoc exploratory study on the previously used 918-image validation
set. The main full PARE definition remains ABC111_R1T1. See
[all results and contrasts](../../benchmarks/followup_20261103/README.md).

The [publication manifest](../../benchmarks/followup_20261103/PUBLICATION_MANIFEST.json)
maps each factor tuple and seed to its exact head SHA-256, rank record, and
archived full score-matrix hash. The files contain a tensor state dictionary,
factors, fit-only IDs, training history and provenance; they contain no images
or caption bodies. Backbone weights are obtained separately from their owners.

Verify hashes from the repository root before loading:

```sh
python tools/followup/verify_public_followup.py
```

Load the factor-aware class, not an ordinary full head that silently ignores the
configuration. From the repository root, for example:

```python
from pathlib import Path
import sys
import torch
root = Path.cwd()
source = root / "experiments/followup_20261103/original"
sys.path[:0] = [str(source / "base_code"), str(source / "code")]
from factorial_head import FactorialHead, Factors
checkpoint = torch.load(
    root / "checkpoints/followup_20261103/seed_0/ABC111_R1T1/head.pt",
    map_location="cpu", weights_only=True)
head = FactorialHead(checkpoint["D"], Factors(**checkpoint["factors"]))
head.load_state_dict(checkpoint["state_dict"], strict=True)
head.eval()
```

The class uses `fuse_batch` with the unchanged cached-document schema. Its legacy
single-document `forward` is deliberately disabled. Scientific parameters and
initialization match the original head; the factor controls change only the
documented paths.
