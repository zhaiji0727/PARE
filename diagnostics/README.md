# Reproducibility diagnostic

`extractor_919/` contains the measured 7 October 2026 CPU comparison on all
919 original validation IDs: summary, per-image scalar measurements, and crop
coordinate records. No source images or captions are included. The selected
image filenames refer to the separately obtained official dataset.

The comparison found changes in coarse box sets for 467/919 images and fine
box sets for 827/919; the repaired code has zero negative backgrounds and zero
threshold-monotonicity violations in this audit. These are implementation
diagnostics, not retrieval improvements or object-detection accuracy.

`reference/saqre_phys_before_20261007.py` is the exact frozen pre-repair
extractor, SHA-256 `38482c06c852f1d45b985fe6b862b0ae009ec4d0d880432e1ddca2f5b655acbc`.
Its comparison with the current extractor isolates float64 statistics and
spatial intensity-tie behavior. Use the main `code/` module for new experiments.

The archived summary retains original source hashes. Only machine-specific
paths have been made relative. For a self-contained checkout, the released
audit script changes the old-extractor lookup path, so its byte hash differs
from that of the original run. The release manifest records the released bytes.
Re-running the diagnostic records the source hashes used on the new machine.

The original 919-ID validation list contains exact-content leakage documented
by the newer split audit. It is retained here for paired implementation
comparison only. Use the regenerated 918-ID clean validation set for fitting,
selection and revised retrieval experiments. Full retrieval results remain
pending as of this release.
