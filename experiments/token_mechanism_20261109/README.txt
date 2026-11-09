Token-conditioning diagnostic, 9 November 2026 (Supplementary S9)

Complete exploratory archive: 15 matched trained heads (five seeds x three
variants), 30 within-split full-gallery evaluations, all rank vectors, training
records, frozen execution-source copies, independent audits and statistical
analysis. The 15 included head files are byte-identical to evaluated originals;
metadata strings were inspected without unpickling. Images, caption bodies,
embedding caches and third-party backbone weights are not redistributed.

All variants use the B0 gate with fine attention still present, original
reliability, and 24x24 fine crops. Conditional uses regional descriptors;
constant_descriptor supplies the per-branch training-only mean to the SAME
token network; token_off bypasses token augmentation. Training uses 6610 images
and 23016 pairs. Development has 918 images; internal_check has 1653 images.
Both sets have prior use. This is exploratory fixed-data evidence, not a new
independent external test and not the PARE-S/CFAR-strict configuration.

Conditional minus constant-descriptor R@10 is -0.3050 pp in development and
+0.2178 pp in internal_check. Neither passes the fixed simultaneous-interval
evidence criterion. Close means do not establish equivalence. Conditional minus
token-off is +12.3747 pp and +10.5989 pp, respectively, passing the prespecified
within-dataset criterion in both splits. Every seed, variant, adverse outcome
and comparison is preserved in results/ and analysis/. These results support
the token module in this particular recipe; they do not establish physical
scattering labels, independent fine-branch benefit, or a result for PARE-S.

PUBLIC_ARCHIVE_MANIFEST.json binds every distributed evidence file to its
original hash and public-copy hash. Personal paths in orchestration/provenance
were redacted transparently. measured_source/ is an execution-source archive,
not a promise that redacted historical paths or anchor dependencies can run
unchanged. The separate simplified-study portable entry is newly authored and
must not be credited with producing these measurements.

Full score matrices are retained in the original evidence store and identified
by hashes in run/audit files; this compact archive distributes their independently
verified rank projections. All three variants' 15 small trained heads are
included. The computational analysis, ordinary and multiplicity-adjusted
intervals, and independent checks retain both positive and negative results.
