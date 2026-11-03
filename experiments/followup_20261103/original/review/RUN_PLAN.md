# Fixed follow-up execution plan

This is a post-hoc exploratory analysis motivated by the original test findings. It does not replace any of the original test results. The validation split was already used for hyperparameter selection.

Protocol revision 2: `42789ee8793b7c5bd59bf950dd8deefe8bf1c3caa2e9be2d54143630f63bf288`. Baseline public commit: `6e4ed9a600e78b800e5c4f61e5516d90a017266d`. The independent workspace is `followup_ablation_20261103`.

- Exactly 14 configurations × seeds 0, 1, 2 = 42 runs. All eight ABC configurations use R=T=1; both ABC endpoints additionally use RT=00,01,10. No config or seed is selected or dropped based on outcomes.
- Physical GPU 0 runs seed 0; GPU 1 runs seed 1; GPU 2 runs seed 2. Each process executes all 14 configurations in protocol order. Launch aborts if any planned GPU already has a compute process. GPU 6/7 currently have unrelated VLLM jobs and are not used.
- Each head starts from exactly the same fresh parameters within a seed; 8,263 fit images / 28,759 pairs only, ten epochs, batch 64, 450 batches per epoch, tail 23, lr=.002, kappa=.03, AdamW weight decay=.0001. Frozen B32 backbone and physics training cache.
- Every epoch pair-order hash is checked against the same seeded independent shuffle stream. Text encoding matches the original sweep: sorted unique fit+validation captions, batches of 256, normalized frozen embeddings.
- Final epoch only: full 918×918 validation matrix, same first-caption query order and cache-row gallery order, torch.argsort descending, all ranks and R@1/R@5/R@10/MRR@10 retained. No official test data/heads/results are inputs.
- Each run saves its actual head, fit-only train IDs, full validation matrix, query/gallery IDs, caption hashes, per-query ranks, full loss history, source/input hashes and execution environment. Original cache and sources are hashed before and after.
- Resume verifies all completed artifacts, hashes and initialization. Interrupted attempts are archived only with explicit retry; there is no outcome-dependent reseeding. Original archives and public repository remain unchanged.

Launch command after root review: `.venv/bin/python scripts/launch_fixed_runs.py` in the independent workspace. Persistent processes use separate logs `results/seed_0.log`, `results/seed_1.log`, `results/seed_2.log`. The launch manifest records PIDs, commands, and frozen source hash. No formal follow-up training was launched while preparing this plan.

The factor adapter changes only the intended three pathways and two legacy switches. Its comparison with the published `fuse_batch` is in `fuse_batch_minimal.diff`. CPU and CUDA checks compare full, strict, and legacy-CFAR anchors for all RT settings, dimensions 512/768, full/mixed/empty masks, both weighted-score and contrastive-loss gradients. The tiny optimizer checks additionally compare two complete epochs, duplicate image/caption identities, a final size-one batch and all three seeds.
