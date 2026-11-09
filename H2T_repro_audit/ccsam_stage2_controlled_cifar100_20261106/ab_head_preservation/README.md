# C4 — CC-SAM with Many-only head-preservation distillation

This directory contains the first constructive A+B method experiment following the sealed C0-C3 CC-SAM Stage-II study.

- A: retain the complete C1 class-conditional feature-gradient perturbation.
- B: distill the matching seed's frozen C2 classifier output distribution on Many-class original GAP features only.
- Student starts from the Clean Stage-I classifier; teacher is the corresponding C2 final-epoch classifier.
- Fixed first-run settings: T=2.0, lambda_HP=1.0.
- Only C4 is newly trained: seeds 123/124/125, 30 epochs each.

Read AB_PROTOCOL.md before execution. Run the exact sequence in CODEX_AB_HEAD_PRESERVATION_20261109.txt.

The scripts deliberately refuse to overwrite results and verify the original C0-C3 artifacts, C2 teacher checkpoints, prepared inputs, and TEST-free provenance. The official CIFAR-100 TEST split is prohibited.

Main entry points:
- train_ab.py — preflight, real-data Smoke, and C4 formal training.
- test_ab_distillation.py — CPU-only KL/masking tests.
- analyze_ab.py — fixed epoch-30 paired analysis against C0-C3.

Large C4 checkpoints and batch traces remain local-only and are excluded by .gitignore. Their SHA256 values must be recorded in the final artifact index.
