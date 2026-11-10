# CC-SAM-TPR — MORE-inspired tail parameter reallocation

This directory contains the next classifier-stage method iteration after the sealed C0-C4 CC-SAM study.

## Method

- Keep the validated C1 class-conditional feature-gradient perturbation and objective.
- Use the matching seed's completed C2 classifier as a frozen general classifier.
- Add a low-rank residual only to non-Many output rows: W_t = D_tail B A.
- Use the MORE class-prior-weighted squared L2 logit discrepancy with a sinusoidal schedule.
- Compute tail-neighbor diversity on the effective classifier residual rows B @ A (not on B coefficients alone).
- Address MORE's two discussion issues with train-only tail-neighbor effective-residual-row separation and a soft same-class row-wise general/residual orthogonality term.
- Merge the residual into one classifier at inference. No experts or additional inference branch.

This is a classifier-stage MORE-inspired adaptation, not a full reproduction of the network-wide MORE method. Read TPR_PROTOCOL.md before execution.

If preflight/Smoke artifacts were created with the earlier coefficient-space draft, archive them before rerunning the corrected protocol; never delete or overwrite them. If any formal TPR output already exists, stop and preserve it rather than overwriting it.

## Files

- train_tpr.py — preflight, TRAIN-only graph extraction, real-data Smoke, and formal training.
- test_tpr.py — CPU-only semantic tests.
- analyze_tpr.py — fixed epoch-30 paired analysis against C0-C4.
- finalize_tpr.py — local/GitHub artifact checksum index.
- TPR_PROTOCOL.md — method definition, assumptions, frozen protocol, and limitations.
- CODEX_TPR_PARAMETER_REALLOCATION_20261110.txt — exact Codex execution instructions.

## Execution order

1. Run syntax and unit tests.
2. Run --preflight.
3. Run --smoke-only.
4. Run --formal.
5. Run analyze_tpr.py.
6. Run finalize_tpr.py.

Do not skip a stage. The official CIFAR-100 TEST split is prohibited. C0-C4 are protected read-only inputs. Local-only final checkpoints and batch traces must remain on the local machine and must not be force-added to Git.


**Execution gate note (2026-11-10):** The runner imports the C3 snapshot helper from
`ab_head_preservation.train_ab`. Smoke receives the `protected_inputs` mapping
directly (not the enclosing preflight report). Run the CPU unit tests before
Preflight or any GPU work.
