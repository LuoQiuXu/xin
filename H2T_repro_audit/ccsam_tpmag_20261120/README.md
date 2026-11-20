# Tail-Preserving Margin-Aware Gate (TP-MAG)

This is the first method-construction experiment following the completed CC-MIA causal/oracle and logit-geometry diagnostics.

## Scope

TP-MAG applies a continuous, per-step gate to the existing CC-SAM direct source-class -> Many-classifier-row perturbation-gradient components. It uses two first-order quantities computed against a frozen Calibration-only reference:

1. Benefit to the target Many class's fixed Many-vs-non-Many margin.
2. Risk to the source non-Many class's margin against the aggregate Many logit set.

The gate is parameter-free apart from a numerical epsilon. It does not change the feature perturbation, clean CE gradient, loss, backbone, optimizer, sampling plan, or training schedule.

## Read in this order

1. TPMAG_PROTOCOL.md
2. gate_core.py and test_gate_core.py
3. CODEX_TPMAG_EXECUTE_20261120.txt

## Commands

From the repository root, using the existing project Conda environment previously referred to as 小菊:

- python H2T_repro_audit/ccsam_tpmag_20261120/validate_cpu.py
- python H2T_repro_audit/ccsam_tpmag_20261120/train_tpmag.py --phase preflight
- python H2T_repro_audit/ccsam_tpmag_20261120/train_tpmag.py --phase smoke
- python H2T_repro_audit/ccsam_tpmag_20261120/train_tpmag.py --phase formal
- python H2T_repro_audit/ccsam_tpmag_20261120/train_tpmag.py --phase analyze

The formal phase trains only the new TP-MAG arm for seeds 123, 124 and 125. The completed CC-MIA parent provides the paired C2, BASE_C3, TARGETED, matched-control and ORACLE_SELECTION references. The run reproduces the original C0 batch trace and uses the same Clean Stage-I checkpoint, sampler, augmentation, frozen ResNet-32 GAP features, optimizer and 30-epoch schedule.

Calibration is used to construct the frozen class-conditional reference. Evaluation is not opened until all three TP-MAG seed runs have completed. Evaluation is a TRAIN-derived development partition, not an independent test set. Official CIFAR-100 TEST access is prohibited.

## Output policy

Runtime outputs are exclusive. No resume, overwrite, best-epoch selection, threshold sweep or Evaluation-driven tuning is permitted. Large checkpoints and NPZ files are ignored by Git; CSV/JSON reports are intended to be pushed after a successful run.
