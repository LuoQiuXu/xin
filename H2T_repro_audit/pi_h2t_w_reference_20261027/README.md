# PI-H2T reference-W paper-parity experiment (one Stage II arm)

**Research status: exploratory replication diagnostic, NOT a verified author PI-H2T reproduction, NOT a novel method.**

## Why this specific experiment?

The archived complete CIFAR100-LT IF100 run used:
- a genuinely trained two-parameter PIF Stage I (200 epochs), fixed checkpoint SHA256
  `64f53e45a1b7f73f494f9c88374607e714b58dc6eb55118fa31154c4389f93ef`;
- Stage II frozen-Stage-I-W H2TF: Overall 36.50%, Many 29.86%, Medium 43.29%, Few 36.55%;
- matched Stage II without fusion: Overall 45.53%, Many 67.08%, Medium 46.34%, Few 17.79%;
- H2TF mean donor mass about 0.772, with train CE about 4 at epoch10.

This is **not** a successful reproduction of the PI author's reported Stage II last 49.04%, and Figure 9's ratio must not be targeted as a fitted hyperparameter.

The original PI-H2T paper (v2 Eq8 / Algorithm1) refers to classifier weights **W** during Stage II. In the prior independent reconstruction the ratio reference was silently fixed to the **Stage I** classifier throughout all 10 epochs. The paper/author-code evidence found thus far DOES NOT resolve whether W must be frozen or reflect the actively optimized **Stage II** classifier; this is a true **implementation parity uncertainty**.

Thus there is exactly one pre-declared fork:

> Keep Eq8 r=(1-cos(f_rec,w_y))/2, Eq7 f_mix=r f_rec+(1-r) f_donor, all data/BN/seed/CE/optimizer/checkpoint; update ONLY the source of w_y from frozen Stage I to current Stage II classifier. 

This test shows if the interpretation matters under otherwise identical implementation. **It does not by itself establish which interpretation the PI-H2T authors used.** The candidate/current W arm was not chosen by a metric sweep. No formula inversion, fixed-mass scaling, donor-selection method, Low-VS, margin protection, or new architecture.

## Experimental budget: only ONE Stage II arm (10 epochs), zero new Stage I training

The finished historical arms stay entirely untouched:
- `H2T_repro_audit/pi_h2t_strict_20261027/runs/pih2t_if100_seed123_42_v1/stage2/h2tf` = frozen Stage I W.
- `.../stage2/no_fusion` = paired no fusion.

New arm:
- `H2T_repro_audit/pi_h2t_w_reference_20261027/runs/w_reference_current_seed42_01/`.
- Reuses exactly the local Stage I checkpoint and original CIFAR100 train/test cache, requires StageI SHA match.
- Stage II seed 42, 10 epochs, batch128, SGD lr0.02 cosine, weight decay0.0005, BS recipients+IS donors, **train-mode BN updated by identical recipient/donor forward calls**.
- Trainable classifier only, frozen PIF/backbone, live `head.fc.weight.detach()` used for ratio.
- No best-epoch selection; evaluate epoch10 on reused official TEST for reproduction **diagnostic only**.

## Required verification before interpreting numbers

- Old source/Stage I SHA and data source identical; historical Stage II artifact for BOTH frozen W and no-fusion exists locally.
- Byte hashes of exact augmented receiver images, IS donor images, receiver/donor labels identical **at every epoch** to frozen W historical run.
- The new classifier initialization matches the old source Stage II initialization hash.
- Backbone and PIF parameter hashes unchanged; final BN running-buffer hash **equals** frozen W run.
- Prediction labels/output length matches frozen/no-fusion saved TEST predictions (10,000 ordered examples, 100 per class).
- No inference fusion in any arm.
- If any check fails, mark **INVALID**, stop; never compare accuracies, quietly switch BN/fusion definitions, or weaken hashes.

## Run from repository root (same Windows GPU environment)

```powershell
python -m unittest discover -s H2T_repro_audit/pi_h2t_w_reference_20261027 -p "test_*.py" -v
python -m py_compile H2T_repro_audit/pi_h2t_w_reference_20261027/ratio_reference.py H2T_repro_audit/pi_h2t_w_reference_20261027/run_current_w.py
python H2T_repro_audit/pi_h2t_w_reference_20261027/run_current_w.py preflight
python H2T_repro_audit/pi_h2t_w_reference_20261027/run_current_w.py run
```

Do NOT re-run the report if the result directory is already partially populated without first diagnosing why; preserve error logs. `report` may be run after a complete saved result, but only for analysis (never model selection).

Outputs: preregistered_protocol.json, epochs.csv, ten local ratio_epochNN.npz, epoch10 classifier checkpoint, 10k TEST logits NPZ, result.json, comparison.json, per_class_current_vs_frozen.csv and SCIENTIFIC_REPORT.md.

## Scientific decision after the run

Only two possible endings:
- VALID single-factor result: describe current-vs-frozen W accuracy and ratio trajectories, whether gap to author result **descriptively** narrows; do not claim full author replication or statistical significance.
- BLOCKED/INVALID: preserve first failure and stop; no model tweaks or next blind arm.

A positive current-W result cannot settle PIF initialization, Stage I 39.91 vs author 42.19, checkpoint selection, exact author BN/detach policy, or original paper Fig9's statistical semantics. A negative result rejects ONLY this parity fork under this run. In either case, next step is independent author-code/author-contact confirmation if available, not an uninformed ratio sweep.

Preserve historical untracked directories `fusion_semantic_audit`, `lowvs_mask_parity`, `mdlt_domain_probe`; do not modify protected original sources or prior runs. All new reports in this new study directory.
