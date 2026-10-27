# PI-H2T v2 · CIFAR-100-LT IF=100 · paper-aligned independent reproduction

**Purpose:** Reconstruct the published PI-H2T PIF + H2TF pipeline on a clean, reproducible CIFAR100-LT/ResNet32 protocol. **Not official author code; not yet certified as exact published-number parity.** No novel mechanism, no HAFE, no Low-VS, no donor affinity, no MarginProtect.

## Confirmed sources

- PI-H2T arXiv v2 `https://arxiv.org/pdf/2506.00625v2`, Eqs (3),(6),(7),(8), figure and pseudocode audited in `../research_gap_20261026/results/evidence_20261026_01/PI_PAPER_CODE_PARITY.md`.
- Author public H2T GitHub `https://github.com/Keke921/H2T`; as checked 2026-10-27 its only public branch is `main` at `e0c849235d9655fca31cd9fa8a5f38f92a77c7e8`. **It provides original H2T, not a verified full PI-H2T implementation.**
- Reuse original CIFAR dataset builder, feature ResNet32 and classifier **without modification** under `H2T_official_clean/H2T-main`.

## Scope of implementation

`paper_components.py`:
- PIF is a single bias-free Conv1d(2,1,1) over channel residual `F-mean_C(F)` and `F`. Exactly two trainable coefficients. PIF operates on [B,64,8,8] maps after the last ResNet conv block, before GAP.
- H2TF first pools recipient/donor maps to [B,64]. Receiver retention `r=(1-cos(f_recipient,w_label))/2`, donor mass `1-r`. Training uses **receiver labels only**. Inference uses PIF and classifier, **not** H2TF.
- PIF is *channel-permutation equivariant*, NOT mathematically invariant as an entire output (the channel-wise mean is invariant). The name from the paper is not a license to claim the full output is invariant.

`run_reproduction.py`:
- Stage I: CIFAR100-LT IF100 **original per-class deterministic index selection seed0**, ResNet32+PIF+CE+MixUp(alpha1), 200 epochs, batch128, SGD .1, momentum .9, weight decay 2e-4, 5-epoch linear warmup, step decay milestones 160/180, fixed seed123.
- Stage II: one StageI fixed **epoch200** checkpoint, independent `no_fusion` and `h2tf` classifier-only runs, seed42, 10 epochs, initial SGD .02, wd5e-4, cosine decay, batch128; BS receiver / long-tail IS donor; PIF and ResNet gradients frozen.
- For a **matched-BN-traffic control**, both StageII arms forward both receiver and donor images, even when fusion is disabled. This is a deliberate fair-control deviation from the author's simpler no-fusion path and must be disclosed.
- Default `--bn-mode update` is inherited from H2T StageII `shift_bn=True`; `--w-reference frozen_stage1`, PIF Conv1d PyTorch default initializer and `detach=True` are **locally declared protocol choices NOT confirmed as PI author's exact choices**. Do NOT tune these after accuracy inspection.
- Official CIFAR100 TEST is used only to **report** protocol results, never to choose a checkpoint or hyperparameters. Therefore checkpoint `fixed epoch200`, not the paper's possibly TEST-selected best. This also prevents presenting reused TEST data as independent final validation.

## Phase gates / cost

1. **Read/compare** PI paper and author repo, mark all unresolved fields `MATCH / MISMATCH / UNVERIFIED`. A complete exact-author reproduction is blocked while any author-specific required field is UNVERIFIED.
2. **CPU formula tests** first. They should not require CIFAR data or a GPU.
3. **Preflight** CUDA and official CIFAR data presence; data download only with opt-in `--download`. Test dependencies and source hashes, inspect stage/source choices and batch memory.
4. **Stage I** (expensive: 200 epochs) only after Steps 1–3 pass and the user-authorized reconstruction scope is understood. No hidden fallback to historical 25-param checkpoint.
5. **Stage II** 2 × 10 epochs from the **same** new StageI fixed epoch200, with no-fusion vs H2TF control. Use locked BN/W reference selected **before** results.
6. **Report:** Overall/Many/Medium/Few, per-class paired correct→wrong and wrong→correct, average donor mass, learned PIF coefficients, exact provenance/hash of weights and selected LT samples.

## Commands

From repository root (Windows PowerShell is fine):

```powershell
python -m unittest discover -s H2T_repro_audit/pi_h2t_strict_20261027 -p "test_*.py" -v
python H2T_repro_audit/pi_h2t_strict_20261027/run_reproduction.py preflight --run-id pih2t_if100_seed123_42_v1 --data-root H2T/data/cifar100
python H2T_repro_audit/pi_h2t_strict_20261027/run_reproduction.py stage1 --run-id pih2t_if100_seed123_42_v1 --data-root H2T/data/cifar100
python H2T_repro_audit/pi_h2t_strict_20261027/run_reproduction.py stage2 --run-id pih2t_if100_seed123_42_v1 --data-root H2T/data/cifar100 --arm no_fusion
python H2T_repro_audit/pi_h2t_strict_20261027/run_reproduction.py stage2 --run-id pih2t_if100_seed123_42_v1 --data-root H2T/data/cifar100 --arm h2tf
python H2T_repro_audit/pi_h2t_strict_20261027/run_reproduction.py report --run-id pih2t_if100_seed123_42_v1
```

Change only `--data-root` to the **existing original CIFAR100 cache root**; never point the tool to a pre-balanced/generated or alternative dataset. If raw files are not downloaded yet, use explicit `--download` during preflight, and retain source provenance.

**Do not change** `--w-reference` or `--bn-mode` between StageI and StageII: the protocol manifest is locked and StageII will refuse divergence. If a genuine protocol error is discovered after StageI, stop and declare that run invalid; write a fresh run ID and disclose the reason rather than silently retune. 

## Output

New directory `runs/<run-id>/`, never overwritten. Includes:
- `protocol.json` with source hashes, author-uncertain choices.
- `dataset_counts.json` with IF100 counts and exact LT image/label hash.
- `stage1_final_epoch200.pth`, `stage1_epochs.csv`, `stage1_summary.json` (including PIF initial/final coefficients).
- `stage2/no_fusion/` and `stage2/h2tf/`: independent FC checkpoints, 10-epoch train history, prediction NPZ and summary JSON.
- `comparison.json`, `per_class_paired_changes.csv` (descriptive; never used for TEST-driven method selection).

Large raw checkpoints and NPZ should **not** be pushed indiscriminately. Preserve them locally and record relative path, size and SHA256 in a concise GitHub report. Reports/CSV/JSON can be committed explicitly on this branch.

## Stop conditions

CUDA/OOM, dataset split discrepancy, nonfinite loss, source/hash drift, missing or untrusted checkpoint, inconsistent class counts, or unsupported assumption ⇒ STOP, log facts, **do not** substitute historical PI weights, remove checks, change batch size, adjust learning rates, or reverse ratio to get higher accuracy.

This is an initial, **single-seed protocol reproduction**, not new SCI Q2 novelty, not proof that published PI-H2T fails. If it differs greatly from paper numbers, first audit paper-to-code parity rather than inventing a fix.
