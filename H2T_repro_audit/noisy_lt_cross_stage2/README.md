# Noise H2T crossed Stage-II experiment

This is one **outcome-first** diagnostic iteration after the completed CIFAR-100N-LT human-label-noise screen. It is NOT a new method and must NOT modify the original H2T experiment.

**Purpose**: determine whether H2T's observed Few benefit attenuation (-3.83/-3.50 pp in two Stage-II seeds) depends more on a noisy Stage-I representation or on the noisy Stage-II observed-label+balanced-sampler condition.

Read [SCIENTIFIC_PROTOCOL.md](./SCIENTIFIC_PROTOCOL.md) then [CODEX_CROSSED_STAGE2.txt](./CODEX_CROSSED_STAGE2.txt).

## Inputs

Requires the original **LOCAL-ONLY** files from `../noisy_lt_failure_screen/results/cifar100n_lt_seed42_20261029_01/`:
- `primary_cache/stage1_clean/epoch200.pth` and `primary_cache/stage1_noisy/epoch200.pth`;
- `protocol/clean_train.npz`, `noisy_train.npz`, `clean_validation.npz`;
- `primary_cache/inputs/lt_images.npy`, `validation_images.npy`;
- old `RUN.json`, `TRAINING_LOCK.json`, `VALIDATION.json`, `evaluation/final_metrics.csv`.

The new branch inherits old source and summary files but GitHub does NOT contain old weights/NPY cache. **Do not create fake weights or retrain the backbone.** No official CIFAR TEST is read.

## CPU check

```powershell
python -m unittest discover -s H2T_repro_audit/noisy_lt_cross_stage2 -p 'test_*.py' -v
python -m compileall -q H2T_repro_audit/noisy_lt_cross_stage2
```

## True one-batch dry-run (will save separate disposable checkpoints)

```powershell
python H2T_repro_audit/noisy_lt_cross_stage2/cross_stage2.py train --representation clean --labels noisy --seed 124 --method no_fusion --dryrun
python H2T_repro_audit/noisy_lt_cross_stage2/cross_stage2.py train --representation clean --labels noisy --seed 124 --method h2t --dryrun
```

## Fixed eight crossed runs, no Stage-I training

```powershell
$script = 'H2T_repro_audit/noisy_lt_cross_stage2/cross_stage2.py'
foreach ($pair in @(@('clean','noisy'),@('noisy','clean'))) {
    foreach ($seed in @(124,125)) {
        python $script train --representation $pair[0] --labels $pair[1] --seed $seed --method no_fusion
        if ($LASTEXITCODE -ne 0) { throw 'NoFusion failed' }
        python $script train --representation $pair[0] --labels $pair[1] --seed $seed --method h2t
        if ($LASTEXITCODE -ne 0) { throw 'H2T failed' }
    }
}
python $script evaluate
python H2T_repro_audit/noisy_lt_cross_stage2/analyze_factorial.py --diagonal H2T_repro_audit/noisy_lt_failure_screen/results/cifar100n_lt_seed42_20261029_01/evaluation/final_metrics.csv --crossed H2T_repro_audit/noisy_lt_cross_stage2/results/cross_stage2_01/crossed_final_metrics.csv --output H2T_repro_audit/noisy_lt_cross_stage2/results/cross_stage2_01/analysis
```

**Output**: `results/cross_stage2_01/`, add scientific report, CSV and JSON. Checkpoint and batch-control outputs under `local_only` (do not push large tensors; log SHA values and local relative paths). Current code is committed but GPU jobs and tests have not been executed by this assistant; Codex must execute and report real status.

## How to interpret

Compare both methods within each of the four Stage-I/Stage-II combinations. `analyze_factorial.py` verifies the four cells, seed IDs, source checkpoint SHA consistency and group denominators. Pure factorial decomposition does not identify a loss-only causal effect because noisy observed labels alter the Stage-II sampling pools. The clean-labelled crossed cell is **oracle diagnostic only**, not a deployable method.

After this experiment pick at most **one** next method design direction, based on actual outcomes. Prior long-tailed noisy-label methods such as RoLT and H2E are valid mechanisms to adapt; existing similar work is not a reason to discard a potentially useful incremental contribution.
