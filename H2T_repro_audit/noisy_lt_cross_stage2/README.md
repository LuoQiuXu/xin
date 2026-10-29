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

**Output**: `results/cross_stage2_01/`. This round has actually completed the eight crossed runs and fixed-final evaluation; results and source checks are described below. Checkpoints, logits and batch controls remain under ignored `local_only/`, with SHA256 and relative paths recorded.

## How to interpret

Compare both methods within each of the four Stage-I/Stage-II combinations. `analyze_factorial.py` verifies the four cells, seed IDs, source checkpoint SHA consistency and group denominators. Pure factorial decomposition does not identify a loss-only causal effect because noisy observed labels alter the Stage-II sampling pools. The clean-labelled crossed cell is **oracle diagnostic only**, not a deployable method.

After this experiment pick at most **one** next method design direction, based on actual outcomes. Prior long-tailed noisy-label methods such as RoLT and H2E are valid mechanisms to adapt; existing similar work is not a reason to discard a potentially useful incremental contribution.

## Actual execution — 2026-10-29

Interpreter: `D:\app\minicondaaa\envs\xiaoju\python.exe`. No environment changes or new dependencies. Both authentic Stage-I epoch200 checkpoints passed SHA/state/finite checks; **zero new Stage-I epochs**. CPU tests 4/4 passed, real paired dry-run passed, and all eight 10-epoch Stage-II runs completed. `VALIDATION.json` confirms 3400 exact paired batches, identical final BN and unchanged source weights/history. All eight final models were locked before one evaluation each; official TEST was not read.

Few mean H2T−NoFusion (pp): clean/clean +4.0833, clean/noisy −2.6667, noisy/clean +2.1667, noisy/noisy +0.4167. Few mean R effect +0.5833pp, S effect −4.2500pp, interaction +5.0000pp. Prioritize Stage-II observed-label/sampling supervision with strong R dependence. Clean conditions are oracle diagnostics; this is not a deployable noise-robust method or evidence that every metric deteriorates.

Read [CROSSED_STAGE2_SCIENTIFIC_REPORT.md](results/cross_stage2_01/CROSSED_STAGE2_SCIENTIFIC_REPORT.md), [factorial results](results/cross_stage2_01/analysis/factorial_accuracy.csv), [contrasts](results/cross_stage2_01/analysis/factorial_contrasts.csv), [JSON summary](results/cross_stage2_01/analysis/summary.json), and [checkpoint index](results/cross_stage2_01/CHECKPOINT_INDEX.csv).

Recorded run time totals **608.10 seconds** (about 10.14 minutes), excluding interpreter import and separate checks/evaluation/reporting. PyTorch allocated peak was **124.47 MiB**; this metric is not total device/process VRAM.

| R | S | seed | 方法 | run耗时(s) | allocated峰值(MiB) |
|---|---|---:|---|---:|---:|
| clean | noisy | 124 | no_fusion | 74.13 | 124.47 |
| clean | noisy | 124 | h2t | 76.96 | 124.47 |
| clean | noisy | 125 | no_fusion | 75.36 | 124.47 |
| clean | noisy | 125 | h2t | 77.39 | 124.47 |
| noisy | clean | 124 | no_fusion | 78.64 | 124.47 |
| noisy | clean | 124 | h2t | 75.53 | 124.47 |
| noisy | clean | 125 | no_fusion | 75.23 | 124.47 |
| noisy | clean | 125 | h2t | 74.85 | 124.47 |

## Source checks and recovery commands

The actual order was `cross_stage2.py preflight`, CPU tests/compilation, the two listed dry-runs, eight training runs, `evaluate`, `analyze_factorial.py`, then `finalize.py`. The real dry-run pair was independently checked and `DRYRUN.json` written before formal training. Preflight and all result files use exclusive creation; do not re-execute them into this existing result directory. A future independent reproduction must set the new wrapper's `OUT` to a fresh directory before preflight; keep all original source assets and this result directory intact.

`LOCAL_ONLY_INDEX.csv` indexes real models/logits/controls/logs and their SHA. Hashes describe actual local bytes; Git may normalize text line endings, while model/array hashes remain binary hashes. GitHub contains no binary backup. Restore missing binaries only from your own backup, preserving paths relative to the repository root; never substitute retraining Stage I. Example below is for one missing file and must be configured with an existing backup root:

```powershell
$backupRoot = '<填写已有备份仓库根目录>'
$relativeFile = 'H2T_repro_audit/noisy_lt_cross_stage2/results/cross_stage2_01/local_only/repr_clean_labels_noisy_seed124_h2t/epoch10.pth'
$expectedSHA = '9ee99d8a9882bc391411f3de8913b685b5e5c59fdd7817627e73e760bca14b08'
$backupFile = Join-Path $backupRoot $relativeFile
if (!(Test-Path -LiteralPath $relativeFile)) {
    if (!(Test-Path -LiteralPath $backupFile)) { throw 'Backup missing: stop, do not fabricate or retrain' }
    if ((Get-FileHash -LiteralPath $backupFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSHA) { throw 'Backup SHA mismatch' }
    New-Item -ItemType Directory -Force -Path (Split-Path $relativeFile) | Out-Null
    Copy-Item -LiteralPath $backupFile -Destination $relativeFile
}
if ((Get-FileHash -LiteralPath $relativeFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSHA) { throw 'Local SHA mismatch; preserve and diagnose' }
```

Source Stage-I SHA values and data paths are separately recorded in `SOURCE_AUDIT.json`. Training rules remain the original H2T; wrapper additions provide IO protection, timing/memory records and saved logits only. No extra seeds, noise settings, modules or training runs were added.
