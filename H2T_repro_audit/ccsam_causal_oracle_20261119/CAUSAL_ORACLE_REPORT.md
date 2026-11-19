# CC-MIA Causal / Oracle Report

Completed: 2026-11-19T23:15:38+08:00

This is a held-out TRAIN-derived development analysis. No official TEST was accessed.

All three seeds completed 30 epochs for six paired actual arms (91,800 optimizer steps). ORACLE_SUPPRESSION is exactly TARGETED and shares its result.

| Seed | Arm | Overall % | Macro % | Many % | Few % | Fixed Many margin |
|---|---|---|---|---|---|---|
| 123 | C2 | 46.4646 | 46.4646 | 62.6471 | 23.0000 | 4.915808 |
| 123 | BASE_C3 | 45.0505 | 45.0505 | 56.7647 | 27.6667 | 4.264955 |
| 123 | TARGETED | 44.7475 | 44.7475 | 58.5294 | 26.3333 | 4.766818 |
| 123 | RANDOM_MATCHED | 44.9495 | 44.9495 | 57.6471 | 27.0000 | 4.391490 |
| 123 | FREQUENCY_MATCHED | 44.9495 | 44.9495 | 56.7647 | 27.6667 | 4.306761 |
| 123 | ORACLE_SELECTION | 44.8485 | 44.8485 | 59.4118 | 26.3333 | 4.990144 |
| 124 | C2 | 45.7576 | 45.7576 | 61.7647 | 24.0000 | 4.897622 |
| 124 | BASE_C3 | 45.1515 | 45.1515 | 57.0588 | 28.3333 | 4.264147 |
| 124 | TARGETED | 45.1515 | 45.1515 | 58.5294 | 27.0000 | 4.765933 |
| 124 | RANDOM_MATCHED | 45.2525 | 45.2525 | 57.9412 | 28.0000 | 4.358571 |
| 124 | FREQUENCY_MATCHED | 45.2525 | 45.2525 | 57.9412 | 28.0000 | 4.333880 |
| 124 | ORACLE_SELECTION | 45.4545 | 45.4545 | 60.0000 | 26.6667 | 4.991300 |
| 125 | C2 | 45.8586 | 45.8586 | 63.8235 | 23.3333 | 4.920338 |
| 125 | BASE_C3 | 45.0505 | 45.0505 | 57.0588 | 28.3333 | 4.257823 |
| 125 | TARGETED | 45.0505 | 45.0505 | 58.8235 | 26.6667 | 4.762877 |
| 125 | RANDOM_MATCHED | 45.0505 | 45.0505 | 57.6471 | 28.0000 | 4.343251 |
| 125 | FREQUENCY_MATCHED | 44.8485 | 44.8485 | 57.3529 | 27.6667 | 4.340621 |
| 125 | ORACLE_SELECTION | 44.7475 | 44.7475 | 60.0000 | 25.6667 | 4.993893 |

Full mean, sample SD, paired effects and class-stratified bootstrap 95% intervals: `analysis/mean_SD_effects.csv`, `analysis/paired_deltas.csv`, `analysis/bootstrap_intervals.csv`.

TARGETED beats BOTH matched controls on BOTH primaries in 3/3 seeds. Controlled-intervention decision: **supported_under_predeclared_controlled_intervention_rule**.

ORACLE_SUPPRESSION: mean Many +1.6667pp; fixed margin +0.502901; Few -1.4444pp; both primary directions positive in 3/3 seeds. Preregistered practical-effect flag: False.

ORACLE_SELECTION: mean Many +2.8431pp; fixed margin +0.729471; Few -1.8889pp; both primary directions positive in 3/3 seeds. Preregistered practical-effect flag: True.

Oracle Suppression has zero distance from TARGETED by construction; this is not independent confirmation. Oracle Selection is a fixed-K retain objective, not a theoretical global bound. `ORACLE_SELECTION_minus_TARGETED` reports its observed gap.

Gate admission: **HUMAN_REVIEW_ONLY**. No Gate module was designed or trained.

## Evidence limits

- TRAIN-derived development split; parent work already examined the original 1980 holdout examples
- Clean Stage-I checkpoint was originally selected on that 1980-example holdout; this split cannot undo historical selection
- class 0 absent; Many metrics cover 34 of 35 TRAIN-Many classes
- conditional bootstrap uncertainty given three fixed training seeds, not population inference
- local analytic direct-gradient intervention; no new end-to-end objective or global upper-bound guarantee
- Oracle Selection retains K, while suppression removes K; different constraints must not be confused
- BASE_C3 autograd and intervention analytic decomposition have measured FP32 rounding residuals

All parent source and artifact fingerprints were rechecked at completion. Masks were sealed before Evaluation was opened and were never revised.
