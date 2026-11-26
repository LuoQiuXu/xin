# Exact-V2 Estimator Alignment Audit Report

- Part A runs the **exact v2 online estimator** (`estimate_anchored_transition_and_prior`, `confidence_power=2.0`, `shrinkage_mass=10.0`, `prior_pseudocount=1.0`, `max_iter=200`, `tolerance=1e-6`) on the exact frozen Stage-1 `train_logits.npy` (`sha256: 6f82d1af1902722284da45214ea8412e1231da7938f2962737b14f6cae3699e8`) across all five regimes (`sym02`, `sym04`, `sym06`, `asym04_coarse_cycle`, `cifar100n_real`).
- Part B extracts the full 100-class and Many/Medium/Few trajectory from the completed online `v2 sym04` run's 36 `em_snapshots` (`Epoch 20..195`).

## Part A: 5-Regime Comparison (Old Stage-1 Joint EM vs Exact-V2 Anchored EM on Frozen v1 Teacher)

| Regime | Realized Noise | Est Noise (Old -> Exact-V2) | Converged (Old -> V2) | T Diag MAE (Old -> V2) | Prior KL (Old -> V2) | Few Mass Ratio (Old -> V2) | V2 Few Spearman | V2 Few CV (true 0.411) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| sym02 | 0.2028 | 0.1568 -> **0.2485** | False -> **True** (18 it) | 0.1178 -> **0.2200** | 0.0470 -> **0.0433** | 2.24x -> **2.70x** | 0.527 | 0.346 |
| sym04 | 0.4037 | 0.0540 -> **0.3120** | False -> **True** (12 it) | 0.3477 -> **0.1551** | 0.1105 -> **0.1238** | 4.67x -> **4.90x** | 0.325 | 0.175 |
| sym06 | 0.6096 | 0.5369 -> **0.5620** | False -> **True** (47 it) | 0.1307 -> **0.1108** | 0.1813 -> **0.1448** | 4.97x -> **4.92x** | 0.096 | 0.738 |
| asym04_coarse_cycle | 0.4037 | 0.3610 -> **0.2777** | False -> **True** (19 it) | 0.0717 -> **0.1621** | 0.0342 -> **0.0291** | 0.76x -> **1.81x** | 0.526 | 0.491 |
| cifar100n_real | 0.3597 | 0.3177 -> **0.4101** | False -> **True** (29 it) | 0.1295 -> **0.1687** | 0.0648 -> **0.0288** | 0.72x -> **1.18x** | 0.229 | 0.733 |

## Part B: Online v2 Sym-0.4 EMA-Teacher EM Trajectory (Realized Noise = 0.4037, True Few Mass = 2.81%, Few CV = 0.411)

| Epoch | Raw Noise | Blended Noise | Blended T Diag MAE | Blended Row TV | Blended Prior KL | Many/Med/Few Mass Ratio | Few Spearman (Raw / Bld) | Few CV (Raw / Bld) | Few <0.5x / >2x |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 20 | 0.6954 | 0.2458 | 0.1508 | 0.3318 | 0.0950 | 0.85x / 1.30x / 3.66x | 0.100 / 0.410 | 1.569 / 0.218 | 0 / 29 |
| 50 | 0.5665 | 0.5387 | 0.2075 | 0.4586 | 0.0761 | 0.89x / 1.30x / 2.54x | 0.150 / 0.261 | 1.697 / 0.927 | 3 / 14 |
| 100 | 0.5274 | 0.5131 | 0.1856 | 0.4286 | 0.0930 | 0.85x / 1.51x / 2.39x | 0.229 / 0.298 | 1.034 / 0.746 | 7 / 17 |
| 150 | 0.4894 | 0.4836 | 0.1696 | 0.4078 | 0.0749 | 0.87x / 1.37x / 2.76x | 0.251 / 0.372 | 2.317 / 1.042 | 7 / 20 |
| 195 | 0.4209 | 0.4271 | 0.1539 | 0.3831 | 0.0633 | 0.87x / 1.35x / 2.81x | 0.307 / 0.324 | 0.622 / 0.616 | 7 / 21 |
