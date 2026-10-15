# RAVS-H2T V1 — first method prototype (2026-10-15)

**Question:** Does receiver-conditioned VS candidate pool improve tail classes with less head loss compared with fixed Low-VS, and does the hardness gate add value over frequency-only?

VS auditing already confirmed that low/high scoring pools alter actual fusion channel identities and statistics. This experiment tests one trainable method prototype, **not** another scoring diagnostic.

## Predeclared algorithm

With frozen Clean Stage-I CE+MixUp ResNet-32 (seed42, TRAIN-only selected epoch184), Stage-II trains only the 100-way linear classifier. The underlying donor VS score, donor variance shrinkage, Uniform/Low-VS masks, and donor count n<20 fallback are reused unchanged from the reviewed Clean training sources.

For every augmented receiver (before fusion), obtain detached clean logits l from its current Stage-II classifier. The clean margin is m=l_y-max_{j != y} l_j, normalized hardness is h=sigmoid(-m/(std(l)+1e-8)); rarity is q=clip(log(500/n_y)/log(100),0,1).

RAVS-V1: omega=q*(0.5+0.5*h).
Frequency-Only: omega=0.75*q (same rule with h=0.5 fixed).
Pool K=clip(floor(64-16*omega+0.5),48,64).
Sample uniformly without replacement 16 channels from the **Bottom-K** donor-VS-scored pool with the same per-sample Gumbel stream. Replace exactly those 16 full 8x8 receiver feature-map channels; train on the original recipient hard label. Keep model, CE, donor, ratio and 10-epoch budget untouched.

**Hypothesis, not established fact:** higher normalized receiver hardness should benefit from stronger low-VS routing. If not, reject this version rather than search more thresholds.

## One new-environment five-arm experiment

| Arm | Intervention |
|---|---|
| No-Fusion | Original receiver feature; donor stream/forward still executed |
| Uniform-PerSample | Uniform16 of 64 |
| Low-VS | Reviewed Bottom48 to random16 |
| Frequency-Only | Receiver class rarity controls Bottom-K |
| RAVS-V1 | Rarity plus online receiver hardness controls Bottom-K |

Run all five at Stage-II seeds124 and125, 10 epochs each, exactly 850 optimizer updates per arm. Classifier starts from identical frozen Clean Stage-I classifier within each seed. BN/backbone remain eval and frozen. Same receiver/donor plans, augmentations, VS stats, Gumbel stream and runtime for all five new runs.

No-Fusion must execute first per seed and records indices/images/features/scores/noise hashes for all batches. Other arms compare each batch before gradient update and abort on mismatch. Older experiments' 47–48% results must not be called paired to these new experiments.

Development evaluation: 1980 original TRAIN holdout images, 20/class over class1..99, class0 absent, all 100 logits kept. Official TEST images/labels are forbidden. Fixed epoch10 is the primary comparison; best epoch is diagnostic. The reused development holdout is **not** an independent final test.

## Entrypoint

From local repository root, in the user's CUDA-capable Python environment:

- python H2T_repro_audit/ravs_v1/run_ravs_v1.py preflight
- Run the 10 precisely ordered commands in CODEX_RAVS_H2T_V1_INSTRUCTIONS.txt, starting with No-Fusion for seed124.
- python H2T_repro_audit/ravs_v1/run_ravs_v1.py report

Results appear exclusively in H2T_repro_audit/ravs_v1/results/. The instructions prohibit Codex from **all version-control operations**; the user controls that independently.

## Exploratory gates (not significance claims)

Gate1: RAVS vs Low-VS: both seeds Overall nonnegative, average +0.15pp or better, average Tail and Head nonnegative.
Gate2: RAVS vs Frequency-Only: consistent direction and average Overall +0.10pp or more, OR no Overall mean loss and mean Tail at least +0.35pp.
Only when both gates pass does a receiver-hardness module merit further independent verification. Otherwise stop; no adaptive search over K, new seeds, thresholds, losses or scores on this repeatedly used holdout.

Current intervention changes Stage-II **classifier training**, not learned backbone representation. This is not full PI-H2T / PIF and does not establish final CIFAR-100-LT performance.
