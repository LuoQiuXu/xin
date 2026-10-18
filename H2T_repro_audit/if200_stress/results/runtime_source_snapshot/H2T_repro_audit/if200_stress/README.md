# H2T IF=200 stress test: when does cross-class fusion become negative transfer?

**Purpose:** stop inventing q/VS-based gates. Test whether Original H2T causes actual negative transfer relative to classifier-only DR when the CIFAR-100-LT tail becomes scarcer (IF200). Determine whether Low-VS helps *beyond* per-sample random channels, without conflating mask granularity.

This is an **exploratory, fixed-BN Stage-II experiment**, not an exact reproduction of every author setting. In the author's CIFAR IF200 stage-II config, \`shift_bn=True\`; previous Clean Low-VS findings have **fixed BN**. This run fixes BN deliberately to isolate the 100-way FC changes. We must never attribute a failure in this branch to the author's *full shift-BN method*. An author-configuration confirmation is required if a reproducible failure is found.

## Why IF200 first, not IF50?

The current, integrity-checked IF100 LT TRAIN subset contains 10,847 samples using the author's per-class seed0 shuffle. IF200 is the **per-class prefix** of that exact shuffled list, of **9,502** samples, class counts \`int(500 * (0.005 ** (c/99)))\`. Thus IF200 training raw indices are an exact nested subset of the existing IF100 LT TRAIN raw indices, and remain disjoint from the existing 1,980-image *original TRAIN* holdout (20 images from classes 1..99, class0 absent). Source files are SHA-locked after preflight.

IF50 would enlarge certain classes and may overlap that holdout: cannot construct it by blindly adding samples without rebuilding the split. Do not opportunistically start IF50 after seeing results.

## Fixed experimental budget

**Stage I:** 1 newly randomized ResNet-32 + 100-way FC (seed42), 200 epochs exactly, CE+MixUp(1), author SGD LR schedule: warmup5, drops after epochs160/180, initial LR .1, momentum .9, weight decay 2e-4, batch128, 75 updates/epoch. Fixed epoch200 checkpoint; *not selected* by validation or TEST. No IF100 checkpoint adaptation, no model freezing in StageI. Epoch200 validation is descriptive only.

Compute **IF200 TRAIN-only** class mean/shrunken variance and epsilon using the newly trained frozen StageI; cache held-out TRAIN validation feature maps solely for evaluation. For 13 classes with 2–4 training samples, apply the previously used variance shrinkage and the established fallback \`donor n<20\` rather than estimating raw variance unregularized.

**Stage II:** same newly trained StageI and same IF200 TRAIN statistics in every arm; only FC SGD is trainable, all Backbone/BN frozen and immutable, seed124/125, epoch10 fixed, 75 updates/epoch. Author IF200 YAML specifies LR .2, \`lr_factor=.1\`, WD 2e-4, batch128. With the original single FC param group, effective LR begins at .02 after the factor. All four arms share one per-seed plan of class-uniform receiver sampling (BS), instance-permutation donor sampling (IS), actual crop/flip images, Gumbel and the source pair, with per-batch combined content hashes; abort any mismatch.

| Arm | What is varied? |
|---|---|
| DR-NoFusion | original receiver features only |
| Original-BatchShared | single random 16/64 channel mask shared across batch |
| Uniform-PerSample | independently random 16/64 channels for each receiver |
| LowVS-PerSample | per receiver Bottom48 VS pool → random16 complete 8×8 channels |

The two truly targeted and preregistered contrasts are:
1. **Original-BatchShared − DR-NoFusion**: Is the original-style *feature fusion intervention* helpful or damaging in IF200? (under fixed BN).
2. **LowVS-PerSample − Uniform-PerSample**: Are donor-class VS-based channel identities valuable beyond unconditioned random channels?

Third, **Uniform-PerSample − Original-BatchShared** diagnoses mask granularity. **LowVS-PerSample − Original-BatchShared is a confounded contrast** (VS selection **and** granularity); don't package it as the VS-only gain.

## Evaluation and decision rules

Only the same historic **99-class original-TRAIN dev holdout** is evaluated (20/class, class0 absent, but 100 output logits). Never load official CIFAR TEST or optimize on these validation images.

Primary endpoint **epoch10** per seed, Overall, Head, OfficialMedium, Tail, Many/Medium/Few, Macro. Include paired correct→wrong and wrong→correct counts, true Tail→predicted Head, true Head→predicted Tail, worst-class accuracy with denominator 20/class. Save final checkpoint and 1980×100 logits for each arm.

**Failure evidence gate:** If Original-BatchShared causes a Tail regression relative to DR in **both** seeds, especially consistent per-class loss or stronger tail→head errors, label **"candidate negative-transfer condition at IF200 fixed-BN"**. Do not call this full original-H2T failure yet; confirm author \`shift_bn=True\` and new StageI seeds before claiming broad defects.

If LowVS beats Uniform only in one seed, do not design a new Low-VS gate. If both settings are broadly positive, this IF200 run did not find a failure; stop, don't automatically add IF50 or an external dataset.

Cross-IF comparisons with historical IF100 must be clearly marked **qualitative** because the StageI checkpoint selection and setup differ (IF100 chose best dev epoch184; IF200 uses fixed epoch200), and the holdout has been re-used repeatedly. Never claim a statistically significant interaction across IF from these data alone.

## Minimal local execution

Follow \`CODEX_H2T_IF200_STRESS.txt\`: one preflight, **one 200-epoch StageI**, one feature/statistics preparation, **8 paired StageII FC-only runs**, one final report. No hyperparameter scanning or new models. GPU memory: original CIFAR ResNet32 with batch128 on a laptop 8GB GPU; stop on OOM and disclose; do not silently alter the batch size.

Writes ONLY in \`H2T_repro_audit/if200_stress/results/\`, source IF100 artifacts remain read-only. User handles **all Git operations** (Codex is forbidden from Git/GitHub, branch/commit/PR tasks).

After uploading results, the next scientific decision is whether any observed candidate loss survives comparison with the author's actual BN configuration and a second StageI seed—not proposing another adaptive classifier trick.
