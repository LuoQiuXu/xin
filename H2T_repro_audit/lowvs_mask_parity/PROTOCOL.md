# Low-VS × Mask Granularity | Preregistered Pilot

## One scientific question

When Stage I, BN state, recipient/donor sampling, augmentations, optimizer,
training time, initial classifier, and mask granularity are strictly matched,
does a Low-VS candidate pool improve long-tailed classifier performance over
uniform random channel replacement?

This fills a missing control. Do not confuse "VS improves over U-Individual"
with "VS improves over author's original batch-shared random H2T".

## Four FRESH training arms (no historical accuracy reused)

| Arm | Randomization | Channel pool | Channel mask |
| --- | --- | --- | --- |
| U-Shared | first receiver's pre-locked 64-channel Gumbel | all 64 | one 16-channel mask shared by batch |
| VS-Shared | SAME batch Gumbel as U-Shared | lowest-48 channels by batch-mean eligible online VS score | one 16-channel mask shared by batch |
| U-Individual | each receiver's pre-locked Gumbel | all 64 | one 16-channel mask per sample |
| VS-Individual | each receiver's SAME Gumbel | lowest-48 by each receiver's VS score | one 16-channel mask per sample |

Batch VS score: for each current minibatch calculate the per-channel arithmetic
mean of the *reviewed individual* scores, excluding donors where n<20 or the
online score is degenerate. If no eligible donor exists, U-Shared is used.
For VS-Individual, existing n<20 / degenerate fallback remains U-Individual.
Do not change pool size 48, replacement count 16, or epsilon/shrinkage.

All four arms use an identical receiver/donor/augmentation/Gumbel schedule;
each policy chooses only the 16-channel mask. Everything else is matched.

**Important:** U-Shared is *original-like* in replacement operator and marginal
mask distribution, but its reproducible Gumbel random stream does NOT equal
the official H2T implementation's torch.randperm sequence. VS-Shared's
batch-mean scoring is a NEW policy definition and has no official predecessor.
Comparisons of "shared" and "individual" do NOT isolate aggregation from
granularity; within-granularity VS vs U comparisons are the primary results.

## Locked protocol

- Dataset: CIFAR-100-LT IF=100; 10,847 LT TRAIN images.
- Stage I: *same existing Clean seed42 epoch184 checkpoint* for every arm.
- Backbone: ResNet-32 [B,64,8,8] features; backbone and BN fixed/eval.
- Stage II: classifier only, 100 logits, CE, SGD, exact existing cosine scheduler.
- Seeds: Stage II 124 and 125 only. 10 epochs × 85 updates = 850 per arm.
- rho=0.25: 16 complete spatial channels replaced in all arms.
- Comparison: same 1980-image TRAIN-held-out dev bank (classes 1..99, 20/class),
  100 logits kept, class 0 absent. No official TEST access.
- Primary endpoint: *fixed epoch10*, not validation-selected best checkpoint.
- Historical results and old 2026-10-15 three-arm mask experiment are contextual
  only. Re-train all four arms in this same local Python/CUDA environment.

## Primary contrasts and predefined decision

1. VS-Shared minus U-Shared (test Low-VS under original-like shared-mask semantics).
2. VS-Individual minus U-Individual (reconfirm previous Low-VS finding within the same environment).

Evaluate Overall, Head, official-ID Medium, Tail, Few, and per-class paired
wrong-to-right/right-to-wrong counts for each Stage II seed separately.

**Provisional GO** to independent confirmation only if both primary contrasts
have Tail delta >= +0.5 percentage points AND Overall delta >= 0 pp at fixed
epoch10 for *both* seed124 and seed125. This is a conservative exploratory
screen, not a significance test. If only Tail improves at an Overall cost,
report the tradeoff and do not declare a method-level win. If the signs
disagree or primary requirement fails, do not search parameters on this
reused development set.

Additional descriptive contrasts:
- U-Individual - U-Shared (random mask granularity)
- VS-Individual - VS-Shared (granularity + VS aggregation)
- VS-Individual - U-Shared (confounded, NOT a scoring-only causal effect)

The interaction (individual VS-vs-U delta minus shared VS-vs-U delta) is
a descriptive interaction including batch score aggregation, not pure
mask granularity.

## Must-pass run integrity checks

- Verify the locked Clean Stage I, statistics, plans, source code and dataset
  checksums, and actual input-array shapes before GPU training.
- U-Shared must be executed first for each seed to create the fresh exact
  paired batch-signature reference.
- For each other arm assert 850/850 exact matches of PAIR_KEYS
  (indices, images, features, enhancement RNG, Gumbel, online GAP/S, ambient RNG).
  No waiver is allowed for numerical environment differences.
- Verify exact 16/64 nonduplicate spatial channel copying, unchanged recipient
  target, frozen BN/backbone throughout, and updates only to classifier.
- Reload saved FC checkpoints for a metrics equality check.
- Preserve original code, old checkpoints and reports. All new outputs must
  stay in lowvs_mask_parity/results. An incomplete run must STOP, not silently
  be replaced by another seed or environment.
- No evaluation on official CIFAR100 TEST during development or parameter
  selection; an input guard aborts if test loading is attempted.
- Do not commit raw image banks, large NPZ arrays, or model weights to GitHub.

## Pre-declared limitations

- Only one Stage I backbone/representation and two Stage II RNG streams.
- Reused 99-class TRAIN dev set is neither pristine held-out test nor a full
  100-class benchmark; every class has just 20 validation images.
- Fixed epoch10 improvements can vanish under best-epoch model selection.
- All experiments freeze Stage I and do not demonstrate end-to-end
  representation improvement.
- A GO demands an **independent** confirmatory evaluation with a fresh
  data split/retrained Stage I or genuinely new dataset, comparing a faithful
  original H2T implementation as well.
- A NO-GO is an informative scientific result: Low-VS's earlier improvement
  cannot be attributed confidently to VS scoring in this controlled setting.
