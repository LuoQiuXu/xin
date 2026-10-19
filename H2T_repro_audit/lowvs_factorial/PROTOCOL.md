# Low-VS Factorial Pilot: Donor Affinity × Margin-Risk Channel Protection

Status: **pre-registered exploratory pilot; not a claimed new method**  
Branch: experiment/donor-affinity-margin-protect-20261019  
Dataset: CIFAR-100-LT, IF=100; ResNet-32 Stage-I checkpoint frozen  
Stage-II seeds: 124, 125; fixed endpoint: epoch 10  
Compute target: RTX 5070 Laptop, 8 GB VRAM; batch size 128; classifier-only updates

## 1. Why this is the next experiment

The previous gradient-utility policy did not pass its predeclared continuation gate. The prior clean-receiver auxiliary objectives increased Tail accuracy in some comparisons but reduced Overall/Head; the TopCompetitor objective did not beat ordinary clean CE on Tail in both seeds. We therefore do not continue tuning those losses.

The current question is whether two feature-fusion decisions matter independently:

1. **Donor affinity:** Does a head-class donor whose frozen Stage-I representation is more compatible with a particular tail receiver yield better augmentation than a frequency-stratified random head donor?
2. **Margin-risk channel protection:** Given the exact same donor and original Low-VS Bottom48 candidate pool, does excluding channels whose proposed swap is predicted to reduce the receiver's current true-vs-strongest-competitor margin improve learning?

This is a 2×2 mechanism pilot. It does not establish that any effect generalizes beyond this fixed Stage-I checkpoint and reused development split.

## 2. Four trained arms and one historical reference

| Arm | Tail-receiver donor policy | Tail-receiver channel policy |
|---|---|---|
| A: RandomDonor | Frequency-stratified random head donor | Original Low-VS Bottom48 → Gumbel-random 16 |
| B: HighAffinityDonor | Top-affinity head donor within the same frequency stratum | Original Low-VS Bottom48 → Gumbel-random 16 |
| C: RandomDonor+MarginProtect | Exactly the same donor indices as A | Protect 16 least-favorable predicted margin-transfer channels inside Bottom48; choose 16 from the remaining 32 |
| D: HighAffinityDonor+MarginProtect | Exactly the same donor indices as B | Same protection rule as C |

E: Historical LowVS-Full is the already completed, source-verified baseline in H2T_repro_audit/lowvs_disc_preserve/results/. **Do not retrain E.** Show it as contextual reference only; A differs from E because A deliberately conditions tail-receiver donors on head classes.

Only receiver draws with label >= 71 receive the experimental donor override and/or protection. All other receiver draws retain the original planned instance-sampled donor and original Low-VS mask policy. The receiver label is never changed.

## 3. Donor affinity without a tail-prototype bottleneck

Do not use validation or official TEST features to build the reference bank.

1. With the frozen Stage-I backbone and the existing deterministic evaluation transform, compute a GAP-64 feature for every image in the existing CIFAR-100-LT TRAIN subset. Reuse a valid source-locked bank if one already exists; otherwise write a new bank only under this experiment's results directory.
2. Build L2-normalized mean prototypes for head classes 0–35 from those TRAIN features. This uses many samples for each head prototype and avoids using a five-example tail-class prototype.
3. For each tail receiver's clean TRAIN feature, compute cosine affinity to all 36 head prototypes.
4. Sort head classes by training count and partition them into six rank strata of six classes each. For each receiver draw, sample a stratum with probability proportional to the sum of class counts in that stratum.
5. A selects a class from the chosen stratum with probability proportional to its training count. B selects from the two highest-affinity classes within the same stratum, again weighted by training count. This preserves the stratum-level donor-frequency prior and uses common random numbers for A/B; it does not guarantee identical class-frequency distributions within each stratum, so log and report the realized donor counts and selected affinities.
6. Select a donor image uniformly within the chosen class. A/B share the same u_stratum, u_class, and u_instance random variates; only the candidate set changes. C reuses A's donor indices exactly; D reuses B's.
7. The chosen donor indices are precomputed for all ten epochs and saved with a hash. Do not reselect donors based on validation metrics or the evolving classifier.

The affinity treatment is sample-conditioned feature-space affinity, not semantic understanding. The Bottom-affinity policy may be used for a no-training diagnostic only; it is not an additional trained arm.

## 4. Margin-risk channel protection: preserve the original Low-VS pool

For each tail receiver, after the frozen backbone has produced receiver and selected-donor feature maps, let f_d and g_d be their spatially averaged channel values. Let y be the true receiver class and let j* be the strongest non-target class predicted on the unfused receiver by the current Stage-II classifier.

For each channel define:

\[
q_d = (W_{y,d}-W_{j^*,d})(g_d-f_d).
\]

Because the classifier is linear after GAP, sum_d q_d is the exact change in the fixed y-versus-j* margin if all channels were replaced, with j* held fixed. A negative q_d means that swapping this channel is predicted to reduce that particular margin. This is an immediate, classifier-conditional proxy, not a guarantee about training utility.

For C/D:

1. Reuse the original online donor-statistic Low-VS score, exact Bottom48 candidate set, original Gumbel values, and 16-channel replacement count.
2. On the normal Low-VS path, among the 48 Bottom-VS candidates, protect the 16 channels with the smallest q_d (the least favorable predicted margin-transfer contributions).
3. Select 16 channels from the remaining 32 using the same Gumbel values used by the corresponding A/B control.
4. If the original helper activates its pre-existing fallback (e.g., insufficient donor count or flat/non-finite score), preserve the original fallback mask unchanged for that draw and log it. Do not force a Bottom48 pool or apply protection to a fallback draw. Non-finite inputs that the original helper treats as fatal must remain fatal.
4. Replace those 16 whole 8×8 feature maps with donor channels. No interpolation, no extra loss, no trainable gate, no change to the original Low-VS score or fallback rule.

Do not choose the replacement channels from all 64 channels outside the protected set. That would change the Low-VS candidate policy and confound the protection effect.

This intervention tests whether protecting the least favorable transfers helps. It does not claim that channels are intrinsically semantic or universally important.

## 5. Fixed training and data protocol

- Stage-I: existing clean seed-42 checkpoint, selected at epoch 184 by the historical TRAIN-only development split; backbone and all BN parameters/buffers frozen in eval mode.
- Stage-II seeds: 124 and 125.
- Epochs: 10; batch size: 128; 85 updates/epoch; 850 real optimizer updates per arm/seed.
- Optimizer, scheduler, initialization, receiver plan, transforms, Gumbel stream, original online VS math and fallback: reuse the existing ravs_v1/sealed local helpers without changing their definitions.
- Train only the existing 100-way linear classifier. No auxiliary loss.
- Fixed primary endpoint: epoch 10. Do not select best epoch for the primary comparison.
- Main comparisons: B−A (donor affinity), C−A (channel protection under random donors), D−B (protection under high affinity), D−B−C+A (Tail interaction).
- Report each seed separately. No p-values or claims of statistical significance from two Stage-II seeds.
- Official CIFAR-100 TEST is prohibited: no loading, decoding, hashing, evaluating, or model selection with it.

The available development set is 1,980 images from original TRAIN, 20 per class for IDs 1–99; class 0 is absent. It has been reused in previous iterations. All validation-based results are exploratory and cannot be reported as independent test generalization.

## 6. Required diagnostics

### Preflight / static checks
- Verify the source checkpoint, cached arrays, receiver plan, Gumbel arrays and old LowVS-Full signatures against the existing source manifests. Do not manufacture or regenerate historical inputs.
- Verify prototypes use only the LT TRAIN subset and exactly match the frozen Stage-I hash.
- Verify A/B share receiver IDs and frequency-stratum draws; C/A and D/B have identical donor IDs.
- Verify selected donor labels match chosen donor classes and all donor image indices belong to LT TRAIN.
- Verify every candidate pool has exactly 48 channels, every mask selects exactly 16 channels, all protected channels belong to Bottom48, and no protected channel is selected.
- Verify the sum of q_d matches the exact fixed-competitor margin delta to numerical tolerance on synthetic CPU tests.
- Verify the backbone, BN state and initial classifier state before training; only the classifier may change.
- Install the existing official-TEST guard and write all new files only below this directory's results/.

### Per-batch diagnostics
Before each optimizer update, on tail receiver draws log:
- mean and quantiles of the unfused receiver margin;
- fixed-competitor margin change and dynamic margin change after fusion;
- fused-minus-clean CE change;
- fraction of positive clean margins that become non-positive;
- selected-channel sum of q_d, and the protected-channel q_d distribution;
- donor class ID, frequency stratum, affinity, and donor class count;
- selected-channel count and candidate-pool integrity.

These diagnose immediate geometry; they are not substitutes for final classification metrics.

### Epoch-10 results
Report Overall, Head, OfficialMedium, Tail, Many, Medium, Few, Macro, validation CE; paired image-level wins/losses/net correct; Tail-only wins/losses/net; per-class changes; true-group versus predicted-group confusion; donor frequency/affinity summaries; and both factorial contrasts and the interaction term.

One validation image equals 0.0505 percentage points Overall, 0.1429 Head points, or 0.1724 Tail points under the 99-class split. Interpret tiny differences as a few image flips, not robust gains.

## 7. Predeclared decision rules

- **Donor affinity supported for follow-up:** B−A and D−C have positive Tail net-correct changes in both seeds, and the gain is not merely Tail-to-other-wrong-class redistribution; Overall/Head costs are also shown.
- **Margin protection supported for follow-up:** C−A and D−B have positive Tail net-correct changes in both seeds, with the online diagnostics confirming that the rule actually removes negative/least-favorable q_d transfers.
- **Complementarity supported:** the Tail interaction D−B−C+A is positive in both seeds and D improves over A without a material Overall/Head regression.
- If D improves only over A but not over B or C, do not claim synergy; prefer the simpler effective factor.
- If directions flip across seeds or gains correspond to only one or two images, classify the result as inconclusive and stop this exact pilot. Do not sweep strata, top-K, protect-K, epochs, coefficients or seeds on this reused development set.
- If all new arms fail to improve on A, stop this candidate mechanism rather than adding losses or making the pipeline more complex.

Even a positive pilot result justifies only a follow-up with an untouched validation protocol and additional seeds; it does not establish a publishable method or end-to-end representation improvement.
