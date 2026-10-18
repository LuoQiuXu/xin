# Low-VS fixed + Tail Clean Discriminative Preservation (controlled pilot)

**We are building ON TOP of Low-VS, not tuning, optimizing, denying or replacing its channel policy.** This is the first minimal test of a distinct complementary objective. Two linked observations motivate it:

1. Existing CIFAR100-LT IF100 Clean Stage-II seed124/125: LowVS-Full beats Uniform-PerSample by approximately **+0.28 pp Overall / +0.95 pp Tail**, while disproportionately modifying the Head/Tail decision trade-off.
2. IF200 shows original-style H2T mainly reduces Head-overprediction and redistributes predictions to Medium/Tail, but many erroneous Head predictions become *other* erroneous classes, not correct Tail labels. Low-VS's stronger donor-conditioned interference does not consistently add Tail accuracy at IF200.

**Specific hypothesis:** Preserve discrimination of *clean tail receivers versus their hardest existing competing class*, while keeping exactly the frozen historical Low-VS cross-class perturbation. Such a complementary constraint might convert changes in predicted category-group allocation into more actual correct tail predictions. That is a hypothesis, NOT an established causal chain.

## Three arms ONLY

All share the same untouched Stage-I frozen backbone+BN and Stage-II online Low-VS hard masks (Bottom48 → random16), same paired receiver/donor crops and Gumbel stream, same training steps, same full fused CE. Frozen means the *representation does not learn*: this is an isolated linear classifier experiment, not a claim of end-to-end representation improvement.

| Arm | Training objective | What it tests |
|---|---|---|
| LowVS | CE on fused labeled feature only | historical frozen Low-VS positive control |
| LowVS-CleanCE | LowVS + λ × **tail-only CE** on original (non-fused) receiver | Added supervision / clean receiver anchor |
| LowVS-TopCompetitor | LowVS + λ × **tail-only log(1+exp(l_j−l_y))** on non-fused receiver | Explicit strongest competitor separation |

Exactly the same 100-way classifier, same tail receiver subset, same clean feature forward and same fixed coefficient **λ=0.5**, no tunable margin (\`m=0\`). Both auxiliary losses are **SUM over tail samples / FULL batch size**, rather than dividing by changing tail count. \`j\` is chosen from the CURRENT clean classifier logits at each step, with the true class excluded and argmax detached before pairwise softplus; it is **never** used to adjust any VS score, channel mask, donor or hard label.

This is a minimal *prior art-compatible* ranking loss pilot. Pairwise highest-negative ranking/margin objectives exist widely; even if the third arm improves performance, it is not by itself a SCI-level novel method. Aux CE and ranking softplus are different functions with potentially different gradient magnitudes; equal λ is a same-coefficient control, NOT full gradient-dose matching. Interpret causality accordingly.

## Predeclared outcome and stop logic

**Primary contrast**: **LowVS-TopCompetitor minus LowVS-CleanCE** at fixed epoch10 on Tail, Overall, Head, and exact paired gain/loss images, separately seed124 and seed125. Check whether it rescues correct Tail *identities* rather than merely reducing Tail→Head confusion.

**Secondary**: each auxiliary arm vs LowVS. All three arms must agree on 850 historical batch source/mask hashes per seed and start at exact same Stage-I classifier weights. We repeat the original LowVS control ONCE per seed to confirm historical run consistency. DO NOT choose best epoch; diagnostic curves may be shown but results judged only at epoch10.

Outcomes:
- Competitor > clean-CE in both seeds on Tail with no material Overall/Head regression and more paired correct Tail labels: supports follow-up as a complementary supervision direction (not publication proof).
- Competitor ≈ clean-CE: extra clean supervision explains apparent gains; no special-competitor claim.
- Both add-ons worsen or disagreement between seeds is comparable to 1–3 images: stop this exact proposal; do not start margin/λ sweeps.

**Important**: exact same repeatedly used 1,980-image original-TRAIN development split (classes1..99, 20 each, class0 absent); no official TEST; one Stage-I seed42 epoch184; just two dependent Stage-II seeds. This experiment cannot demonstrate independent test generalization or backbone representation learning.

## Training resources and protocol

Reuses local historical source material in:
\`H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014/\`
and prior actual training signatures in:
\`H2T_repro_audit/lowvs_norm_direction/results/seed{124,125}/LowVS-Full/batch_signatures.csv\`.

The source checkpoint SHA and VS-stats SHA are asserted by the reused historical helper. Their content is read-only; no old source files are modified. Existing official TEST guard is installed for each action.

Only 6 new classifier-only StageII runs (3 arms × 2 seeds, 10epochs×85SGD steps), no retraining StageI, full test-only CPU loss self-test.

From repository ROOT, in order:

~~~text
python H2T_repro_audit/lowvs_disc_preserve/losses.py
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py preflight
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py train --seed 124 --arm LowVS
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py train --seed 124 --arm LowVS-CleanCE
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py train --seed 124 --arm LowVS-TopCompetitor
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py train --seed 125 --arm LowVS
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py train --seed 125 --arm LowVS-CleanCE
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py train --seed 125 --arm LowVS-TopCompetitor
python H2T_repro_audit/lowvs_disc_preserve/run_pilot.py report
~~~

Results under this directory's **results/** only. The old replay is 850 steps per seed, and every new arm's batch is checked for exact 7-field hash equality to that old experiment AND the current LowVS baseline (receiver/donor IDs, augmented pixels, frozen features, online VS scores, Gumbel, both masks).

Outputs: preflight.json, per-arm final_fc.pth, epoch10 predictions/logits, epochs.csv, batch_signatures.csv, summary.json; aggregate epoch10.csv, paired.csv, per_class.csv, curves.csv, scientific_summary.json, LOWVS_DISC_RESULTS.md.

Codex instructions are in \`CODEX_LOWVS_DISC_PRESERVE.txt\`. **All Git/version control is prohibited for Codex**; the user handles it separately.

If the pilot supports a meaningful complementary module, subsequent experiments should explicitly ask whether the same discriminator helps when real learned backbone representation is optimized, with a new untouched validation/test protocol and stronger baselines. Do not call this pilot a finished new method.
