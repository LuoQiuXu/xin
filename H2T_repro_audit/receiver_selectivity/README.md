# H2T receiver-selectivity ablation — one scientific question only

## Primary scientific question

Original H2T applies a batch-shared, randomly selected 16/64-channel hard replacement to **all** class-balanced receiver features. IF200 fixed-BN historical results show positive aggregate tail performance, with a few small locally harmed tail classes. The key question is not "can we build a smarter module?" but:

> Is direct H2T fusion of tail-class receivers necessary? When we skip it, is the effect specific to skipping **tail** receivers rather than merely lowering total fusion exposure?

We test a **single treatment factor** — which receivers receive the unchanged H2T mask. No new score, attention, loss, donor selection, margin protection or architecture.

## Four arms, one isolated contrast

| Arm | Receiver fusion rule | Purpose |
|---|---|---|
| `DR-NoFusion` | All off | No-fusion reference |
| `Original-BatchShared` | All on | H2T-like whole-batch 16/64 reference |
| `TailOff` | Off for labels 71..99 only | Proposed failure-condition probe |
| `RandomOffMatched` | Uniform random off for **exactly the same number** of receiver draws as TailOff, separately in every minibatch | Dose-matched counterfactual |

**Primary contrast:** `TailOff - RandomOffMatched`, fixed epoch 10, Tail accuracy. The total number of augmented receiver examples per batch is identical in these two arms; the identities differ. Contrasts to original/no-fusion are descriptive checks, not replacements for the dose-matched primary comparison.

The tail-class boundary of 71..99 follows the locked IF200 historical protocol. The four historically harmed classes 72,77,79,87 may be listed after analysis but must not become the treated set.

## Locked protocol

- CIFAR-100-LT IF200, true raw TRAIN, 9502 LT training instances, ResNet-32, frozen Stage-I seed42 checkpoint at fixed epoch200.
- Stage-II seeds 124 and 125, each four arms, 10 epochs × 75 SGD updates. 100-way FC, 6500 parameters. CIFAR author IF200 Stage-II optimizer with effective first learning rate 0.02, cosine schedule.
- BN **fully frozen and eval**. The author's Stage-II YAML sets `shift_bn=True`; results therefore **cannot** be described as full original H2T replication.
- Instance-wise donor draw, balanced receiver draw, augmentation and a 16/64 batch-shared H2T mask are held constant within each seed across arms.
- Validation is the previously reused 1980-image holdout from CIFAR official TRAIN (classes1..99, 20 each; class0 absent), not the official TEST. This remains exploratory: previous use of the holdout and only one Stage-I checkpoint limit certainty.
- Paired inputs/features/donor/mask array SHA hashes must match within the **fresh** same-run DR reference; DO NOT demand bit identity with earlier experiments across changed execution paths. Never relax within-run matching.
- No new Stage-I training; no official CIFAR TEST access. Stop if historical TRAIN-only files or frozen IF200 checkpoint are absent.
- Fresh run outputs only under this directory's `results/<run-id>/` and refuse an existing directory.

## Pre-registered exploratory decision rule

Only report `CONDITIONAL_SIGNAL_ONLY` if all hold:

1. `TailOff - RandomOffMatched` Tail pp strictly positive at **both** seeds;
2. arithmetic mean Tail difference at least **+0.5 pp**;
3. Overall difference at least **−0.25 pp** at **each** seed.

Otherwise `NO_SUPPORT_STOP`. Neither case establishes independent statistical significance or generalization. Do **not** change these thresholds after seeing results.

## Execute from repository root

```powershell
python -m unittest discover -s H2T_repro_audit/receiver_selectivity -p "test_*.py" -v
python H2T_repro_audit/receiver_selectivity/run_experiment.py self-test --run-id receiver_selectivity_20261026_01
python H2T_repro_audit/receiver_selectivity/run_experiment.py preflight --run-id receiver_selectivity_20261026_01
python H2T_repro_audit/receiver_selectivity/run_experiment.py run-all --run-id receiver_selectivity_20261026_01
```

If `python` does not refer to the pre-existing tested GPU environment, use the exact environment interpreter documented in historical IF200 reports. Never silently change the experimental PyTorch/CUDA or dataset protocol.

The `run-all` operation performs 8 Stage-II arms in deterministic order, saving final FC and validation predictions per arm. It generates `epoch10_summary.csv`, `per_class_pairing.csv`, `analysis.json`, and `SCIENTIFIC_REPORT.md`. It does not auto-select best epochs.

See `CODEX_H2T_RECEIVER_FAILURE_TEST.txt` for the complete Codex instruction. Existing code or result trees, especially `H2T_official_clean/`, `H2T_repro_audit/if200_stress/results/`, and previous experiments, must remain unchanged.

## Interpretation guardrails

- The repeated per-class harm signal involves at most a few examples per class. It is not confirmation of a class-specific failure mechanism.
- Skipping direct fusion from a subset of receivers changes all classifier gradients and therefore can affect all test classes. A favorable result alone does not prove the skipped examples were intrinsically "bad" or that individual negative margins caused the outcome.
- A matched random skip controls global augmentation *dose*, not every possible confound (notably class-conditional fusion frequencies and nonlinear SGD interactions).
- No adaptive rule, class-ID exception, new losses, or additional seeds are to be designed in this round. A negative result is a valid reason to stop this hypothesis.
