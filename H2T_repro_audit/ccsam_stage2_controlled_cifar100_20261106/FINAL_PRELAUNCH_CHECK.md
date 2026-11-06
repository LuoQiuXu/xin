# Final CC-SAM Stage II prelaunch check

READY_FOR_FORMAL_TRAINING

Branch: `experiment/ccsam-gradient-numerics-fix-20261106`; checked HEAD: `cc7e60a29068a3548cf2fefd6ebd776c1eb9d8c8`.

| Check | Result |
| --- | --- |
| Branch and baseline | PASS |
| Native formal-entry preconditions | PASS |
| Smoke report original bytes | PASS |
| Smoke result flags | PASS |
| Smoke/prepared manifest binding | PASS |
| Unchanged train_ccsam.py | PASS |
| Unchanged ccsam_common.py | PASS |
| Unchanged experiment_protocol.json | PASS |
| Clean Stage I checkpoint | PASS |
| Prepared train_images.npy | PASS |
| Prepared train_labels.npy | PASS |
| Prepared validation_images.npy | PASS |
| Prepared validation_labels.npy | PASS |
| Prepared validation_gap_features.npy | PASS |
| Sampling plan seed 123 | PASS |
| Sampling plan seed 124 | PASS |
| Sampling plan seed 125 | PASS |
| C0/C1/C2 and frozen hyperparameters | PASS |
| Alpha schedule | PASS |
| Output paths have no old runs or conflicts | PASS |
| Output parent writable | PASS |
| Backbone/BN freeze and checkpoint state | PASS |
| Validated numerical environment | PASS |
| Historical results, inputs and sources protected | PASS |
| Official TEST isolation | PASS |
| No Smoke rerun, forward or formal training | PASS |

The existing 2026-11-06 22:44:34 passing Smoke report was copied byte-for-byte; it was not regenerated or rerun.
Both copies have SHA256 `7c7973531cc4158e4f9e28273eaf2446021cd766b28c5dfd204863906035968e`.
Prepared manifest SHA256: `c4ecbd487c33052fdc20f642325d8c86d9f18efa6e75622d52e8055d85806e02`.
Clean checkpoint SHA256: `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`.

The native main function was never called. Its original pre-loop statements were executed in an isolated function, stopping before `for seed in SEEDS`; the training loop and subsequent statements were not compiled into that function. Smoke/training callbacks were blocked and never invoked. This proves the existing formal gates can read the installed report without starting a run.

The current model check only constructed/loaded the sealed model and inspected frozen CPU state. No forward, backward or optimizer was executed. Real forward/BN invariance and gradient parity remain supported by the immutable validation_20261106_05 evidence: 510 real batches, 10 stress cases, and C0/C1/C2 Smoke PASS.

Authorized preparation edits: README status/usage wording, byte-identical root smoke_report.json, and a narrow .gitattributes rule to preserve the Smoke JSON bytes across Git checkouts. Gradient code, common implementation, protocol, data, sampler, checkpoint and old reports remain unchanged.

Detailed PASS/FAIL evidence, input hashes, frozen-model state, native gate execution and TEST counters: `prelaunch_20261106_01/CHECK_RESULT.json`.

Formal training has NOT started. No results directory was created. Starting the nine runs requires a new user confirmation.
