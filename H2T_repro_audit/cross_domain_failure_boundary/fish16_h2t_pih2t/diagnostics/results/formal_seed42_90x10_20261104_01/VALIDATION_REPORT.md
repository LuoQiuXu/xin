# Fish16 formal validation

Actual execution: COMPLETE_FORMAL_SELECTED_CONDITIONS. Exit code 0. One invocation; no resume/retry.

| Check | Result |
|---|---|
| A0 / B0 actual epochs and steps | 90 / 91,530 each |
| A1 / A2 / B1 / B2 actual epochs and steps | 10 / 10,170 each |
| Total committed epochs / physical logged updates | 220 / 223,740 |
| Attempted windows / actual SGD / gradient audits | 223,740 / 223,740 / 223,740 |
| Optimizer skips / logged NaN / logged Inf | 0 / 0 / 0 |
| Complete raw artifact index | 475 files; 2,463,214,977 bytes; all SHA256 and sizes exact |
| Final checkpoint model tensors | All finite; metadata, epoch, records and counters match |
| Common A0/B0 backbone/head initialization | Exact recorded hashes |
| Stage-I recipient + MixUp | All 90 paired epoch hashes exact |
| Stage-II recipient images/labels | All 4 conditions, all 10,170 batches exact |
| A2/B2 donor views | All 10,170 batches exact |
| Stage-II source | A0/B0 final Epoch 90 only; same source within each pair |
| Extractor + BN freeze | 4 independent endpoint checks exact; 40 runtime full-state gates completed |
| Training source / upstream source fingerprints | 6 / 8 unchanged |
| FP32 policy | Autocast off; disabled scaler; all 223,740 events have scale=1 |
| TEST evaluation / additional seeds | 0 / none |
| Postrun image reads / GPU inference / training | 0 / 0 / 0 |

CPU tests and syntax were **reused**, not rerun: CPU_VALIDATION.json PASS, 41 methods, syntax 14 files.
CPU record SHA256: dbf4fe04619a2e9d6f1e0bfc5a0ebd63272fdfdd262499de1edbc4a42d6bf012.
Preflight record SHA256: 40a9e8af8971223525fdf08b8a227f217cb79308b5c0e6df7e2d0bfbaa5e1626.
Their execution HEAD 960b1be and formal execution HEAD 072d7b77 differ only by archived prior results; actual source fingerprints match.

The CPU analysis verified all original raw files read-only, all 223,740 physical optimizer events, the exact final/best checkpoint metadata, complete frozen extractor state including BN, and batch view controls. Endpoint checks are independently computed; per-epoch freeze evidence comes from the unchanged runner's mandatory pre-commit gate, not fabricated standalone per-epoch hash files.

No new training tests, Smoke or numerical experiment was run. New CPU analysis completed with exit code 0. Matplotlib reported a denied optional user-profile font-cache write; in-memory plotting succeeded. Stage-II and per-class PNG layouts were visually inspected without clipping.

Important scientific limits: one seed, Validation only, 100 Tail images from clustered tracks, frozen BN unlike shift_bn=True, local PI-H2T reconstruction choices, no saved prediction frequencies/confusion/margins, no demonstrated causal root cause.

The two initial external-analysis launches were rejected because automatic permission review exceeded its deadline, before executing. Analysis was therefore safely completed in this workspace archive. This is an analysis-delivery issue; formal training itself completed normally and its raw outputs remained outside Git. A separate delivery record reports whether final external report copying succeeded.
