# Semantic compatibility screen — completed limited-scope run

Read `CODEX_SEMANTIC_COMPATIBILITY_SCREEN.txt` and `SEMANTIC_SCREEN_PROTOCOL.md` first. The user explicitly approved the 210 eligible Few images in 21 classes after the full Few→Many coverage audit. Other Few classes and Medium donors were not substituted.

Canonical run: `results/semantic_screen_20261029_01/`.

- `SEMANTIC_COMPATIBILITY_REPORT.md`: scientific NO-GO, coverage finding, all 21 class results and limitations.
- `analysis_v2/`: final recipient/class/superclass tables, distance audit, decision and figure.
- `screen_output/`: untouched output of the supplied `screen.py`.
- `COVERAGE_*.json`, `*_COVERAGE.csv`: official TRAIN-only eligibility; the 21/30 class coverage is not a method-failure result.
- `ARTIFACT_INDEX.csv`: SHA256 and local-only/submission disposition. Numeric feature banks and pair input remain local in `primary_cache/`.

The Clean Stage-I checkpoint SHA256 is `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`, epoch184/seed42, TRAIN-validation-selected. Its historical model source, checkpoint, original TRAIN/meta and Clean LT cache/split must be available locally; these large historical assets are not uploaded by this study.

Actual commands and validation history are in `results/semantic_screen_20261029_01/VALIDATION_REPORT.md`. For recovery into a **new** result directory, use the same interpreter and seed/plan, never overwrite an existing result:

```powershell
$semanticPython = 'D:\app\minicondaaa\envs\xiaoju\python.exe'
$semanticRun = 'H2T_repro_audit/semantic_compatibility_screen/results/replay_new_unique_directory'
& $semanticPython -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/coverage.py --output $semanticRun
& $semanticPython -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/checkpoint.py --output $semanticRun
# Copy the unchanged SCOPE_RESOLUTION.json and PRE_ANALYSIS_PLAN.json from the
# canonical run into this new directory. Record this as replay of the old lock,
# not a new prospective registration or new authorization.
& $semanticPython -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/extract_features.py plan --output $semanticRun
& $semanticPython -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/extract_features.py pilot --output $semanticRun
& $semanticPython -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/extract_features.py features --output $semanticRun
if (Test-Path -LiteralPath "$semanticRun/screen_output") { throw 'Existing screen output; stop' }
& $semanticPython -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/screen.py --input "$semanticRun/primary_cache/pairs.npz" --output "$semanticRun/screen_output" --lambda-fixed 0.25
& $semanticPython -B -X utf8 H2T_repro_audit/semantic_compatibility_screen/analyze.py --output $semanticRun --analysis-output "$semanticRun/analysis_v2"
```

To recompute statistics from the current cached real input and original matched rows, call `analyze.py --output <canonical run> --analysis-output <canonical run>/new_unique_analysis_directory`; no model forward is needed. The supplied screen itself opens outputs with `w`, so its output-directory absence must be checked before calling it. All new helpers refuse their completed targets.

`verify_outputs.py` is the current run's one-time audit using its original local historical metadata baseline. A replay must establish a new before/after protection baseline rather than claim the old baseline proves a new run. Its validation is not a new scientific replicate.

The first analysis-export attempt stopped on NumPy boolean JSON serialization. Those partial outputs remain locally at the run root and are ignored; final results are exclusively in `analysis_v2/`. The repair preserved all numeric outputs byte-for-byte. No training or additional fusion mechanism is included.
