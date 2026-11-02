# A0/B0 Actual Execution Source Audit

Date: 2026-11-02. CPU-only source/AST review; neither smoke.py nor any training/forward was executed.

A0 execution: `8cbfa69bd194083053917bb0cef615572abdc182`; B0 execution: `d47cbd7293402fd02634c176d7e16db288b936b9`.
Report publications are different: A0 `52c8bf00d55966d0fa7d02fe52f788736083a02c`, B0 `de82e100e200de575380feba540a62ff9e94eca6`. The current audit started at `de82e100e200de575380feba540a62ff9e94eca6`.

## File and immutable upstream evidence

| File | A0/B0 Git blob equality | Interpretation |
|---|---|---|
| models.py | True | Unchanged common initialization/adapters |
| data.py | True | Unchanged manifests, views, transforms and sampling |
| preflight.py | True | Unchanged locked branch/data checks |
| amp_recovery.py | False | Signed PIF-b recovery policy plus structured diagnostics |
| smoke.py | False | Diagnostics, bounded/B0-only entries, accounting/status changes; numerical Stage-I core preserved |

Every recorded execution-file hash is reconciled with its exact Git commit. models.py/data.py use LF bytes; preflight.py/amp_recovery.py/smoke.py recorded CRLF worktree bytes. The LF Git blob converted to CRLF reproduces each execution hash. Line-ending normalization is explicit, not an ignored source mismatch.
All eight recorded upstream H2T/PI-H2T source hashes agree across runs and with the protected current files; their corresponding Git blobs also match. Git diff contains no changes to either upstream directory. The complete raw diff and exact Git source copies remain external under source_evidence/.

## Stage-I numerical path

| Evidence | A0 | B0 |
|---|---|---|
| RNG reset | [smoke.py:53](https://github.com/LuoQiuXu/xin/blob/8cbfa69bd194083053917bb0cef615572abdc182/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L53) | [smoke.py:57](https://github.com/LuoQiuXu/xin/blob/d47cbd7293402fd02634c176d7e16db288b936b9/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L57) |
| Mode, trainable parameters, LR and weight decay | [smoke.py:143](https://github.com/LuoQiuXu/xin/blob/8cbfa69bd194083053917bb0cef615572abdc182/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L143) | [smoke.py:270](https://github.com/LuoQiuXu/xin/blob/d47cbd7293402fd02634c176d7e16db288b936b9/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L270) |
| MixUp/forward/loss branch | [smoke.py:182](https://github.com/LuoQiuXu/xin/blob/8cbfa69bd194083053917bb0cef615572abdc182/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L182) | [smoke.py:312](https://github.com/LuoQiuXu/xin/blob/d47cbd7293402fd02634c176d7e16db288b936b9/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L312) |
| Full one_epoch routine | [smoke.py:138](https://github.com/LuoQiuXu/xin/blob/8cbfa69bd194083053917bb0cef615572abdc182/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L138) | [smoke.py:265](https://github.com/LuoQiuXu/xin/blob/d47cbd7293402fd02634c176d7e16db288b936b9/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L265) |
| Stage-II checkpoint routing | [smoke.py:422](https://github.com/LuoQiuXu/xin/blob/8cbfa69bd194083053917bb0cef615572abdc182/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L422) | [smoke.py:675](https://github.com/LuoQiuXu/xin/blob/d47cbd7293402fd02634c176d7e16db288b936b9/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/smoke.py#L675) |

The first-stage mode branch, optimizer creation/parameter set, GradScaler constructor, accumulation, Views/loader construction, target and raw input loading, batch-control hashing, group-size calculation, MixUp forward/loss, scaled backward, unscale and zero_grad calls all match as AST projections. The guard call arguments also match. seed(), validation metrics and validation forward routines have identical ASTs.
The only extra statement inside the B0 MixUp branch copies the two targets to a CPU diagnostic context and converts the already sampled lambda to float. It draws no randomness, does not edit the mixed input/targets/parameters and does not change the loss. The source also adds a context variable outside the branch and scale/update-counter reads around unscale.
New exception handling saves a post-forward/pre-step snapshot before re-raising. It is a rejected-window diagnostic path, not a replacement forward/loss. The bounded forensic early return is inactive for the completed B0 run: max_batches remains None. The B0-only entry passes the same stage1 schedule, no donor, stage=1 and original first-warmup configuration. Actual result files independently establish full 16269-sample completion.

## Explicit non-equivalences and bookkeeping

The recovery gate changed from one positive Inf to one Inf of either sign at exactly PIF b [0,1,0], while retaining the no-NaN, no-missing, exact-count/location constraints. Diagnostic coordinate logging is capped at 32 entries, but the full Inf count is computed separately; this does not allow a multi-Inf event. Native GradScaler owns skips/backoff, and counter/parameter-hash checks remain.
The original requirement of exactly one B0 skip changed to a required first batch6 skip plus the existing total/consecutive limits and a real following update for every skip. This is an actual numeric-recovery/update-path difference, documented as 1017 versus 1015 updates, not hidden behind SOURCE_EQUIVALENCE=PASS.
Source refactoring also removes the default six-condition branch batch_smoke=PASS status write. That is a status-reporting difference outside the B0-only entry and does not modify its Stage-I numerical calculation. It is retained as a known caveat; no fix is performed here.
**SOURCE_EQUIVALENCE=PASS is limited to the audited initialization/data/normal Stage-I numerical core with the above explicit policy and bookkeeping differences. It does not mean the full source files or actual optimizer trajectories are identical.**

## MixUp determinism and observation limits

[Original H2T methods.py:8](https://github.com/LuoQiuXu/xin/blob/8cbfa69bd194083053917bb0cef615572abdc182/H2T_official_clean/H2T-main/methods.py#L8) draws lambda with NumPy beta and the permutation with CPU torch.randperm(batch_size), then moves the permutation to CUDA. Its source is unchanged.
Both schedules, class/target mapping, batch sizes, pre-MixUp augmented tensor hashes, CPU torch RNG hashes and float64 lambda values match for all 1017 batches. Thus the same MixUp operation is supported by deterministic inputs/code/state evidence. The NumPy RNG state, CUDA RNG state, permutation and mixed tensor were not saved in these epoch controls; no direct comparison of those missing objects is claimed.

## Static Stage-II readiness

[models.py:124](https://github.com/LuoQiuXu/xin/blob/d47cbd7293402fd02634c176d7e16db288b936b9/H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t/models.py#L124) calls model.eval(), extractor.requires_grad_(False), extractor.eval(), classifier.requires_grad_(True) and classifier.train(). The classifier is Linear, so enabling it does not re-enable extractor BN. Stage-II optimizer parameters are classifier-only; extractor forwards are in no_grad and the classifier loss/backward is outside that block.
The unchanged routing loop maps A1/A2 to the same A0 state and B1/B2 to the same B0 state; strict state loading includes classifier weights. It uses the shared recipient schedule for all four and the same donor schedule for A2/B2. Static pairwise classifier hashes come from each validated checkpoint, not from a new Stage-II run.
The current CLI has no standalone Stage-II checkpoint-reuse option: the default six-condition function first trains Stage-I again. Therefore sampling/checkpoint readiness PASS is not a runnable Stage-II command or permission to invoke the default entry. No launch plan or new implementation is added.

## AST evidence results

| Check | Result |
|---|---|
| seed_function_ast_equal | PASS |
| metrics_function_ast_equal | PASS |
| validation_function_ast_equal | PASS |
| gradscaler_constructor_ast_equal | PASS |
| stage_mode_parameters_lr_wd_ast_equal | PASS |
| assignment_opt_ast_equal | PASS |
| assignment_scaler_ast_equal | PASS |
| assignment_step_counter_ast_equal | PASS |
| assignment_optimizer_ids_ast_equal | PASS |
| assignment_named_optimizer_parameters_ast_equal | PASS |
| assignment_initial_loss_scale_ast_equal | PASS |
| assignment_accumulation_ast_equal | PASS |
| assignment_dataset_ast_equal | PASS |
| assignment_batches_ast_equal | PASS |
| stage1_mixup_forward_loss_ast_equal_excluding_readonly_context_copy | PASS |
| loop_assignment_target_ast_equal | PASS |
| loop_assignment_row_ast_equal | PASS |
| loop_assignment_x_ast_equal | PASS |
| loop_assignment_group_start_ast_equal | PASS |
| loop_assignment_group_size_ast_equal | PASS |
| backward_calls_ast_equal | PASS |
| unscale__calls_ast_equal | PASS |
| zero_grad_calls_ast_equal | PASS |
| guard_call_arguments_ast_equal | PASS |
| stage2_routing_loop_ast_equal | PASS |
