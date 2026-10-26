# H2T Component Audit Protocol

## Purpose

Build an evidence-backed map of the non-backbone components in the original H2T implementation. For every candidate component, answer:

1. Where is it defined, and where is it called?
2. Is it enabled in the selected CIFAR-100-LT IF=100 Stage 1 / Stage 2 configuration?
3. What role does the paper assign to it, and what mechanism does the code actually implement?
4. What behavior would likely change if it were removed or replaced?
5. Has the removal effect already been tested in the paper or repository, or is it only a hypothesis?

This is a static audit and scientific interpretation task, not a model-improvement task.

## Frozen source of truth

Default source root:

`H2T_official_clean/H2T-main/`

Default configs:

- `config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml`
- `config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml`

Do not silently substitute `H2T_repro_audit/official/`, an experimental implementation, or a later PI-H2T implementation for the clean original H2T snapshot. If a comparison is useful, report it separately and preserve the default source as the primary evidence.

Do not modify files under the frozen source root.

## Component taxonomy

Use these categories and state the evidence for each assignment:

- **Method-defining component**: directly implements a mechanism proposed by H2T.
- **Data / sampling component**: determines receiver, donor, class distribution, or sample frequency.
- **Training protocol component**: optimizer, gradient flow, model train/eval state, scheduler, MixUp, or augmentation.
- **Normalization / calibration component**: BN running statistics, feature/weight normalization, or explicit scaling.
- **Classifier / loss component**: classifier head and objective.
- **Backbone implementation detail**: convolutional blocks, pooling, residual shortcuts, and other architecture internals. Document for context, but do not mislabel as an H2T-specific innovation.
- **Optional / alternative implementation**: code exists but the selected configuration or entry point does not establish its use.
- **Unresolved**: evidence is insufficient or code behavior is ambiguous.

## Evidence and confidence rules

For every important claim, cite a repository-relative path and line number. Separate the evidence into:

- **Paper evidence**: section, equation, figure, table, or page from the paper/slides. If the paper PDF cannot be accessed, explicitly say so and do not invent a page or equation.
- **Code evidence**: exact definition, call site, branch condition, or tensor operation.
- **Configuration evidence**: the selected YAML value. Mention when a value may be overridden by defaults or command-line arguments.
- **Runtime evidence**: only available if an existing runtime audit or controlled run directly measured it. Do not claim static code inspection is runtime proof.
- **Ablation evidence**: only when an existing paper/repository result directly tests removal or replacement.

For the “remove it” question, label the conclusion as one of:
- **Already tested** — give the exact result and source.
- **Code-derived expectation** — explain the causal path but state it is not yet experimentally verified.
- **Unknown** — evidence does not support a reliable prediction.

Never infer that a module is active merely because it is defined. Never infer that it is inactive solely because a simple keyword search found no call. Follow the actual entry point, factory/dispatch logic, selected config, and relevant control flow.

## Required outputs

Codex should create a new timestamped run directory under:

`H2T_repro_audit/component_audit/results/`

It should contain:

- `source_manifest.json`: hashes of the source files scanned and the local Git branch/HEAD when available.
- `static_inventory.json`: AST symbols, relevant call sites, selected config values, contexts, and parse warnings.
- `STATIC_COMPONENT_MAP.md`: generated code-location inventory.
- `COMPONENT_AUDIT_REPORT.md`: human-reviewed explanation of each component and its role.
- `COMPONENT_EVIDENCE_MATRIX.csv`: one row per component, including category, paper evidence, definition, call site, selected-config status, expected removal effect, evidence class, and unresolved questions.
- `VALIDATION_REPORT.md`: commands executed, test results, source immutability checks, and limitations.

The Python scanner is a locator, not the final scientific conclusion. The human report must inspect function bodies and control flow rather than merely repeat keyword matches.

## Explicit non-goals

- Do not change H2T, H2Tv2, BN, sampler, loss, classifier, or optimizer behavior.
- Do not add attention modules or other new mechanisms.
- Do not launch full training, multi-seed training, or GPU experiments.
- Do not change datasets, checkpoint files, configs, random seeds, or baseline results.
- Do not recommend a new method before the module map is complete.
- Do not overwrite previous result directories; each run must be reproducible and timestamped.
