# Original H2T Component Audit

This directory adds a read-only static inventory tool and a Codex execution protocol for auditing non-backbone components in the clean original H2T source.

## Frozen source and target setting

- Source: `H2T_official_clean/H2T-main/`
- Stage 1 config: `config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml`
- Stage 2 config: `config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml`
- Dataset setting represented by the selected configs: CIFAR-100-LT, imbalance factor 0.01.
- No training, checkpoint loading, dataset loading, or GPU access is required for the static audit.

## Run

From the repository root:

```bash
python -m unittest discover -s H2T_repro_audit/component_audit -p "test_*.py" -v
python H2T_repro_audit/component_audit/audit_components.py
```

The scanner writes a new timestamped directory under `H2T_repro_audit/component_audit/results/` containing:

- `source_manifest.json`
- `static_inventory.json`
- `STATIC_COMPONENT_MAP.md`

The scanner uses only the Python standard library. It parses Python with `ast` and extracts scalar YAML mappings using a deliberately small parser; it is not a general YAML parser. The generated inventory is static evidence, not proof of runtime execution or causal contribution.

## Codex instructions

Use `CODEX_H2T_COMPONENT_AUDIT.txt` for the complete task. It directs Codex to trace the actual Stage 1/Stage 2 path, read the original paper evidence, create a component-by-component report and CSV evidence matrix, run integrity checks, and avoid all training changes.

## Evidence standard

Every component should be classified as active on the selected path, defined but not shown active, architecture-only, optional/alternative, or unresolved. Any predicted removal effect must be explicitly marked as a code-derived expectation unless an existing controlled ablation supports it. Do not alter files under the frozen original source root.
