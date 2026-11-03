"""Single source of truth for the Fish16 experiment branch and protocol identity."""
BRANCH = "experiment/fish16-clean-launch-fix-20261103"
SEED = 42
CONDITIONS = ("A0", "B0", "A1", "A2", "B1", "B2")
STAGE1_EPOCHS = 90
STAGE2_EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
NUMERIC_MODE = "float32_no_autocast_no_grad_scaler"

# Immutable upstream source fingerprints captured from the approved H2T/PI-H2T copies.
# Paths are repository-root relative and keys match models.source_hashes().
LOCKED_UPSTREAM_SOURCE_HASHES = {
    "H2T_official_clean/H2T-main/models/resnet.py": "3c4ac4b66fd4aad2e6682103684605b9e2bf783fc5742e64ea7c1ebdfa4cb2b4",
    "H2T_official_clean/H2T-main/methods.py": "d8266450190511f60771d7e632280c37a4d38a190daf4e5e55898656b00461a1",
    "H2T_official_clean/H2T-main/train_stage1.py": "f97ea0b7524f115aaf08061a728cfe4be4ea93a7cca2180d20413aeec38e45f1",
    "H2T_official_clean/H2T-main/train_stage2.py": "c0e1f794eacb50093a43651b585155600fafac9f3d26f69673aa13ce3d888614",
    "PI_H2T_review/H2T-main/pi_h2t.py": "b63e481d7491bf38752e8071735ecc732c912ceb02a95e9f27b6f403052065a3",
    "PI_H2T_review/H2T-main/methods.py": "d8266450190511f60771d7e632280c37a4d38a190daf4e5e55898656b00461a1",
    "PI_H2T_review/H2T-main/models/resnet_cifar.py": "d8920c3a8b67d1603fc7fd1c8bdaab5db3aff63c4fe37f745282ae2988fd9e2d",
    "PI_H2T_review/H2T-main/PI_H2T_REVIEW_ZH.md": "c8a74bb22bf37f1c09bbd1a9abf2ff5065f082dffdf0dbcc9",
}

# Exact source set validated before the one-epoch Smoke. Paths are relative to this directory.
CPU_VALIDATION_SOURCE_FILES = (
    "protocol_config.py", "preflight.py", "data.py", "models.py", "amp_recovery.py",
    "smoke.py", "formal_train.py", "validate_cpu.py", "test_amp_recovery.py",
    "test_contract.py", "test_formal_train.py", "diagnostics/__init__.py",
    "diagnostics/diagnose_b0.py", "diagnostics/test_diagnostic_stats.py",
)

# The formal runner must be covered by the exact same CPU validation record.
FORMAL_RUNNER_SOURCE_FILES = (
    "formal_train.py", "models.py", "data.py", "amp_recovery.py",
    "preflight.py", "protocol_config.py",
)
EXPECTED_CPU_TEST_METHODS = 41
