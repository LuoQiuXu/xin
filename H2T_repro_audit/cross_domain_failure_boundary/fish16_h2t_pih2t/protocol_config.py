"""Single source of truth for the Fish16 experiment branch and protocol identity."""
BRANCH = "experiment/fish16-clean-launch-20261103"
SEED = 42
CONDITIONS = ("A0", "B0", "A1", "A2", "B1", "B2")
STAGE1_EPOCHS = 90
STAGE2_EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 4
NUMERIC_MODE = "float32_no_autocast_no_grad_scaler"
