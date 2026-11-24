import json
from pathlib import Path

import pytest

from BCL_LTN.train_noisy_bcl import DEFAULT_FORMAL_SMOKE_REPORT, parse_args, verify_smoke_receipt


CLI_ARGS = ["--data-root", "data", "--indices-file", "fixed.npy", "--output-root", "fresh"]


def test_smoke_report_default_depends_on_mode():
    assert parse_args(CLI_ARGS).smoke_report == DEFAULT_FORMAL_SMOKE_REPORT
    assert parse_args([*CLI_ARGS, "--smoke-only"]).smoke_report == Path("fresh/smoke_report.json")


@pytest.mark.parametrize("mode", [[], ["--smoke-only"]])
def test_custom_smoke_report_path_is_preserved(mode):
    args = parse_args([*CLI_ARGS, *mode, "--smoke-report", "other/receipt.json"])
    assert args.smoke_only is bool(mode)
    assert args.smoke_report == Path("other/receipt.json")


def test_missing_formal_smoke_report_errors_before_reading(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required passed Smoke report is missing"):
        verify_smoke_receipt(tmp_path / "missing.json", {})


@pytest.mark.parametrize("field", ["indices_sha256", "noisy_labels_sha256", "noise_manifest_sha256", "passed"])
def test_formal_smoke_gate_rejects_mismatch(tmp_path, field):
    environment = {"source_sha256": {"trainer": "current source"}, "indices_sha256": "indices",
                   "noise_manifest_sha256": "manifest", "noisy_labels_sha256": "labels",
                   "gpu": "gpu", "torch": "torch", "cuda": "cuda", "cudnn": 123,
                   "batch_size": 256, "precision": "fp32", "amp": False, "tf32": False}
    smoke = {**environment, "passed": True, "optimizer_updates": 1,
             "backbone_input_shape": [768, 3, 32, 32], "missing_gradients": 0,
             "nonfinite_gradients": 0, "changed_parameter_tensors": 107,
             "official_test_opens": 0, "official_test_denied": 0}
    path = tmp_path / "smoke_report.json"
    path.write_text(json.dumps(smoke), encoding="utf-8")
    assert verify_smoke_receipt(path, environment) == smoke
    smoke[field] = False if field == "passed" else "different artifact"
    path.write_text(json.dumps(smoke), encoding="utf-8")
    with pytest.raises(RuntimeError, match="passed, TEST-free" if field == "passed" else field):
        verify_smoke_receipt(path, environment)
