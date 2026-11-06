"""Verify and seal the saved diagnostic using CPU arrays and read-only Git commands."""
import ast
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
PREFIX = OUT.relative_to(REPO).as_posix() + "/"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    return hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode()
                          + value.tobytes(order="C")).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def main():
    for name in ("DIAGNOSTIC_VERIFICATION.json", "GIT_STATUS_FINAL.txt", "ARTIFACT_SHA256.csv"):
        assert not (OUT / name).exists(), "Preserve existing output: " + name
    assert (OUT / "C1_GRADIENT_DIAGNOSTIC_REPORT.md").is_file()
    baseline = json.loads((OUT / "PROTECTION_BEFORE.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "gradient_diagnostic.json").read_text(encoding="utf-8"))
    assert result["status"] == "DIAGNOSTIC_COMPLETED"
    assert result["historical_failure_exact_max_abs_reproduced"]
    assert git("branch", "--show-current").strip() == baseline["branch"]
    assert git("rev-parse", "HEAD").strip() == baseline["head"]
    assert git("diff", "--name-status") == ""
    assert git("diff", "--cached", "--name-status") == ""

    changed = []
    for name, entry in baseline["protected_experiment_files"].items():
        if digest(REPO / name) != entry["sha256"]:
            changed.append(name)
    for name, entry in baseline["protected_external_inputs"].items():
        if digest(Path(name)) != entry["sha256"]:
            changed.append(name)
    assert not changed, changed
    before_lines = set((OUT / "GIT_STATUS_BEFORE.txt").read_text(encoding="utf-8").splitlines())
    after_lines = set(git("status", "--short", "--untracked-files=all").splitlines())
    before_outside = {line for line in before_lines if not line[3:].startswith(PREFIX)}
    after_outside = {line for line in after_lines if not line[3:].startswith(PREFIX)}
    assert before_outside == after_outside, sorted(before_outside ^ after_outside)
    assert all(line.startswith("?? ") for line in after_outside)
    tree = ast.parse((OUT / "diagnose_c1.py").read_text(encoding="utf-8"))
    forbidden_calls = [ast.unparse(n.func) for n in ast.walk(tree)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in ("step", "backward", "smoke_test", "verify_analytic_gradient")]
    assert not forbidden_calls, forbidden_calls

    tensors = np.load(OUT / "tensor_comparison.npz", allow_pickle=False)
    assert len(tensors.files) == 90
    assert all(np.isfinite(tensors[name]).all() for name in tensors.files)
    hashes = result["batch"]["hashes"]
    for key, name in (("indices_sha256", "indices"), ("labels_sha256", "labels"),
                      ("augmented_images_sha256", "augmented_images_fp32"),
                      ("gap_features_sha256", "gap_fp32"), ("feature_map_sha256", "feature_map_fp32")):
        assert array_digest(tensors[name]) == hashes[key], key
    classifier = hashlib.sha256()
    for state_name, tensor_name in (("module.fc.weight", "fp32_head_weight"),
                                    ("module.fc.bias", "fp32_head_bias")):
        classifier.update(state_name.encode())
        classifier.update(tensors[tensor_name].tobytes(order="C"))
    assert classifier.hexdigest() == hashes["classifier_state_sha256"]
    stages = ("probability", "raw_gradient", "class_gradient", "class_norm", "direction", "delta")
    for precision in ("fp32", "fp64"):
        for variant in ("A", "C", "D", "E"):
            for stage in stages:
                a = tensors[f"{precision}_{variant}_{stage}"].astype(np.float64)
                b = tensors[f"{precision}_B_{stage}"].astype(np.float64)
                error = np.abs(a - b)
                allowed = 1e-7 + 1e-5 * np.abs(b)
                recorded = result["results"][precision]["variants"][variant][stage]
                assert float(error.max()) == recorded["max_abs"]
                assert int((error > allowed).sum()) == recorded["violations"]
                assert bool(np.all(error <= allowed)) == recorded["torch_allclose"]
                if variant == "E":
                    assert tensors[f"{precision}_{variant}_{stage}"].tobytes() == tensors[f"{precision}_B_{stage}"].tobytes()
    with (OUT / "per_class_gradient_errors.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 512
    assert not any(row["any_nonfinite"] != "False" for row in rows)

    labels = tensors["labels"]
    classes = tensors["fp32_class_ids"]
    a = tensors["fp32_A_delta"].astype(np.float64)
    b = tensors["fp32_B_delta"].astype(np.float64)
    violations = []
    for position, dimension in zip(*np.nonzero(np.abs(a - b) > 1e-7 + 1e-5 * np.abs(b))):
        cls = int(labels[position])
        j = int(np.flatnonzero(classes == cls)[0])
        g = float(tensors["fp32_A_class_gradient"][position, dimension])
        gr = float(tensors["fp32_B_class_gradient"][position, dimension])
        n = float(tensors["fp32_A_class_norm"][j])
        nr = float(tensors["fp32_B_class_norm"][j])
        weight = float(tensors["fp32_class_weights"][cls])
        gradient_component = weight * (g - gr) / nr
        norm_component = weight * g * (1 / n - 1 / nr)
        absolute_error = abs(float(a[position, dimension] - b[position, dimension]))
        allowed_error = 1e-7 + 1e-5 * abs(float(b[position, dimension]))
        violations.append(dict(class_id=cls, batch_position=int(position),
            prepared_train_index=int(tensors["indices"][position]), feature_dimension=int(dimension),
            analytic=float(a[position, dimension]), reference=float(b[position, dimension]),
            absolute_error=absolute_error, allowed_error=allowed_error,
            normalized_error=absolute_error / allowed_error,
            analytic_raw_gradient=g, reference_raw_gradient=gr,
            analytic_class_norm=n, reference_class_norm=nr, class_weight=weight,
            weight_div_reference_norm=weight / nr,
            gradient_error_component=gradient_component, norm_error_component=norm_component,
            fp32_output_rounding_residual=float(a[position, dimension] - b[position, dimension]
                                               - gradient_component - norm_component)))
    norm_only_classes = [int(classes[j]) for j in np.flatnonzero(
        tensors["fp32_C_class_norm"] != tensors["fp32_B_class_norm"])]
    assert len(violations) == 4 and norm_only_classes == [27, 29, 30, 58, 63, 70, 87]
    assert not (OUT.parent / "results").exists()
    ignored_tensor = git("check-ignore", str(OUT / "tensor_comparison.npz")).strip()
    assert ignored_tensor
    verification = dict(status="PASS", verified_at=datetime.now().astimezone().isoformat(),
        branch=baseline["branch"], head=baseline["head"],
        protected_experiment_files=len(baseline["protected_experiment_files"]),
        protected_external_inputs=len(baseline["protected_external_inputs"]), changed_protected_files=changed,
        tracked_diff_empty=True, cached_diff_empty=True, outside_directory_git_status_unchanged=True,
        outside_directory_preexisting_untracked_file_count=len(after_outside),
        diagnostic_syntax_and_forbidden_call_check_passed=True,
        saved_tensors=90, saved_tensor_finiteness="ALL_FINITE", per_class_rows=len(rows),
        batch_hashes_independently_recomputed_from_npz=True, classifier_hash_recomputed_from_npz=True,
        saved_stage_metrics_independently_recomputed=True,
        E_vs_B_all_six_stages_byte_equal_in_both_precisions=True,
        fp32_norm_only_differing_classes=norm_only_classes, fp32_A_delta_violations=violations,
        error_decomposition_note="Using saved FP32 values in double arithmetic: w*(gA-gB)/nB + w*gA*(1/nA-1/nB), plus final FP32-output rounding residual. CPU postprocessing only.",
        official_TEST_used=False, official_TEST_open_attempts=0, optimizer_steps=0,
        new_smoke_invocations=0, formal_results_directory_exists=False,
        production_files_modified=False, tensor_archive_gitignored=True,
        historical_replay_limitation="Old failed Smoke did not persist batch hashes; exact failed max_abs and gate outcome are reproduced, with hashes now recorded.")
    with (OUT / "DIAGNOSTIC_VERIFICATION.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (OUT / "GIT_STATUS_FINAL.txt").open("x", encoding="utf-8", newline="\n") as status_stream, \
            (OUT / "ARTIFACT_SHA256.csv").open("x", encoding="utf-8", newline="") as manifest_stream:
        for args in (("branch", "--show-current"), ("rev-parse", "HEAD"),
                     ("status", "--short", "--branch", "--untracked-files=all"),
                     ("diff", "--stat"), ("diff", "--name-status"),
                     ("diff", "--cached", "--name-status"),
                     ("status", "--short", "--ignored", "--", PREFIX)):
            status_stream.write("$ git " + " ".join(args) + "\n" + git(*args) + "\n")
        status_stream.flush()
        writer = csv.DictWriter(manifest_stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        artifact_count = 0
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "ARTIFACT_SHA256.csv" and path.suffix != ".pyc":
                writer.writerow(dict(path=path.relative_to(OUT).as_posix(),
                                     bytes=path.stat().st_size, sha256=digest(path)))
                artifact_count += 1
    print(json.dumps(dict(status="PASS", artifacts_hashed=artifact_count,
        protected_files_unchanged=len(baseline["protected_experiment_files"]) + len(baseline["protected_external_inputs"]),
        preexisting_untracked_files_unchanged=len(after_outside),
        diagnostic_only=True, commit_created=False, pushed=False)))


if __name__ == "__main__":
    main()
