"""Rebuild the locked BCL-LTN baseline report from saved text artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess


EPOCHS = (40, 100, 160, 180, 200)
GROUPS = ("Overall", "Many", "Medium", "Few")
CSV_FIELDS = (
    "epoch", "clean_source_group", "clean_correct", "clean_total", "clean_accuracy_pct",
    "wrong_label_memorized", "noisy_total", "wrong_label_memorization_pct",
)
CLEAN_REFERENCE = dict(zip(GROUPS, (50.76, 66.54, 49.40, 31.20), strict=True))
TRAINER = "BCL_LTN/train_noisy_bcl.py"
PUBLISHED_BASELINE_HEAD = "3d458546d329050f20ed54db67980054a53d83f9"
SMOKE_FLAGS = {"--smoke-only", "--smoke-report"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON ({error})") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_fraction(data, numerator, denominator, percent, key):
    values = [data.get(field) for field in (numerator, denominator, percent)]
    n, d, reported = values
    _require(type(n) is int and type(d) is int and 0 <= n <= d and d > 0,
             f"{key}: invalid integer fraction {n}/{d}")
    computed = 100.0 * n / d
    _require(type(reported) in (int, float) and reported == computed,
             f"{key}: reported percentage {reported!r} differs from {n}/{d} ({computed})")
    return n, d, computed


def read_diagnostics(directory: Path):
    """Read all five saved diagnostics and validate the original sample counts."""
    saved, rows = [], []
    denominators = {}
    for epoch in EPOCHS:
        path = directory / f"epoch_{epoch:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing epoch-{epoch} diagnostic JSON: {path}")
        record = read_json(path)
        clean = record.get("clean_subset_accuracy")
        noisy = record.get("noisy_subset_wrong_label_memorization_rate")
        _require(record.get("epoch") == epoch, f"{path}: expected epoch {epoch}")
        _require(isinstance(clean, dict) and isinstance(noisy, dict)
                 and set(clean) == set(noisy) == set(GROUPS),
                 f"{path}: expected both metrics for Overall, Many, Medium and Few")
        epoch_rows = []
        for group in GROUPS:
            _require(isinstance(clean[group], dict) and isinstance(noisy[group], dict),
                     f"{path}: invalid metrics object for {group}")
            c = _read_fraction(clean[group], "correct", "total", "accuracy_percent",
                               f"{path} {group} clean")
            n = _read_fraction(noisy[group], "predicted_as_corrupted_observed_label",
                               "total_noisy_examples", "memorization_rate_percent", f"{path} {group} noisy")
            expected = denominators.setdefault(group, (c[1], n[1]))
            _require(expected == (c[1], n[1]), f"{path}: {group} denominators changed across checkpoints")
            epoch_rows.append(dict(zip(CSV_FIELDS, (epoch, group, *c, *n), strict=True)))
        overall, *shots = epoch_rows
        for field in ("clean_correct", "clean_total", "wrong_label_memorized", "noisy_total"):
            _require(overall[field] == sum(row[field] for row in shots),
                     f"{path}: Overall {field} differs from the sum across shot groups")
        _require(record.get("samples") == overall["clean_total"] + overall["noisy_total"],
                 f"{path}: TRAIN diagnostic sample count differs from the two subset totals")
        saved.append(record)
        rows.extend(epoch_rows)
    return saved, rows


def render_csv(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def check_existing_csv(path: Path, expected_rows):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == CSV_FIELDS, f"{path}: incorrect CSV columns")
        actual_rows = list(reader)
    _require(len(actual_rows) == len(expected_rows), f"{path}: expected {len(expected_rows)} CSV rows")
    for row_number, (actual, expected) in enumerate(zip(actual_rows, expected_rows, strict=True), 2):
        _require(set(actual) == set(expected), f"{path}:{row_number}: incorrect CSV fields")
        for key, value in expected.items():
            try:
                observed = type(value)(actual[key])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{row_number}: invalid CSV value for {key}") from error
            _require(observed == value, f"{path}:{row_number}: {key} differs ({observed!r} vs {value!r})")


def read_class_counts(path: Path, manifest, rows):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == ["class_id", "clean_count", "observed_noisy_count"],
                 f"{path}: unexpected class count columns")
        counts = [(int(row["class_id"]), int(row["clean_count"]), int(row["observed_noisy_count"]))
                  for row in reader]
    _require([class_id for class_id, _, _ in counts] == list(range(100)),
             f"{path}: expected all 100 class IDs in order")
    clean = [count for _, count, _ in counts]
    observed = [count for _, _, count in counts]
    total, flipped, unflipped = (manifest[key] for key in ("num_train_samples", "num_flipped", "num_unflipped"))
    _require(sum(clean) == sum(observed) == total == 10_847 == flipped + unflipped
             and min(clean) > 0 and min(observed) > 0 and 0 < flipped < total,
             f"{path}: invalid class counts or manifest totals")
    _require(clean == manifest["clean_class_counts"] and observed == manifest["observed_noisy_class_counts"],
             f"{path}: class count vectors differ from the noise manifest")
    _require(manifest["noise_rate_realized"] == flipped / total, "Manifest realized noise rate differs from its counts")
    shot_groups = {
        "Many": [i for i in range(100) if clean[i] > 100],
        "Medium": [i for i in range(100) if 20 <= clean[i] <= 100],
        "Few": [i for i in range(100) if clean[i] < 20],
    }
    for group, ids in shot_groups.items():
        data = manifest["many_medium_few"][group]
        _require(data["num_classes"] == len(ids)
                 and data["clean_samples"] == sum(clean[i] for i in ids)
                 and data["observed_noisy_label_samples"] == sum(observed[i] for i in ids)
                 and data["clean_samples"] == data["clean_source_samples_flipped"] + data["clean_source_samples_unflipped"],
                 f"Noise manifest {group} counts disagree with its class vectors")
    for row in rows:
        group = row["clean_source_group"]
        data = manifest["many_medium_few"].get(group, {})
        _require(row["clean_total"] == data.get("clean_source_samples_unflipped", unflipped)
                 and row["noisy_total"] == data.get("clean_source_samples_flipped", flipped),
                 f"Epoch {row['epoch']} {group} denominators differ from the noise manifest")
    return counts


def _percent(value, key):
    _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100,
             f"{key}: invalid TEST percentage {value!r}")
    return float(value)


def _source_flags(source: bytes):
    parsed = ast.parse(source)
    return {arg.value for node in ast.walk(parsed)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
            for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)} & SMOKE_FLAGS


def _git_source(head):
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        return None
    try:
        result = subprocess.run(["git", "cat-file", "blob", f"{head}:{TRAINER}"], cwd=REPO_ROOT,
                                capture_output=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout if result.returncode == 0 else None


def inspect_source_provenance(smoke, environment, launch, smoke_receipt, pre_edit_sha=None):
    recorded = smoke.get("source_sha256", {}).get(TRAINER)
    formal_recorded = environment.get("source_sha256", {}).get(TRAINER)
    archived_path = Path(launch["working_directory"]) / TRAINER
    archived = archived_path.read_bytes() if archived_path.is_file() and archived_path.resolve().suffix == ".py" else None
    archived_sha = hashlib.sha256(archived).hexdigest() if archived is not None else None
    archived_flags = _source_flags(archived) if archived is not None else None
    git_versions = []
    for label, head in (("Recorded execution HEAD", launch["head"]), ("Published baseline HEAD", PUBLISHED_BASELINE_HEAD)):
        source = _git_source(head)
        git_versions.append({
            "label": label, "head": head,
            "sha256": hashlib.sha256(source).hexdigest() if source is not None else None,
            "flags": sorted(_source_flags(source)) if source is not None else None,
        })
    return {
        "recorded": recorded, "formal_recorded": formal_recorded, "pre_edit": pre_edit_sha,
        "archive": str(archived_path), "archive_sha": archived_sha,
        "archive_flags": sorted(archived_flags) if archived_flags is not None else None,
        "archive_verified": bool(recorded and archived_sha == recorded == formal_recorded),
        "git_versions": git_versions,
        "smoke_command_has_flag": "--smoke-only" in smoke_receipt["command"],
        "formal_command_has_flag": "--smoke-report" in launch["command"],
    }


def _formal_evidence(output_root, run, noise_path, execution_dir, diagnostics):
    training = read_json(run / "training_summary.json")
    test = read_json(run / "final_test_metrics.json")
    environment = read_json(output_root / "environment.json")
    launch = read_json(execution_dir / "formal_launch.json")
    terminal = read_json(execution_dir / "formal_terminal.json")
    final = read_json(execution_dir / "FINAL_EXECUTION_VERIFICATION.json")
    smoke_path = output_root.parent / "BCL_sym04_seed42_smoke" / "smoke_report.json"
    smoke = read_json(smoke_path)
    smoke_receipt = read_json(execution_dir / "gpu_smoke_receipt.json")
    _require(training["checkpoint_diagnostics"] == diagnostics, "Training summary and checkpoint diagnostic JSON disagree")
    _require(launch["head"] == terminal["head"] == final["head"] and launch["branch"] == terminal["branch"] == final["branch"]
             and launch["command"] == terminal["command"], "Execution receipts disagree on branch, HEAD or command")
    _require(training["epochs_completed"] == terminal["completed_epochs"] == final["epochs"] == 200
             and terminal["state"] == final["state"] == "completed" and terminal["exit_code"] == 0
             and training["optimizer_updates"] == terminal["optimizer_updates_in_completed_epochs"] == final["optimizer_updates"],
             "Formal completion receipts disagree")
    _require(environment["seed"] == 42 and environment["epochs"] == 200
             and terminal["precision"] == smoke["precision"] == environment["precision"] == "fp32"
             and terminal["batch_size"] == smoke["batch_size"] == environment["batch_size"] == 256
             and environment["amp"] is False and environment["tf32"] is False,
             "Historical receipts do not describe the locked Seed 42 FP32 protocol")
    _require(sha256(smoke_path) == launch["smoke_report_sha256"] == terminal["smoke_report_sha256"]
             == smoke_receipt["smoke_report_sha256"] == environment["smoke_report_sha256"]
             and smoke["passed"] is True and smoke_receipt["exit_code"] == 0,
             "The historical Smoke receipt or SHA256 does not match")
    _require(sha256(noise_path) == launch["noise_manifest_sha256"] == terminal["noise_manifest_sha256"]
             == environment["noise_manifest_sha256"] == smoke["noise_manifest_sha256"],
             "The historical noise manifest SHA256 does not match")
    _require(training["official_test_opens_before_gate"] == test["official_test_opens_before_gate"]
             == final["test_opens_before_gate"] == smoke["official_test_opens"] == 0
             and test["official_test_eval_passes"] == final["test_eval_passes"] == 1
             and smoke["formal_training_started"] is False,
             "The published TEST isolation receipts disagree")
    test_values = {group: _percent(test["overall" if group == "Overall" else group], group) for group in GROUPS}
    _require(final["test_metrics"] == test_values, "Published verification and final TEST metrics disagree")
    checkpoints = final["checkpoints"]
    _require(tuple(row["epoch"] for row in checkpoints) == EPOCHS
             and [row["path"] for row in checkpoints] == training["checkpoint_paths"],
             "Execution receipts disagree on checkpoint order or paths")
    for row in checkpoints:
        name = row["path"].replace("\\", "/").rsplit("/", 1)[-1]
        _require(row["sha256"] == training["checkpoint_sha256"].get(name)
                 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                 f"Recorded checkpoint SHA256 disagrees for {name}")
    _require(checkpoints[-1]["sha256"] == test["checkpoint_sha256"], "Final TEST points to a different checkpoint SHA256")
    return training, test, test_values, environment, launch, terminal, final, smoke, smoke_receipt


def _report_tables(manifest, rows, test_values):
    shot, diagnostic, test = [], [], []
    for group in GROUPS[1:]:
        data = manifest["many_medium_few"][group]
        source, flips = data["clean_samples"], data["clean_source_samples_flipped"]
        shot.append(f"| {group} | {data['num_classes']} | {source} | {flips} | "
                    f"{100 * flips / source:.2f}% | {data['observed_noisy_label_samples']} |")
    for offset, epoch in enumerate(EPOCHS):
        subset = rows[offset * len(GROUPS):(offset + 1) * len(GROUPS)]
        values = [row[field] for field in ("clean_accuracy_pct", "wrong_label_memorization_pct") for row in subset]
        diagnostic.append(f"| {epoch} | " + " | ".join(f"{value:.2f}%" for value in values) + " |")
    for group, value in test_values.items():
        ref = CLEAN_REFERENCE[group]
        test.append(f"| {group} | {value:.2f}% | {ref:.2f}% | {value - ref:+.2f} |")
    return shot, diagnostic, test


def compare_original(original_text, scalar_checks, table_checks):
    """Compare independent results to the displayed fields in the legacy report."""
    sections = {}
    section = ""
    for line in original_text.splitlines():
        if line.startswith("## "):
            section = line[3:]
        sections.setdefault(section, []).append(line)
    differences = []
    checked = 0
    for choices in (scalar_checks, table_checks):
        for section, expected_rows in choices.items():
            for prefix, expected in expected_rows:
                actual = [line for line in sections.get(section, []) if line.startswith(prefix)]
                if actual != [expected]:
                    differences.append((section, expected, " / ".join(actual) or "<missing>"))
                checked += 1
    return checked, differences


def _write_report(report_path, original_path, manifest_path, manifest, counts, rows, evidence, provenance):
    training, test, test_values, env, launch, terminal, final, smoke, _ = evidence
    original_bytes = original_path.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    shot, diag, test_table = _report_tables(manifest, rows, test_values)
    noise_lines = {
        "Requested rate": f"{manifest['noise_rate_requested']:.4f}",
        "Realized rate": f"{manifest['noise_rate_realized']:.6f}",
        "Flipped": f"{manifest['num_flipped']} / {manifest['num_train_samples']}",
        "Index SHA256": manifest["indices_sha256"],
        "TRAIN and meta source SHA256": json.dumps(manifest["train_source_sha256"]),
        "Noise manifest SHA256": sha256(manifest_path),
        "Noisy-label SHA256": manifest["noisy_labels_sha256"],
        "Clean-label SHA256 (offline only)": manifest["clean_labels_sha256"],
        "Mask SHA256 (offline only)": manifest["is_noisy_sha256"],
    }
    runtime = {
        "Training seconds": f"{training['total_training_seconds']:.1f}",
        "Max CUDA allocated": f"{training['max_peak_cuda_memory_bytes'] / 1024**3:.3f} GiB",
        "Max CUDA reserved": f"{training['max_peak_cuda_reserved_bytes'] / 1024**3:.3f} GiB",
        "Actual training optimizer updates": str(training["optimizer_updates"]),
        "Initial model state SHA256": training["initial_model_state_sha256"],
    }
    scalars = {section: [(f"- {key}: ", f"- {key}: {value}") for key, value in data.items()]
               for section, data in (("Noise realization", noise_lines), ("Runtime and provenance", runtime))}
    tables = {
        section: [(f"| {key} |", row) for key, row in zip(keys, values, strict=True)]
        for section, keys, values in (("Noise realization", GROUPS[1:], shot),
                                      ("Five-checkpoint TRAIN diagnostics", EPOCHS, diag),
                                      ("Final TEST (raw logits)", GROUPS, test_table))
    }
    checked, differences = compare_original(original_bytes.decode("utf-8-sig"), scalars, tables)
    original_hash_matches = original_sha == final["report_sha256"]
    lines = [
        "# BCL-LTN symmetric-noise baseline reconstructed from saved results", "",
        "## Purpose and protocol", "",
        "Measure the sensitivity of Original BCL to instance-independent synthetic label noise on CIFAR-100-LT IF=100.",
        "The recorded run used the fixed TRAIN subset, Seed 42, ResNet-32, three views, batch 256 and FP32.",
        "The objective was 2.0 × LC + 0.6 × BCL with observed-label counts.",
        "SGD used LR 0.15, momentum 0.9, weight decay 0.0005, five warmup epochs and LR drops after 160 and 180 completed epochs.",
        "Noise was drawn with independent Bernoulli trials at rate 0.4, Seed 42, and uniform replacement among the other 99 classes.", "",
        "## Noise realization", "",
        "| Field | Recorded value |", "| --- | --- |",
        *(f"| {key} | {value} |" for key, value in noise_lines.items()),
        "", "| Clean-defined group | Classes | Source samples | Source flips | Source flip rate | Observed destination labels |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *shot,
        "", "### Per-class counts", "", "| Class ID | Clean count | Observed noisy-label count |", "| ---: | ---: | ---: |",
        *(f"| {class_id} | {clean} | {observed} |" for class_id, clean, observed in counts),
        "", "## Five-checkpoint TRAIN diagnostics", "",
        "Each percentage is recomputed from its saved numerator and denominator. Shot groups use the clean source class.",
        "", "| Epoch | Clean Overall | Clean Many | Clean Medium | Clean Few | Memorization Overall | Memorization Many | Memorization Medium | Memorization Few |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *diag,
        "", "## Final TEST (raw logits)", "", "| Metric | Noisy BCL | Clean BCL reference | Delta (pp) |",
        "| --- | ---: | ---: | ---: |", *test_table, "",
        "These results describe the locked synthetic-noise baseline for Seed 42.", "",
        "## Runtime and environment", "", "| Field | Recorded value |", "| --- | --- |",
        *(f"| {key} | {value} |" for key, value in runtime.items()),
        f"| Formal wall seconds | {terminal['wall_seconds']:.1f} |",
        *(f"| {key} | {env[key]} |" for key in ("gpu", "device", "python", "python_executable", "platform",
                                                   "torch", "torchvision", "cuda", "cudnn", "precision",
                                                   "seed", "batch_size", "epochs", "amp", "tf32")),
        f"| CUDA OOM | {terminal['oom']} |",
        "", "## Checkpoint records", "",
        "Paths and SHA256 values below come from the saved receipts. Rebuilding used text artifacts only.",
        "", "| Epoch | Recorded local path | Recorded SHA256 |", "| ---: | --- | --- |",
        *(f"| {row['epoch']} | {row['path']} | {row['sha256']} |" for row in final["checkpoints"]),
        "", "## Leakage and TEST isolation", "",
        f"- Historical Smoke recorded {smoke['train_batch_tuple_fields']} training fields (views, observed label and source index). "
        "The training-summary class counts agree with the observed-label manifest vector.",
        "- Saved clean truth and the flip mask are reserved for input integrity and TRAIN diagnostics.",
        f"- Recorded TEST opens before the final gate were {test['official_test_opens_before_gate']}. "
        f"The final evaluator ran {test['official_test_eval_passes']} time after all five TRAIN diagnostics "
        f"and the file had {test['official_test_file_opens_during_eval']} opens during that evaluation.",
        "", "## Source and execution provenance", "",
        f"- Execution branch {launch['branch']}. Formal completion, Smoke and noise-manifest receipts agree.",
        f"- Trainer SHA256 recorded by Smoke is {provenance['recorded'] or 'unavailable'}.",
        f"- Trainer SHA256 recorded by the formal environment is {provenance['formal_recorded'] or 'unavailable'}.",
        f"- Pre-edit local trainer SHA256 is {provenance['pre_edit'] or 'unavailable'}.",
        f"- Archived execution source path {provenance['archive']}.",
        f"- Archived execution source SHA256 is {provenance['archive_sha'] or 'unavailable'}. "
        f"Smoke CLI flags in that file are {provenance['archive_flags']}.",
        "- Historical trainer bytes independently match both recorded hashes." if provenance["archive_verified"] else
        "- The historical trainer bytes could not be independently verified from the archived execution checkout.",
        f"- Recorded Smoke command contains --smoke-only ({provenance['smoke_command_has_flag']}). "
        f"The recorded formal command contains --smoke-report ({provenance['formal_command_has_flag']}).",
        "", "| Git source | Commit | Git blob SHA256 | Smoke flags |", "| --- | --- | --- | --- |",
    ]
    for version in provenance["git_versions"]:
        flags = version["flags"]
        flag_text = "unavailable" if flags is None else ", ".join(flags) or "none"
        lines.append(f"| {version['label']} | {version['head']} | {version['sha256'] or 'unavailable'} | {flag_text} |")
    lines += [
        "", "## Comparison with the original report", "",
        f"- Original report SHA256 {original_sha}. Match to recorded report SHA256 ({original_hash_matches}).",
        f"- Compared {checked} displayed scalar or table-row values. Differences found in {len(differences)}.",
        "- This version adds the 100-class count table, an explicit trainer fingerprint comparison and a per-checkpoint hash table. "
        "The report layout and detail differ from the original.", "",
    ]
    for section, expected, actual in differences:
        lines += [f"### {section}", "", f"Original display: {actual}", f"Rebuilt display: {expected}", ""]
    encoded = "\n".join(lines) + "\n"
    if report_path.exists():
        _require(report_path.read_text(encoding="utf-8") == encoded,
                 f"Existing rebuilt report differs; choose a fresh --report-path: {report_path}")
    else:
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return checked, len(differences), original_hash_matches


def rebuild_existing_run(output_root: Path, noise_dir: Path, execution_dir: Path,
                         *, pre_edit_sha=None, report_path: Path | None = None):
    run = output_root / "BCL_sym04_seed42"
    original = output_root.parent / "BCL_sym04_baseline_report.md"
    report = report_path or output_root.parent / "BCL_sym04_baseline_report_rebuilt.md"
    _require(report.suffix.lower() == ".md" and report.resolve() != original.resolve()
             and report.resolve().suffix.lower() == ".md", "Use a separate Markdown path for the rebuilt report")
    manifest_path = noise_dir / "noise_manifest.json"
    manifest = read_json(manifest_path)
    _require(manifest.get("seed") == 42 and manifest.get("noise_rate_requested") == 0.4,
             "Expected the archived 0.4 symmetric-noise manifest for Seed 42")
    saved, rows = read_diagnostics(run / "train_diagnostics")
    counts = read_class_counts(output_root / "class_counts_comparison.csv", manifest, rows)
    evidence = _formal_evidence(output_root, run, manifest_path, execution_dir, saved)
    training, _, _, environment, launch, _, _, smoke, smoke_receipt = evidence
    _require(training["observed_noisy_class_counts"] == manifest["observed_noisy_class_counts"],
             "Training class counts differ from the observed-label manifest")
    _require(manifest["indices_sha256"] == environment["indices_sha256"] == smoke["indices_sha256"]
             and manifest["noisy_labels_sha256"] == environment["noisy_labels_sha256"] == smoke["noisy_labels_sha256"],
             "Manifest, Smoke and formal provenance disagree on the index or noisy-label SHA256")
    provenance = inspect_source_provenance(smoke, environment, launch, smoke_receipt, pre_edit_sha)
    csv_path = run / "train_diagnostics.csv"
    reconstructed_csv = render_csv(rows)
    csv_exists = csv_path.exists()
    if csv_exists:
        check_existing_csv(csv_path, rows)
    checked, differences, original_hash_matches = _write_report(
        report, original, manifest_path, manifest, counts, rows, evidence, provenance)
    if not csv_exists:
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(reconstructed_csv)
    return {
        "csv": str(csv_path), "csv_rows": len(rows), "csv_action": "verified_existing" if csv_exists else "created",
        "csv_byte_identical": csv_path.read_bytes() == reconstructed_csv.encode("utf-8"),
        "denominators_match_all_checkpoints": True,
        "report": str(report), "report_sha256": sha256(report), "original_report_hash_matches_receipt": original_hash_matches,
        "original_display_checks": checked, "original_display_differences": differences,
        "historical_trainer_sha256": provenance["recorded"],
        "historical_trainer_source_verified": provenance["archive_verified"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, help="Use a new path when a rebuilt report already exists.")
    parser.add_argument("--pre-edit-trainer-sha256", help="Previously measured working-tree trainer SHA256.")
    args = parser.parse_args(argv)
    if args.pre_edit_trainer_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.pre_edit_trainer_sha256):
        parser.error("--pre-edit-trainer-sha256 requires a lowercase SHA256 digest")
    try:
        result = rebuild_existing_run(
            args.output_root, args.noise_dir, args.execution_dir,
            pre_edit_sha=args.pre_edit_trainer_sha256, report_path=args.report_path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Postprocessing stopped: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
