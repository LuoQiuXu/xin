import csv
import hashlib
import json
from pathlib import Path

import pytest

from BCL_LTN import postprocess_existing_run as post


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def archive(tmp_path, monkeypatch):
    output = tmp_path / "BCL_sym04_seed42"
    run = output / "BCL_sym04_seed42"
    diag_dir = run / "train_diagnostics"
    noise = tmp_path / "noise"
    execution = tmp_path / "receipt"
    old_checkout = tmp_path / "historic"
    old_source = old_checkout / post.TRAINER
    old_source.parent.mkdir(parents=True)
    old_source.write_text('p.add_argument("--smoke-only")\np.add_argument("--smoke-report")\n', encoding="utf-8")
    source_sha = hashlib.sha256(old_source.read_bytes()).hexdigest()
    monkeypatch.setattr(post, "_git_source", lambda head: b'p.add_argument("--output-root")\n')

    class_counts = [int(500 * (0.01 ** (i / 99.0))) for i in range(100)]
    groups = {
        "Many": [value for value in class_counts if value > 100],
        "Medium": [value for value in class_counts if 20 <= value <= 100],
        "Few": [value for value in class_counts if value < 20],
    }
    group_counts = {}
    for group, counts in groups.items():
        source = sum(counts)
        flips = int(0.4 * source)
        group_counts[group] = {
            "num_classes": len(counts), "clean_samples": source, "observed_noisy_label_samples": source,
            "clean_source_samples_flipped": flips, "clean_source_samples_unflipped": source - flips,
        }
    total, flipped = sum(class_counts), sum(row["clean_source_samples_flipped"] for row in group_counts.values())
    index_sha, label_sha = "a" * 64, "b" * 64
    noise_manifest = {
        "seed": 42, "noise_rate_requested": 0.4, "noise_rate_realized": flipped / total,
        "num_train_samples": total, "num_flipped": flipped, "num_unflipped": total - flipped,
        "clean_class_counts": class_counts, "observed_noisy_class_counts": class_counts,
        "many_medium_few": group_counts, "indices_sha256": index_sha, "noisy_labels_sha256": label_sha,
        "train_source_sha256": {"train": "c" * 64, "meta": "d" * 64},
        "clean_labels_sha256": "e" * 64, "is_noisy_sha256": "f" * 64,
    }
    manifest_path = noise / "noise_manifest.json"
    write_json(manifest_path, noise_manifest)
    manifest_sha = post.sha256(manifest_path)

    diagnostics = []
    diag_dir.mkdir(parents=True)
    for epoch in post.EPOCHS:
        clean = {group: {"correct": i, "total": g["clean_source_samples_unflipped"],
                         "accuracy_percent": 100.0 * i / g["clean_source_samples_unflipped"]}
                 for i, (group, g) in enumerate(group_counts.items(), 1)}
        clean["Overall"] = {"correct": 6, "total": total - flipped, "accuracy_percent": 600.0 / (total - flipped)}
        noisy = {group: {"predicted_as_corrupted_observed_label": i + 1, "total_noisy_examples": g["clean_source_samples_flipped"],
                         "memorization_rate_percent": 100.0 * (i + 1) / g["clean_source_samples_flipped"]}
                 for i, (group, g) in enumerate(group_counts.items(), 1)}
        noisy["Overall"] = {"predicted_as_corrupted_observed_label": 9, "total_noisy_examples": flipped,
                            "memorization_rate_percent": 900.0 / flipped}
        value = {"epoch": epoch, "samples": total, "model_state_unchanged": True,
                 "clean_subset_accuracy": clean, "noisy_subset_wrong_label_memorization_rate": noisy}
        write_json(diag_dir / f"epoch_{epoch:03d}.json", value)
        diagnostics.append(value)
    with (output / "class_counts_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "clean_count", "observed_noisy_count"])
        writer.writerows((i, count, count) for i, count in enumerate(class_counts))

    checkpoints = [{"epoch": e, "path": str(run / "checkpoints" / f"epoch_{e:03d}.pth"), "sha256": f"{e:064x}"}
                   for e in post.EPOCHS]
    test_values = {"Overall": 32.22, "Many": 50.43, "Medium": 35.66, "Few": 6.97}
    write_json(run / "final_test_metrics.json", {
        "overall": test_values["Overall"], **{k: test_values[k] for k in post.GROUPS[1:]},
        "official_test_opens_before_gate": 0, "official_test_eval_passes": 1,
        "official_test_file_opens_during_eval": 2, "checkpoint_sha256": checkpoints[-1]["sha256"],
    })
    write_json(run / "training_summary.json", {
        "epochs_completed": 200, "optimizer_updates": 8600, "official_test_opens_before_gate": 0,
        "total_training_seconds": 12.5, "max_peak_cuda_memory_bytes": 1024, "max_peak_cuda_reserved_bytes": 2048,
        "initial_model_state_sha256": "0" * 64, "observed_noisy_class_counts": class_counts,
        "checkpoint_diagnostics": diagnostics, "checkpoint_paths": [row["path"] for row in checkpoints],
        "checkpoint_sha256": {Path(row["path"]).name: row["sha256"] for row in checkpoints},
    })
    common = {"indices_sha256": index_sha, "noisy_labels_sha256": label_sha, "noise_manifest_sha256": manifest_sha,
              "source_sha256": {post.TRAINER: source_sha}, "precision": "fp32", "batch_size": 256}
    smoke_path = tmp_path / "BCL_sym04_seed42_smoke" / "smoke_report.json"
    write_json(smoke_path, {**common, "passed": True, "official_test_opens": 0, "formal_training_started": False,
                            "train_batch_tuple_fields": 3})
    smoke_sha = post.sha256(smoke_path)
    write_json(output / "environment.json", {
        **common, "smoke_report_sha256": smoke_sha,
        **dict.fromkeys(("gpu", "device", "python", "python_executable", "platform", "torch", "torchvision", "cuda", "cudnn"), "fixture"),
        "seed": 42, "epochs": 200, "amp": False, "tf32": False,
    })
    write_json(execution / "gpu_smoke_receipt.json", {"command": ["--smoke-only"], "exit_code": 0, "smoke_report_sha256": smoke_sha})
    launch = {"head": "1" * 40, "branch": "fixture", "command": ["--smoke-report", str(smoke_path)],
              "working_directory": str(old_checkout), "smoke_report_sha256": smoke_sha, "noise_manifest_sha256": manifest_sha,
              "batch_size": 256, "precision": "fp32"}
    write_json(execution / "formal_launch.json", launch)
    write_json(execution / "formal_terminal.json", {**launch, "state": "completed", "exit_code": 0, "oom": False,
                                                   "wall_seconds": 15.0, "completed_epochs": 200,
                                                   "optimizer_updates_in_completed_epochs": 8600})
    original = tmp_path / "BCL_sym04_baseline_report.md"
    original.write_text("## Final TEST (raw logits)\n| Overall | 99.99% | 50.76% | +49.23 |\n", encoding="utf-8")
    write_json(execution / "FINAL_EXECUTION_VERIFICATION.json", {
        "head": launch["head"], "branch": launch["branch"], "state": "completed",
        "epochs": 200, "optimizer_updates": 8600, "test_opens_before_gate": 0, "test_eval_passes": 1,
        "test_metrics": test_values, "checkpoints": checkpoints, "report_sha256": post.sha256(original),
    })
    return output, noise, execution


def test_reads_all_checkpoints_and_produces_twenty_original_fractions(archive):
    output, noise, _ = archive
    diagnostics, rows = post.read_diagnostics(output / "BCL_sym04_seed42/train_diagnostics")
    assert [d["epoch"] for d in diagnostics] == list(post.EPOCHS)
    assert len(list(csv.DictReader(post.render_csv(rows).splitlines()))) == 20
    manifest = post.read_json(noise / "noise_manifest.json")
    overall, *groups = rows[:4]
    assert overall["clean_correct"] == 6 and overall["wrong_label_memorized"] == 9
    assert overall["clean_total"] == manifest["num_unflipped"]
    assert overall["noisy_total"] == manifest["num_flipped"]
    for index, row in enumerate(groups, 1):
        metadata = manifest["many_medium_few"][row["clean_source_group"]]
        assert row["clean_correct"] == index
        assert row["clean_accuracy_pct"] == 100.0 * index / metadata["clean_source_samples_unflipped"]
        assert row["wrong_label_memorization_pct"] == 100.0 * (index + 1) / metadata["clean_source_samples_flipped"]


def test_missing_checkpoint_has_clear_error(archive):
    directory = archive[0] / "BCL_sym04_seed42/train_diagnostics"
    (directory / "epoch_160.json").unlink()
    with pytest.raises(FileNotFoundError, match="Missing epoch-160 diagnostic JSON"):
        post.read_diagnostics(directory)


@pytest.mark.parametrize(("text", "message"), [
    ("{", "invalid JSON"),
    ("[]", "expected a JSON object"),
    ('{"epoch": 100}', "expected both metrics"),
])
def test_invalid_or_incomplete_json_has_clear_error(archive, text, message):
    directory = archive[0] / "BCL_sym04_seed42/train_diagnostics"
    (directory / "epoch_100.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"epoch_100\.json: .*{message}"):
        post.read_diagnostics(directory)


def test_changed_denominator_is_rejected(archive):
    directory = archive[0] / "BCL_sym04_seed42/train_diagnostics"
    path = directory / "epoch_100.json"
    value = post.read_json(path)
    item = value["clean_subset_accuracy"]["Many"]
    item["total"] += 1
    item["accuracy_percent"] = 100.0 * item["correct"] / item["total"]
    write_json(path, value)
    with pytest.raises(ValueError, match="Many denominators changed"):
        post.read_diagnostics(directory)


def test_rebuilt_report_uses_saved_test_metrics_and_reports_legacy_discrepancy(archive):
    output = archive[0]
    original = output.parent / "BCL_sym04_baseline_report.md"
    initial = original.read_bytes()
    result = post.rebuild_existing_run(*archive)
    result_text = Path(result["report"]).read_text(encoding="utf-8")
    assert result["csv_rows"] == 20 and result["csv_action"] == "created"
    assert result["csv_byte_identical"] and result["historical_trainer_source_verified"]
    assert original.read_bytes() == initial
    metrics = post.read_json(output / "BCL_sym04_seed42/final_test_metrics.json")
    for group in post.GROUPS:
        value = metrics[group if group != "Overall" else "overall"]
        assert f"| {group} | {value:.2f}% |" in result_text
    assert "Original display: | Overall | 99.99%" in result_text
    assert "Rebuilt display: | Overall | 32.22%" in result_text


def test_existing_csv_is_checked_without_rewriting(archive):
    first = post.rebuild_existing_run(*archive)
    path = Path(first["csv"])
    previous = path.read_bytes()
    assert post.rebuild_existing_run(*archive)["csv_action"] == "verified_existing"
    assert path.read_bytes() == previous
    path.write_bytes(previous.replace(b"40,Overall,6,", b"40,Overall,5,"))
    with pytest.raises(ValueError, match="clean_correct differs"):
        post.rebuild_existing_run(*archive)
    assert b"40,Overall,5," in path.read_bytes()


def test_postprocessing_never_reads_checkpoint_bytes(archive, monkeypatch):
    output = archive[0]
    checkpoint = output / "BCL_sym04_seed42/checkpoints/epoch_200.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"invalid checkpoint that must never be opened")
    open_path = Path.open

    def reject_pth(self, *args, **kwargs):
        assert self.suffix != ".pth", f"Unexpected checkpoint open: {self}"
        return open_path(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_pth)
    result = post.rebuild_existing_run(*archive)
    assert result["csv_rows"] == 20
    assert str(checkpoint) in Path(result["report"]).read_text(encoding="utf-8")
