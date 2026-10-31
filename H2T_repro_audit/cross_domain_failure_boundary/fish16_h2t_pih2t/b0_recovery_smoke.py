"""Gate 1 only: reproduce and recover the known B0 PIF AMP overflow.

Runs exactly the first 100 Stage-I batches on Fish4Knowledge-16, seed 42.
No validation/test evaluation, checkpoint selection, Stage-II, or formal run.
All outputs must be written outside the repository.
"""
import argparse
import csv
import hashlib
import json
from itertools import islice
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

import models
from amp_recovery import (
    OptimizerStepCounter,
    guarded_scaler_step,
    make_grad_scaler,
)
from data import Views, load_rows, loader, schedules
from preflight import BRANCH, EXPECTED_SHA, MANIFEST

VALIDATION = Path(r"D:\h2t_cross_domain_results\fish16_validation_seed42_20261031_01")
EXPECTED_ENV = {
    "python_version": "3.12.12",
    "torch": "2.14.1+cu132",
    "cuda_build": "13.2",
    "cudnn": 92400,
    "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU",
}
EXPECTED_INITIAL_PIF = [-0.027100861072540283, -0.604386568069458]
EXPECTED_B6 = {
    "batch": 6,
    "row_indices": [13582, 7987, 15285, 17141, 2724, 13480, 16315, 9510,
                    1680, 15366, 1164, 1640, 18775, 14261, 13777, 19669],
    "raw_input_sha256": "b9ce6c99dd498e3213ebdff893466bbc365833295904b3b153281965b6c81ad5",
    "rng_before_mixup_sha256": "78b3dde6c54238603e730541568ea8f2ef9d5b3531fb11562c69ac38c75a19b3",
    "mixup_lambda": 0.814341567785799,
    "labels_a": [0, 2, 0, 0, 3, 0, 0, 2, 3, 0, 3, 3, 0, 0, 0, 0],
    "labels_b": [2, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 3, 3, 0, 0, 3],
    "pif_parameters_before": [-0.02449430152773857, -0.6112035512924194],
}


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def digest_tensor(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def git_text(*args):
    return subprocess.check_output(
        ["git", *args], cwd=models.ROOT, text=True
    ).strip()


def verify_prerequisites():
    assert torch.cuda.is_available(), "CUDA is required for Gate 1."
    assert git_text("branch", "--show-current") == BRANCH, "Wrong Git branch."
    assert not git_text("status", "--porcelain"), "Worktree is not clean; stop without cleanup."
    assert models.sha(MANIFEST) == EXPECTED_SHA, "Locked manifest SHA mismatch."
    assert models.sha(models.ORIGINAL_WEIGHT) == models.ORIGINAL_SHA, "Official weight SHA mismatch."

    source_hash_file = VALIDATION / "ORIGINAL_SOURCE_HASHES.json"
    cpu_log_file = VALIDATION / "CPU_TEST_RESULTS_FINAL.txt"
    assert source_hash_file.is_file() and cpu_log_file.is_file()
    original_sources = json.loads(source_hash_file.read_text(encoding="utf-8"))
    assert models.source_hashes() == original_sources, "An upstream H2T/PI-H2T source changed."
    cpu_log = cpu_log_file.read_text(encoding="utf-8-sig")
    assert "Ran 10 tests" in cpu_log and "OK" in cpu_log, "Existing 10/10 CPU contract log missing."

    environment = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
    }
    assert environment["python_version"] == EXPECTED_ENV["python_version"], environment
    assert environment["torch"] == EXPECTED_ENV["torch"], environment
    assert environment["cuda_build"] == EXPECTED_ENV["cuda_build"], environment
    assert environment["cudnn"] == EXPECTED_ENV["cudnn"], environment
    assert environment["gpu"] == EXPECTED_ENV["gpu"], environment
    return {
        "branch": BRANCH,
        "head": git_text("rev-parse", "HEAD"),
        "manifest_sha256": EXPECTED_SHA,
        "official_weight_sha256": models.ORIGINAL_SHA,
        "original_upstream_source_hashes": original_sources,
        "environment": environment,
    }


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(output):
    prerequisite = verify_prerequisites()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rows = load_rows()
    assert len(rows) == 27133
    train, val, stage1, recipients, donors = schedules(rows)
    assert len(train) == 16269 and len(val) == 5396 and len(stage1) == 16269

    # Match smoke.py: construct the model first, then reset the global RNG
    # immediately before the training stream begins.
    model = models.FishModel(True)
    seed_everything()
    model.cuda()
    model.train()

    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(
        parameters, lr=0.01 / 5, momentum=0.9, weight_decay=1e-4
    )
    scaler = make_grad_scaler()
    assert scaler.get_scale() == 65536.0
    counter = OptimizerStepCounter(optimizer)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    named_optimizer_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in optimizer_ids
    ]
    assert len(named_optimizer_parameters) == len(parameters)

    initial_pif = [
        float(value)
        for value in model.extractor.pif.fuse.weight.detach().flatten().cpu()
    ]
    assert all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-7)
        for a, b in zip(initial_pif, EXPECTED_INITIAL_PIF)
    ), f"Initial PIF initialization drift: {initial_pif}"
    dataset = Views(rows, stage1, True)
    batches = loader(dataset, batch=16)
    batch_rows, step_rows = [], []
    attempted_steps = 0
    optimizer_updates = 0
    overflow_skips = 0
    consecutive_skips = 0
    max_consecutive_skips = 0
    first_overflow_batch = None
    next_batch_after_overflow_stepped = False
    samples = 0
    loss_sum = 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    save_json(output / "run_manifest.json", {
        **prerequisite,
        "scope": "GATE1_B0_FIRST_100_STAGE1_BATCHES_ONLY",
        "seed": 42,
        "micro_batch": 16,
        "effective_batch": 16,
        "gradient_accumulation": 1,
        "amp": True,
        "grad_scaler": {
            "init_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
        "optimizer": "SGD",
        "lr": 0.002,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "mixup_alpha": 1.0,
        "batches_requested": 100,
        "expected_replay_batch": EXPECTED_B6,
        "experiment_python_source_hashes": {
            source.name: models.sha(source)
            for source in sorted(Path(__file__).parent.glob("*.py"))
        },
        "validation_evaluations": 0,
        "test_evaluations": 0,
        "stage2_epochs": 0,
        "formal_training_started": False,
        "output_directory": str(output),
        "pif_initial_a_b": initial_pif,
        "numeric_config": {
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_tf32": False,
            "matmul_tf32": False,
            "workers": 4,
        },
    })

    optimizer.zero_grad(set_to_none=True)
    try:
        for batch_index, item in enumerate(islice(batches, 100)):
            target = item["label"].cuda(non_blocking=True)
            raw = item["recipient"]
            raw_sha = digest_tensor(raw)
            rng_sha = digest_tensor(torch.get_rng_state())
            row_indices = [int(value) for value in item["index"].tolist()]

            pif_before_batch = [
                float(value)
                for value in model.extractor.pif.fuse.weight.detach().flatten().cpu()
            ]
            if batch_index == EXPECTED_B6["batch"]:
                assert row_indices == EXPECTED_B6["row_indices"], "Batch-6 row schedule drift."
                assert raw_sha == EXPECTED_B6["raw_input_sha256"], "Batch-6 image tensor drift."
                assert rng_sha == EXPECTED_B6["rng_before_mixup_sha256"], "Batch-6 MixUp RNG drift."
                assert all(
                    math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)
                    for a, b in zip(pif_before_batch, EXPECTED_B6["pif_parameters_before"])
                ), f"Batch-6 PIF state drift: {pif_before_batch}"

            x = raw.cuda(non_blocking=True)
            x, labels_a, labels_b, lam = models.methods.mixup_data(
                x, target, alpha=1.0
            )
            if batch_index == EXPECTED_B6["batch"]:
                assert math.isclose(
                    float(lam), EXPECTED_B6["mixup_lambda"],
                    rel_tol=0.0, abs_tol=1e-12
                ), f"Batch-6 MixUp lambda drift: {lam}"
                assert labels_a.detach().cpu().tolist() == EXPECTED_B6["labels_a"], (
                    "Batch-6 MixUp recipient-label order drift."
                )
                assert labels_b.detach().cpu().tolist() == EXPECTED_B6["labels_b"], (
                    "Batch-6 MixUp permutation/label order drift."
                )

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = (
                    lam * torch.nn.functional.cross_entropy(logits, labels_a)
                    + (1.0 - lam) * torch.nn.functional.cross_entropy(logits, labels_b)
                )
            assert torch.isfinite(logits).all(), f"Nonfinite logits at batch {batch_index}."
            assert torch.isfinite(loss), f"Nonfinite loss at batch {batch_index}."

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Full extractor state hashing is intentionally limited to the
            # known batch-6 event; the helper hashes trainable parameters on
            # every recoverable overflow.
            extractor_state_before_step = (
                models.state_sha(model.extractor)
                if batch_index == EXPECTED_B6["batch"] else None
            )
            outcome = guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named_optimizer_parameters,
                step_counter=counter,
                allow_pif_b_overflow=True,
                consecutive_overflow_skips=consecutive_skips,
                max_consecutive_overflow_skips=2,
            )
            extractor_state_after_step = (
                models.state_sha(model.extractor)
                if batch_index == EXPECTED_B6["batch"] else None
            )

            if outcome.skipped:
                overflow_skips += 1
                consecutive_skips += 1
                max_consecutive_skips = max(max_consecutive_skips, consecutive_skips)
                if batch_index == EXPECTED_B6["batch"]:
                    assert (
                        extractor_state_before_step == extractor_state_after_step
                    ), "Extractor parameters/buffers changed during scaler.step/update."
                if first_overflow_batch is None:
                    first_overflow_batch = batch_index
            else:
                optimizer_updates += 1
                consecutive_skips = 0

            attempted_steps += 1
            samples += len(target)
            loss_sum += float(loss.detach()) * len(target)
            step_row = outcome.as_dict()
            step_row.update({
                "batch_index": batch_index,
                "samples": len(target),
                "optimizer_updates_total": optimizer_updates,
                "overflow_skips_total": overflow_skips,
                "consecutive_overflow_skips": consecutive_skips,
                "extractor_state_sha_before_step": extractor_state_before_step,
                "extractor_state_sha_after_step": extractor_state_after_step,
            })
            step_rows.append(step_row)
            batch_rows.append({
                "batch_index": batch_index,
                "row_indices": json.dumps(row_indices),
                "recipient_sha256": raw_sha,
                "mixup_rng_sha256": rng_sha,
                "mixup_lambda": float(lam),
                "labels_a": json.dumps([int(v) for v in labels_a.detach().cpu().tolist()]),
                "labels_b": json.dumps([int(v) for v in labels_b.detach().cpu().tolist()]),
                "loss": float(loss.detach()),
                "logits_finite": True,
                "step_kind": outcome.kind,
                "scale_before": outcome.scale_before,
                "scale_after": outcome.scale_after,
            })

            if batch_index < EXPECTED_B6["batch"]:
                assert not outcome.skipped, (
                    f"Unexpected overflow before archived failing batch: {batch_index}"
                )
            if batch_index == EXPECTED_B6["batch"]:
                assert math.isclose(
                    float(loss.detach()), 1.7094205617904663,
                    rel_tol=0.0, abs_tol=1e-5
                ), f"Batch-6 loss drift: {float(loss.detach())}"
                assert outcome.skipped, (
                    "The archived batch-6 PIF b overflow did not recur under the exact replay."
                )
                assert outcome.scale_before == 65536.0
                assert outcome.scale_after == 32768.0
                assert outcome.optimizer_steps_before == 6
                assert outcome.optimizer_steps_after == 6
            if batch_index == EXPECTED_B6["batch"] + 1:
                assert not outcome.skipped, (
                    "The first batch after scale backoff did not execute an optimizer step."
                )
                assert outcome.scale_before == 32768.0
                next_batch_after_overflow_stepped = True

            optimizer.zero_grad(set_to_none=True)

            if (batch_index + 1) % 10 == 0:
                print(
                    f"B0 recovery {batch_index + 1}/100; "
                    f"updates={optimizer_updates}, skips={overflow_skips}, "
                    f"scale={scaler.get_scale():.0f}",
                    flush=True,
                )
        torch.cuda.synchronize()

    except Exception:
        if batch_rows:
            write_csv(
                output / "batch_log_partial.csv",
                batch_rows,
                [
                    "batch_index", "row_indices", "recipient_sha256", "mixup_rng_sha256",
                    "mixup_lambda", "labels_a", "labels_b", "loss", "logits_finite",
                    "step_kind", "scale_before", "scale_after",
                ],
            )
        if step_rows:
            write_csv(
                output / "optimizer_step_events_partial.csv",
                step_rows,
                [
                    "kind", "scale_before", "scale_after", "optimizer_steps_before",
                    "optimizer_steps_after", "nan_count", "inf_count",
                    "nonfinite_parameters", "nonfinite_positions",
                    "parameter_sha_before", "parameter_sha_after", "batch_index",
                    "samples", "optimizer_updates_total", "overflow_skips_total",
                    "consecutive_overflow_skips", "extractor_state_sha_before_step",
                    "extractor_state_sha_after_step",
                ],
            )
        raise
    finally:
        counter.close()

    duration = time.perf_counter() - started
    write_csv(
        output / "batch_log.csv",
        batch_rows,
        [
            "batch_index", "row_indices", "recipient_sha256", "mixup_rng_sha256",
            "mixup_lambda", "labels_a", "labels_b", "loss", "logits_finite",
            "step_kind", "scale_before", "scale_after",
        ],
    )
    write_csv(
        output / "optimizer_step_events.csv",
        step_rows,
        [
            "kind", "scale_before", "scale_after", "optimizer_steps_before",
            "optimizer_steps_after", "nan_count", "inf_count",
            "nonfinite_parameters", "nonfinite_positions",
            "parameter_sha_before", "parameter_sha_after", "batch_index",
            "samples", "optimizer_updates_total", "overflow_skips_total",
            "consecutive_overflow_skips", "extractor_state_sha_before_step",
            "extractor_state_sha_after_step",
        ],
    )
    assert len(batch_rows) == 100, f"Only {len(batch_rows)} batches completed."
    assert samples == 1600, f"Expected 1600 samples, got {samples}."
    assert attempted_steps == 100
    assert optimizer_updates + overflow_skips == attempted_steps
    assert optimizer_updates == counter.count
    assert first_overflow_batch == 6, f"Unexpected first overflow batch: {first_overflow_batch}"
    assert next_batch_after_overflow_stepped, "No successful optimizer step after scale backoff."
    assert overflow_skips <= 3, f"Too many AMP skips in 100 batches: {overflow_skips}"
    assert max_consecutive_skips <= 2

    final_pif = [
        float(value)
        for value in model.extractor.pif.fuse.weight.detach().flatten().cpu()
    ]
    result = {
        "status": "PASS_GATE1_B0_100_BATCH_RECOVERY_ONLY",
        "branch": BRANCH,
        "head": prerequisite["head"],
        "batches_completed": len(batch_rows),
        "samples_processed": samples,
        "attempted_optimizer_steps": attempted_steps,
        "actual_optimizer_updates": optimizer_updates,
        "recoverable_overflow_skips": overflow_skips,
        "max_consecutive_overflow_skips": max_consecutive_skips,
        "first_overflow_batch_zero_based": first_overflow_batch,
        "next_batch_optimizer_step_executed": next_batch_after_overflow_stepped,
        "initial_loss_scale": 65536.0,
        "final_loss_scale": float(scaler.get_scale()),
        "pif_initial_a_b": initial_pif,
        "pif_final_a_b": final_pif,
        "train_loss_mean": loss_sum / samples,
        "elapsed_seconds": duration,
        "images_per_second": samples / duration,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "all_forward_logits_and_losses_finite": True,
        "only_nonfinite_gradient_allowed": "positive Inf at extractor.pif.fuse.weight[0,1,0]",
        "grad_scaler_native_skip_and_backoff_verified": True,
        "optimizer_step_hook_verified": True,
        "skipped_step_parameter_hash_unchanged": True,
        "bn_note": (
            "BN running statistics update during the training forward pass even when "
            "GradScaler skips the SGD step; the skipped-step hash comparison is taken "
            "after forward and before scaler.step/update."
        ),
        "validation_evaluations": 0,
        "test_evaluations": 0,
        "stage2_epochs": 0,
        "formal_training_started": False,
        "seed43_44_started": False,
        "authorization": "Gate 1 only; Gate 2 and formal pilot remain unauthorized.",
    }
    save_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    assert not output.is_relative_to(models.ROOT), "Outputs must remain outside the repository."
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        run(output)
    except Exception as exc:
        failure = {
            "status": "FAILED_GATE1_STOPPED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "formal_training_started": False,
            "validation_evaluations": 0,
            "test_evaluations": 0,
            "stage2_epochs": 0,
        }
        save_json(output / "FAILURE.json", failure)
        raise


if __name__ == "__main__":
    main()
