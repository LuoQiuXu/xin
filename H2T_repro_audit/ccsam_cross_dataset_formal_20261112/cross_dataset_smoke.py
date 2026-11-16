"""Run the real-data, disposable two-batch gates on the official TRAIN splits."""
from __future__ import annotations

from itertools import chain, islice

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset

from cross_dataset_core import D0, D1, D2, class_conditional_feature_delta, class_weights_from_counts, head_preservation_kl
from cross_dataset_models import assert_encoder_frozen, build_stage1_models, freeze_encoder, state_dict_sha256
from experiment_common import (
    ROOT, OUTPUT_ROOT, array_sha256, make_loader, make_sgd, read_json, record_run, require_complete,
    run_log, save_status, seed_everything, sha256_file, write_json,
)
from extract_features import FEATURES, LABELS, write_cache
from prepare_experiment import PREFLIGHT, open_plan, train_dataset
from train_stage1 import epoch_lr, eval_transform, stage1_batch, train_transform
from train_stage2 import HEAD, assert_teacher_frozen, head_status, head_step, load_head, new_head, save_head

SMOKE = OUTPUT_ROOT / "smoke"
AMP_EVIDENCE = ROOT / "diagnostics" / "amp_gradient_probe" / OUTPUT_ROOT.name


def gradient_equivalence(features, labels, classifier, weights):
    delta, _ = class_conditional_feature_delta(features, labels, classifier, weights)
    reference_input = features.detach().clone().requires_grad_(True)
    losses = F.cross_entropy(classifier(reference_input), labels, reduction="none")
    expected = torch.zeros_like(features)
    classes = torch.unique(labels, sorted=True)
    for index, label in enumerate(classes):
        mask = labels == label
        gradient, = torch.autograd.grad(losses[mask].mean(), reference_input, retain_graph=index + 1 < len(classes))
        norm = torch.linalg.vector_norm(gradient)
        if float(norm) > 1e-12:
            expected[mask] = weights[label] * gradient[mask] / norm
    atol, rtol = 1e-7, 1e-5
    error = (delta - expected).abs()
    limit = atol + rtol * expected.abs()
    result = {
        "rtol": rtol, "atol": atol, "max_abs": float(error.max()),
        "max_error_over_tolerance": float((error / limit).max()),
        "elements_outside_tolerance": int((error > limit).sum()),
        "passed": bool(torch.allclose(delta, expected, rtol=rtol, atol=atol)),
    }
    if not result["passed"]:
        raise RuntimeError(f"Real or synthetic CC-SAM autograd equivalence failed: {result}")
    return result


def smoke_dataset(exp, name, seal, seed, device):
    directory = SMOKE / name
    stage1, stage2 = exp.protocol["stage1"], exp.protocol["stage2"]
    spec = exp.datasets[name]
    with record_run(directory, exp.config("smoke_dataset", dataset=name, seed=seed)) as status:
        seed_everything(seed)
        with train_dataset(exp, name, train_transform(stage1["input_size"]), seal) as (dataset, targets):
            encoder, initial = build_stage1_models(spec["num_classes"])
            status["starting_stage1_state_sha256"] = [state_dict_sha256(model) for model in (encoder, initial)]
            encoder.to(device).train()
            initial.to(device).train()
            sgd = make_sgd(chain(encoder.parameters(), initial.parameters()), {
                **stage1, "learning_rate": epoch_lr(1, stage1["epochs"], stage1["learning_rate"]),
            })
            scaler = torch.amp.GradScaler("cuda")
            generator = torch.Generator(device=device).manual_seed(seed)
            rng = np.random.default_rng(seed)
            loader = make_loader(dataset, stage1["batch_size"], exp.workers, seed, shuffle=True)
            status["stage1_batches"] = []
            status["stage1_amp_evidence"] = {}
            evidence_dir = AMP_EVIDENCE / name
            evidence_dir.mkdir(parents=True, exist_ok=False)
            for number, (x, y) in enumerate(islice(loader, 2), 1):
                status["current_stage1_batch"] = number
                evidence = status["stage1_amp_evidence"][str(number)] = []

                def trace(identity, summary, scaled_rows, rows, delta_rows):
                    path = evidence_dir / f"batch_{number:02}_attempt_{summary['attempt']:02}.json"
                    write_json(path, {
                        "dataset": name, "seed": seed, "batch_one_based": number, "batch_identity": identity,
                        **summary, "scaled_gradients": scaled_rows, "unscaled_gradients": rows,
                        "parameter_deltas": delta_rows, "official_test_access": False,
                    })
                    evidence.append({
                        "file": path.relative_to(AMP_EVIDENCE).as_posix(), "sha256": sha256_file(path),
                        "outcome": summary["outcome"], "scale_before": summary["scale_before"],
                        "scale_after": summary["scale_after"], "optimizer_step": summary["optimizer_step"],
                    })
                    save_status(directory, status)

                result = stage1_batch(encoder, initial, sgd, scaler, x.to(device), y.to(device), rng,
                                      generator, stage1["mixup_alpha"], strict_gradients=True, record_attempt=trace)
                status["stage1_batches"].append(result)
                run_log(directory, f"{name} Stage-I Smoke batch {number}/2 passed, "
                        f"AMP attempts={result['amp_attempt_count']}, scale={result['scale_after']:g}, "
                        f"updated tensors={result['update']['changed_parameter_tensors']}")
            del loader, sgd, scaler
            if len(status["stage1_batches"]) != 2:
                raise RuntimeError("Stage-I Smoke did not complete two real batches")

            encoder.zero_grad(set_to_none=True)
            snapshot = freeze_encoder(encoder)
            plan, plan_record = open_plan(seal, seed, stage2["epochs"])
            indices = plan[0, :stage2["batch_size"]]
            labels = targets[indices]
            dataset.transform = eval_transform()
            loader = make_loader(Subset(dataset, indices.tolist()), stage1["batch_size"], exp.workers, seed)
            status["disposable_cache"] = write_cache(directory, loader, labels, encoder, device)
            del loader
        cache = np.load(directory / FEATURES, mmap_mode="r", allow_pickle=False)
        cached_labels = np.load(directory / LABELS, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(labels, cached_labels):
            raise RuntimeError("Smoke cache round trip changed ordered labels")
        x = torch.from_numpy(np.array(cache)).to(device)
        y = torch.from_numpy(np.array(cached_labels)).to(device)
        counts = torch.tensor(seal["train_class_counts"], dtype=torch.long, device=device)
        weights = class_weights_from_counts(seal["train_class_counts"], device)
        status["real_gradient_equivalence"] = gradient_equivalence(x, y, initial, weights)
        noise = torch.randn(x.shape, generator=generator, device=device)
        status["synthetic_gradient_equivalence"] = gradient_equivalence(noise, y, initial, weights)

        binding = {
            "dataset": name, "seed": seed, "num_classes": spec["num_classes"],
            "initial_classifier_sha256": state_dict_sha256(initial),
            "encoder_state_sha256": state_dict_sha256(encoder),
            "data_fingerprint": seal["data_fingerprint"], "sampling_plan_sha256": plan_record["sha256"],
            "indices_sha256": array_sha256(indices), "feature_cache_sha256": sha256_file(directory / FEATURES),
            "labels_sha256": array_sha256(cached_labels),
        }
        status["pairing"] = binding
        status["arms"] = {}
        base_d1 = None
        for arm in exp.arms:
            student = new_head(spec["num_classes"], initial.state_dict(), device)
            if state_dict_sha256(student) != binding["initial_classifier_sha256"]:
                raise RuntimeError(f"Smoke initialization differs in {arm}")
            teacher = teacher_sha = expected_kl = None
            if arm == D2:
                d0_dir = directory / "disposable_D0_teacher"
                d0 = head_status(d0_dir, binding, D0, 1, phase="smoke_teacher")
                teacher = load_head(d0_dir, d0, binding, D0, 1, device, frozen=True)
                teacher_sha = d0["artifacts"][HEAD]
                assert_teacher_frozen(teacher, d0["classifier_state_sha256"])
                with torch.no_grad():
                    student_logits, teacher_logits = student(x), teacher(x)
                    kl, mask, n_many = head_preservation_kl(student_logits, teacher_logits, y, counts)
                    if n_many == 0 or not torch.equal(mask, counts[y] > 100):
                        raise RuntimeError("Smoke did not validate the strict Many-only distillation mask")
                    if student_logits.shape != (len(y), spec["num_classes"]):
                        raise RuntimeError("D2 KL logits do not span every output class")
                    full = F.kl_div(
                        F.log_softmax(student_logits[mask] / 2.0, dim=1),
                        F.softmax(teacher_logits[mask] / 2.0, dim=1), reduction="batchmean",
                    ) * 4.0
                    if not torch.equal(kl, full):
                        raise RuntimeError("Smoke D2 KL disagrees with the full output distribution")
                    expected_kl = float(kl)
            optimizer = make_sgd(student.parameters(), stage2)
            loss, diag = head_step(arm, student, teacher, weights, counts, x, y, 0, optimizer)
            if state_dict_sha256(student) == binding["initial_classifier_sha256"]:
                raise RuntimeError(f"Smoke classifier did not update in {arm}")
            if arm == D0:
                if diag["delta_mean_l2"] != 0.0:
                    raise RuntimeError("Smoke D0 perturbed features")
                with record_run(directory / "disposable_D0_teacher", exp.config("smoke_teacher", **binding)) as teacher_status:
                    state_sha = save_head(directory / "disposable_D0_teacher" / HEAD, student, binding, D0, 1, None)
                    teacher_status.update(binding, arm=D0, final_epoch=1, epochs_completed=1,
                                          classifier_state_sha256=state_sha)
            else:
                if diag["delta_mean_l2"] <= 0.0:
                    raise RuntimeError(f"Smoke {arm} had no feature perturbation")
                if arm == D1:
                    base_d1 = loss
                else:
                    if not np.isclose(loss, base_d1 + expected_kl, rtol=1e-6, atol=1e-7):
                        raise RuntimeError("Smoke D2 loss differs from paired D1 plus Many-only KL")
                    assert_teacher_frozen(teacher, d0["classifier_state_sha256"])
            assert_encoder_frozen(encoder, snapshot)
            status["arms"][arm] = {
                **diag, "loss": loss, "one_classifier_update": True, "encoder_and_bn_unchanged": True,
                "teacher_checkpoint_sha256": teacher_sha,
                "teacher_eval_frozen_no_grad": teacher is not None, "kl_output_classes": spec["num_classes"] if teacher else None,
            }
        status["passed"] = True


def run_smoke(exp, prepared, device):
    with record_run(SMOKE, exp.config("smoke"), device=device) as status:
        if any((OUTPUT_ROOT / path).exists() for path in ("stage1", "features", "stage2", "final_evaluation", "analysis")):
            raise FileExistsError("Formal output already exists; refusing to run Smoke")
        status["prepared_manifest_sha256"] = sha256_file(PREFLIGHT / "prepared_manifest.json")
        status["datasets"] = []
        for name in exp.datasets:
            smoke_dataset(exp, name, prepared["datasets"][name], exp.seeds[0], device)
            status["datasets"].append(name)
        status["passed"] = True
    return load_smoke(exp)


def load_smoke(exp):
    status = require_complete(SMOKE, phase="smoke", expected={
        "passed": True, "datasets": list(exp.datasets), "official_test_access": False,
        "prepared_manifest_sha256": sha256_file(PREFLIGHT / "prepared_manifest.json"),
    })
    for name in exp.datasets:
        dataset = read_json(SMOKE / name / "status.json")
        batches = dataset["stage1_batches"]
        if len(batches) != 2:
            raise RuntimeError(f"{name} lacks two verified Stage-I Smoke updates")
        for number, batch in enumerate(batches, 1):
            state = batch["final_unscaled_gradient_state"]
            evidence = dataset["stage1_amp_evidence"][str(number)]
            if (not batch["optimizer_step"] or batch["update"]["changed_parameter_tensors"] < 1
                or not batch["update"]["parameters_finite_after"]
                or state["finite_tensors"] != state["trainable_tensors"]
                or len(evidence) != batch["amp_attempt_count"]
                or evidence[-1]["outcome"] != "finite_optimizer_update"):
                raise RuntimeError(f"{name} Smoke batch {number} lacks a finite effective AMP update")
            for record in evidence:
                path = (AMP_EVIDENCE / record["file"]).resolve()
                if not path.is_relative_to(AMP_EVIDENCE.resolve()) or sha256_file(path) != record["sha256"]:
                    raise RuntimeError(f"Stage-I Smoke AMP evidence SHA256 mismatch: {path}")
    return status
