from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ccsam_common import (
    ROOT, NUM_CLASSES, FEATURE_DIM, BATCH_SIZE, STAGE2_EPOCHS, STAGE2_LR,
    MOMENTUM, WEIGHT_DECAY, ADVER_REG, SEEDS, ARMS,
    sha256_file, hash_arrays, state_hash, write_json, read_json,
    install_test_guard, guard_pass, resolve_checkpoint, build_transforms,
    build_models, seed_everything, augment_batch, class_weights_from_counts,
    alpha_for_epoch, evaluate_gap, check_backbone_unchanged, clean_compatible_gap,
)


def analytic_classwise_feature_delta(logits_clean, gap, labels, class_weights, classifier_weight):
    """
    Vectorized exact equivalent of the official CC-SAM Stage-II per-class
    autograd.grad(loss_class, feature) operation for a single linear classifier.

    For z_i = W f_i + b:
      d CE_i / d f_i = (softmax(z_i) - one_hot(y_i)) @ W.
    The official code differentiates the mean loss of each class subset, so
    the per-sample gradient is divided by that class's batch count and then
    normalized jointly over the class subset.
    """
    probabilities = logits_clean.detach().softmax(dim=1)
    one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).to(probabilities.dtype)
    per_sample_gradient = (probabilities - one_hot) @ classifier_weight.detach()

    delta = torch.zeros_like(gap)
    pre_norms = []
    delta_norms = []
    zero_grad_classes = 0
    for class_id in torch.unique(labels, sorted=True):
        mask = labels == class_id
        class_count = int(mask.sum().item())
        class_gradient = per_sample_gradient[mask] / float(class_count)
        gradient_norm = torch.linalg.vector_norm(class_gradient)
        pre_norms.append(float(gradient_norm.detach().cpu()))
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Non-finite class-conditional feature gradient norm")
        if float(gradient_norm.detach().cpu()) <= 1e-12:
            zero_grad_classes += 1
            continue
        class_delta = ADVER_REG * class_weights[class_id] * class_gradient / gradient_norm
        delta[mask] = class_delta
        delta_norms.extend(torch.linalg.vector_norm(class_delta, dim=1).detach().cpu().tolist())

    if not torch.isfinite(delta).all():
        raise RuntimeError("Non-finite CC-SAM feature perturbation")
    feature_norms = torch.linalg.vector_norm(gap.detach(), dim=1).clamp_min(1e-12)
    relative = torch.linalg.vector_norm(delta, dim=1) / feature_norms
    stats = {
        "num_classes_in_batch": int(torch.unique(labels).numel()),
        "zero_gradient_classes": int(zero_grad_classes),
        "mean_class_gradient_norm_before_normalization": float(np.mean(pre_norms)) if pre_norms else 0.0,
        "mean_delta_l2_per_sample": float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()),
        "mean_relative_delta_l2": float(relative.mean().detach().cpu()),
        "max_delta_l2_per_sample": float(torch.linalg.vector_norm(delta, dim=1).max().detach().cpu()),
    }
    return delta.detach(), stats


def autograd_reference_delta(gap, labels, classifier_fc, class_weights):
    """Slow reference used only by the smoke test to verify the vectorized formula."""
    z = gap.detach().clone().requires_grad_(True)
    logits = classifier_fc(z)
    losses = F.cross_entropy(logits, labels, reduction="none")
    reference = torch.zeros_like(z)
    for class_id in torch.unique(labels, sorted=True):
        mask = labels == class_id
        grad = torch.autograd.grad(losses[mask].mean(), z, retain_graph=True)[0]
        norm = torch.linalg.vector_norm(grad)
        if float(norm.detach().cpu()) > 1e-12:
            reference[mask] = ADVER_REG * class_weights[class_id] * grad[mask] / norm
    return reference.detach()


def compute_arm_loss(arm, gap, labels, classifier_fc, class_weights, epoch_index):
    alpha = alpha_for_epoch(epoch_index)
    gap_for_clean = gap.detach()
    logits_clean = classifier_fc(gap_for_clean)
    clean_losses = F.cross_entropy(logits_clean, labels, reduction="none")

    if arm == "C0_BALANCED_CE":
        loss = clean_losses.mean()
        delta = torch.zeros_like(gap_for_clean)
        adv_ce = clean_losses.detach()
        perturb_stats = {
            "num_classes_in_batch": int(torch.unique(labels).numel()),
            "zero_gradient_classes": 0,
            "mean_class_gradient_norm_before_normalization": 0.0,
            "mean_delta_l2_per_sample": 0.0,
            "mean_relative_delta_l2": 0.0,
            "max_delta_l2_per_sample": 0.0,
        }
    elif arm == "C2_WEIGHTED_CE_NO_PERTURB":
        weighted_losses = clean_losses * class_weights[labels]
        loss = alpha * weighted_losses.mean() + (1.0 - alpha) * clean_losses.mean()
        delta = torch.zeros_like(gap_for_clean)
        adv_ce = clean_losses.detach()
        perturb_stats = {
            "num_classes_in_batch": int(torch.unique(labels).numel()),
            "zero_gradient_classes": 0,
            "mean_class_gradient_norm_before_normalization": 0.0,
            "mean_delta_l2_per_sample": 0.0,
            "mean_relative_delta_l2": 0.0,
            "max_delta_l2_per_sample": 0.0,
        }
    elif arm == "C1_CCSAM_FEATURE_GRAD":
        delta, perturb_stats = analytic_classwise_feature_delta(
            logits_clean, gap_for_clean, labels, class_weights, classifier_fc.weight
        )
        logits_adv = classifier_fc(gap_for_clean + delta)
        adv_ce = F.cross_entropy(logits_adv, labels, reduction="none")
        weighted_adv_losses = adv_ce * class_weights[labels]
        loss = alpha * weighted_adv_losses.mean() + (1.0 - alpha) * clean_losses.mean()
        perturb_stats["mean_unweighted_adv_minus_clean_ce"] = float(
            (adv_ce.detach() - clean_losses.detach()).mean().cpu()
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in {arm}")
    return loss, logits_clean, clean_losses, adv_ce, delta, perturb_stats, alpha


def verify_analytic_gradient(gap=None, labels=None, classifier_fc=None, class_weights=None):
    if gap is None:
        device = torch.device("cuda:0")
        torch.manual_seed(20261106)
        classifier_fc = torch.nn.Linear(FEATURE_DIM, NUM_CLASSES).to(device)
        gap = torch.randn(64, FEATURE_DIM, device=device)
        labels = torch.arange(64, device=device, dtype=torch.long) % NUM_CLASSES
        counts = np.linspace(5, 500, NUM_CLASSES, dtype=np.float64)
        class_weights = class_weights_from_counts(counts, device)
        comparison = "synthetic vectorized gradient vs direct autograd reference"
    else:
        gap = gap.detach()
        comparison = "actual Clean Stage I batch/head vectorized gradient vs direct autograd reference"
    logits = classifier_fc(gap.detach())
    analytic, _ = analytic_classwise_feature_delta(
        logits, gap, labels, class_weights, classifier_fc.weight
    )
    reference = autograd_reference_delta(gap, labels, classifier_fc, class_weights)
    max_abs = float((analytic - reference).abs().max().detach().cpu())
    if not torch.allclose(analytic, reference, rtol=1e-5, atol=1e-7):
        raise RuntimeError(f"Vectorized CC-SAM gradient does not match autograd reference; max_abs={max_abs}")
    return {"passed": True, "max_abs_difference": max_abs, "comparison": comparison}


def load_prepared():
    prepared = ROOT / "prepared"
    manifest_path = prepared / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run prepare_ccsam.py successfully before training.")
    manifest = read_json(manifest_path)
    if manifest.get("clean_stage1_sha256") != "7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88":
        raise RuntimeError("Prepared data references a different Clean Stage I checkpoint.")
    if manifest.get("official_TEST_used") is not False:
        raise RuntimeError("Prepared data manifest does not certify TEST-free preparation.")
    if manifest.get("validation_baseline_overall_matches_sealed_clean_protocol") is not True:
        raise RuntimeError("Prepared validation features failed the Clean Stage I reproduction check.")
    train_images = np.load(prepared / "train_images.npy", mmap_mode="r", allow_pickle=False)
    train_labels = np.load(prepared / "train_labels.npy", allow_pickle=False)
    validation_labels = np.load(prepared / "validation_labels.npy", allow_pickle=False)
    validation_gap = np.load(prepared / "validation_gap_features.npy", mmap_mode="r", allow_pickle=False)
    if train_images.shape != (10847, 32, 32, 3) or train_labels.shape != (10847,):
        raise RuntimeError("Prepared TRAIN arrays have unexpected shapes.")
    if validation_gap.shape != (1980, FEATURE_DIM) or validation_labels.shape != (1980,):
        raise RuntimeError("Prepared validation features have unexpected shapes.")
    return manifest, train_images, train_labels, validation_gap, validation_labels


def smoke_test(seed, manifest, train_images, train_labels, validation_gap, validation_labels, train_transform):
    plan_path = ROOT / "prepared" / f"sampling_plan_seed{seed}.npy"
    plan = np.load(plan_path, mmap_mode="r", allow_pickle=False)
    indices = np.asarray(plan[0, :BATCH_SIZE], dtype=np.int64)
    labels = torch.from_numpy(np.asarray(train_labels[indices], dtype=np.int64)).to("cuda:0")
    weights = class_weights_from_counts(np.asarray(manifest["train_class_counts"]), torch.device("cuda:0"))
    reference = None
    arm_results = {}

    for arm in ARMS:
        seed_everything(seed, deterministic=True)
        checkpoint = resolve_checkpoint(manifest["checkpoint_path"])
        backbone, classifier, _ = build_models(checkpoint, torch.device("cuda:0"))
        frozen_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
        initial_backbone_hash = state_hash(backbone)
        initial_classifier_hash = state_hash(classifier)
        generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
        images_cpu = augment_batch(train_images, indices, train_transform, generator)
        images_hash = hash_arrays(indices, labels.cpu().numpy(), images_cpu.numpy())
        with torch.no_grad():
            feature_map = backbone(images_cpu.to("cuda:0"))
            gap = clean_compatible_gap(feature_map)
        feature_hash = hash_arrays(gap.detach().cpu().numpy())

        controls = {
            "indices_and_labels_sha256": hash_arrays(indices, labels.cpu().numpy()),
            "augmented_images_sha256": images_hash,
            "gap_features_sha256": feature_hash,
            "initial_backbone_sha256": initial_backbone_hash,
            "initial_classifier_sha256": initial_classifier_hash,
        }
        if reference is None:
            reference = controls
        elif controls != reference:
            raise RuntimeError(f"Paired smoke inputs or initial states differ in {arm}")

        loss, logits, clean_losses, adv_ce, delta, pstats, alpha = compute_arm_loss(
            arm, gap, labels, classifier.module.fc, weights, epoch_index=0
        )
        if arm == "C1_CCSAM_FEATURE_GRAD":
            analytic_check = verify_analytic_gradient(
                gap, labels, classifier.module.fc, weights
            )
            if float(torch.linalg.vector_norm(delta, dim=1).mean().detach().cpu()) <= 0:
                raise RuntimeError("CC-SAM feature perturbation is identically zero in smoke test.")
        else:
            analytic_check = None
            if not torch.equal(delta, torch.zeros_like(delta)):
                raise RuntimeError(f"{arm} should not perturb features.")

        optimizer = torch.optim.SGD(
            classifier.module.fc.parameters(),
            lr=STAGE2_LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in classifier.module.fc.parameters()):
            raise RuntimeError(f"Classifier gradients are missing or non-finite in {arm}")
        if any(p.grad is not None for p in backbone.parameters()):
            raise RuntimeError(f"Backbone received gradients in {arm}")
        optimizer.step()
        check_backbone_unchanged(backbone, frozen_state)
        if state_hash(classifier) == initial_classifier_hash:
            raise RuntimeError(f"Classifier did not update in {arm}")

        arm_results[arm] = {
            "loss": float(loss.detach().cpu()),
            "alpha_epoch1": float(alpha),
            "clean_ce": float(clean_losses.mean().detach().cpu()),
            "adversarial_ce_unweighted": float(adv_ce.mean().detach().cpu()),
            "perturbation": pstats,
            "classifier_updated": True,
            "backbone_and_bn_bitwise_unchanged": True,
            "analytic_gradient_check": analytic_check,
            "paired_controls": controls,
        }
        del backbone, classifier, optimizer

    return {
        "seed": int(seed),
        "batch_size": int(len(indices)),
        "paired_inputs_and_initialization_match": True,
        "arms": arm_results,
        "controls": reference,
        "validation_features_shape": list(validation_gap.shape),
        "validation_labels_shape": list(validation_labels.shape),
    }


def read_trace(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def train_one_arm(seed, arm, manifest, train_images, train_labels, validation_gap, validation_labels,
                  train_transform, device):
    out = ROOT / "results" / f"seed_{seed}" / arm
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run directory: {out}")
    out.mkdir(parents=True, exist_ok=False)

    seed_everything(seed, deterministic=True)
    checkpoint = resolve_checkpoint(manifest["checkpoint_path"])
    backbone, classifier, _ = build_models(checkpoint, device)
    frozen_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    initial_backbone_hash = state_hash(backbone)
    initial_classifier_hash = state_hash(classifier)
    optimizer = torch.optim.SGD(
        classifier.module.fc.parameters(),
        lr=STAGE2_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=False,
    )
    if optimizer.state:
        raise RuntimeError("Formal run optimizer must start with empty momentum state.")

    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    class_weights = class_weights_from_counts(counts, device)
    plan = np.load(ROOT / "prepared" / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
    if plan.shape != (STAGE2_EPOCHS, len(train_labels)):
        raise RuntimeError(f"Unexpected sampling plan shape: {plan.shape}")
    augmentation_generator = torch.Generator(device="cpu").manual_seed(seed + 300009)

    reference_rows = None
    if arm != "C0_BALANCED_CE":
        reference_path = ROOT / "results" / f"seed_{seed}" / "C0_BALANCED_CE" / "batch_trace.csv"
        if not reference_path.is_file():
            raise FileNotFoundError("C0 batch trace is required before running the paired CC-SAM arms.")
        reference_rows = read_trace(reference_path)
        expected_batches = STAGE2_EPOCHS * ((len(train_labels) + BATCH_SIZE - 1) // BATCH_SIZE)
        if len(reference_rows) != expected_batches:
            raise RuntimeError(f"C0 trace length mismatch: {len(reference_rows)} != {expected_batches}")

    trace_path = out / "batch_trace.csv"
    metrics_path = out / "metrics.csv"
    metric_fields = [
        "epoch", "alpha", "learning_rate", "train_loss", "train_clean_accuracy",
        "validation_Overall", "validation_Macro", "validation_Head", "validation_OfficialMedium",
        "validation_Tail", "validation_Many", "validation_MediumN", "validation_Few",
        "mean_delta_l2_per_sample", "mean_relative_delta_l2",
        "mean_class_gradient_norm_before_normalization", "mean_adv_minus_clean_ce",
        "zero_gradient_classes", "steps", "samples",
    ]
    trace_fields = [
        "epoch", "batch", "batch_size", "indices_labels_sha256",
        "augmented_images_sha256", "gap_features_sha256",
    ]
    status = {
        "state": "running",
        "seed": int(seed),
        "arm": arm,
        "started_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "clean_stage1_sha256": manifest["clean_stage1_sha256"],
        "initial_backbone_sha256": initial_backbone_hash,
        "initial_classifier_sha256": initial_classifier_hash,
        "optimizer": {"type": "SGD", "lr": STAGE2_LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY, "nesterov": False},
        "backbone_frozen": True,
        "bn_eval_and_buffers_frozen": True,
        "official_TEST_used": False,
        "epochs": [],
    }
    write_json(out / "status.json", status)

    global_batch = 0
    with trace_path.open("w", encoding="utf-8", newline="") as trace_file, metrics_path.open("w", encoding="utf-8", newline="") as metric_file:
        trace_writer = csv.DictWriter(trace_file, fieldnames=trace_fields)
        metric_writer = csv.DictWriter(metric_file, fieldnames=metric_fields)
        trace_writer.writeheader()
        metric_writer.writeheader()

        for epoch in range(STAGE2_EPOCHS):
            classifier.train()
            epoch_loss_sum = 0.0
            correct = 0
            seen = 0
            delta_sum = 0.0
            relative_delta_sum = 0.0
            grad_norm_sum = 0.0
            adv_ce_delta_sum = 0.0
            zero_gradient_classes = 0
            epoch_steps = 0
            alpha_value = alpha_for_epoch(epoch)

            for batch_index, start in enumerate(range(0, len(train_labels), BATCH_SIZE)):
                end = min(start + BATCH_SIZE, len(train_labels))
                indices = np.asarray(plan[epoch, start:end], dtype=np.int64)
                target_np = np.asarray(train_labels[indices], dtype=np.int64)
                target = torch.from_numpy(target_np).to(device)
                images_cpu = augment_batch(train_images, indices, train_transform, augmentation_generator)
                with torch.no_grad():
                    feature_map = backbone(images_cpu.to(device, non_blocking=True))
                    gap = clean_compatible_gap(feature_map)
                if gap.shape != (end - start, FEATURE_DIM) or not torch.isfinite(gap).all():
                    raise RuntimeError("Malformed or non-finite 64-D GAP features.")

                trace_row = {
                    "epoch": str(epoch + 1),
                    "batch": str(batch_index),
                    "batch_size": str(end - start),
                    "indices_labels_sha256": hash_arrays(indices, target_np),
                    "augmented_images_sha256": hash_arrays(images_cpu.numpy()),
                    "gap_features_sha256": hash_arrays(gap.detach().cpu().numpy()),
                }
                if reference_rows is not None:
                    expected = reference_rows[global_batch]
                    if any(trace_row[key] != expected[key] for key in trace_fields):
                        raise RuntimeError(
                            f"Paired batch mismatch in {arm}, seed={seed}, epoch={epoch + 1}, batch={batch_index}"
                        )
                trace_writer.writerow(trace_row)

                loss, logits_clean, clean_losses, adv_ce, delta, pstats, alpha_value = compute_arm_loss(
                    arm, gap, target, classifier.module.fc, class_weights, epoch
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in classifier.module.fc.parameters()):
                    raise RuntimeError(f"Missing or non-finite classifier gradients: {arm}, seed={seed}")
                if any(p.grad is not None for p in backbone.parameters()):
                    raise RuntimeError("Frozen backbone received a gradient.")
                optimizer.step()

                batch_size = end - start
                epoch_loss_sum += float(loss.detach().cpu()) * batch_size
                correct += int((logits_clean.detach().argmax(dim=1) == target).sum().item())
                seen += batch_size
                delta_sum += pstats["mean_delta_l2_per_sample"] * batch_size
                relative_delta_sum += pstats["mean_relative_delta_l2"] * batch_size
                grad_norm_sum += pstats["mean_class_gradient_norm_before_normalization"] * batch_size
                if "mean_unweighted_adv_minus_clean_ce" in pstats:
                    adv_ce_delta_sum += pstats["mean_unweighted_adv_minus_clean_ce"] * batch_size
                zero_gradient_classes += int(pstats["zero_gradient_classes"])
                epoch_steps += 1
                global_batch += 1

            if seen != len(train_labels):
                raise RuntimeError(f"Epoch sample count mismatch: {seen} != {len(train_labels)}")
            check_backbone_unchanged(backbone, frozen_state)
            validation_metrics, _ = evaluate_gap(
                classifier, validation_gap, validation_labels, counts, device
            )
            row = {
                "epoch": epoch + 1,
                "alpha": float(alpha_value),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": epoch_loss_sum / seen,
                "train_clean_accuracy": 100.0 * correct / seen,
                "validation_Overall": validation_metrics["Overall"],
                "validation_Macro": validation_metrics["Macro"],
                "validation_Head": validation_metrics["Head"],
                "validation_OfficialMedium": validation_metrics["OfficialMedium"],
                "validation_Tail": validation_metrics["Tail"],
                "validation_Many": validation_metrics["Many"],
                "validation_MediumN": validation_metrics["MediumN"],
                "validation_Few": validation_metrics["Few"],
                "mean_delta_l2_per_sample": delta_sum / seen,
                "mean_relative_delta_l2": relative_delta_sum / seen,
                "mean_class_gradient_norm_before_normalization": grad_norm_sum / seen,
                "mean_adv_minus_clean_ce": adv_ce_delta_sum / seen,
                "zero_gradient_classes": zero_gradient_classes,
                "steps": epoch_steps,
                "samples": seen,
            }
            metric_writer.writerow(row)
            metric_file.flush()
            trace_file.flush()

            status["epochs"].append({
                "epoch": epoch + 1,
                "metrics": row,
                "validation": validation_metrics,
                "frozen_backbone_and_bn_bitwise_unchanged": True,
            })
            write_json(out / "status.json", status)
            print(
                f"seed={seed} arm={arm} epoch={epoch + 1}/{STAGE2_EPOCHS} "
                f"loss={row['train_loss']:.5f} val={row['validation_Overall']:.3f} "
                f"tail={row['validation_Tail']:.3f} few={row['validation_Few']:.3f}",
                flush=True,
            )

    check_backbone_unchanged(backbone, frozen_state)
    final_metrics = status["epochs"][-1]["validation"]
    checkpoint_payload = {
        "epoch": STAGE2_EPOCHS,
        "seed": int(seed),
        "arm": arm,
        "state_dict_classifier": {k: v.detach().cpu().clone() for k, v in classifier.module.state_dict().items()},
        "clean_stage1_sha256": manifest["clean_stage1_sha256"],
        "initial_classifier_sha256": initial_classifier_hash,
        "final_validation": final_metrics,
        "primary_epoch_fixed": True,
        "official_TEST_used": False,
        "backbone_retrained": False,
    }
    torch.save(checkpoint_payload, out / "final_classifier.pth")
    status["state"] = "completed"
    status["completed_at"] = __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")
    status["final_epoch"] = STAGE2_EPOCHS
    status["final_validation"] = final_metrics
    status["final_classifier_sha256"] = sha256_file(out / "final_classifier.pth")
    status["official_TEST_used"] = False
    write_json(out / "status.json", status)
    del backbone, classifier, optimizer
    torch.cuda.empty_cache()
    return status


def main():
    parser = argparse.ArgumentParser(description="Controlled CC-SAM Stage II transfer on the frozen Clean Stage I.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    if tuple(args.seeds) != tuple(SEEDS):
        raise RuntimeError(f"This frozen protocol requires Stage II seeds {SEEDS}; got {tuple(args.seeds)}")
    manifest_path = ROOT / "prepared" / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run prepare_ccsam.py successfully before training.")
    preliminary_manifest = read_json(manifest_path)
    manifest_data_root = Path(preliminary_manifest["official_train_pickle"]).resolve().parent.parent
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else manifest_data_root
    if data_root != manifest_data_root:
        raise RuntimeError("The training data-root override differs from the TRAIN data-root sealed during preparation.")
    install_test_guard(data_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled.")
    device = torch.device("cuda:0")
    manifest, train_images, train_labels, validation_gap, validation_labels = load_prepared()
    train_transform, _, transform_sha = build_transforms()
    if transform_sha != manifest["transform_source_sha256"]:
        raise RuntimeError("The audited CIFAR transform source changed after preparation.")

    if args.smoke_only:
        report_path = ROOT / "smoke_report.json"
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing smoke report: {report_path}")
        analytic_check = verify_analytic_gradient()
        smoke = smoke_test(
            SEEDS[0], manifest, train_images, train_labels, validation_gap,
            validation_labels, train_transform
        )
        guard = guard_pass()
        report = {
            "passed": True,
            "completed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
            "purpose": "one real CIFAR-100-LT batch per arm; all optimizer updates discarded",
            "formal_training_started": False,
            "prepared_manifest_sha256": sha256_file(manifest_path),
            "analytic_gradient_equivalence_synthetic": analytic_check,
            "analytic_gradient_equivalence_actual_batch": smoke["arms"]["C1_CCSAM_FEATURE_GRAD"]["analytic_gradient_check"],
            "seed_smoke": smoke,
            "test_guard": guard,
            "official_TEST_used": False,
        }
        write_json(report_path, report)
        print("CC-SAM SMOKE PASS: all three arms, paired inputs, finite gradients, classifier-only updates, frozen backbone/BN.")
        print("No formal training was started.")
        return

    smoke_path = ROOT / "smoke_report.json"
    if not smoke_path.is_file() or not read_json(smoke_path).get("passed"):
        raise RuntimeError("Formal training is blocked until a successful smoke_report.json exists.")
    if read_json(smoke_path).get("prepared_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Smoke report was produced from a different prepared manifest; rerun preparation and smoke review.")
    if ROOT.joinpath("formal_test_guard.json").exists():
        raise FileExistsError("Formal run appears to have been sealed already; refusing a second run.")
    for seed in SEEDS:
        for arm in ARMS:
            train_one_arm(
                seed, arm, manifest, train_images, train_labels,
                validation_gap, validation_labels, train_transform, device
            )
    guard = guard_pass()
    write_json(ROOT / "formal_test_guard.json", guard)
    write_json(ROOT / "formal_completed.json", {
        "completed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "epochs_per_run": STAGE2_EPOCHS,
        "formal_runs": len(SEEDS) * len(ARMS),
        "official_TEST_used": False,
        "primary_metric_epoch": STAGE2_EPOCHS,
    })
    print("CC-SAM FORMAL TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
