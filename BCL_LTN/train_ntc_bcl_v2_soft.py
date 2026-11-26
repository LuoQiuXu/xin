"""Paired NTC-BCL v2 candidate training using estimated T, pi and frozen gamma.

This script is the Stage-1 v2 candidate, not a v1 edit. It reuses the exact
Seed-42 initialization, fixed IF=100 data order, three-view transforms, LR
schedule and epoch budget from the v1 NTC-FULL-SOFT-BCL run. It reads only
observed labels and estimator outputs; truth/ and official TEST are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import (
    BATCH_SIZE, BASE_LR, CHECKPOINT_EPOCHS, EPOCHS, LAMBDA_LC, MOMENTUM,
    MU_BCL, TEMPERATURE, WEIGHT_DECAY, classification_logits_v1,
    file_sha256, state_dict_sha256, set_seed,
)
from BCL_LTN.audit_epoch160_margin_reliability import load_cifar100_train_pixels_only
from BCL_LTN.ntc_bcl_v2_core import (
    EstimatedTransitionForwardCorrection,
    NoiseAwareSoftBalancedContrastiveLossV2,
    prototype_geometry,
    soft_bcl_batch_diagnostics,
)
from BCL_LTN.train_ntc_bcl import (
    ARM_C, K, N, OUTPUT_ROOT, SEED, ObservedTrainDataset, array_hash,
    batch_identity, seed_training_epoch, train_loader, verified_initialization,
)
from BCL_LTN.train_ntc_bcl import DATA_ROOT, INDICES


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def install_truth_read_guard(regime_dir: Path) -> None:
    truth_root = (regime_dir / "truth").resolve()
    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path == truth_root or truth_root in path.parents:
            raise PermissionError(f"NTC-BCL v2 training cannot open offline truth: {path}")
        if path.name in {"clean_labels.npy", "is_noisy.npy", "true_transition.npy", "truth_manifest.json"}:
            raise PermissionError(f"NTC-BCL v2 training cannot open offline truth file: {path}")
    sys.addaudithook(audit)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def semantic_ramp_for_epoch(epoch_index: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch_index < warmup_epochs:
        return 0.0
    return min(1.0, float(epoch_index - warmup_epochs + 1) / float(ramp_epochs))


def aggregate_soft_weights(gamma: np.ndarray, batch_size: int, floor_ratio: float) -> dict:
    k = gamma.shape[1]
    sums = {
        "class_occurrences": np.zeros(k, dtype=np.float64),
        "effective_class_occurrences": np.zeros(k, dtype=np.float64),
        "prototype_positive_mass_by_class": np.zeros(k, dtype=np.float64),
        "prototype_inverse_denominator_by_class": np.zeros(k, dtype=np.float64),
    }
    batch_count = 0
    anchors_total = 0
    positive_total = 0.0
    for start in range(0, len(gamma), batch_size):
        q = gamma[start:start + batch_size]
        diag = soft_bcl_batch_diagnostics(q, support_floor_ratio=floor_ratio)
        anchors = 2 * len(q)
        sums["class_occurrences"] += diag["class_occurrences"]
        sums["effective_class_occurrences"] += diag["effective_class_occurrences"]
        sums["prototype_positive_mass_by_class"] += diag["prototype_positive_mass_by_class"]
        sums["prototype_inverse_denominator_by_class"] += (
            diag["prototype_inverse_denominator_by_class"] * anchors
        )
        positive_total += float(np.asarray(diag["positive_mass_per_anchor"]).sum())
        anchors_total += anchors
        batch_count += 1
    sums["class_occurrences"] /= max(batch_count, 1)
    sums["effective_class_occurrences"] /= max(batch_count, 1)
    sums["prototype_positive_mass_by_class"] /= max(anchors_total, 1)
    sums["prototype_inverse_denominator_by_class"] /= max(anchors_total, 1)
    return {
        **{key: value.tolist() for key, value in sums.items()},
        "mean_positive_mass_per_anchor": positive_total / max(anchors_total, 1),
        "batch_count": batch_count,
        "support_floor_ratio": float(floor_ratio),
    }


def run_step(
    model,
    optimizer,
    correction,
    contrastive,
    views,
    observed,
    raw_ids,
    gamma_by_raw_id,
    device,
    *,
    semantic_ramp: float,
    prototype_input_mode: str,
    smoke: bool = False,
):
    batch = int(observed.shape[0])
    observed_device = observed.to(device, non_blocking=True)
    views_device = [view.to(device, non_blocking=True) for view in views]
    if (len(views_device) != 3
            or any(view.shape != (batch, 3, 32, 32) or view.dtype != torch.float32 for view in views_device)
            or observed_device.dtype != torch.long):
        raise RuntimeError("Invalid three-view FP32 observed-label batch")

    optimizer.zero_grad(set_to_none=True)
    all_views = torch.cat(views_device, dim=0)
    features, raw_logits, _, _ = model(all_views, compute_contrastive=False)
    projected = model.sample_mlp(features)
    v1_logits = classification_logits_v1(raw_logits, batch)
    _, v2, v3 = projected.split(batch, dim=0)
    two_views = torch.stack((v2, v3), dim=1)
    loss_lc = correction(v1_logits, observed_device)

    if semantic_ramp <= 0.0:
        loss_bcl = contrastive(None, two_views, None, semantic_ramp=0.0)
    else:
        classifier_input = model.classifier.weight
        if prototype_input_mode == "direction":
            classifier_input = F.normalize(classifier_input, dim=1)
        elif prototype_input_mode != "raw":
            raise ValueError("prototype_input_mode must be raw or direction")
        prototypes = model.prototype_mlp(classifier_input)
        raw_ids_np = raw_ids.numpy().astype(np.int64, copy=False)
        gamma_batch = torch.from_numpy(gamma_by_raw_id[raw_ids_np]).to(device=device, dtype=torch.float32)
        loss_bcl = contrastive(
            prototypes, two_views, gamma_batch, semantic_ramp=float(semantic_ramp)
        )

    loss_total = float(LAMBDA_LC) * loss_lc + float(MU_BCL) * loss_bcl
    values = torch.stack((loss_lc.detach(), loss_bcl.detach(), loss_total.detach()))
    if (not torch.isfinite(values).all().item() or not torch.isfinite(raw_logits).all().item()
            or raw_logits.dtype != torch.float32 or loss_total.dtype != torch.float32):
        raise RuntimeError("Non-finite or non-FP32 NTC-BCL v2 objective")

    loss_total.backward()
    parameters = dict(model.named_parameters())
    expected_none = {
        name for name in parameters if semantic_ramp <= 0.0 and name.startswith("prototype_mlp.")
    }
    actual_none = {name for name, parameter in parameters.items() if parameter.grad is None}
    if actual_none != expected_none:
        raise RuntimeError(
            f"Gradient connectivity mismatch at semantic_ramp={semantic_ramp}: "
            f"expected {sorted(expected_none)}, got {sorted(actual_none)}"
        )
    active = {name: p for name, p in parameters.items() if name not in expected_none}
    bad = [name for name, p in active.items() if not torch.isfinite(p.grad).all().item()]
    if bad:
        raise RuntimeError(f"Non-finite NTC-BCL v2 gradients: {bad}")

    before = None
    if smoke:
        before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer.step()
    result = {
        "loss_lc": float(values[0].cpu()),
        "loss_bcl": float(values[1].cpu()),
        "loss_total": float(values[2].cpu()),
        "observed_correct": int((v1_logits.detach().argmax(1) == observed_device).sum().item()),
        "semantic_ramp": float(semantic_ramp),
        "prototype_input_mode": prototype_input_mode,
        "prototype_gradient_active": semantic_ramp > 0.0,
    }
    if smoke:
        changed = [name for name, p in parameters.items() if not torch.equal(p.detach(), before[name])]
        if not changed or not set(changed).issubset(active):
            raise RuntimeError("Smoke SGD did not update active parameters cleanly")
        result.update({
            "changed_parameter_tensors": len(changed),
            "prototype_parameters_changed": any(name.startswith("prototype_mlp.") for name in changed),
            "active_gradient_tensors": len(active),
            "nonfinite_gradients": 0,
        })
    return result


def save_checkpoint(path, model, optimizer, epoch, updates, provenance):
    state = model.state_dict()
    state_sha = state_dict_sha256(state)
    with path.open("xb") as stream:
        torch.save({
            "arm": "NTC_V2_SOFT_BCL",
            "epoch": int(epoch),
            "seed": SEED,
            "model_state": state,
            "model_state_sha256": state_sha,
            "optimizer_state": optimizer.state_dict(),
            "optimizer_updates": int(updates),
            "provenance": provenance,
        }, stream)
    return {"epoch": int(epoch), "path": str(path), "sha256": file_sha256(path), "model_state_sha256": state_sha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-labels", type=Path, required=True)
    parser.add_argument("--regime-manifest", type=Path, required=True)
    parser.add_argument("--estimator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-floor-ratio", type=float, default=0.5)
    parser.add_argument("--instance-warmup-epochs", type=int, default=20)
    parser.add_argument("--semantic-ramp-epochs", type=int, default=10)
    parser.add_argument("--prototype-input-mode", choices=("raw", "direction"), default="direction")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    if args.support_floor_ratio < 0 or args.instance_warmup_epochs < 0 or args.semantic_ramp_epochs < 1:
        raise ValueError("Invalid v2 loss/ramp configuration")
    for path in (
        args.observed_labels, args.regime_manifest,
        args.estimator_dir / "estimator_only.json",
        args.estimator_dir / "estimated_clean_prior.npy",
        args.estimator_dir / "estimated_transition.npy",
        args.estimator_dir / "estimated_posterior_gamma.npy",
        INDICES,
        DATA_ROOT / "cifar-100-python" / "train",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")

    manifest = read_json(args.regime_manifest)
    estimator_receipt = read_json(args.estimator_dir / "estimator_only.json")
    indices = np.load(INDICES, allow_pickle=False)
    observed = np.load(args.observed_labels, allow_pickle=False)
    clean_prior = np.load(args.estimator_dir / "estimated_clean_prior.npy", allow_pickle=False)
    transition = np.load(args.estimator_dir / "estimated_transition.npy", allow_pickle=False)
    gamma = np.load(args.estimator_dir / "estimated_posterior_gamma.npy", allow_pickle=False)
    observed_sha = file_sha256(args.observed_labels)
    if (indices.shape != (N,) or indices.dtype != np.int64 or observed.shape != (N,)
            or observed.dtype != np.int64 or np.unique(indices).size != N
            or np.any((observed < 0) | (observed >= K))):
        raise RuntimeError("Invalid fixed TRAIN IDs or observed labels")
    if (manifest.get("indices_sha256") != file_sha256(INDICES)
            or manifest.get("observed_labels_sha256") != observed_sha
            or estimator_receipt.get("observed_labels_sha256") != observed_sha
            or estimator_receipt.get("truth_loaded") is not False):
        raise RuntimeError("Observed-label or estimator provenance mismatch")
    if (clean_prior.shape != (K,) or transition.shape != (K, K)
            or gamma.shape != (N, K) or not np.isfinite(clean_prior).all()
            or not np.isfinite(transition).all() or not np.isfinite(gamma).all()
            or np.any(clean_prior <= 0) or np.any(transition < 0) or np.any(gamma < 0)
            or not np.allclose(clean_prior.sum(), 1.0, atol=1e-6)
            or not np.allclose(transition.sum(1), 1.0, atol=1e-6)
            or not np.allclose(gamma.sum(1), 1.0, atol=1e-5)):
        raise RuntimeError("Estimated pi, T or gamma failed probability validation")
    if args.output_dir.resolve() == (args.observed_labels.parent / "truth").resolve():
        raise RuntimeError("Output directory cannot overlap offline truth")

    install_truth_read_guard(args.observed_labels.parent)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NTC-BCL v2 formal training")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("AMP must remain disabled")

    baseline_history_path = OUTPUT_ROOT / ARM_C / "history.jsonl"
    if not baseline_history_path.is_file():
        raise FileNotFoundError(
            f"Paired v1 C history is required to verify identical data order: {baseline_history_path}"
        )
    baseline_rows = read_jsonl(baseline_history_path)
    if len(baseline_rows) != EPOCHS:
        raise RuntimeError("v1 C baseline history must contain exactly 200 epochs")

    init_state, init_record, _baseline_summary = verified_initialization()
    set_seed(SEED)
    model = BCLCIFARModel()
    model.load_state_dict(init_state, strict=True)
    initial_state_sha = state_dict_sha256(model.state_dict())
    model.to(device).train()

    correction = EstimatedTransitionForwardCorrection(clean_prior, transition).to(device)
    contrastive = NoiseAwareSoftBalancedContrastiveLossV2(
        K, temperature=TEMPERATURE,
        support_floor_ratio=args.support_floor_ratio,
        denominator_epsilon=1e-6,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    pixels = load_cifar100_train_pixels_only(DATA_ROOT)
    from BCL_AUDIT.cifar100lt.data import build_cifar_transforms
    dataset = ObservedTrainDataset(pixels, indices, observed, build_cifar_transforms())
    loader, generator = train_loader(dataset)
    target_by_raw_id = np.full(50_000, -1, dtype=np.int64)
    target_by_raw_id[indices] = observed
    gamma_by_raw_id = np.zeros((50_000, K), dtype=np.float32)
    gamma_by_raw_id[indices] = gamma.astype(np.float32, copy=False)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = args.output_dir / "checkpoints"
    diagnostics_dir = args.output_dir / "train_diagnostics"
    checkpoint_dir.mkdir()
    diagnostics_dir.mkdir()
    provenance = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "observed_labels_sha256": observed_sha,
        "indices_sha256": file_sha256(INDICES),
        "regime_manifest_sha256": file_sha256(args.regime_manifest),
        "estimator_only_sha256": file_sha256(args.estimator_dir / "estimator_only.json"),
        "estimated_prior_sha256": file_sha256(args.estimator_dir / "estimated_clean_prior.npy"),
        "estimated_transition_sha256": file_sha256(args.estimator_dir / "estimated_transition.npy"),
        "estimated_gamma_sha256": file_sha256(args.estimator_dir / "estimated_posterior_gamma.npy"),
        "initial_state_sha256": initial_state_sha,
        "prototype_input_mode": args.prototype_input_mode,
        "support_floor_ratio": args.support_floor_ratio,
        "instance_warmup_epochs": args.instance_warmup_epochs,
        "semantic_ramp_epochs": args.semantic_ramp_epochs,
        "clean_truth_loaded": False,
        "official_test_opened": False,
    }
    write_json(args.output_dir / "environment.json", provenance)
    prototype_initial_hash = state_dict_sha256(model.prototype_mlp.state_dict())

    if args.smoke_only:
        epoch_seed, lr = seed_training_epoch(0, generator, optimizer)
        model.train()
        batch = next(iter(loader))
        views, observed_batch, raw_ids = batch
        first_identity = batch_identity(views, observed_batch, raw_ids)
        baseline_first = baseline_rows[0]["first_batch"]
        if first_identity != baseline_first:
            raise RuntimeError("v2 Smoke first batch differs from paired v1 C")
        warmup_result = run_step(
            model, optimizer, correction, contrastive, views, observed_batch, raw_ids,
            gamma_by_raw_id, device, semantic_ramp=0.0,
            prototype_input_mode=args.prototype_input_mode, smoke=True,
        )
        semantic_result = run_step(
            model, optimizer, correction, contrastive, views, observed_batch, raw_ids,
            gamma_by_raw_id, device, semantic_ramp=0.1,
            prototype_input_mode=args.prototype_input_mode, smoke=True,
        )
        receipt = {
            **provenance,
            "passed": True,
            "epoch_seed": epoch_seed,
            "lr": lr,
            "first_batch": first_identity,
            "warmup_step": warmup_result,
            "semantic_ramp_step": semantic_result,
            "prototype_initial_state_sha256": prototype_initial_hash,
            "prototype_state_after_smoke_sha256": state_dict_sha256(model.prototype_mlp.state_dict()),
            "formal_training_started": False,
        }
        write_json(args.output_dir / "smoke_receipt.json", receipt)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return

    history_path = args.output_dir / "history.jsonl"
    checkpoints = []
    rows = []
    total_updates = 0
    started = time.perf_counter()
    with history_path.open("x", encoding="utf-8") as history:
        for epoch_index in range(EPOCHS):
            epoch = epoch_index + 1
            epoch_seed, lr = seed_training_epoch(epoch_index, generator, optimizer)
            ramp = semantic_ramp_for_epoch(
                epoch_index, args.instance_warmup_epochs, args.semantic_ramp_epochs
            )
            model.train()
            all_indices = []
            label_counts = np.zeros(K, dtype=np.int64)
            augmentation_hash = hashlib.sha256()
            first_identity = None
            sums = np.zeros(3, dtype=np.float64)
            seen = correct = 0
            for batch_index, (views, observed_batch, raw_ids) in enumerate(loader):
                if batch_index == 0:
                    first_identity = batch_identity(views, observed_batch, raw_ids)
                raw_np = raw_ids.numpy().astype(np.int64, copy=False)
                observed_np = observed_batch.numpy().astype(np.int64, copy=False)
                if not np.array_equal(observed_np, target_by_raw_id[raw_np]):
                    raise RuntimeError("Observed labels do not align with raw TRAIN IDs")
                all_indices.append(raw_np.copy())
                label_counts += np.bincount(observed_np, minlength=K)
                for view in views:
                    augmentation_hash.update(memoryview(view.contiguous().numpy()).cast("B"))

                step = run_step(
                    model, optimizer, correction, contrastive, views, observed_batch, raw_ids,
                    gamma_by_raw_id, device, semantic_ramp=ramp,
                    prototype_input_mode=args.prototype_input_mode,
                )
                sums += np.asarray(
                    [step["loss_lc"], step["loss_bcl"], step["loss_total"]], dtype=np.float64
                ) * len(observed_batch)
                seen += len(observed_batch)
                correct += step["observed_correct"]
                total_updates += 1

            order = np.concatenate(all_indices)
            sample_order_sha = array_hash(order)
            augmentation_sha = augmentation_hash.hexdigest()
            if (seen != N or not np.array_equal(np.sort(order), np.sort(indices))
                    or not np.array_equal(label_counts, np.bincount(observed, minlength=K))):
                raise RuntimeError(f"Epoch {epoch}: fixed TRAIN coverage/count audit failed")

            baseline = baseline_rows[epoch_index]
            for key, current in (
                ("epoch_seed", epoch_seed),
                ("lr", lr),
                ("sample_order_sha256", sample_order_sha),
                ("all_three_view_augmentations_sha256", augmentation_sha),
                ("first_batch", first_identity),
                ("observed_noisy_class_counts", label_counts.tolist()),
            ):
                if baseline.get(key) != current:
                    raise RuntimeError(f"Epoch {epoch}: paired v1 C protocol mismatch in {key}")

            if epoch == args.instance_warmup_epochs:
                prototype_end_hash = state_dict_sha256(model.prototype_mlp.state_dict())
                if (prototype_end_hash != prototype_initial_hash
                        or any(parameter in optimizer.state for parameter in model.prototype_mlp.parameters())):
                    raise RuntimeError("Prototype MLP changed or accumulated optimizer state during instance warmup")
                write_json(args.output_dir / "instance_warmup_completed.json", {
                    "completed_epochs": epoch,
                    "prototype_state_sha256_before": prototype_initial_hash,
                    "prototype_state_sha256_after": prototype_end_hash,
                    "prototype_optimizer_state_entries": 0,
                })

            row = {
                "epoch": epoch,
                "epoch_index": epoch_index,
                "epoch_seed": epoch_seed,
                "lr": lr,
                "semantic_ramp": ramp,
                "prototype_input_mode": args.prototype_input_mode,
                "loss_lc": float(sums[0] / seen),
                "loss_bcl": float(sums[1] / seen),
                "loss_total": float(sums[2] / seen),
                "observed_correct": correct,
                "observed_accuracy_percent": 100.0 * correct / seen,
                "samples": seen,
                "optimizer_updates": total_updates,
                "observed_noisy_class_counts": label_counts.tolist(),
                "sample_order_sha256": sample_order_sha,
                "all_three_view_augmentations_sha256": augmentation_sha,
                "first_batch": first_identity,
                "clean_truth_loaded": False,
            }

            if epoch in CHECKPOINT_EPOCHS:
                geometry = prototype_geometry(model.classifier.weight, model.prototype_mlp)
                soft_diag = aggregate_soft_weights(
                    gamma, BATCH_SIZE, args.support_floor_ratio
                )
                diagnostic = {
                    "epoch": epoch,
                    "semantic_ramp": ramp,
                    "prototype_input_mode": args.prototype_input_mode,
                    "classifier_weight_norm": geometry["classifier_weight_norm"],
                    "classifier_weight_norm_min": geometry["classifier_weight_norm_min"],
                    "classifier_weight_norm_median": geometry["classifier_weight_norm_median"],
                    "classifier_weight_norm_max": geometry["classifier_weight_norm_max"],
                    "prototype_geometry_raw_input": geometry["raw_weight_input"],
                    "prototype_geometry_direction_input": geometry["direction_normalized_weight_input"],
                    "gamma_class_mass": gamma.sum(axis=0).tolist(),
                    "gamma_mean_entropy": float(
                        -(gamma * np.log(np.maximum(gamma, 1e-12))).sum(axis=1).mean()
                    ),
                    "soft_bcl_weight_diagnostics": soft_diag,
                    "model_state_sha256": state_dict_sha256(model.state_dict()),
                }
                write_json(diagnostics_dir / f"epoch_{epoch:03d}.json", diagnostic)
                entry = save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:03d}.pth",
                    model, optimizer, epoch, total_updates, provenance,
                )
                row["checkpoint"] = entry
                checkpoints.append(entry)

            rows.append(row)
            history.write(json.dumps(row, allow_nan=False) + "\n")
            history.flush()
            print(
                f"[NTC-V2] epoch {epoch:03d}/{EPOCHS} ramp={ramp:.2f} "
                f"LC={row['loss_lc']:.5f} BCL={row['loss_bcl']:.5f} "
                f"total={row['loss_total']:.5f} acc_obs={row['observed_accuracy_percent']:.2f}%",
                flush=True,
            )

    if (len(rows) != EPOCHS or total_updates != EPOCHS * len(loader)
            or tuple(item["epoch"] for item in checkpoints) != CHECKPOINT_EPOCHS):
        raise RuntimeError("NTC-BCL v2 did not complete the paired 200-epoch protocol")

    summary = {
        **provenance,
        "epochs_completed": len(rows),
        "optimizer_updates": total_updates,
        "checkpoints": checkpoints,
        "last_epoch": rows[-1],
        "total_training_seconds": time.perf_counter() - started,
        "final_model_state_sha256": state_dict_sha256(model.state_dict()),
        "final_prototype_geometry": prototype_geometry(model.classifier.weight, model.prototype_mlp),
        "gamma_class_mass": gamma.sum(axis=0).tolist(),
        "gamma_mean_entropy": float(
            -(gamma * np.log(np.maximum(gamma, 1e-12))).sum(axis=1).mean()
        ),
        "clean_truth_loaded": False,
        "official_test_opened": False,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps({
        "epochs_completed": summary["epochs_completed"],
        "optimizer_updates": summary["optimizer_updates"],
        "final_model_state_sha256": summary["final_model_state_sha256"],
        "final_prototype_raw_cosine": summary["final_prototype_geometry"]["raw_weight_input"]["mean_off_diagonal_cosine"],
        "final_prototype_direction_cosine": summary["final_prototype_geometry"]["direction_normalized_weight_input"]["mean_off_diagonal_cosine"],
        "total_training_seconds": summary["total_training_seconds"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
