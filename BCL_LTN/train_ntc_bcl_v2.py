"""Single-pass 200-epoch online self-evolving NTC-BCL v2 trainer.

Trains CIFAR-100-LT (ResNet-32 + BCL dual projection heads) under arbitrary
unknown label noise without known noise rates, clean labels, corruption masks,
or symmetry assumptions:
1. Warmup (epochs 1..warmup_epochs): instance-only NT-Xent + observed-prior LC
   with identity transition, while continuously updating an EMA teacher.
2. Progressive self-evolution: every `estimate_interval` epochs starting at
   `warmup_epochs`, an eval-mode deterministic TRAIN pass of the EMA teacher
   produces balanced predictions p_bal(c | x) = softmax(ema_logits).
   `estimate_anchored_transition_and_prior` estimates arbitrary row-stochastic
   T[c, j] and regularized clean prior pi via visual-confidence anchoring +
   concave EM, smoothly blended into Forward Correction buffers.
3. Soft-BCL v2: detached EMA-teacher balanced soft targets q guide contrastive
   learning with progressive semantic_ramp annealing, support_floor_ratio,
   denominator_epsilon, and direction-normalized classifier weights fed to
   prototype_mlp.
4. Strict data isolation: RegimeTruthGate blocks `<regime_dir>/truth/` and
   v1 truth files until all training and frozen TRAIN predictions are complete;
   OfficialTestAudit blocks official CIFAR-100 TEST until all training and
   TRAIN diagnostics finish, opening TEST strictly once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from BCL_AUDIT.cifar100lt.data import build_cifar_transforms, cifar100lt_class_counts
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import (
    BASE_LR,
    BATCH_SIZE,
    CHECKPOINT_EPOCHS,
    EPOCHS,
    LAMBDA_LC,
    LR_GAMMA,
    LR_MILESTONES,
    MOMENTUM,
    MU_BCL,
    TEMPERATURE,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    classification_logits_v1,
    compose_group_loss,
    epoch_seed_for,
    file_sha256,
    learning_rate_for_epoch,
    load_initial_model_state,
    seed_worker,
    set_seed,
    state_dict_sha256,
)
from BCL_LTN.audit_epoch160_margin_reliability import (
    EvalOnlyTrainDataset,
    load_cifar100_train_pixels_only,
)
from BCL_LTN.make_noisy_lt import group_class_ids
from BCL_LTN.ntc_bcl_v2_losses import (
    EMATeacher,
    EstimatedTransitionForwardCorrection,
    NoiseAwareSoftBalancedContrastiveLossV2,
    blend_transition_and_prior,
    estimate_anchored_transition_and_prior,
    measure_prototype_and_classifier_health,
)
from BCL_LTN.ntc_bcl_v21_calibration import (
    calibrate_clean_prior,
    observed_fit_statistics,
)
from BCL_LTN.train_noisy_bcl import (
    FIXED_INDICES_SHA256,
    FIXED_TRAIN_SOURCE_SHA256,
    OfficialTestAudit,
    TrainDiagnosticDataset,
    json_dump,
    safe_torch_load,
    write_train_diagnostics_csv,
)
from BCL_LTN.train_ntc_bcl import (
    INITIAL_STATE_SHA,
    ObservedTrainDataset,
    array_hash,
    batch_identity,
    read_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "H2T" / "data" / "cifar100"
DEFAULT_STAGE1_DATA_ROOT = ROOT / "BCL_LTN" / "outputs" / "ntc_bcl_v2_stage1_data_20261126"
DEFAULT_INDICES = (
    ROOT
    / "BCL_AUDIT"
    / "outputs"
    / "data_audit_recheck_20261122_01"
    / "cifar100lt_if100_train_indices_seed0.npy"
)
DEFAULT_BASE_ROOT = ROOT / "BCL_LTN" / "outputs" / "BCL_sym04_seed42"
LEGACY_NOISE_ROOT = ROOT / "BCL_LTN" / "data"
REGIMES = ("sym02", "sym04", "sym06", "asym04_coarse_cycle", "cifar100n_real")
N, K, SEED = 10_847, 100, 42
STEPS_PER_EPOCH = (N + BATCH_SIZE - 1) // BATCH_SIZE


def checked_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {path}")


class RegimeTruthGate:
    """Block truth/ and clean sidecars until TRAIN predictions are sealed."""

    def __init__(self, regime_dir: Path, legacy_noise_root: Path = LEGACY_NOISE_ROOT):
        self.truth_prefix = os.path.normcase(str((regime_dir / "truth").resolve()))
        self.truth_sep = self.truth_prefix + os.sep
        self.legacy_targets = {
            os.path.normcase(str((legacy_noise_root / name).resolve()))
            for name in ("clean_labels.npy", "is_noisy.npy")
        }
        self.opens = 0
        self.denied = 0
        self.allowed = False
        self._hook_installed = False

    def install(self) -> None:
        if not self._hook_installed:
            sys.addaudithook(self._audit)
            self._hook_installed = True

    def _is_protected(self, raw_path) -> bool:
        if not isinstance(raw_path, (str, bytes)):
            return False
        norm = os.path.normcase(os.path.abspath(os.fsdecode(raw_path)))
        return (
            norm == self.truth_prefix
            or norm.startswith(self.truth_sep)
            or norm in self.legacy_targets
        )

    def _audit(self, event, args) -> None:
        if event not in ("open", "os.listdir", "os.scandir") or not args:
            return
        if self._is_protected(args[0]):
            if not self.allowed:
                self.denied += 1
                raise RuntimeError("Regime truth path was accessed before prediction freeze")
            if event == "open":
                self.opens += 1

    @contextmanager
    def offline_only(self):
        if self.opens or self.denied:
            raise RuntimeError("Unclean sample-truth access history before offline diagnostics")
        self.allowed = True
        try:
            yield
        finally:
            self.allowed = False


def semantic_ramp_for_epoch(
    epoch_index: int, warmup_epochs: int, semantic_ramp_epochs: int
) -> float:
    """Compute progressive semantic contrastive strength in [0, 1]."""
    if epoch_index < warmup_epochs:
        return 0.0
    if semantic_ramp_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch_index - warmup_epochs + 1) / float(semantic_ramp_epochs))


def load_regime_observed_inputs(
    regime_dir: Path, data_root: Path, indices_file: Path
) -> Tuple[ObservedTrainDataset, np.ndarray, Dict[str, object]]:
    """Load CIFAR-100-LT TRAIN pixels and observed labels only (never truth/)."""
    manifest_path = regime_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if int(manifest.get("num_samples", -1)) != N or int(manifest.get("num_classes", -1)) != K:
        raise RuntimeError(f"Unexpected regime dimensions in {manifest_path}")
    if manifest.get("indices_sha256") != FIXED_INDICES_SHA256:
        raise RuntimeError("Regime manifest indices_sha256 mismatch")

    observed_name = str(manifest.get("observed_labels_file", "observed_labels.npy"))
    observed_path = regime_dir / observed_name
    checked_file(observed_path, str(manifest["observed_labels_sha256"]))
    checked_file(indices_file, FIXED_INDICES_SHA256)
    for name, digest in FIXED_TRAIN_SOURCE_SHA256.items():
        checked_file(data_root / "cifar-100-python" / name, digest)

    indices = np.load(indices_file, allow_pickle=False)
    observed = np.load(observed_path, allow_pickle=False)
    if (
        indices.shape != (N,)
        or indices.dtype != np.int64
        or np.unique(indices).size != N
        or np.any((indices < 0) | (indices >= 50_000))
    ):
        raise RuntimeError("Invalid fixed CIFAR-100-LT IF=100 indices")
    if observed.shape != (N,) or observed.dtype != np.int64 or np.any((observed < 0) | (observed >= K)):
        raise RuntimeError("Invalid observed labels array")
    counts = np.bincount(observed, minlength=K)
    if counts.tolist() != manifest["observed_class_counts"]:
        raise RuntimeError("Observed class counts differ from manifest")

    pixels = load_cifar100_train_pixels_only(data_root)
    dataset = ObservedTrainDataset(pixels, indices, observed, build_cifar_transforms())
    return dataset, counts, manifest


def verified_seed42_initialization(base_root: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Restore the locked Seed-42 BCLCIFARModel initialization matching BCL and v1."""
    init_file = base_root / "init_seed42.pth"
    if init_file.is_file():
        state = load_initial_model_state(init_file, expected_seed=SEED, expected_sha256=INITIAL_STATE_SHA)
        source = {"source": str(init_file), "file_sha256": file_sha256(init_file)}
    else:
        set_seed(SEED)
        state = BCLCIFARModel().state_dict()
        source = {"source": "set_seed(42) + BCLCIFARModel"}
    actual = state_dict_sha256(state)
    if actual != INITIAL_STATE_SHA:
        raise RuntimeError(f"Initial Seed-42 SHA256 mismatch: {actual} != {INITIAL_STATE_SHA}")
    return state, {**source, "model_state_sha256": actual}


def make_train_loader(dataset: ObservedTrainDataset) -> Tuple[DataLoader, torch.Generator]:
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    return loader, generator


def configure_fp32_determinism(device: torch.device) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_epoch(epoch_index: int, generator: torch.Generator, optimizer: torch.optim.Optimizer):
    epoch_seed = epoch_seed_for(SEED, epoch_index)
    set_seed(epoch_seed)
    generator.manual_seed(epoch_seed)
    lr = learning_rate_for_epoch(epoch_index)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return epoch_seed, lr


def watch_calls(model: BCLCIFARModel, optimizer: torch.optim.Optimizer) -> Counter:
    counts = Counter({"backbone": 0, "sample_mlp": 0, "prototype_mlp": 0, "optimizer": 0})

    def tally(name: str):
        def hook(*_args):
            counts[name] += 1
        return hook

    for name in ("backbone", "sample_mlp", "prototype_mlp"):
        getattr(model, name).register_forward_hook(tally(name))
    optimizer.register_step_post_hook(tally("optimizer"))
    return counts


def sgd_step_v2(
    model: BCLCIFARModel,
    ema_teacher: EMATeacher,
    views: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    observed: torch.Tensor,
    correction: EstimatedTransitionForwardCorrection,
    contrastive: NoiseAwareSoftBalancedContrastiveLossV2,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    semantic_ramp: float,
    q_temperature: float,
    prototype_input_mode: str,
    smoke: bool = False,
) -> Dict[str, object]:
    """Run one matched 3-view FP32 forward-backward-SGD step and update EMA teacher."""
    if prototype_input_mode not in ("direction", "raw"):
        raise ValueError("prototype_input_mode must be 'direction' or 'raw'")
    observed = observed.to(device, non_blocking=True)
    gpu_views = [v.to(device, non_blocking=True) for v in views]
    batch_size = int(observed.shape[0])
    if (
        len(gpu_views) != 3
        or any(v.shape != (batch_size, 3, 32, 32) or v.dtype != torch.float32 for v in gpu_views)
        or observed.dtype != torch.long
    ):
        raise RuntimeError("Invalid three-view FP32 TRAIN batch")

    optimizer.zero_grad(set_to_none=True)
    all_views = torch.cat(gpu_views, dim=0)
    features, raw_logits, _, _ = model(all_views, compute_contrastive=False)
    sample_embeddings = model.sample_mlp(features)
    v1_logits = classification_logits_v1(raw_logits, batch_size)
    _, view2_emb, view3_emb = sample_embeddings.split(batch_size, dim=0)
    two_views = torch.stack((view2_emb, view3_emb), dim=1)

    expected_unused = semantic_ramp == 0.0
    if expected_unused:
        bcl_loss, bcl_diag = contrastive(
            None, two_views, semantic_ramp=0.0, return_diagnostics=True
        )
    else:
        proto_input = (
            F.normalize(model.classifier.weight, dim=1)
            if prototype_input_mode == "direction"
            else model.classifier.weight
        )
        prototypes = model.prototype_mlp(proto_input)
        with torch.no_grad():
            _, teacher_raw, _, _ = ema_teacher.model(all_views, compute_contrastive=False)
            teacher_three = teacher_raw.detach().reshape(3, batch_size, K).mean(dim=0)
            q = F.softmax(teacher_three / float(q_temperature), dim=1)
        bcl_loss, bcl_diag = contrastive(
            prototypes,
            two_views,
            q,
            semantic_ramp=semantic_ramp,
            return_diagnostics=True,
        )

    lc_loss = correction(v1_logits, observed)
    total_loss = compose_group_loss("BCL", lc_loss, bcl_loss)

    loss_vec = torch.stack((lc_loss.detach(), bcl_loss.detach(), total_loss.detach()))
    if (
        loss_vec.dtype != torch.float32
        or raw_logits.dtype != torch.float32
        or not bool(torch.isfinite(loss_vec).all())
    ):
        raise RuntimeError(f"Non-finite or non-FP32 loss at ramp={semantic_ramp}")

    total_loss.backward()

    parameters = dict(model.named_parameters())
    expected_none = {
        name for name in parameters if expected_unused and name.startswith("prototype_mlp.")
    }
    actual_none = {name for name, p in parameters.items() if p.grad is None}
    if actual_none != expected_none:
        raise RuntimeError(
            f"Gradient connectivity mismatch (ramp={semantic_ramp}): "
            f"expected None={sorted(expected_none)}, got {sorted(actual_none)}"
        )
    active = {name: p for name, p in parameters.items() if name not in expected_none}
    flat_grads = torch.cat([p.grad.detach().flatten() for p in active.values()])
    if not bool(torch.isfinite(flat_grads).all()):
        bad = [n for n, p in active.items() if not bool(torch.isfinite(p.grad).all())]
        raise RuntimeError(f"Non-finite active gradients: {bad}")

    if smoke:
        before = {name: p.detach().clone() for name, p in parameters.items()}

    optimizer.step()
    ema_alpha = ema_teacher.update(model)

    result: Dict[str, object] = {
        "loss_lc": float(loss_vec[0].item()),
        "loss_bcl": float(loss_vec[1].item()),
        "loss_total": float(loss_vec[2].item()),
        "observed_correct": int((v1_logits.detach().argmax(1) == observed).sum().item()),
        "ema_alpha": float(ema_alpha),
        **bcl_diag,
    }
    if smoke:
        flat_after = torch.cat([p.detach().flatten() for p in parameters.values()])
        if not bool(torch.isfinite(flat_after).all()):
            raise RuntimeError("Non-finite parameters after Smoke SGD")
        changed = [n for n, p in parameters.items() if not torch.equal(p.detach(), before[n])]
        if not changed or not set(changed).issubset(active):
            raise RuntimeError("Unexpected inactive parameter update during Smoke")
        for prefix in (
            "backbone.",
            "classifier.",
            "sample_mlp.",
            *(("prototype_mlp.",) if not expected_unused else ()),
        ):
            if not any(n.startswith(prefix) for n in changed):
                raise RuntimeError(f"No Smoke parameter update in active branch {prefix}")
        result.update(
            {
                "active_gradient_tensors": len(active),
                "expected_grad_none": sorted(actual_none),
                "changed_parameter_tensors": len(changed),
                "max_abs_parameter_change": max(
                    float((parameters[n].detach() - before[n]).abs().max()) for n in changed
                ),
                "raw_three_view_logits_sha256": array_hash(raw_logits.detach().cpu().numpy()),
            }
        )
    return result


@torch.inference_mode()
def evaluate_deterministic_train_probabilities(
    eval_model: BCLCIFARModel, dataset: ObservedTrainDataset, device: torch.device
) -> Tuple[torch.Tensor, np.ndarray]:
    """Deterministic eval pass on fixed TRAIN images; returns (probs [N, K], predicted [N])."""
    was_training = eval_model.training
    before_sha = state_dict_sha256(eval_model.state_dict())
    eval_model.eval()
    try:
        eval_ds = EvalOnlyTrainDataset(
            dataset.pixels, dataset.indices, dataset.eval_transform
        )
        loader = DataLoader(
            eval_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        logits_all = np.empty((N, K), dtype=np.float32)
        seen = np.zeros(N, dtype=np.bool_)
        for images, positions in loader:
            pos = positions.numpy()
            if seen[pos].any():
                raise RuntimeError("Duplicate TRAIN position during eval")
            _, logits, _, _ = eval_model(
                images.to(device, non_blocking=True), compute_contrastive=False
            )
            if logits.shape != (len(pos), K) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Model produced invalid TRAIN logits")
            logits_all[pos] = logits.float().cpu().numpy()
            seen[pos] = True
        if not seen.all():
            raise RuntimeError("Eval pass missed TRAIN positions")
    finally:
        eval_model.train(was_training)

    if state_dict_sha256(eval_model.state_dict()) != before_sha:
        raise RuntimeError("Eval pass mutated model weights or BN running stats")
    logits_t = torch.from_numpy(logits_all).to(torch.float64)
    probs = F.softmax(logits_t, dim=1)
    predicted = logits_all.argmax(axis=1).astype(np.int64)
    return probs, predicted


@torch.inference_mode()
def predict_classes(
    eval_model: BCLCIFARModel, loader_dataset, device: torch.device
) -> np.ndarray:
    """Deterministic top-1 prediction on a TrainDiagnosticDataset."""
    was_training = eval_model.training
    before_sha = state_dict_sha256(eval_model.state_dict())
    eval_model.eval()
    preds = []
    try:
        loader = DataLoader(
            loader_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        for images, _indices in loader:
            _, logits, _, _ = eval_model(
                images.to(device, non_blocking=True), compute_contrastive=False
            )
            preds.append(logits.argmax(dim=1).cpu().numpy())
    finally:
        eval_model.train(was_training)
    if state_dict_sha256(eval_model.state_dict()) != before_sha:
        raise RuntimeError("Prediction mutated model state")
    return np.concatenate(preds, axis=0).astype(np.int64)


def build_training_objects(
    init_state: Dict[str, torch.Tensor],
    observed_counts: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
):
    set_seed(SEED)
    model = BCLCIFARModel()
    model.load_state_dict(init_state, strict=True)
    if state_dict_sha256(model.state_dict()) != INITIAL_STATE_SHA:
        raise RuntimeError("Restored model initial SHA256 mismatch")
    model.to(device).train()

    ema_teacher = EMATeacher(model, decay=args.ema_decay, warmup_ramp=True)
    ema_teacher.model.to(device).eval()

    init_prior = torch.as_tensor(observed_counts, dtype=torch.float64)
    init_prior = init_prior / init_prior.sum()
    init_transition = torch.eye(K, dtype=torch.float64)
    correction = EstimatedTransitionForwardCorrection(init_prior, init_transition).to(device)
    contrastive = NoiseAwareSoftBalancedContrastiveLossV2(
        num_classes=K,
        temperature=TEMPERATURE,
        support_floor_ratio=args.support_floor_ratio,
        denominator_epsilon=args.denominator_epsilon,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    return model, ema_teacher, correction, contrastive, optimizer, init_prior, init_transition


def run_gpu_smoke(
    output_dir: Path,
    dataset: ObservedTrainDataset,
    observed_counts: np.ndarray,
    init_state: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    provenance: Dict[str, object],
    device: torch.device,
    truth_gate: RegimeTruthGate,
    test_gate: OfficialTestAudit,
) -> Dict[str, object]:
    """Run 1-batch FP32 Smoke on both warmup (ramp=0) and semantic (ramp>0) + EM check."""
    torch.cuda.empty_cache()
    records = {}
    for phase_name, epoch_idx, ramp_val in (
        ("warmup_instance", 0, 0.0),
        ("semantic_active", max(1, args.warmup_epochs), 0.5),
    ):
        model, ema_teacher, correction, contrastive, optimizer, cur_pi, cur_t = (
            build_training_objects(init_state, observed_counts, args, device)
        )
        calls = watch_calls(model, optimizer)
        loader, generator = make_train_loader(dataset)
        epoch_seed, lr = seed_epoch(epoch_idx, generator, optimizer)
        views, observed, indices = next(iter(loader))
        if len(observed) != BATCH_SIZE or indices.unique().numel() != BATCH_SIZE:
            raise RuntimeError("Smoke batch is not a full unique 256-sample batch")

        proto_before = state_dict_sha256(model.prototype_mlp.state_dict())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        step = sgd_step_v2(
            model,
            ema_teacher,
            views,
            observed,
            correction,
            contrastive,
            optimizer,
            device,
            semantic_ramp=ramp_val,
            q_temperature=args.q_temperature,
            prototype_input_mode=args.prototype_input_mode,
            smoke=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        proto_unchanged = state_dict_sha256(model.prototype_mlp.state_dict()) == proto_before
        expected_calls = {
            "backbone": 1,
            "sample_mlp": 1,
            "prototype_mlp": 0 if ramp_val == 0.0 else 1,
            "optimizer": 1,
        }
        sgd_calls = dict(calls)
        if sgd_calls != expected_calls:
            raise RuntimeError(f"Smoke call count mismatch in {phase_name}: {sgd_calls}")
        if ramp_val == 0.0 and not proto_unchanged:
            raise RuntimeError("prototype_mlp changed during instance warmup Smoke")

        em_test = None
        if phase_name == "semantic_active":
            probs, _ = evaluate_deterministic_train_probabilities(
                ema_teacher.model, dataset, device
            )
            obs_t = torch.from_numpy(dataset.observed).long()
            fit_stats = observed_fit_statistics(probs, obs_t)
            est = estimate_anchored_transition_and_prior(
                probs,
                obs_t,
                confidence_power=args.confidence_power,
                shrinkage_mass=args.shrinkage_mass,
                prior_pseudocount=args.prior_pseudocount,
                reference_prior=cur_pi,
            )
            calibrated_prior, calibration_diag = calibrate_clean_prior(
                torch.as_tensor(observed_counts, dtype=torch.float64),
                est.transition,
                est.clean_prior,
                prior_min=float(args.prior_min_count) / float(N),
                lambda_h=args.prior_calibration_lambda,
                max_iter=args.prior_calibration_max_iter,
                tolerance=args.prior_calibration_tolerance,
            )
            new_pi, new_t = blend_transition_and_prior(
                cur_pi, cur_t, calibrated_prior, est.transition,
                blend_alpha=args.transition_blend
            )
            correction.update_transition_and_prior(new_pi, new_t)
            health = measure_prototype_and_classifier_health(
                model.classifier.weight, model.prototype_mlp
            )
            em_test = {
                "raw_em_noise_rate": est.estimated_noise_rate,
                "calibrated_noise_rate": float(
                    1.0 - (calibrated_prior * est.transition.diag()).sum().item()
                ),
                "em_iterations": est.iterations,
                "em_converged": est.converged,
                "blended_prior_min": float(new_pi.min().item()),
                "prior_calibration": calibration_diag,
                **fit_stats,
                "classifier_norm_min": health["classifier_weight_norm_min"],
                "dir_mean_offdiag_cosine": health["dir_mean_offdiag_cosine"],
            }

        if truth_gate.opens or truth_gate.denied or test_gate.opens or test_gate.denied:
            raise RuntimeError("Isolation gate violated during GPU Smoke")

        records[phase_name] = {
            "passed": True,
            "epoch_index": epoch_idx,
            "epoch_seed": epoch_seed,
            "lr": lr,
            "semantic_ramp": ramp_val,
            "identity": batch_identity(views, observed, indices),
            "actual_calls": sgd_calls,
            "prototype_state_unchanged": proto_unchanged,
            "seconds": elapsed,
            "em_check": em_test,
            **step,
        }
        print(
            f"[SMOKE v2:{phase_name}] LC={step['loss_lc']:.4f} BCL={step['loss_bcl']:.4f} "
            f"calls={sgd_calls}",
            flush=True,
        )
        del model, ema_teacher, correction, contrastive, optimizer

    summary = {
        "passed": True,
        "regime": args.regime,
        "provenance": provenance,
        "official_test_opens": test_gate.opens,
        "truth_opens": truth_gate.opens,
        "phases": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(output_dir / "smoke_receipt.json", summary)
    return summary


def save_checkpoint(
    path: Path,
    model: BCLCIFARModel,
    ema_teacher: EMATeacher,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    prior: torch.Tensor,
    transition: torch.Tensor,
    epoch: int,
    updates: int,
    provenance: Dict[str, object],
) -> Dict[str, object]:
    model_state = model.state_dict()
    ema_state = ema_teacher.model.state_dict()
    model_sha = state_dict_sha256(model_state)
    ema_sha = state_dict_sha256(ema_state)
    with path.open("xb") as stream:
        torch.save(
            {
                "epoch": epoch,
                "seed": SEED,
                "model_state": model_state,
                "model_state_sha256": model_sha,
                "ema_model_state": ema_state,
                "ema_state_sha256": ema_sha,
                "ema_meta": {
                    "decay": ema_teacher.decay,
                    "warmup_ramp": ema_teacher.warmup_ramp,
                    "num_updates": ema_teacher.num_updates,
                },
                "estimated_clean_prior": prior.cpu().numpy(),
                "estimated_transition": transition.cpu().numpy(),
                "optimizer_state": optimizer.state_dict(),
                "optimizer_updates": updates,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
                "loader_generator_state": generator.get_state(),
                "provenance": provenance,
            },
            stream,
        )
    return {
        "epoch": epoch,
        "path": str(path),
        "sha256": file_sha256(path),
        "model_state_sha256": model_sha,
        "ema_state_sha256": ema_sha,
    }


def compute_subset_diagnostics(
    prediction: np.ndarray,
    clean: np.ndarray,
    noisy_mask: np.ndarray,
    observed: np.ndarray,
    groups: Dict[str, np.ndarray],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    clean_acc = {}
    wrong_mem = {}
    for name, classes in groups.items():
        source = np.isin(clean, classes)
        good = source & (~noisy_mask)
        flipped = source & noisy_mask
        nc, nn_cnt = int(good.sum()), int(flipped.sum())
        correct = int((prediction[good] == clean[good]).sum()) if nc > 0 else 0
        memorized = int((prediction[flipped] == observed[flipped]).sum()) if nn_cnt > 0 else 0
        clean_acc[name] = {
            "correct": correct,
            "total": nc,
            "accuracy_percent": (100.0 * correct / nc) if nc > 0 else 0.0,
        }
        wrong_mem[name] = {
            "predicted_as_corrupted_observed_label": memorized,
            "total_noisy_examples": nn_cnt,
            "memorization_rate_percent": (100.0 * memorized / nn_cnt) if nn_cnt > 0 else 0.0,
        }
    return clean_acc, wrong_mem


def offline_truth_and_estimator_audit(
    output_dir: Path,
    dataset: ObservedTrainDataset,
    regime_dir: Path,
    frozen_predictions: List[Dict[str, object]],
    em_snapshots: List[Dict[str, object]],
    truth_gate: RegimeTruthGate,
) -> Dict[str, object]:
    """Audit frozen TRAIN predictions and online EM trajectory against sealed truth."""
    truth_dir = regime_dir / "truth"
    with truth_gate.offline_only():
        t_manifest = read_json(truth_dir / "truth_manifest.json")
        clean_path = truth_dir / "clean_labels.npy"
        noisy_path = truth_dir / "is_noisy.npy"
        trans_path = truth_dir / "true_transition.npy"
        checked_file(clean_path, str(t_manifest["clean_labels_sha256"]))
        checked_file(noisy_path, str(t_manifest["is_noisy_sha256"]))
        checked_file(trans_path, str(t_manifest["true_transition_sha256"]))
        clean = np.load(clean_path, allow_pickle=False)
        noisy_mask = np.load(noisy_path, allow_pickle=False)
        true_t = np.load(trans_path, allow_pickle=False)

    clean_counts = np.bincount(clean, minlength=K)
    if (
        clean.shape != (N,)
        or noisy_mask.shape != (N,)
        or not np.array_equal(noisy_mask, clean != dataset.observed)
        or clean_counts.tolist() != t_manifest["clean_class_counts"]
    ):
        raise RuntimeError("Truth array integrity mismatch")

    pi_true = clean_counts.astype(np.float64) / float(N)
    groups = {"Overall": np.arange(K), **group_class_ids(clean_counts)}

    student_rows = []
    ema_rows = []
    diag_dir = output_dir / "train_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    for entry in frozen_predictions:
        pred_path = Path(entry["prediction_path"])
        checked_file(pred_path, str(entry["prediction_sha256"]))
        with np.load(pred_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["original_train_index"], dataset.indices):
                raise RuntimeError("Frozen prediction indices mismatch")
            s_pred = saved["student_predicted_class"]
            e_pred = saved["ema_predicted_class"]
        s_clean, s_mem = compute_subset_diagnostics(
            s_pred, clean, noisy_mask, dataset.observed, groups
        )
        e_clean, e_mem = compute_subset_diagnostics(
            e_pred, clean, noisy_mask, dataset.observed, groups
        )
        row = {
            "epoch": entry["epoch"],
            "samples": N,
            "model_state_unchanged": True,
            "model_state_sha256": entry["model_state_sha256"],
            "ema_state_sha256": entry["ema_state_sha256"],
            "prediction_sha256": entry["prediction_sha256"],
            "clean_subset_accuracy": s_clean,
            "noisy_subset_wrong_label_memorization_rate": s_mem,
            "ema_clean_subset_accuracy": e_clean,
            "ema_noisy_subset_wrong_label_memorization_rate": e_mem,
        }
        json_dump(diag_dir / f"epoch_{entry['epoch']:03d}.json", row)
        student_rows.append(row)
        ema_rows.append(
            {
                "epoch": entry["epoch"],
                "clean_subset_accuracy": e_clean,
                "noisy_subset_wrong_label_memorization_rate": e_mem,
            }
        )
    write_train_diagnostics_csv(output_dir / "train_diagnostics.csv", student_rows)

    em_audit_rows = []
    pi_eps = 1e-12
    for snap in em_snapshots:
        est_dir = Path(snap["snapshot_path"])
        with np.load(est_dir, allow_pickle=False) as arr:
            pi_hat = arr["blended_prior"].astype(np.float64)
            t_hat = arr["blended_transition"].astype(np.float64)
            raw_pi = arr["raw_em_prior"].astype(np.float64)
            raw_t = arr["raw_em_transition"].astype(np.float64)
            calibrated_pi = arr["calibrated_prior"].astype(np.float64)
        pi_kl = float(np.sum(pi_true * np.log((pi_true + pi_eps) / (pi_hat + pi_eps))))
        raw_kl = float(np.sum(pi_true * np.log((pi_true + pi_eps) / (raw_pi + pi_eps))))
        calibrated_kl = float(
            np.sum(pi_true * np.log((pi_true + pi_eps) / (calibrated_pi + pi_eps)))
        )
        em_audit_rows.append(
            {
                "epoch": snap["epoch"],
                "realized_noise_rate": float(t_manifest["realized_noise_rate"]),
                "raw_estimated_noise_rate": float(1.0 - np.dot(raw_pi, np.diag(raw_t))),
                "calibrated_estimated_noise_rate": float(
                    1.0 - np.dot(calibrated_pi, np.diag(raw_t))
                ),
                "blended_estimated_noise_rate": float(1.0 - np.dot(pi_hat, np.diag(t_hat))),
                "blended_T_mae": float(np.mean(np.abs(t_hat - true_t))),
                "blended_T_diag_mae": float(np.mean(np.abs(np.diag(t_hat) - np.diag(true_t)))),
                "blended_prior_mae": float(np.mean(np.abs(pi_hat - pi_true))),
                "blended_prior_kl": pi_kl,
                "raw_prior_kl": raw_kl,
                "raw_prior_mae": float(np.mean(np.abs(raw_pi - pi_true))),
                "raw_prior_min": float(raw_pi.min()),
                "calibrated_prior_mae": float(np.mean(np.abs(calibrated_pi - pi_true))),
                "calibrated_prior_kl": calibrated_kl,
                "calibrated_prior_min": float(calibrated_pi.min()),
                "blended_prior_min": float(pi_hat.min()),
            }
        )
    json_dump(output_dir / "online_em_truth_audit.json", {"rows": em_audit_rows})
    return {
        "realized_noise_rate": float(t_manifest["realized_noise_rate"]),
        "student_rows": student_rows,
        "ema_rows": ema_rows,
        "em_audit_rows": em_audit_rows,
        "truth_opens": truth_gate.opens,
    }


def open_official_test_once(data_root: Path):
    filename, expected_md5 = torchvision.datasets.CIFAR100.test_list[0]
    if filename != "test":
        raise RuntimeError("Unexpected CIFAR-100 TEST filename")
    payload = (data_root / "cifar-100-python" / filename).read_bytes()
    if hashlib.md5(payload).hexdigest() != expected_md5:
        raise RuntimeError("Official CIFAR-100 TEST MD5 mismatch")
    data = pickle.loads(payload, encoding="bytes")
    pixels = np.asarray(data.get(b"data", data.get("data")))
    targets = np.asarray(data.get(b"fine_labels", data.get("fine_labels")), dtype=np.int64)
    if (
        pixels.shape != (10_000, 3072)
        or pixels.dtype != np.uint8
        or targets.shape != (10_000,)
        or np.any((targets < 0) | (targets >= K))
        or not np.all(np.bincount(targets, minlength=K) == 100)
    ):
        raise RuntimeError("Unexpected CIFAR-100 TEST contents")
    pixels = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return pixels, targets, hashlib.sha256(payload).hexdigest()


def evaluate_final_test_once(
    output_dir: Path,
    eval_transform,
    data_root: Path,
    final_checkpoint: Dict[str, object],
    device: torch.device,
    test_gate: OfficialTestAudit,
) -> Dict[str, object]:
    if test_gate.opens or test_gate.denied:
        raise RuntimeError("Official TEST gate already has an access event")
    ckpt = safe_torch_load(Path(final_checkpoint["path"]), torch.device("cpu"))
    student = BCLCIFARModel().to(device)
    student.load_state_dict(ckpt["model_state"], strict=True)
    teacher = BCLCIFARModel().to(device)
    teacher.load_state_dict(ckpt["ema_model_state"], strict=True)

    with test_gate.final_evaluation():
        pixels, targets, test_sha = open_official_test_once(data_root)
        test_ds = TrainDiagnosticDataset(pixels, np.arange(len(targets)), eval_transform)
        student_pred = predict_classes(student, test_ds, device)
        teacher_pred = predict_classes(teacher, test_ds, device)

    if test_gate.opens != 1 or test_gate.denied:
        raise RuntimeError(f"Expected strictly 1 TEST open, observed {test_gate.opens}")

    groups = group_class_ids(cifar100lt_class_counts())

    def summarize(pred: np.ndarray) -> Dict[str, object]:
        correct = np.bincount(targets[pred == targets], minlength=K)
        metrics = {
            "Overall": 100.0 * int(correct.sum()) / len(targets),
            **{name: float(correct[ids].mean()) for name, ids in groups.items()},
        }
        return {"metrics": metrics, "per_class_correct": correct.tolist()}

    result = {
        "checkpoint_sha256": final_checkpoint["sha256"],
        "model_state_sha256": final_checkpoint["model_state_sha256"],
        "ema_state_sha256": final_checkpoint["ema_state_sha256"],
        "official_test_opens": test_gate.opens,
        "test_file_sha256": test_sha,
        "student": summarize(student_pred),
        "ema_teacher": summarize(teacher_pred),
    }
    json_dump(output_dir / "final_test_metrics.json", result)
    print(f"[FINAL TEST] student={result['student']['metrics']}", flush=True)
    print(f"[FINAL TEST] ema_teacher={result['ema_teacher']['metrics']}", flush=True)
    return result


def write_v2_report(
    output_dir: Path,
    args: argparse.Namespace,
    history_rows: List[Dict[str, object]],
    diag_summary: Dict[str, object],
    test_summary: Optional[Dict[str, object]],
) -> None:
    groups = ("Overall", "Many", "Medium", "Few")
    lines = [
        f"# NTC-BCL v2.1 G3 Report — {args.regime}",
        "",
        f"- Epochs completed: `{args.epochs}` (warmup `{args.warmup_epochs}`, semantic ramp `{args.semantic_ramp_epochs}`, EM interval `{args.estimate_interval}`)",
        f"- Unknown transition & clean prior: no true noise rate or symmetry used during training",
        f"- Prototype input mode: `{args.prototype_input_mode}` | support floor ratio: `{args.support_floor_ratio}` | EMA decay: `{args.ema_decay}`",
        f"- Module B: PiW(raw soft support) + constrained prior calibration (lambda_h={args.prior_calibration_lambda}, pi_min={args.prior_min_count}/N) + transition blend={args.transition_blend}",
        "- No DEATG, no temporal probability smoothing, and no epoch-based transition freeze; fit statistics are diagnostics only.",
        "",
    ]
    fit_rows = [
        (row["epoch"], row["em_update"])
        for row in history_rows
        if row.get("em_update") is not None
    ]
    if fit_rows:
        lines.extend([
            "## Zero-Intervention Observed-Label Fit Diagnostics",
            "",
            "| Epoch | F_top20 | F_top50 | F_all_macro |",
            "| ---: | ---: | ---: | ---: |",
        ])
        for epoch, stats in fit_rows:
            lines.append(
                f"| {epoch} | {stats['f_top20']:.6f} | {stats['f_top50']:.6f} | {stats['f_all_macro']:.6f} |"
            )
        lines.append("")

    if test_summary is not None:
        lines.extend(["## Official TEST Metrics", "", "| Evaluator | Overall | Many | Medium | Few |", "| --- | ---: | ---: | ---: | ---: |"])
        for tag in ("student", "ema_teacher"):
            m = test_summary[tag]["metrics"]
            lines.append(f"| {tag} | " + " | ".join(f"{m[g]:.2f}%" for g in groups) + " |")
        lines.append("")

    lines.extend(["## TRAIN Clean Accuracy & Memorization", "", "| Epoch | Student Clean (O/Ma/Me/Fe) | Student Mem% | EMA Clean (O/Ma/Me/Fe) | EMA Mem% |", "| ---: | --- | ---: | --- | ---: |"])
    for row in diag_summary["student_rows"]:
        sc = " / ".join(f"{row['clean_subset_accuracy'][g]['accuracy_percent']:.2f}" for g in groups)
        ec = " / ".join(f"{row['ema_clean_subset_accuracy'][g]['accuracy_percent']:.2f}" for g in groups)
        sm = row["noisy_subset_wrong_label_memorization_rate"]["Overall"]["memorization_rate_percent"]
        em = row["ema_noisy_subset_wrong_label_memorization_rate"]["Overall"]["memorization_rate_percent"]
        lines.append(f"| {row['epoch']} | {sc} | {sm:.2f}% | {ec} | {em:.2f}% |")

    if diag_summary["em_audit_rows"]:
        last_em = diag_summary["em_audit_rows"][-1]
        lines.extend([
            "",
            "## Online Anchored EM Evolution (Offline Truth Audit)",
            "",
            f"- Realized dataset noise: `{diag_summary['realized_noise_rate']:.4f}`",
            f"- Final blended estimated noise (Epoch {last_em['epoch']}): `{last_em['blended_estimated_noise_rate']:.4f}` (raw `{last_em['raw_estimated_noise_rate']:.4f}`)",
            f"- Final blended `T` MAE: `{last_em['blended_T_mae']:.6f}` | `T` diag MAE: `{last_em['blended_T_diag_mae']:.4f}`",
            f"- Prior MAE (raw / calibrated / blended): `{last_em['raw_prior_mae']:.5f}` / `{last_em['calibrated_prior_mae']:.5f}` / `{last_em['blended_prior_mae']:.5f}`",
            f"- Prior KL (raw / calibrated / blended): `{last_em['raw_prior_kl']:.5f}` / `{last_em['calibrated_prior_kl']:.5f}` / `{last_em['blended_prior_kl']:.5f}` | Min blended prior: `{last_em['blended_prior_min']:.2e}`",
            "",
        ])

    last = history_rows[-1]
    proto = last["prototype_health"]
    lines.extend([
        "## Prototype & Contrastive Health (Final Epoch)",
        "",
        f"- Classifier `||W_c||` min / median / max: `{proto['classifier_weight_norm_min']:.4f}` / `{proto['classifier_weight_norm_median']:.4f}` / `{proto['classifier_weight_norm_max']:.4f}` (small weights <0.15: `{proto['small_weight_count']}`)",
        f"- Prototype off-diag cosine (dir input) mean / max: `{proto['dir_mean_offdiag_cosine']:.4f}` / `{proto['dir_max_offdiag_cosine']:.4f}`",
        f"- Prototype off-diag cosine (raw input) mean / max: `{proto['raw_mean_offdiag_cosine']:.4f}` / `{proto['raw_max_offdiag_cosine']:.4f}`",
        f"- Soft-BCL q entropy mean: `{last['q_entropy_mean']:.4f}` | min denominator weight: `{last['denominator_weight_min']:.4f}` | anchor positive mass mean: `{last['anchor_positive_mass_mean']:.4f}`",
    ])
    (output_dir / "NTC_BCL_V21_G3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_source_provenance() -> Dict[str, object]:
    """Capture git branch/HEAD, dirty paths, and SHA256 hashes of v2 source files."""
    tracked_sources = (
        "BCL_LTN/ntc_bcl_v2_losses.py",
        "BCL_LTN/train_ntc_bcl_v2.py",
        "BCL_LTN/noise_transition_estimator.py",
        "BCL_LTN/ntc_bcl_v21_calibration.py",
    )
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_dirty_raw = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked_sources], cwd=ROOT, text=True
    )
    dirty_lines = [line for line in git_dirty_raw.splitlines() if line.strip()]
    source_sha256 = {
        rel: file_sha256(ROOT / rel) for rel in tracked_sources if (ROOT / rel).is_file()
    }
    return {
        "git_branch": git_branch,
        "git_head": git_head,
        "v2_source_dirty": bool(dirty_lines),
        "v2_source_status": dirty_lines,
        "v2_source_sha256": source_sha256,
    }


def train_v2(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    device = torch.device(args.device)
    configure_fp32_determinism(device)

    regime_dir = args.stage1_data_root / args.regime
    truth_gate = RegimeTruthGate(regime_dir)
    truth_gate.install()
    test_gate = OfficialTestAudit(args.data_root)

    source_prov = collect_source_provenance()
    dataset, observed_counts, regime_manifest = load_regime_observed_inputs(
        regime_dir, args.data_root, args.indices_file
    )
    init_state, init_meta = verified_seed42_initialization(args.base_root)
    provenance = {
        **source_prov,
        "regime": args.regime,
        "regime_manifest_observed_sha256": regime_manifest["observed_labels_sha256"],
        "init": init_meta,
        "hyperparams": {
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "semantic_ramp_epochs": args.semantic_ramp_epochs,
            "estimate_interval": args.estimate_interval,
            "ema_decay": args.ema_decay,
            "confidence_power": args.confidence_power,
            "shrinkage_mass": args.shrinkage_mass,
            "prior_pseudocount": args.prior_pseudocount,
            "transition_blend": args.transition_blend,
            "module_b": "PiW(raw_S)+KL-L2_prior_calibration+safe_floor",
            "prior_calibration_lambda_h": args.prior_calibration_lambda,
            "prior_min_count": args.prior_min_count,
            "prior_min_probability": float(args.prior_min_count) / float(N),
            "prior_calibration_max_iter": args.prior_calibration_max_iter,
            "prior_calibration_tolerance": args.prior_calibration_tolerance,
            "support_floor_ratio": args.support_floor_ratio,
            "denominator_epsilon": args.denominator_epsilon,
            "q_temperature": args.q_temperature,
            "prototype_input_mode": args.prototype_input_mode,
        },
    }

    if args.smoke_only:
        run_gpu_smoke(
            args.output,
            dataset,
            observed_counts,
            init_state,
            args,
            provenance,
            device,
            truth_gate,
            test_gate,
        )
        return

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir()
    em_dir = args.output / "em_snapshots"
    em_dir.mkdir()
    pred_dir = args.output / "train_predictions"
    pred_dir.mkdir()

    model, ema_teacher, correction, contrastive, optimizer, cur_prior, cur_transition = (
        build_training_objects(init_state, observed_counts, args, device)
    )
    calls = watch_calls(model, optimizer)
    loader, generator = make_train_loader(dataset)
    obs_tensor = torch.from_numpy(dataset.observed).long()
    sorted_indices = np.sort(dataset.indices)

    ckpt_epochs = sorted(
        {ep for ep in CHECKPOINT_EPOCHS if ep <= args.epochs} | {args.epochs}
    )
    checkpoints, frozen_predictions, em_snapshots, history_rows = [], [], [], []
    eval_ds = TrainDiagnosticDataset(
        dataset.pixels, dataset.indices, dataset.eval_transform
    )
    eval_ds_sorted = eval_ds

    total_updates = 0
    started = time.perf_counter()
    with (args.output / "history.jsonl").open("x", encoding="utf-8") as log_file:
        for epoch_index in range(args.epochs):
            epoch = epoch_index + 1
            epoch_seed, lr = seed_epoch(epoch_index, generator, optimizer)
            ramp = semantic_ramp_for_epoch(
                epoch_index, args.warmup_epochs, args.semantic_ramp_epochs
            )
            prev_calls = calls.copy()
            loss_sums = np.zeros(3, dtype=np.float64)
            q_entropy_sum = 0.0
            pos_mass_sum = 0.0
            denom_min = float("inf")
            seen = correct = 0
            all_indices = []
            label_counts = np.zeros(K, dtype=np.int64)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            ep_t0 = time.perf_counter()

            for views, observed, indices in loader:
                all_indices.append(indices.numpy().copy())
                label_counts += np.bincount(observed.numpy(), minlength=K)
                step = sgd_step_v2(
                    model,
                    ema_teacher,
                    views,
                    observed,
                    correction,
                    contrastive,
                    optimizer,
                    device,
                    semantic_ramp=ramp,
                    q_temperature=args.q_temperature,
                    prototype_input_mode=args.prototype_input_mode,
                )
                bs = len(observed)
                loss_sums += np.asarray([step["loss_lc"], step["loss_bcl"], step["loss_total"]]) * bs
                q_entropy_sum += step["q_entropy_mean"] * bs
                pos_mass_sum += step["anchor_positive_mass_mean"] * bs
                denom_min = min(denom_min, step["denominator_weight_min"])
                correct += step["observed_correct"]
                seen += bs
                total_updates += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ep_elapsed = time.perf_counter() - ep_t0

            order = np.concatenate(all_indices)
            actual_calls = {k_name: v - prev_calls[k_name] for k_name, v in calls.items()}
            expected_calls = {
                "backbone": STEPS_PER_EPOCH,
                "sample_mlp": STEPS_PER_EPOCH,
                "prototype_mlp": 0 if ramp == 0.0 else STEPS_PER_EPOCH,
                "optimizer": STEPS_PER_EPOCH,
            }
            if (
                seen != N
                or not np.array_equal(np.sort(order), sorted_indices)
                or not np.array_equal(label_counts, observed_counts)
                or actual_calls != expected_calls
            ):
                raise RuntimeError(f"Epoch {epoch} coverage or call mismatch: {actual_calls}")

            proto_health = measure_prototype_and_classifier_health(
                model.classifier.weight, model.prototype_mlp
            )

            em_meta = None
            should_update_em = (
                epoch >= args.warmup_epochs
                and epoch < args.epochs
                and ((epoch - args.warmup_epochs) % args.estimate_interval == 0)
            )
            if should_update_em:
                teacher_probs, _ = evaluate_deterministic_train_probabilities(
                    ema_teacher.model, dataset, device
                )
                # Zero-intervention observation diagnostics: record only; never
                # feed these values into the estimator, optimizer, or update gate.
                fit_stats = observed_fit_statistics(teacher_probs, obs_tensor)
                reference_prior = cur_prior.detach().clone()
                est = estimate_anchored_transition_and_prior(
                    teacher_probs,
                    obs_tensor,
                    confidence_power=args.confidence_power,
                    shrinkage_mass=args.shrinkage_mass,
                    prior_pseudocount=args.prior_pseudocount,
                    reference_prior=reference_prior,
                )
                calibrated_prior, calibration_diag = calibrate_clean_prior(
                    torch.as_tensor(observed_counts, dtype=torch.float64),
                    est.transition,
                    est.clean_prior,
                    prior_min=float(args.prior_min_count) / float(N),
                    lambda_h=args.prior_calibration_lambda,
                    max_iter=args.prior_calibration_max_iter,
                    tolerance=args.prior_calibration_tolerance,
                )
                new_prior, new_transition = blend_transition_and_prior(
                    cur_prior,
                    cur_transition,
                    calibrated_prior,
                    est.transition,
                    blend_alpha=args.transition_blend,
                )
                prior_delta = float((new_prior - cur_prior).abs().max().item())
                trans_delta = float((new_transition - cur_transition).abs().max().item())
                cur_prior, cur_transition = new_prior, new_transition
                correction.update_transition_and_prior(cur_prior, cur_transition)

                snap_path = em_dir / f"epoch_{epoch:03d}.npz"
                with snap_path.open("xb") as sf:
                    np.savez_compressed(
                        sf,
                        reference_prior=reference_prior.cpu().numpy(),
                        raw_em_prior=est.clean_prior.cpu().numpy(),
                        raw_em_transition=est.transition.cpu().numpy(),
                        calibrated_prior=calibrated_prior.cpu().numpy(),
                        blended_prior=cur_prior.cpu().numpy(),
                        blended_transition=cur_transition.cpu().numpy(),
                        anchor_support=est.anchor_support.cpu().numpy(),
                        f_top20=np.asarray(fit_stats["f_top20"], dtype=np.float64),
                        f_top50=np.asarray(fit_stats["f_top50"], dtype=np.float64),
                        f_all_macro=np.asarray(fit_stats["f_all_macro"], dtype=np.float64),
                    )
                em_meta = {
                    "epoch": epoch,
                    "raw_estimated_noise_rate": est.estimated_noise_rate,
                    "calibrated_estimated_noise_rate": float(
                        1.0 - (calibrated_prior * est.transition.diag()).sum().item()
                    ),
                    "blended_estimated_noise_rate": float(
                        1.0 - (cur_prior * cur_transition.diag()).sum().item()
                    ),
                    "global_diagonal_retention": float(est.global_diagonal_retention.item()),
                    "em_iterations": est.iterations,
                    "em_converged": est.converged,
                    "em_final_delta": est.final_delta,
                    "anchor_support_min": float(est.anchor_support.min().item()),
                    "anchor_support_median": float(est.anchor_support.median().item()),
                    "anchor_support_max": float(est.anchor_support.max().item()),
                    "prior_calibration": calibration_diag,
                    **fit_stats,
                    "blended_prior_min": float(cur_prior.min().item()),
                    "blended_prior_max": float(cur_prior.max().item()),
                    "prior_update_delta": prior_delta,
                    "transition_update_delta": trans_delta,
                    "snapshot_path": str(snap_path),
                }
                em_snapshots.append(em_meta)
                print(
                    f"[EM @ Epoch {epoch:03d}] raw_noise={em_meta['raw_estimated_noise_rate']:.4f} "
                    f"blended_noise={em_meta['blended_estimated_noise_rate']:.4f} "
                    f"p_min={em_meta['blended_prior_min']:.2e} conv={est.converged}",
                    flush=True,
                )

            row: Dict[str, object] = {
                "epoch": epoch,
                "epoch_index": epoch_index,
                "epoch_seed": epoch_seed,
                "lr": lr,
                "semantic_ramp": ramp,
                "loss_lc": float(loss_sums[0] / seen),
                "loss_bcl": float(loss_sums[1] / seen),
                "loss_total": float(loss_sums[2] / seen),
                "observed_accuracy_percent": 100.0 * correct / seen,
                "q_entropy_mean": float(q_entropy_sum / seen),
                "anchor_positive_mass_mean": float(pos_mass_sum / seen),
                "denominator_weight_min": float(denom_min),
                "prototype_health": {
                    "classifier_weight_norm_min": proto_health["classifier_weight_norm_min"],
                    "classifier_weight_norm_median": proto_health["classifier_weight_norm_median"],
                    "classifier_weight_norm_max": proto_health["classifier_weight_norm_max"],
                    "small_weight_count": len(proto_health["small_weight_class_ids"]),
                    "raw_mean_offdiag_cosine": proto_health["raw_mean_offdiag_cosine"],
                    "raw_max_offdiag_cosine": proto_health["raw_max_offdiag_cosine"],
                    "dir_mean_offdiag_cosine": proto_health["dir_mean_offdiag_cosine"],
                    "dir_max_offdiag_cosine": proto_health["dir_max_offdiag_cosine"],
                },
                "em_update": em_meta,
                "epoch_seconds": ep_elapsed,
                "sample_order_sha256": array_hash(order),
            }
            if epoch in ckpt_epochs:
                ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
                ckpt_meta = save_checkpoint(
                    ckpt_path,
                    model,
                    ema_teacher,
                    optimizer,
                    generator,
                    cur_prior,
                    cur_transition,
                    epoch,
                    total_updates,
                    provenance,
                )
                checkpoints.append(ckpt_meta)
                s_pred = predict_classes(model, eval_ds_sorted, device)
                e_pred = predict_classes(ema_teacher.model, eval_ds_sorted, device)
                pred_path = pred_dir / f"epoch_{epoch:03d}.npz"
                with pred_path.open("xb") as pf:
                    np.savez_compressed(
                        pf,
                        original_train_index=dataset.indices,
                        student_predicted_class=s_pred,
                        ema_predicted_class=e_pred,
                    )
                frozen_predictions.append(
                    {
                        **ckpt_meta,
                        "prediction_path": str(pred_path),
                        "prediction_sha256": file_sha256(pred_path),
                    }
                )
                row["checkpoint"] = ckpt_meta

            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_file.flush()
            history_rows.append(row)
            if epoch == 1 or epoch % 10 == 0 or epoch in ckpt_epochs:
                print(
                    f"[Epoch {epoch:03d}/{args.epochs:03d}] lr={lr:.4f} ramp={ramp:.2f} "
                    f"LC={row['loss_lc']:.4f} BCL={row['loss_bcl']:.4f} "
                    f"dir_cos={proto_health['dir_mean_offdiag_cosine']:.4f} "
                    f"W_min={proto_health['classifier_weight_norm_min']:.3f} ({ep_elapsed:.1f}s)",
                    flush=True,
                )

    total_seconds = time.perf_counter() - started
    if truth_gate.opens or truth_gate.denied or test_gate.opens or test_gate.denied:
        raise RuntimeError("Isolation gate violated during training loop")

    json_dump(args.output / "train_predictions_frozen.json", {"predictions": frozen_predictions})
    diag_summary = offline_truth_and_estimator_audit(
        args.output, dataset, regime_dir, frozen_predictions, em_snapshots, truth_gate
    )
    test_summary = None
    if not args.skip_test:
        test_summary = evaluate_final_test_once(
            args.output,
            dataset.eval_transform,
            args.data_root,
            checkpoints[-1],
            device,
            test_gate,
        )
    write_v2_report(args.output, args, history_rows, diag_summary, test_summary)
    json_dump(
        args.output / "run_summary.json",
        {
            "regime": args.regime,
            "epochs_completed": args.epochs,
            "total_optimizer_updates": total_updates,
            "total_seconds": total_seconds,
            "provenance": provenance,
            "checkpoints": checkpoints,
            "official_test_opens": test_gate.opens,
            "truth_opens": truth_gate.opens,
            "test_summary": test_summary,
        },
    )


def finalize_saved_run(args: argparse.Namespace) -> None:
    """Evaluate official TEST once for an already completed training directory."""
    if not args.output.is_dir():
        raise FileNotFoundError(args.output)
    if (args.output / "final_test_metrics.json").exists():
        raise FileExistsError("final_test_metrics.json already exists; refusing re-evaluation")
    history_path = args.output / "history.jsonl"
    frozen_path = args.output / "train_predictions_frozen.json"
    em_audit_path = args.output / "online_em_truth_audit.json"
    for req in (history_path, frozen_path, em_audit_path, args.output / "train_diagnostics.csv"):
        if not req.is_file():
            raise FileNotFoundError(req)

    device = torch.device(args.device)
    configure_fp32_determinism(device)
    test_gate = OfficialTestAudit(args.data_root)

    history_rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(history_rows) != args.epochs:
        raise RuntimeError(f"Expected {args.epochs} history rows, got {len(history_rows)}")
    frozen_entries = read_json(frozen_path)["predictions"]
    final_ckpt = frozen_entries[-1]
    if int(final_ckpt["epoch"]) != args.epochs:
        raise RuntimeError("Final frozen checkpoint epoch mismatch")
    checked_file(Path(final_ckpt["path"]), str(final_ckpt["sha256"]))

    student_rows = [
        read_json(args.output / "train_diagnostics" / f"epoch_{int(entry['epoch']):03d}.json")
        for entry in frozen_entries
    ]
    em_audit_rows = read_json(em_audit_path)["rows"]
    diag_summary = {
        "realized_noise_rate": (
            em_audit_rows[-1]["realized_noise_rate"] if em_audit_rows else float("nan")
        ),
        "student_rows": student_rows,
        "em_audit_rows": em_audit_rows,
    }
    test_summary = evaluate_final_test_once(
        args.output,
        build_cifar_transforms()[2],
        args.data_root,
        final_ckpt,
        device,
        test_gate,
    )
    write_v2_report(args.output, args, history_rows, diag_summary, test_summary)
    total_seconds = float(sum(float(r["epoch_seconds"]) for r in history_rows))
    json_dump(
        args.output / "run_summary.json",
        {
            "regime": args.regime,
            "epochs_completed": args.epochs,
            "total_optimizer_updates": args.epochs * STEPS_PER_EPOCH,
            "total_seconds": total_seconds,
            "provenance": safe_torch_load(Path(final_ckpt["path"]), torch.device("cpu")).get("provenance"),
            "checkpoints": [
                {
                    "epoch": e["epoch"],
                    "path": e["path"],
                    "sha256": e["sha256"],
                    "model_state_sha256": e["model_state_sha256"],
                    "ema_state_sha256": e["ema_state_sha256"],
                }
                for e in frozen_entries
            ],
            "official_test_opens": test_gate.opens,
            "truth_opens": 0,
            "test_summary": test_summary,
            "finalized_from_saved_checkpoint": True,
        },
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NTC-BCL v2 single-pass self-evolving trainer")
    parser.add_argument("--regime", choices=REGIMES, default="sym04")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage1-data-root", type=Path, default=DEFAULT_STAGE1_DATA_ROOT)
    parser.add_argument("--indices-file", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--finalize-saved", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--semantic-ramp-epochs", type=int, default=20)
    parser.add_argument("--estimate-interval", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--confidence-power", type=float, default=2.0)
    parser.add_argument("--shrinkage-mass", type=float, default=10.0)
    parser.add_argument("--prior-pseudocount", type=float, default=1.0)
    parser.add_argument("--transition-blend", type=float, default=0.35)
    parser.add_argument(
        "--lambda-h", "--prior-calibration-lambda",
        dest="prior_calibration_lambda", type=float, default=1.0,
        help="Quadratic tether weight in KL(mu_obs || T.T pi) + lambda_h ||pi-pi_EM||_2^2.",
    )
    parser.add_argument("--prior-min-count", type=float, default=3.0)
    parser.add_argument("--prior-calibration-max-iter", type=int, default=100)
    parser.add_argument("--prior-calibration-tolerance", type=float, default=1e-8)
    parser.add_argument("--support-floor-ratio", type=float, default=0.5)
    parser.add_argument("--denominator-epsilon", type=float, default=1e-6)
    parser.add_argument("--q-temperature", type=float, default=1.0)
    parser.add_argument("--prototype-input-mode", choices=("direction", "raw"), default="direction")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.finalize_saved:
        finalize_saved_run(args)
    else:
        train_v2(args)


if __name__ == "__main__":
    main()
