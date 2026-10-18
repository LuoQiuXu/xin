"""OfficeHome donor-domain pilot: audit -> smoke -> 3 seeds -> sealed final test."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from officehome_data import (
    ARMS, CSV_SHA256, CSV_URL, DOMAINS, N_CLASSES, SEEDS, OfficeHomeImages,
    audit_dataset, cell_counts, paired_changes, prediction_metrics, read_protocol,
    sha256, split_rows, validate_test_lock, write_csv, write_json,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPECTED_BRANCH = "experiment/officehome-domain-transfer-pilot-20261018"
CONFIG = {
    "seeds": list(SEEDS), "arms": list(ARMS), "model": "ResNet18",
    "pretrained": "ResNet18_Weights.IMAGENET1K_V1", "outputs": 65,
    "stage1_updates": 5001, "stage1_eval_every": 100, "image_batch_per_domain": 4,
    "image_batch_total": 16, "stage1_optimizer": "Adam", "stage1_lr": 5e-5,
    "stage1_weight_decay": 0.0, "stage1_amp": True,
    "stage1_selection": "val domain-class macro, earliest tie",
    "stage1_bn": "running statistics eval; affine remains trainable",
    "stage2_backbone_bn": "all frozen; eval-transform cached GAP features",
    "stage2_epochs": 10, "stage2_batch_size": 128, "stage2_optimizer": "SGD",
    "stage2_lr": 0.01, "stage2_momentum": 0.9, "stage2_weight_decay": 0.0005,
    "stage2_scheduler": "CosineAnnealingLR(T_max=10, eta_min=0)",
    "stage2_primary": "fixed epoch 10; no best selection", "stage2_precision": "float32",
    "mask": "batch shared uniform without replacement", "rho": 0.25,
    "channels": 512, "replaced_channels": 128, "cache_batch_size": 16,
    "loader_workers": 4, "resnet_dropout": 0.0,
    "domain_matched_fallback": "same global donor only if exact class-domain pool empty",
    "cross_domain_sampling": "uniform instances in same class outside recipient domain",
    "cross_domain_fallback": "same global donor only if other-domain class pool empty",
    "test_policy": "all 3 x 4 epoch10 checkpoints SHA-locked before test dataset construction",
    "train_eval_resize": [224, 224], "cpu_torch_threads": 4,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    return {name: sha256(HERE / name) for name in (
        "officehome_data.py", "officehome_domain_probe.py",
        "test_officehome_domain_probe.py", "README.md")}


def git_info():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Wrong working branch: {branch}")
    return {"branch": branch, "base_commit": git("rev-parse", "HEAD"),
            "initial_status": git("status", "--short")}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def rng_stream(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def stage1_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # MDLT does NOT set affine requires_grad=False at Stage I.


def freeze_stage2(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None


def bn_state(model):
    return {name: value for name, value in model.state_dict().items()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def backbone_state(model):
    return {name: value for name, value in model.state_dict().items() if not name.startswith("fc.")}


def gap_features(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    return model.avgpool(x).flatten(1)


def replace_channels(receiver, donor, channels):
    """Original whole-channel substitution with one mask shared by this real batch."""
    result = receiver.clone()
    result[:, channels] = donor[:, channels]
    return result


def conditional_donors(labels, domains, receivers, global_donors, matched_rng, cross_rng):
    dc = labels[global_donors]
    rd = domains[receivers]
    matched, cross = global_donors.copy(), global_donors.copy()
    mf, cf = np.zeros(len(receivers), bool), np.zeros(len(receivers), bool)
    for c in np.unique(dc):
        for d in range(4):
            at = np.flatnonzero((dc == c) & (rd == d))
            if not len(at):
                continue
            same = np.flatnonzero((labels == c) & (domains == d))
            other = np.flatnonzero((labels == c) & (domains != d))
            if len(same):
                matched[at] = matched_rng.choice(same, size=len(at), replace=True)
            else:
                mf[at] = True
            if len(other):
                cross[at] = cross_rng.choice(other, size=len(at), replace=True)
            else:
                cf[at] = True
    if not np.array_equal(labels[matched], dc) or not np.array_equal(labels[cross], dc):
        raise AssertionError("Donor class changed under domain intervention")
    if not np.all(domains[matched[~mf]] == rd[~mf]):
        raise AssertionError("Matched-domain constraint failed")
    if not np.all(domains[cross[~cf]] != rd[~cf]):
        raise AssertionError("Cross-domain constraint failed")
    if not np.array_equal(matched[mf], global_donors[mf]) or not np.array_equal(cross[cf], global_donors[cf]):
        raise AssertionError("Invalid fallback")
    return matched, cross, mf, cf


def make_plan(rows, seed, epochs=10):
    n = len(rows)
    labels = np.array([r.label for r in rows], dtype=np.int64)
    domains = np.array([r.domain for r in rows], dtype=np.int64)
    pools = [np.flatnonzero(labels == c) for c in range(N_CLASSES)]
    if any(len(p) == 0 for p in pools) or any(r.split != "train" for r in rows):
        raise ValueError("All 65 receiver classes must be present in train")
    rr, gr, mr, cr, kr = [rng_stream(seed, x) for x in range(201, 206)]
    fields = {k: [] for k in ("receiver", "global", "matched", "cross", "matched_fallback", "cross_fallback", "masks")}
    for epoch in range(epochs):
        counts = np.full(N_CLASSES, n // N_CLASSES)
        counts[rr.choice(N_CLASSES, size=n % N_CLASSES, replace=False)] += 1
        receiver_classes = rr.permutation(np.repeat(np.arange(N_CLASSES), counts))
        receiver = np.empty(n, dtype=np.int64)
        for c, pool in enumerate(pools):
            at = np.flatnonzero(receiver_classes == c)
            receiver[at] = rr.choice(pool, size=len(at), replace=True)
        donor = gr.integers(0, n, size=n, dtype=np.int64)
        matched, cross, mf, cf = conditional_donors(labels, domains, receiver, donor, mr, cr)
        masks = np.stack([kr.choice(512, size=128, replace=False)
                          for _ in range(math.ceil(n / 128))]).astype(np.int64)
        actual_counts = np.bincount(labels[receiver], minlength=N_CLASSES)
        if actual_counts.max() - actual_counts.min() > 1:
            raise AssertionError("Class-balanced receiver counts are not floor/ceil")
        for key, value in zip(fields, (receiver, donor, matched, cross, mf, cf, masks)):
            fields[key].append(value)
    return {key: np.stack(value) for key, value in fields.items()}


def simple_metrics(metrics):
    return {k: metrics[k] for k in ("Overall", "DomainClassMacro", "DomainClassMany",
                                    "DomainClassMedium", "DomainClassFew", "WorstCell")}


def append_csv(path, row):
    exists = Path(path).exists()
    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def loader(root, rows, split, seed, augment=False, batch_sampler=None, test_lock=None):
    ds = OfficeHomeImages(root, rows, split, augment, test_lock)
    options = dict(num_workers=4, pin_memory=True, persistent_workers=True,
                   worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed))
    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, **options)
    return DataLoader(ds, batch_size=16, shuffle=False, **options)


@torch.no_grad()
def evaluate_images(model, image_loader, rows, counts, device):
    model.eval()
    predictions = []
    for x, y, domain, index in image_loader:
        logits = model(x.to(device, non_blocking=True))
        predictions.extend(logits.argmax(1).cpu().tolist())
    return prediction_metrics(rows, predictions, counts)[0]


def build_model():
    # No random-backbone fallback. The named V1 weights have a fixed torchvision URL/hash.
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, N_CLASSES)
    return model


def stage1(root, rows, counts, seed, out, device, smoke=False):
    seed_everything(seed)
    model = build_model().to(device)
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet18_Weights.IMAGENET1K_V1.url).name
    pretraining = {"url": ResNet18_Weights.IMAGENET1K_V1.url, "path": str(weight_path),
                   "sha256": sha256(weight_path), "head_initial_state_sha256": state_hash(model.fc.state_dict())}
    write_json(out / "imagenet_initialization.json", pretraining)
    initial_bb = state_hash(backbone_state(model))
    initial_bn = state_hash(bn_state(model))
    train_rows, val_rows = split_rows(rows, "train"), split_rows(rows, "val")
    updates = 20 if smoke else 5001
    sampler_rng = rng_stream(seed, 101)
    domain_pools = [np.array([i for i, r in enumerate(train_rows) if r.domain == d]) for d in range(4)]
    indices = [np.concatenate([sampler_rng.choice(pool, 4, replace=True) for pool in domain_pools]).tolist()
               for _ in range(updates)]
    np.savez_compressed(out / "output" / "stage1_image_plan.npz", indices=np.array(indices),
                        train_csv_indices=np.array([r.index for r in train_rows]))
    tr_loader = loader(root, train_rows, "train", seed + 1001, True, indices)
    val_loader = loader(root, val_rows, "val", seed + 1002)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda")
    best_score, best_update, best_state = -math.inf, None, None
    start, train_seconds, skipped_attempts, first_gradient_audit = time.perf_counter(), 0.0, 0, None
    loss_total = 0.0
    for update, (x, y, domains, raw_indices) in enumerate(tr_loader, 1):
        t0 = time.perf_counter()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if not torch.equal(torch.bincount(domains, minlength=4), torch.full((4,), 4)):
            raise AssertionError("Stage I batch is not 4 images per domain")
        stage1_mode(model)
        for retry in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite Stage I loss at update {update}")
            scaler.scale(loss).backward()
            if first_gradient_audit is None:
                scaler.unscale_(optimizer)
                first_gradient_audit = {
                    "backbone_has_grad": model.conv1.weight.grad is not None,
                    "bn_affine_has_grad": model.bn1.weight.grad is not None,
                    "classifier_has_grad": model.fc.weight.grad is not None,
                    "all_gradients_finite": all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)}
                if not first_gradient_audit["all_gradients_finite"]:
                    first_gradient_audit = None
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                break
            skipped_attempts += 1
        else:
            raise FloatingPointError("AMP could not perform a finite optimizer update")
        loss_total += float(loss.detach())
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - t0
        if update % 100 == 0 or update == updates:
            metrics = evaluate_images(model, val_loader, val_rows, counts, device)
            if abs(metrics["Overall"] - metrics["DomainClassMacro"]) > 1e-9:
                raise AssertionError("Equal-cell validation Overall != domain-class macro")
            if state_hash(bn_state(model)) != initial_bn:
                raise AssertionError("Stage I BN running statistics changed")
            if metrics["DomainClassMacro"] > best_score + 1e-12:
                best_score, best_update = metrics["DomainClassMacro"], update
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            append_csv(out / "stage1_epochs.csv", {
                "seed": seed, "update": update, "mean_train_loss": loss_total / update,
                **simple_metrics(metrics), "selected_update_so_far": best_update,
                "elapsed_seconds": time.perf_counter() - start, "amp_skipped_forward_attempts": skipped_attempts})
            print(f"StageI seed={seed} update={update}/{updates} loss={float(loss.detach()):.5f} "
                  f"val_macro={metrics['DomainClassMacro']:.4f} best_update={best_update}", flush=True)
    final_path = out / "output" / "stage1_final.pth"
    torch.save({"model": model.state_dict(), "seed": seed, "update": updates}, final_path)
    best_path = out / "output" / "stage1_best.pth"
    torch.save({"model": best_state, "seed": seed, "update": best_update,
                "val_domain_class_macro": best_score, "imagenet": pretraining}, best_path)
    info = {"seed": seed, "updates": updates, "best_update": best_update, "best_val_macro": best_score,
            "best_checkpoint": str(best_path), "best_sha256": sha256(best_path),
            "final_sha256": sha256(final_path), "bn_running_bitwise_unchanged": state_hash(bn_state(model)) == initial_bn,
            "backbone_changed": state_hash(backbone_state(model)) != initial_bb,
            "first_gradient_audit": first_gradient_audit, "amp_skipped_forward_attempts": skipped_attempts,
            "train_compute_seconds": train_seconds, "total_seconds": time.perf_counter() - start}
    if not info["backbone_changed"] or not info["bn_running_bitwise_unchanged"]:
        raise AssertionError("Stage I parameter/BN update audit failed")
    write_json(out / "stage1_summary.json", info)
    model.load_state_dict(best_state)
    freeze_stage2(model)
    del tr_loader, val_loader, optimizer
    return model, info


@torch.no_grad()
def cache_features(root, records, split, model, seed, out, device, test_lock=None):
    image_loader = loader(root, records, split, seed + 3100, test_lock=test_lock)
    features, seen_indices = [], []
    model.eval()
    start = time.perf_counter()
    for x, y, d, raw_index in image_loader:
        f = gap_features(model, x.to(device, non_blocking=True))
        if f.ndim != 2 or f.shape[1] != 512 or not torch.isfinite(f).all():
            raise AssertionError("Invalid GAP feature cache")
        features.append(f.cpu())
        seen_indices.extend(raw_index.tolist())
    if seen_indices != [r.index for r in records]:
        raise AssertionError("Feature cache order changed")
    tensor = torch.cat(features)
    path = out / "output" / f"{split}_features.npz"
    np.savez_compressed(path, features=tensor.numpy(), raw_indices=np.array(seen_indices),
                        labels=np.array([r.label for r in records]),
                        domains=np.array([r.domain for r in records]))
    print(f"Cached seed={seed} split={split} n={len(records)} sha={sha256(path)} "
          f"seconds={time.perf_counter()-start:.1f}", flush=True)
    del image_loader
    return tensor.to(device)


def donor_diagnostics(train_rows, plan, seed, out, smoke):
    labels = np.array([r.label for r in train_rows])
    domains = np.array([r.domain for r in train_rows])
    limit = min(len(train_rows), 256) if smoke else len(train_rows)
    receiver = plan["receiver"][:, :limit]
    donor_classes = labels[plan["global"][:, :limit]]
    control_hash = array_hash(receiver, donor_classes, plan["masks"][:, :math.ceil(limit / 128)])
    diagnostics, fallbacks = [], []
    for arm, donor_key in zip(ARMS, ("global", "global", "matched", "cross")):
        donors = plan[donor_key][:, :limit]
        if not np.array_equal(labels[donors], donor_classes):
            raise AssertionError("Unpaired donor classes")
        flags = (plan["matched_fallback"][:, :limit] if donor_key == "matched" else
                 plan["cross_fallback"][:, :limit] if donor_key == "cross" else np.zeros_like(donors, bool))
        diagnostics.append({
            "seed": seed, "arm": arm, "receiver_draws": int(receiver.size),
            "fusion_draws": 0 if arm == ARMS[0] else int(receiver.size),
            "donor_plan_role": "virtual_not_used" if arm == ARMS[0] else "actual",
            "paired_controls_sha256": control_hash, "donor_classes_sha256": array_hash(labels[donors]),
            "fallback_count": int(flags.sum()), "fallback_rate": float(flags.mean()),
            "cross_domain_rate": float((domains[donors] != domains[receiver]).mean()),
            "donor_domain_counts": json.dumps(np.bincount(domains[donors].ravel(), minlength=4).tolist()),
            "donor_class_counts": json.dumps(np.bincount(labels[donors].ravel(), minlength=65).tolist()),
        })
        for e, i in np.argwhere(flags):
            j, g, t = donors[e, i], plan["global"][e, i], receiver[e, i]
            fallbacks.append({"seed": seed, "arm": arm, "epoch": int(e) + 1, "draw": int(i),
                              "receiver_csv_index": train_rows[t].index,
                              "donor_class": int(labels[j]), "receiver_domain": DOMAINS[domains[t]],
                              "global_donor_csv_index": train_rows[g].index,
                              "fallback_donor_csv_index": train_rows[j].index})
    write_csv(out / "donor_diagnostics.csv", diagnostics)
    write_csv(out / "donor_fallbacks.csv", fallbacks, [
        "seed", "arm", "epoch", "draw", "receiver_csv_index", "donor_class",
        "receiver_domain", "global_donor_csv_index", "fallback_donor_csv_index"])
    write_json(out / "sampling_plan_audit.json", {
        "train_csv_index_sha256": array_hash(np.array([r.index for r in train_rows])),
        "seed": seed, "receiver_donor_class_masks_paired": True, "control_sha256": control_hash,
        "plan_sha256": sha256(out / "output" / "sampling_plan.npz"),
        "receiver_draws_per_epoch": len(train_rows), "receiver_class_count_floor": len(train_rows) // 65,
        "receiver_class_count_ceil": math.ceil(len(train_rows) / 65),
        "mask_granularity": "one 128-of-512 mask per actual minibatch",
        "RNG": "independent numpy SeedSequence streams for receiver/global/matched/cross/mask",
        "not_original_torch_randperm_RNG_path": True,
    })
    return diagnostics


def train_classifier(model, train_features, val_features, train_rows, val_rows,
                     counts, plan, seed, arm, out, stage1_sha, smoke=False):
    seed_everything(seed + 5000)
    head = copy.deepcopy(model.fc).to(train_features.device)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    head.train()
    initial_head = state_hash(head.state_dict())
    fixed_backbone = state_hash(backbone_state(model))
    fixed_bn = state_hash(bn_state(model))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    labels = np.array([r.label for r in train_rows], dtype=np.int64)
    train_y = torch.tensor(labels, device=train_features.device)
    donor_key = {ARMS[0]: "global", ARMS[1]: "global", ARMS[2]: "matched", ARMS[3]: "cross"}[arm]
    total_steps, start = 0, time.perf_counter()
    trace = hashlib.sha256()
    metrics = None
    for e in range(1 if smoke else 10):
        loss_sum, processed = 0.0, 0
        lr = optimizer.param_groups[0]["lr"]
        for batch, offset in enumerate(range(0, len(train_rows), 128)):
            if smoke and batch == 2:
                break
            receivers = plan["receiver"][e, offset:offset + 128]
            donors = plan[donor_key][e, offset:offset + 128]
            channels = plan["masks"][e, batch]
            expected_donor_classes = labels[plan["global"][e, offset:offset + 128]]
            if not np.array_equal(labels[donors], expected_donor_classes):
                raise AssertionError("Actual training donor class changed")
            if len(np.unique(channels)) != 128 or channels.min() < 0 or channels.max() >= 512:
                raise AssertionError("Bad batch-shared mask")
            trace.update(array_hash(receivers, expected_donor_classes, channels, labels[receivers]).encode())
            ti = torch.tensor(receivers, device=train_features.device)
            td = torch.tensor(donors, device=train_features.device)
            tc = torch.tensor(channels, device=train_features.device)
            receiver_features = train_features[ti]
            x = (receiver_features if arm == ARMS[0] else
                 replace_channels(receiver_features, train_features[td], tc))
            target = train_y[ti]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite Stage II loss {seed}/{arm}/{e+1}/{batch}")
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()):
                raise FloatingPointError("Nonfinite classifier gradient")
            if any(p.grad is not None for p in model.parameters()):
                raise AssertionError("Backbone/reference model acquired gradients")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(receivers)
            processed += len(receivers)
            total_steps += 1
        scheduler.step()
        head.eval()
        with torch.no_grad():
            predictions = torch.cat([head(v).argmax(1).cpu() for v in val_features.split(128)]).numpy()
        metrics, _, _ = prediction_metrics(val_rows, predictions, counts)
        append_csv(out / "stage2_epochs.csv", {
            "seed": seed, "arm": arm, "epoch": e + 1, "smoke_only": smoke,
            "steps": total_steps, "draws_this_epoch": processed, "lr_used": lr,
            "lr_next": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / processed,
            **simple_metrics(metrics), "elapsed_seconds": time.perf_counter() - start})
        np.savez_compressed(out / "output" / f"{arm}_validation_epoch{e+1:02d}.npz",
                            predictions=predictions, labels=np.array([r.label for r in val_rows]),
                            raw_indices=np.array([r.index for r in val_rows]))
        print(f"StageII seed={seed} arm={arm} epoch={e+1} steps={total_steps} "
              f"val={metrics['Overall']:.4f} Few={metrics['DomainClassFew']:.4f}", flush=True)
        head.train()
    expected_steps = 2 if smoke else 10 * math.ceil(len(train_rows) / 128)
    if total_steps != expected_steps:
        raise AssertionError("Wrong Stage II update budget")
    if state_hash(backbone_state(model)) != fixed_backbone or state_hash(bn_state(model)) != fixed_bn:
        raise AssertionError("Stage II backbone/BN state changed")
    if state_hash(head.state_dict()) == initial_head:
        raise AssertionError("Classifier did not update")
    checkpoint = out / "output" / f"{arm}_{'smoke' if smoke else 'epoch10'}.pth"
    torch.save({"classifier": head.state_dict(), "epoch": 1 if smoke else 10, "seed": seed,
                "arm": arm, "stage1_sha256": stage1_sha, "initial_classifier_sha256": initial_head,
                "steps": total_steps, "backbone_state_sha256": fixed_backbone,
                "bn_state_sha256": fixed_bn, "training_controls_sha256": trace.hexdigest()}, checkpoint)
    result = {"seed": seed, "arm": arm, "epoch": 1 if smoke else 10, "steps": total_steps,
              "path": str(checkpoint), "sha256": sha256(checkpoint),
              "stage1_sha256": stage1_sha, "initial_classifier_sha256": initial_head,
              "training_controls_sha256": trace.hexdigest(), "backbone_bitwise_unchanged": True,
              "bn_bitwise_unchanged": True, "backbone_gradients_absent": True,
              "classifier_changed": True, "seconds": time.perf_counter() - start,
              "validation": simple_metrics(metrics)}
    del head, optimizer
    return result


def seal_final_checkpoints(out, results):
    entries = []
    for item in results:
        path = Path(item["path"]).resolve()
        entries.append({**{k: item[k] for k in ("seed", "arm", "epoch", "sha256", "stage1_sha256")},
                        "path": path.relative_to(out.resolve()).as_posix()})
    lock_path = out / "final_checkpoints.lock.json"
    write_json(lock_path, {"sealed_utc": utc_now(), "csv_sha256": CSV_SHA256, "checkpoints": entries})
    validate_test_lock(lock_path)
    return lock_path


def make_scientific_report(out, summary, differences, stage1_infos, diagnostics):
    lookup = {(r["seed"], r["arm"]): r for r in summary}
    lines = [
        "# OfficeHome-MLT H2T 供体域条件机制试验", "",
        "## 命题和协议", "",
        "本轮检验相同供体类别和 batch 共享通道掩码下，供体域条件是否改变固定特征 Stage II 的结果。",
        "四个域均进入 train/val/test；不是 unseen-domain OOD。使用 ImageNet V1 预训练 ResNet-18、65 logits。",
        "Stage I 为 5001 updates，BN running statistics 固定而 affine 可训练；仅按 val domain-class macro 选模，平局取早。",
        "Stage II 使用 eval transform 的固定 GAP 特征，整个 backbone/BN 冻结；这是机制 pilot，区别于随机图像增强的原始 H2T Stage II。",
        "128/512 完整通道按 batch 共享替换，rho=0.25；NumPy 独立 RNG 不复刻原 torch.randperm 路径。",
        "12 个 epoch10 模型先保存并锁定，之后统一打开 test；Stage II 不选择 best。", "",
        "## 固定 epoch10 最终 TEST 结果（%）", "",
        "| Seed | Arm | Overall | Domain-class macro | Many | Medium | Few |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['Overall']:.4f} | {r['DomainClassMacro']:.4f} | "
                     f"{r['DomainClassMany']:.4f} | {r['DomainClassMedium']:.4f} | {r['DomainClassFew']:.4f} |")
    lines += ["", "### 三 seed 等权均值", "",
              "| Arm | Overall | Domain-class macro | Many | Medium | Few |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        subset = [r for r in summary if r["arm"] == arm]
        values = [np.mean([r[k] for r in subset]) for k in (
            "Overall", "DomainClassMacro", "DomainClassMany", "DomainClassMedium", "DomainClassFew")]
        lines.append(f"| {arm} | " + " | ".join(f"{x:.4f}" for x in values) + " |")
    lines += ["", "## 预注册判读", ""]
    gains, worse_seeds = [], []
    for seed in SEEDS:
        dr, global_, matched, cross = [lookup[seed, arm] for arm in ARMS]
        gain = global_["DomainClassFew"] - dr["DomainClassFew"]
        df = cross["DomainClassFew"] - matched["DomainClassFew"]
        dm = cross["DomainClassMacro"] - matched["DomainClassMacro"]
        gains.append(gain)
        if df < 0 or dm < 0:
            worse_seeds.append(seed)
        lines.append(f"- seed {seed}：Global−DR Few={gain:+.4f} pp；Cross−Matched Few={df:+.4f} pp，macro={dm:+.4f} pp。")
    if all(x > 0 for x in gains):
        lines.append("- Global 在三个 seed 的 Few 上方向一致为正；样本只有三个训练重复，不能宣称统计显著或普适收益。")
    else:
        lines.append("- Global 相对 DR 的 Few 收益未在三个 seed 全部保持为正；不能声称 H2T 在该条件下已有稳定收益。")
    if len(worse_seeds) >= 2:
        lines.append(f"- {len(worse_seeds)}/3 个 seed 满足 Cross 的 Few 或 macro 低于 Matched；在本次配对条件下支持供体域不匹配可能构成风险的初步线索。仍不是新模块有效的证据。")
    else:
        lines.append("- Cross 相对 Matched 的预注册方向没有达到至少 2/3 seed；当前假设证据不足。")
    lines += ["", "## 配对控制、退回与局限", "",
              "每 seed 四组共享同一 Stage I、初始 FC、receiver、donor class、batch 顺序和掩码；actual training control hashes 完全一致。",
              "C/D 仅在相应 domain-class 池为空时退回 B 的同一供体；逐条退回见各 seed donor_fallbacks.csv。",
              "DR 的 donor 仅为配对计划中的虚拟供体，不执行融合；诊断中的域/类别分布应按该角色解释。",
              "Many>=60、Medium20..59、Few<20 按训练 domain-class cell 定义，不能与 CIFAR shot 或论文 CIFAR TEST 比较。",
              "260 个 cell、四域准确率、最差 cell 全部并列项和测试支持数见对应 CSV/JSON；不会只挑一个有利 cell。",
              "三 seed 重复使用相同测试图像，不能视作 7800 个独立训练重复；本轮无调参、无新增损失或模块。",
              "不根据即时 margin/测试结果追加实验。到此停止，后续研究需另行确认。", "",
              "## 文件与审计证据", "",
              "- config.json / environment.json / dataset_audit.json：配置、来源、环境与数据检查。",
              "- seed42/43/44/stage1_epochs.csv：全部 51 次 val 检查，stage1_summary.json：选择、BN、梯度与耗时。",
              "- seed42/43/44/stage2_epochs.csv：每组十 epoch，固定第十 epoch；output 中保留本地计划、特征、模型和预测。",
              "- final_checkpoints.lock.json / test_access.json：12 模型 SHA256 与 test 首次访问时间。",
              "- epoch10_summary.csv / epoch10_mean.csv / per_domain.csv / domain_class_metrics.csv：全部主指标。",
              "- paired_changes.csv：逐图 gain/loss/net；net 与相应准确率差由程序核对。",
              "- donor_diagnostics.csv / state_audit.json：供体、配对控制与 BB/BN 状态。",
              "- execution.log：完整标准输出及异常栈。源码 SHA256 见 config.json。", ""]
    (out / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def execute(args, out):
    git = git_info()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    import torchvision
    environment = {"utc": utc_now(), "python_executable": sys.executable, "python": sys.version,
                   "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
                   "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None,
                   "determinism": "fixed streams, cuDNN deterministic, benchmark/TF32 off; no cross-hardware bitwise claim"}
    config = {**CONFIG, "command": args.command, "argv": sys.argv, "git": git,
              "source_sha256": source_hashes(), "csv_url": CSV_URL, "csv_sha256": CSV_SHA256,
              "image_root": str(args.data_root.resolve()), "csv_path": str(args.csv.resolve()),
              "created_utc": utc_now()}
    write_json(out / "config.json", config)
    write_json(out / "environment.json", environment)
    rows = read_protocol(args.csv)
    audit = audit_dataset(rows, args.data_root, args.csv, out)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if not audit["ready"]:
        (out / "final_report.md").write_text(
            "# OfficeHome metadata audit: BLOCKED\n\n"
            f"Missing images: {audit['missing_image_count']}. See missing_paths.csv and dataset_audit.json.\n"
            "No training, test decoding, evaluation or model selection was performed.\n", encoding="utf-8")
        write_json(out / "status.json", {"status": "BLOCKED_DATA", "utc": utc_now()})
        return 2
    if args.command == "audit":
        (out / "final_report.md").write_text(
            "# OfficeHome metadata audit: PASS\n\n"
            f"All {len(rows)} official CSV paths exist. No duplicate or overlapping paths. "
            "No image was decoded. Test remains locked. See domain_class_counts.csv and dataset_audit.json.\n",
            encoding="utf-8")
        write_json(out / "status.json", {"status": "AUDIT_PASS", "utc": utc_now()})
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the authorized AMP image training; no silent CPU fallback")
    smoke = args.command == "smoke"
    if not smoke:
        smoke_dir = args.smoke_dir.resolve()
        smoke_status = json.loads((smoke_dir / "status.json").read_text(encoding="utf-8"))
        smoke_config = json.loads((smoke_dir / "config.json").read_text(encoding="utf-8"))
        if smoke_status["status"] != "SMOKE_PASS" or smoke_config["source_sha256"] != config["source_sha256"]:
            raise RuntimeError("A passing real-data smoke with the identical source is required")
        if smoke_config["image_root"] != config["image_root"] or smoke_config["csv_sha256"] != CSV_SHA256:
            raise RuntimeError("Smoke dataset provenance mismatch")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    counts = cell_counts(rows)
    train_rows, val_rows = split_rows(rows, "train"), split_rows(rows, "val")
    results, stage1_infos, all_diagnostics = [], [], []
    seeds = (42,) if smoke else SEEDS
    start = time.perf_counter()
    for seed in seeds:
        seed_out = out / f"seed{seed}"
        (seed_out / "output").mkdir(parents=True, exist_ok=False)
        model, s1_info = stage1(args.data_root, rows, counts, seed, seed_out, device, smoke)
        stage1_infos.append(s1_info)
        frozen_state = state_hash(model.state_dict())
        train_f = cache_features(args.data_root, train_rows, "train", model, seed, seed_out, device)
        val_f = cache_features(args.data_root, val_rows, "val", model, seed, seed_out, device)
        write_json(seed_out / "feature_manifest.json", {
            s: {"path": f"output/{s}_features.npz", "sha256": sha256(seed_out / "output" / f"{s}_features.npz"),
                "transform": "Resize((224,224)), RGB tensor, ImageNet normalization", "dtype": "float32"}
            for s in ("train", "val")})
        plan = make_plan(train_rows, seed, epochs=1 if smoke else 10)
        np.savez_compressed(seed_out / "output" / "sampling_plan.npz", **plan,
                            train_csv_indices=np.array([r.index for r in train_rows]))
        diagnostics = donor_diagnostics(train_rows, plan, seed, seed_out, smoke)
        all_diagnostics.extend(diagnostics)
        seed_results = [train_classifier(model, train_f, val_f, train_rows, val_rows, counts, plan,
                                         seed, arm, seed_out, s1_info["best_sha256"], smoke) for arm in ARMS]
        if len({r["training_controls_sha256"] for r in seed_results}) != 1:
            raise AssertionError("Actual training receiver/donor class/mask/label streams differ")
        if len({r["initial_classifier_sha256"] for r in seed_results}) != 1:
            raise AssertionError("Initial FC states differ")
        if state_hash(model.state_dict()) != frozen_state:
            raise AssertionError("Frozen Stage I reference changed during cached Stage II")
        write_json(seed_out / "state_audit.json", seed_results)
        results.extend(seed_results)
        del model, train_f, val_f
        torch.cuda.empty_cache()
    write_json(out / "state_audit.json", results)
    write_csv(out / "donor_diagnostics.csv", all_diagnostics)
    write_csv(out / "validation_epoch10_summary.csv", [
        {"seed": r["seed"], "arm": r["arm"], "epoch": r["epoch"], **r["validation"]} for r in results])
    memory = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    if smoke:
        estimate = {"stage1_train_seconds_20_updates": stage1_infos[0]["train_compute_seconds"],
                    "estimated_3_seed_stage1_train_seconds": stage1_infos[0]["train_compute_seconds"] / 20 * 5001 * 3,
                    "elapsed_smoke_seconds": time.perf_counter() - start,
                    "stage2_total_actual_updates": sum(r["steps"] for r in results),
                    "limitation": "20-update timing is approximate; loader startup and cache time differ", **memory}
        write_json(out / "smoke_timing.json", estimate)
        write_json(out / "status.json", {"status": "SMOKE_PASS", "test_accessed": False, "utc": utc_now()})
        print("SMOKE_PASS " + json.dumps(estimate), flush=True)
        return 0

    # HARD PHASE BOUNDARY: no test dataset/loader has been constructed above this line.
    lock_path = seal_final_checkpoints(out, results)
    validate_test_lock(lock_path)
    write_json(out / "test_access.json", {"started_utc": utc_now(), "phase": "single final evaluation",
                                        "all_final_checkpoints_locked": 12, "manifest_sha256": sha256(lock_path),
                                        "no_stage2_best_selection": True})
    print("FINAL_TEST_BEGIN " + utc_now(), flush=True)
    test_rows = split_rows(rows, "test")
    summary, domain_table, cell_table, pairs, worst = [], [], [], [], []
    for seed in SEEDS:
        seed_out = out / f"seed{seed}"
        model = build_model().to(device)
        best = torch.load(seed_out / "output" / "stage1_best.pth", map_location=device, weights_only=True)
        model.load_state_dict(best["model"])
        freeze_stage2(model)
        test_f = cache_features(args.data_root, test_rows, "test", model, seed, seed_out, device, lock_path)
        predictions = {}
        for arm in ARMS:
            ckpt = torch.load(seed_out / "output" / f"{arm}_epoch10.pth", map_location=device, weights_only=True)
            model.fc.load_state_dict(ckpt["classifier"])
            with torch.no_grad():
                pred = torch.cat([model.fc(f).argmax(1).cpu() for f in test_f.split(128)]).numpy()
            predictions[arm] = pred
            metrics, cells, domains = prediction_metrics(test_rows, pred, counts)
            summary.append({"seed": seed, "arm": arm, "epoch": 10, **simple_metrics(metrics)})
            domain_table.extend({"seed": seed, "arm": arm, **r} for r in domains)
            cell_table.extend({"seed": seed, "arm": arm, **r} for r in cells)
            worst.append({"seed": seed, "arm": arm, "accuracy": metrics["WorstCell"], "cells": metrics["WorstCells"]})
            np.savez_compressed(seed_out / "output" / f"{arm}_final_test_predictions.npz",
                                predictions=pred, labels=np.array([r.label for r in test_rows]),
                                raw_indices=np.array([r.index for r in test_rows]))
            print(f"FINAL_TEST seed={seed} arm={arm} " + json.dumps(simple_metrics(metrics)), flush=True)
        comparisons = [(a, b) for b in ARMS[:2] for a in ARMS if a != b] + [(ARMS[3], ARMS[2])]
        for candidate, baseline in comparisons:
            pairs.extend({"seed": seed, "candidate": candidate, "baseline": baseline, **r}
                         for r in paired_changes(test_rows, predictions[candidate], predictions[baseline], counts))
        del model, test_f
    write_csv(out / "epoch10_summary.csv", summary)
    write_csv(out / "epoch10_mean.csv", [
        {"arm": a, "n_seeds": 3, **{k: float(np.mean([r[k] for r in summary if r["arm"] == a]))
                                  for k in simple_metrics(summary[0])}} for a in ARMS])
    write_csv(out / "per_domain.csv", domain_table)
    write_csv(out / "domain_class_metrics.csv", cell_table)
    write_csv(out / "paired_changes.csv", pairs)
    write_json(out / "worst_cells.json", worst)
    make_scientific_report(out, summary, pairs, stage1_infos, all_diagnostics)
    memory = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    write_json(out / "status.json", {"status": "COMPLETE", "utc": utc_now(),
                                    "elapsed_seconds": time.perf_counter() - start, **memory})
    print("COMPLETE " + json.dumps(memory), flush=True)
    return 0


class Tee:
    def __init__(self, original, handle):
        self.original, self.handle = original, handle
    def write(self, value):
        self.original.write(value)
        self.handle.write(value)
        self.handle.flush()
    def flush(self):
        self.original.flush()
        self.handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "smoke", "run"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=HERE / "protocol" / "OfficeHome.csv")
    parser.add_argument("--out", type=Path, required=True, help="New unique run directory; never overwritten")
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.smoke_dir is None:
        parser.error("run requires --smoke-dir from a successful identical-source real-data smoke")
    out = args.out.resolve()
    if not out.is_relative_to(HERE):
        parser.error("All pilot outputs must remain under this new experiment directory")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "execution.log").open("w", encoding="utf-8", buffering=1) as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            print("START " + utc_now() + " " + repr(sys.argv), flush=True)
            code = execute(args, out)
        except Exception as exc:
            traceback.print_exc()
            write_json(out / "status.json", {"status": "FAILED", "utc": utc_now(),
                                            "error": repr(exc), "traceback": traceback.format_exc()})
            code = 1
        finally:
            sys.stdout, sys.stderr = original_out, original_err
    return code


if __name__ == "__main__":
    raise SystemExit(main())
