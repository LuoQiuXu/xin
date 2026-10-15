"""Observe the actual official H2T train/eval entry points on real CIFAR-100-LT.

The train commands are deliberately limited to one epoch. Hooks inspect behavior
without changing sampling, random streams, gradients, loss or optimizers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch
import torchvision

BASE = Path(__file__).resolve().parent
RUNTIME = BASE / "runtime"
EVIDENCE = BASE / "evidence"
DATA = BASE.parent / "H2T/data/cifar100"
sys.path.insert(0, str(RUNTIME))


def dump(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False),
                                 encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(module, buffers=False):
    pairs = module.named_buffers() if buffers else module.named_parameters()
    return {name: value.detach().cpu().clone() for name, value in pairs}


def delta(before, after):
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    maximum = max((float((before[k].double() - after[k].double()).abs().max())
                   for k in changed), default=0.0)
    return {"changed_count": len(changed), "total_count": len(before),
            "max_abs_delta": maximum, "changed_names": changed}


def base_config(stage):
    from utils import config
    cfg = config.clone()
    cfg.merge_from_file(str(RUNTIME / (
        "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml" if stage == "stage1"
        else "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml")))
    cfg.data_path = str(DATA)
    return cfg


def preflight():
    from datasets.cifar100 import CIFAR100_LT
    from torchvision.datasets.utils import check_integrity
    from models import resnet_cifar
    from loss import CrossEntropyLoss
    from train_stage1 import adjust_learning_rate as adjust1
    from train_stage2 import adjust_learning_rate as adjust2

    raw = torchvision.datasets.CIFAR100(str(DATA), train=True, download=False)
    integrity = {}
    for filename, md5 in raw.train_list + raw.test_list:
        path = DATA / raw.base_folder / filename
        integrity[filename] = {"expected_md5": md5,
                                "verified": check_integrity(str(path), md5),
                                "sha256": sha(path)}
    integrity[raw.meta["filename"]] = {"verified": check_integrity(
        str(DATA / raw.base_folder / raw.meta["filename"]), raw.meta["md5"])}
    assert all(item["verified"] for item in integrity.values())

    cfg1 = base_config("stage1")
    d1 = CIFAR100_LT(cfg1, False, root=str(DATA), imb_factor=.01,
                    batch_size=128, num_works=0)
    counts = np.bincount(d1.train_instance.dataset.targets, minlength=100)
    expected = np.array([int(500 * (.01 ** (i / 99))) for i in range(100)])
    assert np.array_equal(counts, expected)
    assert counts.min() == 5 and counts.max() == 500 and len(counts) == 100
    test_counts = np.bincount(d1.eval.dataset.targets, minlength=100)
    assert np.all(test_counts == 100)
    cfg2 = base_config("stage2")
    d2 = CIFAR100_LT(cfg2, False, root=str(DATA), imb_factor=.01,
                    batch_size=128, num_works=0)
    assert np.array_equal(d1.train_instance.dataset.data, d2.train_instance.dataset.data)
    assert d1.train_instance.dataset.targets == d2.train_instance.dataset.targets

    model = resnet_cifar.resnet32_fe()
    classifier = resnet_cifar.Classifier(64, 100)
    assert model.feature_tuning is False and not hasattr(model, "conv1d")
    with torch.no_grad():
        image, _, _ = next(iter(d1.train_instance))
        feature = model(image)
        logits = classifier(feature)
    assert tuple(feature.shape) == (128, 64, 8, 8)
    assert tuple(logits.shape) == (128, 100)
    opt1 = torch.optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                           lr=cfg1.lr, momentum=cfg1.momentum,
                           weight_decay=cfg1.weight_decay)
    lr1 = {}
    for epoch in [0, 1, 4, 5, 159, 160, 179, 180, 199]:
        adjust1(opt1, epoch, cfg1)
        lr1[str(epoch + 1)] = opt1.param_groups[0]["lr"]
    opt2 = torch.optim.SGD(classifier.parameters(), lr=cfg2.lr,
                           momentum=cfg2.momentum, weight_decay=cfg2.weight_decay)
    lr2 = {}
    for epoch in range(10):
        adjust2(opt2, epoch, cfg2)
        lr2[str(epoch + 1)] = opt2.param_groups[0]["lr"]
    ce = CrossEntropyLoss(cfg1, counts)
    assert ce.per_cls_weights is None
    dump("preflight.json", {
        "integrity": integrity, "raw_train_count": len(raw),
        "long_tail_count": int(counts.sum()), "class_count": 100,
        "class_counts": counts.tolist(), "actual_if": float(counts.max()/counts.min()),
        "test_count": len(d1.eval.dataset), "test_class_counts": test_counts.tolist(),
        "stage1_stage2_same_subset": True,
        "subset_data_sha256": hashlib.sha256(d1.train_instance.dataset.data.tobytes()).hexdigest(),
        "subset_targets_sha256": hashlib.sha256(np.asarray(
            d1.train_instance.dataset.targets, dtype=np.int64).tobytes()).hexdigest(),
        "batches_per_epoch": len(d1.train_instance),
        "last_batch_size": len(d1.train_instance.dataset) % 128,
        "feature_shape": list(feature.shape), "logit_shape": list(logits.shape),
        "backbone_parameters": sum(p.numel() for p in model.parameters()),
        "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
        "batchnorm_modules": sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()),
        "PIF_active": False, "CE_reweighted": False,
        "stage1_lr_by_one_based_epoch": lr1, "stage2_lr_by_one_based_epoch": lr2,
        "stage1_config": cfg1.dump(), "stage2_config": cfg2.dump()})
    print("PREFLIGHT PASS: 100 classes, IF=100,", int(counts.sum()),
          "training images, 10000 test images", flush=True)


class MemoryMonitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                values = [int(x.strip()) for x in result.stdout.strip().split(",")]
                self.samples.append({"time": time.time(), "used_mib": values[0],
                                     "total_mib": values[1], "gpu_util_percent": values[2]})
            except Exception as exc:
                self.samples.append({"time": time.time(), "error": str(exc)})
            self.stop.wait(1)


def observe_train(module, result, stage):
    real_train = module.train
    real_h2t = module.H2T
    real_mixup = module.mixup_data
    real_mixup_loss = module.mixup_criterion
    context = {}
    mixes = []
    mixed_losses = []
    fusion_count = 0

    def observe_mixup(images, targets, *args, **kwargs):
        mixed, ya, yb, lam = real_mixup(images, targets, *args, **kwargs)
        assert torch.equal(ya, targets) and 0 <= lam <= 1
        mixes.append(float(lam))
        return mixed, ya, yb, lam

    def observe_mixup_loss(*args, **kwargs):
        loss = real_mixup_loss(*args, **kwargs)
        assert torch.isfinite(loss).item()
        mixed_losses.append(float(loss.detach()))
        return loss

    def observe_h2t(x1, x2, rho=.3):
        nonlocal fusion_count
        first = fusion_count == 0
        if first:
            before, donor = x1.clone(), x2.clone()
            generator = torch.Generator(device="cpu")
            generator.set_state(torch.get_rng_state())
            selected = torch.randperm(x1.shape[1], generator=generator)[:int(rho*x1.shape[1])]
            selected = selected.to(x1.device)
            other = torch.tensor([c for c in range(x1.shape[1]) if c not in selected.cpu().tolist()],
                                  device=x1.device)
            pointer = x1.data_ptr()
            assert not torch.is_grad_enabled() and not x1.requires_grad and not x2.requires_grad
        fused = real_h2t(x1, x2, rho)
        fusion_count += 1
        if first:
            assert fused.data_ptr() == pointer
            assert torch.equal(fused[:, selected], donor[:, selected])
            assert torch.equal(fused[:, other], before[:, other])
            assert torch.equal(x2, donor)
            result["fusion"] = {"rho": rho, "feature_shape": list(fused.shape),
                "selected_channels": selected.cpu().tolist(),
                "replaced_count": int(selected.numel()), "retained_count": int(other.numel()),
                "same_channel_mask_for_entire_batch": True, "inplace": True,
                "donor_unchanged": True, "replacement_exact": True,
                "retention_exact": True, "no_grad": True, "feature_requires_grad": False}
        return fused

    module.mixup_data = observe_mixup
    module.mixup_criterion = observe_mixup_loss
    module.H2T = observe_h2t

    def train(loader, model, classifier, criterion, optimizer, epoch, cfg, logger, block=None):
        initial_model = snapshot(model)
        initial_classifier = snapshot(classifier)
        initial_buffers = snapshot(model, True)
        if stage == "stage2":
            loaded = module._audit_checkpoint
            assert all(torch.equal(v, loaded["state_dict_model"][k].detach().cpu())
                       for k, v in (initial_model | initial_buffers).items())
            assert all(torch.equal(v, loaded["state_dict_classifier"][k].detach().cpu())
                       for k, v in initial_classifier.items())
            result["loaded_checkpoint"]["strict_backbone_state_equality"] = True
            result["loaded_checkpoint"]["strict_classifier_state_equality"] = True
        parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        backbone_ids = {id(p) for p in model.parameters()}
        classifier_ids = {id(p) for p in classifier.parameters()}
        result["optimizer"] = {"type": type(optimizer).__name__,
            "groups": [{k: v for k, v in group.items() if k != "params"}
                       for group in optimizer.param_groups],
            "contains_backbone": bool(backbone_ids & parameter_ids),
            "contains_all_classifier": classifier_ids <= parameter_ids}
        assert (parameter_ids == classifier_ids if stage == "stage2"
                else parameter_ids == backbone_ids | classifier_ids)
        loss_values = []
        step_count = 0
        classifier_has_grad = True
        backbone_has_grad = False
        recipient_hist = np.zeros(100, dtype=np.int64)
        donor_hist = np.zeros(100, dtype=np.int64)
        batch_sizes = []
        real_step = optimizer.step
        real_forward = criterion.forward

        class ObservedLoader:
            def __len__(self):
                return len(loader)

            def __getattr__(self, name):
                return getattr(loader, name)

            def __iter__(self):
                for images, targets, meta in loader:
                    context["targets"] = targets.clone()
                    recipient_hist[:] += np.bincount(targets.numpy(), minlength=100)
                    if "sample_label" in meta:
                        donor_hist[:] += np.bincount(meta["sample_label"].numpy(), minlength=100)
                    batch_sizes.append(len(targets))
                    yield images, targets, meta

        def forward(logits, targets, extra_info=None):
            loss = real_forward(logits, targets, extra_info)
            assert torch.isfinite(loss).item()
            loss_values.append(float(loss.detach()))
            if stage == "stage2":
                assert torch.equal(targets.detach().cpu(), context["targets"])
                assert logits.requires_grad
                if len(loss_values) == 1:
                    with torch.no_grad():
                        ordinary_ce = torch.nn.functional.cross_entropy(logits, targets)
                    assert torch.equal(loss.detach(), ordinary_ce)
                    result["recipient_labels_preserved"] = True
                    result["loss_matches_plain_CE"] = True
            return loss

        def step(*args, **kwargs):
            nonlocal step_count, classifier_has_grad, backbone_has_grad
            for p in classifier.parameters():
                classifier_has_grad &= p.grad is not None
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all().item()
            for p in model.parameters():
                backbone_has_grad |= p.grad is not None
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all().item()
            step_count += 1
            return real_step(*args, **kwargs)

        criterion.forward = forward
        optimizer.step = step
        real_train(ObservedLoader(), model, classifier, criterion, optimizer,
                   epoch, cfg, logger, block)
        final_model = snapshot(model)
        final_classifier = snapshot(classifier)
        final_buffers = snapshot(model, True)
        model_delta = delta(initial_model, final_model)
        classifier_delta = delta(initial_classifier, final_classifier)
        buffers_delta = delta(initial_buffers, final_buffers)
        bn_tracking = {k: int(final_buffers[k] - initial_buffers[k])
                       for k in initial_buffers if k.endswith("num_batches_tracked")}
        assert step_count == len(loader) and sum(batch_sizes) == len(loader.dataset)
        assert classifier_delta["changed_count"] > 0 and classifier_has_grad
        if stage == "stage1":
            assert model_delta["changed_count"] > 0 and backbone_has_grad
            assert len(mixes) == len(loader)
            assert np.array_equal(recipient_hist, np.bincount(loader.dataset.targets, minlength=100))
        else:
            assert model_delta["changed_count"] == 0 and not backbone_has_grad
            assert fusion_count == len(loader) and buffers_delta["changed_count"] > 0
            assert all(value == 2*len(loader) for value in bn_tracking.values())
            assert np.array_equal(donor_hist, np.bincount(loader.dataset.targets, minlength=100))
        objective_losses = mixed_losses if stage == "stage1" else loss_values
        assert len(objective_losses) == step_count
        result["train"] = {"epochs": 1, "optimizer_steps": step_count,
            "samples": sum(batch_sizes), "batch_sizes": batch_sizes,
            "loss_count": len(objective_losses), "CE_component_count": len(loss_values),
            "loss_min": min(objective_losses), "loss_max": max(objective_losses),
            "loss_mean": sum(objective_losses)/len(objective_losses),
            "loss_sample_weighted_mean": float(np.average(objective_losses, weights=batch_sizes)),
            "all_losses_finite": True, "all_gradients_finite": True,
            "backbone_grad_present": backbone_has_grad,
            "classifier_grad_present": classifier_has_grad,
            "backbone_parameter_change": model_delta,
            "classifier_parameter_change": classifier_delta,
            "backbone_buffer_change": buffers_delta,
            "bn_tracking_increments": bn_tracking,
            "backbone_requires_grad_flags": sorted({p.requires_grad for p in model.parameters()}),
            "backbone_training_after_train": model.training,
            "recipient_class_histogram": recipient_hist.tolist(),
            "donor_class_histogram": donor_hist.tolist(),
            "recipient_group_counts": [int(recipient_hist[:36].sum()),
                                        int(recipient_hist[36:71].sum()), int(recipient_hist[71:].sum())],
            "donor_group_counts": [int(donor_hist[:36].sum()),
                                    int(donor_hist[36:71].sum()), int(donor_hist[71:].sum())],
            "mixup_count": len(mixes), "mixup_lambdas": mixes,
            "fusion_count": fusion_count}
        criterion.forward = real_forward
        optimizer.step = real_step

    module.train = train


def observe_validate(module, result):
    real_validate = module.validate

    def validate(loader, model, classifier, criterion, cfg, logger, *args):
        buffers = snapshot(model, True)
        predictions = []
        def hook(_module, _inputs, output):
            predictions.extend(output.detach().argmax(1).cpu().tolist())
        handle = classifier.register_forward_hook(hook)
        acc, ece = real_validate(loader, model, classifier, criterion, cfg, logger, *args)
        handle.remove()
        assert len(predictions) == len(loader.dataset) == 10000
        assert delta(buffers, snapshot(model, True))["changed_count"] == 0
        targets = np.asarray(loader.dataset.targets)
        predictions = np.asarray(predictions)
        correct = np.bincount(targets[predictions == targets], minlength=100)
        counts = np.bincount(targets, minlength=100)
        per_class = correct/counts*100
        overall = float((targets == predictions).mean()*100)
        assert abs(float(acc)-overall) < 1e-4
        result["evaluation"] = {"samples": len(targets), "overall": overall,
            "head": float(per_class[:36].mean()), "medium": float(per_class[36:71].mean()),
            "tail": float(per_class[71:].mean()), "ece": float(ece),
            "correct_per_class": correct.tolist(), "test_class_counts": counts.tolist(),
            "backbone_eval_mode": not model.training, "BN_unchanged_during_eval": True,
            "original_accuracy_matches_independent_count": True}
        return acc, ece

    module.validate = validate


def run_entry(stage, checkpoint):
    if stage in ("stage2", "eval") and (not checkpoint or not Path(checkpoint).is_file()):
        raise ValueError("An existing audit checkpoint is required")
    if stage == "eval":
        spec = importlib.util.spec_from_file_location("official_eval", RUNTIME / "eval-modifed.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("train_" + stage)
    result = {"stage": stage, "status": "running", "started": time.time(),
              "entrypoint": str(Path(module.__file__).resolve())}
    cfg = "cifar100_imb001_stage1_ce_mixup.yaml" if stage == "stage1" else "cifar100_imb001_stage2_ce_H2T.yaml"
    opts = ["data_path", str(DATA), "resume", str(Path(checkpoint).resolve()) if checkpoint else "''",
            "name", "repro_audit_" + stage + "_1ep"]
    if stage != "eval":
        opts += ["num_epochs", "1"]
    sys.argv = [str(Path(module.__file__)), "--cfg", "./config/cifar100/" + cfg] + opts
    result["argv"] = sys.argv
    real_logger = module.create_logger
    def create_logger(cfg, cfg_name):
        values = real_logger(cfg, cfg_name)
        result["model_dir"] = str(Path(values[1]).resolve())
        result["effective_config"] = cfg.dump()
        return values
    module.create_logger = create_logger
    if stage != "eval":
        observe_train(module, result, stage)
    observe_validate(module, result)
    real_load = torch.load
    def load(path, *args, **kwargs):
        ckpt = real_load(path, *args, **kwargs)
        result["loaded_checkpoint"] = {"path": str(Path(path).resolve()), "sha256": sha(path),
            "epoch": ckpt["epoch"], "keys": sorted(ckpt.keys()),
            "weights_only": kwargs.get("weights_only", "default")}
        module._audit_checkpoint = ckpt
        return ckpt
    torch.load = load
    os.chdir(RUNTIME)
    monitor = MemoryMonitor()
    monitor.thread.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        module.main()
        result["status"] = "pass"
        if stage != "eval":
            current = Path(result["model_dir"]) / "current.pth.tar"
            assert current.is_file()
            result["checkpoint"] = {"path": str(current), "sha256": sha(current)}
    except Exception:
        result["status"] = "fail"
        result["error"] = traceback.format_exc()
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=6)
        result["elapsed_seconds"] = time.time()-result["started"]
        result["gpu_memory"] = {"peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
            "max_total_gpu_used_mib": max((x["used_mib"] for x in monitor.samples
                                           if "used_mib" in x), default=None),
            "device": torch.cuda.get_device_name(0), "samples": monitor.samples}
        dump(stage + ".json", result)
        print("AUDIT", stage, result["status"], "seconds", round(result["elapsed_seconds"], 2), flush=True)


def probe_load(checkpoint):
    try:
        torch.load(checkpoint, map_location="cpu")
        result = {"default_load_pass": True}
    except Exception:
        result = {"default_load_pass": False, "error": traceback.format_exc()}
    dump("checkpoint_load_probe.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "stage1", "stage2", "eval", "probe-load"])
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    EVIDENCE.mkdir(exist_ok=True)
    if args.command == "preflight":
        preflight()
    elif args.command == "probe-load":
        probe_load(args.checkpoint)
    else:
        run_entry(args.command, args.checkpoint)
