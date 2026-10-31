"""Seed-42 real-data smoke ONLY: two Stage-I and four Stage-II single epochs.

There is deliberately no formal-training or test-evaluation entry point here.
"""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn

import models
from data import load_rows, schedules, Views, loader, transform
from preflight import BRANCH, EXPECTED_SHA, MANIFEST
from amp_recovery import OptimizerStepCounter, guarded_scaler_step, make_grad_scaler

VALIDATION = Path(r"D:\h2t_cross_domain_results\fish16_validation_seed42_20261031_01")


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def cleanup(model):
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def finite_gradients(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients), "Nonfinite/missing gradients"
    assert any(g.abs().sum().item() > 0 for g in gradients), "All gradients zero"


def metrics(logits, labels, groups):
    predictions = logits.argmax(1)
    correct = predictions == labels
    per_class = [float(correct[labels == c].float().mean() * 100) for c in range(16)]
    return {"overall": float(correct.float().mean() * 100), "macro": sum(per_class) / 16,
            "worst_class": min(per_class), "per_class": per_class,
            "group_image": {g: float(correct[torch.isin(labels, torch.tensor(ids))].float().mean() * 100)
                            for g, ids in groups.items()},
            "group_macro": {g: sum(per_class[i] for i in ids) / len(ids) for g, ids in groups.items()}}


@torch.no_grad()
def evaluate_validation(model, rows, val, groups, batch):
    model.eval()
    outputs, labels = [], []
    for item in loader(Views(rows, val, False), batch):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(item["recipient"].cuda(non_blocking=True))
        assert torch.isfinite(logits).all()
        outputs.append(logits.float().cpu())
        labels.append(item["label"])
    outputs, labels = torch.cat(outputs), torch.cat(labels)
    assert len(labels) == 5396
    return metrics(outputs, labels, groups)


def batch_smoke(rows, rec, donors, batch, output):
    # This controlled batch is discarded; it never initializes a later smoke phase.
    one = next(iter(loader(Views(rows, rec[:batch], True, donors=donors[:batch]), batch)))
    proofs = []
    for name, pif, mode in (("A1", False, "crt"), ("A2", False, "h2t"), ("B1", True, "crt"), ("B2", True, "pi")):
        seed()
        model = models.FishModel(pif).cuda()
        model.freeze()
        before = models.state_sha(model.extractor)
        weight_before = model.classifier.weight.detach().clone()
        opt = torch.optim.SGD(model.classifier.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
        opt.zero_grad(set_to_none=True)
        scaler = make_grad_scaler()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            r = model.extractor(one["recipient"].cuda())
            d = model.extractor(one["donor"].cuda()) if mode != "crt" else None
            assert r.shape == (batch, 2048, 7, 7)
            fused, extra = models.stage2_features(r, d, one["label"].cuda(), model.classifier, mode)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.cross_entropy(model.classifier(fused), one["label"].cuda())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        finite_gradients(model.classifier.parameters())
        assert all(p.grad is None for p in model.extractor.parameters())
        scaler.step(opt)
        scaler.update()
        assert not torch.equal(model.classifier.weight, weight_before)
        assert before == models.state_sha(model.extractor)
        proofs.append({"condition": name, "mode": mode, "feature_shape": list(r.shape),
                       "dtype": str(r.dtype), "loss": float(loss.detach()), "classifier_updated": True,
                       "extractor_and_bn_unchanged": True, "h2t_channels": len(extra) if mode == "h2t" else None,
                       "ratio_detached": not extra.requires_grad if mode == "pi" else None})
        cleanup(model)
        del model, opt, scaler, r, d, fused, loss, weight_before
        torch.cuda.empty_cache()
    save_json(output / "BATCH_SMOKE.json", proofs)
    print("REAL_BATCH_SMOKE_PASS_ALL_FOUR_CONDITIONS", flush=True)


def one_epoch(name, model, rows, indices, donors, mode, stage, batch, groups, val, out):
    seed()
    folder = out / name
    folder.mkdir(exist_ok=False)
    model.cuda()
    if stage == 1:
        model.train()
        parameters = list(model.parameters())
        lr, wd = .01 / 5, 1e-4  # Pilot's first warmup epoch, not a 1-epoch-rescaled schedule.
    else:
        model.freeze()
        parameters = list(model.classifier.parameters())
        lr, wd = .01, 5e-4  # First epoch of fixed 10-epoch cosine schedule.
    before_extractor = models.state_sha(model.extractor)
    before_classifier = models.state_sha(model.classifier)
    before_pif = [float(p.detach().flatten()[i]) for p in model.extractor.pif.parameters() for i in range(p.numel())]
    opt = torch.optim.SGD(parameters, lr=lr, momentum=.9, weight_decay=wd)
    scaler = make_grad_scaler()
    step_counter = OptimizerStepCounter(opt)
    optimizer_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    named_optimizer_parameters = [
        (name, p) for name, p in model.named_parameters() if id(p) in optimizer_ids
    ]
    assert len(named_optimizer_parameters) == len(parameters)
    initial_loss_scale = scaler.get_scale()
    accumulation = 16 // batch
    dataset = Views(rows, indices, True, donors=donors)
    batches = loader(dataset, batch)
    controls = []
    channel_indices, ratios, ratio_labels = [], [], []
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss_sum, updates, samples, finite_checks = 0., 0, 0, 0
    attempted_steps = 0
    overflow_skips = 0
    consecutive_overflow_skips = 0
    max_consecutive_overflow_skips = 0
    optimizer_step_events = []
    for i, item in enumerate(batches):
        target = item["label"].cuda(non_blocking=True)
        row = {"batch": i, "samples": len(target), "recipient_sha256": digest_tensor(item["recipient"]),
               "donor_sha256": digest_tensor(item["donor"]) if donors is not None else None}
        x = item["recipient"].cuda(non_blocking=True)
        if stage == 1:
            row["mixup_rng_sha256"] = digest_tensor(torch.get_rng_state())
            x, a, b, lam = models.methods.mixup_data(x, target, alpha=1.0)
            row["mixup_lambda"] = float(lam)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = lam * nn.functional.cross_entropy(logits, a) + (1-lam) * nn.functional.cross_entropy(logits, b)
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                r = model.extractor(x)
                d = model.extractor(item["donor"].cuda(non_blocking=True)) if donors is not None else None
                features, extra = models.stage2_features(r, d, target, model.classifier, mode)
                if mode == "h2t":
                    assert len(extra) == 512
                    channel_indices.append(extra.numpy().astype(np.int16))
                elif mode == "pi":
                    assert torch.isfinite(extra).all() and not extra.requires_grad
                    assert (extra >= 0).all() and (extra <= 1).all()
                    ratios.append(extra.float().cpu().numpy())
                    ratio_labels.append(item["label"].numpy())
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model.classifier(features)
                loss = nn.functional.cross_entropy(logits, target)
        assert torch.isfinite(loss) and torch.isfinite(logits).all(), "Nonfinite forward"
        group_start = (i // accumulation) * 16
        group_size = min(16, len(dataset)-group_start)
        scaler.scale(loss * len(target) / group_size).backward()
        if (i+1) % accumulation == 0 or i+1 == len(batches):
            scaler.unscale_(opt)
            assert model.classifier.weight.grad is not None
            if stage == 2:
                assert all(p.grad is None for p in model.extractor.parameters())
            outcome = guarded_scaler_step(
                optimizer=opt,
                scaler=scaler,
                named_parameters=named_optimizer_parameters,
                step_counter=step_counter,
                allow_pif_b_overflow=(stage == 1 and model.use_pif),
                consecutive_overflow_skips=consecutive_overflow_skips,
                max_consecutive_overflow_skips=2,
            )
            attempted_steps += 1
            finite_checks += 1
            if outcome.skipped:
                overflow_skips += 1
                consecutive_overflow_skips += 1
                max_consecutive_overflow_skips = max(
                    max_consecutive_overflow_skips, consecutive_overflow_skips
                )
            else:
                updates += 1
                consecutive_overflow_skips = 0
            event = outcome.as_dict()
            event.update({
                "condition": name,
                "batch_start": group_start,
                "batch_end": i,
                "optimizer_updates_total": updates,
                "overflow_skips_total": overflow_skips,
                "consecutive_overflow_skips": consecutive_overflow_skips,
            })
            optimizer_step_events.append(event)
            opt.zero_grad(set_to_none=True)
        samples += len(target)
        loss_sum += float(loss.detach()) * len(target)
        controls.append(row)
        if (i+1) % 100 == 0:
            print(f"{name} smoke epoch1 batch {i+1}/{len(batches)}, loss={float(loss.detach()):.5f}", flush=True)
    step_counter.close()
    torch.cuda.synchronize()
    duration = time.perf_counter()-start
    expected_attempts = math.ceil(16269 / 16)
    assert samples == 16269 and attempted_steps == expected_attempts
    assert updates + overflow_skips == attempted_steps
    assert updates == step_counter.count
    assert overflow_skips <= 3 and max_consecutive_overflow_skips <= 2
    if not (stage == 1 and model.use_pif):
        assert overflow_skips == 0, "Unexpected AMP overflow outside B0 PIF Stage-I"
    else:
        first_skip_index = next(
            (i for i, row in enumerate(optimizer_step_events)
             if row["kind"] == "recoverable_pif_b_overflow_skip"),
            None,
        )
        assert first_skip_index is not None, "Expected B0 PIF overflow was not reproduced"
        first_skip = optimizer_step_events[first_skip_index]
        if batch == 16:
            assert first_skip["batch_end"] == 6, "B0 overflow moved away from archived batch 6"
            assert first_skip["scale_before"] == 65536.0 and first_skip["scale_after"] == 32768.0
        assert first_skip_index + 1 < len(optimizer_step_events), "No optimizer window after the overflow"
        next_step = optimizer_step_events[first_skip_index + 1]
        assert next_step["kind"] == "optimizer_step", "No successful optimizer step after overflow"
        assert next_step["scale_before"] == first_skip["scale_after"]
    save_json(folder / "optimizer_step_events.json", optimizer_step_events)
    if stage == 2:
        assert before_extractor == models.state_sha(model.extractor), "Extractor or BN mutated"
    assert before_classifier != models.state_sha(model.classifier)
    validation = evaluate_validation(model, rows, val, groups, batch)
    after_pif = [float(p.detach().flatten()[i]) for p in model.extractor.pif.parameters() for i in range(p.numel())]
    checkpoint = folder / "smoke_epoch001.pth"
    with checkpoint.open("xb") as f:
        torch.save({"smoke_only": True, "condition": name, "epoch": 1, "seed": 42,
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "validation": validation}, f)
    save_json(folder / "input_controls.json", controls)
    result = {"condition": name, "scope": "ONE_EPOCH_SMOKE_ONLY", "stage": stage,
              "samples": samples, "optimizer_updates": updates,
              "attempted_optimizer_steps": attempted_steps, "amp_overflow_skips": overflow_skips,
              "max_consecutive_overflow_skips": max_consecutive_overflow_skips,
              "initial_loss_scale": initial_loss_scale, "final_loss_scale": float(scaler.get_scale()),
              "finite_gradient_checks": finite_checks,
              "lr": lr, "momentum": .9, "weight_decay": wd, "micro_batch": batch,
              "accumulation": accumulation, "effective_batch": 16, "amp": True, "workers": 4,
              "train_loss": loss_sum / samples, "train_seconds": duration,
              "train_images_per_second": samples / duration,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
              "validation": validation, "pif_before": before_pif, "pif_after": after_pif,
              "extractor_initial_sha256": before_extractor,
              "extractor_final_sha256": models.state_sha(model.extractor),
              "classifier_initial_sha256": before_classifier,
              "checkpoint": str(checkpoint), "checkpoint_sha256": models.sha(checkpoint),
              "selection": "Only epoch1; smoke best/current/final refer to this same file; validation only"}
    if channel_indices:
        channels = np.stack(channel_indices)
        np.savez_compressed(folder / "h2t_channel_indices.npz", indices=channels)
        save_json(folder / "h2t_channel_audit.json", {"batches": len(channels), "count_per_batch": 512,
                  "batch_shared": True, "first_indices": channels[0].tolist(),
                  "selection_counts": np.bincount(channels.ravel(), minlength=2048).tolist()})
    if ratios:
        values, labels = np.concatenate(ratios), np.concatenate(ratio_labels)
        np.savez_compressed(folder / "pi_ratios.npz", recipient_mass=values, target=labels)
        def stats(v):
            return {"count": len(v), "mean": float(v.mean()), "std": float(v.std()),
                    "min": float(v.min()), "max": float(v.max()), "histogram_20": np.histogram(v, bins=20, range=(0, 1))[0].tolist()}
        save_json(folder / "pi_ratio_audit.json", {"ratio_detach": True, "recipient_mass": stats(values),
                  "donor_mass": stats(1-values), "per_class": {str(c): {"recipient_mass": stats(values[labels==c]),
                  "donor_mass": stats(1-values[labels==c])} for c in range(16)}})
    save_json(folder / "result.json", result)
    cleanup(model)
    del opt, scaler, model, batches
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{name} SMOKE_DONE train_seconds={duration:.2f}, val={validation['overall']:.4f}", flush=True)
    return result, controls


def run(output, batch):
    assert torch.cuda.is_available()
    assert models.sha(models.ORIGINAL_WEIGHT) == models.ORIGINAL_SHA
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=models.ROOT, text=True).strip() == BRANCH
    original_sources = json.loads((VALIDATION / "ORIGINAL_SOURCE_HASHES.json").read_text(encoding="utf-8"))
    assert models.source_hashes() == original_sources
    log = (VALIDATION / "CPU_TEST_RESULTS_FINAL.txt").read_text(encoding="utf-8-sig")
    assert "Ran 10 tests" in log and "OK" in log
    assert models.sha(MANIFEST) == EXPECTED_SHA
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed()
    rows = load_rows()
    train, val, stage1, rec, donors = schedules(rows)
    counts = Counter(rows[int(i)]["target"] for i in train)
    names = {r["target"]: r["label"] for r in rows}
    ranked = sorted(range(16), key=lambda c: (-counts[c], names[c]))
    groups = {"Head": ranked[:6], "Medium": ranked[6:11], "Tail": ranked[11:]}
    group_by_class = {c: g for g, ids in groups.items() for c in ids}
    np.savez_compressed(output / "smoke_sampling_schedule.npz", stage1=stage1, recipients=rec, donors=donors)
    with (output / "donor_pair_audit.csv").open("x", newline="", encoding="utf-8") as f:
        fields = ["position", "recipient_index", "donor_index", "recipient_class", "donor_class", "donor_group", "same_image", "same_track", "same_class"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (r, d) in enumerate(zip(rec, donors)):
            a, b = rows[int(r)], rows[int(d)]
            w.writerow(dict(zip(fields, [i, int(r), int(d), a["target"], b["target"], group_by_class[b["target"]],
                                        bool(r==d), a["base_group"]==b["base_group"], a["target"]==b["target"]])))
    save_json(output / "run_manifest.json", {"scope": "SMOKE_ONLY_NO_FORMAL_TRAINING_AUTHORIZATION", "seed": 42,
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=models.ROOT, text=True).strip(),
              "branch": BRANCH, "manifest_sha256": EXPECTED_SHA, "weight_sha256": models.ORIGINAL_SHA,
              "original_source_hashes": original_sources, "new_source_hashes": {p.name: models.sha(p) for p in Path(__file__).parent.glob("*.py")},
              "micro_batch": batch, "effective_batch": 16, "amp": True, "workers": 4,
              "stage1_epochs_executed": 1, "stage2_epochs_executed": 1, "schedule_denominators": [90, 10],
              "train_transform": str(transform(True)), "validation_transform": str(transform(False)),
              "numeric_config": {"cudnn_deterministic": True, "benchmark": False, "cudnn_tf32": False, "matmul_tf32": False},
              "test_model_evaluations": 0, "class_groups": groups, "note": "BN frozen for Stage-II; differs from original shift_bn=True"})
    batch_smoke(rows, rec, donors, batch, output)
    results, controls = {}, {}
    for name, pif in (("A0", False), ("B0", True)):
        results[name], controls[name] = one_epoch(name, models.FishModel(pif), rows, stage1, None, "stage1", 1, batch, groups, val, output)
    for key in ("recipient_sha256", "mixup_rng_sha256", "mixup_lambda"):
        assert [r[key] for r in controls["A0"]] == [r[key] for r in controls["B0"]], key
    for name, source, mode in (("A1", "A0", "crt"), ("A2", "A0", "h2t"), ("B1", "B0", "crt"), ("B2", "B0", "pi")):
        model = models.FishModel(source == "B0")
        state = torch.load(results[source]["checkpoint"], map_location="cpu", weights_only=True)
        assert state["smoke_only"] and state["seed"] == 42
        model.load_state_dict(state["model"], strict=True)
        results[name], controls[name] = one_epoch(name, model, rows, rec, donors if mode != "crt" else None, mode, 2, batch, groups, val, output)
    assert all([r["recipient_sha256"] for r in controls[n]] == [r["recipient_sha256"] for r in controls["A1"]] for n in ("A2", "B1", "B2"))
    assert [r["donor_sha256"] for r in controls["A2"]] == [r["donor_sha256"] for r in controls["B2"]]
    assert results["A1"]["extractor_initial_sha256"] == results["A2"]["extractor_initial_sha256"]
    assert results["B1"]["extractor_initial_sha256"] == results["B2"]["extractor_initial_sha256"]
    assert results["A1"]["classifier_initial_sha256"] == results["A2"]["classifier_initial_sha256"]
    assert results["B1"]["classifier_initial_sha256"] == results["B2"]["classifier_initial_sha256"]
    assert models.source_hashes() == original_sources
    assert models.sha(MANIFEST) == EXPECTED_SHA
    save_json(output / "SMOKE_SUMMARY.json", {"status": "PASS_SMOKE_ONLY", "results": results,
              "all_recipient_views_exactly_paired": True, "fusion_donor_views_exactly_paired": True,
              "stage1_mixup_and_views_exactly_paired": True, "source_initializations_matched": True,
              "formal_training_started": False, "test_evaluations": 0, "additional_seeds": []})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--micro-batch", type=int, choices=(16, 8), default=16)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    assert not output.is_relative_to(models.ROOT)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        run(output, args.micro_batch)
    except Exception as exc:
        save_json(output / "FAILURE.json", {"error_type": type(exc).__name__, "traceback": traceback.format_exc(),
                  "scope": "SMOKE_ONLY", "elapsed_seconds": time.time()-started, "formal_training_started": False})
        raise
    print(f"SMOKE_COMPLETE_ONLY elapsed_seconds={time.time()-started:.2f}; FORMAL_TRAINING_NOT_STARTED", flush=True)
