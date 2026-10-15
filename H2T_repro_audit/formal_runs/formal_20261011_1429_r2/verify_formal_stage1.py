"""Verify completed Stage I artifacts and independently evaluate saved states.

Uses the audited official eval-modifed.py, with passive prediction/BN hooks.
No training or Stage II entry point is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
RUNTIME = BASE / "runtime"
sys.path.insert(0, str(RUNTIME))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def verify(folder):
    status, preflight = read(folder / "status.json"), read(folder / "preflight.json")
    assert status["state"] == "completed" and status["completed_epochs"] == 200
    assert status["checkpoint_load_calls"] == []
    assert len(status["epochs"]) == 200
    assert [e["epoch"] for e in status["epochs"]] == list(range(1, 201))
    assert all(e["steps"] == 85 and e["samples"] == 10847 and e["finite_train_loss"] for e in status["epochs"])
    assert all(np.isfinite(e["train_loss"]) and np.isfinite(e["eval_overall_percent"]) for e in status["epochs"])
    expected_lr = [(.1 * e / 5 if e <= 5 else .001 if e > 180 else .01 if e > 160 else .1) for e in range(1, 201)]
    assert np.allclose([e["lr"] for e in status["epochs"]], expected_lr, rtol=0, atol=1e-15)
    assert all(sha(RUNTIME / p) == value for p, value in preflight["runtime_files"].items())
    assert all(sha(Path(status["code_dir"]) / p) == value for p, value in preflight["runtime_files"].items())
    checkpoints = {}
    ckps = Path(status["model_dir"])
    for filename in ["model_best.pth.tar", "current.pth.tar", "model_final.pth.tar"]:
        path = ckps / filename
        assert sha(path) == status["checkpoints_sha256"][filename]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert all(torch.isfinite(v).all().item() for key in ["state_dict_model", "state_dict_classifier"] for v in checkpoint[key].values())
        checkpoints[filename] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
            "epoch": int(checkpoint["epoch"]), "best_acc1_percent": float(checkpoint["best_acc1"]),
            "all_state_tensors_finite": True,
            "bn_counts": {k: int(v) for k, v in checkpoint["state_dict_model"].items() if k.endswith("num_batches_tracked")}}
    assert checkpoints["current.pth.tar"]["epoch"] == 200
    assert checkpoints["model_final.pth.tar"]["sha256"] == checkpoints["current.pth.tar"]["sha256"]
    assert checkpoints["model_best.pth.tar"]["epoch"] == status["best_epoch"]
    assert all(v == 17000 for v in checkpoints["current.pth.tar"]["bn_counts"].values())
    assert len(checkpoints["current.pth.tar"]["bn_counts"]) == 31
    assert status["best_epoch"] == int(np.argmax([e["eval_overall_percent"] for e in status["epochs"]])) + 1
    write(folder / "artifact_verification.json", {"passed": True, "completed_epochs": 200,
        "optimizer_steps": 17000, "training_loss_finite_every_step": True,
        "no_checkpoint_loaded_during_training": True, "original_lr_schedule_verified": True,
        "runtime_source_unchanged": True, "source_snapshot_verified": True,
        "checkpoints": checkpoints})
    print("ARTIFACT VERIFICATION PASS: 200 epochs / 17000 steps; best/final/source/LR verified", flush=True)


class LogMetrics(logging.Handler):
    def __init__(self):
        super().__init__()
        self.value = None

    def emit(self, record):
        m = re.search(r"\* Acc@1 ([\d.]+)% Acc@5 ([\d.]+)% HAcc ([\d.]+)% MAcc ([\d.]+)% TAcc ([\d.]+)%", record.getMessage())
        if m:
            self.value = dict(zip(["overall", "top5", "head", "medium", "tail"], map(float, m.groups())))


def evaluate(folder, kind):
    status = read(folder / "status.json")
    assert status["state"] == "completed"
    filename = "model_best.pth.tar" if kind == "best" else "model_final.pth.tar"
    checkpoint_path = Path(status["model_dir"]) / filename
    spec = importlib.util.spec_from_file_location("original_formal_eval", RUNTIME / "eval-modifed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_load, real_validate, real_logger = torch.load, module.validate, module.create_logger
    loaded = []
    metrics = LogMetrics()
    observed = {"class_counts": np.zeros(100, dtype=np.int64), "correct_counts": np.zeros(100, dtype=np.int64)}

    def load(path, *args, **kwargs):
        assert Path(path).resolve() == checkpoint_path.resolve()
        loaded.append({"path": str(path), "sha256": sha(path)})
        return real_load(path, *args, **kwargs)

    def logger(cfg, name):
        result = real_logger(cfg, name)
        result[0].addHandler(metrics)
        return result

    def validate(loader, model, classifier, lws_model, criterion, cfg, log, block=None):
        before = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        class IterationObserver:
            def __len__(self):
                return len(loader)
            def __iter__(self):
                for images, targets in loader:
                    observed["targets"] = targets.numpy()
                    yield images, targets
        def prediction_hook(_module, _inputs, output):
            targets = observed["targets"]
            predicted = output.argmax(1).detach().cpu().numpy()
            observed["class_counts"] += np.bincount(targets, minlength=100)
            observed["correct_counts"] += np.bincount(targets[targets == predicted], minlength=100)
        handle = classifier.register_forward_hook(prediction_hook)
        try:
            value = real_validate(IterationObserver(), model, classifier, lws_model, criterion, cfg, log, block)
        finally:
            handle.remove()
        assert all(torch.equal(v, dict(model.named_buffers())[k].detach().cpu()) for k, v in before.items())
        observed["bn_buffers_unchanged_during_eval"] = True
        observed["official_acc1"] = float(value[0])
        observed["official_ece"] = float(value[1])
        return value

    torch.load, module.validate, module.create_logger = load, validate, logger
    sys.argv = ["eval-modifed.py", "--cfg", "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml",
        "data_path", str(BASE.parent / "H2T/data/cifar100"), "resume", str(checkpoint_path),
        "name", status["run_name"] + "_" + kind + "_eval"]
    os.chdir(RUNTIME)
    try:
        module.main()
    finally:
        torch.load = real_load
    assert len(loaded) == 1 and np.all(observed["class_counts"] == 100)
    correct, counts = observed["correct_counts"], observed["class_counts"]
    per_class = correct / counts * 100
    exact = {"overall": float(correct.sum() / counts.sum() * 100),
             "head": float(per_class[:36].mean()), "medium": float(per_class[36:71].mean()),
             "tail": float(per_class[71:].mean())}
    assert all(abs(exact[k] - metrics.value[k]) < .0006 for k in exact)
    logged_epoch = status["epochs"][status["best_epoch"] - 1 if kind == "best" else -1]
    assert all(abs(exact[k] - logged_epoch["eval_metrics_percent_rounded_3dp"][k]) < .0006 for k in exact)
    write(folder / (kind + "_evaluation.json"), {"checkpoint": loaded[0],
        "epoch": status["best_epoch"] if kind == "best" else 200, "metrics_percent": exact,
        "official_metrics_rounded_3dp": metrics.value, "ece_percent": observed["official_ece"],
        "class_counts": counts.tolist(), "correct_counts": correct.tolist(),
        "per_class_accuracy_percent": per_class.tolist(), "bn_buffers_unchanged_during_eval": True,
        "matches_training_epoch_evaluation": True, "stage2_started": False})
    print(f"INDEPENDENT {kind.upper()} EVALUATION PASS: {exact}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["artifacts", "best", "final"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    folder = BASE / "formal_runs" / args.run_id
    if args.action == "artifacts":
        verify(folder)
    else:
        evaluate(folder, args.action)
