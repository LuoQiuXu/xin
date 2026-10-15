"""Launch the audited, unchanged Stage I entry point and save passive evidence.

This observer never replaces the model, data sampler, optimizer, loss, scheduler
or training loop. It invokes the original functions and observes their results.
Only data_path, empty resume, and the unique output name override the YAML.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
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
DATA = BASE.parent / "H2T/data/cifar100"
CFG_REL = "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml"
sys.path.insert(0, str(RUNTIME))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    target = Path(path)
    temporary = target.with_name(target.name + ".writing")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def check_sources():
    expected = json.loads((BASE / "evidence/runtime_manifest.json").read_text(encoding="utf-8"))["files"]
    actual = {p: sha(RUNTIME / p) for p in expected}
    assert actual == expected, "Audited compatibility copy changed"
    source = json.loads((BASE / "evidence/source_manifest.json").read_text(encoding="utf-8"))
    assert all(sha(BASE / "official" / f["path"]) == f["sha256"] for f in source["files"])
    return actual


def config_for(name):
    from utils import config, update_config
    cfg = config.clone()
    update_config(cfg, argparse.Namespace(cfg=str(RUNTIME / CFG_REL),
        opts=["data_path", str(DATA), "resume", "''", "name", name]))
    assert cfg.num_epochs == 200 and cfg.batch_size == 128 and cfg.imb_factor == .01
    assert cfg.backbone == "resnet32_fe" and cfg.classifier == "Classifier"
    assert cfg.mixup and cfg.alpha == 1.0 and cfg.loss.type == "CrossEntropyLoss"
    assert not cfg.loss.CE.reweight_CE and cfg.gpu is None and cfg.workers == 0
    assert cfg.lr == .1 and cfg.momentum == .9 and cfg.weight_decay == 2e-4
    assert cfg.lr_scheduler.type == "multistep" and cfg.lr_scheduler.lr_step == [160, 180]
    assert cfg.resume == "" and cfg.deterministic and not cfg.multiprocessing_distributed
    return cfg


def prepare(folder, name):
    from datasets.cifar100 import CIFAR100_LT
    from torchvision.datasets.utils import check_integrity
    assert not folder.exists(), f"Output already exists: {folder}"
    assert not list((RUNTIME / "saved").glob(name + "_*")), "Official output name is not unique"
    manifest = check_sources()
    cfg = config_for(name)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.cuda.get_device_properties(0)
    probe = torch.ones(32, device="cuda").sum().item()
    assert probe == 32
    raw = torchvision.datasets.CIFAR100(str(DATA), train=True, download=False)
    integrity = {}
    for filename, md5 in raw.train_list + raw.test_list + [(raw.meta["filename"], raw.meta["md5"])]:
        path = DATA / raw.base_folder / filename
        integrity[filename] = {"md5": md5, "verified": check_integrity(str(path), md5), "sha256": sha(path)}
    assert all(v["verified"] for v in integrity.values())
    dataset = CIFAR100_LT(cfg, False, root=str(DATA), imb_factor=cfg.imb_factor,
                          batch_size=cfg.batch_size, num_works=cfg.workers)
    counts = np.bincount(dataset.train_instance.dataset.targets, minlength=100)
    expected = np.array([int(500 * (.01 ** (i / 99))) for i in range(100)])
    assert np.array_equal(counts, expected) and counts.sum() == 10847
    assert np.all(np.bincount(dataset.eval.dataset.targets, minlength=100) == 100)
    old = json.loads((BASE / "evidence/preflight.json").read_text(encoding="utf-8"))
    subset_sha = hashlib.sha256(dataset.train_instance.dataset.data.tobytes()).hexdigest()
    assert subset_sha == old["subset_data_sha256"]
    free = shutil.disk_usage(RUNTIME).free
    assert free > 2 * 1024**3
    folder.mkdir(parents=True)
    (folder / "original.yaml").write_bytes((RUNTIME / CFG_REL).read_bytes())
    (folder / "effective.yaml").write_text(cfg.dump(), encoding="utf-8")
    for filename in ["compatibility.diff", "source_manifest.json", "runtime_manifest.json"]:
        shutil.copy2(BASE / "evidence" / filename, folder / filename)
    shutil.copy2(BASE / "requirements-repro.txt", folder / "requirements-repro.txt")
    shutil.copy2(__file__, folder / "formal_stage1.py")
    command = [sys.executable, "-u", "train_stage1.py", "--cfg", CFG_REL,
               "data_path", str(DATA), "resume", "''", "name", name]
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (folder / "nvidia_smi_preflight.txt").write_text(smi.stdout, encoding="utf-8")
    dump(folder / "preflight.json", {"prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable, "python_version": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": device.name,
        "compute_capability": [device.major, device.minor], "gpu_memory_bytes": device.total_memory,
        "cuda_probe_passed": True, "data_path": str(DATA), "integrity": integrity,
        "class_counts": counts.tolist(), "train_samples": int(counts.sum()), "test_samples": 10000,
        "batches_per_epoch": len(dataset.train_instance), "subset_sha256": subset_sha,
        "disk_free_bytes": free, "run_name": name, "run_evidence_dir": str(folder),
        "resume": "", "old_checkpoint_loading_enabled": False,
        "seed": 123, "cwd": str(RUNTIME), "original_entry_command_argv": command,
        "launcher_sha256": sha(__file__), "runtime_files": manifest,
        "stage2_authorized": False})
    print(f"PREFLIGHT PASS; evidence={folder}; original config retained; resume empty", flush=True)


class GpuMonitor:
    def __init__(self, folder):
        self.path = folder / "gpu_samples.jsonl"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.peak_mib = 0

    def work(self):
        with self.path.open("w", encoding="utf-8", buffering=1) as stream:
            while not self.stop.is_set():
                sample = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                try:
                    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    v = [float(x.strip()) for x in p.stdout.strip().split(",")]
                    sample.update(dict(zip(["used_mib", "total_mib", "gpu_util_percent", "temperature_c", "power_w"], v)))
                    self.peak_mib = max(self.peak_mib, v[0])
                except Exception as e:
                    sample["error"] = str(e)
                stream.write(json.dumps(sample) + "\n")
                self.stop.wait(5)


class MetricsHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.metrics = None

    def emit(self, record):
        message = record.getMessage()
        m = re.search(r"\* Acc@1 ([\d.]+)% Acc@5 ([\d.]+)% HAcc ([\d.]+)% MAcc ([\d.]+)% TAcc ([\d.]+)%", message)
        if m:
            self.metrics = dict(zip(["overall", "top5", "head", "medium", "tail"], map(float, m.groups())))


def tensor_state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(folder, name):
    assert (folder / "preflight.json").is_file(), "Run preflight first"
    preflight = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
    assert check_sources() == preflight["runtime_files"]
    assert sha(__file__) == preflight["launcher_sha256"]
    assert preflight["run_name"] == name
    assert not (folder / "status.json").exists(), "A run already used this evidence directory"
    assert not list((RUNTIME / "saved").glob(name + "_*")), "Output name already used"
    os.chdir(RUNTIME)
    import train_stage1 as original
    status = {"state": "running", "pid": os.getpid(), "run_name": name,
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "completed_epochs": 0,
              "checkpoint_load_calls": [], "epochs": []}
    monitor = GpuMonitor(folder)
    metrics = MetricsHandler()
    real_logger, real_parse = original.create_logger, original.parse_args
    real_train, real_validate = original.train, original.validate
    real_save, real_loss = original.save_checkpoint, original.mixup_criterion
    real_load = torch.load
    context = {}

    def parse():
        args = real_parse()
        assert original.config.dump() == config_for(name).dump()
        assert not original.config.resume
        return args

    def create_logger(cfg, path):
        logger, model_dir, code_dir = real_logger(cfg, path)
        logger.addHandler(metrics)
        status.update({"model_dir": str(Path(model_dir).resolve()), "code_dir": str(Path(code_dir).resolve()),
                       "official_log": str(Path(code_dir).resolve().parent / "logs" / (name + ".txt"))})
        dump(folder / "status.json", status)
        return logger, model_dir, code_dir

    def observe_load(*args, **kwargs):
        status["checkpoint_load_calls"].append(str(args[0]))
        raise RuntimeError("Stage I from scratch must never load a checkpoint")

    def loss(*args, **kwargs):
        value = real_loss(*args, **kwargs)
        number = float(value.detach())
        assert np.isfinite(number), "Nonfinite training loss"
        batch = args[1].shape[0]
        context["loss_sum"] += number * batch
        context["samples"] += batch
        context["steps"] += 1
        context["loss_min"] = min(context["loss_min"], number)
        context["loss_max"] = max(context["loss_max"], number)
        return value

    def train(loader, model, classifier, criterion, optimizer, epoch, cfg, logger, block=None):
        if epoch == 0:
            assert model.module.feature_tuning is False and not hasattr(model.module, "conv1d")
            assert type(optimizer) is torch.optim.SGD and len(optimizer.param_groups) == 2
            assert optimizer.state == {}
            status["initial_model_state_sha256"] = tensor_state_hash(model)
            status["initial_classifier_state_sha256"] = tensor_state_hash(classifier)
            status["optimizer_groups"] = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
            counts = np.bincount(loader.dataset.targets, minlength=100)
            assert counts.tolist() == preflight["class_counts"]
            status["train_dataset_confirmed"] = True
        context.update({"loss_sum": 0.0, "samples": 0, "steps": 0,
                        "loss_min": float("inf"), "loss_max": float("-inf")})
        started = time.perf_counter()
        real_train(loader, model, classifier, criterion, optimizer, epoch, cfg, logger, block)
        assert context["steps"] == 85 and context["samples"] == 10847
        status["epochs"].append({"epoch": epoch + 1, "official_zero_based_epoch": epoch,
            "lr": optimizer.param_groups[-1]["lr"], "train_seconds": time.perf_counter() - started,
            "train_loss": context["loss_sum"] / context["samples"],
            "train_loss_min": context["loss_min"], "train_loss_max": context["loss_max"],
            "finite_train_loss": True, "steps": context["steps"], "samples": context["samples"]})

    def validate(*args, **kwargs):
        started = time.perf_counter()
        metrics.metrics = None
        result = real_validate(*args, **kwargs)
        assert metrics.metrics is not None and np.isfinite(float(result[0]))
        status["epochs"][-1].update({"eval_metrics_percent_rounded_3dp": metrics.metrics,
            "eval_overall_percent": float(result[0]), "ece_percent": float(result[1]),
            "eval_seconds": time.perf_counter() - started})
        return result

    def save(state, is_best, model_dir):
        real_save(state, is_best, model_dir)
        status.update({"completed_epochs": int(state["epoch"]), "best_epoch": int(original.best_epoch + 1),
            "official_best_epoch_zero_based": int(original.best_epoch), "best_overall_percent": float(original.best_acc1),
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "gpu_global_peak_sampled_mib": monitor.peak_mib})
        status["epochs"][-1]["new_best"] = bool(is_best)
        dump(folder / "status.json", status)

    original.parse_args, original.create_logger = parse, create_logger
    original.train, original.validate = train, validate
    original.save_checkpoint, original.mixup_criterion = save, loss
    torch.load = observe_load
    # Python consumes '-u'; argparse must see the script name as argv[0].
    sys.argv = preflight["original_entry_command_argv"][2:]
    torch.cuda.reset_peak_memory_stats()
    monitor.thread.start()
    started = time.perf_counter()
    with (folder / "stdout_stderr.log").open("w", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                original.main()
                assert status["completed_epochs"] == 200 and not status["checkpoint_load_calls"]
                assert check_sources() == preflight["runtime_files"]
                assert all(sha(Path(status["code_dir"]) / p) == v for p, v in preflight["runtime_files"].items())
                ckps = Path(status["model_dir"])
                shutil.copy2(ckps / "current.pth.tar", ckps / "model_final.pth.tar")
                status["checkpoints_sha256"] = {p.name: sha(p) for p in ckps.glob("*.pth.tar")}
                status["source_snapshot_verified"] = True
                status["state"] = "completed"
            except BaseException:
                status["state"] = "failed"
                status["error"] = traceback.format_exc()
                traceback.print_exc()
                raise
            finally:
                torch.load = real_load
                monitor.stop.set()
                monitor.thread.join(timeout=7)
                status.update({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               "elapsed_seconds": time.perf_counter() - started,
                               "gpu_global_peak_sampled_mib": monitor.peak_mib})
                dump(folder / "status.json", status)
    print(f"STAGE I COMPLETE: 200 epochs; best={status['best_overall_percent']:.3f}% epoch={status['best_epoch']}; evidence={folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id)
    run_folder = BASE / "formal_runs" / args.run_id
    run_name = "cifar100_imb001_stage1_ce_mixup_" + args.run_id
    if args.action == "prepare":
        prepare(run_folder, run_name)
    else:
        run(run_folder, run_name)
