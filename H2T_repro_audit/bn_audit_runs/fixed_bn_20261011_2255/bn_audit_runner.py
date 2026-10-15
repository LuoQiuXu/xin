"""Fixed-BN paired-input control; calls the original training and H2T functions.
This is a mechanism control experiment, separate from the official reproduction.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torchvision

from bn_audit_helpers import check_sources as verify_sources, dump, sha, GpuMonitor, MetricsHandler, protected_files

SUITE = Path(__file__).resolve().parent
BASE = SUITE.parent.parent
RUNTIME = SUITE / "runtime"
CHANNEL_SEED = 123

def check_sources():
    return verify_sources(BASE, RUNTIME)
DATA = BASE.parent / "H2T/data/cifar100"
CFG = "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml"
STAGE1_EVIDENCE = BASE / "formal_runs/formal_20261011_1429_r2/best_evaluation.json"
STAGE1 = json.loads(STAGE1_EVIDENCE.read_text(encoding="utf-8"))["checkpoint"]
SEED = 123
sys.path.insert(0, str(RUNTIME))


def config_for(suite_id, arm):
    from utils import config, update_config
    cfg = config.clone()
    opts = ["data_path", str(DATA), "resume", STAGE1["path"],
            "name", f"cifar100_if100_stage2_{suite_id}_{arm}", "shift_bn", "False"]
    if arm == "dr":
        opts += ["H2T", "0.0"]
    update_config(cfg, argparse.Namespace(cfg=str(RUNTIME / CFG), opts=opts))
    assert cfg.num_epochs == 10 and cfg.batch_size == 128 and cfg.imb_factor == .01
    assert cfg.backbone == "resnet32_fe" and cfg.gpu is None and cfg.workers == 0
    assert cfg.loss.type == "CrossEntropyLoss" and not cfg.loss.CE.reweight_CE
    assert not cfg.mixup and cfg.alpha is None and not cfg.shift_bn
    assert not cfg.deterministic and not cfg.multiprocessing_distributed
    assert cfg.lr == .1 and cfg.weight_decay == 5e-4 and cfg.momentum == .9
    assert cfg.lr_scheduler.type == "cos" and cfg.lr_scheduler.lr_factor == .2
    assert cfg.sampler.type == "weighted sampler" and cfg.sampler.weighted_sampler.type == "balance"
    assert cfg.sampler.dual_sample.enable and cfg.sampler.dual_sample.type == "long-tailed"
    assert cfg.H2T == (0.0 if arm == "dr" else .25)
    return cfg, opts


def snapshot(module, buffers=False):
    return {k: v.detach().cpu().clone() for k, v in (module.named_buffers() if buffers else module.named_parameters())}


def delta(before, after):
    assert before.keys() == after.keys()
    changed = [k for k in before if before[k].numpy().tobytes() != after[k].contiguous().numpy().tobytes()]
    return {"changed_count": len(changed), "total_count": len(before),
            "max_abs_delta": max((float((before[k].double()-after[k].double()).abs().max()) for k in changed), default=0.),
            "changed_names": changed,
            "per_tensor_max_abs_delta": {k: float((before[k].double()-after[k].double()).abs().max()) for k in changed}}


def state_hash(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode())
        h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_hashes():
    return {"torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "torch_cuda": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()}


def prepare(folder, suite_id):
    from datasets.cifar100 import CIFAR100_LT
    from torchvision.datasets.utils import check_integrity
    assert folder == SUITE and not (folder/"preflight.json").exists(), "Use a new suite output directory"
    assert protected_files(BASE) == json.loads((folder/"protected_original_before.json").read_text(encoding="utf-8"))
    manifest = check_sources()
    assert sha(STAGE1["path"]) == STAGE1["sha256"] == "0cf3e67a87f2a8c45f4470fadc06c3d7c47796c691797d21ed6c2fa8c6b7033e"
    loaded = torch.load(STAGE1["path"], map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 198
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.ones(8, device="cuda").sum().item() == 8
    raw = torchvision.datasets.CIFAR100(str(DATA), train=True, download=False)
    checks = {filename: check_integrity(str(DATA/raw.base_folder/filename), md5)
              for filename, md5 in raw.train_list + raw.test_list + [(raw.meta["filename"], raw.meta["md5"])]}
    assert all(checks.values())
    cfg, _ = config_for(suite_id, "h2t")
    dataset = CIFAR100_LT(cfg, False, root=str(DATA), imb_factor=.01, batch_size=128, num_works=0)
    counts = np.bincount(dataset.train_instance.dataset.targets, minlength=100)
    prior = json.loads((BASE/"evidence/preflight.json").read_text(encoding="utf-8"))
    assert counts.tolist() == prior["class_counts"] and counts.sum() == 10847
    assert hashlib.sha256(dataset.train_instance.dataset.data.tobytes()).hexdigest() == prior["subset_data_sha256"]
    assert len(dataset.eval.dataset) == 10000
    assert np.all(np.bincount(dataset.eval.dataset.targets, minlength=100) == 100)
    configs = {arm: config_for(suite_id, arm)[0] for arm in ["dr", "h2t"]}
    a, b = configs["dr"].clone(), configs["h2t"].clone()
    a.name = b.name
    a.H2T = b.H2T
    assert a.dump() == b.dump(), "Unexpected algorithm/config difference"
    for cfg in configs.values():
        assert not list((RUNTIME/"saved").glob(cfg.name + "_*"))
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ["runtime_manifest.json", "source_manifest.json", "compatibility.diff"]:
        shutil.copy2(BASE/"evidence"/filename, folder/filename)
    shutil.copy2(STAGE1_EVIDENCE, folder/"stage1_reference.json")
    assert Path(__file__).parent == folder
    for arm, cfg in configs.items():
        target = folder/arm
        target.mkdir()
        (target/"original.yaml").write_bytes((RUNTIME/CFG).read_bytes())
        (target/"effective.yaml").write_text(cfg.dump(), encoding="utf-8")
    device = torch.cuda.get_device_properties(0)
    dump(folder/"preflight.json", {"prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite_id": suite_id, "python": sys.executable, "python_version": sys.version,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": device.name, "gpu_memory_bytes": device.total_memory,
        "data_path": str(DATA), "data_integrity": checks, "class_counts": counts.tolist(),
        "train_samples": 10847, "test_samples": 10000, "batches_per_epoch": 85,
        "stage1_checkpoint": STAGE1, "stage1_epoch": 198, "seed": SEED,
        "deterministic_yaml": False, "runtime_files": manifest,
        "launcher_sha256": sha(__file__), "helpers_sha256": sha(SUITE/"bn_audit_helpers.py"),
        "channel_seed": CHANNEL_SEED, "shift_bn": False, "is_official_reproduction": False,
        "independent_runtime": str(RUNTIME), "core_source_modifications": [],
        "control_differences_from_original": ["shift_bn=False", "isolated CPU channel RNG"],
        "only_arm_config_differences": ["H2T", "name"], "disk_free_bytes": shutil.disk_usage(RUNTIME).free})
    print("STAGE II PREFLIGHT PASS; same Stage I checksum; configs differ only by H2T/name", flush=True)


def run(folder, suite_id, arm):
    preflight = json.loads((folder/"preflight.json").read_text(encoding="utf-8"))
    assert check_sources() == preflight["runtime_files"]
    assert sha(__file__) == preflight["launcher_sha256"]
    assert sha(SUITE/"bn_audit_helpers.py") == preflight["helpers_sha256"]
    assert protected_files(BASE) == json.loads((folder/"protected_original_before.json").read_text(encoding="utf-8"))
    assert sha(STAGE1["path"]) == STAGE1["sha256"]
    evidence = folder/arm
    assert not (evidence/"status.json").exists()
    assert os.environ.get("PYTHONHASHSEED") == str(SEED)
    cfg, opts = config_for(suite_id, arm)
    assert not list((RUNTIME/"saved").glob(cfg.name + "_*"))
    os.chdir(RUNTIME)
    import train_stage2 as original
    status = {"state": "running", "arm": arm, "pid": os.getpid(), "run_name": cfg.name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "completed_epochs": 0,
        "checkpoint_loads": [], "epochs": [], "h2t_calls": 0, "seed": SEED,
        "deterministic_yaml": cfg.deterministic, "shift_bn": cfg.shift_bn,
        "H2T": cfg.H2T, "stage2_best_epoch": None, "stage2_best_overall": -1.}
    real_logger, real_parse = original.create_logger, original.parse_args
    real_train, real_validate, real_save, real_fuse = original.train, original.validate, original.save_checkpoint, original.H2T
    real_load = torch.load
    monitor, metrics = GpuMonitor(evidence), MetricsHandler()
    ctx = {"phase": "setup"}
    initial = {}
    channel_generator = torch.Generator(device="cpu").manual_seed(CHANNEL_SEED)
    status["channel_rng"] = {"seed": CHANNEL_SEED, "isolation": "torch.random.fork_rng(devices=[])",
        "initial_state_sha256": hashlib.sha256(channel_generator.get_state().numpy().tobytes()).hexdigest()}
    assert Path(original.__file__).resolve().is_relative_to(RUNTIME.resolve())
    import methods
    assert Path(methods.__file__).resolve().is_relative_to(RUNTIME.resolve())
    status["imported_source_files"] = {"train_stage2": original.__file__, "methods": methods.__file__}

    def parse():
        result = real_parse()
        assert original.config.dump() == cfg.dump()
        return result

    def logger(config, name):
        result = real_logger(config, name)
        result[0].addHandler(metrics)
        status.update({"model_dir": str(Path(result[1]).resolve()), "code_dir": str(Path(result[2]).resolve()),
            "official_log": str(Path(result[2]).resolve().parent/"logs"/(config.name+".txt"))})
        dump(evidence/"status.json", status)
        return result

    def load(path, *args, **kwargs):
        assert Path(path).resolve() == Path(STAGE1["path"]).resolve(), "Unexpected prior Stage II checkpoint"
        assert sha(path) == STAGE1["sha256"]
        value = real_load(path, *args, **kwargs)
        initial["loaded"] = value
        status["checkpoint_loads"].append({"path": str(path), "sha256": sha(path), "epoch": int(value["epoch"])})
        return value

    def fuse(x1, x2, rho=.3):
        assert arm == "h2t" and rho == .25 and not torch.is_grad_enabled()
        assert x1.shape[1:] == (64,8,8)
        before, donor = x1.clone(), x2.clone()
        ambient = torch.get_rng_state().clone()
        predictor = torch.Generator(device="cpu")
        predictor.set_state(channel_generator.get_state())
        chosen = torch.randperm(64, generator=predictor)[:16].to(x1.device)
        others = torch.tensor([i for i in range(64) if i not in chosen.cpu().tolist()],device=x1.device)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(channel_generator.get_state())
            value = real_fuse(x1,x2,rho)
            channel_generator.set_state(torch.get_rng_state())
        assert torch.equal(torch.get_rng_state(), ambient), "Channel RNG leaked into data/augmentation stream"
        assert torch.equal(channel_generator.get_state(),predictor.get_state()), "Unexpected channel RNG consumption"
        assert value.data_ptr() == x1.data_ptr() and torch.equal(value[:,chosen],donor[:,chosen])
        assert torch.equal(value[:,others],before[:,others]) and torch.equal(x2,donor)
        status["h2t_calls"] += 1
        ctx["channel_masks"].append(chosen.cpu().tolist())
        if status["h2t_calls"] == 1:
            status["fusion_verified"] = {"rho":rho,"replaced_channels":16,"retained_channels":48,
                "chosen_channels_first_call":chosen.cpu().tolist(),"feature_shape":list(value.shape),
                "same_mask_for_entire_batch":True,"inplace":True,"donor_unchanged":True,
                "ambient_cpu_rng_bitwise_unchanged_every_call":True,"checked_every_fusion_call":True}
        return value

    class LoaderObserver:
        def __init__(self, loader):
            self.loader = loader
        def __getattr__(self, name):
            return getattr(self.loader, name)
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            for step, (images, targets, meta) in enumerate(self.loader):
                ctx["targets"] = targets
                ctx["bs_hist"] += np.bincount(targets.numpy(), minlength=100)
                ctx["is_hist"] += np.bincount(meta["sample_label"].numpy(), minlength=100)
                item = {"batch": step, "bs_labels_sha256": hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
                    "is_labels_sha256": hashlib.sha256(meta["sample_label"].numpy().tobytes()).hexdigest(),
                    "cpu_rng_at_batch_ready_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
                if True:  # fingerprint every batch, including the last partial batch
                    item.update({"bs_images_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                        "is_images_sha256": hashlib.sha256(meta["sample_image"].numpy().tobytes()).hexdigest()})
                ctx["batch_fingerprints"].append(item)
                yield images, targets, meta

    def train(loader, model, classifier, criterion, optimizer, epoch, config, log, block=None):
        if epoch == 0:
            assert len(status["checkpoint_loads"]) == 1
            initial["parameters"], initial["buffers"], initial["classifier"] = snapshot(model), snapshot(model, True), snapshot(classifier)
            loaded = initial["loaded"]
            assert all(torch.equal(v, loaded["state_dict_model"][k].detach().cpu()) for k,v in (initial["parameters"]|initial["buffers"]).items())
            assert all(torch.equal(v, loaded["state_dict_classifier"][k].detach().cpu()) for k,v in initial["classifier"].items())
            assert model.module.feature_tuning is False and not hasattr(model.module, "conv1d")
            assert type(optimizer) is torch.optim.SGD and optimizer.state == {}
            assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in classifier.parameters()}
            assert len(optimizer.param_groups) == 1 and criterion.per_cls_weights is None
            status.update({"initial_backbone_equals_stage1": True, "initial_classifier_equals_stage1": True,
                "initial_backbone_parameters_sha256": state_hash(initial["parameters"]),
                "initial_backbone_buffers_sha256": state_hash(initial["buffers"]),
                "initial_classifier_sha256": state_hash(initial["classifier"]),
                "optimizer_groups": [{k:v for k,v in g.items() if k != "params"} for g in optimizer.param_groups],
                "backbone_requires_grad_parameter_count": sum(p.requires_grad for p in model.parameters()),
                "initial_bn_counters": {k:int(v) for k,v in initial["buffers"].items() if k.endswith("num_batches_tracked")},
                "core_sources_equal_audited_original": True})
        before_bn = snapshot(model, True)
        ctx.update({"phase": "train", "loss_sum": 0., "samples": 0, "steps": 0, "backbone_calls": 0,
            "loss_min": float("inf"), "loss_max": float("-inf"), "batch_fingerprints": [], "channel_masks": [],
            "bs_hist": np.zeros(100, dtype=np.int64), "is_hist": np.zeros(100, dtype=np.int64)})
        rng_start = rng_hashes()
        def loss_hook(_module, args, output):
            if ctx["phase"] != "train":
                return
            assert torch.equal(args[1].detach().cpu(), ctx["targets"]), "CE labels must remain BS"
            value = float(output.detach())
            assert np.isfinite(value)
            count = args[1].shape[0]
            ctx["loss_sum"] += value*count
            ctx["samples"] += count
            ctx["steps"] += 1
            ctx["loss_min"], ctx["loss_max"] = min(ctx["loss_min"],value), max(ctx["loss_max"],value)
        def backbone_hook(_module, _args):
            assert not torch.is_grad_enabled() and not model.training
            assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
            ctx["backbone_calls"] += 1
        loss_handle = criterion.register_forward_hook(loss_hook)
        backbone_handle = model.register_forward_pre_hook(backbone_hook)
        started = time.perf_counter()
        try:
            real_train(LoaderObserver(loader), model, classifier, criterion, optimizer, epoch, config, log, block)
        finally:
            loss_handle.remove()
            backbone_handle.remove()
            ctx["phase"] = "between"
        assert ctx["steps"] == 85 and ctx["samples"] == 10847
        assert ctx["backbone_calls"] == (85 if arm == "dr" else 170)
        assert all(p.grad is None for p in model.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in classifier.parameters())
        model_after, bn_after = snapshot(model), snapshot(model, True)
        backbone_delta = delta(initial["parameters"], model_after)
        assert backbone_delta["changed_count"] == 0
        bn_increment = {k: int(bn_after[k]-before_bn[k]) for k in before_bn if k.endswith("num_batches_tracked")}
        assert len(bn_increment) == 31 and all(v == 0 for v in bn_increment.values())
        assert state_hash(bn_after) == state_hash(initial["buffers"]), "BN buffers changed at the byte level"
        assert state_hash(model_after) == state_hash(initial["parameters"]), "Conv/BN affine parameters changed"
        assert ctx["is_hist"].tolist() == preflight["class_counts"]
        status["epochs"].append({"epoch": epoch+1, "official_zero_based_epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"], "train_seconds": time.perf_counter()-started,
            "steps": 85, "samples": 10847, "train_loss": ctx["loss_sum"]/ctx["samples"],
            "train_loss_min": ctx["loss_min"], "train_loss_max": ctx["loss_max"], "finite_train_loss": True,
            "backbone_forward_calls": ctx["backbone_calls"], "backbone_parameters_delta": backbone_delta,
            "classifier_delta_from_stage1": delta(initial["classifier"], snapshot(classifier)),
            "bn_delta_this_epoch": delta(before_bn, bn_after), "bn_count_increment": bn_increment,
            "backbone_gradients_absent": True, "classifier_gradients_finite": True, "ce_labels_equal_bs": True,
            "bs_histogram": ctx["bs_hist"].tolist(), "is_histogram": ctx["is_hist"].tolist(),
            "batch_fingerprints": ctx["batch_fingerprints"], "channel_masks": ctx["channel_masks"],
            "backbone_parameters_bitwise_equal_stage1": True, "bn_buffers_bitwise_equal_stage1": True,
            "backbone_parameters_sha256": state_hash(model_after), "bn_buffers_sha256": state_hash(bn_after),
            "rng_at_epoch_start": rng_start,
            "rng_at_epoch_end": rng_hashes()})

    def validate(loader, model, classifier, criterion, config, log, block=None):
        before = snapshot(model, True)
        metrics.metrics = None
        started = time.perf_counter()
        result = real_validate(loader, model, classifier, criterion, config, log, block)
        assert state_hash(before) == state_hash(snapshot(model, True)) == state_hash(initial["buffers"])
        assert metrics.metrics is not None
        status["epochs"][-1].update({"eval_metrics_percent_rounded_3dp": metrics.metrics,
            "eval_overall_percent": float(result[0]), "ece_percent": float(result[1]),
            "eval_seconds": time.perf_counter()-started, "bn_unchanged_during_eval": True})
        return result

    def save(state, is_best, model_dir):
        real_save(state, is_best, model_dir)
        current = status["epochs"][-1]
        if current["eval_overall_percent"] > status["stage2_best_overall"]:
            shutil.copy2(Path(model_dir)/"current.pth.tar", Path(model_dir)/"model_best_stage2.pth.tar")
            status["stage2_best_overall"], status["stage2_best_epoch"] = current["eval_overall_percent"], current["epoch"]
        current["official_new_best"] = bool(is_best)
        status.update({"completed_epochs": int(state["epoch"]), "official_best_epoch_zero_based": int(original.best_epoch),
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
            "gpu_global_peak_sampled_mib": monitor.peak_mib})
        dump(evidence/"status.json", status)

    original.parse_args, original.create_logger = parse, logger
    original.train, original.validate, original.save_checkpoint, original.H2T = train, validate, save, fuse
    torch.load = load
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    status["initial_rng_hashes"] = rng_hashes()
    status["backend_settings"] = {"cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    sys.argv = ["train_stage2.py", "--cfg", CFG] + opts
    status["original_entry_argv"] = sys.argv.copy()
    torch.cuda.reset_peak_memory_stats()
    monitor.thread.start()
    started = time.perf_counter()
    with (evidence/"stdout_stderr.log").open("w", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                original.main()
                assert status["completed_epochs"] == 10 and len(status["checkpoint_loads"]) == 1
                assert status["h2t_calls"] == (0 if arm == "dr" else 850)
                status["channel_rng"]["final_state_sha256"] = hashlib.sha256(channel_generator.get_state().numpy().tobytes()).hexdigest()
                assert protected_files(BASE) == json.loads((folder/"protected_original_before.json").read_text(encoding="utf-8"))
                status["original_artifacts_unchanged"] = True
                assert check_sources() == preflight["runtime_files"]
                assert sha(STAGE1["path"]) == STAGE1["sha256"]
                for relative, expected in preflight["runtime_files"].items():
                    target = Path(status["code_dir"])/relative
                    if not target.exists():
                        assert target.suffix.lower() in {".pdf", ".pptx"}
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(RUNTIME/relative, target)
                    assert sha(target) == expected
                ckps = Path(status["model_dir"])
                shutil.copy2(ckps/"current.pth.tar", ckps/"model_final.pth.tar")
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
                    "elapsed_seconds": time.perf_counter()-started,
                    "gpu_global_peak_sampled_mib": monitor.peak_mib})
                dump(evidence/"status.json", status)
    print(f"{arm.upper()} COMPLETE: 10 epochs; best epoch={status['stage2_best_epoch']}; best={status['stage2_best_overall']:.3f}%", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--arm", choices=["dr", "h2t"])
    args = parser.parse_args()
    assert all(c.isalnum() or c in "_-" for c in args.suite_id)
    folder = SUITE
    assert SUITE.name == args.suite_id
    if args.action == "prepare":
        prepare(folder, args.suite_id)
    else:
        assert args.arm
        run(folder, args.suite_id, args.arm)
