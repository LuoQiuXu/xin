"""Locked seed-42 Fish16 formal pilot. No smoke or TEST entry point."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch
from torch import nn

import data
import models
from amp_recovery import OptimizerStepCounter, guarded_scaler_step

BRANCH = "experiment/fish16-amp-scale-stability-20261102"
SEED = 42
ORDER = ("A0", "B0", "A1", "A2", "B1", "B2")
ROUTES = {"A1": ("A0", "crt"), "A2": ("A0", "h2t"),
          "B1": ("B0", "crt"), "B2": ("B0", "pi")}
SCHEMA = 1

# The previous formal run let the default GradScaler double its scale every
# 2,000 successful updates. A0 then failed at scale 262,144. Keep the original
# 65,536 starting scale, but prevent automatic growth during any one condition.
# The interval exceeds the maximum 90-epoch Stage-I attempts (90 * 1,017).
FORMAL_AMP_POLICY = {
    "device": "cuda",
    "enabled": True,
    "init_scale": 65536.0,
    "growth_factor": 2.0,
    "backoff_factor": 0.5,
    "growth_interval": 100000,
    "stage1_max_attempts": 90 * 1017,
    "stage2_max_attempts": 10 * 1017,
    "automatic_growth_during_condition": False,
}


def make_formal_grad_scaler():
    if not torch.cuda.is_available():
        raise RuntimeError("The Fish16 formal AMP policy requires CUDA.")
    return torch.amp.GradScaler(
        FORMAL_AMP_POLICY["device"],
        init_scale=FORMAL_AMP_POLICY["init_scale"],
        growth_factor=FORMAL_AMP_POLICY["growth_factor"],
        backoff_factor=FORMAL_AMP_POLICY["backoff_factor"],
        growth_interval=FORMAL_AMP_POLICY["growth_interval"],
        enabled=FORMAL_AMP_POLICY["enabled"],
    )


def stage_of(condition):
    if condition not in ORDER:
        raise ValueError("Unknown formal condition")
    return 1 if condition in ("A0", "B0") else 2


def epoch_budget(condition):
    return 90 if stage_of(condition) == 1 else 10


def stage2_source(condition):
    if condition not in ROUTES:
        raise ValueError("Only Stage-II conditions have a source checkpoint")
    return ROUTES[condition][0]


def epoch_lr(stage, epoch):
    """Zero-based epoch. Warmup peaks at epoch5; Stage-I cosine ends at epoch90."""
    budget = 90 if stage == 1 else 10 if stage == 2 else 0
    if not 0 <= epoch < budget:
        raise ValueError("Epoch outside fixed formal budget")
    if stage == 1 and epoch < 5:
        return .01 * (epoch + 1) / 5
    position = (epoch - 4) / 85 if stage == 1 else epoch / 10
    return .005 * (1 + math.cos(math.pi * position))


class EpochSchedule:
    def __init__(self, stage):
        if stage not in (1, 2):
            raise ValueError("Unknown stage scheduler")
        self.stage, self.next_epoch = stage, 0

    def apply(self, optimizer, epoch):
        if epoch != self.next_epoch:
            raise ValueError("Scheduler cannot restart or skip a committed epoch")
        lr = epoch_lr(self.stage, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def complete(self, epoch):
        if epoch != self.next_epoch:
            raise ValueError("Nonsequential epoch completion")
        self.next_epoch += 1

    def state_dict(self):
        return {"stage": self.stage, "next_epoch": self.next_epoch, "version": SCHEMA}

    def load_state_dict(self, state):
        if state["stage"] != self.stage or state["version"] != SCHEMA:
            raise ValueError("Scheduler protocol changed")
        budget = 90 if self.stage == 1 else 10
        if not 0 <= state["next_epoch"] <= budget:
            raise ValueError("Invalid scheduler cursor")
        self.next_epoch = state["next_epoch"]


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def json_normalize(value):
    """Normalize tuple/list and other JSON round-trip representations."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False).encode("utf-8"))


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp." + uuid.uuid4().hex)
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def capture_rng(cuda=True):
    name, values, position, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"name": name,
            "values": torch.from_numpy(values.astype(np.int64)), "position": position,
            "gaussian": gaussian, "cached": cached}, "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if cuda else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], n["values"].numpy().astype(np.uint32),
                         n["position"], n["gaussian"], n["cached"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_checkpoint_metadata(checkpoint, condition, protocol, final_source=False):
    required = {"model", "optimizer", "scaler", "scheduler", "rng", "records",
                "epoch_completed", "protocol_sha256", "formal_run", "smoke_only"}
    if not required <= checkpoint.keys():
        raise ValueError("Incomplete formal resume checkpoint")
    if (checkpoint["formal_run"] is not True or checkpoint["smoke_only"] is not False
            or checkpoint.get("seed") != SEED or checkpoint.get("condition") != condition
            or checkpoint.get("stage") != stage_of(condition)
            or checkpoint.get("epochs_total") != epoch_budget(condition)
            or checkpoint["protocol_sha256"] != protocol):
        raise ValueError("Checkpoint is not the matching formal seed-42 protocol")
    epoch = checkpoint["epoch_completed"]
    if not 0 <= epoch <= epoch_budget(condition):
        raise ValueError("Invalid completed-epoch cursor")
    if [r["epoch"] for r in checkpoint["records"]] != list(range(1, epoch + 1)):
        raise ValueError("Resume history has missing or repeated epochs")
    if checkpoint["scheduler"]["next_epoch"] != epoch:
        raise ValueError("Scheduler and checkpoint epoch disagree")
    if final_source and (stage_of(condition) != 1 or epoch != 90):
        raise ValueError("Stage-II requires the final formal Stage-I epoch90")


def checked_checkpoint(path, expected_sha):
    if not expected_sha or models.sha(path) != expected_sha:
        raise ValueError("Source checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if models.sha(path) != expected_sha:
        raise ValueError("Source checkpoint changed while loading")
    return checkpoint


def verify_pair_sources(sources):
    for a, b in (("A1", "A2"), ("B1", "B2")):
        if a in sources and b in sources and sources[a] != sources[b]:
            raise ValueError("Paired Stage-II source checkpoint hashes differ")


def require_split(rows, indices, split):
    if split not in ("train", "val"):
        raise ValueError("TEST is inaccessible to the formal runner")
    if any(rows[int(i)]["split_primary"] != split for i in indices):
        raise ValueError("Attempt to read an image outside the allowed train/validation split")


class SafeViews(data.Views):
    def __init__(self, rows, recipients, training, epoch=0, donors=None):
        self.allowed_split = "train" if training else "val"
        require_split(rows, recipients, self.allowed_split)
        if donors is not None:
            if not training:
                raise ValueError("Validation cannot have a donor stream")
            require_split(rows, donors, "train")
        super().__init__(rows, recipients, training, epoch, donors)

    def image(self, index, position, stream):
        require_split(self.rows, [index], self.allowed_split)
        return super().image(index, position, stream)


def metrics(logits, labels, groups):
    correct = logits.argmax(1) == labels
    per_class = [float(correct[labels == c].float().mean() * 100) for c in range(16)]
    if not all(math.isfinite(v) for v in per_class):
        raise ValueError("Missing validation class or nonfinite accuracy")
    return {"overall": float(correct.float().mean() * 100), "macro": sum(per_class)/16,
            "per_class": per_class, "worst_class": min(per_class),
            "group_image": {g: float(correct[torch.isin(labels, torch.tensor(ids))].float().mean()*100)
                            for g, ids in groups.items()},
            "group_macro": {g: sum(per_class[i] for i in ids)/len(ids)
                            for g, ids in groups.items()}}


@torch.no_grad()
def evaluate(model, rows, val, groups, epoch):
    model.eval()
    outputs, targets = [], []
    for item in data.loader(SafeViews(rows, val, False, epoch), 16):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(item["recipient"].cuda(non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite validation logits")
        outputs.append(logits.float().cpu())
        targets.append(item["label"])
    labels = torch.cat(targets)
    if len(labels) != 5396:
        raise ValueError("Validation sample count changed")
    return metrics(torch.cat(outputs), labels, groups)


def pif_values(model):
    return model.extractor.pif.fuse.weight.detach().cpu().flatten().tolist() if model.use_pif else []


def pif_gradient_signature(model):
    if not model.use_pif:
        return None
    gradient = model.extractor.pif.fuse.weight.grad
    if gradient is None:
        return {"missing": True}
    values = gradient.detach().cpu()
    if torch.isfinite(values).all():
        return None
    return {"missing": False, "shape": list(values.shape), "dtype": str(values.dtype),
            "nan_count": int(torch.isnan(values).sum()),
            "positive_inf_count": int(torch.isposinf(values).sum()),
            "negative_inf_count": int(torch.isneginf(values).sum()),
            "nonfinite_positions": (~torch.isfinite(values)).nonzero().tolist()}


def epoch_csv(path, records):
    if not records:
        return
    flat = []
    for r in records:
        row = {k: v for k, v in r.items() if k not in ("validation", "overflow_events")}
        row.update({"validation_"+k: v for k, v in r["validation"].items()})
        flat.append({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


class FormalRun:
    def __init__(self, args):
        self.args, self.output = args, Path(args.output_dir).resolve()
        self.owns_output = False
        self.context, self.active, self.source_cache = {}, None, {}
        self.state = {"status": "INITIALIZING", "scope": "FORMAL_SEED42_90_PLUS_10",
                      "completed_conditions": [], "conditions": {}, "current_condition": None,
                      "current_epoch": None, "test_evaluations": 0, "additional_seeds": [],
                      "formal_training_started": False, "resume_events": []}

    def status(self):
        atomic_json(self.output/"ACTUAL_EXECUTION_STATUS.json", self.state)

    def current_hashes(self):
        base = Path(__file__).parent
        return {n: models.sha(base/n) for n in ("formal_train.py", "models.py", "data.py", "amp_recovery.py")}

    def initialize(self):
        if self.output.is_relative_to(models.ROOT):
            raise ValueError("Formal runtime outputs must be outside Git")
        if self.args.resume:
            if not self.output.is_dir():
                raise ValueError("Resume directory is missing")
            if not (self.output/"FORMAL_RUN_MANIFEST.json").is_file():
                raise ValueError("Not an existing formal run")
            self.owns_output = True
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            self.owns_output = True
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=models.ROOT, text=True).strip()
        if branch != BRANCH:
            raise ValueError("Wrong formal branch")
        if models.sha(data.MANIFEST) != data.EXPECTED_SHA or models.sha(models.ORIGINAL_WEIGHT) != models.ORIGINAL_SHA:
            raise ValueError("Locked manifest or official initialization hash mismatch")
        expected = json.loads(Path(r"D:\h2t_cross_domain_results\fish16_validation_seed42_20261031_01\ORIGINAL_SOURCE_HASHES.json").read_text(encoding="utf-8"))
        if models.source_hashes() != expected:
            raise ValueError("Original H2T/PI-H2T source hash changed")
        self.hashes = self.current_hashes()
        cpu_path = Path(self.args.cpu_validation_record) if self.args.cpu_validation_record else None
        if self.args.resume and cpu_path is None:
            old = json.loads((self.output/"FORMAL_RUN_MANIFEST.json").read_text(encoding="utf-8"))
            cpu_path = Path(old["cpu_validation_record"])
        if cpu_path is None:
            raise ValueError("CPU validation record is required")
        cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
        if cpu["status"] != "PASS" or any(s["exit_code"] for s in cpu["checks"]):
            raise ValueError("Required CPU checks did not pass")
        if cpu["source_hashes"] != self.hashes:
            raise ValueError("Runner sources changed after CPU validation")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        environment = {"python_executable": sys.executable, "python": platform.python_version(),
                       "torch": str(torch.__version__), "cuda": torch.version.cuda,
                       "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0)}
        validated={"python":"3.12.12","torch":"2.14.1+cu132","cuda":"13.2",
                   "cudnn":92400,"gpu":"NVIDIA GeForce RTX 5070 Laptop GPU"}
        if any(environment[k]!=v for k,v in validated.items()):
            raise RuntimeError("Validated runtime changed: "+json.dumps(environment))
        protocol = {"schema": SCHEMA, "seed": SEED, "batch": 16, "workers": 4,
                    "stage1_epochs": 90, "stage2_epochs": 10, "momentum": .9,
                    "stage1_weight_decay": 1e-4, "stage2_weight_decay": 5e-4,
                    "stage1_lrs": [epoch_lr(1,e) for e in range(90)],
                    "stage2_lrs": [epoch_lr(2,e) for e in range(10)],
                    "source_hashes": self.hashes, "original_sources": expected,
                    "manifest_sha256": data.EXPECTED_SHA, "official_weight_sha256": models.ORIGINAL_SHA,
                    "numeric_flags": {"deterministic": True, "benchmark": False,
                                      "cudnn_tf32": False, "matmul_tf32": False},
                    "amp_scaler": FORMAL_AMP_POLICY,
                    "stage2_source_policy": "final_formal_stage1_epoch90",
                    "best_selection_metric": "validation_overall", "environment": environment}
        self.protocol = hashlib.sha256(json_bytes(protocol)).hexdigest()
        self.selected = list(ORDER) if self.args.condition == "all" else [self.args.condition]
        if self.args.condition != "all" and stage_of(self.args.condition) == 2:
            if not self.args.source_checkpoint or not self.args.source_sha256:
                raise ValueError("Independent Stage-II requires explicit final source checkpoint and SHA256")
        elif self.args.source_checkpoint or self.args.source_sha256:
            raise ValueError("External source is only accepted for independent Stage-II")
        self.rows = data.load_rows()
        self.train, self.val, self.stage1, self.rec, self.donors = data.schedules(self.rows)
        require_split(self.rows, self.train, "train")
        require_split(self.rows, self.val, "val")
        require_split(self.rows, self.rec, "train")
        require_split(self.rows, self.donors, "train")
        counts = {c: sum(self.rows[int(i)]["target"] == c for i in self.train) for c in range(16)}
        names = {self.rows[int(i)]["target"]: self.rows[int(i)]["label"] for i in self.train}
        ranked = sorted(range(16), key=lambda c: (-counts[c], names[c]))
        self.groups = {"Head": ranked[:6], "Medium": ranked[6:11], "Tail": ranked[11:]}
        if self.args.resume:
            old = json.loads((self.output/"FORMAL_RUN_MANIFEST.json").read_text(encoding="utf-8"))
            if old["protocol_sha256"] != self.protocol or old["selected_conditions"] != self.selected:
                raise ValueError("Resume protocol or condition selection changed")
            self.state = json.loads((self.output/"ACTUAL_EXECUTION_STATUS.json").read_text(encoding="utf-8"))
            self.state["resume_events"].append({"previous_status": self.state["status"],
                "previous_current_epoch": self.state.get("current_epoch"),
                "policy": "Restore last committed epoch checkpoint; explicitly replay any uncommitted partial epoch. No completed epoch restarts or skips."})
        else:
            np.savez_compressed(self.output/"sampling_plan.npz", train=self.train, val=self.val,
                                stage1=self.stage1, recipients=self.rec, donors=self.donors)
            atomic_json(self.output/"FORMAL_RUN_MANIFEST.json", {
                "protocol_sha256": self.protocol, "protocol": protocol,
                "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=models.ROOT, text=True).strip(),
                "branch": branch, "selected_conditions": self.selected,
                "cpu_validation_record": str(cpu_path), "groups": self.groups,
                "train_transform": str(data.transform(True)), "validation_transform": str(data.transform(False)),
                "sampling_plan_sha256": models.sha(self.output/"sampling_plan.npz"),
                "resume_granularity": "Atomic committed epoch; uncommitted partial epoch replay is explicitly logged, not silently counted twice.",
                "physical_attempt_logs": "Each epoch attempt has a separate exclusive JSONL; interrupted attempts remain local.",
                "epoch_seconds_scope": "Training plus validation and pairing/freeze checks; checkpoint serialization excluded.",
                "test_access": "Forbidden by SafeViews", "test_evaluations": 0})
        self.state["status"] = "RUNNING"
        self.status()
        self.summaries()

    def source(self, condition):
        src = stage2_source(condition)
        if self.args.condition != "all":
            path, digest = Path(self.args.source_checkpoint).resolve(), self.args.source_sha256
        else:
            state = self.state["conditions"][src]
            if state["status"] != "COMPLETE" or state["completed_epochs"] != 90:
                raise ValueError("Stage-I source not actually complete")
            path, digest = self.output/src/"epoch_090.pth", state["final_checkpoint_sha256"]
        if models.sha(path) != digest:
            raise ValueError("Final source checkpoint changed")
        key = str(path), digest
        if key not in self.source_cache:
            cp = checked_checkpoint(path, digest)
            validate_checkpoint_metadata(cp, src, self.protocol, final_source=True)
            self.source_cache[key] = cp["model"]
        sources = {k: v["source_checkpoint_sha256"] for k,v in self.state["conditions"].items()
                   if stage_of(k) == 2}
        sources[condition] = digest
        verify_pair_sources(sources)
        return self.source_cache[key], str(path), digest

    def check_pairing(self, condition, epoch, record):
        peer = "A0" if condition == "B0" else "A1" if stage_of(condition) == 2 and condition != "A1" else None
        if peer and peer in self.state["conditions"]:
            history = self.state["conditions"][peer].get("records", [])
            if epoch < len(history):
                fields = ["recipient_control_sha256"] + (["mixup_control_sha256"] if condition == "B0" else [])
                if any(record[f] != history[epoch][f] for f in fields):
                    raise ValueError("Paired input/MixUp controls differ")
        if condition == "B2" and "A2" in self.state["conditions"]:
            if record["donor_control_sha256"] != self.state["conditions"]["A2"]["records"][epoch]["donor_control_sha256"]:
                raise ValueError("Paired fusion donor views differ")

    def train_epoch(self, name, model, optimizer, scaler, counter, epoch, pending, consecutive, lr):
        stage = stage_of(name)
        model.train() if stage == 1 else model.freeze()
        if stage == 2 and any(m.training for m in model.extractor.modules() if isinstance(m, nn.BatchNorm2d)):
            raise ValueError("Stage-II BN was not frozen")
        indices = self.stage1 if stage == 1 else self.rec
        donors = self.donors if name in ("A2", "B2") else None
        batches = data.loader(SafeViews(self.rows, indices, True, epoch, donors), 16)
        attempt = 1
        while (self.output/name/f"epoch_{epoch+1:03d}_attempt_{attempt:03d}.jsonl").exists():
            attempt += 1
        logfile = self.output/name/f"epoch_{epoch+1:03d}_attempt_{attempt:03d}.jsonl"
        recipient_digest, mixup_digest, donor_digest = (hashlib.sha256() for _ in range(3))
        samples, attempted, skips, consecutive_max = 0, 0, 0, consecutive
        loss_sum, start, steps_start = 0., time.perf_counter(), counter.count
        overflows = []
        initial_scale = float(scaler.get_scale())
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        named = list(model.named_parameters()) if stage == 1 else [("classifier."+n,p) for n,p in model.classifier.named_parameters()]
        with logfile.open("x", encoding="utf-8") as handle:
            for batch, item in enumerate(batches):
                self.context = {"condition":name,"epoch":epoch+1,"batch":batch,
                    "samples_before_window":samples,"attempted_before_window":attempted,
                    "actual_epoch_updates_before_window":counter.count-steps_start,
                    "process_optimizer_calls_before_window":counter.count,
                    "scale_before_unscale":float(scaler.get_scale())}
                self.active = {"model":model,"optimizer":optimizer,"scaler":scaler,
                    "step_counter":counter,"input":item["recipient"],"targets":item["label"]}
                targets = item["label"].cuda(non_blocking=True)
                x = item["recipient"].cuda(non_blocking=True)
                control = {"batch": batch, "samples": len(targets),
                           "recipient_sha256": tensor_sha(item["recipient"]), "labels_sha256": tensor_sha(item["label"])}
                recipient_digest.update(json_bytes(control))
                target_a = target_b = None
                lam = None
                if stage == 1:
                    control["mixup_rng_sha256"] = tensor_sha(torch.get_rng_state())
                    x, target_a, target_b, lam = models.methods.mixup_data(x, targets, alpha=1.)
                    control["mixup_lambda"] = float(lam)
                    mixup_digest.update(json_bytes(control))
                    self.active.update(input=x,targets=targets,target_a=target_a,target_b=target_b,mixup_lambda=lam,control=control)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(x)
                        loss = lam*nn.functional.cross_entropy(logits,target_a)+(1-lam)*nn.functional.cross_entropy(logits,target_b)
                else:
                    self.active.update(input=x,targets=targets,control=control)
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                        r = model.extractor(x)
                        d = model.extractor(item["donor"].cuda(non_blocking=True)) if donors is not None else None
                        features, extra = models.stage2_features(r,d,targets,model.classifier,ROUTES[name][1])
                    if donors is not None:
                        donor = {"batch":batch,"samples":len(targets),"donor_sha256":tensor_sha(item["donor"])}
                        donor_digest.update(json_bytes(donor))
                        control.update(donor)
                        if name == "A2":
                            if len(extra) != 512: raise ValueError("H2T channel count changed")
                            control["h2t_channels"] = extra.tolist()
                        else:
                            if extra.requires_grad or not torch.isfinite(extra).all() or not ((extra>=0)&(extra<=1)).all():
                                raise ValueError("Invalid detached PI-H2TF recipient mass")
                            control["pi_recipient_mass"] = extra.float().cpu().tolist()
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model.classifier(features)
                        loss = nn.functional.cross_entropy(logits,targets)
                self.active.update(input=x,targets=targets,target_a=target_a,target_b=target_b,
                                   mixup_lambda=lam,logits=logits,loss=loss,control=control)
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite training forward")
                window_size = min(16, len(indices)-batch*16)
                scaler.scale(loss * len(targets) / window_size).backward()
                scaler.unscale_(optimizer)
                signature = pif_gradient_signature(model)
                if stage == 2 and any(p.grad is not None for p in model.extractor.parameters()):
                    raise ValueError("Frozen Stage-II extractor received gradients")
                attempted += 1
                outcome = guarded_scaler_step(optimizer=optimizer,scaler=scaler,named_parameters=named,
                    step_counter=counter,allow_pif_b_overflow=(name=="B0"),
                    consecutive_overflow_skips=consecutive,max_consecutive_overflow_skips=2)
                event = {**outcome.as_dict(),"condition":name,"epoch":epoch+1,"batch":batch,
                         "pif_gradient_signature":signature}
                if pending is not None:
                    if outcome.skipped or outcome.scale_before != pending["scale_after"]:
                        raise ValueError("PIF overflow not followed by a real update at the reduced scale")
                    pending = None
                if outcome.skipped:
                    skips += 1
                    consecutive += 1
                    consecutive_max = max(consecutive_max,consecutive)
                    pending = {"epoch":epoch+1,"batch":batch,"scale_after":outcome.scale_after}
                    overflows.append(event)
                    print(f"AMP_SKIP {name} epoch={epoch+1} batch={batch} {outcome.scale_before}->{outcome.scale_after} {signature}",flush=True)
                else:
                    consecutive = 0
                samples += len(targets)
                loss_sum += float(loss.detach())*len(targets)
                handle.write(json.dumps({"control":control,"event":event,"loss":float(loss.detach())},ensure_ascii=False,allow_nan=False)+"\n")
                handle.flush()
                optimizer.zero_grad(set_to_none=True)
                self.active = {"model":model,"optimizer":optimizer,"scaler":scaler,"step_counter":counter}
                if (batch+1)%100 == 0:
                    self.state["progress"] = {**self.context,"completed_batches":batch+1,"samples":samples,
                                              "actual_epoch_updates":counter.count-steps_start,"amp_skips":skips}
                    self.status()
                    print(f"{name} epoch={epoch+1}/{epoch_budget(name)} batch={batch+1}/1017 lr={lr:.9g} loss={float(loss.detach()):.5f}",flush=True)
        torch.cuda.synchronize()
        duration = time.perf_counter()-start
        updates = counter.count-steps_start
        if samples != 16269 or attempted != 1017 or updates+skips != attempted:
            raise ValueError("Incomplete formal epoch/update accounting")
        if name != "B0" and skips:
            raise ValueError("Unexpected optimizer skip outside B0")
        return {"epoch":epoch+1,"lr":lr,"samples":samples,"attempted_optimizer_windows":attempted,
                "actual_optimizer_updates":updates,"amp_skips":skips,"gradient_audits":attempted,
                "max_consecutive_skips":consecutive_max,"loss_scale_final":float(scaler.get_scale()),
                "loss_scale_initial":initial_scale,
                "train_loss":loss_sum/samples,"train_seconds":duration,"images_per_second":samples/duration,
                "recipient_control_sha256":recipient_digest.hexdigest(),
                "mixup_control_sha256":mixup_digest.hexdigest() if stage==1 else None,
                "donor_control_sha256":donor_digest.hexdigest() if donors is not None else None,
                "attempt_log":logfile.name,"overflow_events":overflows},pending,consecutive

    def condition(self, name):
        stage, budget = stage_of(name), epoch_budget(name)
        folder = self.output/name
        prior = self.state["conditions"].get(name)
        if prior and prior["status"] == "COMPLETE":
            if prior["completed_epochs"] != budget or models.sha(folder/f"epoch_{budget:03d}.pth") != prior["final_checkpoint_sha256"]:
                raise ValueError("Completed-condition evidence changed")
            return
        self.context={"condition":name,"phase":"condition_initialization"}
        self.state.update(current_condition=name,current_epoch=None)
        self.status()
        folder.mkdir(exist_ok=self.args.resume)
        model = models.FishModel(name.startswith("B"))
        source_path, source_sha = None, None
        if stage == 2:
            state, source_path, source_sha = self.source(name)
            model.load_state_dict(state,strict=True)
        initial = {"backbone":models.state_sha(model.extractor.backbone),
                   "classifier":models.state_sha(model.classifier),"extractor":models.state_sha(model.extractor)}
        source_condition = "A0" if name=="B0" else "A1" if name=="A2" else "B1" if name=="B2" else None
        if source_condition in self.state["conditions"]:
            reference = self.state["conditions"][source_condition]["initialization"]
            if any(initial[k]!=reference[k] for k in (["backbone","classifier"] if name=="B0" else ["extractor","classifier"])):
                raise ValueError("Paired common initialization mismatch")
        seed_all()
        model.cuda()
        model.train() if stage==1 else model.freeze()
        parameters = list(model.parameters()) if stage==1 else list(model.classifier.parameters())
        optimizer = torch.optim.SGD(parameters,lr=epoch_lr(stage,0),momentum=.9,weight_decay=1e-4 if stage==1 else 5e-4)
        scaler, scheduler = make_formal_grad_scaler(), EpochSchedule(stage)
        records, pending, consecutive, best = [], None, 0, {"overall":-1.,"epoch":None}
        update_offset = 0
        last = folder/"last.pth"
        if self.args.resume and last.is_file():
            cp = torch.load(last,map_location="cpu",weights_only=True)
            validate_checkpoint_metadata(cp,name,self.protocol)
            if cp["source_checkpoint_sha256"] != source_sha:
                raise ValueError("Resume Stage-II source changed")
            model.load_state_dict(cp["model"],strict=True)
            optimizer.load_state_dict(cp["optimizer"])
            scaler.load_state_dict(cp["scaler"])
            if scaler.get_growth_interval() != FORMAL_AMP_POLICY["growth_interval"]:
                raise ValueError("Resume checkpoint uses a different formal AMP growth policy")
            scheduler.load_state_dict(cp["scheduler"])
            records,pending,consecutive,best = cp["records"],cp["pending_overflow"],cp["consecutive_skips"],cp["best"]
            update_offset=cp["actual_updates_total"]
            restore_rng(cp["rng"])
            print(f"RESUME {name} committed_epochs={len(records)}; next_epoch={len(records)+1}; any uncommitted partial attempt is explicitly replayed",flush=True)
        elif prior and prior.get("completed_epochs",0):
            raise ValueError("Committed epochs exist without their resumable checkpoint")
        state = {"status":"RUNNING","completed_epochs":len(records),"epochs_total":budget,
                 "records":records,"initialization":initial,"source_checkpoint":source_path,
                 "source_checkpoint_sha256":source_sha,"best":best,
                 "actual_optimizer_updates":sum(r["actual_optimizer_updates"] for r in records),
                 "amp_skips":sum(r["amp_skips"] for r in records)}
        self.state["conditions"][name]=state
        if not (folder/"CONDITION_MANIFEST.json").exists():
            atomic_json(folder/"CONDITION_MANIFEST.json",{"condition":name,"stage":stage,"seed":SEED,
                "epochs":budget,"protocol_sha256":self.protocol,"initialization":initial,
                "source_checkpoint":source_path,"source_checkpoint_sha256":source_sha,
                "per_epoch_lr":[epoch_lr(stage,e) for e in range(budget)]})
        counter=OptimizerStepCounter(optimizer)
        self.active={"model":model,"optimizer":optimizer,"scaler":scaler,"step_counter":counter}

        def checkpoint():
            return {"schema":SCHEMA,"formal_run":True,"smoke_only":False,"condition":name,"stage":stage,
                "seed":SEED,"epoch":len(records),"epoch_completed":len(records),"epochs_total":budget,"protocol_sha256":self.protocol,
                "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                "scheduler":scheduler.state_dict(),"rng":capture_rng(),"records":records,"pending_overflow":pending,
                "consecutive_skips":consecutive,"best":best,"actual_updates_total":update_offset+counter.count,
                "source_checkpoint_sha256":source_sha}
        if not last.exists():
            atomic_checkpoint(last,checkpoint())
        epoch_csv(folder/"epoch_metrics.csv",records)
        for record in records:
            path=folder/f"epoch_{record['epoch']:03d}_metrics.json"
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != json_normalize(record):
                    raise ValueError("Committed epoch metric differs from resume checkpoint")
            else:
                atomic_json(path,record)
        for epoch in range(scheduler.next_epoch,budget):
            if self.current_hashes()!=self.hashes:
                raise ValueError("Training source changed during execution")
            self.state.update(current_condition=name,current_epoch=epoch+1,formal_training_started=True)
            self.status()
            lr=scheduler.apply(optimizer,epoch)
            epoch_start=time.perf_counter()
            record,pending,consecutive=self.train_epoch(name,model,optimizer,scaler,counter,epoch,pending,consecutive,lr)
            self.context={"condition":name,"epoch":epoch+1,"phase":"validation"}
            validation_start=time.perf_counter()
            record["validation"]=evaluate(model,self.rows,self.val,self.groups,epoch)
            record["validation_seconds"]=time.perf_counter()-validation_start
            record["peak_allocated_mib"]=torch.cuda.max_memory_allocated()/1024**2
            record["pif_values"]=pif_values(model)
            if stage==2 and models.state_sha(model.extractor)!=initial["extractor"]:
                raise ValueError("Stage-II extractor or BN buffers mutated")
            self.check_pairing(name,epoch,record)
            record["epoch_seconds"]=time.perf_counter()-epoch_start
            self.context={"condition":name,"epoch":epoch+1,"phase":"epoch_checkpoint_commit"}
            is_best=record["validation"]["overall"]>best["overall"]
            if is_best:best={"overall":record["validation"]["overall"],"epoch":epoch+1}
            records=records+[record]
            scheduler.complete(epoch)
            atomic_checkpoint(last,checkpoint())
            state.update(completed_epochs=len(records),records=records,best=best,
                         actual_optimizer_updates=sum(r["actual_optimizer_updates"] for r in records),
                         amp_skips=sum(r["amp_skips"] for r in records))
            self.status()
            if is_best:atomic_copy(last,folder/"best_validation.pth")
            epoch_csv(folder/"epoch_metrics.csv",records)
            atomic_json(folder/f"epoch_{epoch+1:03d}_metrics.json",record)
            self.state.pop("progress",None)
            self.status()
            self.summaries()
            print(f"EPOCH_COMPLETE {name} {epoch+1}/{budget} loss={record['train_loss']:.6f} val={record['validation']['overall']:.4f} macro={record['validation']['macro']:.4f} updates={record['actual_optimizer_updates']} skips={record['amp_skips']}",flush=True)
        if pending is not None:
            raise ValueError("Final overflow has no subsequent successful optimizer window")
        final=folder/f"epoch_{budget:03d}.pth"
        atomic_copy(last,final)
        state.update(status="COMPLETE",completed_epochs=budget,final_checkpoint_sha256=models.sha(final))
        self.state["completed_conditions"]=[n for n in ORDER if self.state["conditions"].get(n,{}).get("status")=="COMPLETE"]
        self.state.update(current_condition=None,current_epoch=None)
        self.status()
        self.summaries()
        counter.close()
        self.active=None
        model.cpu()
        del model,optimizer,scaler
        torch.cuda.empty_cache()

    def summaries(self):
        rows=[]
        for name in ORDER:
            s=self.state["conditions"].get(name,{})
            records=s.get("records",[])
            last=records[-1] if records else None
            rows.append({"condition":name,"status":s.get("status","NOT_STARTED"),"completed_epochs":len(records),
                "required_epochs":epoch_budget(name),"validation_overall":last["validation"]["overall"] if last else None,
                "validation_macro":last["validation"]["macro"] if last else None,
                "validation_tail_image":last["validation"]["group_image"]["Tail"] if last else None,
                "actual_optimizer_updates":sum(r["actual_optimizer_updates"] for r in records),
                "attempted_windows":sum(r["attempted_optimizer_windows"] for r in records),
                "amp_skips":sum(r["amp_skips"] for r in records),"source_checkpoint_sha256":s.get("source_checkpoint_sha256")})
        stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        atomic_bytes(self.output/"RESULTS_SUMMARY.csv",stream.getvalue().encode("utf-8"))
        lines=["# Seed-42 formal Fish16 pilot", "", f"Actual status: {self.state['status']}", "",
               "| Condition | Actual epochs | Actual updates | AMP skips | Validation overall |", "|---|---:|---:|---:|---:|"]
        for r in rows:lines.append(f"| {r['condition']} | {r['completed_epochs']}/{r['required_epochs']} | {r['actual_optimizer_updates']} | {r['amp_skips']} | {r['validation_overall']} |")
        for a,b in (("A2","A1"),("B2","B1")):
            sa,sb=self.state["conditions"].get(a,{}),self.state["conditions"].get(b,{})
            if sa.get("status")==sb.get("status")=="COMPLETE":
                va,vb=sa["records"][-1]["validation"],sb["records"][-1]["validation"]
                lines.append(f"\n{a}-{b}: overall={va['overall']-vb['overall']:.6f} pp; macro={va['macro']-vb['macro']:.6f} pp; tail-image={va['group_image']['Tail']-vb['group_image']['Tail']:.6f} pp")
        lines += ["", "Validation only; TEST evaluations=0. One seed is not a cross-domain generalization claim.",
                  "Attempt JSONLs retain physical calls, including any interrupted/uncommitted attempts; epoch totals describe committed checkpoint lineage."]
        atomic_bytes(self.output/"RESULTS_SUMMARY.md",("\n".join(lines)+"\n").encode("utf-8"))

    def output_index(self):
        rows=[]
        for p in sorted(self.output.rglob("*")):
            if p.is_file() and p.name!="OUTPUT_SHA256.csv":
                rows.append({"relative_path":p.relative_to(self.output).as_posix(),"bytes":p.stat().st_size,"sha256":models.sha(p)})
        stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=["relative_path","bytes","sha256"]);writer.writeheader();writer.writerows(rows)
        atomic_bytes(self.output/"OUTPUT_SHA256.csv",stream.getvalue().encode("utf-8"))

    def execute(self):
        try:
            self.initialize()
            for condition in self.selected:self.condition(condition)
            self.state.update(status="COMPLETE_FORMAL_SELECTED_CONDITIONS",current_condition=None,current_epoch=None)
            self.status();self.summaries();self.output_index()
            print("FORMAL_RUN_COMPLETE; TEST_EVALUATIONS=0; ADDITIONAL_SEEDS=[]",flush=True)
        except BaseException as exc:
            if self.owns_output:
                snapshot_error=None
                try:
                    if self.active:
                        a=self.active
                        tensor_or_none=lambda key:a[key].detach().cpu() if torch.is_tensor(a.get(key)) else None
                        atomic_checkpoint(self.output/"FAILURE_SNAPSHOT.pt",{
                            "context":self.context,"model":a["model"].state_dict(),
                            "optimizer":a["optimizer"].state_dict(),"scaler":a["scaler"].state_dict(),
                            "rng":capture_rng(),"mixed_input":tensor_or_none("input"),
                            "targets":tensor_or_none("targets"),"target_a":tensor_or_none("target_a"),"target_b":tensor_or_none("target_b"),
                            "mixup_lambda":a.get("mixup_lambda"),"logits":tensor_or_none("logits"),
                            "loss":tensor_or_none("loss"),"control":a.get("control"),
                            "process_optimizer_calls_at_failure":a["step_counter"].count,
                            "parameter_gradients":{n:p.grad.detach().cpu() for n,p in a["model"].named_parameters() if p.grad is not None}})
                except BaseException as snapshot_exc:snapshot_error=repr(snapshot_exc)
                self.state.update(status="FAILED_STOPPED",failure_type=type(exc).__name__,failure_message=str(exc),failure_context=self.context)
                self.status();self.summaries()
                atomic_json(self.output/"FAILURE.json",{"error_type":type(exc).__name__,"error":str(exc),
                    "traceback":traceback.format_exc(),"context":self.context,
                    "gradient_diagnostics":getattr(exc,"diagnostics",{}),"snapshot_error":snapshot_error,
                    "test_evaluations":0,"additional_seeds":[],"automatic_retry":False})
                self.output_index()
            raise


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--condition",choices=("all",)+ORDER,default="all")
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--source-sha256")
    parser.add_argument("--cpu-validation-record")
    parser.add_argument("--resume",action="store_true")
    return parser.parse_args()


if __name__=="__main__":
    FormalRun(parse_args()).execute()
