"""Crossed Stage-II ONLY: existing clean/noisy Stage-I x opposite-label Stage-II.

Reuses the frozen author train core from the previous audited noisy H2T screen.
No Stage-I training, no algorithm changes, no official TEST or best-epoch selection.
Run from the repository root on the same machine with original local checkpoints.
"""
import argparse
import csv
import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
SOURCE = ROOT / "H2T_repro_audit/noisy_lt_failure_screen"
SRC_OUT = SOURCE / "results/cifar100n_lt_seed42_20261029_01"
SRC_CACHE = SRC_OUT / "primary_cache"
OUT = THIS / "results/cross_stage2_01"
LOCAL = OUT / "local_only"
sys.path.insert(0, str(SOURCE))

from runtime import (Core, ObservedDataset, AuditedLoader, AuditedSGD,
                     config, seed, models, transform, state_hash, cpu_state)
from train import (read, sha, save_weights, load_weights, counters, bn_state,
                   parameter_hash)

GROUPS = ("Overall", "Many", "Medium", "Few")


def safe_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def name(rep, labels, seed2, method):
    if rep not in ("clean", "noisy") or labels not in ("clean", "noisy") or rep == labels:
        raise ValueError("Only missing crossed cells allowed, not diagonal re-runs")
    return f"repr_{rep}_labels_{labels}_seed{seed2}_{method}"


def source_proof(rep):
    assert read(SRC_OUT / "VALIDATION.json")["passed"]
    orig = read(SRC_OUT / "stage1_" + rep / "RUN.json")
    # The previous experiment's Stage-I models are immutable input objects.
    parent = ROOT / orig["checkpoint"]["file"]
    assert parent.exists(), f"Missing source weight: {parent}"
    assert sha(parent) == orig["checkpoint"]["sha256"]
    assert orig["checkpoint"]["epoch"] == 200
    assert sha(SRC_OUT / "protocol" / ("clean_train.npz")) == read(SRC_OUT / "TRAINING_LOCK.json")["regime_label_sha256"]["clean"]
    assert sha(SRC_OUT / "protocol" / ("noisy_train.npz")) == read(SRC_OUT / "TRAINING_LOCK.json")["regime_label_sha256"]["noisy"]
    return parent, orig


def run(rep, labels, seed2, method):
    tag = name(rep, labels, seed2, method)
    if seed2 not in (124, 125) or method not in ("no_fusion", "h2t"):
        raise ValueError("Unsupported arm")
    parent, src = source_proof(rep)
    runpath = OUT / tag
    runpath.mkdir(parents=True, exist_ok=False)
    local = LOCAL / tag
    local.mkdir(parents=True, exist_ok=False)
    cfg = config(2)
    seed(seed2, cfg)
    bb, fc = models()
    loaded = load_weights(parent, bb, fc)
    assert loaded["epoch"] == 200
    for p in bb.parameters():
        p.requires_grad_(False)
    init = dict(bb_state_sha=state_hash(cpu_state(bb)),
                fc_state_sha=state_hash(cpu_state(fc)),
                bb_parameters_sha=parameter_hash(bb),
                BN_counters=counters(bb))
    ds = ObservedDataset(labels, cfg)  # Reads existing observed-label pools, not clean oracle
    loader = AuditedLoader(ds, 2)
    optimizer = AuditedSGD([{"params": fc.parameters()}], cfg.lr,
                           momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    assert not optimizer.state
    baseline = None
    if method == "h2t":
        baseline_tag = name(rep, labels, seed2, "no_fusion")
        nf = read(OUT / baseline_tag / "RUN.json")
        assert nf["source_checkpoint_sha"] == sha(parent) and nf["initial"] == init
        assert nf["labels_file_sha"] == sha(SRC_OUT / "protocol" / (labels + "_train.npz"))
        with (LOCAL / baseline_tag / "batch_controls.csv").open(newline="", encoding="utf8") as f:
            baseline = list(csv.DictReader(f))
        assert len(baseline) == 850
    rows = []
    all_controls = []
    with (runpath / "native_train.log").open("x", encoding="utf8") as logfile:
        core = Core(2, cfg, loader, method, logfile)
        for epoch in range(10):
            s = core.epoch_run(optimizer, epoch, bb, fc, cfg, baseline)
            rows.append({"epoch": epoch + 1, **s})
            assert len(loader.rows) == 85
            assert optimizer.updates == 85 * (epoch + 1)
            assert all(now - init["BN_counters"][key] == 170 * (epoch+1)
                       for key, now in counters(bb).items())
            assert parameter_hash(bb) == init["bb_parameters_sha"]
            all_controls.extend(loader.rows)
            print(tag, "epoch", epoch+1, "loss", round(float(s["Loss"]), 4), flush=True)
    assert len(all_controls) == 850
    assert core.replacements == (850 if method == "h2t" else 0)
    with (local / "batch_controls.csv").open("x", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_controls[0]))
        writer.writeheader()
        writer.writerows(all_controls)
    with (runpath / "epochs.csv").open("x", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bn_hash = state_hash(bn_state(bb))
    if method == "h2t":
        assert bn_hash == nf["BN_final_sha"], "paired BN shift not identical"
    checkpoint = save_weights(local / "epoch10.pth", bb, fc, 10)
    safe_json(runpath / "RUN.json", dict(
        representation_regime=rep, observed_label_regime=labels,
        stage1_seed=42, stage2_seed=seed2, method=method,
        initial=init, source_checkpoint_sha=sha(parent),
        source_checkpoint_path=str(parent.relative_to(ROOT)),
        labels_file_sha=sha(SRC_OUT / "protocol" / (labels + "_train.npz")),
        checkpoint=checkpoint, BN_final_sha=bn_hash,
        BB_parameters_frozen=(parameter_hash(bb) == init["bb_parameters_sha"]),
        paired_all_850_steps=(baseline is not None),
        H2T_replacements=core.replacements, optimizer_updates=optimizer.updates,
        controls_file_sha=sha(local / "batch_controls.csv"),
        nonoracle_note="Only observed labels used for sampler/targets; representation checkpoint is a preexisting controlled diagnostic input."
    ))


def evaluate():
    # Cannot evaluate until ALL eight crossed arms finish.
    tags = [name(r, l, s, m) for r,l in (("clean", "noisy"), ("noisy", "clean"))
            for s in (124, 125) for m in ("no_fusion", "h2t")]
    runs = {tag: read(OUT / tag / "RUN.json") for tag in tags}
    checks = {}
    for tag, runinfo in runs.items():
        cp = ROOT / runinfo["checkpoint"]["file"]
        assert cp.exists() and sha(cp) == runinfo["checkpoint"]["sha256"]
        checks[tag] = runinfo["checkpoint"]["sha256"]
    safe_json(OUT / "EVAL_LOCK.json", {"checkpoints": checks, "all_8_complete_before_eval": True})
    with np.load(SRC_OUT / "protocol/clean_validation.npz", allow_pickle=False) as f:
        ids, targets = f["ids"].copy(), f["labels"].copy()
    images = np.load(SRC_CACHE / "inputs/validation_images.npy", mmap_mode="r", allow_pickle=False)
    assert ids.shape == targets.shape == (1980,) and images.shape == (1980,32,32,3)
    assert len(np.unique(ids)) == 1980 and np.array_equal(np.bincount(targets, minlength=100), [0]+[20]*99)
    counts = np.asarray(read(SRC_OUT/"protocol/DATA_AUDIT.json")["true_lt_counts"])
    mask = dict(Overall=np.ones(1980, bool), Many=counts[targets]>100,
                Medium=(counts[targets]>=20)&(counts[targets]<=100), Few=counts[targets]<20)
    assert [sum(mask[g]) for g in GROUPS] == [1980,680,700,600]
    rows, perclass = [], []
    for tag in tags:
        rec = runs[tag]
        cfg = config(2)
        seed(rec["stage2_seed"], cfg)
        bb, fc = models()
        load_weights(ROOT / rec["checkpoint"]["file"], bb, fc)
        bb.eval(); fc.eval()
        before = state_hash(cpu_state(bb))
        t = transform("eval")
        pred = []
        with torch.inference_mode():
            for k in range(0,1980,128):
                batch = torch.stack([t(Image.fromarray(im)) for im in images[k:k+128]]).cuda()
                logits = fc(bb(batch))
                assert torch.isfinite(logits).all()
                pred.append(logits.argmax(1).cpu().numpy())
        prediction = np.concatenate(pred)
        assert state_hash(cpu_state(bb)) == before
        correct = prediction == targets
        for g in GROUPS:
            rows.append(dict(representation_regime=rec["representation_regime"],
                label_regime=rec["observed_label_regime"], stage2_seed=rec["stage2_seed"],
                method=rec["method"], group=g, n=int(mask[g].sum()),
                accuracy_pct=float(100*correct[mask[g]].mean()),
                source_checkpoint_sha=rec["source_checkpoint_sha"],
                checkpoint_sha=rec["checkpoint"]["sha256"]))
        for cls in range(100):
            ix=targets==cls
            perclass.append(dict(tag=tag,class_id=cls,n=int(ix.sum()),
                                 correct=int(correct[ix].sum()) if ix.any() else 0,
                                 accuracy_pct=float(100*correct[ix].mean()) if ix.any() else None))
    for filename, data in (("crossed_final_metrics.csv", rows), ("crossed_per_class.csv", perclass)):
        with (OUT/filename).open("x", newline="", encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    print("EVALUATED 8 crossed models; no official TEST accessed", flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("train","evaluate"))
    p.add_argument("--representation",choices=("clean","noisy"))
    p.add_argument("--labels",choices=("clean","noisy"))
    p.add_argument("--seed",type=int,choices=(124,125),default=124)
    p.add_argument("--method",choices=("no_fusion","h2t"),default="no_fusion")
    a=p.parse_args()
    if a.mode=="train":
        assert a.representation and a.labels
        run(a.representation, a.labels, a.seed, a.method)
    else:
        evaluate()


if __name__=="__main__":
    main()
