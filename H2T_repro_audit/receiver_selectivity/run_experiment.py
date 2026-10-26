#!/usr/bin/env python3
"""One-factor H2T receiver-selectivity experiment; fixed BN, IF200.

Causal contrast: TailOff vs RandomOffMatched, with identical fusion dose per
batch and otherwise identical H2T masks, training sequences and optimization.
This deliberately does NOT implement new learned gates or losses.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IF200 = ROOT / "H2T_repro_audit/if200_stress"

# Prevent the two unrelated files called policy.py from shadowing each other.
spec = importlib.util.spec_from_file_location("receiver_selectivity_policy", HERE / "policy.py")
receiver_policy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(receiver_policy)
ARMS = receiver_policy.ARMS
choose_active = receiver_policy.choose_active

sys.path.insert(0, str(IF200))
import run_if200 as old  # noqa: E402

SEEDS = (124, 125)
COMP_A = "TailOff"
COMP_B = "RandomOffMatched"
PRIMARY_TAIL_THRESHOLD_PP = 0.5
PRIMARY_OVERALL_FLOOR_PP = -0.25
FORMER_HARM_CLASSES = (72, 77, 79, 87)  # secondary/descriptive ONLY


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("refusing empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        w = csv.DictWriter(file, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def sha(path: Path) -> str:
    return old.sha(path)


def required_files() -> list[Path]:
    base = old.RESULTS
    return [
        base / "stage1/stage1_fixed_epoch200.pth",
        base / "stage1/summary.json",
        base / "validation_maps.npy",
        old.SOURCE / "train_images.npy",
        old.SOURCE / "train_labels.npy",
        old.SOURCE / "validation_images.npy",
        old.SOURCE / "validation_labels.npy",
        old.SOURCE / "split_indices.npz",
    ]


def preflight(out: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA environment required for paired training")
    missing = [str(p) for p in required_files() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing locked historic inputs; NO hidden retraining: " + repr(missing))
    ck = old.load_s1()
    old.cfg("stage2")  # assert original IF200 config and optimizer schedule
    xx, y, vx, vy, indices, raw = old.data()
    assert len(xx) == 9502 and len(y) == 9502 and len(vy) == 1980
    assert np.array_equal(np.bincount(y, minlength=100), old.counts200())
    maps = np.load(old.RESULTS / "validation_maps.npy", mmap_mode="r", allow_pickle=False)
    assert maps.shape == (1980, 64, 8, 8)
    assert np.bincount(vy, minlength=100)[0] == 0
    provenance_sources = required_files() + [IF200/"run_if200.py", IF200/"policy.py", HERE/"policy.py", HERE/"run_experiment.py"]
    input_hash = {str(p.relative_to(ROOT)): sha(p) for p in provenance_sources}
    return {
        "source_hashes": input_hash,
        "stage1_sha": sha(old.RESULTS / "stage1/stage1_fixed_epoch200.pth"),
        "backbone": "ResNet32; fixed BN eval",
        "stage1_epoch": int(ck["epoch"]),
        "dataset": "CIFAR100-LT IF200; 9502 TRAIN / 1980 TRAIN-only holdout",
        "validation_note": "class0 ABSENT, reused developer holdout, not independent TEST",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "epochs": 10,
        "updates_per_arm": old.STEPS * old.S2_EPOCHS,
        "optimizer": "SGD lr=0.02 initial (author factor 0.1), momentum=0.9, wd=0.0002, cosine",
        "BN": "frozen/eval, differs from original author shift_bn=true",
        "shared_mask": "16/64 channels, one shared mask per minibatch",
        "primary": "TailOff - RandomOffMatched, fixed epoch10; paired seeds",
        "decision_rule": {
            "both_seeds_tail_delta_strictly_positive": True,
            "mean_tail_delta_min_pp": PRIMARY_TAIL_THRESHOLD_PP,
            "both_seeds_overall_delta_min_pp": PRIMARY_OVERALL_FLOOR_PP,
            "if_failed": "Do not promote receiver selectivity to a new method.",
        },
        "former_harmed_tail_classes": list(FORMER_HARM_CLASSES),
        "former_classes_role": "descriptive only; no post hoc arm tuning",
        "official_CIFAR100_TEST": "prohibited/not accessed",
    }


def apply_original_mask(fa: torch.Tensor, fb: torch.Tensor, shared: np.ndarray,
                        active: np.ndarray) -> torch.Tensor:
    """Exactly 16 entire channels for eligible receivers, zero otherwise."""
    n = fa.shape[0]
    assert fa.shape == fb.shape and fa.shape[1:] == (64, 8, 8)
    assert shared.shape == (16,) and active.shape == (n,)
    sel = torch.as_tensor(shared.astype(np.int64), device=fa.device)
    m = torch.zeros((64,), dtype=torch.bool, device=fa.device)
    m[sel] = True
    assert int(m.sum()) == 16
    apply = torch.as_tensor(active, device=fa.device, dtype=torch.bool)[:, None, None, None]
    return torch.where(apply & m[None, :, None, None], fb, fa)


def train_arm(seed: int, arm: str, out: Path, pre: dict) -> None:
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("seed/arm outside preregistration")
    run = out / f"seed{seed}" / arm
    if run.exists():
        raise FileExistsError(f"Refuse overwrite: {run}")
    run.mkdir(parents=True)
    ck = old.load_s1()
    xx, y, vx, vy, indices, raw = old.data()
    maps = np.load(old.RESULTS / "validation_maps.npy", mmap_mode="r", allow_pickle=False)
    old.seedall(seed)
    bb, fc = old.model()
    bb.load_state_dict(ck["backbone"], strict=True)
    fc.load_state_dict(ck["classifier"], strict=True)
    bb.eval().requires_grad_(False)
    fc.train()
    bb_hash = old.statehash(bb)
    initial_fc_hash = old.statehash(fc)
    q = old.cfg("stage2")
    optimizer = torch.optim.SGD(fc.parameters(), lr=q.lr, momentum=q.momentum, weight_decay=q.weight_decay)
    train_transform, _ = old.tfms()
    aug_gen = torch.Generator().manual_seed(seed + 300009)
    schedule = old.plans(seed, y)
    counts = {"total_receivers": 0, "tail_receivers": 0,
              "fused_receivers": 0, "tail_fused": 0, "headmedium_fused": 0}
    signatures = []
    control = {}
    control_path = out / f"seed{seed}" / ARMS[0] / "paired_batches.csv"
    if arm != ARMS[0]:
        if not control_path.exists():
            raise FileNotFoundError(f"Run DR arm first: {control_path}")
        with control_path.open(newline="", encoding="utf-8") as f:
            control = {int(r["step"]): r["signature"] for r in csv.DictReader(f)}
        assert len(control) == 750
    steps = 0
    loss_records = []
    for epoch, (rec, don, gn, sn) in enumerate(schedule):
        lr = .2 * .1 * .5 * (1 + math.cos(epoch * math.pi / 10))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_ce = 0.
        fc.train()
        for batch, start in enumerate(range(0, old.N, old.BATCH)):
            end = min(old.N, start + old.BATCH)
            r, d = rec[start:end], don[start:end]
            xa, xb = old.augment(xx, r, d, aug_gen, train_transform)
            with torch.no_grad():
                fa, fb = bb(xa.cuda()), bb(xb.cuda())
                # Exactly the batch-shared mask analogue in original IF200 run.
                shared = np.argsort(-sn[batch], kind="stable")[:16].astype(np.int64)
                active = choose_active(y[r], arm, seed, epoch, batch)
                mixed = apply_original_mask(fa, fb, shared, active)
            # Pair original inputs/features/shuffle/masks, NOT arm-specific gating.
            signature = old.hash_data(r, d, xa.numpy(), xb.numpy(),
                                      fa.cpu().numpy(), fb.cpu().numpy(),
                                      gn[start:end], sn[batch], shared)
            if control and signature != control[steps]:
                raise RuntimeError(f"Input/feature/mask mismatch at seed={seed} epoch={epoch} batch={batch}")
            signatures.append({"step": steps, "epoch": epoch+1, "batch": batch,
                               "signature": signature, "active_count": int(active.sum()),
                               "tail_count": int((y[r] >= 71).sum())})
            counts["total_receivers"] += len(r)
            counts["tail_receivers"] += int((y[r] >= 71).sum())
            counts["fused_receivers"] += int(active.sum())
            counts["tail_fused"] += int((active & (y[r] >= 71)).sum())
            counts["headmedium_fused"] += int((active & (y[r] < 71)).sum())
            labels = torch.as_tensor(y[r], dtype=torch.long, device="cuda")
            logits = fc(mixed)
            loss = nn.functional.cross_entropy(logits, labels)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_ce += float(loss.detach().cpu()) * len(r)
            steps += 1
        assert old.statehash(bb) == bb_hash, "Backbone or BN state mutated"
        loss_records.append({"epoch": epoch+1, "lr": lr, "train_CE": total_ce/old.N})
    assert steps == 750
    save_csv(run / "paired_batches.csv", signatures)
    save_csv(run / "train_epochs.csv", loss_records)
    if arm in ("TailOff", "RandomOffMatched"):
        # Both have precisely equal augmented receiver counts per batch.
        for r in signatures:
            assert int(r["active_count"]) == (old.BATCH if r["batch"] < 74 else 30) - int(r["tail_count"])
    metrics, val_logits, val_preds = old.vals(fc, vy, maps=maps)
    assert val_logits.shape == (1980, 100)
    assert np.array_equal(val_preds, val_logits.argmax(axis=1))
    assert old.statehash(bb) == bb_hash and old.statehash(fc) != initial_fc_hash
    torch.save({"seed": seed, "arm": arm, "classifier": {
        k:v.detach().cpu().clone() for k,v in fc.state_dict().items()},
        "stage1_sha": pre["stage1_sha"], "epoch": 10}, run/"final_fc.pth")
    np.savez_compressed(run / "validation_epoch10.npz", labels=vy, logits=val_logits,
                        predictions=val_preds)
    # Independent reload sanity check of the completed classifier artifact.
    saved = torch.load(run/"final_fc.pth", map_location="cpu", weights_only=False)
    assert saved["seed"] == seed and saved["arm"] == arm and saved["epoch"] == 10
    fc.load_state_dict(saved["classifier"], strict=True)
    replay_metrics, replay_logits, replay_preds = old.vals(fc, vy, maps=maps)
    assert np.array_equal(replay_preds, val_preds)
    assert np.array_equal(replay_logits, val_logits)
    assert all(abs(replay_metrics[k] - metrics[k]) < 1e-10 for k in metrics)
    dump_json(run / "summary.json", {
        "seed":seed,"arm":arm,"epoch":10,"total_steps":steps,"final":metrics,
        "batch_pairing": "PASS" if arm != ARMS[0] else "reference",
        "backbone_BN_hash_unchanged": True, "initial_fc_hash": initial_fc_hash,
        "final_fc_hash": old.statehash(fc), "exposure": counts,
        "official_TEST_used": False,
    })
    print(f"completed {seed} {arm}: Overall={metrics['Overall']:.3f} Tail={metrics['Tail']:.3f}", flush=True)


def compare(xpred: np.ndarray, ypred: np.ndarray, labels: np.ndarray) -> tuple[dict, list[dict]]:
    """First arm (x) minus second arm (y); same ordered validation examples."""
    out = []
    for k in range(1, 100):
        mask = labels == k
        assert mask.sum() == 20
        wins = int((xpred[mask] == labels[mask]).sum() -
                   ((xpred[mask] == labels[mask]) & (ypred[mask] == labels[mask])).sum())
        losses = int((ypred[mask] == labels[mask]).sum() -
                     ((xpred[mask] == labels[mask]) & (ypred[mask] == labels[mask])).sum())
        out.append({"class":k,"wins":wins,"losses":losses,"net":wins-losses,"delta_pp":5*(wins-losses)})
    tail = [r for r in out if r["class"] >= 71]
    overall_wins = sum(row["wins"] for row in out)
    overall_losses = sum(row["losses"] for row in out)
    return {
        "overall_wins":overall_wins,"overall_losses":overall_losses,
        "Overall_delta_pp":100*(overall_wins-overall_losses)/len(labels),
        "Tail_delta_pp":100*sum(r["net"] for r in tail)/(len(tail)*20),
        "tail_wins":sum(r["wins"] for r in tail),
        "tail_losses":sum(r["losses"] for r in tail),
        "tail_net":sum(r["net"] for r in tail),
        "persistent_harm_candidates":[r["class"] for r in tail if r["net"]<0],
    }, out


def analyze(out: Path) -> None:
    pre = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    for rel,h in pre["source_hashes"].items():
        if sha(ROOT/rel) != h:
            raise RuntimeError(f"Historical source changed: {rel}")
    summary, per_class = [], []
    all_results = {}
    for seed in SEEDS:
        maps = {}
        for arm in ARMS:
            run = out / f"seed{seed}" / arm
            with np.load(run/"validation_epoch10.npz",allow_pickle=False) as v:
                y,p = v["labels"].copy(),v["predictions"].copy()
                lg = v["logits"].copy()
            assert lg.shape == (1980,100) and np.isfinite(lg).all()
            assert np.array_equal(p, lg.argmax(1))
            maps[arm] = (y,p)
            with (run/"summary.json").open(encoding="utf-8") as f:
                rec = json.load(f)
            summary.append({"seed":seed,"arm":arm,**rec["final"],**rec["exposure"]})
        labels = maps[ARMS[0]][0]
        assert all(np.array_equal(maps[a][0],labels) for a in ARMS)
        comp_names = [
            ("Original-BatchShared","DR-NoFusion"),
            ("TailOff","Original-BatchShared"),
            ("RandomOffMatched","Original-BatchShared"),
            (COMP_A,COMP_B),
        ]
        for a,b in comp_names:
            result, rows=compare(maps[a][1],maps[b][1],labels)
            all_results[f"{seed}:{a} - {b}"] = result
            for row in rows:
                per_class.append({"seed":seed,"contrast":f"{a} - {b}",**row})
    save_csv(out/"epoch10_summary.csv", summary)
    save_csv(out/"per_class_pairing.csv", per_class)
    primary = [all_results[f"{seed}:{COMP_A} - {COMP_B}"] for seed in SEEDS]
    tail_delta = [p["Tail_delta_pp"] for p in primary]
    overall_delta = [p["Overall_delta_pp"] for p in primary]
    # Preregistered exploratory check, NOT a significance or publication claim.
    passes = all(x > 0 for x in tail_delta) and              sum(tail_delta)/len(tail_delta) >= PRIMARY_TAIL_THRESHOLD_PP and              all(x >= PRIMARY_OVERALL_FLOOR_PP for x in overall_delta)
    decision = "CONDITIONAL_SIGNAL_ONLY" if passes else "NO_SUPPORT_STOP"
    dump_json(out/"analysis.json", {
        "contrasts":all_results,"primary_tail_delta_pp":tail_delta,
        "primary_overall_delta_pp":overall_delta,
        "decision":decision,
        "interpretation": "Exploratory only: 2 Stage II seeds share same Stage I & reused 99-class TRAIN holdout.",
        "known_harmed_class_ids":list(FORMER_HARM_CLASSES),
        "warning": "The 4 IDs above were selected from the previous IF200 experiment. They are NOT a prospective primary test.",
        "official_TEST_used":False
    })
    tab = ["| Seed | TailOff − RandomOffMatched Tail pp | Overall pp | Tail wins/losses |",
           "|---|---:|---:|---:|"]
    for seed, rec in zip(SEEDS, primary):
        tab.append(f"| {seed} | {rec['Tail_delta_pp']:+.4f} | {rec['Overall_delta_pp']:+.4f} | {rec['tail_wins']}/{rec['tail_losses']} |")
    (out/"SCIENTIFIC_REPORT.md").write_text("\n".join([
        "# H2T Receiver Selectivity: One-Factor Ablation",
        "", "## Question",
        "Does removing H2T fusion specifically from tail receivers outperform "
        "randomly removing the same number of fusion events per minibatch?",
        "", "The only manipulation is which receiver examples receive the "
        "original batch-shared 16/64 hard replacement. Donors, labels, "
        "BN mode, backbone, optimizer and step count remain unchanged.",
        "", "## Primary results", *tab, "",
        f"Pre-registered exploratory decision: **{decision}**.",
        "", "## Interpretation limits",
        "- The comparison isolates receiver eligibility conditional on a "
        "dose-matched random skip, not the causal history of a particular damaged validation class.",
        "- All results are on the repeatedly used 1980-image TRAIN holdout "
        "(class0 absent), not the official CIFAR TEST. Two seeds share the same Stage I.",
        "- Class 72/77/79/87 are descriptive historical observations, never the primary selection target.",
        "- Fixed BN differs from the official shift_bn=True recipe; "
        "this is a mechanism stress experiment, not a claim that original H2T fails generally.",
        "- A small or sign-changing Tail effect means STOP; do not tune classes, "
        "mask ratios, donor policies, difficulty proxies or seeds after reading the outcome.",
        "", "## Files", "- epoch10_summary.csv", "- per_class_pairing.csv",
        "- analysis.json", "- per-seed NPZ, checkpoint, paired hashes and exposure counts", ""
    ]),encoding="utf-8")
    print(f"ANALYSIS COMPLETE: {decision}; Tail pp={tail_delta}; Overall pp={overall_delta}")


def self_test() -> None:
    """No dataset/GPU: verify the concrete hard-replacement path and win/loss counting."""
    x = torch.arange(7*64*8*8, dtype=torch.float32).reshape(7,64,8,8)
    donor = x + 1_000_000.
    selected = np.arange(0,64,4,dtype=np.int64)  # 16 channels
    labels = np.array([0,35,36,70,71,95,99])
    for arm in ARMS:
        active = choose_active(labels,arm,124,0,0)
        z = apply_original_mask(x, donor, selected, active)
        expected = x.clone()
        for i,on in enumerate(active):
            if on:
                expected[i, selected] = donor[i, selected]
        assert torch.equal(z,expected), f"hard-channel replacement bug in {arm}"
    # 99 classes, 20 examples per class; exactly one changed correction in tail class71.
    yy = np.repeat(np.arange(1,100,dtype=np.int64),20)
    a = yy.copy()
    b = yy.copy()
    b[np.flatnonzero(yy==71)[0]] = 0
    stats,rows=compare(a,b,yy)
    assert stats["tail_wins"] == 1 and stats["tail_losses"] == 0
    assert abs(stats["Tail_delta_pp"] - 100/580)<1e-12
    assert abs(stats["Overall_delta_pp"] - 100/1980)<1e-12
    assert next(x for x in rows if x["class"]==71)["delta_pp"] == 5.
    print("SELF TEST PASS: complete-channel fusion flags and paired result arithmetic")


def run_all(run_id: str) -> Path:
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise ValueError("run-id must contain only alnum, underscore and hyphen")
    out = HERE / "results" / run_id
    if out.exists():
        raise FileExistsError(f"Refuse overwrite of old run: {out}")
    # Missing historical inputs must not leave a misleadingly started run dir.
    pre=preflight(out)
    out.mkdir(parents=True)
    dump_json(out/"protocol.json",pre)
    for seed in SEEDS:
        for arm in ARMS:
            train_arm(seed,arm,out,pre)
    analyze(out)
    return out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test","preflight","run-all","analyze"))
    parser.add_argument("--run-id", required=True)
    a=parser.parse_args()
    out=HERE/"results"/a.run_id
    if a.command=="self-test":
        self_test()
    elif a.command=="preflight":
        print(json.dumps(preflight(out),ensure_ascii=False,indent=2))
    elif a.command=="run-all":
        print("RESULT DIR",run_all(a.run_id))
    elif a.command=="analyze":
        if not out.is_dir():
            raise FileNotFoundError(out)
        analyze(out)


if __name__ == "__main__":
    main()
