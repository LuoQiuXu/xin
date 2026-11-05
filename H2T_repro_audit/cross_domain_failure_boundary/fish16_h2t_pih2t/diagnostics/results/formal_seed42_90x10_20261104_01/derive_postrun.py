"""CPU-only analysis of immutable Fish16 formal outputs; never opens images."""
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(r"D:\xiangmuxunlian\xin")
RUN = Path(r"D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261104_01")
# Small review artifacts stay in the authorized workspace when external write approval times out.
# The original RUN is always read-only. This directory contains no checkpoints, images or NPZ.
OUT = Path(__file__).parent / "analysis"
TASK = REPO / "H2T_repro_audit/cross_domain_failure_boundary/fish16_h2t_pih2t"
ORDER = ["A0", "B0", "A1", "A2", "B1", "B2"]
BUDGET = {n: 90 if n in ("A0", "B0") else 10 for n in ORDER}
torch.set_num_threads(2)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def csvout(name, rows):
    with (OUT / name).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat(value):
    result = {k: value[k] for k in ("overall", "macro", "worst_class")}
    result.update({f"{kind}_{g}": value[kind][g]
                   for kind in ("group_image", "group_macro")
                   for g in ("Head", "Medium", "Tail")})
    return result


def state_hash(state, prefix):
    digest = hashlib.sha256()
    for key in sorted(k for k in state if k.startswith(prefix)):
        digest.update(key[len(prefix):].encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(count=int(len(values)), mean=float(values.mean()), std=float(values.std()),
                minimum=float(values.min()), maximum=float(values.max()),
                p05=float(np.quantile(values, .05)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), donor_mean=float((1-values).mean()),
                donor_more_than_half_fraction=float((values < .5).mean()),
                donor_more_than_075_fraction=float((values < .25).mean()))


assert subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() == "experiment/fish16-clean-launch-fix-20261103"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == "072d7b77e0b77b8ca0187064519f5665f6422d0e"
assert not OUT.exists(), "Never overwrite a prior analysis"
OUT.mkdir()
s = read(RUN / "ACTUAL_EXECUTION_STATUS.json")
m = read(RUN / "FORMAL_RUN_MANIFEST.json")
assert s["status"] == "COMPLETE_FORMAL_SELECTED_CONDITIONS"
assert s["completed_conditions"] == ORDER and not s["resume_events"]
assert s["test_evaluations"] == 0 and s["additional_seeds"] == []
assert not (RUN / "FAILURE.json").exists()
assert m["selected_conditions"] == ORDER and m["protocol"]["seed"] == 42
assert m["protocol"]["amp_scaler"]["enabled"] is False
indexed = list(csv.DictReader((RUN / "OUTPUT_SHA256.csv").open(encoding="utf-8")))
index = {r["relative_path"]: r for r in indexed}
actual = {p.relative_to(RUN).as_posix() for p in RUN.rglob("*")
          if p.is_file() and p.name != "OUTPUT_SHA256.csv"}
assert actual == set(index)
for relative, row in index.items():
    path = RUN / relative
    assert path.stat().st_size == int(row["bytes"]) and sha(path) == row["sha256"], relative
for name, expected in m["protocol"]["source_hashes"].items():
    assert sha(TASK / name) == expected, name
for name, expected in m["protocol"]["original_sources"].items():
    assert sha(REPO / name) == expected, name
print("PASS original artifact index and source locks", len(index), flush=True)

manifest = Path(r"D:\h2t_cross_domain_audit\fish4knowledge\manifest_primary.csv")
assert sha(manifest) == m["protocol"]["manifest_sha256"]
rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
targets = np.full(len(rows), -1, dtype=np.int64)
names = {}
for i, row in enumerate(rows):
    # Metadata parsing only for train/val. Never resolve/stat/open any image path.
    if row["split_primary"] in ("train", "val"):
        c = int(row["relative_path"].replace("\\", "/").split("/")[0].removeprefix("fish_")) - 1
        assert 0 <= c < 16
        targets[i] = c
        if c in names:
            assert names[c] == row["label"]
        names[c] = row["label"]
plan = np.load(RUN / "sampling_plan.npz", allow_pickle=False)
tr, va, rec, don = (plan[k] for k in ("train", "val", "recipients", "donors"))
assert len(tr) == len(rec) == len(don) == 16269 and len(va) == 5396
assert all(rows[int(i)]["split_primary"] == "train" for i in np.concatenate([tr, rec, don]))
assert all(rows[int(i)]["split_primary"] == "val" for i in va)
assert np.array_equal(np.sort(plan["stage1"]), np.sort(tr))
groupof = {i: g for g, ids in m["groups"].items() for i in ids}
trn, van = (np.bincount(targets[a], minlength=16) for a in (tr, va))
rc, dc = targets[rec], targets[don]
classes = []
for c in range(16):
    classes.append(dict(class_id=c, class_folder=f"fish_{c+1:02d}", species=names[c], group=groupof[c],
        train_images=int(trn[c]), validation_images=int(van[c]),
        train_groups=len({rows[int(i)]["group_id"] for i in tr if targets[i] == c}),
        validation_groups=len({rows[int(i)]["group_id"] for i in va if targets[i] == c}),
        recipient_schedule_images=int((rc == c).sum()), donor_schedule_images=int((dc == c).sum())))
csvout("CLASS_COVERAGE.csv", classes)
finals, bests, epochs, perclass, effects, checkpoints, freeze, pairing = ([] for _ in range(8))
models_final = {}
for name in ORDER:
    cs, budget = s["conditions"][name], BUDGET[name]
    records = cs["records"]
    assert cs["status"] == "COMPLETE" and cs["completed_epochs"] == len(records) == budget
    assert cs["actual_optimizer_updates"] == budget * 1017 and cs["amp_skips"] == 0
    for ei, record in enumerate(records, 1):
        assert record["epoch"] == ei and record["samples"] == 16269
        assert record["attempted_optimizer_windows"] == record["actual_optimizer_updates"] == record["gradient_audits"] == 1017
        assert record["amp_skips"] == 0 and not record["overflow_events"]
        assert record["loss_scale_final"] == record["loss_scale_initial"] == 1
        assert record["lr"] == m["protocol"]["stage1_lrs" if budget == 90 else "stage2_lrs"][ei-1]
        assert read(RUN / name / f"epoch_{ei:03d}_metrics.json") == record
        pv = record["pif_values"]
        epochs.append(dict(condition=name, epoch=ei, train_loss=record["train_loss"], lr=record["lr"],
            actual_updates=record["actual_optimizer_updates"], skips=record["amp_skips"],
            **flat(record["validation"]), pif_a=pv[0] if pv else "", pif_b=pv[1] if pv else ""))
    finals.append(dict(condition=name, epoch=budget, **flat(records[-1]["validation"])))
    be = cs["best"]["epoch"]
    br = records[be-1]
    assert be == next(i+1 for i, r in enumerate(records)
        if r["validation"]["overall"] == max(v["validation"]["overall"] for v in records))
    bests.append(dict(condition=name, selected_by="validation_overall_earliest_tie", epoch=be, **flat(br["validation"])))
    finalpath = RUN / name / f"epoch_{budget:03d}.pth"
    assert sha(finalpath) == cs["final_checkpoint_sha256"]
    cp = torch.load(finalpath, map_location="cpu", weights_only=True)
    assert cp["formal_run"] and not cp["smoke_only"] and cp["seed"] == 42 and cp["epoch_completed"] == budget
    assert cp["records"] == records and cp["actual_updates_total"] == budget * 1017 and cp["scaler"] == {}
    assert cp["source_checkpoint_sha256"] == cs["source_checkpoint_sha256"] and cp["protocol_sha256"] == m["protocol_sha256"]
    assert all(bool(torch.isfinite(value).all()) for value in cp["model"].values())
    models_final[name] = cp["model"]
    for artifact, epoch in ((f"epoch_{budget:03d}.pth", budget), ("last.pth", budget), ("best_validation.pth", be)):
        path = RUN / name / artifact
        checkpoints.append(dict(condition=name, artifact=artifact, epoch=epoch, bytes=path.stat().st_size,
            sha256=sha(path), source_checkpoint_sha256=cs["source_checkpoint_sha256"] or ""))
    assert sha(RUN / name / "last.pth") == sha(finalpath)
    bcp = torch.load(RUN / name / "best_validation.pth", map_location="cpu", weights_only=True)
    assert bcp["epoch_completed"] == be and bcp["records"][-1] == br
    if budget == 10:
        src = "A0" if name.startswith("A") else "B0"
        cm = read(RUN / name / "CONDITION_MANIFEST.json")
        assert cs["source_checkpoint_sha256"] == s["conditions"][src]["final_checkpoint_sha256"]
        h = state_hash(models_final[src], "extractor.")
        assert h == cm["initialization"]["extractor"] == state_hash(cp["model"], "extractor.")
        assert state_hash(models_final[src], "classifier.") == cm["initialization"]["classifier"]
        keys = [k for k in cp["model"] if k.startswith("extractor.")]
        assert all(torch.equal(cp["model"][k], models_final[src][k]) for k in keys)
        freeze.append(dict(condition=name, source=src, epochs_with_runtime_freeze_gate=10,
            extractor_tensors=len(keys), bn_stat_buffers=sum(k.endswith(("running_mean", "running_var", "num_batches_tracked")) for k in keys),
            source_extractor_state_sha256=h, final_extractor_state_sha256=h,
            source_checkpoint_sha256=cs["source_checkpoint_sha256"], all_tensors_exact=True))
assert all(s["conditions"]["A0"]["initialization"][k] == s["conditions"]["B0"]["initialization"][k] for k in ("backbone", "classifier"))
for ei in range(90):
    a, b = (s["conditions"][n]["records"][ei] for n in ("A0", "B0"))
    assert all(a[k] == b[k] for k in ("recipient_control_sha256", "mixup_control_sha256"))
    pairing.append(dict(pair="A0_B0", epoch=ei+1, recipient_equal=True, mixup_equal=True, donor_equal="not_applicable"))
for ei in range(10):
    rr = {n: s["conditions"][n]["records"][ei] for n in ("A1", "A2", "B1", "B2")}
    assert len({r["recipient_control_sha256"] for r in rr.values()}) == 1
    assert rr["A2"]["donor_control_sha256"] == rr["B2"]["donor_control_sha256"]
    pairing.append(dict(pair="ALL_STAGE_II", epoch=ei+1, recipient_equal=True, mixup_equal="not_applicable", donor_equal=True))
    for plus, base in (("A2", "A1"), ("B2", "B1")):
        a, b = flat(rr[plus]["validation"]), flat(rr[base]["validation"])
        effects.append(dict(comparison=f"{plus}-{base}", epoch=ei+1, **{k: a[k]-b[k] for k in a}))
for c in range(16):
    v = {n: s["conditions"][n]["records"][-1]["validation"]["per_class"][c] for n in ORDER}
    perclass.append(dict(**classes[c], **v, A2_minus_A1=v["A2"]-v["A1"], B2_minus_B1=v["B2"]-v["B1"], B0_minus_A0=v["B0"]-v["A0"]))
for filename, table in (("FINAL_VALIDATION.csv", finals), ("BEST_VALIDATION.csv", bests),
    ("EPOCH_METRICS_ALL.csv", epochs), ("PER_CLASS_FINAL.csv", perclass), ("PAIRWISE_EPOCH_EFFECTS.csv", effects),
    ("CHECKPOINT_SHA256.csv", checkpoints), ("PAIRING_CHECKS.csv", pairing), ("EXTRACTOR_FREEZE_CHECKS.csv", freeze)):
    csvout(filename, table)
print("PASS counts, checkpoint lineage and pairing", flush=True)

events = Counter()
entries = 0
controls, all_mass, massrows = {}, [], []
channelcounts = np.zeros(2048, dtype=np.int64)
for name in ORDER:
    files = sorted((RUN / name).glob("epoch_*_attempt_*.jsonl"))
    assert len(files) == BUDGET[name]
    for ei, path in enumerate(files, 1):
        count, samples = 0, 0
        cr, mass = [], []
        assert path.name == f"epoch_{ei:03d}_attempt_001.jsonl"
        with path.open(encoding="utf-8") as stream:
            for bi, line in enumerate(stream):
                x = json.loads(line)
                c, e = x["control"], x["event"]
                assert c["batch"] == bi and c["samples"] == (16 if bi < 1016 else 13)
                assert e["condition"] == name and e["epoch"] == ei and e["batch"] == bi
                assert e["nan_count"] == e["inf_count"] == 0 and not e["nonfinite_parameters"]
                assert e["scale_before"] == e["scale_after"] == 1
                assert e["optimizer_steps_after"] - e["optimizer_steps_before"] == 1
                assert math.isfinite(x["loss"])
                events[e["kind"]] += 1
                count += 1
                samples += c["samples"]
                entries += 1
                if name in ("A1", "A2", "B1", "B2"):
                    cr.append((c["recipient_sha256"], c["labels_sha256"], c.get("donor_sha256")))
                if name == "A2":
                    ids = np.asarray(c["h2t_channels"], dtype=int)
                    assert len(ids) == len(set(ids)) == 512 and ids.min() >= 0 and ids.max() < 2048
                    channelcounts[ids] += 1
                if name == "B2":
                    vals = c["pi_recipient_mass"]
                    assert len(vals) == c["samples"] and all(math.isfinite(v) and 0 <= v <= 1 for v in vals)
                    mass.extend(vals)
        assert count == 1017 and samples == 16269
        if cr:
            controls[(name, ei)] = cr
        if name == "B2":
            v = np.asarray(mass)
            assert len(v) == 16269
            all_mass.append(v)
            masks = [("All", np.ones(len(v), dtype=bool))] + [(g, np.isin(rc, ids)) for g, ids in m["groups"].items()] + [(f"class_{c}", rc == c) for c in range(16)]
            for scope, mask in masks:
                massrows.append(dict(epoch=ei, scope=scope, **stats(v[mask])))
assert entries == 223740
for ei in range(1, 11):
    for name in ("A2", "B1", "B2"):
        assert all(a[:2] == b[:2] for a, b in zip(controls[("A1", ei)], controls[(name, ei)]))
    assert all(a[2] == b[2] for a, b in zip(controls[("A2", ei)], controls[("B2", ei)]))
assert channelcounts.sum() == 10 * 1017 * 512
csvout("PI_RECIPIENT_MASS_STATS.csv", massrows)
csvout("H2T_CHANNEL_COUNTS.csv", [dict(channel=i, selected_batches=int(v), fraction_of_10170_batches=float(v/10170)) for i, v in enumerate(channelcounts)])
csvout("DONOR_RECIPIENT_FREQUENCY.csv", [dict(class_id=c, species=names[c], group=groupof[c], train_images=int(trn[c]),
    train_fraction=float(trn[c]/len(tr)), donor_images_per_epoch=int((dc == c).sum()), donor_fraction=float((dc == c).mean()),
    recipient_images_per_epoch=int((rc == c).sum()), recipient_fraction=float((rc == c).mean())) for c in range(16)])
csvout("DONOR_RECIPIENT_CLASS_PAIRS.csv", [dict(recipient_class=a, donor_class=b, recipient_group=groupof[a], donor_group=groupof[b],
    pairs_per_epoch=int(((rc == a) & (dc == b)).sum())) for a in range(16) for b in range(16)])
masses = np.stack(all_mass)
mechanisms = dict(pi_all_epochs=stats(masses.ravel()), pi_epoch1=stats(masses[0]), pi_epoch10=stats(masses[-1]),
    pi_epoch10_by_recipient_group={g: stats(masses[-1, np.isin(rc, ids)]) for g, ids in m["groups"].items()},
    donor_group_fraction={g: float(np.isin(dc, ids).mean()) for g, ids in m["groups"].items()},
    recipient_group_fraction={g: float(np.isin(rc, ids).mean()) for g, ids in m["groups"].items()},
    same_class_pair_fraction=float((rc == dc).mean()), same_class_pairs_per_epoch=int((rc == dc).sum()),
    h2t_channels=dict(channels=2048, selected_per_batch=512, batches=10170, total_selections=int(channelcounts.sum()),
        minimum_channel_selections=int(channelcounts.min()), maximum_channel_selections=int(channelcounts.max()), mean_channel_selections=float(channelcounts.mean())),
    prediction_distribution="NOT_RECORDED; no new inference was run",
    semantic_compatibility="UNVERIFIED; same/different class metadata is not a semantic-distance measure")
dump("MECHANISM_SUMMARY.json", mechanisms)
counts = dict(epochs=220, attempted_windows=entries, actual_sgd_updates=entries, gradient_audits=entries,
    physical_log_entries=entries, optimizer_skips=0, logged_nan_count=0, logged_inf_count=0, event_kinds=dict(events))
provenance = dict(formal_run_manifest_sha256=sha(RUN / "FORMAL_RUN_MANIFEST.json"),
    original_output_index_sha256=sha(RUN / "OUTPUT_SHA256.csv"), actual_execution_status_sha256=sha(RUN / "ACTUAL_EXECUTION_STATUS.json"),
    console_sha256=sha(Path(str(RUN) + "_console.log")), source_sha256=m["protocol"]["source_hashes"], original_source_sha256=m["protocol"]["original_sources"])
verify = dict(status="PASS", execution_head=m["run_commit"], branch=m["branch"], run_root=str(RUN),
    original_artifact_files_verified=len(index), original_artifact_bytes_verified=sum(int(r["bytes"]) for r in indexed), counts=counts,
    conditions={n: dict(status=s["conditions"][n]["status"], epochs=BUDGET[n], updates=s["conditions"][n]["actual_optimizer_updates"], skips=0) for n in ORDER},
    source_hashes_unchanged=True, common_backbone_classifier_initialization_exact=True,
    epoch_pairing_checks=len(pairing), all_pairing_checks_pass=True, all_stage2_batch_view_hashes_equal=True,
    frozen_extractor_endpoint_checks=freeze, runtime_stage2_freeze_gates_completed=40,
    runtime_freeze_gate_evidence="Pinned formal_train.py:696-697 checks full extractor state before each Stage-II epoch checkpoint; all 40 committed.",
    no_resume_or_retry=True, precision="FP32; autocast and scaler disabled; all logged scales exactly 1",
    test_evaluations=0, test_images_opened_by_analysis=0, additional_seeds=[], postrun_gpu_inference_or_training=False, provenance=provenance)
dump("POSTRUN_VALIDATION.json", verify)
summary = dict(final=finals, best_validation=bests, final_pairwise_effects=[r for r in effects if r["epoch"] == 10], counts=counts,
    stage1_final_B0_minus_A0={k: finals[1][k]-finals[0][k] for k in flat(s["conditions"]["A0"]["records"][-1]["validation"])},
    group_validation_counts={g: int(van[ids].sum()) for g, ids in m["groups"].items()},
    group_train_counts={g: int(trn[ids].sum()) for g, ids in m["groups"].items()}, mechanism=mechanisms)
dump("ANALYSIS_SUMMARY.json", summary)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25, "savefig.dpi": 160})
figdir = OUT / "figures"
figdir.mkdir()
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(figdir / f"{name}.png")
    fig.savefig(figdir / f"{name}.pdf")
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for name in ("A0", "B0"):
    rr = [r for r in epochs if r["condition"] == name]
    for ax, key in zip(axes.ravel(), ("train_loss", "overall", "macro", "group_macro_Tail")):
        ax.plot([r["epoch"] for r in rr], [r[key] for r in rr], label=name)
        ax.set(xlabel="Stage-I epoch", ylabel=key + (" (%)" if key != "train_loss" else ""))
        ax.legend()
savefig(fig, "stage1_curves")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for name in ("A1", "A2", "B1", "B2"):
    rr = [r for r in epochs if r["condition"] == name]
    for ax, key in zip(axes.ravel(), ("train_loss", "overall", "macro", "group_image_Head", "group_macro_Tail", "group_image_Tail")):
        ax.plot([r["epoch"] for r in rr], [r[key] for r in rr], label=name, marker=".")
        ax.set(xlabel="Stage-II epoch", ylabel=key + (" (%)" if key != "train_loss" else ""))
        ax.legend()
savefig(fig, "stage2_curves")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for comparison in ("A2-A1", "B2-B1"):
    rr = [r for r in effects if r["comparison"] == comparison]
    for ax, key in zip(axes.ravel(), ("macro", "group_macro_Tail", "group_image_Tail", "overall", "group_image_Head", "worst_class")):
        ax.plot([r["epoch"] for r in rr], [r[key] for r in rr], marker=".", label=comparison)
        ax.axhline(0, color="black", lw=.7)
        ax.set(xlabel="Stage-II epoch", ylabel=key + " delta (pp)")
        ax.legend()
savefig(fig, "paired_epoch_effects")
fig, ax = plt.subplots(figsize=(10, 4))
rr = [r for r in epochs if r["condition"] == "B0"]
for key in ("pif_a", "pif_b"):
    ax.plot([r["epoch"] for r in rr], [r[key] for r in rr], label=key)
ax.set(xlabel="B0 Stage-I epoch", ylabel="PIF coefficient")
ax.legend()
savefig(fig, "pif_evolution")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for scope in ("All", "Head", "Medium", "Tail"):
    rr = [r for r in massrows if r["scope"] == scope]
    axes[0].plot([r["epoch"] for r in rr], [r["mean"] for r in rr], label=scope, marker=".")
axes[0].set(xlabel="B2 Stage-II epoch", ylabel="Mean recipient mass")
axes[0].legend()
axes[1].hist(masses.ravel(), bins=40, range=(0, 1), density=True)
axes[1].set(xlabel="Recipient mass (all epochs)", ylabel="Density")
savefig(fig, "pi_recipient_mass")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
colors = [{"Head": "#4477aa", "Medium": "#ddaa33", "Tail": "#228833"}[groupof[c]] for c in range(16)]
for ax, key in zip(axes, ("A2_minus_A1", "B2_minus_B1")):
    ax.bar(range(16), [r[key] for r in perclass], color=colors)
    ax.axhline(0, color="black", lw=.7)
    ax.set(ylabel=key + " (pp)", title="Final Validation per-class change; blue=Head, gold=Medium, green=Tail")
axes[1].set_xticks(range(16), [f"{c}: {names[c]}" for c in range(16)], rotation=65, ha="right")
savefig(fig, "per_class_effects")
print(json.dumps(dict(status="PASS", counts=counts, final=finals, effects=summary["final_pairwise_effects"],
    group_validation_counts=summary["group_validation_counts"], mechanism=mechanisms), ensure_ascii=False), flush=True)
