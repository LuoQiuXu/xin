"""Fixed Low-VS donor-affinity x margin-protection pilot; no version-control calls."""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import policy

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = PARENT.parent
RESULTS = HERE / "results"
OLD = PARENT / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014"
DISC = PARENT / "lowvs_disc_preserve"
BANK_OLD = PARENT / "gu_lowvs" / "results" / "balanced_clean_train_bank.npz"
SEEDS = (124, 125)
ARMS = ("RandomDonor", "HighAffinityDonor", "RandomDonor+MarginProtect", "HighAffinityDonor+MarginProtect")
LETTERS = dict(zip(ARMS, "ABCD"))
METRICS = ("Overall", "Head", "OfficialMedium", "Tail", "Many", "Medium", "Few", "Macro", "validation_CE")
N, BATCH, EPOCHS = 10847, 128, 10
SOURCE_NAMES = ("run_factorial.py", "policy.py", "test_policy.py", "PROTOCOL.md", "CODEX_LOWVS_FACTORIAL.txt")
BASE = None
TEST_EVENTS, WRITE_EVENTS, VCS_EVENTS = [], [], []
INITIAL_READ_ONLY_VCS_QUERIES = (
    "Before the task file was read, read-only status/branch/log/ls-files queries located it. "
    "No version-control mutation occurred. After reading the prohibition, no further VCS operations are permitted."
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    paths = [HERE / n for n in SOURCE_NAMES]
    paths += [PARENT / "ravs_v1" / n for n in ("run_ravs_v1.py", "run_safeguards.py", "ravs_policy.py")]
    paths += [DISC / "run_pilot.py", PARENT / "gu_lowvs" / "run_gu_lowvs.py"]
    paths += [OLD / n for n in ("common.py", "stability_models.py", "online_policy.py",
                               "reused_train_helpers.py", "reused_vs_math.py")]
    return {p.relative_to(REPO).as_posix(): sha(p) for p in paths}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to(RESULTS.resolve()):
        raise RuntimeError(f"Output escapes this experiment results/: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def new_json(path, value):
    p = output_path(path)
    with p.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def new_npz(path, **arrays):
    with output_path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def new_text(path, text):
    with output_path(path).open("x", encoding="utf-8") as handle:
        handle.write(text)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CsvSink:
    def __init__(self, path):
        self.handle = output_path(path).open("x", encoding="utf-8", newline="")
        self.writer = None
    def row(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
    def flush(self):
        self.handle.flush()
    def close(self):
        self.handle.close()


def new_csv(path, rows):
    sink = CsvSink(path)
    try:
        for row in rows:
            sink.row(row)
    finally:
        sink.close()


def bn_hash(bb):
    h = hashlib.sha256()
    for k, value in bb.state_dict().items():
        if k.endswith(("running_mean", "running_var", "num_batches_tracked")):
            h.update(k.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def install_local_guard():
    forbidden_root = str((REPO / "H2T" / "data" / "cifar100" / "cifar-100-python" / "test").resolve()).casefold()
    audit_root = str(PARENT.resolve()).casefold()
    allowed = str(RESULTS.resolve()).casefold()
    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(args[0])))
            key = str(path).casefold()
            if key == forbidden_root or (path.name.casefold() == "test" and path.parent.name.casefold() == "cifar-100-python"):
                TEST_EVENTS.append({"event": event, "path": str(path)})
                raise RuntimeError("Official CIFAR100 TEST file open/hash is forbidden")
            if key == audit_root or key.startswith(audit_root + os.sep):
                mode = args[1] if len(args) > 1 else None
                flags = args[2] if len(args) > 2 else 0
                writing = ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
                           (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
                if writing and not (key == allowed or key.startswith(allowed + os.sep)):
                    WRITE_EVENTS.append({"event": event, "path": str(path)})
                    raise RuntimeError("Historical/source files are read-only: " + str(path))
        if event in ("os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.chmod", "os.utime", "os.link", "os.symlink"):
            for value in args[:2]:
                if isinstance(value, (str, bytes, os.PathLike)):
                    key = os.path.abspath(os.fsdecode(value)).casefold()
                    if (key == audit_root or key.startswith(audit_root + os.sep)) and not (key == allowed or key.startswith(allowed + os.sep)):
                        WRITE_EVENTS.append({"event": event, "path": str(value)})
                        raise RuntimeError("Modification outside this results directory is forbidden")
        if event == "subprocess.Popen":
            executable = str(args[0]).replace("\\", "/").rsplit("/", 1)[-1].lower()
            if executable in ("git", "git.exe", "gh", "gh.exe"):
                VCS_EVENTS.append(str(args[0]))
                raise RuntimeError("Version-control operations are prohibited")
    sys.addaudithook(audit)


def guard_ok():
    assert not TEST_EVENTS and not WRITE_EVENTS and not VCS_EVENTS
    if BASE is not None:
        BASE.check_test_guard()


def get_base():
    global BASE
    if BASE is None:
        sys.path.insert(0, str(PARENT / "ravs_v1"))
        import run_ravs_v1
        BASE = run_ravs_v1
        BASE.forbid_official_test()
    return BASE


def sealed_history():
    """Check existing historical locks; never manufacture a replacement history."""
    b = get_base()
    b.check_sources()
    hist = read_json(DISC / "results" / "historical_manifest.json")
    manifest_path = DISC / "results" / "delivery_manifest.json"
    delivery = read_json(manifest_path)["files"]
    checked = {}
    for rel, expected in hist.items():
        p = (REPO / rel).resolve()
        if p.name == "test" and p.parent.name == "cifar-100-python":
            raise RuntimeError("History lock would require forbidden TEST access")
        if not p.is_relative_to(PARENT):
            raise RuntimeError("Unexpected path outside historical audit directory: " + str(p))
        assert p.is_file() and sha(p) == expected, ("Historical manifest mismatch", str(p))
        checked[str(p)] = expected
    for rel, entry in delivery.items():
        p = (DISC / rel).resolve()
        expected = entry["sha256"] if isinstance(entry, dict) else entry
        assert p.is_relative_to(DISC) and p.is_file() and sha(p) == expected, ("Historical delivery mismatch", str(p))
        checked[str(p)] = expected
    for p in (DISC / "results" / "historical_manifest.json", manifest_path):
        checked[str(p)] = sha(p)
    return checked


def verify_preflight():
    p = read_json(RESULTS / "preflight.json")
    assert p["status"] == "PASS" and p["source_hashes"] == sources()
    assert sealed_history() == read_json(RESULTS / "historical_inputs.json")
    guard_ok()
    return p


def selection(trace, target_labels, noise, q, protect):
    """Use the actual helper pools; preserve fallback indices in their original order."""
    pool = np.asarray(trace["pools"]["Low-VS"], dtype=bool)
    original = np.asarray(trace["masks"]["Low-VS"])
    reason = np.asarray(trace["fallback"])
    assert pool.shape == q.shape == noise.shape == (len(target_labels), 64)
    normal = reason == 0
    assert (pool[normal].sum(1) == 48).all()
    assert (pool[~normal].sum(1) == 64).all()
    rebuilt = np.argsort(-np.where(pool, noise, -np.inf), axis=1, kind="stable")[:, :16].astype(np.uint8)
    assert np.array_equal(rebuilt, original), "Trace pool does not reconstruct the exact original Low-VS mask"
    assert np.array_equal(original[~normal], trace["masks"]["Uniform-PerSample"][~normal])
    chosen = original.copy()
    protected = np.zeros_like(pool)
    applied = (np.asarray(target_labels) >= 71) & normal & bool(protect)
    if applied.any():
        picked, excluded = policy.protected_lowvs_mask(
            torch.from_numpy(pool[applied].copy()), torch.from_numpy(q[applied].copy()),
            torch.from_numpy(noise[applied].copy()))
        picked_np = picked.numpy()
        protected[applied] = excluded.numpy()
        chosen[applied] = np.argsort(-np.where(picked_np, noise[applied], -np.inf),
                                     axis=1, kind="stable")[:, :16].astype(np.uint8)
    actual_bool = np.eye(64, dtype=bool)[chosen].any(1)
    original_bool = np.eye(64, dtype=bool)[original].any(1)
    assert (actual_bool.sum(1) == 16).all()
    assert not (actual_bool & ~pool).any()
    assert not (actual_bool & protected).any()
    assert (protected[applied].sum(1) == 16).all() and not protected[~applied].any()
    assert np.array_equal(chosen[~applied], original[~applied]), "Fallback or non-tail changed"
    if applied.any():
        improvement = (q * (actual_bool.astype(np.int8) - original_bool.astype(np.int8))).sum(1)
        assert (improvement[applied] >= -1e-10).all(), "Protection worsens fixed-q sum"
    return chosen, protected, applied


def unit():
    import test_policy
    tests = [getattr(test_policy, name) for name in sorted(dir(test_policy)) if name.startswith("test_") and callable(getattr(test_policy, name))]
    for test in tests:
        test()
    rng = np.random.default_rng(20261019)
    noise = rng.gumbel(size=(6, 64))
    pool = np.zeros((6, 64), bool)
    pool[:, :48] = True
    fallback = np.array([0, 0, 1, 2, 0, 0], dtype=np.uint8)
    pool[fallback > 0] = True
    original = np.argsort(-np.where(pool, noise, -np.inf), axis=1, kind="stable")[:, :16].astype(np.uint8)
    uniform = np.argsort(-noise, axis=1, kind="stable")[:, :16].astype(np.uint8)
    trace = {"pools": {"Low-VS": pool}, "masks": {"Low-VS": original, "Uniform-PerSample": uniform}, "fallback": fallback}
    target = np.array([5, 71, 80, 90, 35, 99])
    q = rng.normal(size=(6, 64))
    baseline, _, _ = selection(trace, target, noise, q, False)
    picked, protected, applied = selection(trace, target, noise, q, True)
    assert np.array_equal(baseline, original)
    assert np.array_equal(picked[[0, 2, 3, 4]], original[[0, 2, 3, 4]])
    assert applied.tolist() == [False, True, False, False, False, True]
    new_json(RESULTS / "unit.json", {"status": "PASS", "provided_policy_tests": len(tests),
             "additional_fallback_pool_coupling_test": True, "TEST_used": False, "source_hashes": sources()})
    print("UNIT PASS: 4 provided policy tests plus exact pool/fallback/non-tail coupling.", flush=True)


def historical_rows(seed):
    b = get_base()
    paths = [DISC / "results" / f"seed{seed}" / "LowVS" / "batch_signatures.csv",
             b.OLD / f"lowvs_seed{seed}" / "batches.csv"]
    rows = [read_csv(p) for p in paths]
    assert all(len(r) == 850 for r in rows)
    return rows


def preflight():
    b = get_base()
    assert read_json(RESULTS / "unit.json")["status"] == "PASS"
    assert read_json(RESULTS / "unit.json")["source_hashes"] == sources()
    if not torch.cuda.is_available():
        raise RuntimeError("Required CUDA environment missing; stop, do not replace environment")
    b.configure_compute()
    history = sealed_history()
    split = np.load(OLD / "split_indices.npz", allow_pickle=False)
    assert len(split["train_raw_indices"]) == N and len(split["validation_raw_indices"]) == 1980
    assert not np.intersect1d(split["train_raw_indices"], split["validation_raw_indices"]).size
    checks = {}
    for seed in SEEDS:
        y, vy, vbank, oldplan, noise = b.load_data(seed)
        legacy, clean = historical_rows(seed)
        for i, row in enumerate(legacy):
            assert int(row["epoch"]) == i // 85 + 1 and int(row["batch"]) == i % 85
        cfg, tr, bb, cls, CE, initial, s1 = b.fresh(seed)
        before = {k: v.detach().cpu().clone() for k, v in bb.state_dict().items()}
        b.warmup(tr, bb)
        aug = torch.Generator().manual_seed(seed + 300009)
        rec, don = oldplan["receivers"][0, :BATCH], oldplan["donors"][0, :BATCH]
        ri, di = b.augment(rec, don, aug, tr["train_transform"])
        with torch.no_grad():
            receiver, donor = bb(ri.cuda()), bb(di.cuda())
        g = noise[0, :BATCH]
        trace = b.online_masks(receiver, y[don], g)
        sig = b.signatures(rec, don, y, ri, di, receiver, donor, g, trace)
        sig["uniform_mask_sha256"] = b.array_hash(trace["masks"]["Uniform-PerSample"])
        sig["lowvs_mask_sha256"] = b.array_hash(trace["masks"]["Low-VS"])
        assert all(sig[k] == legacy[0][k] for k in sig), ("Historical first-batch parity failure", seed)
        assert b.array_hash(aug.get_state().numpy()) == clean[0]["augmentation_state_after_sha256"]
        target = torch.from_numpy(y[rec].copy()).long().cuda()
        q_t, j_star, z0 = policy.channel_margin_delta(receiver, donor, target, cls)
        q = q_t.cpu().numpy().astype(np.float64)
        losses = {}
        for protected in (False, True):
            chosen, excluded, applied = selection(trace, y[rec], g, q, protected)
            mixed, mask = b.fuse(receiver, donor, chosen)
            full = mask[:, :, None, None].expand_as(mixed)
            assert torch.equal(mixed[full], donor[full]) and torch.equal(mixed[~full], receiver[~full])
            with torch.no_grad():
                z = cls(mixed)
                loss = CE(cfg, np.bincount(y, minlength=100)).cuda()(z, target, extra_info=None)
                assert torch.isfinite(loss) and z.shape == (len(rec), 100)
                rows_t = torch.arange(len(target), device=target.device)
                exact = z[rows_t, target] - z[rows_t, j_star] - z0[rows_t, target] + z0[rows_t, j_star]
                predicted = (q_t * mask).sum(1)
                torch.testing.assert_close(exact, predicted, rtol=2e-5, atol=2e-5)
                losses[str(protected)] = float(loss)
        b.check_frozen(bb, before)
        assert b.state_hash(cls) == initial["classifier"]
        checks[str(seed)] = {"initial": initial, "StageI_sha256": b.sha(s1), "BN_sha256": bn_hash(bb),
                             "historical_first_batch_signature": sig, "forward_losses_no_update": losses,
                             "train_plan_sha256": b.sha(b.plan_dir(seed) / "sampling_plan.npz"),
                             "gumbel_sha256": b.sha(b.plan_dir(seed) / "gumbel_noise_float64.npy")}
        print(f"PREFLIGHT seed={seed}: historical signature and actual trace/fallback/fusion validated", flush=True)
    assert sealed_history() == history
    new_json(RESULTS / "historical_inputs.json", history)
    new_json(RESULTS / "environment.json", b.environment())
    new_json(RESULTS / "preflight.json", {"status": "PASS", "checks": checks, "source_hashes": sources(),
             "history_file_count": len(history), "TRAIN_VAL_overlap": 0, "TEST_used": False,
             "runtime_TEST_guard_events": b.AUDIT_EVENTS, "initial_read_only_query_note": INITIAL_READ_ONLY_VCS_QUERIES,
             "pairing_scope": "receiver/RNG/Gumbel across all arms; donor/VS within A-C and B-D; non-tail across all"})
    guard_ok()
    print("PREFLIGHT PASS; no optimizer updates.", flush=True)


def load_bank_if_eligible(y, raw, s1_sha, eval_transform):
    meta_path = BANK_OLD.with_name("reference_bank_provenance.json")
    if not BANK_OLD.is_file() or not meta_path.is_file():
        return None, ["optional historical bank or provenance absent"]
    meta = read_json(meta_path)
    reasons = []
    if meta.get("stage1_sha256") != s1_sha:
        reasons.append("historical bank Stage-I SHA mismatch")
    if meta.get("eval_transform") != eval_transform:
        reasons.append("historical bank eval transform mismatch")
    if meta.get("TEST_used") is not False:
        reasons.append("historical bank does not certify TRAIN-only input")
    if meta.get("bank_sha256") != sha(BANK_OLD):
        reasons.append("historical bank no longer matches its provenance")
    with np.load(BANK_OLD, allow_pickle=False) as data:
        needed = ("features", "labels", "raw_indices", "stage1_sha256", "split")
        if any(k not in data for k in needed):
            reasons.append("historical bank lacks index/label/checkpoint evidence")
            return None, reasons
        x = data["features"]
        if x.shape != (N, 64) or x.dtype != np.float32 or not np.isfinite(x).all():
            reasons.append("historical bank feature shape/type/finiteness mismatch")
        if not np.array_equal(data["labels"], y) or not np.array_equal(data["raw_indices"], raw):
            reasons.append("historical bank labels/raw TRAIN index order mismatch")
        if str(data["stage1_sha256"]) != s1_sha or str(data["split"]) != "TRAIN only":
            reasons.append("historical bank embedded TRAIN/checkpoint provenance mismatch")
        return (x.copy() if not reasons else None), reasons


def prepare():
    b = get_base()
    pre = verify_preflight()
    outputs = [RESULTS / n for n in ("train_gap_bank.npz", "head_prototypes.npz", "train_bank_provenance.json",
                                     "donor_plan_seed124.npz", "donor_plan_seed125.npz",
                                     "donor_plan_summary.csv", "donor_stratum_summary.csv", "prepare.json")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("A planned prepare output already exists")
    y = np.load(OLD / "train_labels.npy", allow_pickle=False)
    split = np.load(OLD / "split_indices.npz", allow_pickle=False)
    raw = split["train_raw_indices"]
    counts = np.bincount(y, minlength=100)
    assert y.shape == raw.shape == (N,) and not np.intersect1d(raw, split["validation_raw_indices"]).size
    cfg, tr, bb, cls, CE, initial, s1 = b.fresh(124)
    before = {k: v.detach().cpu().clone() for k, v in bb.state_dict().items()}
    eval_text = str(tr["eval_transform"])
    x, not_eligible = load_bank_if_eligible(y, raw, b.sha(s1), eval_text)
    reused = x is not None
    if reused:
        with output_path(RESULTS / "train_gap_bank.npz").open("xb") as handle:
            handle.write(BANK_OLD.read_bytes())
    else:
        images = np.load(OLD / "train_images.npy", mmap_mode="r", allow_pickle=False)
        assert images.shape == (N, 32, 32, 3)
        chunks = []
        with torch.no_grad():
            for start in range(0, N, BATCH):
                tensor = torch.stack([tr["eval_transform"](Image.fromarray(im))
                                      for im in images[start:start + BATCH]]).cuda()
                feature = bb(tensor)
                chunks.append(F.avg_pool2d(feature, 8).flatten(1).cpu().numpy())
        x = np.concatenate(chunks).astype(np.float32)
        assert x.shape == (N, 64) and np.isfinite(x).all()
        new_npz(RESULTS / "train_gap_bank.npz", features=x, labels=y, raw_indices=raw,
                stage1_sha256=np.array(b.sha(s1)), split=np.array("TRAIN only"))
    head_ids, prototypes = policy.build_head_prototypes(x, y)
    strata = policy.build_frequency_strata(counts)
    affinity = policy.receiver_prototype_affinity(x, head_ids, prototypes)
    assert head_ids.tolist() == list(range(36)) and prototypes.shape == (36, 64)
    assert all(len(s) == 6 for s in strata)
    bank_sha = sha(RESULTS / "train_gap_bank.npz")
    new_npz(RESULTS / "head_prototypes.npz", head_ids=head_ids, prototypes=prototypes,
            class_counts=counts, train_raw_indices=raw, stage1_sha256=np.array(b.sha(s1)))
    b.check_frozen(bb, before)
    assert b.state_hash(bb) == initial["backbone"] and b.state_hash(cls) == initial["classifier"]
    new_json(RESULTS / "train_bank_provenance.json", {
        "shape": list(x.shape), "StageI_sha256": b.sha(s1), "bank_sha256": bank_sha,
        "reused_existing_bank": reused, "existing_bank_path": str(BANK_OLD) if reused else None,
        "optional_bank_ineligible_reasons": not_eligible,
        "train_labels_sha256": b.array_hash(y), "train_raw_indices_sha256": b.array_hash(raw),
        "eval_transform": eval_text, "class_counts": counts.tolist(), "prototype_classes": head_ids.tolist(),
        "head_prototypes_sha256": sha(RESULTS / "head_prototypes.npz"),
        "backbone_before": initial["backbone"], "backbone_after": b.state_hash(bb),
        "BN_bitwise_unchanged": True, "classifier_unchanged": True,
        "validation_features_used_for_bank_or_prototypes": False, "TEST_used": False})
    donor_rows, stratum_rows, prepared = [], [], {}
    for seed in SEEDS:
        with np.load(b.plan_dir(seed) / "sampling_plan.npz", allow_pickle=False) as oldplan:
            receiver = oldplan["receivers"].copy()
            original = oldplan["donors"].copy()
        assert receiver.shape == original.shape == (10, N)
        assert ((receiver >= 0) & (receiver < N)).all() and ((original >= 0) & (original < N)).all()
        tail = y[receiver] >= 71
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261019, 1]))
        us, uc, ui = [rng.random(receiver.shape) for _ in range(3)]
        aff = affinity[receiver[tail]]
        random_cls, random_s, random_a = policy.choose_donor_classes(
            aff, head_ids, counts, strata, us[tail], uc[tail], "random")
        high_cls, high_s, high_a = policy.choose_donor_classes(
            aff, head_ids, counts, strata, us[tail], uc[tail], "high")
        assert np.array_equal(random_s, high_s)
        plans = {"receivers": receiver, "original_donors": original, "tail_override": tail,
                 "u_stratum": us, "u_class": uc, "u_instance": ui,
                 "train_raw_indices": raw, "stage1_sha256": np.array(b.sha(s1)), "bank_sha256": np.array(bank_sha)}
        for letter, chosen_classes, chosen_strata, chosen_affinity in (
                ("A", random_cls, random_s, random_a), ("B", high_cls, high_s, high_a),
                ("C", random_cls, random_s, random_a), ("D", high_cls, high_s, high_a)):
            donors = original.copy()
            donors[tail] = policy.sample_donor_indices(chosen_classes, y, ui[tail])
            classes = y[original].copy()
            classes[tail] = chosen_classes
            st = np.full(receiver.shape, -1, dtype=np.int64)
            st[tail] = chosen_strata
            af = np.zeros(receiver.shape, dtype=np.float32)
            af[tail] = chosen_affinity
            assert np.array_equal(donors[~tail], original[~tail])
            assert np.array_equal(y[donors], classes) and ((donors >= 0) & (donors < N)).all()
            assert ((classes[tail] >= 0) & (classes[tail] <= 35)).all()
            plans.update({f"donors_{letter}": donors, f"classes_{letter}": classes,
                          f"stratum_{letter}": st, f"affinity_{letter}": af})
            arm = ARMS["ABCD".index(letter)]
            for c in range(100):
                selected = tail & (classes == c)
                donor_rows.append({"seed": seed, "arm": arm, "donor_class": c, "train_class_count": int(counts[c]),
                                   "all_receiver_donor_draws": int((classes == c).sum()),
                                   "tail_receiver_donor_draws": int(selected.sum()),
                                   "non_tail_receiver_donor_draws": int(((~tail) & (classes == c)).sum()),
                                   "mean_tail_affinity": float(af[selected].mean()) if selected.any() else None})
            masses = np.array([counts[s].sum() for s in strata], dtype=np.float64)
            for si, ids in enumerate(strata):
                at = tail & (st == si)
                stratum_rows.append({"seed": seed, "arm": arm, "stratum": si,
                                     "head_classes": json.dumps(ids.tolist()),
                                     "stratum_prior": float(masses[si] / masses.sum()),
                                     "tail_draws": int(at.sum()), "all_tail_draws": int(tail.sum()),
                                     "mean_affinity": float(af[at].mean()) if at.any() else None})
        assert np.array_equal(plans["donors_A"], plans["donors_C"])
        assert np.array_equal(plans["donors_B"], plans["donors_D"])
        assert np.array_equal(plans["stratum_A"], plans["stratum_B"])
        path = RESULTS / f"donor_plan_seed{seed}.npz"
        new_npz(path, **plans)
        prepared[str(seed)] = {"path": path.name, "sha256": sha(path), "tail_draws": int(tail.sum()),
                              "receivers_sha256": b.array_hash(receiver),
                              "common_uniforms_sha256": b.array_hash(us, uc, ui),
                              "original_sampling_plan_sha256": b.sha(b.plan_dir(seed) / "sampling_plan.npz")}
        print(f"PREPARE seed={seed}: paired TRAIN-only plans; tail draws={int(tail.sum())}", flush=True)
    new_csv(RESULTS / "donor_plan_summary.csv", donor_rows)
    new_csv(RESULTS / "donor_stratum_summary.csv", stratum_rows)
    new_json(RESULTS / "prepare.json", {"status": "PASS", "source_hashes": sources(),
             "bank_sha256": bank_sha, "prototypes_sha256": sha(RESULTS / "head_prototypes.npz"),
             "seed_uniform_stream": "numpy SeedSequence([seed,20261019,1]); us/uc/ui each [10,N]",
             "plans": prepared, "class_frequency_caveat": "Stratum IDs paired; within-stratum donor-class frequencies can differ.",
             "non_tail_affinity_sentinel": "affinity=0 and stratum=-1 for unmodified non-tail draws; affinity undefined there",
             "TEST_used": False, "validation_used_for_treatments": False})
    guard_ok()
    print("PREPARE PASS; original banks/plans/Gumbel untouched.", flush=True)


def load_prepared(seed):
    b = get_base()
    prepared = read_json(RESULTS / "prepare.json")
    assert prepared["status"] == "PASS" and prepared["source_hashes"] == sources()
    assert sha(RESULTS / "train_gap_bank.npz") == prepared["bank_sha256"]
    assert sha(RESULTS / "head_prototypes.npz") == prepared["prototypes_sha256"]
    path = RESULTS / prepared["plans"][str(seed)]["path"]
    assert sha(path) == prepared["plans"][str(seed)]["sha256"]
    with np.load(path, allow_pickle=False) as data:
        plan = {k: data[k] for k in data.files}
    with np.load(b.plan_dir(seed) / "sampling_plan.npz", allow_pickle=False) as oldplan:
        assert np.array_equal(plan["receivers"], oldplan["receivers"])
        assert np.array_equal(plan["original_donors"], oldplan["donors"])
    return prepared, plan


def batch_signatures(rec, don, y, ri, di, receiver, donor, g, trace, aug_before, aug_after, chosen, protected):
    b = get_base()
    tail = y[rec] >= 71
    keep = ~tail
    rn, dn = receiver.detach().cpu().numpy(), donor.detach().cpu().numpy()
    sig = b.signatures(rec, don, y, ri, di, receiver, donor, g, trace)
    sig.update({
        "receiver_ids_labels_sha256": b.array_hash(rec, y[rec]),
        "donor_ids_labels_sha256": b.array_hash(don, y[don]),
        "receiver_pixels_sha256": b.array_hash(ri.numpy()),
        "donor_pixels_sha256": b.array_hash(di.numpy()),
        "receiver_features_sha256": b.array_hash(rn), "donor_features_sha256": b.array_hash(dn),
        "augmentation_rng_before_sha256": b.array_hash(aug_before.numpy()),
        "augmentation_state_after_sha256": b.array_hash(aug_after.numpy()),
        "original_lowvs_mask_sha256": b.array_hash(trace["masks"]["Low-VS"]),
        "original_pool_sha256": b.array_hash(trace["pools"]["Low-VS"]),
        "actual_mask_sha256": b.array_hash(chosen),
        "protected_mask_sha256": b.array_hash(protected),
        "fallback_sha256": b.array_hash(trace["fallback"]),
        "non_tail_donor_sha256": b.array_hash(don[keep], y[don[keep]]),
        "non_tail_donor_pixels_sha256": b.array_hash(di.numpy()[keep]),
        "non_tail_donor_features_sha256": b.array_hash(dn[keep]),
        "non_tail_scores_sha256": b.array_hash(trace["scores"][keep]),
        "non_tail_original_mask_sha256": b.array_hash(trace["masks"]["Low-VS"][keep]),
        "non_tail_actual_mask_sha256": b.array_hash(chosen[keep]),
    })
    return sig


COMMON_KEYS = ("receiver_ids_labels_sha256", "receiver_pixels_sha256", "receiver_features_sha256",
               "augmentation_rng_before_sha256", "augmentation_state_after_sha256", "gumbel_sha256",
               "non_tail_donor_sha256", "non_tail_donor_pixels_sha256", "non_tail_donor_features_sha256",
               "non_tail_scores_sha256", "non_tail_original_mask_sha256", "non_tail_actual_mask_sha256")
DONOR_PAIR_KEYS = ("indices_sha256", "images_sha256", "features_sha256", "scores_sha256",
                   "donor_ids_labels_sha256", "donor_pixels_sha256", "donor_features_sha256",
                   "original_lowvs_mask_sha256", "original_pool_sha256", "fallback_sha256")


def compare_keys(current, reference, keys):
    for key in keys:
        if current[key] != reference[key]:
            raise RuntimeError("Paired input assertion failed for " + key)


def quantiles(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    names = ("mean", "min", "q05", "q25", "q50", "q75", "q95", "max")
    if not values.size:
        return {prefix + "_" + k: None for k in names}
    points = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    vals = [float(values.mean()), *map(float, points)]
    return {prefix + "_" + k: value for k, value in zip(names, vals)}


def write_geometry(sink, *, epoch, batch, start, rec, don, y, raw, plan, letter,
                   target, q_t, j_star, clean_logits, fused_logits, mask, original, protected, applied, trace):
    rows_t = torch.arange(len(target), device=target.device)
    with torch.no_grad():
        clean_margin = clean_logits[rows_t, target] - clean_logits[rows_t, j_star]
        detached = fused_logits.detach()
        fused_fixed = detached[rows_t, target] - detached[rows_t, j_star]
        others = detached.clone()
        others[rows_t, target] = -torch.inf
        fused_dynamic = detached[rows_t, target] - others.max(1).values
        delta_ce = F.cross_entropy(detached, target, reduction="none") - F.cross_entropy(clean_logits, target, reduction="none")
        exact = fused_fixed - clean_margin
        predicted = (q_t * mask).sum(1)
        torch.testing.assert_close(exact, predicted, rtol=2e-5, atol=2e-5)
    q = q_t.detach().cpu().numpy().astype(np.float64)
    cm, fx, dy, ce = [v.detach().cpu().numpy() for v in (clean_margin, exact, fused_dynamic - clean_margin, delta_ce)]
    competitor = j_star.cpu().numpy()
    selected = mask.cpu().numpy()
    original_bool = np.eye(64, dtype=bool)[original].any(1)
    tail = np.flatnonzero(y[rec] >= 71)
    class_counts = np.bincount(y, minlength=100)
    for i in tail:
        chosen_q = q[i, selected[i]]
        protected_q = q[i, protected[i]]
        sink.row({"epoch": epoch + 1, "batch": batch, "draw": start + int(i),
                  "receiver_index": int(rec[i]), "receiver_raw_index": int(raw[rec[i]]),
                  "receiver_label": int(y[rec[i]]), "donor_index": int(don[i]), "donor_raw_index": int(raw[don[i]]),
                  "donor_class": int(y[don[i]]), "donor_train_count": int(class_counts[y[don[i]]]),
                  "stratum": int(plan[f"stratum_{letter}"][epoch, start + i]),
                  "affinity": float(plan[f"affinity_{letter}"][epoch, start + i]),
                  "competitor_id": int(competitor[i]), "clean_margin": float(cm[i]),
                  "fixed_competitor_margin_delta": float(fx[i]), "dynamic_margin_delta": float(dy[i]),
                  "fused_minus_clean_CE": float(ce[i]),
                  "positive_clean_margin": bool(cm[i] > 0),
                  "positive_to_nonpositive": bool(cm[i] > 0 and cm[i] + dy[i] <= 0),
                  "selected_q_sum": float(chosen_q.sum()), "original_lowvs_q_sum": float(q[i, original_bool[i]].sum()),
                  "selected_q_gain_vs_original": float(chosen_q.sum() - q[i, original_bool[i]].sum()),
                  "fallback_reason": int(trace["fallback"][i]), "protection_applied": bool(applied[i]),
                  "protected_count": int(protected[i].sum()), "selected_count": int(selected[i].sum()),
                  **quantiles(chosen_q, "selected_q"), **quantiles(protected_q, "protected_q")})
    tmask = y[rec] >= 71
    clean_positive = tmask & (cm > 0)
    crossed = clean_positive & (cm + dy <= 0)
    return {"tail_draws": int(tmask.sum()), "protection_applied_draws": int(applied.sum()),
            "tail_fallback_draws": int(((trace["fallback"] > 0) & tmask).sum()),
            "tail_fixed_margin_delta_mean": float(fx[tmask].mean()) if tmask.any() else None,
            "tail_dynamic_margin_delta_mean": float(dy[tmask].mean()) if tmask.any() else None,
            "tail_delta_CE_mean": float(ce[tmask].mean()) if tmask.any() else None,
            "tail_selected_q_sum_mean": float((q * selected).sum(1)[tmask].mean()) if tmask.any() else None,
            "tail_original_q_sum_mean": float((q * original_bool).sum(1)[tmask].mean()) if tmask.any() else None,
            "tail_clean_positive": int(clean_positive.sum()), "tail_positive_to_nonpositive": int(crossed.sum()),
            "tail_conditional_cross_rate": float(crossed.sum() / clean_positive.sum()) if clean_positive.any() else None,
            **quantiles(cm[tmask], "tail_clean_margin"),
            **quantiles(q[protected], "protected_q"),
            **quantiles(q[selected & tmask[:, None]], "selected_q")}


def train(seed, arm):
    pre = verify_preflight()
    prep, plan = load_prepared(seed)
    b = get_base()
    output = RESULTS / f"seed{seed}" / arm
    if output.exists():
        raise FileExistsError(f"Refuse overwrite/resume: {output}")
    order = [(s, a) for s in SEEDS for a in ARMS]
    for s, a in order[:order.index((seed, arm))]:
        prior = read_json(RESULTS / f"seed{s}" / a / "summary.json")
        assert prior["status"] == "completed" and prior["updates"] == 850
    y, val_y, val_maps, original_plan, noise = b.load_data(seed)
    raw = np.load(OLD / "split_indices.npz", allow_pickle=False)["train_raw_indices"].copy()
    _, historical_augmentation = historical_rows(seed)
    cfg, tr, bb, cls, CE, initial, s1 = b.fresh(seed)
    assert initial == pre["checks"][str(seed)]["initial"]
    assert sha(s1) == pre["checks"][str(seed)]["StageI_sha256"]
    frozen = {k: v.detach().cpu().clone() for k, v in bb.state_dict().items()}
    fc_initial = {k: v.detach().cpu().clone() for k, v in cls.state_dict().items()}
    initial_bn = bn_hash(bb)
    optimizer = b.optimizer_for(cfg, cls)
    assert not optimizer.state
    schedule = b.scheduler()
    criterion = CE(cfg, np.bincount(y, minlength=100)).cuda()
    b.warmup(tr, bb)
    b.check_frozen(bb, frozen)
    aug = torch.Generator().manual_seed(seed + 300009)
    letter, use_protection = LETTERS[arm], arm in ARMS[2:]
    control = read_csv(RESULTS / f"seed{seed}" / ARMS[0] / "batch_signatures.csv") if arm != ARMS[0] else None
    donor_control_arm = ARMS[0] if arm == ARMS[2] else ARMS[1] if arm == ARMS[3] else None
    donor_control = read_csv(RESULTS / f"seed{seed}" / donor_control_arm / "batch_signatures.csv") if donor_control_arm else None
    for reference in (control, donor_control):
        if reference is not None:
            assert len(reference) == 850
    output.mkdir(parents=True)
    new_json(output / "config.json", {
        "seed": seed, "arm": arm, "factorial_letter": letter, "epochs": 10,
        "updates_per_epoch": 85, "total_updates": 850, "batch_size": 128, "last_batch_size": 95,
        "dataset": "CIFAR-100-LT TRAIN IF100", "N_train": N, "N_validation": 1980,
        "validation": "original TRAIN holdout, classes1..99 x20, class0 absent;100 logits",
        "StageI_sha256": sha(s1), "StageI_seed": 42, "StageI_epoch": 184,
        "initial_hashes": initial, "initial_BN_hash": initial_bn,
        "BN_and_backbone_frozen": True, "classifier_parameters": sum(p.numel() for p in cls.parameters()),
        "optimizer": "SGD", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005,
        "loss": "official CE, no auxiliary loss", "rho": 0.25, "channels": "16 whole maps /64",
        "VS_policy": "unmodified original online_masks Low-VS; original fallback exact",
        "tail_override": "receiver_label>=71", "protection": use_protection,
        "protection_rule": "lowest16 current fixed-competitor q inside original Bottom48; Gumbel16 from remaining32; fallback unchanged",
        "augmentation_generator_seed": seed + 300009,
        "donor_plan_sha256": prep["plans"][str(seed)]["sha256"],
        "source_hashes": sources(), "TEST_used": False, "effective_original_yaml": cfg.dump(),
    })
    new_text(output / "effective.yaml", cfg.dump())
    signatures = CsvSink(output / "batch_signatures.csv")
    geometry = CsvSink(output / "geometry_diagnostics.csv")
    epochs_csv = CsvSink(output / "epochs.csv")
    started = time.perf_counter()
    updates = 0
    epoch_rows = []
    matching = {"receiver_stream_and_non_tail_batches": 0, "historical_augmentation_rng_batches": 0,
                "donor_VS_pair_batches": 0, "mask_integrity_batches": 0}
    peak_allocated, peak_reserved = 0.0, 0.0
    try:
        for epoch in range(EPOCHS):
            cls.train()
            schedule(optimizer, epoch, cfg)
            torch.cuda.reset_peak_memory_stats()
            epoch_started = time.perf_counter()
            loss_sum = 0.0
            epoch_tail, epoch_protected, epoch_fallback = 0, 0, 0
            for batch, start in enumerate(range(0, N, BATCH)):
                end = min(start + BATCH, N)
                rec = plan["receivers"][epoch, start:end]
                don = plan[f"donors_{letter}"][epoch, start:end]
                assert np.array_equal(rec, original_plan["receivers"][epoch, start:end])
                non_tail = y[rec] < 71
                assert np.array_equal(don[non_tail], original_plan["donors"][epoch, start:end][non_tail])
                target = torch.from_numpy(y[rec].copy()).cuda()
                aug_before = aug.get_state().clone()
                ri, di = b.augment(rec, don, aug, tr["train_transform"])
                aug_after = aug.get_state().clone()
                g = noise[epoch, start:end]
                with torch.no_grad():
                    receiver, donor = bb(ri.cuda()), bb(di.cuda())
                    assert receiver.shape == donor.shape == (end - start, 64, 8, 8)
                    assert not receiver.requires_grad and not donor.requires_grad
                    trace = b.online_masks(receiver, y[don], g)
                    q, competitor, clean_logits = policy.channel_margin_delta(receiver, donor, target, cls)
                    q_np = q.cpu().numpy().astype(np.float64)
                    chosen, protected, applied = selection(trace, y[rec], g, q_np, use_protection)
                    fused, copy_mask = b.fuse(receiver, donor, chosen)
                    full_mask = copy_mask[:, :, None, None].expand_as(fused)
                    assert torch.equal(fused[full_mask], donor[full_mask])
                    assert torch.equal(fused[~full_mask], receiver[~full_mask])
                    assert torch.equal(copy_mask.sum(1), torch.full((len(rec),), 16, device=copy_mask.device))
                    assert np.array_equal(target.cpu().numpy(), y[rec])
                sig = batch_signatures(rec, don, y, ri, di, receiver, donor, g, trace,
                                       aug_before, aug_after, chosen, protected)
                expected_old = historical_augmentation[updates]
                assert int(expected_old["epoch"]) == epoch + 1 and int(expected_old["batch"]) == batch
                assert sig["augmentation_state_after_sha256"] == expected_old["augmentation_state_after_sha256"]
                matching["historical_augmentation_rng_batches"] += 1
                if control is not None:
                    compare_keys(sig, control[updates], COMMON_KEYS)
                matching["receiver_stream_and_non_tail_batches"] += 1
                if donor_control is not None:
                    compare_keys(sig, donor_control[updates], DONOR_PAIR_KEYS)
                    matching["donor_VS_pair_batches"] += 1
                logits = cls(fused)
                loss = criterion(logits, target, extra_info=None)
                assert torch.isfinite(logits).all() and torch.isfinite(loss)
                geom = write_geometry(geometry, epoch=epoch, batch=batch, start=start, rec=rec, don=don,
                                      y=y, raw=raw, plan=plan, letter=letter, target=target, q_t=q,
                                      j_star=competitor, clean_logits=clean_logits, fused_logits=logits,
                                      mask=copy_mask, original=trace["masks"]["Low-VS"], protected=protected,
                                      applied=applied, trace=trace)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cls.parameters())
                assert all(p.grad is None and not p.requires_grad for p in bb.parameters())
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in cls.parameters())
                updates += 1
                matching["mask_integrity_batches"] += 1
                signatures.row({"epoch": epoch + 1, "batch": batch, "size": len(rec), **sig,
                                "q_sha256": b.array_hash(q_np), "loss": float(loss.detach()), **geom})
                loss_sum += float(loss.detach()) * len(rec)
                epoch_tail += geom["tail_draws"]
                epoch_protected += geom["protection_applied_draws"]
                epoch_fallback += geom["tail_fallback_draws"]
                if batch in (0, 40, 84):
                    print(f"seed={seed} {letter} epoch={epoch+1}/10 batch={batch+1}/85 updates={updates} loss={float(loss.detach()):.6f}", flush=True)
            signatures.flush(); geometry.flush()
            torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - epoch_started
            b.check_frozen(bb, frozen)
            assert bn_hash(bb) == initial_bn and b.state_hash(bb) == initial["backbone"]
            changed = {k: float((v.detach().cpu() - fc_initial[k]).abs().max()) for k, v in cls.state_dict().items()}
            assert all(v > 0 for v in changed.values())
            val, val_logits, val_pred = b.evaluate(cls, val_y, feature_maps=val_maps)
            alloc, reserve = torch.cuda.max_memory_allocated() / 1024**2, torch.cuda.max_memory_reserved() / 1024**2
            peak_allocated, peak_reserved = max(peak_allocated, alloc), max(peak_reserved, reserve)
            row = {"seed": seed, "arm": arm, "epoch": epoch + 1, "updates": updates,
                   "lr": optimizer.param_groups[0]["lr"], "train_CE": loss_sum / N,
                   **{k: float(val[k]) for k in METRICS}, "training_seconds": epoch_seconds,
                   "tail_draws": epoch_tail, "protected_draws": epoch_protected, "tail_fallback_draws": epoch_fallback,
                   "peak_allocated_mib": alloc, "peak_reserved_mib": reserve}
            epoch_rows.append(row); epochs_csv.row(row); epochs_csv.flush()
            print(f"EPOCH seed={seed} {letter} epoch={epoch+1} Overall={val['Overall']:.4f} Head={val['Head']:.4f} Tail={val['Tail']:.4f} Few={val['Few']:.4f} seconds={epoch_seconds:.2f}", flush=True)
            if epoch == 9:
                payload = {"epoch": 10, "seed": seed, "arm": arm, "classifier": {k: v.detach().cpu().clone() for k, v in cls.state_dict().items()},
                           "StageI_sha256": sha(s1), "TEST_used": False, "updates": updates}
                with output_path(output / "final_fc.pth").open("xb") as handle:
                    torch.save(payload, handle)
                new_npz(output / "validation_epoch10.npz", labels=val_y, logits=val_logits, predictions=val_pred)
            guard_ok()
    finally:
        signatures.close(); geometry.close(); epochs_csv.close()
    assert updates == 850 and len(epoch_rows) == 10
    assert all(v == 850 for k, v in matching.items() if k != "donor_VS_pair_batches")
    if donor_control is not None:
        assert matching["donor_VS_pair_batches"] == 850
    final_state_hash = b.state_hash(cls)
    payload = torch.load(output / "final_fc.pth", map_location="cpu", weights_only=False)
    cls.load_state_dict(payload["classifier"], strict=True)
    assert b.state_hash(cls) == final_state_hash
    val_reload, logits_reload, pred_reload = b.evaluate(cls, val_y, feature_maps=val_maps)
    assert np.array_equal(val_pred, pred_reload) and np.array_equal(val_logits, logits_reload)
    assert all(abs(val_reload[k] - epoch_rows[-1][k]) < 1e-9 for k in METRICS)
    b.check_frozen(bb, frozen)
    guard_ok()
    best = max(epoch_rows, key=lambda row: row["Overall"])
    new_json(output / "summary.json", {
        "status": "completed", "seed": seed, "arm": arm, "letter": letter, "updates": updates,
        "last": epoch_rows[-1], "best_epoch": best["epoch"], "best_Overall": best["Overall"],
        "best_Tail": best["Tail"], "best_is_diagnostic_only": True, "best_checkpoint_saved": False,
        "initial_state_hashes": initial, "final_classifier_sha256": final_state_hash,
        "classifier_max_abs_changes": changed, "classifier_parameters": sum(p.numel() for p in cls.parameters()),
        "backbone_state_entries_bitwise_unchanged": len(frozen), "backbone_parameters_changed": 0,
        "final_backbone_sha256": b.state_hash(bb), "BN_before_sha256": initial_bn, "BN_after_sha256": bn_hash(bb),
        "BN_counters": sorted({int(v) for k, v in frozen.items() if k.endswith("num_batches_tracked")}),
        "pairing_counts": matching, "validation_reload_bitwise_exact": True,
        "final_fc_file_sha256": sha(output / "final_fc.pth"),
        "validation_npz_sha256": sha(output / "validation_epoch10.npz"),
        "elapsed_seconds": time.perf_counter() - started, "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved, "TEST_used": False, "TEST_guard_events": 0,
        "source_hashes": sources(), "donor_plan_sha256": prep["plans"][str(seed)]["sha256"],
    })
    print(f"COMPLETE seed={seed} {letter} {arm}:10 epochs,850 updates; BB/BN unchanged; reload exact", flush=True)

def group_id(labels):
    labels = np.asarray(labels)
    return np.where(labels <= 35, 0, np.where(labels <= 70, 1, 2))


def report():
    verify_preflight()
    b = get_base()
    from common import metrics as calculate_metrics
    metrics_rows, paired_rows, class_rows, confusion_rows, transition_rows, effects, geometry_rows = [], [], [], [], [], [], []
    all_summaries, checks, donor_summaries = {}, {}, []
    train_y = np.load(OLD / "train_labels.npy", allow_pickle=False)
    counts = np.bincount(train_y, minlength=100)
    for seed in SEEDS:
        prepared, plan = load_prepared(seed)
        cache = {}
        sign = {}
        labels = None
        for arm in ARMS:
            folder = RESULTS / f"seed{seed}" / arm
            summary = read_json(folder / "summary.json")
            assert summary["status"] == "completed" and summary["updates"] == 850 and summary["source_hashes"] == sources()
            assert summary["TEST_used"] is False and summary["BN_before_sha256"] == summary["BN_after_sha256"]
            assert summary["initial_state_hashes"]["backbone"] == summary["final_backbone_sha256"]
            assert sha(folder / "final_fc.pth") == summary["final_fc_file_sha256"]
            assert sha(folder / "validation_epoch10.npz") == summary["validation_npz_sha256"]
            epochs = read_csv(folder / "epochs.csv")
            assert len(epochs) == 10 and [int(r["updates"]) for r in epochs] == list(range(85, 851, 85))
            with np.load(folder / "validation_epoch10.npz", allow_pickle=False) as saved:
                this_y, pred, logits = [saved[k].copy() for k in ("labels", "predictions", "logits")]
            if labels is None:
                labels = this_y
            assert np.array_equal(labels, this_y)
            assert logits.shape == (1980, 100) and np.isfinite(logits).all() and np.array_equal(pred, logits.argmax(1))
            assert np.bincount(labels, minlength=100).tolist() == [0] + [20] * 99
            met = calculate_metrics(pred, labels)
            for key in METRICS[:-1]:
                assert abs(met[key] - summary["last"][key]) < 1e-9
            ce_cpu = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)))
            assert abs(ce_cpu - summary["last"]["validation_CE"]) < 1e-5
            met["validation_CE"] = summary["last"]["validation_CE"]
            letter = LETTERS[arm]
            cache[letter] = {"pred": pred, "metrics": met}
            all_summaries[f"{seed}_{letter}"] = summary
            metrics_rows.append({"seed": seed, "arm": arm, "letter": letter, "role": "new_factorial_arm", "epoch": 10,
                                 **met, "best_epoch_diagnostic": summary["best_epoch"], "best_Overall_diagnostic": summary["best_Overall"],
                                 "updates": 850, "training_seconds": sum(float(r["training_seconds"]) for r in epochs),
                                 "elapsed_seconds": summary["elapsed_seconds"], "peak_allocated_mib": summary["peak_allocated_mib"],
                                 "peak_reserved_mib": summary["peak_reserved_mib"]})
            sign[letter] = read_csv(folder / "batch_signatures.csv")
            assert len(sign[letter]) == 850
            for i, row in enumerate(sign[letter]):
                assert int(row["epoch"]) == i // 85 + 1 and int(row["batch"]) == i % 85
                assert int(row["size"]) == (95 if i % 85 == 84 else 128)
            geo = read_csv(folder / "geometry_diagnostics.csv")
            assert len(geo) == prepared["plans"][str(seed)]["tail_draws"]
            assert all(int(r["selected_count"]) == 16 for r in geo)
            protected_geo = [r for r in geo if r["protection_applied"] == "True"]
            fallback_geo = [r for r in geo if int(r["fallback_reason"]) > 0]
            assert all(int(r["protected_count"]) == 0 for r in fallback_geo)
            assert all(int(r["protected_count"]) == 16 for r in protected_geo)
            gain = np.array([float(r["selected_q_gain_vs_original"]) for r in protected_geo])
            if len(gain):
                assert gain.min() >= -1e-10
            true_positive = sum(r["positive_clean_margin"] == "True" for r in geo)
            crossings = sum(r["positive_to_nonpositive"] == "True" for r in geo)
            grow = {"seed": seed, "arm": arm, "tail_draws": len(geo), "protected_draws": len(protected_geo),
                    "fallback_draws": len(fallback_geo), "clean_positive_tail_draws": true_positive,
                    "positive_to_nonpositive_draws": crossings,
                    "conditional_cross_rate": crossings / true_positive if true_positive else None,
                    **quantiles([float(r["clean_margin"]) for r in geo], "clean_margin"),
                    **quantiles([float(r["fixed_competitor_margin_delta"]) for r in geo], "fixed_margin_delta"),
                    **quantiles([float(r["dynamic_margin_delta"]) for r in geo], "dynamic_margin_delta"),
                    **quantiles([float(r["fused_minus_clean_CE"]) for r in geo], "delta_CE"),
                    **quantiles([float(r["selected_q_sum"]) for r in geo], "selected_q_sum"),
                    **quantiles(gain, "protected_gain_vs_same_head_original"),
                    **quantiles([float(r["protected_q_mean"]) for r in protected_geo], "per_draw_protected_q_mean")}
            geometry_rows.append(grow)
            del geo, protected_geo, fallback_geo
        _, historical_augmentation = historical_rows(seed)
        for i in range(850):
            for letter in "ABCD":
                compare_keys(sign[letter][i], sign["A"][i], COMMON_KEYS)
                assert sign[letter][i]["augmentation_state_after_sha256"] == historical_augmentation[i]["augmentation_state_after_sha256"]
            compare_keys(sign["C"][i], sign["A"][i], DONOR_PAIR_KEYS)
            compare_keys(sign["D"][i], sign["B"][i], DONOR_PAIR_KEYS)
        checks[str(seed)] = {"all_arm_common_batches": 850, "A_C_donor_VS_batches": 850, "B_D_donor_VS_batches": 850,
                             "all_arm_historical_augmentation_rng_batches": 850, "mask_integrity": True}
        # E is an existing replayed context, never another trained arm in this task.
        old_folder = DISC / "results" / f"seed{seed}" / "LowVS"
        old_summary = read_json(old_folder / "summary.json")
        with np.load(old_folder / "validation_epoch10.npz", allow_pickle=False) as historical:
            assert np.array_equal(historical["labels"], labels)
            old_pred = historical["predictions"].copy()
            assert np.array_equal(old_pred, historical["logits"].argmax(1))
        em = calculate_metrics(old_pred, labels)
        old_epoch = next(r for r in read_csv(DISC / "results" / "epoch10.csv") if int(r["seed"]) == seed and r["arm"] == "LowVS")
        for k in METRICS[:-1]:
            assert abs(em[k] - float(old_epoch[k])) < 1e-9
        em["validation_CE"] = float(old_epoch["validation_CE"])
        metrics_rows.append({"seed": seed, "arm": "Historical LowVS-Full", "letter": "E", "role": "historical_context_only", "epoch": 10,
                             **em, "best_epoch_diagnostic": old_summary.get("best_epoch_diagnostic"),
                             "best_Overall_diagnostic": old_summary.get("best_Overall_diagnostic"),
                             "updates": "historical", "training_seconds": None, "elapsed_seconds": None,
                             "peak_allocated_mib": None, "peak_reserved_mib": None})
        cache["E"] = {"pred": old_pred, "metrics": em}
        true_g = group_id(labels)
        for letter, saved in cache.items():
            pred = saved["pred"]
            predicted_g = group_id(pred)
            correct = pred == labels
            for tg in range(3):
                for pg in range(3):
                    take = (true_g == tg) & (predicted_g == pg)
                    confusion_rows.append({"seed": seed, "arm": letter, "true_group": ("Head", "OfficialMedium", "Tail")[tg],
                                           "predicted_group": ("Head", "OfficialMedium", "Tail")[pg],
                                           "count": int(take.sum()), "true_class_correct": int((take & correct).sum()),
                                           "wrong_even_if_same_group": int((take & ~correct).sum())})
        for new, ref in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "A"), ("D", "C")):
            p, rp = cache[new]["pred"], cache[ref]["pred"]
            correct, rc = p == labels, rp == labels
            win, loss = correct & ~rc, rc & ~correct
            tail, head = labels >= 71, labels <= 35
            net = int(win.sum() - loss.sum())
            tn = int(win[tail].sum() - loss[tail].sum())
            hn = int(win[head].sum() - loss[head].sum())
            delta = {"delta_" + k: float(cache[new]["metrics"][k] - cache[ref]["metrics"][k]) for k in METRICS}
            assert abs(delta["delta_Overall"] - net / 1980 * 100) < 1e-10
            assert abs(delta["delta_Tail"] - tn / 580 * 100) < 1e-10
            assert abs(delta["delta_Head"] - hn / 700 * 100) < 1e-10
            paired_rows.append({"seed": seed, "contrast": f"{new}-{ref}", "new_arm": new, "reference_arm": ref,
                                "wins": int(win.sum()), "losses": int(loss.sum()), "net_correct": net,
                                "tail_wins": int(win[tail].sum()), "tail_losses": int(loss[tail].sum()), "tail_net_correct": tn,
                                "head_net_correct": hn, "tail_wrong_to_wrong_changed": int((tail & ~correct & ~rc & (p != rp)).sum()), **delta})
            for c in range(100):
                at = labels == c
                n = int(at.sum())
                class_rows.append({"seed": seed, "contrast": f"{new}-{ref}", "class_id": c, "validation_n": n,
                                   "train_count": int(counts[c]), "reference_correct": int(rc[at].sum()), "new_correct": int(correct[at].sum()),
                                   "wins": int(win[at].sum()), "losses": int(loss[at].sum()),
                                   "net_correct": int(win[at].sum() - loss[at].sum()),
                                   "delta_accuracy_pp": float((correct[at].sum() - rc[at].sum()) / n * 100) if n else None})
            old_g, new_g = group_id(rp), group_id(p)
            for tg in range(3):
                for og in range(3):
                    for ng in range(3):
                        at = (true_g == tg) & (old_g == og) & (new_g == ng)
                        transition_rows.append({"seed": seed, "contrast": f"{new}-{ref}", "true_group": tg,
                                                "reference_predicted_group": og, "new_predicted_group": ng,
                                                "images": int(at.sum()), "wrong_to_right": int((at & win).sum()),
                                                "right_to_wrong": int((at & loss).sum()),
                                                "wrong_to_wrong": int((at & ~rc & ~correct).sum())})
        tail = labels >= 71
        ncorrect = {letter: int(((saved["pred"] == labels) & tail).sum()) for letter, saved in cache.items()}
        inter = ncorrect["D"] - ncorrect["B"] - ncorrect["C"] + ncorrect["A"]
        effects.append({"seed": seed, **{f"{k}_tail_correct": v for k, v in ncorrect.items()},
                        "tail_interaction_net_correct": inter, "tail_interaction_pp": inter / 580 * 100,
                        "B_minus_A_tail_net": ncorrect["B"] - ncorrect["A"], "D_minus_C_tail_net": ncorrect["D"] - ncorrect["C"],
                        "C_minus_A_tail_net": ncorrect["C"] - ncorrect["A"], "D_minus_B_tail_net": ncorrect["D"] - ncorrect["B"],
                        "D_minus_A_tail_net": ncorrect["D"] - ncorrect["A"],
                        "D_minus_A_Overall_pp": cache["D"]["metrics"]["Overall"] - cache["A"]["metrics"]["Overall"],
                        "D_minus_A_Head_pp": cache["D"]["metrics"]["Head"] - cache["A"]["metrics"]["Head"]})
        for letter in "AB":
            target_tail = plan["tail_override"].astype(bool)
            classes = plan[f"classes_{letter}"][target_tail]
            affinity = plan[f"affinity_{letter}"][target_tail]
            donor_summaries.append({"seed": seed, "arm": letter, "tail_draws": len(classes),
                                    "mean_donor_training_count": float(counts[classes].mean()),
                                    **quantiles(affinity, "affinity"),
                                    "class_draw_counts": np.bincount(classes, minlength=100).tolist(),
                                    "stratum_draw_counts": np.bincount(plan[f"stratum_{letter}"][target_tail], minlength=6).tolist()})
    new_csv(RESULTS / "epoch10.csv", metrics_rows)
    new_csv(RESULTS / "paired.csv", paired_rows)
    new_csv(RESULTS / "per_class.csv", class_rows)
    new_csv(RESULTS / "group_confusion.csv", confusion_rows)
    new_csv(RESULTS / "paired_group_transitions.csv", transition_rows)
    new_csv(RESULTS / "factorial_effects.csv", effects)
    new_csv(RESULTS / "geometry_summary.csv", geometry_rows)
    affinity_positive = all(r["B_minus_A_tail_net"] > 0 and r["D_minus_C_tail_net"] > 0 for r in effects)
    protection_positive = all(r["C_minus_A_tail_net"] > 0 and r["D_minus_B_tail_net"] > 0 for r in effects)
    tiny_contrasts = [dict(seed=r["seed"], contrast=r["contrast"], tail_net_correct=r["tail_net_correct"])
                      for r in paired_rows if abs(r["tail_net_correct"]) <= 2]
    sign_changes = []
    for contrast in ("B-A", "C-A", "D-B", "D-A", "D-C"):
        net = [r["tail_net_correct"] for r in paired_rows if r["contrast"] == contrast]
        if net[0] * net[1] < 0:
            sign_changes.append(contrast)
    synergy_direction = all(r["tail_interaction_net_correct"] > 0 and r["D_minus_A_tail_net"] > 0 for r in effects)
    no_regression = all(r["D_minus_A_Overall_pp"] >= -1e-10 and r["D_minus_A_Head_pp"] >= -1e-10 for r in effects)
    inconclusive = bool(tiny_contrasts or sign_changes)
    decision = {
        "affinity_direction_gate_passed": affinity_positive,
        "margin_protection_direction_gate_passed": protection_positive,
        "tail_interaction_and_D_gt_A_both_seeds": synergy_direction,
        "D_vs_A_no_Overall_or_Head_regression": no_regression,
        "material_regression_threshold": "not numerically specified in preregistration; not invented after observing results",
        "tiny_tail_changes_0_to_2_images": tiny_contrasts, "tail_sign_flips": sign_changes,
        "inconclusive_under_stop_rule": inconclusive,
        "affinity_supported_for_followup": affinity_positive and not inconclusive,
        "protection_supported_for_followup": protection_positive and not inconclusive,
        "complementarity_conservative_support": synergy_direction and no_regression and not inconclusive,
        "no_more_training": True,
    }
    guard_ok()
    history = sealed_history()
    assert history == read_json(RESULTS / "historical_inputs.json")
    science = {"status": "completed", "primary_epoch": 10, "seeds": list(SEEDS), "updates_total": 6800,
               "epoch10": metrics_rows, "paired": paired_rows, "factorial_effects": effects, "decision": decision,
               "geometry": geometry_rows, "donor_distributions": donor_summaries, "pairing": checks,
               "protocol": "1980 reused original TRAIN holdout, classes1..99 x20; full100 logits; not independent TEST",
               "missing_class": 0, "historical_E_context_only": True, "TEST_used": False,
               "TEST_events": TEST_EVENTS, "historical_write_events": WRITE_EVENTS, "runtime_vcs_events": VCS_EVENTS,
               "initial_read_only_vcs_query_deviation": INITIAL_READ_ONLY_VCS_QUERIES,
               "history_files_checked": len(history), "source_hashes": sources()}
    new_json(RESULTS / "scientific_summary.json", science)
    write_reports(science, all_summaries, counts)
    new_json(RESULTS / "STOP_COMPLETED.json", {"status": "completed_and_stopped", "updates": 6800,
             "trained_arms": list(ARMS), "seeds": list(SEEDS), "official_TEST_used": False,
             "additional_experiments_started": False, "source_hashes": sources()})
    print("REPORT COMPLETE: epoch10.csv,paired.csv,per_class.csv,group_confusion.csv,factorial_effects.csv and two Chinese reports.", flush=True)
    print("STOP: no more seeds, models, Stage-I, official TEST or version-control operations.", flush=True)


def write_reports(science, summaries, counts):
    rows, paired, effects, decision = [science[k] for k in ("epoch10", "paired", "factorial_effects", "decision")]
    lines = ["# Low-VS 供体亲和度 × 间隔风险保护：2×2 实验结果", "",
             "主终点：固定 epoch10。仅报告 TRAIN 留出集，不是官方 CIFAR-100 TEST 准确率。", "",
             "## 协议与数据", "",
             "冻结 Clean Stage-I seed42 epoch184，SHA256 `7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88`。",
             "CIFAR-100-LT IF100 原 LT TRAIN 10847张；ResNet32；100-way FC；每组10epoch×85更新=850；batch128末批95。",
             "原 SGD lr0.1/momentum0.9/wd0.0005、原 scheduler、原 CE。Backbone/BN affine和全部统计逐位冻结。",
             "验证集是原始 TRAIN 的1980张留出，class1..99每类20，class0缺失，保留100维logits；历史重复使用，只支持探索性比较。",
             "有效组：Head35类/700张，OfficialMedium35类/700张，Tail29类/580张。Many/Medium/Few按TRAIN数量>100、20..100、<20划分；与OfficialMedium定义不同。", "",
             "|组|有效类别数|图像数|", "|---|---:|---:|"]
    for name, take in (("Many", counts > 100), ("Medium", (counts >= 20) & (counts <= 100)), ("Few", counts < 20)):
        active = take.copy(); active[0] = False
        lines.append(f"|{name}|{int(active.sum())}|{int(active.sum())*20}|")
    lines += ["", "## 四臂与配对", "",
              "A=RandomDonor；B=HighAffinityDonor；C=A+MarginProtect；D=B+MarginProtect。仅y>=71接收者改变，其他draw沿用原供体和Low-VS。",
              "A/B共享6个频次层的逐draw抽取；层内A按类频次随机、B从affinity最高2类按频次抽样；C供体逐项=A，D逐项=B。",
              "所有臂共用接收者、增强随机流、Gumbel和非尾类输入；A/C与B/D的供体像素/特征/原VS分数逐batch完全一致。",
              "正常路径保持原Bottom48，C/D保护其中q最小16个，再用同一Gumbel从余下32选16；fallback保持原mask，不执行保护。",
              "每组850batch配对与整图通道复制检查通过；仅6500个FC参数更新，186项Backbone状态逐位不变，31个BN计数保持15640。",
              "A/B同层内供体类别频次仍可不同，因此亲和度处理并未隔离全部类频率因素。", "",
              "## 固定epoch10结果（%）", "",
              "|seed|臂|Overall|Head|OfficialMedium|Tail|Many|Medium|Few|Macro|valCE|best epoch/Overall仅诊断|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        best = f"{r['best_epoch_diagnostic']}/{r['best_Overall_diagnostic']:.4f}" if r["best_Overall_diagnostic"] is not None else "历史未提供"
        lines.append(f"|{r['seed']}|{r['letter']}|" + "|".join(f"{r[k]:.4f}" for k in METRICS) + f"|{best}|")
    lines += ["", "E=历史LowVS-Full，仅作上下文，未重训。A将尾接收者供体限定为head，与E供体协议不同，E−A不作单因素因果解释。", "",
              "## 逐图配对与真实Tail得失", "",
              "|seed|比较|全体win/loss/net|Tail win/loss/net|ΔOverall pp|ΔHead pp|ΔTail pp|Tail wrong→wrong改预测|",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"|{r['seed']}|{r['contrast']}|{r['wins']}/{r['losses']}/{r['net_correct']:+d}|{r['tail_wins']}/{r['tail_losses']}/{r['tail_net_correct']:+d}|{r['delta_Overall']:+.4f}|{r['delta_Head']:+.4f}|{r['delta_Tail']:+.4f}|{r['tail_wrong_to_wrong_changed']}|")
    lines += ["", "Tail净收益只计真实标签正确与否的变化；wrong→wrong转移单独列出，没有计为准确率收益。",
              "一张验证图像=0.050505pp Overall、0.142857pp Head、0.172414pp Tail；两个seed共享同一组图像，不能把3960次评估视为独立样本。", "",
              "## Tail交互项", "", "|seed|A正确|B正确|C正确|D正确|I=D−B−C+A（张）|I pp|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in effects:
        lines.append(f"|{r['seed']}|{r['A_tail_correct']}|{r['B_tail_correct']}|{r['C_tail_correct']}|{r['D_tail_correct']}|{r['tail_interaction_net_correct']:+d}|{r['tail_interaction_pp']:+.4f}|")
    lines += ["", "## 预注册判据", "",
              f"亲和度方向门槛（两seed B−A与D−C Tail均>0）：**{decision['affinity_direction_gate_passed']}**。",
              f"保护方向门槛（两seed C−A与D−B Tail均>0）：**{decision['margin_protection_direction_gate_passed']}**。",
              f"两seed交互为正且D>A：**{decision['tail_interaction_and_D_gt_A_both_seeds']}**；D对A完全无Overall/Head下降：**{decision['D_vs_A_no_Overall_or_Head_regression']}**。",
              f"Tail跨seed变号比较：{decision['tail_sign_flips']}；仅0–2张净变化：{decision['tiny_tail_changes_0_to_2_images']}。",
              f"按停止规则结果不确定：**{decision['inconclusive_under_stop_rule']}**。",
              "协议没有给‘material regression’数值阈值，因此没有看结果后另设阈值。若有Overall/Head损失，列明成本，不擅自认定无实质退化。",
              "n=2不做p值或显著性声明；即使方向门槛通过，也不能声称已建立泛化有效的新方法。", "",
              "## 在线几何（更新之前；仅描述即时行为）", "",
              "|seed|臂|Tail draws|保护draws|fallback|平均固定marginΔ|平均动态marginΔ|平均CEΔ|正margin跨界比例|保护相对同一当前FC原mask的平均q增益|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in science["geometry"]:
        gain = r["protected_gain_vs_same_head_original_mean"]
        gain_text = f"{gain:.6f}" if gain is not None else "NA"
        cross = r["conditional_cross_rate"]
        lines.append(f"|{r['seed']}|{LETTERS[r['arm']]}|{r['tail_draws']}|{r['protected_draws']}|{r['fallback_draws']}|{r['fixed_margin_delta_mean']:.6f}|{r['dynamic_margin_delta_mean']:.6f}|{r['delta_CE_mean']:.6f}|{cross:.6f}|{gain_text}|")
    lines += ["", "q是当前FC固定最强非真实竞争类别下的即时贡献，既不是语义重要性，也不是泛化效用。",
              "保护路径逐draw验证确实剔除Bottom48中最小16个q，且同一当前FC下选中q和不低于原mask；不同训练臂的FC已分化，跨臂几何均值不视作纯掩码因果效应。", "",
              "## 供体频率与亲和度", "",
              "|seed|臂|Tail draws|平均供体训练样本数|affinity mean|affinity q05/q50/q95|6层实际draw数|",
              "|---|---|---:|---:|---:|---|---|"]
    for r in science["donor_distributions"]:
        lines.append(f"|{r['seed']}|{r['arm']}|{r['tail_draws']}|{r['mean_donor_training_count']:.4f}|{r['affinity_mean']:.6f}|{r['affinity_q05']:.6f}/{r['affinity_q50']:.6f}/{r['affinity_q95']:.6f}|{r['stratum_draw_counts']}|")
    lines += ["", "完整类频次见 donor_plan_summary.csv；各层先验/实际次数见 donor_stratum_summary.csv；这些数据来自TRAIN，不依赖验证结果。", "",
              "## 耗时与显存", "", "|seed|臂|10epoch训练秒数（含诊断）|全程秒数|CUDA峰值allocated MiB|reserved MiB|", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["letter"] != "E":
            lines.append(f"|{r['seed']}|{r['letter']}|{r['training_seconds']:.2f}|{r['elapsed_seconds']:.2f}|{r['peak_allocated_mib']:.2f}|{r['peak_reserved_mib']:.2f}|")
    lines += ["", "耗时包含新增在线诊断/CSV写入，不能直接与历史未记录相同诊断的用时比较为纯算法开销。", "",
              "## 防护、异常与协议偏差", "",
              f"重新验证历史封存文件{science['history_files_checked']}项，训练完成后哈希保持不变。官方TEST open/hash/构造guard事件0；无历史文件写入事件。",
              "在首次打开任务说明之前，为定位任务文件执行过只读status/branch/log/ls-files查询；未改变版本状态。看到禁止条款后未再执行版本控制操作。此初始只读查询属于执行偏差，明确保留记录。",
              "新入口静态审查阶段修正了草稿中的helper字段/参数名；正式CPU测试之前完成，没有改动任何历史源文件、数学定义或配对约束。",
              "训练失败或其他异常以 FAILED_STOP.json 和 console_*.txt 为准；正常完成标记为 STOP_COMPLETED.json。没有自动重跑、补seed或超参数搜索。", "",
              "## 可复查文件与运行命令", "",
              f"本地目录：`{RESULTS}`。所有新文件仅位于此目录。源文件SHA见preflight.json/scientific_summary.json，完整环境见environment.json。",
              "统计：epoch10.csv、paired.csv、per_class.csv、group_confusion.csv、paired_group_transitions.csv、factorial_effects.csv、geometry_summary.csv。",
              "来源：historical_inputs.json、train_bank_provenance.json、head_prototypes.npz、train_gap_bank.npz、donor_plan_seed124.npz/125.npz、prepare.json。",
              "每组：config.json、effective.yaml、epochs.csv、batch_signatures.csv、geometry_diagnostics.csv、final_fc.pth、validation_epoch10.npz、summary.json。",
              "命令stdout和错误栈：results/console_unit.txt、console_preflight.txt、console_prepare.txt、console_train_seed{124,125}_{A,B,C,D}.txt、console_report.txt。", "", "```powershell",
              f"Set-Location '{REPO}'", f"$py = '{sys.executable}'",
              "$entry = 'H2T_repro_audit/lowvs_factorial/run_factorial.py'", "& $py $entry unit", "& $py $entry preflight", "& $py $entry prepare"]
    for seed in SEEDS:
        for arm in ARMS:
            lines.append(f"& $py $entry train --seed {seed} --arm '{arm}'")
    lines += ["& $py $entry report", "```", "",
              "以上是首次执行顺序；输出存在时脚本拒绝覆写，不应直接在当前results上重跑。每条命令必须成功后才执行下一条。", "",
              "## 停止与下一步", "",
              "本轮8组、6800次更新完成后停止。不新增训练臂/seed，不重训Stage-I，不接触官方TEST，不自动执行版本控制。",
              "若方向翻转或只涉及1–2张图，停止这一具体pilot，不在重复使用的验证集上扫参数。即使门槛通过，后续也仅值得在事先锁定、未使用的评估协议上独立复核，须另行确认。"]
    new_text(RESULTS / "LOWVS_FACTORIAL_RESULTS.md", "\n".join(lines) + "\n")
    scientific_lines = ["# Low-VS 2×2 科学审计报告", "",
                        "## 准确命题", "",
                        "在固定Clean Backbone与历史开发集上，频次分层内head供体亲和度与当前FC间隔风险通道保护是否分别改善尾类正确分类，是否表现正交互？", "",
                        "本实验只更新分类器，无法证明端到端表示学习提升。数学固定竞争类margin分解成立、采样按设计生效、最终分类改善是三个不同证据层次。", ""] + lines[2:]
    new_text(RESULTS / "LOWVS_FACTORIAL_SCIENTIFIC_REPORT.md", "\n".join(scientific_lines) + "\n")


class Tee:
    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle
    def write(self, value):
        self.stream.write(value); self.handle.write(value); self.handle.flush()
        return len(value)
    def flush(self):
        self.stream.flush(); self.handle.flush()
    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("unit", "preflight", "prepare", "train", "report"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "train" and (args.seed is None or args.arm is None):
        parser.error("train requires seed and arm")
    install_local_guard()
    if (RESULTS / "FAILED_STOP.json").exists():
        raise RuntimeError("Existing FAILED_STOP.json: stop; do not retry/overwrite failed run")
    if (RESULTS / "STOP_COMPLETED.json").exists():
        raise RuntimeError("Experiment already completed: refuse rerun")
    tag = f"train_seed{args.seed}_{LETTERS[args.arm]}" if args.action == "train" else args.action
    output_path(RESULTS / "placeholder")
    log_path = RESULTS / f"console_{tag}.txt"
    status_path = RESULTS / f"status_{tag}.json"
    if log_path.exists() or status_path.exists():
        raise FileExistsError("Refuse command log/status overwrite: " + tag)
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        started = time.perf_counter()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
        try:
            torch.set_num_threads(4)
            print("COMMAND", sys.executable, str(Path(__file__).resolve()), " ".join(sys.argv[1:]), flush=True)
            print("START", stamp, "CWD", os.getcwd(), flush=True)
            if args.action == "unit": unit()
            elif args.action == "preflight": preflight()
            elif args.action == "prepare": prepare()
            elif args.action == "train": train(args.seed, args.arm)
            else: report()
            guard_ok()
            new_json(status_path, {"status": "completed", "started": stamp, "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                     "seconds": time.perf_counter() - started, "TEST_events": TEST_EVENTS,
                     "historical_write_events": WRITE_EVENTS, "runtime_vcs_events": VCS_EVENTS,
                     "source_hashes": sources(), "command": sys.argv})
        except BaseException:
            failure = traceback.format_exc()
            print(failure, file=sys.stderr, flush=True)
            failed = {"status": "FAILED_STOP", "command": sys.argv, "started": stamp,
                      "traceback": failure, "TEST_events": TEST_EVENTS,
                      "historical_write_events": WRITE_EVENTS, "runtime_vcs_events": VCS_EVENTS,
                      "partial_outputs_preserved": True, "no_retry": True}
            if not (RESULTS / "FAILED_STOP.json").exists():
                new_json(RESULTS / "FAILED_STOP.json", failed)
            if not status_path.exists():
                new_json(status_path, failed)
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()