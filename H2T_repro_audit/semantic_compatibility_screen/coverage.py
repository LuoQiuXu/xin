"""Official TRAIN-only Few-to-Many superclass eligibility; no model forwards."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
DEFAULT_OUT = TASK / "results/semantic_screen_20261029_01"
CLEAN = ROOT / "H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013"
SEED = 20261029


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def guard_io():
    counts = dict(official_test_open_attempts=0, historical_write_attempts=0)
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.lower() == "test" and "cifar" in path.as_posix().lower():
            counts["official_test_open_attempts"] += 1
            raise PermissionError("Official CIFAR TEST content access forbidden")
        mode, flags = args[1] or "", args[2] or 0
        writing = any(x in str(mode) for x in "wax+") or flags & (os.O_WRONLY | os.O_RDWR)
        if writing and path.is_relative_to(ROOT) and not path.is_relative_to(TASK):
            counts["historical_write_attempts"] += 1
            raise PermissionError("Historical write forbidden: " + str(path))
    sys.addaudithook(audit)
    return counts


def official_mapping(fine, coarse, classes=100):
    fine, coarse = np.asarray(fine), np.asarray(coarse)
    if fine.shape != coarse.shape or fine.ndim != 1 or fine.dtype.kind not in "iu" or coarse.dtype.kind not in "iu":
        raise ValueError("Expected paired integer official fine/coarse labels")
    mapping = []
    for cls in range(classes):
        observed = np.unique(coarse[fine == cls])
        if len(observed) != 1:
            raise ValueError("Fine class does not have exactly one official superclass")
        mapping.append(int(observed[0]))
    return np.asarray(mapping, dtype=np.int64)


def class_availability(counts, mapping):
    counts, mapping = np.asarray(counts), np.asarray(mapping)
    if counts.shape != mapping.shape or counts.ndim != 1 or (counts <= 0).any():
        raise ValueError("Invalid class counts/mapping")
    many = counts > 100
    same = np.array([counts[many & (mapping == mapping[c]) & (np.arange(len(counts)) != c)].sum()
                     for c in range(len(counts))], dtype=np.int64)
    different = np.array([counts[many & (mapping != mapping[c])].sum()
                          for c in range(len(counts))], dtype=np.int64)
    eligible = (counts < 20) & (same >= 4) & (different >= 4)
    return same, different, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(TASK)
    out.mkdir(parents=True, exist_ok=True)
    names = ["COVERAGE_SUMMARY.json", "FINE_CLASS_COVERAGE.csv", "SUPERCLASS_COVERAGE.csv",
             "RECIPIENT_COVERAGE.csv", "PREFERRED_300_COVERAGE.csv", "primary_cache/feasibility_plan.npz", "COVERAGE_GUARD.json"]
    if any((out / n).exists() for n in names):
        raise FileExistsError("Refuse to overwrite feasibility evidence")
    guard = guard_io()
    train = ROOT / "H2T/data/cifar100/cifar-100-python/train"
    meta = train.with_name("meta")
    raw_bytes = train.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"
    assert hashlib.md5(raw_bytes).hexdigest() == "16019d7e3df5f24257cddd939b257f8d"
    meta_bytes = meta.read_bytes()
    assert hashlib.md5(meta_bytes).hexdigest() == "7973b15100ade9c7d40fb424638fde48"
    raw, official_meta = pickle.loads(raw_bytes, encoding="latin1"), pickle.loads(meta_bytes, encoding="latin1")
    fine = np.asarray(raw["fine_labels"], dtype=np.int64)
    coarse = np.asarray(raw["coarse_labels"], dtype=np.int64)
    mapping = official_mapping(fine, coarse)
    assert fine.shape == (50000,) and np.array_equal(np.bincount(fine), np.full(100, 500))
    assert np.array_equal(np.bincount(mapping), np.full(20, 5))
    counts = np.array([int(500 * .01 ** (c / 99)) for c in range(100)], dtype=np.int64)
    rng = np.random.RandomState(0)
    ids = []
    for c, n in enumerate(counts):
        pool = np.flatnonzero(fine == c)
        rng.shuffle(pool)
        ids.extend(pool[:n])
    ids = np.asarray(ids, dtype=np.int64)
    with np.load(CLEAN / "split_indices.npz", allow_pickle=False) as z:
        assert np.array_equal(ids, z["train_raw_indices"])
    images = raw["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    assert np.array_equal(np.load(CLEAN / "train_labels.npy", allow_pickle=False), fine[ids])
    assert np.array_equal(np.load(CLEAN / "train_images.npy", allow_pickle=False, mmap_mode="r"), images[ids])
    same, different, eligible = class_availability(counts, mapping)
    few_ids, many_ids = ids[counts[fine[ids]] < 20], ids[counts[fine[ids]] > 100]
    eligible_ids = few_ids[eligible[fine[few_ids]]]
    preferred = np.random.default_rng(SEED).choice(few_ids, 300, replace=False)
    def csv_save(name, rows):
        with (out / name).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    classes = [dict(fine_id=c, fine_name=official_meta["fine_label_names"][c],
        coarse_id=int(mapping[c]), coarse_name=official_meta["coarse_label_names"][mapping[c]],
        LT_count=int(counts[c]), frequency_group="Many" if counts[c] > 100 else ("Few" if counts[c] < 20 else "Medium"),
        same_coarse_Many_images=int(same[c]), different_coarse_Many_images=int(different[c]),
        eligible_Few_recipient=bool(eligible[c])) for c in range(100)]
    csv_save(names[1], classes)
    csv_save(names[2], [dict(coarse_id=c, coarse_name=official_meta["coarse_label_names"][c],
        Many_fine_classes=int(((mapping == c) & (counts > 100)).sum()),
        Many_images=int(counts[(mapping == c) & (counts > 100)].sum()),
        Few_fine_classes=int(((mapping == c) & (counts < 20)).sum()),
        Few_images=int(counts[(mapping == c) & (counts < 20)].sum())) for c in range(20)])
    def recipient_rows(selected):
        return [dict(raw_id=int(i), fine_id=int(fine[i]), coarse_id=int(mapping[fine[i]]),
            LT_class_count=int(counts[fine[i]]), same_coarse_Many_candidates=int(same[fine[i]]),
            different_coarse_Many_candidates=int(different[fine[i]]), eligible=bool(eligible[fine[i]])) for i in selected]
    csv_save(names[3], recipient_rows(few_ids))
    csv_save(names[4], recipient_rows(preferred))
    (out / "primary_cache").mkdir(exist_ok=True)
    with (out / names[5]).open("xb") as f:
        np.savez_compressed(f, LT_raw_ids=ids, Few_raw_ids=few_ids, Many_raw_ids=many_ids,
                            eligible_Few_raw_ids=eligible_ids, preferred_300_raw_ids=preferred,
                            fine_to_coarse=mapping, LT_class_counts=counts)
    write_json(out / names[0], dict(seed=SEED, official_train_sha256=sha(train), official_meta_sha256=sha(meta),
        LT_images=len(ids), LT_training_cache_exactly_verified=True, class_counts=counts.tolist(),
        Few_classes=int((counts < 20).sum()), Few_images=len(few_ids), Many_classes=int((counts > 100).sum()), Many_images=len(many_ids),
        eligible_Few_classes=int(eligible.sum()), eligible_Few_images=len(eligible_ids),
        full_Few_coverage_fraction=float(len(eligible_ids) / len(few_ids)),
        preferred_300_eligible=int(eligible[fine[preferred]].sum()), preferred_300_total=len(preferred),
        excluded_Few_classes=[r for r in classes if r["frequency_group"] == "Few" and not r["eligible_Few_recipient"]],
        minimum_candidates_per_semantic_arm=4, no_frequency_group_broadening=True,
        no_model_forward=True, no_official_test_read=True,
        coverage_interpretation="Only 210 eligible Few images; 300 fully covered Few recipients impossible. Full target versus permitted 150-300 limited subset is awaiting explicit user clarification."))
    assert not any(guard.values())
    write_json(out / names[6], guard)
    print(f"COVERAGE COMPLETE: {len(eligible_ids)}/{len(few_ids)} eligible Few images, {eligible.sum()}/30 classes; preferred300 has {eligible[fine[preferred]].sum()}/300 covered.")


if __name__ == "__main__":
    main()
