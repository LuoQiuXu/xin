"""CPU recheck of real prediction archives, provenance and historical protection."""
import csv
import json
from pathlib import Path

import numpy as np

from inference import ROOT, TASK, OUT, CACHE, sha, write_json


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    targets = [OUT / "ARTIFACT_VALIDATION.json", OUT / "ARTIFACT_INDEX.csv"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refuse to overwrite validation outputs")
    initial = json.loads((CACHE / "historical_metadata_before.json").read_text(encoding="utf-8"))
    before = {p: (size, mtime) for p, size, mtime in initial}
    after = {}
    for name in ("H2T_official_clean", "H2T", "PI_H2T_review", "H2T_repro_audit"):
        for p in (ROOT / name).rglob("*"):
            if p.is_file() and not p.is_relative_to(TASK):
                stat = p.stat()
                after[p.relative_to(ROOT).as_posix()] = (stat.st_size, str(stat.st_mtime_ns))
    assert before == after, "Historical path/size/mtime metadata changed"
    checkpoints = read("CHECKPOINT_METADATA_NUMPY_COMPAT.json")["checkpoints"]
    for ck in checkpoints:
        assert sha(ROOT / ck["path"]) == ck["sha256"]
    availability = read("CHECKPOINT_AVAILABILITY.json")
    for path, expected in availability["source_sha256"].items():
        assert sha(ROOT / path) == expected
    protocol_sources = {r["path"]: r["sha256"] for r in rows(OUT / "SOURCE_EVIDENCE.csv")}
    for path, expected in protocol_sources.items():
        assert sha(ROOT / path) == expected
    for checks in availability["pairs"].values():
        assert checks["same_stage1"] and checks["same_eval_architecture"] and checks["identical_final_backbone_and_BN"]
    validation = read("INFERENCE_VALIDATION.json")
    records = {r["model"] + "_" + r["condition"]: r for r in validation["predictions"]}
    assert len(records) == 20
    for r in records.values():
        assert r["state_before"] == r["state_after"] and r["all_finite"] and r["BN_unchanged"]
        assert r["dropout_modules"] == 0 and r["all_parameters_require_grad"] is False
    for filename in ["PREPARE_GUARD.json", "INFER_GUARD.json"]:
        assert not any(read(filename).values())
    rules_hash = sha(OUT / "PRE_REGISTERED_SCREEN_RULES.json")
    assert read("VISUAL_AUDIT.json")["screen_rules_sha256"] == rules_hash == validation["runtime"]["screen_rules_sha256"]
    shared_ids = shared_y = None
    verified_metrics, verified_did = 0, 0
    for manifest_name, directory in [("manifest.json", OUT), ("manifest_pi_local.json", OUT / "pi_local_pair")]:
        cfg = json.loads((TASK / manifest_name).read_text(encoding="utf-8"))
        counts = np.asarray(cfg["class_counts"])
        correct = {}
        for method, files in cfg["methods"].items():
            for condition, rel in files.items():
                p = TASK / rel
                r = records[method + "_" + condition]
                assert sha(p) == r["npz_sha256"]
                with np.load(p, allow_pickle=False) as z:
                    assert set(z.files) == {"ids", "labels", "logits"}
                    ids, y, logits = z["ids"], z["labels"], z["logits"]
                assert ids.shape == y.shape == (1980,) and ids.dtype == y.dtype == np.int64
                assert logits.shape == (1980, 100) and logits.dtype == np.float32 and np.isfinite(logits).all()
                assert len(np.unique(ids)) == 1980
                if shared_ids is None:
                    shared_ids, shared_y = ids, y
                assert np.array_equal(ids, shared_ids) and np.array_equal(y, shared_y)
                correct[method, condition] = (logits.argmax(1) == y).astype(np.int16)
        groups = {"all": np.ones(len(y), bool), "many": counts[y] > 100,
                  "medium": (counts[y] >= 20) & (counts[y] <= 100), "few": counts[y] < 20}
        for r in rows(directory / "metrics.csv"):
            c = correct[r["method"], r["condition"]]
            expected = {k: float(c[m].mean() * 100) for k, m in groups.items()}
            class_acc = [c[y == cls].mean() * 100 for cls in np.unique(y)]
            expected.update(macro=float(np.mean(class_acc)), worst_class=float(np.min(class_acc)))
            assert all(abs(float(r[k]) - v) < 1e-10 for k, v in expected.items())
            verified_metrics += 1
        clean = correct[cfg["candidate"], "clean"] - correct[cfg["reference"], "clean"]
        for r in rows(directory / "difference_in_differences.csv"):
            condition = r["condition"]
            d = correct[cfg["candidate"], condition] - correct[cfg["reference"], condition] - clean
            d = d[groups[r["group"]]]
            assert abs(float(r["difference_in_differences_pp"]) - 100 * d.mean()) < 1e-10
            assert int(r["excess_net_correct"]) == int(d.sum())
            assert int(r["negative_DiD_images"]) == int((d < 0).sum())
            verified_did += 1
    write_json(targets[0], dict(
        passed=True, historical_files_metadata_unchanged=len(before),
        metadata_fields=["relative path", "size bytes", "mtime_ns (exact string)"],
        metadata_is_not_full_content_hashing=True, checkpoints_sha256_rechecked=len(checkpoints),
        imported_model_dataset_sources_sha256_rechecked=len(availability["source_sha256"]),
        source_and_protocol_files_sha256_rechecked=len(protocol_sources),
        prediction_archives_verified=len(records), recomputed_metric_rows=verified_metrics,
        recomputed_DiD_rows=verified_did, shared_raw_ids_and_labels=True,
        all_NPZ_no_pickle=True, official_test_open_attempts=0, historical_write_attempts=0,
        guards_scope="Python open events in prepare/infer; metadata inventory stats do not open dataset contents",
        no_training_or_optimizer=True, screen_rules_sha256=rules_hash))
    index = []
    for p in sorted(TASK.rglob("*")):
        if not p.is_file() or p == targets[1] or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(TASK).as_posix()
        local = "/primary_cache/" in rel or p.suffix in (".npz", ".npy", ".log")
        index.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                          sha256=sha(p), disposition="LOCAL_ONLY_IGNORED" if local else "SUBMIT"))
    p = ROOT / ".gitignore"
    index.append(dict(path=".gitignore", bytes=p.stat().st_size, sha256=sha(p), disposition="SUBMIT"))
    with targets[1].open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(index)
    print(f"VERIFY PASS: {len(before)} historical file metadata unchanged; 6 checkpoint SHA256s, 20 predictions, 20 metrics and 40 DiD rows verified.")


if __name__ == "__main__":
    main()
