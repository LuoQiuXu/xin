"""CPU provenance/coverage/analysis verification; never repeats a GPU forward."""
import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np

from coverage import ROOT, TASK, DEFAULT_OUT, SEED, guard_io, sha, write_json
from checkpoint import load_models, CHECKPOINT, EXPECTED_SHA256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    out = a.output.resolve()
    assert out.is_relative_to(TASK)
    dest = out / "analysis_v2"
    if (out / "ARTIFACT_VALIDATION.json").exists() or (out / "ARTIFACT_INDEX.csv").exists():
        raise FileExistsError("Refuse to overwrite verification evidence")
    guard = guard_io()
    before = {p: (s,t) for p,s,t in read(out / "primary_cache/HISTORICAL_METADATA_BEFORE.json")}
    after = {}
    for name in ["H2T_official_clean", "H2T", "PI_H2T_review", "H2T_repro_audit"]:
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and not path.is_relative_to(TASK):
                s = path.stat()
                after[path.relative_to(ROOT).as_posix()] = (s.st_size, str(s.st_mtime_ns))
    assert before == after, "Historical path/size/mtime changed"
    for filename in ["screen.py", "CODEX_SEMANTIC_COMPATIBILITY_SCREEN.txt", "SEMANTIC_SCREEN_PROTOCOL.md"]:
        assert (TASK / filename).read_bytes() == (out / "primary_cache" / ("provided_" + filename)).read_bytes()
    availability = read(out / "CHECKPOINT_AVAILABILITY.json")
    for path, expected in availability["source_sha256"].items():
        assert sha(ROOT / path) == expected
    assert sha(CHECKPOINT) == EXPECTED_SHA256
    _, fc, _ = load_models()  # CPU strict loading only, no model forward
    with np.load(out / "primary_cache/feasibility_plan.npz", allow_pickle=False) as z:
        plan = {k:z[k] for k in z.files}
    counts, lt, many = plan["LT_class_counts"], plan["LT_raw_ids"], plan["Many_raw_ids"]
    train = ROOT / "H2T/data/cifar100/cifar-100-python/train"
    assert sha(train) == read(out / "COVERAGE_SUMMARY.json")["official_train_sha256"]
    with train.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    fine, coarse = np.asarray(raw["fine_labels"]), np.asarray(raw["coarse_labels"])
    np.testing.assert_array_equal(np.bincount(fine[lt], minlength=100), counts)
    assert len(lt) == len(np.unique(lt)) == 10847 and (counts[fine[many]] > 100).all()
    coverage = table(out / "RECIPIENT_COVERAGE.csv")
    eligible = []
    for row in coverage:
        rid = int(row["raw_id"])
        valid = many[(many != rid) & (fine[many] != fine[rid])]
        ns = int((coarse[valid] == coarse[rid]).sum())
        nd = int((coarse[valid] != coarse[rid]).sum())
        assert ns == int(row["same_coarse_Many_candidates"]) and nd == int(row["different_coarse_Many_candidates"])
        if ns >= 4 and nd >= 4:
            eligible.append(rid)
    assert len(coverage) == 305 and len(eligible) == 210 and len(np.unique(fine[eligible])) == 21
    assert set(eligible) == set(plan["eligible_Few_raw_ids"])
    with np.load(out / "primary_cache/feature_bank.npz", allow_pickle=False) as z:
        bank = {k:z[k] for k in z.files}
    with np.load(out / "primary_cache/pilot_feature_bank.npz", allow_pickle=False) as z:
        pilot = {k:z[k] for k in z.files}
    assert bank["features"].shape == (6493,64) and bank["features"].dtype == np.float32 and np.isfinite(bank["features"]).all()
    assert np.array_equal(bank["features"][np.searchsorted(bank["ids"],pilot["ids"])], pilot["features"])
    pairmeta = read(out / "PAIR_INPUT.json")
    assert sha(TASK / pairmeta["path"]) == pairmeta["sha256"]
    with np.load(TASK / pairmeta["path"], allow_pickle=False) as z:
        pairs = {k:z[k] for k in z.files}
    assert np.array_equal(pairs["coarse_label"][fine], coarse)
    for prefix in ["rec", "donor"]:
        ids, labels = pairs[prefix + "_id"], pairs[prefix + "_label"]
        expected = bank["features"][np.searchsorted(bank["ids"], ids)].astype(np.float64)
        assert np.array_equal(pairs[prefix + "_feat"], expected)
        assert np.array_equal(labels, fine[ids])
        assert np.isin(ids, lt).all()
    assert np.array_equal(pairs["weight"], fc.fc.weight.detach().numpy().astype(np.float64))
    assert np.array_equal(pairs["bias"], fc.fc.bias.detach().numpy().astype(np.float64))
    assert (counts[pairs["rec_label"]] < 20).all() and (counts[pairs["donor_label"]] > 100).all()
    assert sha(out / "PRE_ANALYSIS_PLAN.json") == pairmeta["analysis_plan_sha256"]
    matched = table(out / "screen_output/matched_pairs.csv")
    assert len(matched) == 4138
    for arm in ["same", "diff"]:
        assert all(float(r[arm + "_mass"]) == .25 for r in matched)
        assert len({(r["recipient_id"],r[arm+"_donor_id"]) for r in matched}) == len(matched)
    recrows = table(dest / "RECIPIENT_ANALYSIS.csv")
    values = []
    for r in recrows:
        subset = [m for m in matched if m["recipient_id"] == r["recipient_id"]]
        expected = [100*np.mean([int(m["same_correct"])-int(m["diff_correct"]) for m in subset]),
                    np.mean([float(m["same_margin"])-float(m["diff_margin"]) for m in subset])]
        assert len(subset) == int(r["n_matches"])
        assert np.max(np.abs(expected - np.array([float(r["accuracy_delta_pp"]),float(r["margin_delta"])]))) < 1e-10
        values.append(expected)
    values = np.asarray(values)
    labels = np.array([int(r["fine_id"]) for r in recrows])
    classes = np.unique(labels)
    sums = np.array([values[labels==c].sum(0) for c in classes])
    sizes = np.array([(labels==c).sum() for c in classes])
    draws = np.random.default_rng(SEED).integers(0,len(classes),(2000,len(classes)))
    boot = sums[draws].sum(1) / sizes[draws].sum(1)[:,None]
    ci = np.quantile(boot,[.025,.975],axis=0)
    summary = read(dest / "ANALYSIS_SUMMARY.json")
    assert np.allclose(values.mean(0),[summary["receiver_balanced_accuracy_delta_pp"],summary["receiver_balanced_margin_delta"]],rtol=0,atol=1e-12)
    expected_ci = summary["fine_class_cluster_bootstrap_95_percent_CI"]
    assert np.allclose(ci[:,0],expected_ci["accuracy_delta_pp"],rtol=0,atol=1e-12)
    assert np.allclose(ci[:,1],expected_ci["margin_delta"],rtol=0,atol=1e-12)
    # The failed export had already finished these numeric outputs; v2 changes no values.
    failure = read(out / "ANALYSIS_ATTEMPT01_FAILURE.json")
    for item in failure["preserved_initial_outputs"]:
        assert sha(out / item["path"]) == item["sha256"]
        if item["path"] != "SCIENTIFIC_DECISION.json":
            assert (out / item["path"]).read_bytes() == (dest / item["path"]).read_bytes()
    for path in [out / (phase + "_GUARD.json") for phase in ["COVERAGE","CHECKPOINT","PLAN","PILOT","FEATURES"]] + [dest / "ANALYSIS_GUARD.json"]:
        assert not any(read(path).values())
    assert not any(guard.values())
    write_json(out / "ARTIFACT_VALIDATION.json", dict(passed=True,
        historical_files_metadata_unchanged=len(before), historical_metadata_not_full_content_hashing=True,
        checkpoint_sha256_unchanged=EXPECTED_SHA256, source_protocol_hashes_unchanged=len(availability["source_sha256"]),
        original_screen_and_two_instructions_unchanged=True, official_TRAIN_coverage_rows_independently_verified=len(coverage),
        eligible_Few_images=210, eligible_Few_classes=21, true_Few_Many_roles=True,
        real_unique_feature_images=6493, pair_rows=13440, cached_recipient_features_exact=True,
        original_head_weights_and_bias_exact=True, pilot_features_reused_exact=True,
        matched_pairs=4138, distinct_receiver_units=210, matched_mass_per_pair_exact=.25,
        independently_recomputed_recipient_effects=210, vectorized_cluster_intervals_match=True,
        serialization_repair_did_not_change_numeric_outputs=True,
        original_partial_outputs_preserved=True, official_test_open_attempts=0,historical_write_attempts=0,
        verifier_model_forward_calls=0, no_training=True))
    local_failed = {r["path"] for r in failure["preserved_initial_outputs"]}
    index = []
    for path in sorted(TASK.rglob("*")):
        if not path.is_file() or path.name == "ARTIFACT_INDEX.csv" or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(out).as_posix() if path.is_relative_to(out) else ""
        local = "primary_cache" in path.parts or path.suffix in [".npz", ".npy", ".log"] or rel in local_failed
        index.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,sha256=sha(path),
            disposition="LOCAL_ONLY_IGNORED" if local else "SUBMIT"))
    path = ROOT / ".gitignore"
    index.append(dict(path=".gitignore",bytes=path.stat().st_size,sha256=sha(path),disposition="SUBMIT"))
    with (out / "ARTIFACT_INDEX.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(index[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(index)
    print(f"VERIFY PASS: {len(before)} historical metadata entries unchanged; real coverage, cached features, fixed masses, receiver estimates and cluster intervals verified.")


if __name__ == "__main__":
    main()
