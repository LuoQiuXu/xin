"""Read pinned Git blobs only; never import or execute historical experiment code.

This is the branch-aware source inventory permitted by task step 3, not a
replacement PASS for analyze_ledger.py. Existing outputs are never overwritten.
"""
import ast
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
STUDY = OUT.parents[1]


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, check=True
    ).stdout


SUPPORT = {
    "H2T_BN": [
        "artifact_verification.json", "paired_inputs_verification.json",
        "stage1_reference.json", "dr/epochs.csv", "h2t/epochs.csv",
        "dr/effective.yaml", "h2t/effective.yaml", "bn_audit_runner.py",
        "runtime/methods.py", "runtime/train_stage2.py",
    ],
    "DONOR_K4": [
        "analysis_summary.json", "correctness_verification.json",
        "preflight.json", "sampling_distribution.csv", "audit_donors.py",
    ],
    "MASK": [
        "all_three_epoch10.csv", "paired_differences.csv", "pairing_hash.json",
        "continuation_authorization.json", "parameter_BN_verification.json",
        "../mask_policy.py", "../run_mask_granularity.py",
    ],
    "LOWVS_NORM": [
        "epoch10_verified.csv", "paired_verified.csv", "hybrid_parameter_effects.csv",
        "norm_matching_verification.csv", "dataset_protocol.json",
        "../norm_policy.py", "../run_experiment.py",
    ],
    "GU": [
        "epoch10_verified.csv", "paired_verified.csv", "cross_arm_action_histograms.csv",
        "preregistered_decision_rules.json", "reference_bank_provenance.json",
        "../utility_policy.py", "../run_gu_lowvs.py",
    ],
    "RAVS": [
        "all_arms_epoch10.csv", "paired_epoch10.csv", "preregistered_decision_rules.json",
        "scientific_summary.json", "../ravs_policy.py", "../run_ravs_v1.py",
    ],
    "IF200": [
        "epoch10.csv", "paired.csv", "dataset_attestation.json",
        "scientific_verification.json", "stage1/summary.json",
        "../policy.py", "../run_if200.py",
    ],
    "BOUNDARY": [
        "group_summary.csv", "population_diagnostics.csv", "instantaneous_rates.csv",
        "replay_verification.csv", "../geometry.py", "../run_probe.py",
    ],
    "CLEAN_CE": [
        "epoch10.csv", "paired.csv", "tail_error_transitions.csv",
        "scientific_verification.json", "../losses.py", "../run_pilot.py",
    ],
    "OFFICE_RISK": [
        "config.json", "summary.json", "per_seed_arm_summary.csv",
        "risk_predictor_summary.csv", "paired_cross_minus_matched.csv",
        "../../transfer_risk_probe.py", "../pilot_20261018_01/config.json",
        "../pilot_20261018_01/final_checkpoints.lock.json",
    ],
    "PARITY": [
        "PRIOR_EXPERIMENT_EVIDENCE.md", "HISTORICAL_CONTRASTS.csv",
        "HISTORICAL_RECALC.json", "PI_PAPER_CODE_PARITY.csv",
    ],
    "TARGET": ["policy.py"],
    "RECEIVER": ["policy.py", "run_experiment.py"],
    "SEMANTIC": [
        "../diagnostics/feature_hash_diagnosis_20261025_01/HASH_REPLAY_DIAGNOSIS.md",
        "../diagnostics/feature_hash_diagnosis_20261025_01/STATIC_DIAGNOSIS.json",
        "../audit_fusion_semantics.py",
    ],
}

EXTRA = [
    ("AUTHOR_SOURCE", "audit/pi-h2t-author-source-gate-20261027",
     "H2T_repro_audit/pi_h2t_author_audit_20261027/results/source_gate_01/", [
         "STRICT_REPRODUCTION_DECISION.md", "AUTHOR_REPO_INVENTORY.md",
         "PI_PAPER_PROTOCOL_MATRIX.csv", "PAPER_VERSION_AND_SUPPLEMENT_AUDIT.md",
     ]),
    ("STRICT_REPRO", "repro/pi-h2t-cifar100-paper-parity-20261027",
     "H2T_repro_audit/pi_h2t_strict_20261027/runs/pih2t_if100_seed123_42_v1/", [
         "SCIENTIFIC_RESULTS.md", "PARITY_RUN_REPORT.md", "PARITY_LOCK.md",
         "stage1_epochs.csv", "stage2/h2tf/train_epochs.csv",
         "stage2/no_fusion/train_epochs.csv", "per_class_paired_changes.csv",
     ]),
    ("BIAS_PROPOSAL", "research/h2t-bias-vs-discrimination-20261028",
     "H2T_repro_audit/h2t_bias_discrimination_20261028/", ["README.md"]),
]


def normalize(base, suffix):
    parts = []
    for item in (base + "/" + suffix).split("/"):
        if item == "..":
            parts.pop()
        elif item and item != ".":
            parts.append(item)
    return "/".join(parts)


def inspect_blob(record_id, branch, commit, path, role):
    raw = git("show", f"{commit}:{path}")
    blob = git("rev-parse", f"{commit}:{path}").decode().strip()
    calculated = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != calculated:
        raise RuntimeError(f"Git blob mismatch: {path}")
    text = raw.decode("utf-8-sig")
    item = dict(id=record_id, branch=branch, commit=commit, path=path, role=role,
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                git_blob_sha1=blob, blob_hash_verified=True,
                current_worktree_exists=(REPO / path).is_file(),
                source_url=f"https://github.com/LuoQiuXu/xin/blob/{commit}/{path}")
    suffix = Path(path).suffix
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = list(reader)
        item.update(csv_rows=len(rows), csv_columns=reader.fieldnames)
    elif suffix == ".json":
        json.loads(text)
        item["json_parses"] = True
    elif suffix == ".py":
        ast.parse(text)
        item["ast_parses_without_import"] = True
    return item, text


def main():
    targets = [OUT / "SOURCE_MANIFEST.json", OUT / "BRANCH_SOURCE_INDEX.csv",
               OUT / "primary_cache" / "supporting_sources.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refuse to overwrite evidence index or cache")
    ledger = json.loads((STUDY / "EVIDENCE_LEDGER.json").read_text(encoding="utf-8-sig"))
    heads = {r["branch"]: r["sha"] for r in json.loads(
        (OUT / "BRANCH_HEADS.json").read_text(encoding="utf-8"))}
    manifest, cache, index = [], [], []
    for r in ledger["records"]:
        commit = heads[r["branch"]]
        base = r["path"].rsplit("/", 1)[0]
        paths = [(r["path"], "registered_primary")]
        paths += [(normalize(base, s), "supporting_artifact_or_code") for s in SUPPORT[r["id"]]]
        scope = "/".join(r["path"].split("/")[:2])
        tree = git("ls-tree", "-r", "--name-only", commit, "--", scope).decode().splitlines()
        for path, role in paths:
            item, content = inspect_blob(r["id"], r["branch"], commit, path, role)
            manifest.append(item)
            cache.append(dict(**item, text=content))
        primary = manifest[-len(paths)]
        index.append(dict(id=r["id"], branch=r["branch"], commit=commit,
                          registered_path=r["path"], sha256=primary["sha256"],
                          git_blob_sha1=primary["git_blob_sha1"],
                          current_worktree_exists=primary["current_worktree_exists"],
                          scope_files=len(tree), scope_results_files=sum("/results/" in p for p in tree),
                          registered_status=r["status"], source_url=primary["source_url"]))
    for key, branch, base, names in EXTRA:
        for name in names:
            item, content = inspect_blob(key, branch, heads[branch], normalize(base, name),
                                         "chronology_or_dedup_supplement")
            manifest.append(item)
            cache.append(dict(**item, text=content))
    with targets[0].open("x", encoding="utf-8", newline="\n") as f:
        json.dump(dict(repo="LuoQiuXu/xin", purpose="pinned_branch_read_only_inventory",
                       standard_local_gate_status="FAILED_CLOSED_NOT_OVERRIDDEN",
                       registered_records=len(index), sources=len(manifest), files=manifest),
                  f, ensure_ascii=False, indent=2)
        f.write("\n")
    with targets[1].open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(index)
    with targets[2].open("x", encoding="utf-8", newline="\n") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(dict(records=len(index), sources=len(manifest),
                          missing_local_ids=[r["id"] for r in index if not r["current_worktree_exists"]],
                          all_blob_hashes_verified=True)))


if __name__ == "__main__":
    main()
