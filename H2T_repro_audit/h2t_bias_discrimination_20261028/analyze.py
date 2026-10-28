#!/usr/bin/env python3
"""Read-only, no-training falsification probe: group bias vs within-group ranking.

Consumes ONLY IF200 fixed-BN StageII epoch10 paired TRAIN-holdout logits.
No author-H2T parity claim; no official CIFAR TEST. NumPy dependency only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IF200 = ROOT / "H2T_repro_audit/if200_stress/results"
OLD = ROOT / "H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014"
SEEDS = (124, 125)
ARMS = ("DR-NoFusion", "Original-BatchShared")
GROUPS = {"Head": (0, 36), "OfficialMedium": (36, 71), "Tail": (71, 100)}
PINNED_S1 = "5abef38180bf85cd5c60c38bae08efe23173bdf4775fe44bc38299458448548f"
PINNED_INIT = "a8e9d399d61537ca6d085a55f131e6924634ff077f12cbb40ad2c9f5f55f44f9"
GRID_UNITS = tuple(float(i) / 4 for i in range(-16, 17))
EPS = 1e-9


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(100.0 * np.mean(pred == y))


def group_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    result = {"Overall": scalar_accuracy(pred, y)}
    for group, (lo, hi) in GROUPS.items():
        mask = (y >= lo) & (y < hi)
        if not np.any(mask):
            raise ValueError("Missing validation group " + group)
        result[group] = scalar_accuracy(pred[mask], y[mask])
    return result


def within_group(logits: np.ndarray, y: np.ndarray) -> dict:
    """Class ranking conditioned on *true group*: immune to group-constant bias."""
    out = {}
    for group, (lo, hi) in GROUPS.items():
        mask = (y >= lo) & (y < hi)
        restricted = np.argmax(logits[mask, lo:hi], axis=1) + lo
        out[group] = {
            "correct": int(np.sum(restricted == y[mask])),
            "count": int(np.sum(mask)),
            "accuracy": scalar_accuracy(restricted, y[mask]),
        }
    return out


def validate_holdout(source_root: Path) -> tuple[np.ndarray, dict]:
    labels_path = source_root / "validation_labels.npy"
    split_path = source_root / "split_indices.npz"
    if not labels_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("Pinned TRAIN-holdout labels/split not present; do not use TEST")
    labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
    if labels.shape != (1980,):
        raise ValueError("Unexpected TRAIN holdout size")
    counts = np.bincount(labels, minlength=100)
    if counts[0] != 0 or not np.all(counts[1:] == 20):
        raise ValueError("Expected exactly 20 per class 1..99 and none of class 0")
    with np.load(split_path, allow_pickle=False) as split:
        if not {"validation_raw_indices", "train_raw_indices"}.issubset(split.files):
            raise ValueError("TRAIN split provenance missing expected raw indices")
        v = split["validation_raw_indices"]
        t = split["train_raw_indices"]
    if len(v) != 1980 or len(np.unique(v)) != 1980 or np.intersect1d(v, t).size:
        raise ValueError("Validation IDs duplicate or overlap TRAIN")
    return labels, {str(labels_path.relative_to(ROOT)): digest(labels_path),
                    str(split_path.relative_to(ROOT)): digest(split_path)}


def load_arm(seed: int, arm: str, labels: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    folder = IF200 / f"seed{seed}" / arm
    path = folder / "validation_epoch10.npz"
    meta = folder / "summary.json"
    if not path.is_file() or not meta.is_file():
        raise FileNotFoundError(f"Missing saved validation logits: {path}")
    summary = read_json(meta)
    if (summary.get("seed") != seed or summary.get("arm") != arm
        or not summary.get("completed")
        or summary.get("stage1_sha") != PINNED_S1
        or summary.get("classifier_initial_hash") != PINNED_INIT
        or summary.get("steps") != 750 or summary.get("epochs") != 10
        or summary.get("backbone_unchanged") is not True
        or summary.get("BN_frozen") is not True
        or summary.get("TEST_used") is not False):
        raise ValueError(f"Unverified protocol for {seed}/{arm}")
    if arm == "Original-BatchShared" and summary.get("paired_all_batches") is not True:
        raise ValueError("H2T paired-batch verification absent")
    with np.load(path, allow_pickle=False) as z:
        if not {"labels", "logits", "predictions"}.issubset(z.files):
            raise ValueError("Saved NPZ missing required arrays")
        zlabels = z["labels"]
        logits = np.asarray(z["logits"], dtype=np.float64)
        saved_pred = z["predictions"]
    if (logits.shape != (1980, 100) or not np.all(np.isfinite(logits))
        or not np.array_equal(labels, zlabels)
        or not np.array_equal(saved_pred, np.argmax(logits, axis=1))):
        raise ValueError(f"Invalid aligned label/logit/prediction arrays for {seed}/{arm}")
    metrics = group_metrics(np.argmax(logits, axis=1), labels)
    for k in ("Overall", "Head", "OfficialMedium", "Tail"):
        if abs(metrics[k] - float(summary["final"][k])) > 1e-7:
            raise ValueError(f"Logit accuracy does not reproduce summary: {seed}/{arm}/{k}")
    return logits, summary, {str(path.relative_to(ROOT)): digest(path),
                             str(meta.relative_to(ROOT)): digest(meta)}


def group_candidate_max(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxes, indices = [], []
    for lo, hi in GROUPS.values():
        local = logits[:, lo:hi]
        idx = np.argmax(local, axis=1)
        indices.append(idx + lo)
        maxes.append(local[np.arange(len(local)), idx])
    return np.stack(maxes, axis=1), np.stack(indices, axis=1)


def oracle_grid(dr: np.ndarray, h2t: np.ndarray, y: np.ndarray) -> tuple[list[dict], dict]:
    """Optimistic *diagnostic upper bound*, not a deployable tuned calibrator."""
    base, arg = group_candidate_max(dr)
    target = group_metrics(h2t.argmax(axis=1), y)
    tops = np.partition(dr, -2, axis=1)[:, -2:]
    scale = float(np.median(tops[:, 1] - tops[:, 0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Non-positive median reference logit gap")
    # Bias applied to entire medium and tail bucket; head fixed at zero.
    # Argmax of bucket maxima: groups occupy contiguous class indices,
    # so np.argmax naturally implements smallest-label tie break.
    rows = []
    dominated = 0
    for m in GRID_UNITS:
        for t in GRID_UNITS:
            candidates = base + scale * np.array([0.0, m, t])[None, :]
            win_group = np.argmax(candidates, axis=1)
            pred = arg[np.arange(len(y)), win_group]
            score = group_metrics(pred, y)
            covers = all(score[k] + EPS >= target[k] for k in target)
            dominated += int(covers)
            rows.append({"medium_shift_in_gap_units": m,
                         "tail_shift_in_gap_units": t, "gap_scale": scale,
                         **score, "covers_h2t_all_groups": int(covers)})
    return rows, {"oracle_covers_h2t": bool(dominated),
                  "oracle_covering_points": dominated,
                  "grid_points": len(rows),
                  "gap_scale": scale,
                  "group_bias_search_is_validation_oracle": True}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty CSV")
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(root: Path, output: Path) -> dict:
    global ROOT, IF200, OLD
    ROOT = root.resolve()
    IF200 = ROOT / "H2T_repro_audit/if200_stress/results"
    OLD = ROOT / "H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014"
    destination = (ROOT / output).resolve()
    allowed = (ROOT / "H2T_repro_audit/h2t_bias_discrimination_20261028/results").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed:
        raise ValueError("Output must be a new child of this study's results directory")
    if destination.exists():
        raise FileExistsError("Refuse to overwrite existing results: " + str(destination))

    y, files = validate_holdout(OLD)
    by_seed = []
    rows = []
    grids = []
    for seed in SEEDS:
        dr, _, dr_files = load_arm(seed, ARMS[0], y)
        h2t, _, h_files = load_arm(seed, ARMS[1], y)
        files.update(dr_files); files.update(h_files)
        drm, h2tm = group_metrics(dr.argmax(axis=1), y), group_metrics(h2t.argmax(axis=1), y)
        dw, hw = within_group(dr, y), within_group(h2t, y)
        grid, oracle = oracle_grid(dr, h2t, y)
        for item in grid:
            grids.append({"seed": seed, **item})
        for group in GROUPS:
            rows.append({"seed": seed, "group": group,
                         "n": dw[group]["count"],
                         "dr_conditional_correct": dw[group]["correct"],
                         "h2t_conditional_correct": hw[group]["correct"],
                         "dr_conditional_pct": dw[group]["accuracy"],
                         "h2t_conditional_pct": hw[group]["accuracy"],
                         "conditional_delta_pp": hw[group]["accuracy"]-dw[group]["accuracy"],
                         "dr_standard_pct": drm[group],
                         "h2t_standard_pct": h2tm[group],
                         "standard_delta_pp": h2tm[group]-drm[group]})
        signal = (hw["Tail"]["accuracy"] - dw["Tail"]["accuracy"] >= 1.0
                  and h2tm["Tail"] > drm["Tail"]
                  and h2tm["Overall"] > drm["Overall"]
                  and not oracle["oracle_covers_h2t"])
        by_seed.append({"seed": seed, "standard_DR": drm, "standard_H2T": h2tm,
                        "conditional_DR": dw, "conditional_H2T": hw,
                        **oracle, "preregistered_seed_signal": bool(signal)})
    decision = ("EXPLORATORY_NON_GROUP_BIAS_SIGNAL"
                if all(s["preregistered_seed_signal"] for s in by_seed)
                else "NO_SUPPORT_STOP_THIS_HYPOTHESIS")
    result = {"status": "COMPLETE", "decision": decision, "per_seed": by_seed,
              "source_hashes": files,
              "interpretation": {
                  "effect": "Frozen-backbone StageII classifier differences; NOT better backbone representation",
                  "primary": "True-group restricted top1 ranking invariant to uniform group-bias shifts",
                  "oracle": "Optimistic 2D development-set scan; NOT deployable or independent evaluation",
                  "boundary": "Only group-constant shifts. Does NOT rule out per-class bias or norm/calibration explanations",
                  "validation": "Reused TRAIN-only 1980-image 99-class holdout; no official TEST or novel generalization claim",
                  "independence": "Two StageII seeds share one StageI and one development holdout; not independent datasets",
                  "checkpoint": "IF200 frozen-BN, original-like batch-shared mask, not shift_bn=True original training",
                  "future": "Do not build a method from this alone; require independent validation and novelty review"
              }}
    lines = [
        "# H2T: group-level bias versus within-group discrimination",
        "",
        "**Decision:** " + decision,
        "",
        "This is a retrospective **TRAIN-holdout only** falsification screen, NOT an",
        "optimization of a model and NOT a new H2T method. All comparisons use",
        "fixed epoch10 and the same IF200 frozen-BN StageI checkpoint.",
        "",
        "|Seed|DR Overall|H2T Overall|DR Tail|H2T Tail|Tail within-group Δpp|Oracle group-bias dominates?|Signal|",
        "|---:|---:|---:|---:|---:|---:|---|---|"
    ]
    for s in by_seed:
        t = s["conditional_H2T"]["Tail"]["accuracy"]-s["conditional_DR"]["Tail"]["accuracy"]
        lines.append(f'|{s["seed"]}|{s["standard_DR"]["Overall"]:.3f}|{s["standard_H2T"]["Overall"]:.3f}|{s["standard_DR"]["Tail"]:.3f}|{s["standard_H2T"]["Tail"]:.3f}|{t:+.3f}|{s["oracle_covers_h2t"]}|{s["preregistered_seed_signal"]}|')
    lines.extend([
        "",
        "## Exact interpretation",
        "",
        "For each validation image, only logits from its **true label group** are compared.",
        "Adding the same scalar to every class inside that group cannot change the",
        "within-group predicted class. A positive difference therefore contradicts",
        "a **group-constant bias only** explanation for this conditional ranking;",
        "it does not establish semantic enhancement, nor rule out class-specific",
        "biases, learned class norms, or training optimization effects.",
        "",
        "The DR oracle tries fixed predeclared medium/tail bucket shifts relative to",
        "head on the **same reused development set**. Its envelope is deliberately",
        "optimistic. Matching H2T here does not prove a practical calibrator generalizes;",
        "failing to match only excludes this finite grid of 2D group shifts.",
        "",
        "## Preregistered screen",
        "",
        "For BOTH seeds require: (1) H2T standard Overall and Tail > DR;",
        "(2) H2T Tail within-group ≥ DR by 1.0 percentage point;",
        "(3) no 2D group shift in fixed grid matches/exceeds H2T Overall and",
        "Head/OfficialMedium/Tail simultaneously. Otherwise STOP this hypothesis.",
        "",
        "Even PASS is only motivation for an independent experiment; never claim",
        "a publishable contribution, 'semantic damage' or superiority from this scan.",
        "",
        "## Source integrity",
        "",
        "Every input file SHA256 is recorded in decision.json. Script checked the",
        "source summary assertions, predictions, 1980-image TRAIN split and 100-class",
        "logit shapes. A manual Codex check MUST corroborate actual checkpoint/feature",
        "origins; source metadata assertions are not proof of provenance.",
        "",
    ])
    # All source reads and validation precede output creation.
    destination.mkdir(parents=True)
    write_csv(destination/"within_group.csv", rows)
    write_csv(destination/"oracle_grid.csv", grids)
    (destination/"decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    (destination/"SCIENTIFIC_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, required=True,
                    help="New relative path under this study's results/")
    a = ap.parse_args()
    try:
        out = run(a.root, a.output)
    except (ValueError, OSError, KeyError, AssertionError, json.JSONDecodeError) as e:
        print("STOP WITHOUT RETRAINING:", type(e).__name__, str(e))
        return 2
    print("COMPLETE:", out["decision"], "seeds:", len(out["per_seed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
