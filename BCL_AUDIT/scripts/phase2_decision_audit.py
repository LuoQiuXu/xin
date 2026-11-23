from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261123
EXPECTED_EPOCHS = (40, 100, 160, 180, 200)
PAIR_METRICS = (
    "delta_confusion_BCL_minus_LC",
    "delta_raw_margin_BCL_minus_LC",
    "delta_adjusted_margin_BCL_minus_LC",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty diagnostic table: {path.name}")
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _safe_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(values.size), controls])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def _partial_spearman(exposure, outcome, controls) -> float:
    exposure = np.asarray(exposure, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    valid = np.isfinite(exposure) & np.isfinite(outcome)
    if controls.size:
        valid &= np.isfinite(controls).all(axis=1)
    exposure, outcome, controls = exposure[valid], outcome[valid], controls[valid]
    if exposure.size < controls.shape[1] + 4 or np.unique(exposure).size < 2:
        return float("nan")
    rx, ry = _rankdata_average_ties(exposure), _rankdata_average_ties(outcome)
    rc = (
        np.column_stack([_rankdata_average_ties(controls[:, i]) for i in range(controls.shape[1])])
        if controls.shape[1] else np.empty((exposure.size, 0), dtype=np.float64)
    )
    residual_x, residual_y = _residualize(rx, rc), _residualize(ry, rc)
    denominator = float(np.linalg.norm(residual_x) * np.linalg.norm(residual_y))
    return float(np.dot(residual_x, residual_y) / denominator) if denominator > 1e-12 else float("nan")


def _standardized_beta(exposure, outcome, controls) -> float:
    exposure = np.asarray(exposure, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    valid = np.isfinite(exposure) & np.isfinite(outcome)
    if controls.size:
        valid &= np.isfinite(controls).all(axis=1)
    exposure, outcome, controls = exposure[valid], outcome[valid], controls[valid]
    if exposure.size < controls.shape[1] + 4:
        return float("nan")
    sx, sy = float(exposure.std()), float(outcome.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    x, y = (exposure - exposure.mean()) / sx, (outcome - outcome.mean()) / sy
    if controls.shape[1]:
        control_std = controls.std(axis=0)
        keep = control_std > 1e-12
        controls = controls[:, keep]
        if controls.shape[1]:
            controls = (controls - controls.mean(axis=0)) / controls.std(axis=0)
    design = np.column_stack([np.ones(x.size), x, controls])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def _association_with_bootstrap(exposure, outcome, controls, rng, reps=BOOTSTRAP_REPS):
    beta = _standardized_beta(exposure, outcome, controls)
    rho = _partial_spearman(exposure, outcome, controls)
    beta_draws, rho_draws = [], []
    n = int(len(exposure))
    for _ in range(reps):
        indices = rng.randint(0, n, size=n)
        sampled_controls = controls[indices]
        sampled_beta = _standardized_beta(exposure[indices], outcome[indices], sampled_controls)
        sampled_rho = _partial_spearman(exposure[indices], outcome[indices], sampled_controls)
        if math.isfinite(sampled_beta):
            beta_draws.append(sampled_beta)
        if math.isfinite(sampled_rho):
            rho_draws.append(sampled_rho)

    def interval(values):
        if len(values) < max(100, reps // 4):
            return float("nan"), float("nan")
        return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))

    beta_low, beta_high = interval(beta_draws)
    rho_low, rho_high = interval(rho_draws)
    return {
        "standardized_beta": beta,
        "standardized_beta_bootstrap_ci_low": beta_low,
        "standardized_beta_bootstrap_ci_high": beta_high,
        "partial_spearman": rho,
        "partial_spearman_bootstrap_ci_low": rho_low,
        "partial_spearman_bootstrap_ci_high": rho_high,
        "beta_valid_bootstrap_reps": len(beta_draws),
        "rho_valid_bootstrap_reps": len(rho_draws),
        "bootstrap_reps_requested": reps,
    }


def load_and_validate_dino_topology(data_audit_dir: Path, class_meta: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Validate the saved TRAIN-only DINO topology before any expensive Phase-2 probe."""
    required = (
        "DINO_CLASS_MEAN_FEATURES.npy", "DINO_CLASS_SIMILARITY.npy",
        "dino_metadata.json", "head_tail_pair_topology_dino.csv",
        "DINO_CLASS_TOPOLOGY.csv", "data_audit_dino_summary.json",
    )
    missing = [name for name in required if not (data_audit_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Completed Phase-1 TRAIN-only DINO artifacts are required; no DINO recomputation "
            "is allowed. Missing: " + ", ".join(missing)
        )
    with (data_audit_dir / "dino_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    with (data_audit_dir / "data_audit_dino_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if (
        metadata.get("test_split_loaded") is not False
        or "training split" not in str(metadata.get("source_split", ""))
        or summary.get("dino_enrichment_completed") is not True
        or summary.get("test_split_loaded") is not False
        or "training split" not in str(summary.get("source_split", ""))
    ):
        raise RuntimeError("DINO topology does not certify TRAIN-only provenance")
    topk = int(metadata.get("topk", -1))
    if topk != 5 or "symmetric union" not in str(metadata.get("topk_pair_rule", "")).lower():
        raise RuntimeError("Expected the Phase-1 DINO symmetric-union Top-5 topology")

    similarity = np.load(data_audit_dir / "DINO_CLASS_SIMILARITY.npy", allow_pickle=False)
    class_means = np.load(data_audit_dir / "DINO_CLASS_MEAN_FEATURES.npy", allow_pickle=False)
    if (
        similarity.shape != (100, 100) or class_means.ndim != 2 or class_means.shape[0] != 100
        or not np.isfinite(similarity).all() or not np.isfinite(class_means).all()
        or not np.allclose(similarity, similarity.T, atol=1e-5, rtol=0)
        or not np.allclose(np.diag(similarity), 1.0, atol=1e-5, rtol=0)
    ):
        raise RuntimeError("Saved DINO similarity / class-feature arrays failed integrity checks")
    class_topology_rows = _read_csv(data_audit_dir / "DINO_CLASS_TOPOLOGY.csv")
    if len(class_topology_rows) != 100 or {
        int(row["fine_id"]) for row in class_topology_rows
    } != set(range(100)):
        raise RuntimeError("DINO_CLASS_TOPOLOGY.csv must contain all 100 fine classes")

    many_ids = sorted(i for i, meta in class_meta.items() if str(meta["shot_group"]).lower() == "many")
    few_ids = sorted(i for i, meta in class_meta.items() if str(meta["shot_group"]).lower() == "few")
    if len(many_ids) != 35 or len(few_ids) != 30:
        raise RuntimeError(f"Expected 35 Many and 30 Few classes, got {len(many_ids)} and {len(few_ids)}")

    near_pairs: set[tuple[int, int]] = set()
    for class_id in range(100):
        row = similarity[class_id].astype(np.float64, copy=True)
        row[class_id] = -np.inf
        for neighbor in np.argsort(-row)[:topk].tolist():
            near_pairs.add(tuple(sorted((class_id, int(neighbor)))))

    sidecar = _read_csv(data_audit_dir / "head_tail_pair_topology_dino.csv")
    expected_pairs = {(head, tail) for head in many_ids for tail in few_ids}
    observed_pairs = {}
    for row in sidecar:
        identity = (int(row["head_fine_id"]), int(row["tail_fine_id"]))
        if identity in observed_pairs:
            raise RuntimeError(f"Duplicate Many-Few pair in DINO sidecar: {identity}")
        observed_pairs[identity] = row
    if len(sidecar) != 1050 or set(observed_pairs) != expected_pairs:
        raise RuntimeError("DINO sidecar must contain the complete 35 x 30 Many-Few pair set")

    for (head_id, tail_id), row in observed_pairs.items():
        expected_same = int(class_meta[head_id]["coarse_id"]) == int(class_meta[tail_id]["coarse_id"])
        if _as_bool(row["same_superclass"]) != expected_same:
            raise RuntimeError(f"Superclass assignment mismatch for pair {(head_id, tail_id)}")
        observed_similarity = _finite_float(row.get("dino_cosine_similarity"))
        if not math.isfinite(observed_similarity) or abs(observed_similarity - float(similarity[head_id, tail_id])) > 1e-5:
            raise RuntimeError(f"DINO pair similarity mismatch for {(head_id, tail_id)}")
        expected_near = tuple(sorted((head_id, tail_id))) in near_pairs
        if _as_bool(row.get("dino_topk_union")) != expected_near:
            raise RuntimeError(f"DINO Top-5 relation mismatch for {(head_id, tail_id)}")

    expected_near_count = sum(tuple(sorted((h, t))) in near_pairs for h in many_ids for t in few_ids)
    if int(summary.get("dino_topk_union_many_few_pairs", -1)) != expected_near_count:
        raise RuntimeError("DINO summary Many-Few Top-5 count does not match the similarity matrix")

    exposure_by_class = {}
    for tail_id in few_ids:
        near_many = [h for h in many_ids if tuple(sorted((h, tail_id))) in near_pairs]
        same_many = [h for h in many_ids if int(class_meta[h]["coarse_id"]) == int(class_meta[tail_id]["coarse_id"])]
        exposure_by_class[tail_id] = {
            "dino_many_neighbor_count": len(near_many),
            "has_dino_many_neighbor": bool(near_many),
            "max_dino_similarity_to_many": float(np.max(similarity[tail_id, many_ids])),
            "same_superclass_many_count": len(same_many),
            "has_same_superclass_many": bool(same_many),
        }
    return {
        "similarity": similarity, "near_pairs": near_pairs,
        "many_ids": many_ids, "few_ids": few_ids,
        "exposure_by_class": exposure_by_class,
        "sidecar_rows": sidecar, "metadata": metadata, "summary": summary,
    }


def _load_test_class_accuracy(output_root: Path):
    loaded = {}
    for group in ("LC_only", "BCL"):
        path = output_root / f"{group}_seed42" / "final_test_predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            labels, predictions = archive["labels"].astype(np.int64), archive["predictions"].astype(np.int64)
        if labels.shape != (10_000,) or predictions.shape != labels.shape:
            raise RuntimeError(f"Invalid saved TEST predictions in {path}")
        if not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 100)):
            raise RuntimeError(f"Saved TEST labels are not balanced in {path}")
        loaded[group] = (labels, predictions)
    if not np.array_equal(loaded["LC_only"][0], loaded["BCL"][0]):
        raise RuntimeError("LC-only and BCL saved TEST labels are not paired")
    labels = loaded["LC_only"][0]
    accuracy = {}
    for group, (_, predictions) in loaded.items():
        accuracy[group] = {
            i: float(np.mean(predictions[labels == i] == labels[labels == i]))
            for i in range(100)
        }
    return accuracy["LC_only"], accuracy["BCL"]


def _build_h1_rows(class_rows, accuracy_lc, accuracy_bcl):
    rows = []
    for meta in class_rows:
        class_id = int(meta["fine_id"])
        rows.append({
            "class_id": class_id, "fine_name": meta["fine_name"],
            "coarse_id": int(meta["coarse_id"]), "shot_group": meta["shot_group"],
            "train_count": int(meta["train_count"]),
            "LC_only_test_accuracy_pct": 100.0 * accuracy_lc[class_id],
            "BCL_test_accuracy_pct": 100.0 * accuracy_bcl[class_id],
            "delta_accuracy_pp": 100.0 * (accuracy_bcl[class_id] - accuracy_lc[class_id]),
            "few_gain_rank_descending": "", "few_gain_rank_ascending": "",
        })
    few = [row for row in rows if row["shot_group"] == "few"]
    descending = sorted(few, key=lambda row: (-row["delta_accuracy_pp"], row["class_id"]))
    ascending = sorted(few, key=lambda row: (row["delta_accuracy_pp"], row["class_id"]))
    for rank, row in enumerate(descending, 1):
        row["few_gain_rank_descending"] = rank
    for rank, row in enumerate(ascending, 1):
        row["few_gain_rank_ascending"] = rank
    by_group = {}
    for group in ("many", "medium", "few"):
        values = np.asarray([r["delta_accuracy_pp"] for r in rows if r["shot_group"] == group])
        by_group[group] = {
            "num_classes": int(values.size), "mean_gain_pp": float(values.mean()),
            "median_gain_pp": float(np.median(values)), "std_gain_pp": float(values.std()),
            "q25_gain_pp": float(np.percentile(values, 25)), "q75_gain_pp": float(np.percentile(values, 75)),
            "min_gain_pp": float(values.min()), "max_gain_pp": float(values.max()),
            "positive_classes": int(np.sum(values > 1e-12)),
            "zero_classes": int(np.sum(np.abs(values) <= 1e-12)),
            "negative_classes": int(np.sum(values < -1e-12)),
        }
    few_values = np.asarray([r["delta_accuracy_pp"] for r in few])
    summary = {
        "by_shot_group": by_group,
        "few_gain_mixed_sign": bool(np.any(few_values > 1e-12) and np.any(few_values < -1e-12)),
        "few_top10": [
            {"class_id": r["class_id"], "fine_name": r["fine_name"], "delta_accuracy_pp": r["delta_accuracy_pp"]}
            for r in descending[:10]
        ],
        "few_bottom10": [
            {"class_id": r["class_id"], "fine_name": r["fine_name"], "delta_accuracy_pp": r["delta_accuracy_pp"]}
            for r in ascending[:10]
        ],
    }
    return rows, summary


def _build_h2_exposure_rows(class_rows, h1_rows, pair_effect_rows, topology):
    h1_by_id = {int(row["class_id"]): row for row in h1_rows}
    tail_confusion = {}
    for tail_id in topology["few_ids"]:
        rows = [
            row for row in pair_effect_rows
            if row["direction"] == "Tail_to_Head" and int(row["tail_class"]) == tail_id
        ]
        if len(rows) != len(topology["many_ids"]):
            raise RuntimeError(f"Expected 35 Tail-to-Head pairs for Few class {tail_id}, got {len(rows)}")
        tail_confusion[tail_id] = {
            "lc": 100.0 * _safe_mean(_finite_float(r["LC_confusion_rate"]) for r in rows),
            "bcl": 100.0 * _safe_mean(_finite_float(r["BCL_confusion_rate"]) for r in rows),
            "delta": 100.0 * _safe_mean(_finite_float(r["delta_confusion_BCL_minus_LC"]) for r in rows),
        }
    rows = []
    for meta in class_rows:
        tail_id = int(meta["fine_id"])
        if meta["shot_group"] != "few":
            continue
        exposure = topology["exposure_by_class"][tail_id]
        confusion, gain = tail_confusion[tail_id], h1_by_id[tail_id]
        rows.append({
            "class_id": tail_id, "fine_name": meta["fine_name"],
            "coarse_id": int(meta["coarse_id"]), "train_count": int(meta["train_count"]),
            "log_train_count": float(np.log(int(meta["train_count"]))),
            "LC_only_test_accuracy_pct": gain["LC_only_test_accuracy_pct"],
            "delta_accuracy_pp": gain["delta_accuracy_pp"],
            "has_dino_many_neighbor": exposure["has_dino_many_neighbor"],
            "dino_many_neighbor_count": exposure["dino_many_neighbor_count"],
            "max_dino_similarity_to_many": exposure["max_dino_similarity_to_many"],
            "has_same_superclass_many": exposure["has_same_superclass_many"],
            "same_superclass_many_count": exposure["same_superclass_many_count"],
            "LC_tail_to_head_mean_confusion_pp": confusion["lc"],
            "BCL_tail_to_head_mean_confusion_pp": confusion["bcl"],
            "delta_tail_to_head_mean_confusion_pp": confusion["delta"],
        })
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 Few-class exposure rows, got {len(rows)}")
    return rows


def _build_h2_associations(exposure_rows):
    exposures = (
        ("max_dino_similarity_to_many", "Max DINO similarity to any Many class"),
        ("has_dino_many_neighbor", "Any symmetric-union DINO Top-5 Many neighbor"),
        ("has_same_superclass_many", "Any same-superclass Many class"),
    )
    outcomes = (
        ("delta_accuracy_pp", "Few-class TEST accuracy gain (pp)",
         ("LC_only_test_accuracy_pct", "log_train_count")),
        ("delta_tail_to_head_mean_confusion_pp", "Few-class mean Tail-to-Head confusion change (pp)",
         ("LC_only_test_accuracy_pct", "log_train_count", "LC_tail_to_head_mean_confusion_pp")),
    )
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    rows = []
    for exposure_key, exposure_label in exposures:
        for outcome_key, outcome_label, control_keys in outcomes:
            x = np.asarray([float(row[exposure_key]) for row in exposure_rows])
            y = np.asarray([float(row[outcome_key]) for row in exposure_rows])
            controls = np.column_stack([[float(row[key]) for row in exposure_rows] for key in control_keys])
            result = _association_with_bootstrap(x, y, controls, rng)
            unadjusted = _association_with_bootstrap(
                x, y, np.empty((len(x), 0), dtype=np.float64), rng
            )
            rows.append({
                "exposure": exposure_key, "exposure_label": exposure_label,
                "outcome": outcome_key, "outcome_label": outcome_label,
                "n_few_classes": len(exposure_rows), "control_variables": ";".join(control_keys),
                **result,
                "unadjusted_standardized_beta": unadjusted["standardized_beta"],
                "unadjusted_standardized_beta_ci_low": unadjusted["standardized_beta_bootstrap_ci_low"],
                "unadjusted_standardized_beta_ci_high": unadjusted["standardized_beta_bootstrap_ci_high"],
                "unadjusted_spearman": unadjusted["partial_spearman"],
                "unadjusted_spearman_ci_low": unadjusted["partial_spearman_bootstrap_ci_low"],
                "unadjusted_spearman_ci_high": unadjusted["partial_spearman_bootstrap_ci_high"],
                "interpretation_note": (
                    "Exploratory class-level association only; bootstrap resamples the 30 Few classes, "
                    "not training seeds, and does not establish causality. Delta accuracy contains the "
                    "LC-only baseline with a negative sign, so baseline-adjusted and unadjusted results "
                    "must be read together to assess mathematical-coupling/regression-to-mean sensitivity."
                ),
            })
    return rows


def _weighted_mean(values, weights):
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0 else float("nan")


def _weighted_smd(a, wa, b, wb):
    ma, mb = _weighted_mean(a, wa), _weighted_mean(b, wb)
    va, vb = _weighted_mean((a - ma) ** 2, wa), _weighted_mean((b - mb) ** 2, wb)
    pooled = math.sqrt(max(0.5 * (va + vb), 0.0))
    return (ma - mb) / pooled if pooled > 1e-12 else float("nan")


def _matched_pair_effects(directional_rows, relation_key, pair_weights):
    relation = np.asarray([_as_bool(row[relation_key]) for row in directional_rows], dtype=bool)
    baseline = np.asarray([_finite_float(row["LC_confusion_rate"]) for row in directional_rows])
    weights = np.asarray(pair_weights, dtype=np.float64)
    positive = np.where(relation & (weights > 0) & np.isfinite(baseline))[0]
    negative = np.where((~relation) & (weights > 0) & np.isfinite(baseline))[0]
    if positive.size == 0 or negative.size == 0:
        return None
    matched_negative = []
    for index in positive:
        distances = np.abs(baseline[negative] - baseline[index])
        matched_negative.append(int(negative[np.lexsort((negative, distances))[0]]))
    matched_negative = np.asarray(matched_negative, dtype=np.int64)
    positive_weights, negative_weights = weights[positive], weights[negative]
    result = {
        "n_exposed_pair_rows": int(positive.size), "n_control_pair_rows": int(negative.size),
        "baseline_smd_before": _weighted_smd(baseline[positive], positive_weights, baseline[negative], negative_weights),
        "baseline_smd_after": _weighted_smd(baseline[positive], positive_weights, baseline[matched_negative], positive_weights),
        "mean_abs_baseline_gap_after": _weighted_mean(np.abs(baseline[positive] - baseline[matched_negative]), positive_weights),
    }
    for metric in PAIR_METRICS:
        values = np.asarray([_finite_float(row[metric]) for row in directional_rows])
        result[f"{metric}__unadjusted_did"] = (
            _weighted_mean(values[positive], positive_weights) - _weighted_mean(values[negative], negative_weights)
        )
        result[f"{metric}__matched_did"] = (
            _weighted_mean(values[positive], positive_weights) - _weighted_mean(values[matched_negative], positive_weights)
        )
    return result


def _pair_cluster_bootstrap(directional_rows, relation_key, many_ids, few_ids, seed, reps=BOOTSTRAP_REPS):
    rng = np.random.RandomState(seed)
    draws = {metric: [] for metric in PAIR_METRICS}
    valid = 0
    for _ in range(reps):
        sampled_many = rng.choice(many_ids, size=len(many_ids), replace=True)
        sampled_few = rng.choice(few_ids, size=len(few_ids), replace=True)
        many_mult = np.bincount(sampled_many, minlength=100)
        few_mult = np.bincount(sampled_few, minlength=100)
        weights = np.asarray([
            many_mult[int(row["head_class"])] * few_mult[int(row["tail_class"])]
            for row in directional_rows
        ], dtype=np.float64)
        result = _matched_pair_effects(directional_rows, relation_key, weights)
        if result is None:
            continue
        valid += 1
        for metric in PAIR_METRICS:
            value = result[f"{metric}__matched_did"]
            if math.isfinite(value):
                draws[metric].append(value)
    output = {"valid_bootstrap_reps": valid, "bootstrap_reps_requested": reps}
    for metric in PAIR_METRICS:
        values = draws[metric]
        if len(values) >= max(100, reps // 4):
            low, high = np.percentile(values, [2.5, 97.5])
            output[f"{metric}__ci_low"], output[f"{metric}__ci_high"] = float(low), float(high)
        else:
            output[f"{metric}__ci_low"], output[f"{metric}__ci_high"] = float("nan"), float("nan")
    return output


def _build_h2_pair_matched_did(pair_effect_rows, topology):
    relations = (
        ("same_superclass", "Same-superclass Many-Few pair"),
        ("dino_topk_union", "Symmetric-union DINO Top-5 Many-Few pair"),
    )
    output = []
    for relation_index, (relation_key, relation_label) in enumerate(relations):
        for direction_index, direction in enumerate(("Head_to_Tail", "Tail_to_Head")):
            directional = [row for row in pair_effect_rows if row["direction"] == direction]
            if len(directional) != 1050:
                raise RuntimeError(f"Expected 1050 {direction} pair rows, got {len(directional)}")
            observed = _matched_pair_effects(
                directional, relation_key, np.ones(len(directional), dtype=np.float64)
            )
            if observed is None:
                raise RuntimeError(f"Cannot match exposed/control pairs for {relation_key}, {direction}")
            bootstrap = _pair_cluster_bootstrap(
                directional, relation_key, topology["many_ids"], topology["few_ids"],
                seed=BOOTSTRAP_SEED + 100 * relation_index + direction_index,
            )
            exposed_count = sum(_as_bool(row[relation_key]) for row in directional)
            control_count = len(directional) - exposed_count
            for metric in PAIR_METRICS:
                scale = 100.0 if metric == "delta_confusion_BCL_minus_LC" else 1.0
                output.append({
                    "relation": relation_key, "relation_label": relation_label,
                    "direction": direction, "metric": metric,
                    "exposed_pair_count": exposed_count, "control_pair_count": control_count,
                    "matched_exposed_pair_count": observed["n_exposed_pair_rows"],
                    "unadjusted_did_exposed_minus_control": observed[f"{metric}__unadjusted_did"] * scale,
                    "baseline_confusion_smd_before_matching": observed["baseline_smd_before"],
                    "baseline_confusion_smd_after_matching": observed["baseline_smd_after"],
                    "mean_absolute_baseline_confusion_gap_after_matching_pp": observed["mean_abs_baseline_gap_after"] * 100.0,
                    "matched_did_exposed_minus_control": observed[f"{metric}__matched_did"] * scale,
                    "bootstrap_ci_2_5": bootstrap[f"{metric}__ci_low"] * scale,
                    "bootstrap_ci_97_5": bootstrap[f"{metric}__ci_high"] * scale,
                    "valid_two_way_class_cluster_bootstrap_reps": bootstrap["valid_bootstrap_reps"],
                    "bootstrap_reps_requested": bootstrap["bootstrap_reps_requested"],
                    "effect_unit": "percentage_points" if scale == 100.0 else "logit_margin",
                    "bootstrap_unit": "independent resampling of Many IDs and Few IDs; pair weight is endpoint multiplicity product",
                    "matching_rule": "nearest control pair with replacement on LC-only baseline directional confusion rate",
                })
    return output


def _numeric_or_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return 1.0 if text == "true" else 0.0
    return _finite_float(value)


def _build_h3_gradient_summary(output_root, topology):
    specs = (
        ("common_input_eval", "offline_dual_gradient_metrics.csv", (
            ("W_y_gradient_cosine", "W_y_gradient_cosine"),
            ("W_y_conflict_fraction", "W_y_conflict_negative_cosine"),
            ("W_y_raw_BCL_to_LC_norm_ratio", "W_y_BCL_to_LC_norm_ratio"),
            ("W_y_weighted_BCL_to_LC_norm_ratio", "W_y_objective_weighted_BCL_to_LC_norm_ratio"),
            ("same_feature_gradient_cosine", "same_f_feature_gradient_cosine_mean"),
            ("same_feature_conflict_fraction", "same_f_feature_conflict_fraction"),
            ("same_feature_raw_BCL_to_LC_norm_ratio", "same_f_BCL_to_LC_norm_ratio"),
            ("same_feature_weighted_BCL_to_LC_norm_ratio", "same_f_objective_weighted_BCL_to_LC_norm_ratio"),
        )),
        ("native_three_view_train", "offline_native_three_view_gradient_class_metrics.csv", (
            ("W_y_gradient_cosine", "W_y_gradient_cosine"),
            ("W_y_conflict_fraction", "W_y_conflict_negative_cosine"),
            ("W_y_raw_BCL_to_LC_norm_ratio", "W_y_BCL_to_LC_norm_ratio"),
            ("W_y_weighted_BCL_to_LC_norm_ratio", "W_y_objective_weighted_BCL_to_LC_norm_ratio"),
            ("feature_v1_gradient_cosine", "feature_gradient_cosine_v1_mean"),
            ("feature_v1_conflict_fraction", "feature_gradient_conflict_v1_fraction"),
            ("feature_v1_raw_BCL_to_LC_norm_ratio", "feature_gradient_BCL_to_LC_v1_norm_ratio"),
            ("feature_v1_weighted_BCL_to_LC_norm_ratio", "feature_gradient_objective_weighted_BCL_to_LC_v1_norm_ratio"),
            ("feature_v2_gradient_cosine", "feature_gradient_cosine_v2_mean"),
            ("feature_v2_conflict_fraction", "feature_gradient_conflict_v2_fraction"),
            ("feature_v2_raw_BCL_to_LC_norm_ratio", "feature_gradient_BCL_to_LC_v2_norm_ratio"),
            ("feature_v2_weighted_BCL_to_LC_norm_ratio", "feature_gradient_objective_weighted_BCL_to_LC_v2_norm_ratio"),
            ("feature_v3_gradient_cosine", "feature_gradient_cosine_v3_mean"),
            ("feature_v3_conflict_fraction", "feature_gradient_conflict_v3_fraction"),
            ("feature_v3_raw_BCL_to_LC_norm_ratio", "feature_gradient_BCL_to_LC_v3_norm_ratio"),
            ("feature_v3_weighted_BCL_to_LC_norm_ratio", "feature_gradient_objective_weighted_BCL_to_LC_v3_norm_ratio"),
        )),
    )
    output, few_ids = [], set(topology["few_ids"])
    exposure_specs = (
        ("dino_many_neighbor", "has_dino_many_neighbor"),
        ("same_superclass_many", "has_same_superclass_many"),
    )
    for probe_mode, filename, metrics in specs:
        source_rows = _read_csv(output_root / filename)
        if any(row.get("source_split") != "LT_train_probe" for row in source_rows):
            raise RuntimeError(f"{filename} contains a non-TRAIN gradient source")
        for exposure_label, exposure_key in exposure_specs:
            for epoch in EXPECTED_EPOCHS:
                epoch_rows = [
                    row for row in source_rows
                    if int(row["epoch"]) == epoch and int(row["class_id"]) in few_ids
                ]
                by_class = {int(row["class_id"]): row for row in epoch_rows}
                if set(by_class) != few_ids:
                    raise RuntimeError(f"{filename} epoch {epoch} lacks one or more Few classes")
                exposed_ids = [
                    i for i in sorted(few_ids)
                    if topology["exposure_by_class"][i][exposure_key]
                ]
                control_ids = sorted(few_ids - set(exposed_ids))
                for metric_label, source_key in metrics:
                    exp_values = np.asarray([_numeric_or_bool(by_class[i][source_key]) for i in exposed_ids])
                    ctl_values = np.asarray([_numeric_or_bool(by_class[i][source_key]) for i in control_ids])
                    exp_mean, ctl_mean = _safe_mean(exp_values), _safe_mean(ctl_values)
                    output.append({
                        "probe_mode": probe_mode, "exposure_definition": exposure_label,
                        "epoch": epoch, "metric": metric_label,
                        "n_exposed_few_classes": len(exposed_ids),
                        "n_unexposed_few_classes": len(control_ids),
                        "exposed_mean": exp_mean, "unexposed_mean": ctl_mean,
                        "difference_exposed_minus_unexposed": exp_mean - ctl_mean,
                        "interpretation_note": "Five checkpoints from one seed are trajectory snapshots, not five independent replications.",
                    })
    global_rows = _read_csv(output_root / "offline_native_three_view_gradient_global_metrics.csv")
    if len(global_rows) != len(EXPECTED_EPOCHS) or any(
        not _as_bool(row.get("bn_running_buffers_restored")) for row in global_rows
    ):
        raise RuntimeError("Native train-mode gradient audit does not certify BN buffer restoration")
    if any(row.get("gradient_source_split") != "LT_train_probe_only" for row in global_rows):
        raise RuntimeError("Native gradient audit contains an unexpected source split")
    return output


def _build_h3_margin_trajectory(output_root, topology):
    train_rows = _read_csv(output_root / "offline_probe_margin_metrics.csv")
    test_rows = _read_csv(output_root / "offline_test_margin_metrics.csv")
    few_ids = set(topology["few_ids"])
    train_lookup = {(r["group"], int(r["epoch"]), int(r["class_id"])): r for r in train_rows}
    test_lookup = {(r["group"], int(r["class_id"])): r for r in test_rows}
    exposure_specs = (
        ("dino_many_neighbor", "has_dino_many_neighbor"),
        ("same_superclass_many", "has_same_superclass_many"),
    )
    output = []
    for source_split, epochs in (
        ("LT_train_probe", EXPECTED_EPOCHS),
        ("balanced_TEST_read_only", (200,)),
    ):
        for exposure_label, exposure_key in exposure_specs:
            for epoch in epochs:
                class_deltas = []
                for class_id in sorted(few_ids):
                    if source_split == "LT_train_probe":
                        lc, bcl = train_lookup.get(("LC_only", epoch, class_id)), train_lookup.get(("BCL", epoch, class_id))
                    else:
                        lc, bcl = test_lookup.get(("LC_only", class_id)), test_lookup.get(("BCL", class_id))
                    if lc is None or bcl is None:
                        raise RuntimeError(f"Missing paired {source_split} margin row for class {class_id}, epoch {epoch}")
                    class_deltas.append({
                        "exposed": bool(topology["exposure_by_class"][class_id][exposure_key]),
                        "raw": _finite_float(bcl["raw_margin_mean"]) - _finite_float(lc["raw_margin_mean"]),
                        "adjusted": _finite_float(bcl["adjusted_margin_mean"]) - _finite_float(lc["adjusted_margin_mean"]),
                        "top1": _finite_float(bcl["raw_top1"]) - _finite_float(lc["raw_top1"]),
                    })
                for exposed in (True, False):
                    subset = [r for r in class_deltas if r["exposed"] is exposed]
                    if not subset:
                        continue
                    output.append({
                        "source_split": source_split, "exposure_definition": exposure_label,
                        "epoch": epoch, "exposure_group": "exposed" if exposed else "unexposed",
                        "n_few_classes": len(subset),
                        "mean_delta_raw_margin_BCL_minus_LC": _safe_mean(r["raw"] for r in subset),
                        "median_delta_raw_margin_BCL_minus_LC": float(np.median([r["raw"] for r in subset])),
                        "mean_delta_adjusted_margin_BCL_minus_LC": _safe_mean(r["adjusted"] for r in subset),
                        "mean_delta_top1_pp_BCL_minus_LC": _safe_mean(r["top1"] for r in subset),
                        "interpretation_note": "Train margins use the fixed LT TRAIN probe; TEST margins are read-only from saved final logits.",
                    })
    return output


def _format(value, digits=3):
    number = _finite_float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _build_decision_brief(h1_summary, h2_associations, h2_pair_did, h3_gradients, h3_margins):
    few = h1_summary["by_shot_group"]["few"]
    top10, bottom10 = h1_summary["few_top10"], h1_summary["few_bottom10"]
    group_lines = [
        "| " + group + " | " + str(values["num_classes"]) + " | "
        + _format(values["mean_gain_pp"]) + " | " + _format(values["median_gain_pp"]) + " | "
        + _format(values["std_gain_pp"]) + " | " + _format(values["q25_gain_pp"]) + "–"
        + _format(values["q75_gain_pp"]) + " | " + str(values["positive_classes"]) + "/"
        + str(values["zero_classes"]) + "/" + str(values["negative_classes"]) + " |"
        for group, values in h1_summary["by_shot_group"].items()
    ]
    dino_t2h = next(r for r in h2_pair_did if r["relation"] == "dino_topk_union" and r["direction"] == "Tail_to_Head" and r["metric"] == "delta_confusion_BCL_minus_LC")
    superclass_t2h = next(r for r in h2_pair_did if r["relation"] == "same_superclass" and r["direction"] == "Tail_to_Head" and r["metric"] == "delta_confusion_BCL_minus_LC")
    w_conflicts = [
        r for r in h3_gradients
        if r["exposure_definition"] == "dino_many_neighbor"
        and r["probe_mode"] == "native_three_view_train"
        and r["metric"] == "W_y_conflict_fraction"
    ]
    positive_w_conflict_epochs = sum(_finite_float(r["difference_exposed_minus_unexposed"]) > 0 for r in w_conflicts)
    dino_ci_support = _finite_float(dino_t2h["bootstrap_ci_2_5"]) > 0 and math.isfinite(_finite_float(dino_t2h["bootstrap_ci_97_5"]))
    superclass_ci_support = _finite_float(superclass_t2h["bootstrap_ci_2_5"]) > 0 and math.isfinite(_finite_float(superclass_t2h["bootstrap_ci_97_5"]))

    maxsim_gain = next(
        r for r in h2_associations
        if r["exposure"] == "max_dino_similarity_to_many" and r["outcome"] == "delta_accuracy_pp"
    )
    maxsim_confusion = next(
        r for r in h2_associations
        if r["exposure"] == "max_dino_similarity_to_many"
        and r["outcome"] == "delta_tail_to_head_mean_confusion_pp"
    )
    class_association_support = (
        _finite_float(maxsim_gain["standardized_beta"]) < 0
        and _finite_float(maxsim_gain["standardized_beta_bootstrap_ci_high"]) < 0
        and _finite_float(maxsim_confusion["standardized_beta"]) > 0
        and _finite_float(maxsim_confusion["standardized_beta_bootstrap_ci_low"]) > 0
    )
    dino_test_margin_rows = [
        r for r in h3_margins
        if r["exposure_definition"] == "dino_many_neighbor"
        and r["source_split"] == "balanced_TEST_read_only"
        and int(r["epoch"]) == 200
    ]
    dino_test_exposed = next((r for r in dino_test_margin_rows if r["exposure_group"] == "exposed"), None)
    dino_test_unexposed = next((r for r in dino_test_margin_rows if r["exposure_group"] == "unexposed"), None)
    dino_test_margin_gap = (
        _finite_float(dino_test_exposed["mean_delta_raw_margin_BCL_minus_LC"])
        - _finite_float(dino_test_unexposed["mean_delta_raw_margin_BCL_minus_LC"])
        if dino_test_exposed is not None and dino_test_unexposed is not None else float("nan")
    )

    if dino_ci_support and class_association_support and positive_w_conflict_epochs > 0 and dino_test_margin_gap < 0:
        candidate = (
            "One stop-gradient counterfactual: detach classifier weights only on the "
            "classifier-derived prototype path (prototype_mlp(W.detach())). Keep forward "
            "prototypes unchanged; remove only BCL-to-W gradient flow."
        )
        candidate_reason = (
            "The max-similarity class-level associations point to lower Few gain and higher "
            "Tail-to-Head confusion after baseline controls; DINO-neighbor Tail-to-Head matched "
            "confusion DID has a positive endpoint-cluster lower bound; native W_y conflict is "
            "higher for exposed Few classes at one or more checkpoints; and the exposed group "
            "has a smaller BCL-minus-LC final TEST margin change."
        )
    elif dino_ci_support or superclass_ci_support or class_association_support:
        candidate = (
            "One localized counterfactual: suppress only the high-overlap Many-Few "
            "negative-prototype interactions identified by the supported topology slice. "
            "Do not add a general semantic gate or modify other pairs."
        )
        candidate_reason = "A topology-localized Tail-to-Head confusion increase remains after matching on LC-only baseline confusion, but shared-W conflict is not clearly localized."
    elif few["std_gain_pp"] > 1e-12:
        candidate = (
            "At most one falsification experiment: detach the classifier-derived prototype "
            "gradient path into W while preserving forward prototypes. If it does not change "
            "the low-gain Few classes without unacceptable Many-class loss, stop this BCL-specific route."
        )
        candidate_reason = (
            "Few-class gains have a non-degenerate class-level spread, but one-seed H2/H3 "
            "evidence does not localize a more specific mechanism. This is the single "
            "bounded falsification allowed by the intermediate decision tier."
        )
    else:
        candidate = "No mechanism-aligned intervention is justified; stop the BCL-specific direction."
        candidate_reason = "The one-seed evidence does not show a sufficiently localized pattern to justify another intervention."

    association_lines = [
        "| " + r["exposure"] + " | " + r["outcome"] + " | "
        + _format(r["standardized_beta"]) + " [" + _format(r["standardized_beta_bootstrap_ci_low"])
        + ", " + _format(r["standardized_beta_bootstrap_ci_high"]) + "] | "
        + _format(r["partial_spearman"]) + " [" + _format(r["partial_spearman_bootstrap_ci_low"])
        + ", " + _format(r["partial_spearman_bootstrap_ci_high"]) + "] | "
        + _format(r["unadjusted_spearman"]) + " ["
        + _format(r["unadjusted_spearman_ci_low"]) + ", "
        + _format(r["unadjusted_spearman_ci_high"]) + "] |"
        for r in h2_associations
    ]
    did_lines = [
        "| " + r["relation"] + " | " + r["direction"] + " | "
        + str(r["exposed_pair_count"]) + "/" + str(r["control_pair_count"]) + " | "
        + _format(r["matched_did_exposed_minus_control"]) + " pp ["
        + _format(r["bootstrap_ci_2_5"]) + ", " + _format(r["bootstrap_ci_97_5"]) + "] | "
        + _format(r["baseline_confusion_smd_before_matching"]) + " → "
        + _format(r["baseline_confusion_smd_after_matching"]) + " |"
        for r in h2_pair_did if r["metric"] == "delta_confusion_BCL_minus_LC"
    ]
    trajectory_lines = []
    for relation in ("dino_many_neighbor", "same_superclass_many"):
        rows = [
            r for r in h3_gradients
            if r["exposure_definition"] == relation
            and r["probe_mode"] == "native_three_view_train"
            and r["metric"] in {"W_y_gradient_cosine", "W_y_conflict_fraction", "feature_v2_conflict_fraction"}
        ]
        for metric in ("W_y_gradient_cosine", "W_y_conflict_fraction", "feature_v2_conflict_fraction"):
            values = [r for r in rows if r["metric"] == metric]
            if metric == "W_y_gradient_cosine":
                signs = sum(_finite_float(r["difference_exposed_minus_unexposed"]) < 0 for r in values)
                direction_text = "exposed-minus-unexposed < 0 (lower cosine / more opposed)"
            else:
                signs = sum(_finite_float(r["difference_exposed_minus_unexposed"]) > 0 for r in values)
                direction_text = "exposed-minus-unexposed > 0 (higher conflict fraction)"
            trajectory_lines.append(f"- {relation} / {metric}: {direction_text} at {signs}/{len(values)} checkpoints.")

    margin_lines = []
    for relation in ("dino_many_neighbor", "same_superclass_many"):
        for source_split, epochs in (("LT_train_probe", EXPECTED_EPOCHS), ("balanced_TEST_read_only", (200,))):
            for epoch in epochs:
                rows = [
                    r for r in h3_margins
                    if r["exposure_definition"] == relation
                    and r["source_split"] == source_split
                    and int(r["epoch"]) == epoch
                ]
                exposed = next((r for r in rows if r["exposure_group"] == "exposed"), None)
                unexposed = next((r for r in rows if r["exposure_group"] == "unexposed"), None)
                if exposed is not None and unexposed is not None:
                    raw_gap = (
                        _finite_float(exposed["mean_delta_raw_margin_BCL_minus_LC"])
                        - _finite_float(unexposed["mean_delta_raw_margin_BCL_minus_LC"])
                    )
                    adjusted_gap = (
                        _finite_float(exposed["mean_delta_adjusted_margin_BCL_minus_LC"])
                        - _finite_float(unexposed["mean_delta_adjusted_margin_BCL_minus_LC"])
                    )
                    margin_lines.append(
                        f"- {relation} / {source_split} / epoch {epoch}: "
                        f"exposed-minus-unexposed BCL-LC raw margin={_format(raw_gap)}, "
                        f"adjusted margin={_format(adjusted_gap)}."
                    )

    brief = f"""# Phase 2 H1-H3 Decision Brief

## Scope and evidential boundary

- Dataset/protocol: CIFAR-100-LT IF=100, ResNet-32, paired LC-only vs BCL, seed 42.
- This is one paired seed. Class/pair bootstrap intervals quantify class-endpoint sampling uncertainty conditional on this trained seed; they do **not** quantify training-seed variability and do not establish reproducibility or causality.
- DINOv2 topology is TRAIN-only and fixed before final TEST analysis. TEST is read only from saved epoch-200 prediction/logit archives. No TEST forward or autograd is used.
- No new training, Oracle, extra seed, or method is executed by this analysis.

## H1 — Class-level gain heterogeneity

Few classes: {few['num_classes']}; positive/zero/negative BCL-minus-LC gains: {few['positive_classes']}/{few['zero_classes']}/{few['negative_classes']}.
Mean/median gain: {_format(few['mean_gain_pp'])}/{_format(few['median_gain_pp'])} pp; IQR: {_format(few['q25_gain_pp'])} to {_format(few['q75_gain_pp'])} pp; range: {_format(few['min_gain_pp'])} to {_format(few['max_gain_pp'])} pp.
Mixed positive and negative Few-class gains: **{h1_summary['few_gain_mixed_sign']}**.

| Group | Classes | Mean gain (pp) | Median gain (pp) | SD (pp) | IQR (pp) | Positive/zero/negative |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(group_lines)}

### Few-class top 10 gains

| Class | ID | Gain (pp) |
|---|---:|---:|
{chr(10).join(f"| {r['fine_name']} | {r['class_id']} | {_format(r['delta_accuracy_pp'])} |" for r in top10)}

### Few-class bottom 10 gains

| Class | ID | Gain (pp) |
|---|---:|---:|
{chr(10).join(f"| {r['fine_name']} | {r['class_id']} | {_format(r['delta_accuracy_pp'])} |" for r in bottom10)}

## H2 — Topology exposure and deconfounded associations

Few-class exposure is reported separately as (1) any Many-class neighbor in the symmetric-union DINO Top-5 graph, (2) any same-superclass Many class, and (3) maximum DINO cosine similarity to the 35 Many classes.

Class-level adjusted associations control LC-only TEST accuracy and log training count for gain; Tail-to-Head confusion change additionally controls LC-only baseline Tail-to-Head confusion. The table also reports unadjusted Spearman as a sensitivity check: because delta accuracy is BCL accuracy minus LC-only accuracy, controlling LC-only accuracy can create mathematical coupling/regression-to-mean sensitivity. All intervals are descriptive Few-class bootstrap intervals.

| Exposure | Outcome | Adjusted standardized beta [95% class-bootstrap interval] | Adjusted partial Spearman [95% class-bootstrap interval] | Unadjusted Spearman [95% class-bootstrap interval] |
|---|---|---:|---:|---:|
{chr(10).join(association_lines)}

Pair-level DID uses C[a→b] = P(prediction=b | true=a), delta C = BCL − LC-only. Each exposed pair is matched with replacement to the nearest unexposed pair on LC-only baseline directional confusion. Intervals use two-way endpoint cluster bootstrap: Many IDs and Few IDs are independently resampled; pair weights are the product of endpoint multiplicities.

| Relation | Direction | Exposed/control pairs | Matched DID in delta C (pp) [95% endpoint-cluster interval] | Baseline-confusion SMD before → after |
|---|---|---:|---:|---:|
{chr(10).join(did_lines)}

## H3 — Optimization and margin trajectory

Five-checkpoint diagnostics are stratified by both Few-class exposure definitions. Common-input eval and native three-view train-mode gradients are separate estimands. Positive conflict-fraction differences indicate more negative-cosine observations in the exposed group; lower W_y cosine indicates more opposed branch gradients.

{chr(10).join(trajectory_lines)}

Exposed-minus-unexposed margin trajectory (each margin is already BCL-minus-LC):
{chr(10).join(margin_lines)}

The margin table includes fixed LT TRAIN probe margins at epochs 40/100/160/180/200 and read-only TEST margin differences from saved final logits at epoch 200. Native train-mode BN restoration is checked after every diagnostic batch.

## Required decision questions

1. **Specific candidate weakness:** The only candidate supported by this design is a possible mismatch between global class-level contrastive balancing and local Many–Few semantic-neighbor interactions. This is a hypothesis, not an established BCL defect.
2. **Evidence strength:** Exploratory, single-seed, conditional on seed 42. Even a positive class/pair bootstrap interval is not evidence of cross-seed reproducibility.
3. **At most one next intervention:** **{candidate}**

Rationale: {candidate_reason}

## Stop line

- Do not add passive probes, extra metrics, extra seeds, or additional topology definitions after seeing this report.
- If the single recommended counterfactual is approved and fails to change the identified low-gain/high-overlap Few classes without unacceptable Many-class loss, stop this BCL-specific direction.
- Do not turn these results into a method claim until an independently repeated paired seed supports the same direction.
"""
    decision = {
        "status": "ONE_SEED_EXPLORATORY_HUMAN_REVIEW",
        "seed": 42,
        "h1_few_gain_summary": few,
        "h1_by_shot_group": h1_summary["by_shot_group"],
        "h1_few_gain_mixed_sign": h1_summary["few_gain_mixed_sign"],
        "h2_dino_tail_to_head_matched_delta_confusion_pp": dino_t2h["matched_did_exposed_minus_control"],
        "h2_dino_tail_to_head_cluster_ci": [dino_t2h["bootstrap_ci_2_5"], dino_t2h["bootstrap_ci_97_5"]],
        "h2_same_superclass_tail_to_head_matched_delta_confusion_pp": superclass_t2h["matched_did_exposed_minus_control"],
        "h2_same_superclass_tail_to_head_cluster_ci": [superclass_t2h["bootstrap_ci_2_5"], superclass_t2h["bootstrap_ci_97_5"]],
        "h3_dino_exposed_higher_native_W_conflict_epochs": positive_w_conflict_epochs,
        "h2_max_similarity_adjusted_association_support": class_association_support,
        "h3_dino_exposed_minus_unexposed_test_raw_margin_gap": dino_test_margin_gap,
        "candidate_intervention": candidate, "candidate_intervention_reason": candidate_reason,
        "no_training_executed": True, "no_oracle_executed": True, "no_new_method_executed": True,
        "class_and_pair_bootstrap_is_not_seed_uncertainty": True, "dino_test_split_loaded": False,
    }
    return brief, decision


def run_phase2_decision_audit(output_root: Path, data_audit_dir: Path, class_meta, topology):
    """Create the single H1-H3 synthesis from outputs of the merged Phase-2 analyzer."""
    class_rows = _read_csv(data_audit_dir / "class_counts.csv")
    accuracy_lc, accuracy_bcl = _load_test_class_accuracy(output_root)
    h1_rows, h1_summary = _build_h1_rows(class_rows, accuracy_lc, accuracy_bcl)

    pair_effect_rows = _read_csv(output_root / "head_tail_pair_effects.csv")
    h2_exposure_rows = _build_h2_exposure_rows(class_rows, h1_rows, pair_effect_rows, topology)
    h2_associations = _build_h2_associations(h2_exposure_rows)
    h2_pair_did = _build_h2_pair_matched_did(pair_effect_rows, topology)

    h3_gradients = _build_h3_gradient_summary(output_root, topology)
    h3_margins = _build_h3_margin_trajectory(output_root, topology)
    brief, decision = _build_decision_brief(
        h1_summary, h2_associations, h2_pair_did, h3_gradients, h3_margins
    )

    # Compute and validate every table before creating any Phase-2 synthesis file.
    _write_csv(output_root / "phase2_h1_class_gain.csv", h1_rows)
    _write_csv(output_root / "phase2_h2_few_exposure.csv", h2_exposure_rows)
    _write_csv(output_root / "phase2_h2_adjusted_associations.csv", h2_associations)
    _write_csv(output_root / "phase2_h2_pair_matched_did.csv", h2_pair_did)
    _write_csv(output_root / "phase2_h3_overlap_gradient_summary.csv", h3_gradients)
    _write_csv(output_root / "phase2_h3_overlap_margin_trajectory.csv", h3_margins)
    with (output_root / "PHASE2_DECISION_BRIEF.md").open("x", encoding="utf-8") as stream:
        stream.write(brief)
    _write_json(output_root / "phase2_decision_summary.json", decision)
    return decision
