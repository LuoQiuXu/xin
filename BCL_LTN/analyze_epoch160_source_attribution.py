from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = (
    "raw_margin",
    "lc_plus_log_prior_margin",
    "inverse_prior_margin",
    "class_normalized_raw_margin",
)
EXPECTED_GROUP_COUNTS = {
    "Many": {"total": 6752, "clean": 5237, "noisy": 1515},
    "Medium": {"total": 2559, "clean": 1037, "noisy": 1522},
    "Few": {"total": 1536, "clean": 194, "noisy": 1342},
}
GROUP_ORDER = ("Many", "Medium", "Few")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_ids(clean_class_counts: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(clean_class_counts, dtype=np.int64)
    if counts.shape != (100,) or np.any(counts <= 0):
        raise ValueError("noise_manifest.clean_class_counts must be a positive 100-vector")
    return {
        "Many": np.flatnonzero(counts > 100),
        "Medium": np.flatnonzero((counts >= 20) & (counts <= 100)),
        "Few": np.flatnonzero(counts < 20),
    }


def class_group_lookup(clean_class_counts: np.ndarray) -> np.ndarray:
    groups = group_ids(clean_class_counts)
    lookup = np.empty(100, dtype=object)
    for name, ids in groups.items():
        lookup[ids] = name
    if any(value is None for value in lookup):
        raise RuntimeError("Some class IDs were not assigned to Many/Medium/Few")
    return lookup


def binary_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Tie-aware ROC-AUC and step-wise average precision; positive means clean."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError("labels and scores must be matching 1-D arrays")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}

    order_asc = np.argsort(s, kind="mergesort")
    sorted_scores = s[order_asc]
    ranks = np.empty(s.size, dtype=np.float64)
    starts = np.concatenate(([0], np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1))
    ends = np.concatenate((starts[1:], [s.size]))
    for start, end in zip(starts, ends):
        ranks[order_asc[start:end]] = ((start + 1) + end) / 2.0
    auc = (
        ranks[y == 1].sum() - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    order_desc = np.argsort(-s, kind="mergesort")
    desc_scores = s[order_desc]
    desc_labels = y[order_desc]
    tie_ends = np.concatenate((
        np.flatnonzero(desc_scores[1:] != desc_scores[:-1]),
        np.asarray([s.size - 1], dtype=np.int64),
    ))
    cumulative_tp = np.cumsum(desc_labels)[tie_ends]
    cumulative_fp = (tie_ends + 1) - cumulative_tp
    recall = np.concatenate(([0.0], cumulative_tp / positives))
    precision = np.concatenate((
        [1.0],
        cumulative_tp / (cumulative_tp + cumulative_fp),
    ))
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return {"roc_auc": float(auc), "average_precision": ap}


def distribution_row(frame: pd.DataFrame, group_name: str, score_name: str) -> dict:
    values = frame[score_name].to_numpy(dtype=np.float64)
    if values.size == 0:
        return {
            "subset": group_name, "score": score_name, "n": 0,
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "q10": np.nan, "q25": np.nan, "q75": np.nan, "q90": np.nan,
            "fraction_gt_zero": np.nan,
        }
    return {
        "subset": group_name,
        "score": score_name,
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "fraction_gt_zero": float(np.mean(values > 0)),
    }


def compute_gate_reference(frame: pd.DataFrame, class_lookup: np.ndarray) -> dict:
    """
    Build a label-free training reference.

    Only observed_label and margin scores are used here. clean_label/is_clean are
    intentionally not passed into this function.
    """
    required = {"observed_label", "raw_margin", "class_normalized_raw_margin"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing gate-reference columns: {sorted(missing)}")

    observed = frame["observed_label"].to_numpy(dtype=np.int64)
    raw = frame["raw_margin"].to_numpy(dtype=np.float64)
    zscore = frame["class_normalized_raw_margin"].to_numpy(dtype=np.float64)
    if not (np.isfinite(raw).all() and np.isfinite(zscore).all()):
        raise ValueError("Nonfinite values in saved margin scores")

    class_stats = []
    reconstructed_zscore = np.empty_like(raw)
    for class_id in range(100):
        selected = observed == class_id
        values = raw[selected]
        if values.size == 0:
            raise RuntimeError(f"Observed class {class_id} has no audit samples")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        reconstructed_zscore[selected] = (values - mean) / (std + 1e-8)
        class_stats.append({
            "class_id": class_id,
            "observed_count": int(values.size),
            "raw_margin_mean": mean,
            "raw_margin_std": std,
        })
    if not np.allclose(reconstructed_zscore, zscore, rtol=0.0, atol=1e-10):
        raise RuntimeError("Saved class_normalized_raw_margin does not match reconstructed per-class z-scores")

    few_mask = np.asarray([class_lookup[c] == "Few" for c in observed], dtype=bool)
    few_z = zscore[few_mask]
    if few_z.size != EXPECTED_GROUP_COUNTS["Few"]["total"]:
        raise RuntimeError(f"Expected 1536 observed-Few scores, found {few_z.size}")
    threshold = float(np.quantile(few_z, 0.90, method="linear"))
    return {
        "reference": "Epoch-160 TRAIN-only audit sample_scores.csv",
        "score": "class_normalized_raw_margin",
        "formula": "s_i=(m_i-mu_observed_label)/(sigma_observed_label+1e-8)",
        "scope": "observed Few labels only; Many and Medium are not gated",
        "reliable_rule": "Many/Medium: always reliable; Few: s_i > few_q90_threshold",
        "unreliable_rule": "Few and s_i <= few_q90_threshold",
        "threshold_rule": "q90 over all 1536 observed-Few normalized margins; score-only, no clean truth",
        "few_q90_threshold": threshold,
        "normalization_epsilon": 1e-8,
        "class_stats": class_stats,
        "n_observed_few": int(few_z.size),
        "expected_few_acceptance_at_reference": int(np.sum(few_z > threshold)),
    }


def run(args) -> dict:
    sample_path = args.sample_scores.expanduser().resolve()
    manifest_path = args.noise_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Local offline-only sample_scores.csv not found: {sample_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Noise manifest not found: {manifest_path}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Attribution output directory must be created fresh by main(): {output_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_counts = np.asarray(manifest["clean_class_counts"], dtype=np.int64)
    lookup = class_group_lookup(clean_counts)

    frame = pd.read_csv(sample_path)
    required = {
        "position", "original_train_index", "observed_label", "clean_label",
        "is_clean", "observed_label_group", *SCORE_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"sample_scores.csv is missing required columns: {sorted(missing)}")
    if len(frame) != int(manifest["num_train_samples"]) or len(frame) != 10_847:
        raise RuntimeError(f"Expected 10847 rows, found {len(frame)}")
    if frame["position"].duplicated().any() or frame["original_train_index"].duplicated().any():
        raise RuntimeError("Duplicate position or original_train_index in sample_scores.csv")

    for column in ("observed_label", "clean_label", "is_clean"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    for column in SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.float64)
        if not np.isfinite(frame[column].to_numpy()).all():
            raise RuntimeError(f"Nonfinite values in {column}")

    observed = frame["observed_label"].to_numpy(dtype=np.int64)
    clean = frame["clean_label"].to_numpy(dtype=np.int64)
    is_clean = frame["is_clean"].to_numpy(dtype=np.int64)
    if np.any((observed < 0) | (observed >= 100) | (clean < 0) | (clean >= 100)):
        raise RuntimeError("Class IDs must be in [0,99]")
    if not np.isin(is_clean, [0, 1]).all():
        raise RuntimeError("is_clean must contain only 0/1")
    if not np.array_equal(is_clean.astype(bool), observed == clean):
        raise RuntimeError("is_clean does not agree with observed_label == clean_label")

    frame["source_group"] = lookup[clean]
    frame["observed_group"] = lookup[observed]
    if not np.array_equal(
        frame["observed_label_group"].astype(str).to_numpy(),
        frame["observed_group"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Saved observed_label_group disagrees with manifest-derived class groups")

    # Verify the exact audited bucket composition before interpreting source flows.
    group_rows = []
    for name in GROUP_ORDER:
        subset = frame[frame["observed_group"] == name]
        clean_n = int(subset["is_clean"].sum())
        noisy_n = int(len(subset) - clean_n)
        expected = EXPECTED_GROUP_COUNTS[name]
        actual = {"total": int(len(subset)), "clean": clean_n, "noisy": noisy_n}
        if actual != expected:
            raise RuntimeError(f"{name} bucket mismatch: expected {expected}, got {actual}")
        group_rows.append({"observed_group": name, **actual})
    pd.DataFrame(group_rows).to_csv(
        output_dir / "observed_group_composition.csv", index=False, encoding="utf-8-sig"
    )

    if int(frame["is_clean"].sum()) != int(manifest["num_unflipped"]):
        raise RuntimeError("Global clean count disagrees with the noise manifest")
    if int((1 - frame["is_clean"]).sum()) != int(manifest["num_flipped"]):
        raise RuntimeError("Global noisy count disagrees with the noise manifest")

    noisy = frame[frame["is_clean"] == 0].copy()
    transition = pd.crosstab(noisy["source_group"], noisy["observed_group"])
    transition = transition.reindex(index=GROUP_ORDER, columns=GROUP_ORDER, fill_value=0)
    transition_counts = transition.copy()
    transition_counts["row_total"] = transition_counts.sum(axis=1)
    transition_pct = transition.div(transition.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    transition_pct["row_total_pct"] = transition_pct.sum(axis=1)
    transition_counts.to_csv(output_dir / "noisy_source_to_observed_counts.csv", encoding="utf-8-sig")
    transition_pct.to_csv(output_dir / "noisy_source_to_observed_row_percent.csv", encoding="utf-8-sig")

    few = frame[frame["observed_group"] == "Few"].copy()
    few_breakdown = []
    for source in GROUP_ORDER:
        subset = few[few["source_group"] == source]
        clean_n = int(subset["is_clean"].sum())
        noisy_n = int(len(subset) - clean_n)
        few_breakdown.append({
            "source_group": source,
            "observed_destination_group": "Few",
            "total": int(len(subset)),
            "clean": clean_n,
            "noisy": noisy_n,
            "fraction_of_few_bucket": float(len(subset) / len(few)),
            "fraction_of_few_noisy": float(noisy_n / int((few["is_clean"] == 0).sum()))
            if int((few["is_clean"] == 0).sum()) else np.nan,
        })
    few_breakdown_frame = pd.DataFrame(few_breakdown)
    few_breakdown_frame.to_csv(output_dir / "few_source_breakdown.csv", index=False, encoding="utf-8-sig")

    clean_few = few[few["is_clean"] == 1]
    if len(clean_few) != EXPECTED_GROUP_COUNTS["Few"]["clean"]:
        raise RuntimeError("Clean Few count mismatch")

    pairwise_rows = []
    distribution_rows = []
    source_subsets = {
        "Clean_Few": clean_few,
        "Many_to_Few": few[(few["is_clean"] == 0) & (few["source_group"] == "Many")],
        "Medium_to_Few": few[(few["is_clean"] == 0) & (few["source_group"] == "Medium")],
        "Few_to_Few": few[(few["is_clean"] == 0) & (few["source_group"] == "Few")],
    }
    for subset_name, subset in source_subsets.items():
        for score in SCORE_COLUMNS:
            distribution_rows.append(distribution_row(subset, subset_name, score))

    for source in GROUP_ORDER:
        negatives = few[(few["is_clean"] == 0) & (few["source_group"] == source)]
        if len(negatives) == 0:
            continue
        for score in SCORE_COLUMNS:
            combined = pd.concat([clean_few[[score]], negatives[[score]]], ignore_index=True)
            labels = np.concatenate((
                np.ones(len(clean_few), dtype=np.int64),
                np.zeros(len(negatives), dtype=np.int64),
            ))
            metrics = binary_ranking_metrics(labels, combined[score].to_numpy(dtype=np.float64))
            pairwise_rows.append({
                "positive_subset": "Clean_Few",
                "negative_subset": f"{source}_to_Few",
                "score": score,
                "n_clean_few": int(len(clean_few)),
                "n_negative_source": int(len(negatives)),
                **metrics,
            })

    # Decompose each pooled Few AUC into source-group pairwise AUCs.
    total_few_noisy = int((few["is_clean"] == 0).sum())
    reconstruction_rows = []
    pooled_metrics = {}
    for score in SCORE_COLUMNS:
        labels = np.concatenate((
            np.ones(len(clean_few), dtype=np.int64),
            np.zeros(total_few_noisy, dtype=np.int64),
        ))
        scores = np.concatenate((
            clean_few[score].to_numpy(dtype=np.float64),
            few.loc[few["is_clean"] == 0, score].to_numpy(dtype=np.float64),
        ))
        pooled = binary_ranking_metrics(labels, scores)
        source_rows = [row for row in pairwise_rows if row["score"] == score]
        weighted = float(sum(
            row["roc_auc"] * row["n_negative_source"] / total_few_noisy
            for row in source_rows
        ))
        if not np.isclose(weighted, pooled["roc_auc"], atol=1e-12):
            raise RuntimeError(
                f"{score} source-wise AUC decomposition failed: weighted={weighted}, pooled={pooled['roc_auc']}"
            )
        pooled_metrics[score] = pooled
        reconstruction_rows.append({
            "score": score,
            "pooled_few_auc": pooled["roc_auc"],
            "weighted_source_pairwise_auc": weighted,
            "absolute_difference": abs(weighted - pooled["roc_auc"]),
        })
    pd.DataFrame(reconstruction_rows).to_csv(
        output_dir / "few_auc_source_decomposition.csv", index=False, encoding="utf-8-sig"
    )
    pooled_raw = pooled_metrics["raw_margin"]["roc_auc"]
    weighted_auc = next(
        row["weighted_source_pairwise_auc"] for row in reconstruction_rows
        if row["score"] == "raw_margin"
    )

    for score in SCORE_COLUMNS:
        distribution_rows.append(distribution_row(few, "Observed_Few_All", score))
    pd.DataFrame(distribution_rows).to_csv(
        output_dir / "few_margin_distribution_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "few_pairwise_source_metrics.csv", index=False, encoding="utf-8-sig"
    )

    # The gate reference is deliberately constructed from score-only columns.
    score_only = frame[[
        "observed_label", "raw_margin", "class_normalized_raw_margin",
    ]].copy()
    gate_reference = compute_gate_reference(score_only, lookup)
    gate_reference["sample_scores_sha256"] = sha256_file(sample_path)
    gate_reference["noise_manifest_sha256"] = sha256_file(manifest_path)
    (output_dir / "gate_reference.json").write_text(
        json.dumps(gate_reference, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # This CSV was generated by the audited script with margins/labels only.
    # Do not infer logits, predictions or entropy from a scalar target margin.
    logit_columns = [
        column for column in frame.columns
        if column.lower().startswith("logit_") or column.lower() in {
            "max_logit", "max_head_logit", "predicted_class", "prediction_entropy",
            "entropy", "max_softmax_probability",
        }
    ]
    logit_status = (
        "Unexpected logit-like columns were present but are not interpreted by this audit: "
        f"{logit_columns}. No logits/entropy metrics are computed."
        if logit_columns
        else "NOT AVAILABLE: the audited sample_scores.csv writer stores position/index, "
             "observed/clean labels, clean/noisy status, observed-label group and four margin "
             "scores only. It contains no logits, predicted class, softmax probabilities or "
             "entropy. No model loading or re-inference was performed."
    )

    return {
        "rows": int(len(frame)),
        "few_raw_margin_pooled_auc": float(pooled_raw),
        "few_raw_margin_weighted_source_auc": float(weighted_auc),
        "few_pooled_metrics": pooled_metrics,
        "few_noisy_count": total_few_noisy,
        "many_to_few_count": int(len(source_subsets["Many_to_Few"])),
        "many_to_few_share_of_few_noisy": float(
            len(source_subsets["Many_to_Few"]) / total_few_noisy
        ),
        "logit_entropy_status": logit_status,
        "output_dir": str(output_dir),
        "gate_reference": gate_reference,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only source-to-observed-group attribution from saved Epoch-160 sample scores"
    )
    parser.add_argument(
        "--sample-scores",
        type=Path,
        default=Path(
            "BCL_LTN/outputs/epoch160_margin_reliability_audit_20261124/sample_scores.csv"
        ),
    )
    parser.add_argument(
        "--noise-manifest",
        type=Path,
        default=Path("BCL_LTN/data/noise_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("BCL_LTN/outputs/epoch160_source_attribution_20261124"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Create a fresh output directory before writing any files.
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite attribution output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args)
    except Exception:
        # Preserve the partial local diagnostic directory for transparent debugging.
        raise
    few_sources = pd.read_csv(args.output_dir / "few_source_breakdown.csv")
    pairwise = pd.read_csv(args.output_dir / "few_pairwise_source_metrics.csv")
    decomposition = pd.read_csv(args.output_dir / "few_auc_source_decomposition.csv")
    raw_pairwise = pairwise[pairwise["score"] == "raw_margin"]
    lc_pairwise = pairwise[pairwise["score"] == "lc_plus_log_prior_margin"]
    pairwise_rows = []
    for source in GROUP_ORDER:
        raw_row = raw_pairwise[raw_pairwise["negative_subset"] == f"{source}_to_Few"]
        lc_row = lc_pairwise[lc_pairwise["negative_subset"] == f"{source}_to_Few"]
        if raw_row.empty:
            continue
        rr = raw_row.iloc[0]
        lr = lc_row.iloc[0]
        pairwise_rows.append(
            f"| {source}->Few | {int(rr['n_negative_source'])} | "
            f"{rr['roc_auc']:.4f} | {rr['average_precision']:.4f} | "
            f"{lr['roc_auc']:.4f} | {lr['average_precision']:.4f} |"
        )
    source_rows = [
        f"| {row.source_group} | {int(row.total)} | {int(row.clean)} | {int(row.noisy)} | "
        f"{100.0 * row.fraction_of_few_noisy:.2f}% |"
        for row in few_sources.itertuples(index=False)
    ]
    decomp_rows = [
        f"| {row.score} | {row.pooled_few_auc:.6f} | "
        f"{row.weighted_source_pairwise_auc:.6f} | {row.absolute_difference:.2e} |"
        for row in decomposition.itertuples(index=False)
    ]
    report_lines = [
        "# Epoch-160 source-to-observed-group attribution",
        "",
        f"- Samples: {result['rows']}",
        f"- Observed-Few raw-margin pooled AUC: {result['few_raw_margin_pooled_auc']:.6f}",
        f"- Weighted source-wise AUC reconstruction: {result['few_raw_margin_weighted_source_auc']:.6f}",
        f"- Observed-Few noisy samples: {result['few_noisy_count']}",
        f"- Many-to-Few noisy samples: {result['many_to_few_count']} "
        f"({100.0 * result['many_to_few_share_of_few_noisy']:.2f}% of Few noise)",
        f"- Class-adaptive Few q90 threshold (score-only): {result['gate_reference']['few_q90_threshold']:.10f}",
        f"- Reference Few samples above q90: {result['gate_reference']['expected_few_acceptance_at_reference']}",
        "",
        "## Observed-Few source breakdown",
        "",
        "| Source group | Total in observed Few | Clean | Noisy | Share of Few noise |",
        "|---|---:|---:|---:|---:|",
        *source_rows,
        "",
        "## Pairwise separation: Clean Few vs noisy source-to-Few",
        "",
        "| Noisy source | n negatives | Raw-margin AUC | Raw-margin AP | LC+prior AUC | LC+prior AP |",
        "|---|---:|---:|---:|---:|---:|",
        *pairwise_rows,
        "",
        "## Pooled AUC decomposition check",
        "",
        "| Score | Pooled Few AUC | Weighted source AUC | Absolute difference |",
        "|---|---:|---:|---:|",
        *decomp_rows,
        "",
        "## Logit / entropy availability",
        "",
        result["logit_entropy_status"],
        "",
        "## Interpretation guardrails",
        "",
        "- Source group is derived from clean_label only for this offline evaluation.",
        "- Observed group is derived from observed_label and the fixed clean IF=100 class-count vector.",
        "- The gate reference uses observed_label and margin scores only; clean_label/is_clean are not used.",
        "- Pairwise AUC compares Clean Few positives against one source-group's noisy Few negatives.",
        "- The weighted pairwise AUC must reconstruct the pooled Few AUC; otherwise the script stops.",
        "- No model is loaded, no GPU is initialized, and no inference or training is run.",
        "- A scalar margin cannot recover the predicted class, maximum head-class logit, or prediction entropy.",
        "",
        "## Outputs",
        "",
        "- observed_group_composition.csv",
        "- noisy_source_to_observed_counts.csv",
        "- noisy_source_to_observed_row_percent.csv",
        "- few_source_breakdown.csv",
        "- few_margin_distribution_summary.csv",
        "- few_pairwise_source_metrics.csv",
        "- few_auc_source_decomposition.csv",
        "- gate_reference.json (score-only; usable by the subsequent training gate)",
    ]
    (args.output_dir / "ATTRIBUTION_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print((args.output_dir / "ATTRIBUTION_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
