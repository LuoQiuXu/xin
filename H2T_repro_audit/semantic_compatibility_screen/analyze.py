"""Recipient-level audit of the provided screen; no feature or model updates."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from coverage import TASK, DEFAULT_OUT, SEED, guard_io, sha, write_json


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_save(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cluster_intervals(labels, values, repetitions=2000):
    """Resample fine-class clusters, preserving all recipients in every draw."""
    classes = np.unique(labels)
    sums = np.array([values[labels == c].sum(0) for c in classes])
    counts = np.array([(labels == c).sum() for c in classes])
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(classes), len(classes))
        estimates.append(sums[chosen].sum(0) / counts[chosen].sum())
    bounds = np.quantile(estimates, [.025, .975], axis=0)
    return bounds.tolist()


def grouped(rows, key):
    result = []
    for value in sorted({r[key] for r in rows}):
        subset = [r for r in rows if r[key] == value]
        active = [r for r in subset if r["n_matches"]]
        result.append(dict(group_id=int(value), n_recipients=len(subset), n_matched_recipients=len(active),
            clean_correct=sum(r["clean_correct"] for r in subset),
            clean_accuracy_pp=100*np.mean([r["clean_correct"] for r in subset]),
            n_matched_pairs=sum(r["n_matches"] for r in subset),
            accuracy_delta_pp=float(np.mean([r["accuracy_delta_pp"] for r in active])) if active else None,
            margin_delta=float(np.mean([r["margin_delta"] for r in active])) if active else None,
            mean_relative_distance_gap=float(np.mean([r["relative_gap_mean"] for r in active])) if active else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(TASK)
    dest = args.analysis_output.resolve() if args.analysis_output else out
    assert dest.is_relative_to(out)
    dest.mkdir(parents=True, exist_ok=True)
    targets = [dest / n for n in ["RECIPIENT_ANALYSIS.csv", "CLASS_ANALYSIS.csv", "SUPERCLASS_ANALYSIS.csv",
        "MATCH_DISTANCE_AUDIT.csv", "ANALYSIS_SUMMARY.json", "SCIENTIFIC_DECISION.json", "SEMANTIC_SCREEN_DIAGNOSTICS.png", "ANALYSIS_GUARD.json"]]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refuse overwriting analysis outputs")
    guard = guard_io()
    metadata = json.loads((out / "PAIR_INPUT.json").read_text(encoding="utf-8"))
    rules = json.loads((out / "PRE_ANALYSIS_PLAN.json").read_text(encoding="utf-8"))
    assert sha(out / "PRE_ANALYSIS_PLAN.json") == metadata["analysis_plan_sha256"]
    path = TASK / metadata["path"]
    assert sha(path) == metadata["sha256"]
    with np.load(path, allow_pickle=False) as z:
        x = {k: z[k] for k in z.files}
    with np.load(out / "primary_cache/feasibility_plan.npz", allow_pickle=False) as z:
        counts, lt = z["LT_class_counts"], z["LT_raw_ids"]
    ri, y, di, dy = (x[k] for k in ["rec_id", "rec_label", "donor_id", "donor_label"])
    fr, fd, mapping, w, b = (x[k] for k in ["rec_feat", "donor_feat", "coarse_label", "weight", "bias"])
    assert ri.shape == y.shape == di.shape == dy.shape == (13440,)
    assert fr.shape == fd.shape == (13440, 64) and w.shape == (100, 64) and b.shape == (100,)
    assert np.isfinite(fr).all() and np.isfinite(fd).all() and np.isfinite(w).all() and np.isfinite(b).all()
    assert np.isin(ri, lt).all() and np.isin(di, lt).all()
    assert (counts[y] < 20).all() and (counts[dy] > 100).all() and (ri != di).all() and (y != dy).all()
    assert "lambda_donor" not in x
    is_same = mapping[y] == mapping[dy]
    distance = np.linalg.norm(fd-fr, axis=1)
    clean_logits, mixed_logits = fr@w.T+b, (.75*fr+.25*fd)@w.T+b
    def margin(logits):
        return logits[np.arange(len(y)), y] - np.max(np.where(np.arange(100)[None, :] == y[:, None], -np.inf, logits), axis=1)
    clean_correct, correct = clean_logits.argmax(1) == y, mixed_logits.argmax(1) == y
    clean_margin, mixed_margin = margin(clean_logits), margin(mixed_logits)
    keys = {(int(r), int(d)): i for i, (r, d) in enumerate(zip(ri, di))}
    assert len(keys) == len(ri)
    original = table(out / "screen_output/matched_pairs.csv")
    enriched, errors = [], []
    for row in original:
        rid, same_id, diff_id = (int(row[k]) for k in ["recipient_id", "same_donor_id", "diff_donor_id"])
        si, ni = keys[rid, same_id], keys[rid, diff_id]
        assert is_same[si] and not is_same[ni] and y[si] == y[ni]
        assert float(row["same_mass"]) == float(row["diff_mass"]) == .25
        for field, expected in [("same_correct", int(correct[si])), ("diff_correct", int(correct[ni])),
                                ("clean_correct", int(clean_correct[si]))]:
            assert int(row[field]) == expected
        for field, expected in [("same_distance", distance[si]), ("diff_distance", distance[ni]),
                                ("same_margin", mixed_margin[si]), ("diff_margin", mixed_margin[ni])]:
            errors.append(abs(float(row[field])-expected))
        assert max(errors[-4:]) < 1e-9
        ds, dn = distance[si], distance[ni]
        gap = float(abs(ds-dn))
        relative = float(2*gap/(ds+dn)) if ds+dn else 0.
        enriched.append(dict(recipient_id=rid, same_donor_id=same_id, diff_donor_id=diff_id,
            fine_id=int(y[si]), coarse_id=int(mapping[y[si]]), same_distance=float(ds), diff_distance=float(dn),
            distance_gap=gap, signed_distance_gap=float(ds-dn), relative_gap=relative,
            tight_caliper=bool(relative <= .05), accuracy_delta=int(correct[si])-int(correct[ni]),
            margin_delta=float(mixed_margin[si]-mixed_margin[ni]),
            same_correct=int(correct[si]), diff_correct=int(correct[ni]),
            same_margin=float(mixed_margin[si]), diff_margin=float(mixed_margin[ni]),
            same_donor_LT_count=int(counts[dy[si]]), diff_donor_LT_count=int(counts[dy[ni]]),
            same_donor_norm=float(np.linalg.norm(fd[si])), diff_donor_norm=float(np.linalg.norm(fd[ni]))))
    csv_save(dest / "MATCH_DISTANCE_AUDIT.csv", [{k:r[k] for k in ["recipient_id", "same_donor_id", "diff_donor_id",
        "fine_id", "coarse_id", "same_distance", "diff_distance", "distance_gap", "signed_distance_gap", "relative_gap", "tight_caliper"]} for r in enriched])
    recipients = []
    for rid in np.unique(ri):
        idx = np.flatnonzero(ri == rid)
        assert np.array_equal(fr[idx], np.repeat(fr[idx[:1]], len(idx), axis=0))
        assert np.unique(y[idx]).size == 1
        si, ni = idx[is_same[idx]], idx[~is_same[idx]]
        assert len(si) == len(ni) == 32
        match = [r for r in enriched if r["recipient_id"] == rid]
        tight = [r for r in match if r["tight_caliper"]]
        row = dict(recipient_id=int(rid), fine_id=int(y[idx[0]]), coarse_id=int(mapping[y[idx[0]]]),
            clean_correct=int(clean_correct[idx[0]]), clean_margin=float(clean_margin[idx[0]]),
            same_candidates=len(si), different_candidates=len(ni),
            unmatched_accuracy_delta_pp=float(100*(correct[si].mean()-correct[ni].mean())),
            unmatched_margin_delta=float(mixed_margin[si].mean()-mixed_margin[ni].mean()),
            unmatched_same_distance=float(distance[si].mean()), unmatched_diff_distance=float(distance[ni].mean()),
            n_matches=len(match), n_tight_matches=len(tight))
        for field in ["accuracy_delta", "margin_delta", "distance_gap", "signed_distance_gap", "relative_gap",
                      "same_distance", "diff_distance", "same_correct", "diff_correct", "same_margin", "diff_margin",
                      "same_donor_LT_count", "diff_donor_LT_count", "same_donor_norm", "diff_donor_norm"]:
            name = "accuracy_delta_pp" if field == "accuracy_delta" else ("relative_gap_mean" if field == "relative_gap" else field)
            row[name] = float(np.mean([r[field] for r in match])) * (100 if field == "accuracy_delta" else 1) if match else None
        row["relative_gap_p90"] = float(np.quantile([r["relative_gap"] for r in match], .9)) if match else None
        row["tight_accuracy_delta_pp"] = float(100*np.mean([r["accuracy_delta"] for r in tight])) if tight else None
        row["tight_margin_delta"] = float(np.mean([r["margin_delta"] for r in tight])) if tight else None
        recipients.append(row)
    csv_save(dest / "RECIPIENT_ANALYSIS.csv", recipients)
    by_class, by_coarse = grouped(recipients, "fine_id"), grouped(recipients, "coarse_id")
    csv_save(dest / "CLASS_ANALYSIS.csv", by_class)
    csv_save(dest / "SUPERCLASS_ANALYSIS.csv", by_coarse)
    active = [r for r in recipients if r["n_matches"]]
    values = np.array([[r["accuracy_delta_pp"], r["margin_delta"]] for r in active])
    labels = np.array([r["fine_id"] for r in active])
    bounds = cluster_intervals(labels, values)
    mean = values.mean(0)
    sd = float(values[:, 1].std(ddof=1))
    standardized = float(mean[1]/sd) if sd else None
    relative = np.array([r["relative_gap"] for r in enriched])
    sign = np.sign(mean[1])
    class_consistency = float(np.mean([sign*r["margin_delta"] > 1e-12 for r in by_class if r["margin_delta"] is not None]))
    coarse_consistency = float(np.mean([sign*r["margin_delta"] > 1e-12 for r in by_coarse if r["margin_delta"] is not None]))
    clean_rate = float(np.mean([r["clean_correct"] for r in recipients]))
    maximum_class_share = float(max(r["n_matched_recipients"] for r in by_class)/len(active))
    quality = dict(unique_matched_recipients=len(active) >= rules["minimum_unique_matched_recipients"],
        matched_fine_classes=len(np.unique(labels)) >= rules["minimum_matched_fine_classes"],
        matched_superclasses=sum(r["n_matched_recipients"] > 0 for r in by_coarse) >= rules["minimum_matched_superclasses"],
        class_concentration=maximum_class_share <= rules["maximum_single_fine_recipient_share"],
        distance_median=float(np.median(relative)) <= rules["acceptable_global_median_relative_gap"],
        distance_p90=float(np.quantile(relative, .9)) <= rules["acceptable_global_p90_relative_gap"],
        reference_clean_correct_fraction=clean_rate >= rules["minimum_reference_clean_correct_fraction"])
    signal = dict(practical_accuracy=abs(mean[0]) >= rules["practical_minimum_accuracy_difference_pp"],
        practical_standardized_margin=standardized is not None and abs(standardized) >= rules["practical_minimum_standardized_margin_difference"],
        aligned_accuracy_margin_direction=np.sign(mean[0]) == sign and sign != 0,
        fine_class_direction=class_consistency >= 2/3, superclass_direction=coarse_consistency >= 2/3,
        both_cluster_intervals_exclude_zero_same_direction=bool(
            (bounds[0][0] > 0 and bounds[0][1] > 0) or (bounds[1][0] < 0 and bounds[1][1] < 0)))
    signal = {k: bool(v) for k, v in signal.items()}
    state = "INCONCLUSIVE" if not all(quality.values()) else ("GEOMETRIC_ASSOCIATION_CANDIDATE" if all(signal.values()) else "NO_GO_NO_ROBUST_GEOMETRIC_SIGNAL")
    tight = [r for r in recipients if r["n_tight_matches"]]
    summary = dict(n_recipients=len(recipients), n_matched_pairs=len(enriched), n_independent_recipient_units=len(active),
        n_fine_classes=len(by_class), n_superclasses=len(by_coarse),
        clean_correct=sum(r["clean_correct"] for r in recipients), clean_correct_percent=100*clean_rate,
        clean_wrong_percent=100*(1-clean_rate), clean_margin_mean=float(np.mean([r["clean_margin"] for r in recipients])),
        clean_margin_median=float(np.median([r["clean_margin"] for r in recipients])),
        maximum_fine_class_recipient_share=maximum_class_share,
        receiver_balanced_accuracy_delta_pp=float(mean[0]), receiver_balanced_margin_delta=float(mean[1]),
        standardized_margin_difference=standardized,
        class_balanced_accuracy_delta_pp=float(np.mean([r["accuracy_delta_pp"] for r in by_class])),
        class_balanced_margin_delta=float(np.mean([r["margin_delta"] for r in by_class])),
        fine_class_cluster_bootstrap_95_percent_CI=dict(accuracy_delta_pp=[bounds[0][0], bounds[1][0]], margin_delta=[bounds[0][1], bounds[1][1]]),
        fine_class_direction_fraction=class_consistency, superclass_direction_fraction=coarse_consistency,
        mean_absolute_distance_gap=float(np.mean([r["distance_gap"] for r in enriched])),
        mean_signed_distance_gap=float(np.mean([r["signed_distance_gap"] for r in enriched])),
        relative_distance_gap=dict(mean=float(relative.mean()), median=float(np.median(relative)), p90=float(np.quantile(relative,.9)), maximum=float(relative.max())),
        exact_donor_mass_all_matches=.25,
        tight_distance_sensitivity=dict(relative_caliper=.05, n_pairs=sum(r["tight_caliper"] for r in enriched), n_recipients=len(tight),
            accuracy_delta_pp=float(np.mean([r["tight_accuracy_delta_pp"] for r in tight])),
            margin_delta=float(np.mean([r["tight_margin_delta"] for r in tight]))),
        unmatched_receiver_balanced_accuracy_delta_pp=float(np.mean([r["unmatched_accuracy_delta_pp"] for r in recipients])),
        unmatched_receiver_balanced_margin_delta=float(np.mean([r["unmatched_margin_delta"] for r in recipients])),
        clean_correctness_strata=[dict(clean_correct=bool(flag), n_recipients=len(part),
            accuracy_delta_pp=float(np.mean([r["accuracy_delta_pp"] for r in part])),
            margin_delta=float(np.mean([r["margin_delta"] for r in part])))
            for flag in [True, False] if (part := [r for r in active if bool(r["clean_correct"]) == flag])],
        donor_confounding=dict(same_Many_LT_class_count_mean=float(np.mean([r["same_donor_LT_count"] for r in enriched])),
            diff_Many_LT_class_count_mean=float(np.mean([r["diff_donor_LT_count"] for r in enriched])),
            same_donor_norm_mean=float(np.mean([r["same_donor_norm"] for r in enriched])),
            diff_donor_norm_mean=float(np.mean([r["diff_donor_norm"] for r in enriched]))),
        independent_recomputed_max_source_output_error=float(max(errors)),
        margin_drop_contrast_is_algebraically_same_as_margin_contrast=True,
        caveats=["Same fixed training images, same frozen checkpoint; not a generalization test or training benefit.",
                 "Shared donors and superclasses create dependence beyond the fine-class bootstrap; intervals are descriptive.",
                 "Matching Euclidean distance does not randomize semantic relationships or remove label/frequency/head-geometry confounding.",
                 "Only21 eligible Few classes; omitted9 classes are a selection limitation, not a method failure."])
    write_json(dest / "ANALYSIS_SUMMARY.json", summary)
    write_json(dest / "SCIENTIFIC_DECISION.json", dict(
        scientific_decision="GO_GEOMETRIC_CANDIDATE_ONLY" if state == "GEOMETRIC_ASSOCIATION_CANDIDATE" else "NO_GO",
        diagnostic_status=state, pre_analysis_plan_sha256=metadata["analysis_plan_sha256"],
        design_and_proxy_quality=quality, semantic_signal_requirements=signal,
        failed_quality_requirements=[k for k,v in quality.items() if not v],
        failed_signal_requirements=[k for k,v in signal.items() if not v],
        scope="Frozen scalar-mixture diagnostic in the user-approved210-image/21-class Few subset; Many donors only",
        training_executed=False, publication_or_author_method_failure_claim=False))
    os.environ.setdefault("MPLCONFIGDIR", str(out / "primary_cache/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ds, dn = [r["same_distance"] for r in enriched], [r["diff_distance"] for r in enriched]
    axes[0,0].scatter(ds,dn,s=4,alpha=.2)
    limits=[min(min(ds),min(dn)),max(max(ds),max(dn))]
    axes[0,0].plot(limits,limits,color="#666666",ls="--")
    axes[0,0].set(xlabel="Same-superclass donor distance",ylabel="Different-superclass donor distance",title="Within-recipient matched Euclidean distances")
    axes[0,1].hist(relative*100,bins=35,color="#437ca2")
    axes[0,1].axvline(5,color="#bc4c45",ls="--",label="5% sensitivity caliper")
    axes[0,1].set(xlabel="Relative distance gap (%)",ylabel="Matched pairs",title="Residual distance mismatch (not exact control)")
    axes[0,1].legend(fontsize=8)
    deltas=[r["accuracy_delta_pp"] for r in by_class]
    axes[1,0].barh([str(r["group_id"]) for r in by_class],deltas,color=["#16866c" if v>0 else "#bf5b50" for v in deltas])
    axes[1,0].axvline(0,color="#555555",lw=.8)
    axes[1,0].set(xlabel="Same-minus-different label consistency (pp)",ylabel="Approved Few fine class ID",title="Class-level recipient-balanced contrasts")
    axes[1,1].scatter([r["clean_margin"] for r in active],[r["margin_delta"] for r in active],
        c=["#16866c" if r["clean_correct"] else "#bf5b50" for r in active],s=12,alpha=.65)
    axes[1,1].axhline(0,color="#555555",lw=.8)
    axes[1,1].axvline(0,color="#555555",lw=.8)
    axes[1,1].set(xlabel="Clean receiver margin",ylabel="Same-minus-different mixed margin",title="Frozen-head proxy: green clean-correct, red wrong")
    fig.suptitle("Semantic compatibility: 210 eligible Few receivers, Many donors, fixed donor mass .25\nOne frozen Clean Stage I; exploratory associations, not Stage-II utility",fontsize=12)
    fig.savefig(dest / "SEMANTIC_SCREEN_DIAGNOSTICS.png",dpi=155)
    plt.close(fig)
    assert not any(guard.values())
    write_json(dest / "ANALYSIS_GUARD.json", guard)
    print(json.dumps(dict(status=state, recipient_accuracy_delta_pp=float(mean[0]), recipient_margin_delta=float(mean[1]),
                         quality=quality, signal=signal),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
