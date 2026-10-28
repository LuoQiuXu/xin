"""Apply the locked screening rules and plot existing paired predictions only."""
import csv
import json
import os
from pathlib import Path

from inference import OUT, CACHE, sha, write_json

PAIRS = {"h2t": OUT, "pi_local": OUT / "pi_local_pair"}
COLORS = {"all": "#30343b", "many": "#c13b39", "medium": "#2e78ad", "few": "#13866a"}


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def screen_record(pair, family, group, by, rules):
    conditions = rules["families"][family]
    rows = [by[c, group] for c in conditions]
    all_rows = [by[c, "all"] for c in conditions]
    passed = all(float(r["difference_in_differences_pp"]) <= -rules["minimum_effect_pp"]
                 and int(r["excess_net_correct"]) <= -rules["minimum_excess_net_loss"]
                 and int(r["negative_DiD_images"]) >= rules["minimum_distinct_negative_DiD_images"]
                 for r in rows)
    specific = group != "all" and passed and all(
        float(r["difference_in_differences_pp"]) <=
        float(a["difference_in_differences_pp"]) - rules["minimum_effect_pp"]
        for r, a in zip(rows, all_rows))
    return dict(pair=pair, family=family, group=group, n=int(rows[0]["n"]),
        low_severity_DiD_pp=float(rows[0]["difference_in_differences_pp"]),
        high_severity_DiD_pp=float(rows[1]["difference_in_differences_pp"]),
        low_excess_net_correct=int(rows[0]["excess_net_correct"]),
        high_excess_net_correct=int(rows[1]["excess_net_correct"]),
        low_negative_DiD_images=int(rows[0]["negative_DiD_images"]),
        high_negative_DiD_images=int(rows[1]["negative_DiD_images"]),
        within_checkpoint_candidate=passed, frequency_specific_candidate=specific,
        independent_seed_confirmation=False)


def main():
    rules = json.loads((OUT / "PRE_REGISTERED_SCREEN_RULES.json").read_text(encoding="utf-8"))
    runtime = json.loads((OUT / "INFERENCE_VALIDATION.json").read_text(encoding="utf-8"))["runtime"]
    assert runtime["screen_rules_sha256"] == sha(OUT / "PRE_REGISTERED_SCREEN_RULES.json")
    targets = [OUT / n for n in ("SCREEN_DECISION.json", "SCREEN_CANDIDATES.csv",
                                "GROUPED_PAIRED_DELTAS.png", "GROUPED_DID.png")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refuse to overwrite screening decision or figures")
    did, contrasts, records = {}, {}, []
    for pair, directory in PAIRS.items():
        did[pair] = {(r["condition"], r["group"]): r for r in table(directory / "difference_in_differences.csv")}
        contrasts[pair] = {(r["condition"], r["group"]): r for r in table(directory / "paired_contrasts.csv")}
        for family in rules["families"]:
            for group in rules["groups"]:
                records.append(screen_record(pair, family, group, did[pair], rules))
    with (OUT / "SCREEN_CANDIDATES.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    write_json(OUT / "SCREEN_DECISION.json", dict(
        locked_rules_sha256=sha(OUT / "PRE_REGISTERED_SCREEN_RULES.json"),
        scientific_decision="NO_GO_FOR_NEW_RESEARCH_DIRECTION",
        conditional_screening_candidates=[r for r in records if r["within_checkpoint_candidate"]],
        all_checks=records,
        interpretation="Candidate means repeated severity direction within these saved heads, not independent training-seed replication or a published-method failure.",
        reasons=["The local 25-parameter PIF prototype is not verified author PI-H2T.",
                 "One Stage-I and one Stage-II seed per pair; reused development holdout; class 0 absent.",
                 "Head-tail trade-offs and overly strong fusion semantic noise are discussed by PI-H2T; input-corruption novelty is not established.",
                 "Local Few gains remain positive in every condition; Overall DiD is positive partly while Many accuracy approaches a floor."],
        next_critical_evidence="Author-verified paper-parity PI-H2T/DR matched checkpoint pair, with full PIF/H2TF and checkpoint provenance; unavailable in the present evaluation. No new fusion strategy."))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for data, value, name, ylabel in [
        (contrasts, "delta_pp", "GROUPED_PAIRED_DELTAS.png", "candidate - same-source no-fusion (pp)"),
        (did, "difference_in_differences_pp", "GROUPED_DID.png", "change from clean paired delta (DiD, pp)")]:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
        for row, pair in enumerate(PAIRS):
            for col, family in enumerate(rules["families"]):
                ax = axes[row, col]
                conditions = ["clean"] + rules["families"][family]
                for group in rules["groups"]:
                    ys = [float(data[pair][c, group][value]) for c in conditions]
                    ax.plot(range(3), ys, marker="o", color=COLORS[group], label=group.capitalize(), lw=2)
                ax.axhline(0, color="#777777", lw=.8, ls="--")
                ax.set_xticks(range(3), ["clean", *(["sigma .03", "sigma .08"] if family == "noise" else ["sigma .5", "sigma 1.0"])])
                ax.set_title(("H2T fixed-BN pair" if pair == "h2t" else "Local 25-param PIF ratio-only pair") + " / " + family)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
                if col == 0:
                    ax.legend(ncol=2, frameon=False, fontsize=9)
        fig.suptitle("CIFAR-100 TRAIN development holdout (N=1980, 99 classes)\nSeparate Stage-I cohorts; one saved seed per pair; descriptive screening only", fontsize=13)
        fig.savefig(OUT / name, dpi=155)
        plt.close(fig)
    print("SUMMARY PASS: locked rules applied to all 16 pair/family/group combinations; two figures saved.")


if __name__ == "__main__":
    main()
