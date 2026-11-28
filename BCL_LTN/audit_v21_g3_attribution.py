"""Offline boundary/representation attribution & Module A shadow replay for NTC-BCL v2.1 G3.

Uses ONLY frozen Ep200 checkpoints (v2 & G3), saved 36 EM snapshots, saved
final_test_metrics.json, and TRAIN clean subset (never opening official TEST).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from BCL_AUDIT.cifar100lt.data import build_cifar_transforms
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_LTN.audit_epoch160_margin_reliability import EvalOnlyTrainDataset
from BCL_LTN.make_noisy_lt import group_class_ids
from BCL_LTN.train_noisy_bcl import OfficialTestAudit, json_dump, safe_torch_load
from BCL_LTN.train_ntc_bcl_v2 import (
    BATCH_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_INDICES,
    DEFAULT_STAGE1_DATA_ROOT,
    K,
    N,
    load_regime_observed_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "BCL_LTN" / "outputs" / "NTC_BCL_V21_G3_OFFLINE_ATTRIBUTION_20261128"

RUN_PAIRS = {
    "sym04": (
        ROOT / "BCL_LTN" / "outputs" / "NTC_BCL_V2_sym04_seed42_20261126",
        ROOT / "BCL_LTN" / "outputs" / "NTC_BCL_V21_G3_sym04_seed42_20261127",
    ),
    "cifar100n_real": (
        ROOT / "BCL_LTN" / "outputs" / "NTC_BCL_V2_cifar100n_real_seed42_20261127",
        ROOT / "BCL_LTN" / "outputs" / "NTC_BCL_V21_G3_cifar100n_real_seed42_20261127",
    ),
}


@torch.inference_mode()
def extract_features_and_logits(
    ckpt_path: Path, state_key: str, dataset, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ckpt = safe_torch_load(ckpt_path, torch.device("cpu"))
    model = BCLCIFARModel().to(device).eval()
    model.load_state_dict(ckpt[state_key], strict=True)
    eval_ds = EvalOnlyTrainDataset(dataset.pixels, dataset.indices, dataset.eval_transform)
    loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
    )
    feats = np.empty((N, 64), dtype=np.float64)
    logits = np.empty((N, K), dtype=np.float64)
    for images, pos in loader:
        f, z, _, _ = model(images.to(device, non_blocking=True), compute_contrastive=False)
        idx = pos.numpy()
        feats[idx] = f.double().cpu().numpy()
        logits[idx] = z.double().cpu().numpy()
    w = model.classifier.weight.detach().double().cpu().numpy()
    b = model.classifier.bias.detach().double().cpu().numpy()
    return feats, logits, w, b


def group_acc_and_leakage(
    preds: np.ndarray,
    targets: np.ndarray,
    logits_eff: np.ndarray,
    groups: Dict[str, np.ndarray],
) -> Dict[str, object]:
    class_to_group = {}
    for gname in ("Many", "Medium", "Few"):
        for c in groups[gname]:
            class_to_group[int(c)] = gname
    out: Dict[str, object] = {}
    for gname in ("Overall", "Many", "Medium", "Few"):
        ids = np.arange(K) if gname == "Overall" else groups[gname]
        mask = np.isin(targets, ids)
        acc = float(100.0 * np.mean(preds[mask] == targets[mask]))
        # Compute logit margin z_y - max_{j != y} z_j
        sub_l = logits_eff[mask]
        sub_y = targets[mask]
        true_l = sub_l[np.arange(len(sub_y)), sub_y]
        masked_l = sub_l.copy()
        masked_l[np.arange(len(sub_y)), sub_y] = -1e18
        best_other = masked_l.max(axis=1)
        out[gname] = {
            "accuracy": acc,
            "count": int(mask.sum()),
            "mean_margin": float(np.mean(true_l - best_other)),
        }
    few_mask = np.isin(targets, groups["Few"])
    few_pred = preds[few_mask]
    few_targ = targets[few_mask]
    wrong_few = few_mask & (preds != targets)
    to_many = float(np.mean(np.isin(preds[few_mask], groups["Many"]))) * 100.0
    to_med = float(np.mean(np.isin(preds[few_mask], groups["Medium"]))) * 100.0
    to_few_wrong = float(np.mean((preds[few_mask] != targets[few_mask]) & np.isin(preds[few_mask], groups["Few"]))) * 100.0
    pred_shares = {
        gname: float(np.mean(np.isin(preds, groups[gname]))) * 100.0
        for gname in ("Many", "Medium", "Few")
    }
    out["few_confusion_pct"] = {
        "correct": float(100.0 * np.mean(few_pred == few_targ)),
        "to_many": to_many,
        "to_medium": to_med,
        "to_other_few": to_few_wrong,
        "wrong_total": float(100.0 * np.mean(wrong_few[few_mask])),
    }
    out["pred_group_share_pct"] = pred_shares
    return out


def feature_layer_metrics(
    feats: np.ndarray,
    clean_mask: np.ndarray,
    clean_labels: np.ndarray,
    weights: np.ndarray,
    groups: Dict[str, np.ndarray],
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Layer 2 & Layer 3 metrics strictly on TRAIN clean subset (with LOO NCM & k-NN)."""
    h = feats[clean_mask]
    y = clean_labels[clean_mask]
    m = len(y)
    h_norm = h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-12, None)

    centroids = np.zeros((K, 64), dtype=np.float64)
    centroids_norm = np.zeros((K, 64), dtype=np.float64)
    counts = np.bincount(y, minlength=K)
    valid = counts > 0
    for c in range(K):
        if counts[c] == 0:
            continue
        idx = y == c
        centroids[c] = h[idx].mean(axis=0)
        centroids_norm[c] = h_norm[idx].mean(axis=0)
    cent_unit = np.zeros_like(centroids)
    cent_unit[valid] = centroids[valid] / np.clip(
        np.linalg.norm(centroids[valid], axis=1, keepdims=True), 1e-12, None
    )

    # Intra-class & inter-class centroid stats per class
    intra_l2 = np.full(K, np.nan, dtype=np.float64)
    intra_cos_dist = np.full(K, np.nan, dtype=np.float64)
    inter_min_l2 = np.full(K, np.nan, dtype=np.float64)
    inter_max_cos = np.full(K, np.nan, dtype=np.float64)
    for c in range(K):
        if counts[c] == 0:
            continue
        idx = y == c
        diff = h[idx] - centroids[c]
        intra_l2[c] = float(np.mean(np.linalg.norm(diff, axis=1)))
        intra_cos_dist[c] = float(np.mean(1.0 - h_norm[idx] @ cent_unit[c]))
        other_l2 = np.where(valid, np.linalg.norm(centroids - centroids[c], axis=1), np.inf)
        other_l2[c] = np.inf
        inter_min_l2[c] = float(other_l2.min())
        other_cos = np.where(valid, cent_unit @ cent_unit[c], -np.inf)
        other_cos[c] = -np.inf
        inter_max_cos[c] = float(other_cos.max())

    # Standard NCM & LOO NCM (mask empty-clean classes so they never win argmin/argmax)
    dists_std = np.sum((h[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    dists_std[:, ~valid] = np.inf
    ncm_std_pred = np.argmin(dists_std, axis=1)

    cos_std_mat = h_norm @ cent_unit.T
    cos_std_mat[:, ~valid] = -np.inf
    cos_std_pred = np.argmax(cos_std_mat, axis=1)

    sum_h = centroids * counts[:, None]
    ncm_loo_l2_pred = np.empty(m, dtype=np.int64)
    ncm_loo_cos_pred = np.empty(m, dtype=np.int64)
    for i in range(m):
        c = int(y[i])
        d_l2 = dists_std[i].copy()
        if counts[c] > 1:
            mu_loo = (sum_h[c] - h[i]) / float(counts[c] - 1)
            d_l2[c] = float(np.sum((h[i] - mu_loo) ** 2))
        else:
            d_l2[c] = np.inf
        ncm_loo_l2_pred[i] = int(np.argmin(d_l2))

        c_scores = cos_std_mat[i].copy()
        if counts[c] > 1:
            mu_loo_c = (sum_h[c] - h[i]) / float(counts[c] - 1)
            mu_loo_u = mu_loo_c / max(float(np.linalg.norm(mu_loo_c)), 1e-12)
            c_scores[c] = float(h_norm[i] @ mu_loo_u)
        else:
            c_scores[c] = -np.inf
        ncm_loo_cos_pred[i] = int(np.argmax(c_scores))

    # LOO k-NN (cosine, k=5 and k=15)
    sim = h_norm @ h_norm.T
    np.fill_diagonal(sim, -np.inf)
    knn5_pred = np.empty(m, dtype=np.int64)
    knn15_pred = np.empty(m, dtype=np.int64)
    for k_val, out_arr in ((5, knn5_pred), (15, knn15_pred)):
        top_idx = np.argpartition(-sim, kth=k_val - 1, axis=1)[:, :k_val]
        for i in range(m):
            nbr_y = y[top_idx[i]]
            nbr_s = np.maximum(sim[i, top_idx[i]], 0.0)
            scores = np.bincount(nbr_y, weights=nbr_s, minlength=K)
            out_arr[i] = int(np.argmax(scores))

    # Classifier weight vs centroid alignment
    w_norm = np.linalg.norm(weights, axis=1)
    w_unit = weights / np.clip(w_norm[:, None], 1e-12, None)
    cos_w_mu = np.where(valid, np.sum(w_unit * cent_unit, axis=1), np.nan)

    def summarize_pred(pred: np.ndarray) -> Dict[str, float]:
        res = {"Overall": float(100.0 * np.mean(pred == y))}
        for gname in ("Many", "Medium", "Few"):
            mk = np.isin(y, groups[gname])
            res[gname] = float(100.0 * np.mean(pred[mk] == y[mk]))
        return res

    summary: Dict[str, object] = {
        "ncm_std_l2": summarize_pred(ncm_std_pred),
        "ncm_std_cos": summarize_pred(cos_std_pred),
        "ncm_loo_l2": summarize_pred(ncm_loo_l2_pred),
        "ncm_loo_cos": summarize_pred(ncm_loo_cos_pred),
        "knn5_loo_cos": summarize_pred(knn5_pred),
        "knn15_loo_cos": summarize_pred(knn15_pred),
        "geometry_by_group": {},
    }
    for gname in ("Many", "Medium", "Few"):
        ids = groups[gname]
        vids = ids[valid[ids]]
        summary["geometry_by_group"][gname] = {
            "clean_classes_present": int(len(vids)),
            "clean_classes_total": int(len(ids)),
            "intra_l2_mean": float(intra_l2[vids].mean()),
            "intra_cos_dist_mean": float(intra_cos_dist[vids].mean()),
            "inter_min_l2_mean": float(inter_min_l2[vids].mean()),
            "inter_max_cos_mean": float(inter_max_cos[vids].mean()),
            "weight_norm_mean": float(w_norm[ids].mean()),
            "weight_norm_min": float(w_norm[ids].min()),
            "cos_w_mu_mean": float(cos_w_mu[vids].mean()),
            "cos_w_mu_min": float(cos_w_mu[vids].min()),
        }
    per_class = {
        "w_norm": w_norm,
        "cos_w_mu": cos_w_mu,
        "intra_l2": intra_l2,
        "inter_min_l2": inter_min_l2,
        "ncm_loo_cos_class_acc": np.array(
            [100.0 * np.mean(ncm_loo_cos_pred[y == c] == c) if counts[c] > 0 else np.nan for c in range(K)],
            dtype=np.float64,
        ),
        "knn5_loo_class_acc": np.array(
            [100.0 * np.mean(knn5_pred[y == c] == c) if counts[c] > 0 else np.nan for c in range(K)],
            dtype=np.float64,
        ),
        "clean_count": counts,
    }
    return summary, per_class


def module_a_decomposition_and_shadow(regime: str, g3_dir: Path) -> Dict[str, object]:
    snaps = sorted((g3_dir / "em_snapshots").glob("epoch_*.npz"))
    rows = [
        json.loads(l)
        for l in (g3_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip() and json.loads(l).get("em_update") is not None
    ]
    assert len(snaps) == 36 and len(rows) == 36
    epochs = [int(r["epoch"]) for r in rows]

    raw_pi, raw_diag, raw_eta = [], [], []
    bl_pi, bl_diag, bl_eta = [], [], []
    f20, f50, fall = [], [], []
    for sp, r in zip(snaps, rows):
        z = np.load(sp, allow_pickle=False)
        rp = z["raw_em_prior"].astype(np.float64)
        rt = z["raw_em_transition"].astype(np.float64)
        bp = z["blended_prior"].astype(np.float64)
        bt = z["blended_transition"].astype(np.float64)
        raw_pi.append(rp)
        raw_diag.append(np.diag(rt))
        raw_eta.append(float(1.0 - np.dot(rp, np.diag(rt))))
        bl_pi.append(bp)
        bl_diag.append(np.diag(bt))
        bl_eta.append(float(1.0 - np.dot(bp, np.diag(bt))))
        eu = r["em_update"]
        f20.append(float(eu["f_top20"]))
        f50.append(float(eu["f_top50"]))
        fall.append(float(eu["f_all_macro"]))

    g_gap = np.array(f20) - np.array(fall)
    raw_eta_arr = np.array(raw_eta)
    bl_eta_arr = np.array(bl_eta)

    decomp_rows = []
    for i in range(1, len(epochs)):
        d_raw_t = float(-0.5 * np.dot(raw_pi[i] + raw_pi[i - 1], raw_diag[i] - raw_diag[i - 1]))
        d_raw_p = float(-0.5 * np.dot(raw_pi[i] - raw_pi[i - 1], raw_diag[i] + raw_diag[i - 1]))
        d_bl_t = float(-0.5 * np.dot(bl_pi[i] + bl_pi[i - 1], bl_diag[i] - bl_diag[i - 1]))
        d_bl_p = float(-0.5 * np.dot(bl_pi[i] - bl_pi[i - 1], bl_diag[i] + bl_diag[i - 1]))
        assert abs((d_raw_t + d_raw_p) - (raw_eta_arr[i] - raw_eta_arr[i - 1])) < 1e-12
        assert abs((d_bl_t + d_bl_p) - (bl_eta_arr[i] - bl_eta_arr[i - 1])) < 1e-12
        decomp_rows.append({
            "epoch_from": epochs[i - 1],
            "epoch_to": epochs[i],
            "raw_eta_from": float(raw_eta_arr[i - 1]),
            "raw_eta_to": float(raw_eta_arr[i]),
            "d_raw_total": float(raw_eta_arr[i] - raw_eta_arr[i - 1]),
            "d_raw_T": d_raw_t,
            "d_raw_pi": d_raw_p,
            "bl_eta_from": float(bl_eta_arr[i - 1]),
            "bl_eta_to": float(bl_eta_arr[i]),
            "d_bl_total": float(bl_eta_arr[i] - bl_eta_arr[i - 1]),
            "d_bl_T": d_bl_t,
            "d_bl_pi": d_bl_p,
            "f_top20": f20[i],
            "f_all_macro": fall[i],
            "G_t": float(g_gap[i]),
            "d_G_step": float(g_gap[i] - g_gap[i - 1]),
        })

    # Write CSV
    csv_path = OUT_ROOT / f"{regime}_module_a_decomposition.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(decomp_rows[0].keys()))
        w.writeheader()
        w.writerows(decomp_rows)

    # Window summaries: Ep55->160 vs Ep160->195 vs Ep160->180
    def span_sum(ep_a: int, ep_b: int) -> Dict[str, float]:
        ia, ib = epochs.index(ep_a), epochs.index(ep_b)
        sub = decomp_rows[ia:ib]
        return {
            "raw_eta_change": float(raw_eta_arr[ib] - raw_eta_arr[ia]),
            "raw_T_part": float(sum(r["d_raw_T"] for r in sub)),
            "raw_pi_part": float(sum(r["d_raw_pi"] for r in sub)),
            "blended_eta_change": float(bl_eta_arr[ib] - bl_eta_arr[ia]),
            "blended_T_part": float(sum(r["d_bl_T"] for r in sub)),
            "blended_pi_part": float(sum(r["d_bl_pi"] for r in sub)),
            "df_top20": float(f20[ib] - f20[ia]),
            "df_all": float(fall[ib] - fall[ia]),
            "dG": float(g_gap[ib] - g_gap[ia]),
        }

    # Shadow replay of 3-step (15-ep) & 4-step (20-ep) candidate trigger
    shadow_events_3step = []
    # History windows prior to index i (starting after Ep40 ramp completes, i >= 5 -> Ep45+)
    for i in range(6, len(epochs)):
        # Consecutive 3-step monotonic drop in raw_eta
        consec3 = (
            raw_eta_arr[i] < raw_eta_arr[i - 1] < raw_eta_arr[i - 2] < raw_eta_arr[i - 3]
        )
        d3_eta = float(raw_eta_arr[i] - raw_eta_arr[i - 3])
        d3_g = float(g_gap[i] - g_gap[i - 3])
        # Historical 3-step distribution up to i-1 (only k >= 4 so after warmup+ramp)
        hist_d3_eta = [raw_eta_arr[k] - raw_eta_arr[k - 3] for k in range(7, i)]
        hist_d3_g = [g_gap[k] - g_gap[k - 3] for k in range(7, i)]
        eta_p90_drop = float(np.quantile(np.abs(hist_d3_eta), 0.90)) if len(hist_d3_eta) >= 5 else None
        g_p90_rise = float(np.quantile(hist_d3_g, 0.90)) if len(hist_d3_g) >= 5 else None
        sig1 = bool(consec3 and eta_p90_drop is not None and (-d3_eta > eta_p90_drop))
        sig2 = bool(d3_g > 0 and g_p90_rise is not None and d3_g > g_p90_rise)
        shadow_events_3step.append({
            "epoch": epochs[i],
            "consec3_drop": bool(consec3),
            "d3_raw_eta": d3_eta,
            "hist_3step_abs_eta_p90": eta_p90_drop,
            "d3_G": d3_g,
            "hist_3step_dG_p90": g_p90_rise,
            "sig1_eta_drift": sig1,
            "sig2_G_excess": sig2,
            "both_triggered": bool(sig1 and sig2),
        })

    # Pre-Ep160 vs Ep160..195 stats for 15-ep & 20-ep windows
    idx160 = epochs.index(160)
    pre_20_f20 = [abs(f20[k] - f20[k - 4]) for k in range(8, idx160 + 1)]
    pre_20_g = [g_gap[k] - g_gap[k - 4] for k in range(8, idx160 + 1)]
    pre_20_eta = [raw_eta_arr[k] - raw_eta_arr[k - 4] for k in range(8, idx160 + 1)]
    return {
        "spans": {
            "Ep55_to_Ep160": span_sum(55, 160),
            "Ep135_to_Ep155": span_sum(135, 155),
            "Ep160_to_Ep180": span_sum(160, 180),
            "Ep160_to_Ep195": span_sum(160, 195),
            "Ep55_to_Ep195": span_sum(55, 195),
        },
        "pre160_20ep_baseline": {
            "abs_df20_median": float(np.median(pre_20_f20)),
            "abs_df20_p90": float(np.quantile(pre_20_f20, 0.90)),
            "abs_df20_max": float(np.max(pre_20_f20)),
            "dG_median": float(np.median(pre_20_g)),
            "dG_p90": float(np.quantile(pre_20_g, 0.90)),
            "dG_max": float(np.max(pre_20_g)),
            "d_raw_eta_min": float(np.min(pre_20_eta)),
            "d_raw_eta_p10": float(np.quantile(pre_20_eta, 0.10)),
        },
        "shadow_3step_events": shadow_events_3step,
        "decomposition_csv": str(csv_path),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_gate = OfficialTestAudit(DEFAULT_DATA_ROOT)
    # Audit hook check: guarantee ZERO TEST reads anywhere
    summary_all: Dict[str, object] = {}

    for regime, (v2_dir, g3_dir) in RUN_PAIRS.items():
        rdir = DEFAULT_STAGE1_DATA_ROOT / regime
        dataset, _, _ = load_regime_observed_inputs(rdir, DEFAULT_DATA_ROOT, DEFAULT_INDICES)
        clean = np.load(rdir / "truth" / "clean_labels.npy", allow_pickle=False)
        noisy = np.load(rdir / "truth" / "is_noisy.npy", allow_pickle=False)
        clean_mask = ~noisy
        clean_counts = np.bincount(clean, minlength=K)
        groups = group_class_ids(clean_counts)
        clean_y = clean[clean_mask]

        ftm_v2 = json.loads((v2_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
        ftm_g3 = json.loads((g3_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
        p_v2_195 = np.load(v2_dir / "em_snapshots" / "epoch_195.npz", allow_pickle=False)["blended_prior"].astype(np.float64)
        p_g3_195 = np.load(g3_dir / "em_snapshots" / "epoch_195.npz", allow_pickle=False)["blended_prior"].astype(np.float64)

        regime_res: Dict[str, object] = {"clean_subset_sizes": {g: int(np.isin(clean_y, groups[g]).sum()) for g in ("Many", "Medium", "Few")}}
        regime_res["clean_subset_sizes"]["Overall"] = int(clean_mask.sum())

        per_class_store = {}
        for tag, run_dir, p_own in (("v2", v2_dir, p_v2_195), ("g3", g3_dir, p_g3_195)):
            ckpt_path = run_dir / "checkpoints" / "epoch_200.pth"
            eval_dict: Dict[str, object] = {}
            for state_key in ("model_state", "ema_model_state"):
                feats, logits, w, _ = extract_features_and_logits(ckpt_path, state_key, dataset, device)
                z_clean = logits[clean_mask]
                # Layer 1 counterfactuals on TRAIN clean
                cf = {}
                cf["y0_raw_tau0"] = group_acc_and_leakage(np.argmax(z_clean, axis=1), clean_y, z_clean, groups)
                cf["y1_plus_log_pi_g3"] = group_acc_and_leakage(
                    np.argmax(z_clean + np.log(p_g3_195)[None, :], axis=1),
                    clean_y,
                    z_clean + np.log(p_g3_195)[None, :],
                    groups,
                )
                cf["y2_plus_log_pi_v2"] = group_acc_and_leakage(
                    np.argmax(z_clean + np.log(p_v2_195)[None, :], axis=1),
                    clean_y,
                    z_clean + np.log(p_v2_195)[None, :],
                    groups,
                )
                tau_grid = {}
                for tau in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
                    zt = z_clean + float(tau) * np.log(p_g3_195)[None, :]
                    tau_grid[f"{tau:+.2f}"] = group_acc_and_leakage(np.argmax(zt, axis=1), clean_y, zt, groups)
                cf["tau_grid_g3_prior"] = tau_grid

                # Also counterfactual re-centering from own prior -> other prior: z + log(p_other) - log(p_own)
                p_other = p_v2_195 if tag == "g3" else p_g3_195
                z_swap = z_clean + np.log(p_other)[None, :] - np.log(p_own)[None, :]
                cf["swap_to_other_prior_diff"] = group_acc_and_leakage(np.argmax(z_swap, axis=1), clean_y, z_swap, groups)

                # Layer 2 & 3 on features
                feat_metrics, per_cls = feature_layer_metrics(feats, clean_mask, clean, w, groups)
                # Also per-class raw classifier accuracy & margin on TRAIN clean
                raw_pred = np.argmax(z_clean, axis=1)
                raw_margins = []
                raw_cls_acc = []
                for c in range(K):
                    mk = clean_y == c
                    if not bool(mk.any()):
                        raw_margins.append(np.nan)
                        raw_cls_acc.append(np.nan)
                        continue
                    zc = z_clean[mk]
                    other = np.delete(zc, c, axis=1).max(axis=1)
                    raw_margins.append(float(np.mean(zc[:, c] - other)))
                    raw_cls_acc.append(float(100.0 * np.mean(raw_pred[mk] == c)))
                per_cls["train_clean_acc"] = np.array(raw_cls_acc)
                per_cls["train_clean_margin"] = np.array(raw_margins)

                eval_dict[state_key] = {
                    "layer1_counterfactuals": cf,
                    "layer2_3_representation_and_weights": feat_metrics,
                }
                if state_key == "model_state":
                    per_class_store[tag] = per_cls
            regime_res[tag] = eval_dict

        # Write reproducible per_class_prior_boundary_audit.csv
        class_to_group = {}
        for gname in ("Many", "Medium", "Few"):
            for c in groups[gname]:
                class_to_group[int(c)] = gname
        pc_test_v2 = ftm_v2["student"]["per_class_correct"]
        pc_test_g3 = ftm_g3["student"]["per_class_correct"]
        audit_csv = OUT_ROOT / f"{regime}_per_class_prior_boundary_audit.csv"
        fieldnames = [
            "class_id", "group", "train_clean_count",
            "prior_v2", "prior_g3", "log_prior_ratio",
            "test_correct_v2", "test_correct_g3", "delta_accuracy",
            "classifier_weight_norm_v2", "classifier_weight_norm_g3",
            "cos_w_mu_v2", "cos_w_mu_g3",
            "train_clean_acc_v2", "train_clean_acc_g3",
            "ncm_loo_cos_acc_v2", "ncm_loo_cos_acc_g3",
            "knn5_loo_acc_v2", "knn5_loo_acc_g3",
            "train_clean_margin_v2", "train_clean_margin_g3",
        ]
        with audit_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            pv, pg = per_class_store["v2"], per_class_store["g3"]
            def _opt(val: float):
                return "" if np.isnan(val) else float(val)
            for c in range(K):
                w.writerow({
                    "class_id": c,
                    "group": class_to_group[c],
                    "train_clean_count": int(pv["clean_count"][c]),
                    "prior_v2": float(p_v2_195[c]),
                    "prior_g3": float(p_g3_195[c]),
                    "log_prior_ratio": float(np.log(p_g3_195[c] / p_v2_195[c])),
                    "test_correct_v2": int(pc_test_v2[c]),
                    "test_correct_g3": int(pc_test_g3[c]),
                    "delta_accuracy": int(pc_test_g3[c]) - int(pc_test_v2[c]),
                    "classifier_weight_norm_v2": float(pv["w_norm"][c]),
                    "classifier_weight_norm_g3": float(pg["w_norm"][c]),
                    "cos_w_mu_v2": _opt(pv["cos_w_mu"][c]),
                    "cos_w_mu_g3": _opt(pg["cos_w_mu"][c]),
                    "train_clean_acc_v2": _opt(pv["train_clean_acc"][c]),
                    "train_clean_acc_g3": _opt(pg["train_clean_acc"][c]),
                    "ncm_loo_cos_acc_v2": _opt(pv["ncm_loo_cos_class_acc"][c]),
                    "ncm_loo_cos_acc_g3": _opt(pg["ncm_loo_cos_class_acc"][c]),
                    "knn5_loo_acc_v2": _opt(pv["knn5_loo_class_acc"][c]),
                    "knn5_loo_acc_g3": _opt(pg["knn5_loo_class_acc"][c]),
                    "train_clean_margin_v2": _opt(pv["train_clean_margin"][c]),
                    "train_clean_margin_g3": _opt(pg["train_clean_margin"][c]),
                })
        regime_res["per_class_audit_csv"] = str(audit_csv)
        regime_res["module_a_shadow"] = module_a_decomposition_and_shadow(regime, g3_dir)
        summary_all[regime] = regime_res

    if test_gate.opens or test_gate.denied:
        raise RuntimeError("Official TEST was touched unexpectedly")
    summary_all["official_test_opens"] = test_gate.opens
    json_dump(OUT_ROOT / "attribution_and_shadow_summary.json", summary_all)
    print(f"DONE official_test_opens={test_gate.opens} output={OUT_ROOT}")


if __name__ == "__main__":
    main()
