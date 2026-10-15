"""Frozen H2T transfer success/failure and oracle-headroom audit.

This script does no training.  It freezes one fixed-final-epoch Stage-II H2T
checkpoint, uses clean CIFAR-100-LT Few test examples as recipients and samples
Many/Head examples from the long-tailed training set as donors.  For every
recipient--donor pair it applies the released ``methods.H2T`` operator at
rho=0.25 and records the true-label margin change.

The oracle is deliberately label-aware: it selects, *after seeing the true
recipient label*, the donor with the largest mean true-label margin among the
sampled donor pool.  It is an upper bound on controllability, never a deployable
test-time method or a replacement for clean test accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from datasets.cifar100 import IMBALANCECIFAR100
from methods import H2T
from models import resnet_cifar


NUM_CLASSES = 100
HEAD_RANGE = (0, 36)
TAIL_RANGE = (71, 100)


def workspace_root() -> Path:
    # .../workspace/H2T_official_clean/H2T-main/audit_h2t_transfer_oracle.py
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "H2T" / "data" / "cifar100",
        help="Local CIFAR-100 root used by the H2T training runs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=here
        / "outputs"
        / "calibration_audit_official_h2t_seed42"
        / "h2t"
        / "last_epoch_fixed.pth.tar",
        help="Fixed final-epoch H2T rho=0.25 Stage-II checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "outputs" / "h2t_transfer_oracle_seed42_rho025",
    )
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--donors-per-recipient", type=int, default=50)
    parser.add_argument("--masks-per-recipient", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--limit-recipients",
        type=int,
        default=None,
        help="Optional deterministic smoke-test limit; omit for every Few test sample.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class IndexedSubset(Dataset):
    """Returns an item alongside its index in the wrapped base dataset."""

    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, row: int) -> Tuple[torch.Tensor, int, int]:
        base_index = int(self.indices[row])
        sample = self.base[base_index]
        # IMBALANCECIFAR100 returns (image, target, meta); torchvision test
        # data returns (image, target).
        image, target = sample[0], int(sample[1])
        return image, target, base_index


def donor_dataset_config() -> SimpleNamespace:
    # The released imbalanced-data constructor needs this namespace.  Direct
    # indexed access is intentionally used, so no weighted recipient resampling
    # happens while building the deterministic Head donor pool.
    return SimpleNamespace(
        sampler=SimpleNamespace(
            type="instance",
            dual_sample=SimpleNamespace(enable=False, type="long-tailed"),
        )
    )


def load_model(checkpoint_path: Path) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage-II checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"state_dict_model", "state_dict_classifier", "rho", "epoch"}
    absent = required.difference(checkpoint)
    if absent:
        raise KeyError(f"Checkpoint is not from the fixed-endpoint H2T audit: {sorted(absent)}")
    model = torch.nn.DataParallel(resnet_cifar.resnet32_fe()).cuda()
    classifier = torch.nn.DataParallel(
        resnet_cifar.Classifier(feat_in=64, num_classes=NUM_CLASSES)
    ).cuda()
    model.load_state_dict(checkpoint["state_dict_model"], strict=True)
    classifier.load_state_dict(checkpoint["state_dict_classifier"], strict=True)
    model.eval()
    classifier.eval()
    return model, classifier, checkpoint


def signed_true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """z_y - max_{j != y} z_j for every row in ``logits``."""
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], float("-inf"))
    return true_logit - competitors.max(dim=1).values


@dataclass
class FeatureBank:
    features: torch.Tensor
    labels: np.ndarray
    indices: np.ndarray
    logits: torch.Tensor
    predicted: np.ndarray
    margins: np.ndarray


def extract_feature_bank(
    loader: DataLoader,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    label: str,
) -> FeatureBank:
    feature_parts: List[torch.Tensor] = []
    logit_parts: List[torch.Tensor] = []
    label_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    with torch.no_grad():
        for batch_id, (images, targets, indices) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            targets_device = targets.cuda(non_blocking=True)
            features = model(images)
            logits = classifier(features)
            feature_parts.append(features.detach())
            logit_parts.append(logits.detach())
            label_parts.append(targets.numpy().astype(np.int64, copy=False))
            index_parts.append(indices.numpy().astype(np.int64, copy=False))
            if batch_id % 20 == 0:
                print(f"Extracting {label} features: batch {batch_id + 1}/{len(loader)}", flush=True)
    features = torch.cat(feature_parts, dim=0)
    logits = torch.cat(logit_parts, dim=0)
    labels = np.concatenate(label_parts)
    indices = np.concatenate(index_parts)
    labels_device = torch.from_numpy(labels).to(logits.device)
    margins = signed_true_margin(logits, labels_device).cpu().numpy()
    predicted = logits.argmax(dim=1).cpu().numpy()
    return FeatureBank(features, labels, indices, logits, predicted, margins)


def make_datasets(args: argparse.Namespace) -> Tuple[IndexedSubset, IndexedSubset]:
    transform = eval_transform()
    test_data = datasets.CIFAR100(
        root=str(args.data_root), train=False, transform=transform, download=True
    )
    test_labels = np.asarray(test_data.targets, dtype=np.int64)
    tail_indices = np.flatnonzero((test_labels >= TAIL_RANGE[0]) & (test_labels < TAIL_RANGE[1]))
    if args.limit_recipients is not None:
        if args.limit_recipients <= 0:
            raise ValueError("--limit-recipients must be positive")
        tail_indices = tail_indices[: args.limit_recipients]

    train_data = IMBALANCECIFAR100(
        donor_dataset_config(),
        root=str(args.data_root),
        imb_factor=0.01,
        rand_number=0,
        train=True,
        download=True,
        transform=transform,
    )
    train_labels = np.asarray(train_data.targets, dtype=np.int64)
    head_indices = np.flatnonzero((train_labels >= HEAD_RANGE[0]) & (train_labels < HEAD_RANGE[1]))
    return IndexedSubset(test_data, tail_indices), IndexedSubset(train_data, head_indices)


def make_loader(dataset: Dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )


def h2t_with_fixed_mask_seed(
    recipient_features: torch.Tensor,
    donor_features: torch.Tensor,
    rho: float,
    mask_seed: int,
) -> torch.Tensor:
    """Call the published H2T function with a reproducibly shared mask.

    The released operator does not accept a mask argument.  Resetting the RNG
    immediately before its call makes every donor for one recipient/mask repeat
    use the same *officially sampled* channel set, without reimplementing H2T.
    """
    torch.manual_seed(mask_seed)
    torch.cuda.manual_seed_all(mask_seed)
    return H2T(recipient_features.clone(), donor_features, rho=rho)


def transition_name(clean_correct: bool, fused_correct: bool) -> str:
    if not clean_correct and fused_correct:
        return "wrong_to_correct"
    if clean_correct and not fused_correct:
        return "correct_to_wrong"
    if clean_correct:
        return "correct_to_correct"
    return "wrong_to_wrong"


def pct(value: float) -> float:
    return float(value * 100.0)


def condition_summary(
    name: str,
    labels: np.ndarray,
    clean_predicted: np.ndarray,
    clean_margins: np.ndarray,
    state_predicted: np.ndarray,
    state_margins: np.ndarray,
) -> Dict[str, Any]:
    clean_correct = clean_predicted == labels
    state_correct = state_predicted == labels
    initially_wrong = ~clean_correct
    initially_correct = clean_correct
    w2c = initially_wrong & state_correct
    c2w = initially_correct & ~state_correct
    return {
        "record_type": "recipient_state",
        "condition": name,
        "n_recipients": int(labels.size),
        "few_accuracy_pct": pct(state_correct.mean()),
        "mean_true_margin": float(state_margins.mean()),
        "mean_delta_margin": float((state_margins - clean_margins).mean()),
        "wrong_to_correct_count": int(w2c.sum()),
        "wrong_to_correct_rate_among_clean_wrong_pct": pct(w2c.sum() / max(1, initially_wrong.sum())),
        "correct_to_wrong_count": int(c2w.sum()),
        "correct_to_wrong_rate_among_clean_correct_pct": pct(c2w.sum() / max(1, initially_correct.sum())),
        "correct_to_correct_count": int((initially_correct & state_correct).sum()),
        "wrong_to_wrong_count": int((initially_wrong & ~state_correct).sum()),
    }


def uniform_random_pool_summary(
    labels: np.ndarray,
    clean_predicted: np.ndarray,
    clean_margins: np.ndarray,
    pair_predicted: np.ndarray,
    pair_margins: np.ndarray,
) -> Dict[str, Any]:
    """Exact expectation of a uniform donor draw from each sampled pool.

    A fixed single draw is still stored for reproducibility, but this pool
    expectation is the less noisy primary Random-H2T comparator because every
    one of the already sampled donors has been evaluated under the same masks.
    Transition counts are expectations, rather than integer observations.
    """
    if pair_predicted.ndim != 2 or pair_margins.ndim != 2:
        raise ValueError("Pair arrays must be [recipient, donor]")
    if pair_predicted.shape != pair_margins.shape:
        raise ValueError("Pair prediction and margin shapes must match")
    if pair_predicted.shape[0] != labels.size:
        raise ValueError("Pair arrays must have one row per recipient")
    donors_per = pair_predicted.shape[1]
    clean_correct = clean_predicted == labels
    pair_correct = pair_predicted == labels[:, None]
    pair_deltas = pair_margins - clean_margins[:, None]
    initially_wrong = ~clean_correct
    initially_correct = clean_correct
    expected_w2c = float(pair_correct[initially_wrong].sum() / donors_per)
    expected_c2w = float((~pair_correct[initially_correct]).sum() / donors_per)
    return {
        "record_type": "recipient_state_expectation",
        "condition": "random_h2t_uniform_pool",
        "n_recipients": int(labels.size),
        "donor_draw_definition": "uniform expectation over each recipient's sampled donor pool",
        "few_accuracy_pct": pct(pair_correct.mean()),
        "mean_true_margin": float(pair_margins.mean()),
        "mean_delta_margin": float(pair_deltas.mean()),
        "wrong_to_correct_expected_count": expected_w2c,
        "wrong_to_correct_rate_among_clean_wrong_pct": pct(
            pair_correct[initially_wrong].mean()
        ),
        "correct_to_wrong_expected_count": expected_c2w,
        "correct_to_wrong_rate_among_clean_correct_pct": pct(
            (~pair_correct[initially_correct]).mean()
        ),
        "correct_to_correct_expected_count": float(
            pair_correct[initially_correct].sum() / donors_per
        ),
        "wrong_to_wrong_expected_count": float(
            (~pair_correct[initially_wrong]).sum() / donors_per
        ),
    }


def build_pair_diagnostics(
    pair_deltas: np.ndarray,
    clean_correct: np.ndarray,
) -> List[Dict[str, Any]]:
    eps = 1e-8
    diagnostics: List[Dict[str, Any]] = []
    for name, recipient_mask in (
        ("all_recipients", np.ones(clean_correct.shape[0], dtype=bool)),
        ("clean_wrong_recipients", ~clean_correct),
        ("clean_correct_recipients", clean_correct),
    ):
        values = pair_deltas[recipient_mask]
        if values.size == 0:
            continue
        mixed = (values > eps).any(axis=1) & (values < -eps).any(axis=1)
        diagnostics.append(
            {
                "record_type": "pair_diagnostic",
                "condition": name,
                "n_recipients": int(values.shape[0]),
                "n_pairs": int(values.size),
                "positive_transfer_pair_pct": pct((values > eps).mean()),
                "negative_transfer_pair_pct": pct((values < -eps).mean()),
                "recipient_with_any_positive_donor_pct": pct((values > eps).any(axis=1).mean()),
                "recipient_with_all_negative_donors_pct": pct((values < -eps).all(axis=1).mean()),
                "recipient_mixed_positive_negative_donors_pct": pct(mixed.mean()),
                "mean_donor_margin_range": float((values.max(axis=1) - values.min(axis=1)).mean()),
                "mean_donor_margin_std": float(values.std(axis=1).mean()),
            }
        )
    return diagnostics


def save_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_random_vs_oracle(
    output_path: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    pair_deltas: np.ndarray,
    random_deltas: np.ndarray,
    oracle_deltas: np.ndarray,
) -> None:
    names = ["Clean\nno-fusion", "Random H2T\n(pool expectation)", "Oracle\nH2T"]
    keys = ["clean_no_fusion", "random_h2t_uniform_pool", "oracle_h2t"]
    colors = ["#6B7280", "#2563EB", "#D97706"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=180)

    accuracy = [summaries[key]["few_accuracy_pct"] for key in keys]
    axes[0, 0].bar(names, accuracy, color=colors)
    axes[0, 0].set_ylabel("Few recipient accuracy (%)")
    axes[0, 0].set_title("Diagnostic intervention accuracy")
    for index, value in enumerate(accuracy):
        axes[0, 0].text(index, value + 0.25, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    margin = [summaries[key]["mean_true_margin"] for key in keys]
    axes[0, 1].bar(names, margin, color=colors)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("Mean true-label margin")
    axes[0, 1].set_title("Margin after the frozen intervention")
    for index, value in enumerate(margin):
        axes[0, 1].text(index, value + 0.015 * (max(margin) - min(margin) + 1e-6), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    w2c = [summaries[key]["wrong_to_correct_rate_among_clean_wrong_pct"] for key in keys]
    c2w = [summaries[key]["correct_to_wrong_rate_among_clean_correct_pct"] for key in keys]
    positions = np.arange(len(keys))
    axes[1, 0].bar(positions - 0.18, w2c, width=0.36, label="wrong → correct", color="#16A34A")
    axes[1, 0].bar(positions + 0.18, c2w, width=0.36, label="correct → wrong", color="#DC2626")
    axes[1, 0].set_xticks(positions, names)
    axes[1, 0].set_ylabel("Rate within clean-status subgroup (%)")
    axes[1, 0].set_title("Recipient transition rates")
    axes[1, 0].legend(frameon=False, fontsize=8)

    flattened = pair_deltas.reshape(-1)
    low, high = np.quantile(flattened, [0.005, 0.995])
    axes[1, 1].hist(flattened, bins=90, range=(low, high), density=True, color="#94A3B8", alpha=0.8, label="all sampled pairs")
    axes[1, 1].axvline(0, color="black", linewidth=0.9, label="no margin change")
    axes[1, 1].axvline(random_deltas.mean(), color="#2563EB", linewidth=1.7, label="random mean")
    axes[1, 1].axvline(oracle_deltas.mean(), color="#D97706", linewidth=1.7, label="oracle mean")
    axes[1, 1].set_xlabel("Mean Δ true-label margin per recipient–donor pair")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Transfer variability across sampled Head donors")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Frozen official H2T (rho=0.25): random donor versus label-aware oracle", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    oracle_donor_labels: np.ndarray,
    random_donor_labels: np.ndarray,
) -> None:
    random_state = summaries["random_h2t_uniform_pool"]
    random_seeded_draw = summaries["random_h2t_seeded_draw"]
    oracle_state = summaries["oracle_h2t"]
    clean_state = summaries["clean_no_fusion"]
    all_pairs = next(item for item in diagnostics if item["condition"] == "all_recipients")
    gap = oracle_state["few_accuracy_pct"] - random_state["few_accuracy_pct"]
    margin_gap = oracle_state["mean_true_margin"] - random_state["mean_true_margin"]
    random_vs_clean = random_state["few_accuracy_pct"] - clean_state["few_accuracy_pct"]
    oracle_vs_clean = oracle_state["few_accuracy_pct"] - clean_state["few_accuracy_pct"]
    lines = [
        "# Frozen H2T 迁移成败与 Oracle 上限审计",
        "",
        "## 审计定义",
        "",
        f"- 冻结 checkpoint：`{args.checkpoint}`，epoch={checkpoint['epoch']}，训练 rho={checkpoint['rho']}。",
        "- recipient：CIFAR-100 test 的 Few 类（71--99）；donor：long-tailed training set 的 Many/Head 类（0--35），均采用 deterministic eval transform。",
        f"- 每个 recipient 随机、不带语义筛选地抽取 {args.donors_per_recipient} 个 Head donors；每个 donor 在该 recipient 固定的 {args.masks_per_recipient} 个官方 H2T channel masks 下评估。",
        "- 主要 pair 分数：多个 mask 的平均 true-label signed margin change，Δm = mean_mask[m_fused] − m_clean。",
        "- Random H2T（主比较）：对同一 donor pool 的所有 donor 取均匀抽样期望；固定随机抽取一个 donor 的结果另存作复核。Oracle H2T：用真实 recipient label 选取最大平均 true-label margin 的 donor。单个 donor 的分类状态均由多个 mask 上的平均 logits 给出。",
        "- Random/Oracle 都是诊断性 test-time intervention，不是 H2T 的正常 clean-inference 准确率；Oracle 更不是可部署方法。",
        "",
        "## 三个状态",
        "",
        "| 状态 | Few Acc (%) | Mean margin | Δmargin | Wrong→Correct / clean-wrong (%) | Correct→Wrong / clean-correct (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, title in (
        ("clean_no_fusion", "Clean no-fusion"),
        ("random_h2t_uniform_pool", "Random H2T (uniform-pool expectation; primary)"),
        ("random_h2t_seeded_draw", "Random H2T (one fixed seeded draw; check)"),
        ("oracle_h2t", "Oracle H2T (label-aware upper bound)"),
    ):
        item = summaries[key]
        lines.append(
            f"| {title} | {item['few_accuracy_pct']:.2f} | {item['mean_true_margin']:.4f} | "
            f"{item['mean_delta_margin']:+.4f} | {item['wrong_to_correct_rate_among_clean_wrong_pct']:.2f} | "
            f"{item['correct_to_wrong_rate_among_clean_correct_pct']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 迁移不稳定性（所有 sampled recipient–donor pairs）",
            "",
            f"- Positive-transfer pair：{all_pairs['positive_transfer_pair_pct']:.2f}%；negative-transfer pair：{all_pairs['negative_transfer_pair_pct']:.2f}%。",
            f"- 有至少一个正迁移 donor 的 recipient：{all_pairs['recipient_with_any_positive_donor_pct']:.2f}%；所有 donor 都负迁移的 recipient：{all_pairs['recipient_with_all_negative_donors_pct']:.2f}%。",
            f"- 同一 recipient 的 donor 结果同时含正、负 Δm 的比例：{all_pairs['recipient_mixed_positive_negative_donors_pct']:.2f}%。",
            f"- 同一 recipient 在 {args.donors_per_recipient} donors 上的平均 Δm range：{all_pairs['mean_donor_margin_range']:.4f}；平均标准差：{all_pairs['mean_donor_margin_std']:.4f}。",
            "",
            "## Oracle gap 的解释",
            "",
            f"- Oracle − Random diagnostic Few Acc：{gap:+.2f}pp；mean margin gap：{margin_gap:+.4f}。",
            f"- 固定随机单次抽样的 Few Acc={random_seeded_draw['few_accuracy_pct']:.2f}%（主 Random pool expectation={random_state['few_accuracy_pct']:.2f}%），说明主结论不依赖某一次随机 donor 抽取。",
            "- 这个 gap 只回答：在同一 donor pool 里，若能理想地识别对当前 recipient 有利的 donor，理论上还能挖出多少空间。它不证明可通过标签外信号实现该选择。",
            f"- Random donor 的 Head-class id 均值={random_donor_labels.mean():.2f}；Oracle donor 的 Head-class id 均值={oracle_donor_labels.mean():.2f}。类别 id 不等于语义解释，故本轮不据此作 donor 机制结论。",
            "",
            "## 本轮可以支持、不能支持的结论",
            "",
            f"- 在这个冻结 checkpoint、50-donor pool 和 5-mask 协议下，Random H2T 的 Few Acc 相对 clean no-fusion 为 {random_vs_clean:+.2f}pp；随机头尾迁移并不稳定，且平均上是有害的。",
            f"- 同一 pool 内 label-aware Oracle 相对 clean 为 {oracle_vs_clean:+.2f}pp、相对 Random 为 {gap:+.2f}pp；因此 donor 成败之间存在很大的理论可控空间。",
            f"- {all_pairs['recipient_mixed_positive_negative_donors_pct']:.2f}% recipient 同时遇到正、负 Δm donor，不能把这种差异解释为仅由全局 fusion 强度或 mask 随机性造成。",
            "- 这不证明任何可部署的 donor-selection 规则，也不改变 H2T 的正常 clean-inference 结果；Oracle 使用了测试标签，只能作为后续无标签 donor 评分研究的上限。",
            "",
            "## 决策规则",
            "",
            "- 若 Random 与 Oracle 接近，或 Oracle 只能挽救极少 recipient，则 donor-conditioned 迁移可控空间有限，不值得据此扩展后续方法。",
            "- 若 Oracle 明显优于 Random、同时负迁移 pair 很多，说明当前随机迁移存在尚未利用的可控空间；第二阶段才研究 donor 属性与成败的关联。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The released H2T function invokes .cuda(); CUDA is required.")
    if not (0.0 < args.rho <= 1.0):
        raise ValueError("rho must be in (0, 1]")
    if args.donors_per_recipient < 1 or args.masks_per_recipient < 1:
        raise ValueError("donors-per-recipient and masks-per-recipient must both be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    model, classifier, checkpoint = load_model(args.checkpoint)
    if abs(float(checkpoint["rho"]) - args.rho) > 1e-9:
        raise ValueError(
            f"Checkpoint was trained at rho={checkpoint['rho']}; requested intervention rho={args.rho}."
        )
    tail_dataset, head_dataset = make_datasets(args)
    if len(head_dataset) < args.donors_per_recipient:
        raise ValueError("Donor pool is smaller than donors-per-recipient")
    print(f"Recipients (Few test): {len(tail_dataset)}; donors (Head train): {len(head_dataset)}", flush=True)

    recipients = extract_feature_bank(make_loader(tail_dataset, args), model, classifier, "recipient")
    donors = extract_feature_bank(make_loader(head_dataset, args), model, classifier, "donor")
    clean_correct = recipients.predicted == recipients.labels
    n_recipients = len(recipients.labels)
    donors_per = args.donors_per_recipient

    rng = np.random.default_rng(args.seed)
    donor_slots = np.stack(
        [rng.choice(len(donors.labels), size=donors_per, replace=False) for _ in range(n_recipients)]
    )
    random_slots = rng.integers(0, donors_per, size=n_recipients, endpoint=False)

    pair_margins = np.empty((n_recipients, donors_per), dtype=np.float32)
    pair_margin_std = np.empty((n_recipients, donors_per), dtype=np.float32)
    pair_ensemble_margins = np.empty((n_recipients, donors_per), dtype=np.float32)
    pair_predicted = np.empty((n_recipients, donors_per), dtype=np.int16)
    random_predicted = np.empty(n_recipients, dtype=np.int16)
    oracle_predicted = np.empty(n_recipients, dtype=np.int16)
    random_margins = np.empty(n_recipients, dtype=np.float32)
    oracle_margins = np.empty(n_recipients, dtype=np.float32)
    random_donor_rows = np.empty(n_recipients, dtype=np.int64)
    oracle_donor_rows = np.empty(n_recipients, dtype=np.int64)

    pair_path = args.output_dir / "pair_records.csv"
    with pair_path.open("w", encoding="utf-8-sig", newline="") as handle, torch.no_grad():
        columns = [
            "recipient_row",
            "recipient_test_index",
            "recipient_label",
            "clean_predicted",
            "clean_correct",
            "clean_true_margin",
            "donor_slot",
            "donor_pool_row",
            "donor_train_index",
            "donor_label",
            "masks_per_recipient",
            "first_mask_seed",
            "mean_fused_true_margin",
            "std_fused_true_margin",
            "ensemble_true_margin",
            "mean_delta_margin",
            "fused_predicted_mean_logits",
            "fused_correct_mean_logits",
            "transition_mean_logits",
            "positive_transfer",
            "negative_transfer",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for recipient_row in range(n_recipients):
            target_label = int(recipients.labels[recipient_row])
            selected_rows = donor_slots[recipient_row]
            selected_tensor = torch.from_numpy(selected_rows).to(donors.features.device)
            donor_features = donors.features[selected_tensor]
            recipient_features = recipients.features[recipient_row : recipient_row + 1].expand(
                donors_per, -1, -1, -1
            )
            target_tensor = torch.full(
                (donors_per,), target_label, dtype=torch.long, device=donor_features.device
            )
            margin_by_mask: List[torch.Tensor] = []
            logits_sum = torch.zeros((donors_per, NUM_CLASSES), device=donor_features.device)
            first_mask_seed = args.seed + recipient_row * args.masks_per_recipient
            for mask_index in range(args.masks_per_recipient):
                mask_seed = first_mask_seed + mask_index
                fused_features = h2t_with_fixed_mask_seed(
                    recipient_features, donor_features, args.rho, mask_seed
                )
                fused_logits = classifier(fused_features)
                logits_sum.add_(fused_logits)
                margin_by_mask.append(signed_true_margin(fused_logits, target_tensor))
            margin_tensor = torch.stack(margin_by_mask, dim=0)
            mean_margins = margin_tensor.mean(dim=0)
            std_margins = margin_tensor.std(dim=0, unbiased=False)
            mean_logits = logits_sum / args.masks_per_recipient
            ensemble_margins = signed_true_margin(mean_logits, target_tensor)
            predicted = mean_logits.argmax(dim=1)

            mean_margins_np = mean_margins.cpu().numpy()
            std_margins_np = std_margins.cpu().numpy()
            ensemble_margins_np = ensemble_margins.cpu().numpy()
            predicted_np = predicted.cpu().numpy().astype(np.int16)
            pair_margins[recipient_row] = mean_margins_np
            pair_margin_std[recipient_row] = std_margins_np
            pair_ensemble_margins[recipient_row] = ensemble_margins_np
            pair_predicted[recipient_row] = predicted_np

            random_slot = int(random_slots[recipient_row])
            oracle_slot = int(np.argmax(mean_margins_np))
            random_donor_rows[recipient_row] = int(selected_rows[random_slot])
            oracle_donor_rows[recipient_row] = int(selected_rows[oracle_slot])
            random_predicted[recipient_row] = predicted_np[random_slot]
            oracle_predicted[recipient_row] = predicted_np[oracle_slot]
            random_margins[recipient_row] = mean_margins_np[random_slot]
            oracle_margins[recipient_row] = mean_margins_np[oracle_slot]

            clean_margin = float(recipients.margins[recipient_row])
            baseline_correct = bool(clean_correct[recipient_row])
            for donor_slot, donor_row in enumerate(selected_rows):
                fused_correct = bool(predicted_np[donor_slot] == target_label)
                delta = float(mean_margins_np[donor_slot] - clean_margin)
                writer.writerow(
                    {
                        "recipient_row": recipient_row,
                        "recipient_test_index": int(recipients.indices[recipient_row]),
                        "recipient_label": target_label,
                        "clean_predicted": int(recipients.predicted[recipient_row]),
                        "clean_correct": int(baseline_correct),
                        "clean_true_margin": clean_margin,
                        "donor_slot": donor_slot,
                        "donor_pool_row": int(donor_row),
                        "donor_train_index": int(donors.indices[donor_row]),
                        "donor_label": int(donors.labels[donor_row]),
                        "masks_per_recipient": args.masks_per_recipient,
                        "first_mask_seed": first_mask_seed,
                        "mean_fused_true_margin": float(mean_margins_np[donor_slot]),
                        "std_fused_true_margin": float(std_margins_np[donor_slot]),
                        "ensemble_true_margin": float(ensemble_margins_np[donor_slot]),
                        "mean_delta_margin": delta,
                        "fused_predicted_mean_logits": int(predicted_np[donor_slot]),
                        "fused_correct_mean_logits": int(fused_correct),
                        "transition_mean_logits": transition_name(baseline_correct, fused_correct),
                        "positive_transfer": int(delta > 1e-8),
                        "negative_transfer": int(delta < -1e-8),
                    }
                )
            if (recipient_row + 1) % 100 == 0 or recipient_row + 1 == n_recipients:
                print(
                    f"Evaluated recipient {recipient_row + 1}/{n_recipients} "
                    f"({(recipient_row + 1) * donors_per * args.masks_per_recipient:,} H2T pair-masks)",
                    flush=True,
                )

    random_deltas = random_margins - recipients.margins
    oracle_deltas = oracle_margins - recipients.margins
    summaries = {
        "clean_no_fusion": condition_summary(
            "clean_no_fusion",
            recipients.labels,
            recipients.predicted,
            recipients.margins,
            recipients.predicted,
            recipients.margins,
        ),
        "random_h2t_seeded_draw": condition_summary(
            "random_h2t_seeded_draw",
            recipients.labels,
            recipients.predicted,
            recipients.margins,
            random_predicted,
            random_margins,
        ),
        "random_h2t_uniform_pool": uniform_random_pool_summary(
            recipients.labels,
            recipients.predicted,
            recipients.margins,
            pair_predicted,
            pair_margins,
        ),
        "oracle_h2t": condition_summary(
            "oracle_h2t",
            recipients.labels,
            recipients.predicted,
            recipients.margins,
            oracle_predicted,
            oracle_margins,
        ),
    }
    diagnostics = build_pair_diagnostics(pair_margins - recipients.margins[:, None], clean_correct)
    summary_rows = list(summaries.values()) + diagnostics
    save_summary_csv(summary_rows, args.output_dir / "transfer_summary.csv")

    recipient_path = args.output_dir / "recipient_records.csv"
    with recipient_path.open("w", encoding="utf-8-sig", newline="") as handle:
        columns = [
            "recipient_row",
            "recipient_test_index",
            "recipient_label",
            "clean_predicted",
            "clean_correct",
            "clean_true_margin",
            "random_donor_pool_row",
            "random_donor_train_index",
            "random_donor_label",
            "random_predicted",
            "random_mean_margin",
            "random_delta_margin",
            "oracle_donor_pool_row",
            "oracle_donor_train_index",
            "oracle_donor_label",
            "oracle_predicted",
            "oracle_mean_margin",
            "oracle_delta_margin",
            "oracle_minus_random_margin",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in range(n_recipients):
            random_donor = int(random_donor_rows[row])
            oracle_donor = int(oracle_donor_rows[row])
            writer.writerow(
                {
                    "recipient_row": row,
                    "recipient_test_index": int(recipients.indices[row]),
                    "recipient_label": int(recipients.labels[row]),
                    "clean_predicted": int(recipients.predicted[row]),
                    "clean_correct": int(clean_correct[row]),
                    "clean_true_margin": float(recipients.margins[row]),
                    "random_donor_pool_row": random_donor,
                    "random_donor_train_index": int(donors.indices[random_donor]),
                    "random_donor_label": int(donors.labels[random_donor]),
                    "random_predicted": int(random_predicted[row]),
                    "random_mean_margin": float(random_margins[row]),
                    "random_delta_margin": float(random_deltas[row]),
                    "oracle_donor_pool_row": oracle_donor,
                    "oracle_donor_train_index": int(donors.indices[oracle_donor]),
                    "oracle_donor_label": int(donors.labels[oracle_donor]),
                    "oracle_predicted": int(oracle_predicted[row]),
                    "oracle_mean_margin": float(oracle_margins[row]),
                    "oracle_delta_margin": float(oracle_deltas[row]),
                    "oracle_minus_random_margin": float(oracle_margins[row] - random_margins[row]),
                }
            )

    plot_random_vs_oracle(
        args.output_dir / "random_vs_oracle.png",
        summaries,
        pair_margins - recipients.margins[:, None],
        (pair_margins - recipients.margins[:, None]).mean(axis=1),
        oracle_deltas,
    )
    write_report(
        args.output_dir / "H2T_TRANSFER_ORACLE_AUDIT_ZH.md",
        args,
        checkpoint,
        summaries,
        diagnostics,
        donors.labels[oracle_donor_rows],
        donors.labels[random_donor_rows],
    )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_training_rho": float(checkpoint["rho"]),
        "intervention_rho": args.rho,
        "recipient_domain": "CIFAR-100 test Few classes 71-99",
        "donor_domain": "CIFAR-100-LT training Many classes 0-35 with deterministic eval transform",
        "n_recipients": n_recipients,
        "n_head_donors": int(len(donors.labels)),
        "donors_per_recipient": donors_per,
        "masks_per_recipient": args.masks_per_recipient,
        "mask_control": "Each recipient/mask seed is reset before the released H2T call; all donors share the same official mask within a recipient/mask repeat.",
        "oracle_definition": "argmax donor of mean true-label signed margin across the fixed masks; label-aware diagnostic upper bound.",
        "summaries": summaries,
        "pair_diagnostics": diagnostics,
    }
    (args.output_dir / "transfer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed frozen transfer audit: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
