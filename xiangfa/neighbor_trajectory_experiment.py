from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from support_dynamics_experiment import (
    ANGLE_NAMES,
    DESCRIPTOR_INDEX,
    DESCRIPTOR_NAMES,
    NUM_CLASSES,
    SHAPE_NAMES,
    ArrayCIFAR100,
    ResNetCifar,
    classification_accuracy,
    cosine_learning_rate,
    eval_transform,
    extract_features,
    l2_normalize,
    make_nested_class_orders,
    save_json,
    set_seed,
    train_transform,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Select similar classes in an independent balanced CIFAR-100 "
            "reference space, then test whether they have similar controlled "
            "500->200->50->20->5 support trajectories."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--support-root",
        type=Path,
        default=project_dir / "outputs" / "support_dynamics",
        help="Directory containing formal five-environment support_states.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "balanced_neighbor_trajectory",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 20, 50, 200, 500])
    parser.add_argument("--balanced-train-per-class", type=int, default=400)
    parser.add_argument("--balanced-val-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--neighbor-k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    result = spearmanr(x, y)
    return float(result.statistic) if np.isfinite(result.statistic) else 0.0


def safe_wilcoxon(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0.0):
        return {"statistic": 0.0, "pvalue": 1.0}
    result = wilcoxon(values)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def balanced_reference_split(
    targets: np.ndarray,
    train_per_class: int,
    val_per_class: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_per_class + val_per_class > 500:
        raise ValueError("Balanced train + validation samples cannot exceed 500/class.")
    orders = make_nested_class_orders(targets, subset_seed=split_seed)
    train_indices = np.concatenate(
        [order[:train_per_class] for order in orders]
    ).astype(np.int64)
    val_indices = np.concatenate(
        [
            order[train_per_class : train_per_class + val_per_class]
            for order in orders
        ]
    ).astype(np.int64)
    return train_indices, val_indices


def train_balanced_reference(
    seed: int,
    raw_train,
    train_indices: np.ndarray,
    val_loader: DataLoader,
    seed_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    checkpoint_path = seed_dir / "balanced_reference_final.pth"
    if args.skip_train and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("epochs") == args.epochs
            and checkpoint.get("train_per_class")
            == args.balanced_train_per_class
            and checkpoint.get("val_per_class") == args.balanced_val_per_class
        ):
            print(f"[seed {seed}] reusing balanced reference checkpoint")
            return checkpoint_path
        raise ValueError(f"Incompatible balanced checkpoint: {checkpoint_path}")

    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    dataset = ArrayCIFAR100(
        raw_train.data,
        train_targets,
        train_indices,
        train_transform(),
    )
    set_seed(seed)
    model = ResNetCifar(depth=32).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        lr = cosine_learning_rate(args.lr, epoch, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and not args.no_amp,
            ):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += int(labels.numel())
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print_every = max(1, args.epochs // 5)
        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(
                f"[seed {seed} balanced] epoch {epoch + 1:03d}/{args.epochs}, "
                f"loss={row['loss']:.4f}, "
                f"train={row['train_accuracy'] * 100:.2f}%, "
                f"elapsed={row['elapsed_seconds']:.1f}s"
            )
    val_accuracy = classification_accuracy(model, val_loader, device)
    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "epochs": args.epochs,
            "train_per_class": args.balanced_train_per_class,
            "val_per_class": args.balanced_val_per_class,
            "val_accuracy": val_accuracy,
            "history": history,
        },
        checkpoint_path,
    )
    save_json(history, seed_dir / "balanced_history.json")
    print(
        f"[seed {seed} balanced] validation accuracy="
        f"{val_accuracy * 100:.2f}%"
    )
    return checkpoint_path


def load_model(path: Path, device: torch.device) -> ResNetCifar:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNetCifar(depth=32)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def balanced_similarity_scores(
    features: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    knn_k: int,
) -> dict[str, np.ndarray]:
    normalized = l2_normalize(features.astype(np.float64))
    prototypes = np.zeros((NUM_CLASSES, normalized.shape[1]), dtype=np.float64)
    for class_id in range(NUM_CLASSES):
        current = normalized[labels == class_id]
        prototypes[class_id] = l2_normalize(
            current.mean(axis=0, keepdims=True)
        )[0]
    prototype = prototypes @ prototypes.T
    np.fill_diagonal(prototype, -np.inf)

    predictions = logits.argmax(axis=1)
    confusion_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    np.add.at(confusion_counts, (labels, predictions), 1.0)
    class_sizes = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    confusion_rate = confusion_counts / np.maximum(class_sizes[:, None], 1.0)
    np.fill_diagonal(confusion_rate, 0.0)
    confusion_symmetric = 0.5 * (confusion_rate + confusion_rate.T)
    np.fill_diagonal(confusion_symmetric, -np.inf)

    k = min(knn_k, len(normalized) - 1)
    nearest = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nearest.fit(normalized)
    neighbor_indices = nearest.kneighbors(
        normalized, return_distance=False
    )[:, 1:]
    directed_knn = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for sample_index, neighbors in enumerate(neighbor_indices):
        source_class = int(labels[sample_index])
        np.add.at(directed_knn[source_class], labels[neighbors], 1.0)
    directed_knn = directed_knn / np.maximum(
        class_sizes[:, None] * float(k), 1.0
    )
    knn = 0.5 * (directed_knn + directed_knn.T)
    np.fill_diagonal(knn, -np.inf)
    return {
        "prototype": prototype,
        "confusion": confusion_symmetric,
        "knn": knn,
    }


def reciprocal_rank_fusion(
    score_matrices: list[np.ndarray], constant: float = 60.0
) -> np.ndarray:
    fused = np.zeros_like(score_matrices[0], dtype=np.float64)
    for target_class in range(NUM_CLASSES):
        for matrix in score_matrices:
            values = matrix[target_class].copy()
            values[target_class] = -np.inf
            order = np.argsort(-values)
            ranks = np.empty(NUM_CLASSES, dtype=np.int64)
            ranks[order] = np.arange(1, NUM_CLASSES + 1)
            fused[target_class] += 1.0 / (constant + ranks)
        fused[target_class, target_class] = -np.inf
    return fused


def read_fine_to_coarse(data_root: Path) -> np.ndarray:
    train_pickle = data_root / "cifar-100-python" / "train"
    with train_pickle.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    fine = np.asarray(
        payload.get("fine_labels", payload.get(b"fine_labels")),
        dtype=np.int64,
    )
    coarse = np.asarray(
        payload.get("coarse_labels", payload.get(b"coarse_labels")),
        dtype=np.int64,
    )
    mapping = np.zeros(NUM_CLASSES, dtype=np.int64)
    for class_id in range(NUM_CLASSES):
        mapping[class_id] = int(
            np.bincount(coarse[fine == class_id]).argmax()
        )
    return mapping


def load_support_states(
    path: Path, levels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing five-frequency support states: {path}. "
            "Run support_dynamics_experiment.py for this seed first."
        )
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    descriptors = np.zeros(
        (NUM_CLASSES, len(levels), len(DESCRIPTOR_NAMES)),
        dtype=np.float64,
    )
    accuracies = np.zeros((NUM_CLASSES, len(levels)), dtype=np.float64)
    level_position = {int(level): i for i, level in enumerate(levels)}
    seen = np.zeros((NUM_CLASSES, len(levels)), dtype=bool)
    for row in rows:
        class_id = int(row["class_id"])
        count = int(row["count"])
        if count not in level_position:
            continue
        position = level_position[count]
        descriptors[class_id, position] = [
            float(row[name]) for name in DESCRIPTOR_NAMES
        ]
        accuracies[class_id, position] = float(row["accuracy"])
        seen[class_id, position] = True
    if not seen.all():
        raise ValueError("Support-state CSV does not contain every class-level pair.")
    return descriptors, accuracies


def standardized_trajectory_similarity(
    support_states: np.ndarray,
    descriptor_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(
        [DESCRIPTOR_INDEX[name] for name in descriptor_names],
        dtype=np.int64,
    )
    # levels are stored low->high; reverse so the trajectory is head->tail.
    ordered = support_states[:, ::-1, :][:, :, indices]
    increments = np.diff(ordered, axis=1)
    center = increments.mean(axis=0, keepdims=True)
    scale = np.maximum(increments.std(axis=0, ddof=1, keepdims=True), 1e-8)
    standardized = (increments - center) / scale
    vectors = standardized.reshape(NUM_CLASSES, -1)
    normalized = l2_normalize(vectors)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, np.nan)
    return vectors, similarity


def evaluate_neighbors(
    scores: dict[str, np.ndarray],
    trajectory_similarity: np.ndarray,
    fine_to_coarse: np.ndarray,
    neighbor_ks: list[int],
) -> tuple[dict, list[dict]]:
    off_diagonal = ~np.eye(NUM_CLASSES, dtype=bool)
    random_expectation = np.nanmean(trajectory_similarity, axis=1)
    summary = {}
    selected_rows = []
    for method, score_matrix in scores.items():
        method_result = {
            "pairwise_spearman": safe_spearman(
                score_matrix[off_diagonal],
                trajectory_similarity[off_diagonal],
            ),
            "topk": {},
        }
        for k in neighbor_ks:
            topk_similarity = np.zeros(NUM_CLASSES, dtype=np.float64)
            same_coarse_rate = np.zeros(NUM_CLASSES, dtype=np.float64)
            for target_class in range(NUM_CLASSES):
                values = score_matrix[target_class].copy()
                values[target_class] = -np.inf
                selected = np.argsort(-values)[:k]
                topk_similarity[target_class] = np.nanmean(
                    trajectory_similarity[target_class, selected]
                )
                same_coarse_rate[target_class] = np.mean(
                    fine_to_coarse[selected]
                    == fine_to_coarse[target_class]
                )
                for rank, neighbor in enumerate(selected, 1):
                    selected_rows.append(
                        {
                            "method": method,
                            "k": k,
                            "target_class": target_class,
                            "neighbor_rank": rank,
                            "neighbor_class": int(neighbor),
                            "balanced_similarity_score": float(
                                score_matrix[target_class, neighbor]
                            ),
                            "trajectory_similarity": float(
                                trajectory_similarity[target_class, neighbor]
                            ),
                            "same_coarse": int(
                                fine_to_coarse[target_class]
                                == fine_to_coarse[neighbor]
                            ),
                        }
                    )
            delta = topk_similarity - random_expectation
            method_result["topk"][str(k)] = {
                "mean_trajectory_similarity": float(
                    topk_similarity.mean()
                ),
                "random_expected_similarity": float(
                    random_expectation.mean()
                ),
                "delta_vs_random": float(delta.mean()),
                "better_than_random_fraction": float(np.mean(delta > 0.0)),
                "wilcoxon": safe_wilcoxon(delta),
                "same_coarse_hit_rate": float(same_coarse_rate.mean()),
            }
        summary[method] = method_result

    same_coarse_similarity = np.zeros(NUM_CLASSES, dtype=np.float64)
    for target_class in range(NUM_CLASSES):
        siblings = np.flatnonzero(
            (fine_to_coarse == fine_to_coarse[target_class])
            & (np.arange(NUM_CLASSES) != target_class)
        )
        same_coarse_similarity[target_class] = np.nanmean(
            trajectory_similarity[target_class, siblings]
        )
    semantic_delta = same_coarse_similarity - random_expectation
    summary["same_coarse_oracle"] = {
        "mean_trajectory_similarity": float(
            same_coarse_similarity.mean()
        ),
        "random_expected_similarity": float(random_expectation.mean()),
        "delta_vs_random": float(semantic_delta.mean()),
        "better_than_random_fraction": float(
            np.mean(semantic_delta > 0.0)
        ),
        "wilcoxon": safe_wilcoxon(semantic_delta),
    }
    return summary, selected_rows


def build_report(seed_summaries: list[dict], output_dir: Path) -> None:
    primary_deltas = []
    primary_fractions = []
    primary_pvalues = []
    lines = [
        "# 平衡空间相似类与头尾支撑轨迹一致性报告",
        "",
        "相似类只在独立的平衡训练/验证划分中确定；多频率轨迹来自另一组受控长尾环境的固定平衡测试集。",
        "",
    ]
    for summary in seed_summaries:
        seed = summary["seed"]
        reference = summary["balanced_reference"]
        neighbors = summary["internal_trajectory"]["neighbors"]
        lines.extend(
            [
                f"## Seed {seed}",
                "",
                f"- 平衡参考模型验证准确率：**{reference['validation_accuracy'] * 100:.2f}%**。",
                "- 主轨迹：角分位数 + 切空间协方差结构的 500→200→50→20→5 相邻增量，不包含筛选相似类所用的混淆/KNN关系。",
                "",
                "| 平衡空间筛选方法 | Top-3轨迹相似度 | 相对随机 | 改善类别 | p值 | 同超类命中率 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in ["prototype", "confusion", "knn", "fusion"]:
            row = neighbors[method]["topk"]["3"]
            lines.append(
                f"| {method} | {row['mean_trajectory_similarity']:.4f} | "
                f"{row['delta_vs_random']:+.4f} | "
                f"{row['better_than_random_fraction'] * 100:.1f}% | "
                f"{row['wilcoxon']['pvalue']:.3g} | "
                f"{row['same_coarse_hit_rate'] * 100:.1f}% |"
            )
        oracle = neighbors["same_coarse_oracle"]
        lines.append(
            f"| same_coarse_oracle | "
            f"{oracle['mean_trajectory_similarity']:.4f} | "
            f"{oracle['delta_vs_random']:+.4f} | "
            f"{oracle['better_than_random_fraction'] * 100:.1f}% | "
            f"{oracle['wilcoxon']['pvalue']:.3g} | 100.0% |"
        )
        lines.append("")
        primary = neighbors["fusion"]["topk"]["3"]
        primary_deltas.append(primary["delta_vs_random"])
        primary_fractions.append(primary["better_than_random_fraction"])
        primary_pvalues.append(primary["wilcoxon"]["pvalue"])

    supported = (
        float(np.mean(primary_deltas)) >= 0.05
        and float(np.mean(primary_fractions)) >= 0.60
        and max(primary_pvalues) < 0.05
    )
    verdict = (
        "支持：平衡空间预先筛选的相似类具有更一致的头尾支撑轨迹。"
        if supported
        else "暂未通过：平衡空间相似类的头尾支撑轨迹未达到预设一致性门槛。"
    )
    lines.extend(
        [
            "## 预设判定",
            "",
            f"**{verdict}**",
            "",
            "- 主方法：平衡空间原型、对称混淆、局部KNN的秩融合 Top-3。",
            "- 门槛：相对随机轨迹相似度至少 +0.05；至少60%类别改善；Wilcoxon p<0.05。",
            f"- 平均相对随机：**{np.mean(primary_deltas):+.4f}**。",
            f"- 平均改善类别比例：**{np.mean(primary_fractions) * 100:.1f}%**。",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_seed(
    seed: int,
    raw_train,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    fine_to_coarse: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    val_dataset = ArrayCIFAR100(
        raw_train.data,
        train_targets,
        val_indices,
        eval_transform(),
        return_index=True,
    )
    val_loader = DataLoader(
        ArrayCIFAR100(
            raw_train.data,
            train_targets,
            val_indices,
            eval_transform(),
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    checkpoint = train_balanced_reference(
        seed,
        raw_train,
        train_indices,
        val_loader,
        seed_dir,
        args,
        device,
    )
    checkpoint_data = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    model = load_model(checkpoint, device)
    val_features, val_logits, val_labels, _ = extract_features(
        model, val_dataset, args, device
    )
    scores = balanced_similarity_scores(
        val_features, val_logits, val_labels, args.knn_k
    )
    scores["fusion"] = reciprocal_rank_fusion(
        [scores["prototype"], scores["confusion"], scores["knn"]]
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    support_csv = (
        args.support_root / f"seed_{seed}" / "support_states.csv"
    )
    support_states, accuracies = load_support_states(
        support_csv, args.levels
    )
    _, internal_similarity = standardized_trajectory_similarity(
        support_states, ANGLE_NAMES + SHAPE_NAMES
    )
    _, full_similarity = standardized_trajectory_similarity(
        support_states, DESCRIPTOR_NAMES
    )
    internal_neighbors, selected_rows = evaluate_neighbors(
        scores, internal_similarity, fine_to_coarse, args.neighbor_k
    )
    full_neighbors, _ = evaluate_neighbors(
        scores, full_similarity, fine_to_coarse, args.neighbor_k
    )
    write_csv(selected_rows, seed_dir / "selected_neighbors.csv")

    pair_rows = []
    for target_class in range(NUM_CLASSES):
        for candidate_class in range(NUM_CLASSES):
            if target_class == candidate_class:
                continue
            row = {
                "target_class": target_class,
                "candidate_class": candidate_class,
                "same_coarse": int(
                    fine_to_coarse[target_class]
                    == fine_to_coarse[candidate_class]
                ),
                "internal_trajectory_similarity": float(
                    internal_similarity[target_class, candidate_class]
                ),
                "full_trajectory_similarity": float(
                    full_similarity[target_class, candidate_class]
                ),
            }
            row.update(
                {
                    name: float(matrix[target_class, candidate_class])
                    for name, matrix in scores.items()
                }
            )
            pair_rows.append(row)
    write_csv(pair_rows, seed_dir / "all_class_pairs.csv")

    summary = {
        "seed": seed,
        "balanced_reference": {
            "validation_accuracy": float(
                checkpoint_data["val_accuracy"]
            ),
            "train_per_class": args.balanced_train_per_class,
            "validation_per_class": args.balanced_val_per_class,
        },
        "trajectory_accuracy": {
            "five_shot_mean": float(accuracies[:, 0].mean()),
            "five_hundred_shot_mean": float(accuracies[:, -1].mean()),
        },
        "internal_trajectory": {"neighbors": internal_neighbors},
        "full_trajectory": {"neighbors": full_neighbors},
    }
    save_json(summary, seed_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    args.levels = sorted(args.levels)
    args.neighbor_k = sorted(set(args.neighbor_k))
    if args.smoke:
        args.epochs = 2
        args.num_workers = 0
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), args.output_dir / "args.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    train_indices, val_indices = balanced_reference_split(
        train_targets,
        args.balanced_train_per_class,
        args.balanced_val_per_class,
        args.split_seed,
    )
    fine_to_coarse = read_fine_to_coarse(args.data_root)
    save_json(
        {
            "balanced_train_indices": train_indices,
            "balanced_validation_indices": val_indices,
            "fine_to_coarse": fine_to_coarse,
            "trajectory_levels": args.levels,
        },
        args.output_dir / "design.json",
    )

    summaries = []
    for seed in args.seeds:
        summaries.append(
            run_seed(
                seed,
                raw_train,
                train_indices,
                val_indices,
                fine_to_coarse,
                args,
                device,
            )
        )
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    build_report(summaries, args.output_dir)
    print(f"Finished. Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
