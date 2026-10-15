from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, RandomSampler

from support_dynamics_experiment import (
    DESCRIPTOR_INDEX,
    DESCRIPTOR_NAMES,
    NUM_CLASSES,
    ArrayCIFAR100,
    ResNetCifar,
    classification_accuracy,
    cosine_learning_rate,
    describe_supports,
    eval_transform,
    extract_features,
    l2_normalize,
    make_nested_class_orders,
    normalize_vector,
    orthogonal_procrustes,
    save_json,
    set_seed,
    train_transform,
    write_csv,
)


PRIMARY_NAMES = [
    "angle_q50",
    "angle_q80",
    "angle_q95",
    "angle_std",
    "tangent_trace",
    "tangent_effective_rank",
    "tangent_eigshare_1",
    "tangent_eigshare_2",
    "tangent_eigshare_3",
    "tangent_eigshare_4",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Matched CIFAR-100-LT IF=100 head/tail rank swap for one "
            "balanced-space similar class pair."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "lt_rank_swap",
    )
    parser.add_argument("--class-a", type=int, default=47)
    parser.add_argument("--class-b", type=int, default=52)
    parser.add_argument("--imbalance-factor", type=float, default=100.0)
    parser.add_argument("--max-count", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument("--rank-seed", type=int, default=20260801)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_class_names(data_root: Path) -> list[str]:
    with (data_root / "cifar-100-python" / "meta").open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    return list(
        payload.get("fine_label_names", payload.get(b"fine_label_names"))
    )


def exponential_counts(max_count: int, imbalance_factor: float) -> np.ndarray:
    if not 2 <= max_count <= 500:
        raise ValueError("max_count must lie in [2, 500] for CIFAR-100.")
    if imbalance_factor <= 1.0:
        raise ValueError("Imbalance factor must be greater than one.")
    counts = np.asarray(
        [
            int(
                max_count
                * (1.0 / imbalance_factor)
                ** (rank / (NUM_CLASSES - 1))
            )
            for rank in range(NUM_CLASSES)
        ],
        dtype=np.int64,
    )
    if counts[0] != max_count or counts[-1] != int(
        max_count / imbalance_factor
    ):
        raise AssertionError("Unexpected long-tail endpoints.")
    if not np.all(counts[:-1] >= counts[1:]):
        raise AssertionError("Long-tail counts must be non-increasing.")
    return counts


def make_rank_swap_design(
    class_a: int,
    class_b: int,
    curve: np.ndarray,
    rank_seed: int,
) -> dict:
    if class_a == class_b:
        raise ValueError("A and B must be different classes.")
    if not (0 <= class_a < NUM_CLASSES and 0 <= class_b < NUM_CLASSES):
        raise ValueError("Class ids must lie in [0, 99].")
    remaining = np.asarray(
        [
            class_id
            for class_id in range(NUM_CLASSES)
            if class_id not in {class_a, class_b}
        ],
        dtype=np.int64,
    )
    generator = np.random.default_rng(rank_seed)
    generator.shuffle(remaining)
    counts_ab = np.zeros(NUM_CLASSES, dtype=np.int64)
    counts_ba = np.zeros(NUM_CLASSES, dtype=np.int64)
    counts_ab[class_a] = curve[0]
    counts_ab[class_b] = curve[-1]
    counts_ba[class_a] = curve[-1]
    counts_ba[class_b] = curve[0]
    for rank, class_id in enumerate(remaining, start=1):
        counts_ab[class_id] = curve[rank]
        counts_ba[class_id] = curve[rank]
    if not np.array_equal(np.sort(counts_ab), np.sort(curve)):
        raise AssertionError("AB does not use the requested LT histogram.")
    if not np.array_equal(np.sort(counts_ba), np.sort(curve)):
        raise AssertionError("BA does not use the requested LT histogram.")
    background = np.asarray(
        sorted(set(range(NUM_CLASSES)) - {class_a, class_b}),
        dtype=np.int64,
    )
    if not np.array_equal(counts_ab[background], counts_ba[background]):
        raise AssertionError("Background counts differ between AB and BA.")
    return {
        "counts_ab": counts_ab,
        "counts_ba": counts_ba,
        "background_classes": background,
        "background_rank_order": remaining,
    }


def select_indices(
    class_orders: list[np.ndarray], counts: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [
            class_orders[class_id][: int(counts[class_id])]
            for class_id in range(NUM_CLASSES)
        ]
    ).astype(np.int64)


def indices_checksum(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def model_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def primary_shape_state(descriptors: np.ndarray) -> np.ndarray:
    state = np.zeros(
        (len(descriptors), len(PRIMARY_NAMES)), dtype=np.float64
    )
    direct_names = [
        "angle_q50",
        "angle_q80",
        "angle_q95",
        "angle_std",
        "tangent_trace",
        "tangent_effective_rank",
    ]
    for column, name in enumerate(direct_names):
        state[:, column] = descriptors[:, DESCRIPTOR_INDEX[name]]
    trace = np.maximum(
        descriptors[:, DESCRIPTOR_INDEX["tangent_trace"]], 1e-12
    )
    for rank in range(1, 5):
        state[:, 5 + rank] = (
            descriptors[:, DESCRIPTOR_INDEX[f"tangent_eig_{rank}"]]
            / trace
        )
    return state


def train_model(
    seed: int,
    order_name: str,
    counts: np.ndarray,
    dataset: ArrayCIFAR100,
    test_loader: DataLoader,
    seed_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    path = seed_dir / f"model_{order_name}_final.pth"
    if args.skip_train and path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        compatible = (
            int(checkpoint.get("seed", -1)) == seed
            and checkpoint.get("order_name") == order_name
            and int(checkpoint.get("epochs", -1)) == args.epochs
            and int(checkpoint.get("samples_per_epoch", -1))
            == args.samples_per_epoch
            and np.array_equal(
                np.asarray(checkpoint.get("counts", []), dtype=np.int64),
                counts,
            )
        )
        if compatible:
            print(
                f"[seed {seed} {order_name}] reusing {path.name}",
                flush=True,
            )
            return path
        raise ValueError(f"Incompatible checkpoint: {path}")

    set_seed(seed)
    model = ResNetCifar(depth=32).to(device)
    initial_checksum = model_checksum(model)
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
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=args.samples_per_epoch,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
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
        finite = True
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
            if not torch.isfinite(loss):
                finite = False
                raise FloatingPointError(
                    f"Non-finite loss for seed={seed}, order={order_name}."
                )
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
            "finite": finite,
            "examples_drawn": seen,
            "steps": len(loader),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print_every = max(1, args.epochs // 10)
        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(
                f"[seed {seed} {order_name}] "
                f"epoch {epoch + 1:03d}/{args.epochs}, "
                f"loss={row['loss']:.4f}, "
                f"train={row['train_accuracy'] * 100:.2f}%, "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    test_accuracy = classification_accuracy(model, test_loader, device)
    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "order_name": order_name,
            "counts": counts,
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "initial_checksum": initial_checksum,
            "test_accuracy": test_accuracy,
            "history": history,
        },
        path,
    )
    save_json(history, path.with_suffix(".history.json"))
    print(
        f"[seed {seed} {order_name}] "
        f"test={test_accuracy * 100:.2f}% saved={path.name}",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path


def load_model(path: Path, device: torch.device) -> ResNetCifar:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNetCifar(depth=32)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def evaluate_checkpoint(
    path: Path,
    test_dataset: ArrayCIFAR100,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model = load_model(path, device)
    features, logits, labels, _ = extract_features(
        model, test_dataset, args, device
    )
    support = describe_supports(features, labels, logits)
    result = {
        "features": features.astype(np.float32),
        "labels": labels,
        "descriptors": support.descriptors,
        "accuracy": support.class_accuracy,
        "prototypes": support.prototypes,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def analyze_pair(
    seed: int,
    class_a: int,
    class_b: int,
    class_names: list[str],
    background: np.ndarray,
    state_ab: dict,
    state_ba: dict,
    seed_dir: Path,
) -> dict:
    primary_ab = primary_shape_state(state_ab["descriptors"])
    primary_ba = primary_shape_state(state_ba["descriptors"])
    pooled_scale = np.std(
        np.concatenate([primary_ab, primary_ba], axis=0),
        axis=0,
        ddof=1,
    )
    pooled_scale = np.maximum(pooled_scale, 1e-8)
    diff_ba_minus_ab = primary_ba - primary_ab
    background_median = np.median(
        diff_ba_minus_ab[background], axis=0
    )
    delta_a = (
        diff_ba_minus_ab[class_a] - background_median
    ) / pooled_scale
    delta_b = (
        -diff_ba_minus_ab[class_b] + background_median
    ) / pooled_scale
    background_adjusted = (
        diff_ba_minus_ab[background] - background_median
    ) / pooled_scale
    background_norms = np.linalg.norm(background_adjusted, axis=1)

    labels = state_ba["labels"]
    anchor_mask = np.isin(labels, background)
    rotation = orthogonal_procrustes(
        state_ba["features"][anchor_mask],
        state_ab["features"][anchor_mask],
    )
    aligned_ba = l2_normalize(
        state_ba["features"].astype(np.float64) @ rotation
    )
    normalized_ab = l2_normalize(
        state_ab["features"].astype(np.float64)
    )
    aligned_prototypes = np.zeros_like(state_ab["prototypes"])
    for class_id in range(NUM_CLASSES):
        aligned_prototypes[class_id] = normalize_vector(
            aligned_ba[labels == class_id].mean(axis=0)
        )
    prototype_diff = aligned_prototypes - state_ab["prototypes"]
    prototype_background = prototype_diff[background].mean(axis=0)
    prototype_delta_a = (
        prototype_diff[class_a] - prototype_background
    )
    prototype_delta_b = (
        -prototype_diff[class_b] + prototype_background
    )
    alignment_cosine = float(
        np.mean(
            np.sum(
                aligned_ba[anchor_mask]
                * normalized_ab[anchor_mask],
                axis=1,
            )
        )
    )

    accuracy_diff = state_ba["accuracy"] - state_ab["accuracy"]
    accuracy_background = float(np.median(accuracy_diff[background]))
    accuracy_delta_a = float(
        accuracy_diff[class_a] - accuracy_background
    )
    accuracy_delta_b = float(
        -accuracy_diff[class_b] + accuracy_background
    )

    response_rows = []
    for index, name in enumerate(PRIMARY_NAMES):
        response_rows.append(
            {
                "seed": seed,
                "metric": name,
                "delta_a_tail_minus_head": float(delta_a[index]),
                "delta_b_tail_minus_head": float(delta_b[index]),
                "same_sign": int(
                    np.sign(delta_a[index]) == np.sign(delta_b[index])
                ),
            }
        )
    write_csv(response_rows, seed_dir / "ab_responses.csv")

    state_rows = []
    for model_name, state in [("AB", state_ab), ("BA", state_ba)]:
        for role, class_id in [("A", class_a), ("B", class_b)]:
            head_or_tail = (
                "head"
                if (model_name == "AB" and role == "A")
                or (model_name == "BA" and role == "B")
                else "tail"
            )
            row = {
                "seed": seed,
                "model": model_name,
                "role": role,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "head_or_tail": head_or_tail,
                "accuracy": float(state["accuracy"][class_id]),
            }
            row.update(
                {
                    name: float(
                        state["descriptors"][
                            class_id, DESCRIPTOR_INDEX[name]
                        ]
                    )
                    for name in DESCRIPTOR_NAMES
                }
            )
            state_rows.append(row)
    write_csv(state_rows, seed_dir / "ab_support_states.csv")

    result = {
        "seed": seed,
        "class_a": {
            "id": class_a,
            "name": class_names[class_a],
        },
        "class_b": {
            "id": class_b,
            "name": class_names[class_b],
        },
        "alignment_same_image_cosine": alignment_cosine,
        "primary_response_names": PRIMARY_NAMES,
        "delta_a": delta_a,
        "delta_b": delta_b,
        "primary_response_cosine": cosine(delta_a, delta_b),
        "primary_response_correlation": correlation(delta_a, delta_b),
        "primary_sign_agreement": float(
            np.mean(np.sign(delta_a) == np.sign(delta_b))
        ),
        "delta_a_norm": float(np.linalg.norm(delta_a)),
        "delta_b_norm": float(np.linalg.norm(delta_b)),
        "background_norm_median": float(np.median(background_norms)),
        "background_norm_q95": float(np.quantile(background_norms, 0.95)),
        "a_above_background_q95": bool(
            np.linalg.norm(delta_a) > np.quantile(background_norms, 0.95)
        ),
        "b_above_background_q95": bool(
            np.linalg.norm(delta_b) > np.quantile(background_norms, 0.95)
        ),
        "prototype_response_cosine": cosine(
            prototype_delta_a, prototype_delta_b
        ),
        "accuracy_delta_a_tail_minus_head": accuracy_delta_a,
        "accuracy_delta_b_tail_minus_head": accuracy_delta_b,
    }
    save_json(result, seed_dir / "summary.json")
    return result


def run_seed(
    seed: int,
    raw_train,
    raw_test,
    class_orders: list[np.ndarray],
    design: dict,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    test_targets = np.asarray(raw_test.targets, dtype=np.int64)
    indices_ab = select_indices(class_orders, design["counts_ab"])
    indices_ba = select_indices(class_orders, design["counts_ba"])
    dataset_ab = ArrayCIFAR100(
        raw_train.data,
        train_targets,
        indices_ab,
        train_transform(),
    )
    dataset_ba = ArrayCIFAR100(
        raw_train.data,
        train_targets,
        indices_ba,
        train_transform(),
    )
    test_dataset = ArrayCIFAR100(
        raw_test.data,
        test_targets,
        None,
        eval_transform(),
        return_index=True,
    )
    test_loader = DataLoader(
        ArrayCIFAR100(
            raw_test.data, test_targets, None, eval_transform()
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    checkpoint_ab = train_model(
        seed,
        "A_head_B_tail",
        design["counts_ab"],
        dataset_ab,
        test_loader,
        seed_dir,
        args,
        device,
    )
    checkpoint_ba = train_model(
        seed,
        "B_head_A_tail",
        design["counts_ba"],
        dataset_ba,
        test_loader,
        seed_dir,
        args,
        device,
    )
    initial_ab = torch.load(
        checkpoint_ab, map_location="cpu", weights_only=False
    )["initial_checksum"]
    initial_ba = torch.load(
        checkpoint_ba, map_location="cpu", weights_only=False
    )["initial_checksum"]
    if initial_ab != initial_ba:
        raise AssertionError("AB and BA models did not share initialization.")
    print(f"[seed {seed}] extracting fixed-test supports", flush=True)
    state_ab = evaluate_checkpoint(
        checkpoint_ab, test_dataset, args, device
    )
    state_ba = evaluate_checkpoint(
        checkpoint_ba, test_dataset, args, device
    )
    return analyze_pair(
        seed,
        args.class_a,
        args.class_b,
        class_names,
        design["background_classes"],
        state_ab,
        state_ba,
        seed_dir,
    )


def aggregate_results(
    summaries: list[dict],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    pair_cosines = np.asarray(
        [summary["primary_response_cosine"] for summary in summaries]
    )
    sign_agreements = np.asarray(
        [summary["primary_sign_agreement"] for summary in summaries]
    )
    result = {
        "num_seeds": len(summaries),
        "seeds": [summary["seed"] for summary in summaries],
        "epochs": args.epochs,
        "primary_response_cosines": pair_cosines,
        "primary_response_cosine_mean": float(pair_cosines.mean()),
        "both_seed_cosines_positive": bool(np.all(pair_cosines > 0.0)),
        "sign_agreement_mean": float(sign_agreements.mean()),
        "a_above_background_q95_all_seeds": bool(
            all(summary["a_above_background_q95"] for summary in summaries)
        ),
        "b_above_background_q95_all_seeds": bool(
            all(summary["b_above_background_q95"] for summary in summaries)
        ),
    }
    if len(summaries) >= 2:
        result["a_response_cross_seed_cosine"] = cosine(
            np.asarray(summaries[0]["delta_a"]),
            np.asarray(summaries[1]["delta_a"]),
        )
        result["b_response_cross_seed_cosine"] = cosine(
            np.asarray(summaries[0]["delta_b"]),
            np.asarray(summaries[1]["delta_b"]),
        )
    else:
        result["a_response_cross_seed_cosine"] = None
        result["b_response_cross_seed_cosine"] = None
    result["preliminary_supported"] = bool(
        len(summaries) >= 2
        and result["both_seed_cosines_positive"]
        and result["primary_response_cosine_mean"] >= 0.20
        and result["sign_agreement_mean"] >= 0.60
        and result["a_response_cross_seed_cosine"] > 0.0
        and result["b_response_cross_seed_cosine"] > 0.0
    )
    save_json(result, output_dir / "aggregate.json")
    return result


def build_report(
    summaries: list[dict],
    aggregate: dict,
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    verdict = (
        "初步支持：A、B在两个随机种子中呈现方向一致且可重复的头到尾支撑变化。"
        if aggregate["preliminary_supported"]
        else "暂未通过：A、B响应一致性或跨种子重复性未同时达到预设门槛。"
    )
    lines = [
        "# IF=100相似类别头尾秩交换实验",
        "",
        f"- A：{class_names[args.class_a]}（{args.class_a}）",
        f"- B：{class_names[args.class_b]}（{args.class_b}）",
        f"- 两个长尾数据集具有相同IF={args.imbalance_factor:g}频率直方图，只有A、B交换500-shot和5-shot身份。",
        f"- 每个模型训练{args.epochs}轮；支撑状态在固定平衡测试集上估计。",
        "",
        "## 每个随机种子",
        "",
        "| Seed | A–B响应余弦 | 响应相关 | 符号一致率 | A超过背景95% | B超过背景95% | 对齐余弦 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['seed']} | "
            f"{summary['primary_response_cosine']:.4f} | "
            f"{summary['primary_response_correlation']:.4f} | "
            f"{summary['primary_sign_agreement'] * 100:.1f}% | "
            f"{summary['a_above_background_q95']} | "
            f"{summary['b_above_background_q95']} | "
            f"{summary['alignment_same_image_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 初步判断",
            "",
            f"**{verdict}**",
            "",
            f"- A–B响应余弦均值：**{aggregate['primary_response_cosine_mean']:.4f}**",
            f"- 平均符号一致率：**{aggregate['sign_agreement_mean'] * 100:.1f}%**",
            f"- A跨种子响应余弦：**{aggregate['a_response_cross_seed_cosine']}**",
            f"- B跨种子响应余弦：**{aggregate['b_response_cross_seed_cosine']}**",
            "",
            "本实验只验证一个相似类别对的500-shot/5-shot端点响应，不能单独证明该规律对所有相似类普遍成立。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.seeds = args.seeds[:1]
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(args.data_root)
    curve = exponential_counts(args.max_count, args.imbalance_factor)
    design = make_rank_swap_design(
        args.class_a, args.class_b, curve, args.rank_seed
    )

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    raw_test = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False
    )
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    class_orders = make_nested_class_orders(
        train_targets, subset_seed=args.subset_seed
    )
    indices_ab = select_indices(class_orders, design["counts_ab"])
    indices_ba = select_indices(class_orders, design["counts_ba"])
    background = design["background_classes"]
    background_a = np.concatenate(
        [
            class_orders[class_id][
                : int(design["counts_ab"][class_id])
            ]
            for class_id in background
        ]
    )
    background_b = np.concatenate(
        [
            class_orders[class_id][
                : int(design["counts_ba"][class_id])
            ]
            for class_id in background
        ]
    )
    audit = {
        "class_a": {
            "id": args.class_a,
            "name": class_names[args.class_a],
        },
        "class_b": {
            "id": args.class_b,
            "name": class_names[args.class_b],
        },
        "imbalance_factor": args.imbalance_factor,
        "curve": curve,
        "dataset_size_ab": len(indices_ab),
        "dataset_size_ba": len(indices_ba),
        "same_frequency_histogram": bool(
            np.array_equal(
                np.sort(design["counts_ab"]),
                np.sort(design["counts_ba"]),
            )
        ),
        "same_total_size": bool(len(indices_ab) == len(indices_ba)),
        "background_counts_identical": bool(
            np.array_equal(
                design["counts_ab"][background],
                design["counts_ba"][background],
            )
        ),
        "background_indices_identical": bool(
            np.array_equal(background_a, background_b)
        ),
        "background_indices_checksum_ab": indices_checksum(background_a),
        "background_indices_checksum_ba": indices_checksum(background_b),
        "a_tail_nested_in_head": bool(
            set(class_orders[args.class_a][: curve[-1]])
            <= set(class_orders[args.class_a][: curve[0]])
        ),
        "b_tail_nested_in_head": bool(
            set(class_orders[args.class_b][: curve[-1]])
            <= set(class_orders[args.class_b][: curve[0]])
        ),
        "trained_from_balanced_reference": False,
        "fixed_balanced_test_support": True,
        "arguments": vars(args),
    }
    if not all(
        [
            audit["same_frequency_histogram"],
            audit["same_total_size"],
            audit["background_counts_identical"],
            audit["background_indices_identical"],
            audit["a_tail_nested_in_head"],
            audit["b_tail_nested_in_head"],
        ]
    ):
        raise AssertionError("Rank-swap data audit failed.")
    save_json(audit, args.output_dir / "design_audit.json")
    count_rows = []
    for class_id in range(NUM_CLASSES):
        count_rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "count_A_head_B_tail": int(
                    design["counts_ab"][class_id]
                ),
                "count_B_head_A_tail": int(
                    design["counts_ba"][class_id]
                ),
                "changed": int(
                    design["counts_ab"][class_id]
                    != design["counts_ba"][class_id]
                ),
            }
        )
    write_csv(count_rows, args.output_dir / "class_counts.csv")
    if args.design_only:
        print(
            f"Design validated: {args.output_dir / 'design_audit.json'}",
            flush=True,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"A={class_names[args.class_a]}({args.class_a}), "
        f"B={class_names[args.class_b]}({args.class_b}), "
        f"IF={args.imbalance_factor:g}, models/seed=2",
        flush=True,
    )
    summaries = [
        run_seed(
            seed,
            raw_train,
            raw_test,
            class_orders,
            design,
            class_names,
            args,
            device,
        )
        for seed in args.seeds
    ]
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    aggregate = aggregate_results(summaries, args, args.output_dir)
    build_report(summaries, aggregate, class_names, args)
    print(f"Finished: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
