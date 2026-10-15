from __future__ import annotations

import argparse
import csv
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
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


PRIMARY_NAMES = ANGLE_NAMES + SHAPE_NAMES


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Strict one-class-at-a-time CIFAR-100 frequency intervention. "
            "Only the probe class changes count; all other classes stay at 500."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--similarity-csv",
        type=Path,
        default=(
            project_dir
            / "outputs"
            / "balanced_neighbor_trajectory"
            / "seed_42"
            / "all_class_pairs.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "single_class_intervention",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[5, 20, 50, 200, 500]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument("--target-class", type=int)
    parser.add_argument("--similar-class", type=int)
    parser.add_argument("--far-class", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse compatible completed checkpoints.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one seed and two epochs in a separate smoke directory.",
    )
    return parser.parse_args()


def read_class_names(data_root: Path) -> list[str]:
    with (data_root / "cifar-100-python" / "meta").open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    return list(
        payload.get("fine_label_names", payload.get(b"fine_label_names"))
    )


def choose_probe_classes(args: argparse.Namespace) -> dict:
    explicit = [
        args.target_class,
        args.similar_class,
        args.far_class,
    ]
    if all(value is not None for value in explicit):
        target, similar, far = (int(value) for value in explicit)
        if len({target, similar, far}) != 3:
            raise ValueError("Target, similar, and far classes must differ.")
        return {"target": target, "similar": similar, "far": far}
    if any(value is not None for value in explicit):
        raise ValueError(
            "Set all of --target-class/--similar-class/--far-class or none."
        )
    if not args.similarity_csv.exists():
        raise FileNotFoundError(
            f"Balanced-space pair file not found: {args.similarity_csv}"
        )
    rows = list(
        csv.DictReader(args.similarity_csv.open("r", encoding="utf-8-sig"))
    )
    undirected = [
        row
        for row in rows
        if int(row["target_class"]) < int(row["candidate_class"])
        and int(row["same_coarse"]) == 1
    ]
    if not undirected:
        raise ValueError("No same-coarse candidate pairs in similarity CSV.")
    best = max(undirected, key=lambda row: float(row["knn"]))
    target = int(best["target_class"])
    similar = int(best["candidate_class"])
    target_rows = [
        row
        for row in rows
        if int(row["target_class"]) == target
        and int(row["candidate_class"]) != similar
    ]
    # Fusion combines independent balanced prototype, confusion, and local-KNN
    # ranks. Its minimum is the deliberately dissimilar negative control.
    far_row = min(target_rows, key=lambda row: float(row["fusion"]))
    return {
        "target": target,
        "similar": similar,
        "far": int(far_row["candidate_class"]),
    }


def selected_indices(
    class_orders: list[np.ndarray], probe_class: int, probe_count: int
) -> np.ndarray:
    pieces = []
    for class_id, order in enumerate(class_orders):
        count = probe_count if class_id == probe_class else 500
        pieces.append(order[:count])
    return np.concatenate(pieces).astype(np.int64)


def checkpoint_name(probe_class: int | None, count: int) -> str:
    if probe_class is None:
        return f"model_shared_head_count_{count:03d}_final.pth"
    return f"model_probe_{probe_class:03d}_count_{count:03d}_final.pth"


def train_single_model(
    seed: int,
    probe_class: int | None,
    probe_count: int,
    train_dataset: ArrayCIFAR100,
    test_loader: DataLoader,
    seed_dir: Path,
    fixed_steps_per_epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    path = seed_dir / checkpoint_name(probe_class, probe_count)
    expected_probe = -1 if probe_class is None else int(probe_class)
    if args.skip_train and path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        compatible = (
            int(checkpoint.get("seed", -1)) == seed
            and int(checkpoint.get("probe_class", -999)) == expected_probe
            and int(checkpoint.get("probe_count", -1)) == probe_count
            and int(checkpoint.get("epochs", -1)) == args.epochs
            and int(checkpoint.get("fixed_steps_per_epoch", -1))
            == fixed_steps_per_epoch
        )
        if compatible:
            print(f"[seed {seed}] reusing {path.name}", flush=True)
            return path
        raise ValueError(f"Incompatible checkpoint: {path}")

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
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    if len(loader) < fixed_steps_per_epoch:
        raise ValueError("Training loader is shorter than fixed step budget.")

    label = "shared-head" if probe_class is None else f"class-{probe_class}"
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
        for step, (images, labels) in enumerate(loader):
            if step >= fixed_steps_per_epoch:
                break
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
            "examples_seen": seen,
            "steps": fixed_steps_per_epoch,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print_every = max(1, args.epochs // 5)
        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(
                f"[seed {seed} {label} n={probe_count}] "
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
            "probe_class": expected_probe,
            "probe_count": probe_count,
            "epochs": args.epochs,
            "fixed_steps_per_epoch": fixed_steps_per_epoch,
            "test_accuracy": test_accuracy,
            "history": history,
        },
        path,
    )
    save_json(history, path.with_suffix(".history.json"))
    print(
        f"[seed {seed} {label} n={probe_count}] "
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


def cosine(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def correlation(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    if np.std(vector_a) <= 1e-12 or np.std(vector_b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(vector_a, vector_b)[0, 1])


def evaluate_checkpoint(
    checkpoint: Path,
    test_dataset: ArrayCIFAR100,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model = load_model(checkpoint, device)
    features, logits, labels, indices = extract_features(
        model, test_dataset, args, device
    )
    support = describe_supports(features, labels, logits)
    result = {
        "features": features.astype(np.float32),
        "logits": logits.astype(np.float32),
        "labels": labels,
        "indices": indices,
        "descriptors": support.descriptors,
        "prototypes": support.prototypes,
        "accuracy": support.class_accuracy,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def make_trajectory(
    probe: int,
    states: dict[int, dict],
    head_scale: np.ndarray,
) -> dict:
    counts = sorted(states, reverse=True)
    primary_indices = np.asarray(
        [DESCRIPTOR_INDEX[name] for name in PRIMARY_NAMES], dtype=np.int64
    )
    all_increment_rows = []
    primary_vectors = []
    prototype_vectors = []
    accuracy_vector = []
    for high_count, low_count in zip(counts[:-1], counts[1:]):
        high = states[high_count]
        low = states[low_count]
        descriptor_delta = low["descriptors"] - high["descriptors"]
        control_mask = np.arange(NUM_CLASSES) != probe
        control_median = np.median(descriptor_delta[control_mask], axis=0)
        adjusted = (
            descriptor_delta[probe] - control_median
        ) / np.maximum(head_scale, 1e-8)
        primary_vectors.append(adjusted[primary_indices])

        accuracy_delta = low["accuracy"] - high["accuracy"]
        adjusted_accuracy = float(
            accuracy_delta[probe] - np.median(accuracy_delta[control_mask])
        )
        accuracy_vector.append(adjusted_accuracy)

        prototype_vectors.append(
            low["aligned_probe_prototype"]
            - high["aligned_probe_prototype"]
        )
        row = {
            "probe_class": probe,
            "high_count": high_count,
            "low_count": low_count,
            "adjusted_accuracy_delta": adjusted_accuracy,
        }
        row.update(
            {
                f"adjusted_{name}": float(adjusted[index])
                for index, name in enumerate(DESCRIPTOR_NAMES)
            }
        )
        all_increment_rows.append(row)
    primary = np.asarray(primary_vectors)
    prototype = np.asarray(prototype_vectors)
    accuracy = np.asarray(accuracy_vector)
    return {
        "counts": counts,
        "primary_steps": primary,
        "primary_vector": primary.reshape(-1),
        "prototype_steps": prototype,
        "prototype_vector": prototype.reshape(-1),
        "accuracy_vector": accuracy,
        "increment_rows": all_increment_rows,
    }


def compare_trajectories(first: dict, second: dict) -> dict:
    step_cosines = [
        cosine(first_step, second_step)
        for first_step, second_step in zip(
            first["primary_steps"], second["primary_steps"]
        )
    ]
    endpoint_first = first["primary_steps"].sum(axis=0)
    endpoint_second = second["primary_steps"].sum(axis=0)
    prototype_step_cosines = [
        cosine(first_step, second_step)
        for first_step, second_step in zip(
            first["prototype_steps"], second["prototype_steps"]
        )
    ]
    return {
        "primary_trajectory_cosine": cosine(
            first["primary_vector"], second["primary_vector"]
        ),
        "primary_trajectory_correlation": correlation(
            first["primary_vector"], second["primary_vector"]
        ),
        "primary_step_cosine_mean": float(np.mean(step_cosines)),
        "primary_step_cosines": step_cosines,
        "primary_endpoint_cosine": cosine(endpoint_first, endpoint_second),
        "primary_sign_agreement": float(
            np.mean(
                np.sign(first["primary_vector"])
                == np.sign(second["primary_vector"])
            )
        ),
        "accuracy_trajectory_cosine": cosine(
            first["accuracy_vector"], second["accuracy_vector"]
        ),
        "accuracy_trajectory_correlation": correlation(
            first["accuracy_vector"], second["accuracy_vector"]
        ),
        "prototype_trajectory_cosine": cosine(
            first["prototype_vector"], second["prototype_vector"]
        ),
        "prototype_step_cosine_mean": float(
            np.mean(prototype_step_cosines)
        ),
        "prototype_endpoint_cosine": cosine(
            first["prototype_steps"].sum(axis=0),
            second["prototype_steps"].sum(axis=0),
        ),
    }


def run_seed(
    seed: int,
    raw_train,
    raw_test,
    class_orders: list[np.ndarray],
    probes: dict[str, int],
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    test_targets = np.asarray(raw_test.targets, dtype=np.int64)
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
    fixed_steps = (99 * 500 + min(args.levels)) // args.batch_size
    if fixed_steps < 1:
        raise ValueError("Batch size is too large for the fixed step budget.")

    head_indices = selected_indices(class_orders, probes["target"], 500)
    head_dataset = ArrayCIFAR100(
        raw_train.data,
        train_targets,
        head_indices,
        train_transform(),
    )
    head_path = train_single_model(
        seed,
        None,
        500,
        head_dataset,
        test_loader,
        seed_dir,
        fixed_steps,
        args,
        device,
    )

    checkpoints: dict[int, dict[int, Path]] = {}
    for probe in probes.values():
        checkpoints[probe] = {500: head_path}
        for count in sorted(args.levels, reverse=True):
            if count == 500:
                continue
            indices = selected_indices(class_orders, probe, count)
            dataset = ArrayCIFAR100(
                raw_train.data,
                train_targets,
                indices,
                train_transform(),
            )
            checkpoints[probe][count] = train_single_model(
                seed,
                probe,
                count,
                dataset,
                test_loader,
                seed_dir,
                fixed_steps,
                args,
                device,
            )

    print(f"[seed {seed}] evaluating support states", flush=True)
    head_state = evaluate_checkpoint(
        head_path, test_dataset, args, device
    )
    states_by_probe: dict[int, dict[int, dict]] = {}
    state_rows = []
    for role, probe in probes.items():
        states_by_probe[probe] = {}
        for count in sorted(args.levels, reverse=True):
            if count == 500:
                state = {
                    key: value
                    for key, value in head_state.items()
                    if key not in {"features", "logits", "indices"}
                }
                state["aligned_probe_prototype"] = head_state[
                    "prototypes"
                ][probe]
                state["alignment_cosine"] = 1.0
            else:
                evaluated = evaluate_checkpoint(
                    checkpoints[probe][count],
                    test_dataset,
                    args,
                    device,
                )
                control_mask = evaluated["labels"] != probe
                rotation = orthogonal_procrustes(
                    evaluated["features"][control_mask],
                    head_state["features"][control_mask],
                )
                aligned = l2_normalize(
                    evaluated["features"].astype(np.float64) @ rotation
                )
                aligned_head = l2_normalize(
                    head_state["features"].astype(np.float64)
                )
                aligned_prototype = normalize_vector(
                    aligned[evaluated["labels"] == probe].mean(axis=0)
                )
                state = {
                    "labels": evaluated["labels"],
                    "descriptors": evaluated["descriptors"],
                    "prototypes": evaluated["prototypes"],
                    "accuracy": evaluated["accuracy"],
                    "aligned_probe_prototype": aligned_prototype,
                    "alignment_cosine": float(
                        np.mean(
                            np.sum(
                                aligned[control_mask]
                                * aligned_head[control_mask],
                                axis=1,
                            )
                        )
                    ),
                }
            states_by_probe[probe][count] = state
            row = {
                "seed": seed,
                "role": role,
                "class_id": probe,
                "class_name": class_names[probe],
                "count": count,
                "class_accuracy": float(state["accuracy"][probe]),
                "alignment_cosine": float(state["alignment_cosine"]),
            }
            row.update(
                {
                    name: float(
                        state["descriptors"][
                            probe, DESCRIPTOR_INDEX[name]
                        ]
                    )
                    for name in DESCRIPTOR_NAMES
                }
            )
            state_rows.append(row)
    write_csv(state_rows, seed_dir / "support_states.csv")

    head_scale = np.std(
        head_state["descriptors"], axis=0, ddof=1
    )
    trajectories = {
        role: make_trajectory(
            probe, states_by_probe[probe], head_scale
        )
        for role, probe in probes.items()
    }
    increment_rows = []
    for role, trajectory in trajectories.items():
        for row in trajectory["increment_rows"]:
            row["seed"] = seed
            row["role"] = role
            row["class_name"] = class_names[row["probe_class"]]
            increment_rows.append(row)
    write_csv(increment_rows, seed_dir / "adjusted_increments.csv")

    similar_comparison = compare_trajectories(
        trajectories["target"], trajectories["similar"]
    )
    far_comparison = compare_trajectories(
        trajectories["target"], trajectories["far"]
    )
    summary = {
        "seed": seed,
        "fixed_steps_per_epoch": fixed_steps,
        "examples_per_epoch": fixed_steps * args.batch_size,
        "probes": {
            role: {
                "class_id": class_id,
                "class_name": class_names[class_id],
            }
            for role, class_id in probes.items()
        },
        "comparisons": {
            "target_similar": similar_comparison,
            "target_far": far_comparison,
            "similar_minus_far": {
                key: float(similar_comparison[key] - far_comparison[key])
                for key in similar_comparison
                if isinstance(similar_comparison[key], (int, float))
            },
        },
    }
    save_json(summary, seed_dir / "summary.json")
    del head_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def build_report(
    summaries: list[dict],
    probes: dict[str, int],
    class_names: list[str],
    output_dir: Path,
) -> None:
    metric_names = [
        "primary_trajectory_cosine",
        "primary_step_cosine_mean",
        "primary_endpoint_cosine",
        "primary_sign_agreement",
        "prototype_trajectory_cosine",
        "accuracy_trajectory_cosine",
    ]
    lines = [
        "# 单类别频率干预实验",
        "",
        "本实验每次只改变一个探针类别的训练样本数；其余99类始终保持每类500个样本。不同频率模型从相同随机初始化独立训练，并使用相同的每轮优化步数。",
        "",
        "## 探针类别",
        "",
        f"- 目标类：{class_names[probes['target']]}（{probes['target']}）",
        f"- 平衡空间相似类：{class_names[probes['similar']]}（{probes['similar']}）",
        f"- 平衡空间远类：{class_names[probes['far']]}（{probes['far']}）",
        "",
        "主要轨迹由角分位数和切空间协方差结构组成。每一步变化均减去99个固定频率类别的中位变化，再按共享500-shot模型的类间标准差归一化。",
        "",
        "## 轨迹比较",
        "",
        "| Seed | 指标 | 目标–相似类 | 目标–远类 | 差值 |",
        "|---:|---|---:|---:|---:|",
    ]
    for summary in summaries:
        similar = summary["comparisons"]["target_similar"]
        far = summary["comparisons"]["target_far"]
        for metric in metric_names:
            lines.append(
                f"| {summary['seed']} | {metric} | "
                f"{similar[metric]:.4f} | {far[metric]:.4f} | "
                f"{similar[metric] - far[metric]:+.4f} |"
            )

    primary_similar = np.asarray(
        [
            summary["comparisons"]["target_similar"][
                "primary_trajectory_cosine"
            ]
            for summary in summaries
        ]
    )
    primary_far = np.asarray(
        [
            summary["comparisons"]["target_far"][
                "primary_trajectory_cosine"
            ]
            for summary in summaries
        ]
    )
    margin = primary_similar - primary_far
    preliminary_pass = bool(
        np.all(margin > 0.0) and float(margin.mean()) >= 0.10
    )
    verdict = (
        "初步通过：两个种子中相似类轨迹都比远类更接近，且平均优势达到预设的0.10。"
        if preliminary_pass
        else "暂未通过：相似类轨迹没有在所有种子中稳定优于远类，或平均优势不足0.10。"
    )
    lines.extend(
        [
            "",
            "## 初步判定",
            "",
            f"**{verdict}**",
            "",
            f"- 相似类主要轨迹余弦均值：**{primary_similar.mean():.4f}**",
            f"- 远类主要轨迹余弦均值：**{primary_far.mean():.4f}**",
            f"- 平均优势：**{margin.mean():+.4f}**",
            f"- 相似类优于远类的种子比例：**{np.mean(margin > 0) * 100:.1f}%**",
            "",
            "两种子只用于首轮可行性筛查，不能替代多类别、多种子统计检验。",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.levels = sorted(set(args.levels))
    if args.levels != [5, 20, 50, 200, 500]:
        raise ValueError("Formal design expects levels 5, 20, 50, 200, 500.")
    if args.smoke:
        args.epochs = 2
        args.seeds = args.seeds[:1]
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probes = choose_probe_classes(args)
    class_names = read_class_names(args.data_root)
    save_json(
        {
            "arguments": vars(args),
            "design": {
                "intervention": (
                    "one probe class varies; other 99 classes remain at 500"
                ),
                "probes": {
                    role: {
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                    }
                    for role, class_id in probes.items()
                },
                "shared_head_checkpoint": True,
                "nested_subsets": True,
                "fixed_steps_per_epoch": True,
                "primary_descriptors": PRIMARY_NAMES,
            },
        },
        args.output_dir / "design.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        "Probes: "
        + ", ".join(
            f"{role}={class_names[class_id]}({class_id})"
            for role, class_id in probes.items()
        ),
        flush=True,
    )

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    raw_test = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False
    )
    class_orders = make_nested_class_orders(
        np.asarray(raw_train.targets, dtype=np.int64),
        subset_seed=args.subset_seed,
    )
    summaries = [
        run_seed(
            seed,
            raw_train,
            raw_test,
            class_orders,
            probes,
            class_names,
            args,
            device,
        )
        for seed in args.seeds
    ]
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    build_report(summaries, probes, class_names, args.output_dir)
    print(f"Finished: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
