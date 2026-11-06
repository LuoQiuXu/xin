from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
AUDIT_ROOT = ROOT.parent
REPO_ROOT = ROOT.parents[1]
CLEAN_RUN = AUDIT_ROOT / "clean_vs_runs" / "clean_s1_seed42_vs_seed123_20261013"
STABILITY_RUN = AUDIT_ROOT / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014"
RUNTIME = STABILITY_RUN / "runtime"
DEFAULT_DATA_ROOT = REPO_ROOT / "H2T" / "data" / "cifar100"

EXPECTED_CLEAN_SHA256 = "7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88"
EXPECTED_CLEAN_EPOCH = 184
EXPECTED_TRAIN_PICKLE_SHA256 = "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"

NUM_CLASSES = 100
FEATURE_DIM = 64
TRAIN_SAMPLES = 10847
VALIDATION_SAMPLES = 1980
BATCH_SIZE = 64
STAGE2_EPOCHS = 30
STAGE2_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-4
CLASS_WEIGHT_DIV = 2.0
ADVER_REG = 1.0
ALPHA_MAX = 0.3
ALPHA_RAMP_EPOCHS = 30
SEEDS = (123, 124, 125)
ARMS = ("C0_BALANCED_CE", "C1_CCSAM_FEATURE_GRAD", "C2_WEIGHTED_CE_NO_PERTURB")

_TEST_GUARD = None


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for value in arrays:
        arr = np.asarray(value)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def hash_tensor(tensor: torch.Tensor) -> str:
    return hash_arrays(tensor.detach().cpu().contiguous().numpy())


def state_hash(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def install_test_guard(data_root: Path):
    global _TEST_GUARD
    if _TEST_GUARD is not None:
        raise RuntimeError("TEST guard is already installed in this process")

    forbidden = (Path(data_root) / "cifar-100-python" / "test").resolve()
    _TEST_GUARD = {
        "installed_at": now(),
        "official_test_file_open_attempts": 0,
        "CIFAR100_train_false_attempts": 0,
        "blocked_attempts": [],
        "official_TEST_used": False,
    }

    def block(kind, value):
        _TEST_GUARD["blocked_attempts"].append({"kind": kind, "value": str(value), "at": now()})
        raise RuntimeError("OFFICIAL CIFAR-100 TEST ACCESS IS PROHIBITED: " + kind + ": " + str(value))

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                candidate = Path(os.fsdecode(args[0])).resolve()
                if candidate == forbidden or forbidden in candidate.parents:
                    _TEST_GUARD["official_test_file_open_attempts"] += 1
                    block("test pickle open", candidate)
            except (OSError, TypeError, ValueError):
                return

    sys.addaudithook(audit)

    original_init = torchvision.datasets.CIFAR100.__init__

    def guarded_cifar100_init(self, root, train=True, *args, **kwargs):
        if train is False:
            _TEST_GUARD["CIFAR100_train_false_attempts"] += 1
            block("torchvision.datasets.CIFAR100(train=False)", root)
        return original_init(self, root, train, *args, **kwargs)

    torchvision.datasets.CIFAR100.__init__ = guarded_cifar100_init
    return _TEST_GUARD


def guard_pass():
    if _TEST_GUARD is None:
        raise RuntimeError("TEST guard was not installed")
    if _TEST_GUARD["blocked_attempts"]:
        raise RuntimeError("TEST guard recorded blocked attempts")
    if _TEST_GUARD["official_test_file_open_attempts"] != 0:
        raise RuntimeError("Official TEST file access was attempted")
    if _TEST_GUARD["CIFAR100_train_false_attempts"] != 0:
        raise RuntimeError("CIFAR100(train=False) was attempted")
    return dict(_TEST_GUARD)


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)


def resolve_checkpoint(override=None) -> Path:
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend([
        STABILITY_RUN / "locked" / "clean_stage1_best.pth",
        CLEAN_RUN / "stage1" / "checkpoints" / "clean_stage1_best.pth",
    ])
    for candidate in candidates:
        if candidate.is_file():
            digest = sha256_file(candidate)
            if digest != EXPECTED_CLEAN_SHA256:
                raise RuntimeError(
                    "Found a Clean Stage I checkpoint, but its SHA256 does not match the sealed checkpoint. "
                    f"Path={candidate}; actual={digest}; expected={EXPECTED_CLEAN_SHA256}"
                )
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find the sealed Clean Stage I checkpoint. Checked: "
        + "; ".join(str(p) for p in candidates)
        + ". Do not retrain Stage I or substitute another checkpoint."
    )


def build_transforms():
    source_file = RUNTIME / "datasets" / "cifar100.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"Expected frozen runtime source is missing: {source_file}")
    source = source_file.read_text(encoding="utf-8")
    required = [
        "transforms.RandomCrop(32, padding=4)",
        "transforms.RandomHorizontalFlip()",
        "transforms.ToTensor()",
        "transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))",
    ]
    missing = [item for item in required if item not in source]
    if missing:
        raise RuntimeError("The audited CIFAR transform source changed; refusing silent protocol drift: " + repr(missing))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_transform, eval_transform, sha256_file(source_file)


def build_models(checkpoint_path: Path, device: torch.device):
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    from models.resnet_cifar import resnet32_fe, Classifier

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != EXPECTED_CLEAN_EPOCH:
        raise RuntimeError(f"Expected Clean Stage I epoch {EXPECTED_CLEAN_EPOCH}, got {checkpoint.get('epoch')}")
    if checkpoint.get("seed") != 42:
        raise RuntimeError(f"Expected Clean Stage I seed 42, got {checkpoint.get('seed')}")
    if checkpoint.get("TEST_used") is not False:
        raise RuntimeError("Checkpoint provenance does not explicitly certify TEST_used=False")
    if checkpoint.get("selection_partition") != "official TRAIN holdout":
        raise RuntimeError("Checkpoint selection provenance is not the sealed official TRAIN holdout")

    backbone = torch.nn.DataParallel(resnet32_fe()).to(device)
    classifier = torch.nn.DataParallel(Classifier(FEATURE_DIM, NUM_CLASSES)).to(device)
    backbone.load_state_dict(checkpoint["state_dict_model"], strict=True)
    classifier.load_state_dict(checkpoint["state_dict_classifier"], strict=True)

    backbone.eval()
    backbone.requires_grad_(False)
    for module in backbone.modules():
        module.eval()
    classifier.train()

    backbone_params = sum(p.numel() for p in backbone.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    if backbone_params != 463504 or classifier_params != 6500:
        raise RuntimeError(
            f"Unexpected architecture size: backbone={backbone_params}, classifier={classifier_params}"
        )
    return backbone, classifier, checkpoint


def extract_gap(backbone, classifier, images: np.ndarray, eval_transform, device, batch_size=128):
    feature_chunks = []
    classifier.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            batch = torch.stack([eval_transform(Image.fromarray(im)) for im in chunk]).to(device)
            feature_map = backbone(batch)
            if feature_map.ndim != 4 or feature_map.shape[1:] != (64, 8, 8):
                raise RuntimeError(f"Expected [B,64,8,8] Clean feature maps, got {tuple(feature_map.shape)}")
            gap = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            logits_from_gap = classifier.module.fc(gap)
            logits_from_map = classifier(feature_map)
            if not torch.allclose(logits_from_gap, logits_from_map, rtol=1e-5, atol=1e-6):
                raise RuntimeError("GAP(64-D) classifier path is not equivalent to the frozen Clean classifier")
            feature_chunks.append(gap.cpu().numpy().astype(np.float32, copy=False))
    classifier.train()
    return np.concatenate(feature_chunks, axis=0)


def metrics_from_predictions(pred: np.ndarray, labels: np.ndarray, train_counts: np.ndarray):
    pred = np.asarray(pred, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    correct = np.bincount(labels[pred == labels], minlength=NUM_CLASSES)
    per_class = np.divide(
        correct * 100.0,
        counts,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    groups = {
        "Head": np.arange(0, 36),
        "OfficialMedium": np.arange(36, 71),
        "Tail": np.arange(71, 100),
        "Many": np.flatnonzero(train_counts > 100),
        "MediumN": np.flatnonzero((train_counts >= 20) & (train_counts <= 100)),
        "Few": np.flatnonzero(train_counts < 20),
    }
    out = {
        "Overall": float(np.mean(pred == labels) * 100.0),
        "Macro": float(np.nanmean(per_class)),
        "samples": int(len(labels)),
        "logits_classes": NUM_CLASSES,
        "missing_validation_classes": np.flatnonzero(counts == 0).tolist(),
        "class_counts": counts.tolist(),
        "class_correct": correct.tolist(),
        "per_class_accuracy": [None if np.isnan(x) else float(x) for x in per_class],
        "effective_classes": {},
    }
    for name, ids in groups.items():
        active = ids[counts[ids] > 0]
        out[name] = float(np.nanmean(per_class[active]))
        out["effective_classes"][name] = active.tolist()
    return out


def evaluate_gap(classifier, gap_features: np.ndarray, labels: np.ndarray, train_counts: np.ndarray, device):
    classifier.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(labels), 512):
            x = torch.from_numpy(np.array(gap_features[start:start + 512], copy=True)).to(device)
            logits = classifier.module.fc(x)
            if logits.shape != (len(x), NUM_CLASSES) or not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite or malformed validation logits")
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(predictions)
    classifier.train()
    return metrics_from_predictions(pred, labels, train_counts), pred


def make_class_aware_plan(labels: np.ndarray, seed: int, epochs=STAGE2_EPOCHS):
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(NUM_CLASSES)):
        raise RuntimeError("Class-aware sampling requires all 100 classes to be present")

    class_lists = [np.flatnonzero(labels == c).tolist() for c in range(NUM_CLASSES)]
    if any(len(x) == 0 for x in class_lists):
        raise RuntimeError("A class has no training examples")

    rng = random.Random(int(seed))

    class RandomCycleIter:
        def __init__(self, data):
            self.data_list = list(data)
            self.length = len(self.data_list)
            self.i = self.length - 1

        def next_value(self):
            self.i += 1
            if self.i == self.length:
                self.i = 0
                rng.shuffle(self.data_list)
            return self.data_list[self.i]

    class_iter = RandomCycleIter(range(NUM_CLASSES))
    data_iters = [RandomCycleIter(x) for x in class_lists]
    plan = np.empty((epochs, len(labels)), dtype=np.int32)
    for epoch in range(epochs):
        order = np.fromiter(
            (data_iters[class_iter.next_value()].next_value() for _ in range(len(labels))),
            dtype=np.int32,
            count=len(labels),
        )
        per_class = np.bincount(labels[order], minlength=NUM_CLASSES)
        if int(per_class.max() - per_class.min()) > 1:
            raise RuntimeError("Class-aware sampler did not produce a balanced epoch")
        plan[epoch] = order
    return plan


def augment_batch(raw_images: np.ndarray, indices: np.ndarray, transform, generator: torch.Generator):
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        batch = torch.stack([transform(Image.fromarray(raw_images[int(i)])) for i in indices])
        generator.set_state(torch.get_rng_state())
    if not torch.equal(before, torch.get_rng_state()):
        raise RuntimeError("Augmentation changed the ambient CPU RNG state")
    return batch


def class_weights_from_counts(counts: np.ndarray, device):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (NUM_CLASSES,) or np.any(counts <= 0):
        raise RuntimeError("Invalid class counts for CC-SAM class weights")
    max_count = float(counts.max())
    weights = np.log(max_count / counts + 0.01) / CLASS_WEIGHT_DIV
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def alpha_for_epoch(epoch_index: int):
    return min((epoch_index + 1) / float(ALPHA_RAMP_EPOCHS), ALPHA_MAX)


def check_backbone_unchanged(backbone, frozen_state):
    backbone.eval()
    if any(p.requires_grad for p in backbone.parameters()):
        raise RuntimeError("A frozen backbone parameter became trainable")
    if any(p.grad is not None for p in backbone.parameters()):
        raise RuntimeError("A frozen backbone parameter unexpectedly received a gradient")
    current = backbone.state_dict()
    for name, value in current.items():
        if not torch.equal(value.detach().cpu(), frozen_state[name]):
            raise RuntimeError(f"Frozen backbone or BN buffer changed: {name}")
