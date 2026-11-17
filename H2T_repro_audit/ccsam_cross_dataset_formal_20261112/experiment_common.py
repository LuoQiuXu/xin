"""Shared paths, provenance, and run records for the frozen transfer protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cuda_statistics import DEFAULT_CUDA_DEVICE, initialize_cuda

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROTOCOL = ROOT / "experiment_protocol.json"
BRANCH = "experiment/ccsam-selective-margin-guard-20261117"
TRAIN_MANIFESTS = {
    "imagenet_lt": REPO / "H2T/datasets/data_txt/ImageNet_LT_train.txt",
    "places_lt": REPO / "H2T/datasets/data_txt/Places_LT_train.txt",
}
METRICS = ("Overall", "Many", "Medium", "Few")


def attempt_output_root() -> Path:
    attempt = os.environ.get("CCSAM_CROSS_DATASET_ATTEMPT")
    if attempt is None:
        return ROOT
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", attempt):
        raise ValueError("CCSAM_CROSS_DATASET_ATTEMPT must be a simple lowercase directory name")
    path = ROOT / "runs" / attempt
    if path.resolve() != path:
        raise ValueError(f"Experiment output path traverses a link: {path}")
    return path


OUTPUT_ROOT = attempt_output_root()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "--no-optional-locks", *args], cwd=REPO, text=True, encoding="utf-8"
    ).strip()


@dataclass(frozen=True)
class Experiment:
    protocol: dict
    workers: int
    branch: str
    head: str
    source_hashes: dict[str, str]
    dirty_sources: list[str]

    @property
    def datasets(self):
        return self.protocol["datasets"]

    @property
    def seeds(self):
        return self.protocol["seeds"]

    @property
    def arms(self):
        return tuple(self.protocol["arms"])

    def jobs(self):
        for name in self.datasets:
            for seed in self.seeds:
                yield name, seed

    def config(self, phase: str, **values) -> dict:
        return {
            "phase": phase, "git_branch": self.branch, "git_head": self.head,
            "uncommitted_sources": self.dirty_sources, "source_sha256": self.source_hashes,
            "protocol": self.protocol, "num_workers": self.workers,
            "output_root": str(OUTPUT_ROOT), **values,
        }


def load_experiment(workers: int | None) -> Experiment:
    branch = git("branch", "--show-current")
    if branch != BRANCH or Path(git("rev-parse", "--show-toplevel")).resolve() != REPO:
        raise RuntimeError(f"Expected {REPO} on {BRANCH}, observed branch {branch}")
    protocol = read_json(PROTOCOL)
    if protocol["implementation_branch"] != BRANCH:
        raise RuntimeError("Frozen protocol names a different implementation branch")
    if workers is None:
        workers = int(protocol["resource_target"]["num_workers_default"])
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    sources = sorted([*ROOT.glob("*.py"), PROTOCOL, ROOT / "PROTOCOL.md",
                      ROOT / "diagnostics/amp_gradient_probe/gradient_evidence.py"])
    relative = [path.relative_to(REPO).as_posix() for path in sources]
    dirty = git("status", "--porcelain=v1", "--untracked-files=all", "--", *relative)
    return Experiment(protocol, workers, branch, git("rev-parse", "HEAD"),
                      dict(zip(relative, map(sha256_file, sources))), dirty.splitlines())


def environment() -> dict:
    import PIL
    import pyarrow
    import torchvision

    selected = initialize_cuda(DEFAULT_CUDA_DEVICE)
    return {
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "torchvision": str(torchvision.__version__), "pyarrow": pyarrow.__version__,
        "pillow": PIL.__version__, "cudnn": torch.backends.cudnn.version(),
        **selected.diagnostics,
    }


def cuda_device() -> torch.device:
    selected = initialize_cuda(DEFAULT_CUDA_DEVICE)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return selected.device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch: int, workers: int, seed: int, *, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, num_workers=workers, drop_last=False,
        generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker,
        pin_memory=True, persistent_workers=workers > 0,
    )


def make_sgd(parameters, config: dict):
    return torch.optim.SGD(parameters, lr=config["learning_rate"], momentum=config["momentum"],
                           weight_decay=config["weight_decay"], nesterov=config["nesterov"])


def run_path(phase: str, dataset: str, seed: int, arm: str | None = None) -> Path:
    path = OUTPUT_ROOT / phase / dataset / f"seed_{seed}"
    return path / arm if arm is not None else path


def run_log(directory: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with (directory / "run.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def save_status(directory: Path, status: dict) -> None:
    partial = directory / "status.json.writing"
    write_json(partial, status)
    partial.replace(directory / "status.json")


@contextmanager
def record_run(directory: Path, config: dict, *, device: torch.device | None = None):
    """Reserve a run directory and preserve the recorded state on failure."""
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    status = {"state": "running", "started_at": now(), "official_test_access": False}
    save_status(directory, status)
    run_log(directory, f"Starting {config['phase']} in {directory}")
    pending_peak = None
    try:
        if device is not None:
            status["cuda_diagnostics"] = {}
            selected = initialize_cuda(device, status["cuda_diagnostics"])
            selected.reset_peak()
            pending_peak = selected
            save_status(directory, status)
        yield status
        if pending_peak is not None:
            selected, pending_peak = pending_peak, None
            status.update(selected.peak())
        run_log(directory, "Completed")
        status["artifacts"] = {
            path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.relative_to(directory).as_posix() != "status.json"
        }
        status["state"] = "completed"
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        run_log(directory, traceback.format_exc())
        if pending_peak is not None:
            try:
                status.update(pending_peak.peak())
            except Exception as stats_error:
                status["cuda_statistics_error"] = f"{type(stats_error).__name__}: {stats_error}"
                run_log(directory, traceback.format_exc())
        raise
    finally:
        status["ended_at"] = now()
        save_status(directory, status)


def require_complete(directory: Path, *, phase: str, expected: dict | None = None,
                     verify_hashes: bool = True) -> dict:
    status = read_json(directory / "status.json")
    if status.get("state") != "completed":
        raise RuntimeError(f"Run is incomplete in {directory}: {status.get('state')}")
    if any(status.get(key) != value for key, value in (expected or {}).items()):
        raise RuntimeError(f"Run provenance mismatch in {directory}: expected {expected}")
    config = read_json(directory / "config.json")
    if config.get("phase") != phase:
        raise RuntimeError(f"Wrong phase in {directory}: {config.get('phase')}")
    artifacts = status.get("artifacts", {})
    if not {"config.json", "run.log"} <= artifacts.keys():
        raise RuntimeError(f"Missing artifact hashes in {directory}")
    if verify_hashes:
        for filename, expected_sha in artifacts.items():
            path = (directory / filename).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected_sha:
                raise RuntimeError(f"Artifact SHA256 mismatch: {path}")
    return status


def checked_artifact(directory: Path, status: dict, filename: str) -> str:
    expected = status["artifacts"][filename]
    if sha256_file(directory / filename) != expected:
        raise RuntimeError(f"Artifact SHA256 mismatch: {directory / filename}")
    return expected
