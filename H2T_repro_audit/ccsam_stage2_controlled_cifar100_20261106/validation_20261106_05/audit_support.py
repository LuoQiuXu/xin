"""Shared read-only audit helpers. All new artifacts use exclusive creation."""
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
import ccsam_common as c
import train_ccsam as production

BRANCH = "experiment/ccsam-gradient-numerics-fix-20261106"
HEAD = "6dd45e70040b5fd14a6d563a9fa19d0e52ac4753"
RTOL, ATOL = 1e-5, 1e-7


def now():
    return datetime.now().astimezone().isoformat()


def json_write(name, value):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def json_read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def record(path):
    return dict(bytes=path.stat().st_size, sha256=c.sha256_file(path))


def environment():
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0), device_count=torch.cuda.device_count(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        default_dtype=str(torch.get_default_dtype()), autocast_enabled=torch.is_autocast_enabled("cuda"))


def assert_branch():
    assert git("branch", "--show-current").strip() == BRANCH
    assert git("rev-parse", "HEAD").strip() == HEAD
    assert not (ROOT / "results").exists(), "Formal results must not be created"


def check_protection(allow_production=False):
    baseline = json_read("PROTECTION_BEFORE.json")
    changed = []
    for name, expected in baseline["protected_files"].items():
        path = REPO / name
        if not path.is_file() or record(path) != expected:
            changed.append(name)
    if not allow_production and record(ROOT / "train_ccsam.py") != baseline["production_before"]:
        changed.append("train_ccsam.py")
    assert not changed, changed
    assert_branch()
    return dict(protected_file_count=len(baseline["protected_files"]), changed_protected_files=[])


def csv_open(name, fields):
    stream = (OUT / name).open("x", encoding="utf-8", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    return stream, writer


def array(t):
    return t.detach().cpu().contiguous().numpy().copy()


def save_tensors(name, tensors):
    values = {k: array(v) if isinstance(v, torch.Tensor) else np.asarray(v) for k, v in tensors.items()}
    with (OUT / name).open("xb") as stream:
        np.savez_compressed(stream, **values)
    json_write(name + ".json", dict(file=name, **record(OUT / name),
        tensors={k:dict(shape=list(v.shape), dtype=str(v.dtype), sha256=c.hash_arrays(v))
                 for k,v in values.items()}))


def install_guard():
    return c.install_test_guard(c.DEFAULT_DATA_ROOT)


def load_inputs():
    manifest = c.read_json(ROOT / "prepared/prepared_manifest.json")
    assert manifest["clean_stage1_sha256"] == c.EXPECTED_CLEAN_SHA256
    assert manifest["official_TEST_used"] is False
    images = np.load(ROOT / "prepared/train_images.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(ROOT / "prepared/train_labels.npy", allow_pickle=False)
    return manifest, images, labels
