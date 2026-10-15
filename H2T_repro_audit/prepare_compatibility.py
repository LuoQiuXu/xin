"""Prepare an isolated, minimal compatibility copy of the pinned H2T archive."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

BASE = Path(__file__).resolve().parent
OFFICIAL = BASE / "official"
RUNTIME = BASE / "runtime"
EVIDENCE = BASE / "evidence"


def write_diff():
    patches = []
    changed = []
    for source in sorted(OFFICIAL.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(OFFICIAL)
        target = RUNTIME / relative
        if source.read_bytes() == target.read_bytes():
            continue
        changed.append(str(relative).replace("\\", "/"))
        before = source.read_text(encoding="utf-8").splitlines()
        after = target.read_text(encoding="utf-8").splitlines()
        patches.extend(difflib.unified_diff(
            before, after, fromfile="a/" + relative.as_posix(),
            tofile="b/" + relative.as_posix(), lineterm=""))
    (EVIDENCE / "compatibility.diff").write_text(
        "\n".join(patches) + "\n", encoding="utf-8")
    files = {p.relative_to(RUNTIME).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(RUNTIME.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and "saved" not in p.parts}
    (EVIDENCE / "runtime_manifest.json").write_text(json.dumps({
        "modified_official_files": changed, "files": files,
        "methods_unchanged": (OFFICIAL / "methods.py").read_bytes() ==
                             (RUNTIME / "methods.py").read_bytes(),
        "datasets_cifar100_unchanged": (OFFICIAL / "datasets/cifar100.py").read_bytes() ==
                                      (RUNTIME / "datasets/cifar100.py").read_bytes(),
        "stage1_yaml_unchanged": (OFFICIAL / "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml").read_bytes() ==
                                 (RUNTIME / "config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml").read_bytes(),
        "stage2_yaml_unchanged": (OFFICIAL / "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml").read_bytes() ==
                                 (RUNTIME / "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml").read_bytes(),
    }, indent=2), encoding="utf-8")
    print("Compatibility files:", changed)


def patch_checkpoint_loading():
    for filename in ["train_stage1.py", "train_stage2.py", "eval-modifed.py"]:
        path = RUNTIME / filename
        text = path.read_text(encoding="utf-8")
        first = "torch.load(config.resume)"
        second = "torch.load(config.resume, map_location=loc)"
        assert text.count(first) == 1 and text.count(second) == 1
        text = text.replace(first, "torch.load(config.resume, weights_only=False)")
        text = text.replace(second, "torch.load(config.resume, map_location=loc, weights_only=False)")
        path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    if RUNTIME.exists():
        raise SystemExit("Runtime already exists. Refusing to overwrite experiments.")
    shutil.copytree(OFFICIAL, RUNTIME, ignore=shutil.ignore_patterns("__pycache__"))
    model = RUNTIME / "models/resnet_cifar.py"
    text = model.read_text(encoding="utf-8")
    old = "class ResNet_fe(nn.Module):\n\n    def __init__(self, block, num_blocks):"
    assert text.count(old) == 1
    text = text.replace(old, old[:-2] + ", feature_tuning=False):")
    old = "bias=False, feature_tuning=False)"
    assert text.count(old) == 1
    text = text.replace(old, "bias=False)")
    model.write_text(text, encoding="utf-8", newline="\n")
    util = RUNTIME / "utils/util.py"
    text = util.read_text(encoding="utf-8")
    assert text.count("from pathlib2 import Path") == 1
    util.write_text(text.replace("from pathlib2 import Path", "from pathlib import Path"),
                    encoding="utf-8", newline="\n")
    patch_checkpoint_loading()
    write_diff()
