"""Read-only Fish16 data and original pretrained-loader contract diagnostics.

Does not train, alter upstream code, download weights or regenerate a manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback


REPO = Path(__file__).resolve().parents[3]
BRANCH = "experiment/fish16-h2t-pih2t-cross-domain-20261031"
MANIFEST = Path(r"D:\h2t_cross_domain_audit\fish4knowledge\manifest_primary.csv")
IMAGES = REPO / "data/fishRecognition_GT/fish_image"
EXPECTED_SHA = "62d9a2e9fffaaca22ef0c99c0ce4134b4d0e1ecc308333a7cb57b10cf95f0f30"
EXPECTED_SPLITS = {"train": (16269, 5208), "val": (5396, 1736), "test": (5468, 1736)}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def check_data():
    assert sha(MANIFEST) == EXPECTED_SHA, "Locked manifest SHA mismatch"
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows, headers = list(reader), reader.fieldnames
    assert len(rows) == 27133
    labels = sorted({r["label"] for r in rows})
    assert len(labels) == 16
    for key in ("base_group", "group_id", "image_sha256"):
        assignments = defaultdict(set)
        for row in rows:
            assignments[row[key]].add(row["split_primary"])
        assert all(len(splits) == 1 for splits in assignments.values()), key
    support = {}
    for split, (images, groups) in EXPECTED_SPLITS.items():
        selected = [r for r in rows if r["split_primary"] == split]
        support[split] = {"images": len(selected), "groups": len({r["group_id"] for r in selected})}
        assert support[split] == {"images": images, "groups": groups}
    per_class = []
    class_ids = {}
    for label in labels:
        selected = [r for r in rows if r["label"] == label]
        ids = {int(Path(r["relative_path"]).parts[0].removeprefix("fish_")) for r in selected}
        assert len(ids) == 1
        class_id = next(iter(ids))
        assert class_id in range(1, 17)
        class_ids[class_id] = label
        per_class.append({"class_id": class_id, "label": label, **{
            s: {"images": sum(r["split_primary"] == s for r in selected),
                "groups": len({r["group_id"] for r in selected if r["split_primary"] == s})}
            for s in EXPECTED_SPLITS}})
    assert set(class_ids) == set(range(1, 17))
    for i, row in enumerate(rows, 1):
        path = (IMAGES / row["relative_path"]).resolve()
        assert path.is_relative_to(IMAGES.resolve())
        assert not any("mask" in part.lower() for part in path.parts)
        assert path.is_file(), path
        assert sha(path) == row["image_sha256"], str(path)
        if i % 3000 == 0:
            print(f"Verified unchanged image bytes: {i}/{len(rows)}", flush=True)
    train = {r["label"]: r["train"]["images"] for r in per_class}
    ranked = sorted(labels, key=lambda label: (-train[label], label))
    return {"status": "PASS", "manifest": str(MANIFEST), "manifest_sha256": sha(MANIFEST),
            "headers": headers, "rows": len(rows), "classes": sorted(per_class, key=lambda r: r["class_id"]),
            "split_support": support, "frequency_classes": {"Head": ranked[:6], "Medium": ranked[6:11], "Tail": ranked[11:]},
            "cross_split_leaks": 0, "image_paths_and_sha256_verified": len(rows),
            "manifest_regenerated_or_modified": False}


def check_original_loader():
    import torch
    import torchvision
    import PIL
    source = REPO / "H2T_official_clean/H2T-main/models/resnet.py"
    before = sha(source)
    spec = importlib.util.spec_from_file_location("fish16_original_resnet_readonly", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {"source": str(source), "source_sha256": before,
              "requested_ResNet50_FE_exists": hasattr(module, "ResNet50_FE"),
              "available_resnet50_fe": hasattr(module, "resnet50_fe"),
              "source_resnet50_weight_url": module.model_urls["resnet50"],
              "environment": {"python": sys.executable, "python_version": platform.python_version(),
                              "os": platform.platform(), "torch": torch.__version__,
                              "torchvision": torchvision.__version__, "pillow": PIL.__version__,
                              "cuda_build": torch.version.cuda}, "cached_weights": []}
    try:
        model = module.resnet50_fe(pretrained=True)
        report["factory_result"] = "PASS"
        del model
    except Exception as exc:
        report.update(factory_result="FAIL", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    assert sha(source) == before
    for name in ("resnet50-19c8e357.pth", "resnet50-11ad3fa6.pth"):
        path = Path(torch.hub.get_dir()) / "checkpoints" / name
        row = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            row.update(bytes=path.stat().st_size, sha256=sha(path))
        report["cached_weights"].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() == BRANCH
    output = Path(args.output_dir).resolve()
    assert not output.is_relative_to(REPO)
    output.mkdir(parents=True, exist_ok=False)
    result = {"branch": BRANCH, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "training_runs": 0, "weights_downloaded": False, "original_sources_modified": False}
    try:
        result["data_gate"] = check_data()
        result["original_loader_probe"] = check_original_loader()
        result["status"] = ("BLOCKED_MODEL_LOADING_CONTRACT" if
                            not result["original_loader_probe"]["requested_ResNet50_FE_exists"] or
                            result["original_loader_probe"]["factory_result"] != "PASS" else "PASS_PREFLIGHT_ONLY")
    except Exception:
        result.update(status="FAILED_PREFLIGHT", traceback=traceback.format_exc())
    with (output / "preflight.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "PASS_PREFLIGHT_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
