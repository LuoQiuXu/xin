"""Strict CPU availability check for the pre-existing TRAIN-validation-selected base."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from coverage import ROOT, TASK, CLEAN, DEFAULT_OUT, guard_io, sha, write_json

CHECKPOINT = CLEAN / "stage1/checkpoints/clean_stage1_best.pth"
EXPECTED_SHA256 = "7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88"
MODEL_SOURCE = CLEAN / "runtime/models/resnet_cifar.py"


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def preprocessing():
    path = CLEAN / "runtime/datasets/cifar100.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    n = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "eval_transform" for t in n.targets))
    transforms = n.value.args[0].elts
    assert [n.func.attr for n in transforms] == ["ToTensor", "Normalize"]
    return [list(ast.literal_eval(x)) for x in transforms[1].args]


def load_models():
    import torch
    assert sha(CHECKPOINT) == EXPECTED_SHA256
    types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64, np.bool_]
    numeric = [np.dtype, np.ndarray, np._core.multiarray.scalar, np._core.multiarray._reconstruct]
    numeric += [type(np.dtype(t)) for t in types]
    with torch.serialization.safe_globals(numeric):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 184 and checkpoint["seed"] == 42
    assert checkpoint["selection_partition"] == "official TRAIN holdout"
    def strip(state):
        assert all(k.startswith("module.") for k in state)
        return {k[7:]: v for k, v in state.items()}
    backbone = strip(checkpoint["state_dict_model"])
    classifier = strip(checkpoint["state_dict_classifier"])
    assert not any(".shortcut.0.weight" in k or k.startswith("conv1d.") for k in backbone)
    spec = importlib.util.spec_from_file_location("semantic_clean_original_model", MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bb = module.ResNet_fe(module.BasicBlock, [5, 5, 5], feature_tuning=False)
    fc = module.Classifier(feat_in=64, num_classes=100)
    bb.load_state_dict(backbone, strict=True)
    fc.load_state_dict(classifier, strict=True)
    bb.requires_grad_(False).eval()
    fc.requires_grad_(False).eval()
    assert fc.fc.weight.shape == (100, 64) and fc.fc.bias.shape == (100,)
    assert all(torch.isfinite(v).all() for v in list(backbone.values()) + list(classifier.values()))
    return bb, fc, checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(TASK)
    if (out / "CHECKPOINT_AVAILABILITY.json").exists() or (out / "CHECKPOINT_GUARD.json").exists():
        raise FileExistsError("Refuse to overwrite checkpoint evidence")
    guard = guard_io()
    bb, fc, checkpoint = load_models()
    protocol_path = CLEAN / "clean_stage1_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["passed"] and not protocol["TEST_loaded"] and not protocol["TEST_selected"]
    assert protocol["best_checkpoint_sha256"] == EXPECTED_SHA256
    source_paths = [MODEL_SOURCE, CLEAN / "runtime/datasets/cifar100.py", CLEAN / "common.py",
                    CLEAN / "clean_stage1.py", CLEAN / "clean_stage1_protocol.json",
                    CLEAN / "CLEAN_STAGE1_PROTOCOL.md", CLEAN / "clean_stage1_effective.yaml",
                    CLEAN / "split_indices.npz"]
    write_json(out / "CHECKPOINT_AVAILABILITY.json", dict(
        available=True, checkpoint_path=CHECKPOINT.relative_to(ROOT).as_posix(), checkpoint_sha256=EXPECTED_SHA256,
        epoch=184, seed=42, selection_partition=checkpoint["selection_partition"], selection_metric=checkpoint["selection_metric"],
        historical_validation_Overall=checkpoint["validation"]["Overall"],
        historical_selection_notes="Max over 200 epochs on reused 1980-image TRAIN holdout, 99 observed classes; no official TEST selection",
        strict_loading=True, weights_only=True, map_location="cpu", architecture="ResNet32 Option-A, feature_tuning=False, original 100-way biased linear head",
        feature_dimension=64, weight_shape=list(fc.fc.weight.shape), bias_shape=list(fc.fc.bias.shape), dtype=str(fc.fc.weight.dtype),
        normalization=preprocessing(), backbone_state_sha256=state_hash(bb.state_dict()), classifier_state_sha256=state_hash(fc.state_dict()),
        CPU_model_forward_calls=0, GPU_model_forward_calls=0, trained_this_run=False, BN_updated=False,
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
        author_PI_H2T_evaluated=False, use_status="Availability only; feature extraction waits for coverage-gate resolution"))
    assert not any(guard.values())
    write_json(out / "CHECKPOINT_GUARD.json", guard)
    print("CHECKPOINT PASS: Clean Stage-I epoch184/seed42, strict CPU load, TRAIN-validation selection; zero model forwards.")


if __name__ == "__main__":
    main()
