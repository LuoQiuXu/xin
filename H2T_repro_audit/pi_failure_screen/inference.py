"""Evaluate existing matched checkpoints. Never instantiate a dataset wrapper or train.

prepare: CPU strict loading, TRAIN-only holdout/perturbations and visual contact sheet.
infer: requires a completed visual audit, writes predictions before any manifest.
Historical model classes are imported unchanged from their own saved source files.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
OUT = TASK / "results"
CACHE = OUT / "primary_cache"
CONDITIONS = ("clean", "noise_003", "noise_008", "blur_05", "blur_10")
SEED = 20261028
SPLIT = ROOT / "H2T_repro_audit/clean_vs_stability_runs/clean_s1_seed42_s2_124_125_20261014"
PI_SOURCE = "H2T/saved/cifar100_imb001_stage1_ce_mixup_pif_202608082320/codes"
H2T_SOURCE = "H2T_repro_audit/bn_audit_runs/fixed_bn_20261011_2255/runtime"
PAIRS = {
    "h2t": dict(reference="h2t_no_fusion", candidate="h2t", stage1="h2t_stage1",
                source=H2T_SOURCE, pif=False, stage1_epoch=198, stage2_seed=123),
    "pi_local": dict(reference="pi_no_fusion", candidate="pi_local", stage1="pi_stage1",
                     source=PI_SOURCE, pif=True, stage1_epoch=164, stage2_seed=42),
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def array_hash(array):
    a = np.asarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def corrupt(image, raw_id, condition):
    """Pixel space, before ToTensor/Normalize; exact 3x3 reflect-padded blur."""
    x = np.asarray(image)
    if x.shape != (32, 32, 3) or x.dtype != np.uint8:
        raise ValueError("expected raw uint8 HWC CIFAR image")
    if condition == "clean":
        return x.copy()
    value = x.astype(np.float64) / 255.0
    if condition in ("noise_003", "noise_008"):
        sigma = .03 if condition == "noise_003" else .08
        # Common standard noise for both severities; seed depends only on raw ID.
        rng = np.random.default_rng(np.random.SeedSequence([SEED, int(raw_id)]))
        value = value + sigma * rng.standard_normal(x.shape)
    elif condition in ("blur_05", "blur_10"):
        sigma = .5 if condition == "blur_05" else 1.0
        k = np.exp(-np.arange(-1, 2, dtype=np.float64) ** 2 / (2 * sigma ** 2))
        k /= k.sum()
        padded = np.pad(value, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        value = sum(k[i] * k[j] * padded[i:i+32, j:j+32] for i in range(3) for j in range(3))
    else:
        raise ValueError(condition)
    return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)


def install_guard():
    events = {"official_test_open_attempts": 0, "historical_write_attempts": 0}

    def hook(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.name.lower() == "test" and "cifar" in p.as_posix().lower():
            events["official_test_open_attempts"] += 1
            raise PermissionError("Official CIFAR TEST is forbidden")
        mode = args[1] or ""
        flags = args[2] if len(args) > 2 else 0
        writing = any(c in str(mode) for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR)
        if writing and p.is_relative_to(ROOT) and not p.is_relative_to(TASK):
            events["historical_write_attempts"] += 1
            raise PermissionError(f"Historical write forbidden: {p}")

    sys.addaudithook(hook)
    return events


def load_checkpoint(info):
    import torch
    p = ROOT / info["path"]
    if sha(p) != info["sha256"]:
        raise ValueError(f"checkpoint SHA changed: {p}")
    types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64, np.bool_]
    primitives = [np.dtype, np.ndarray, np._core.multiarray.scalar,
                  np._core.multiarray._reconstruct] + [type(np.dtype(t)) for t in types]
    with torch.serialization.safe_globals(primitives):
        return torch.load(p, map_location="cpu", weights_only=True)


def states(checkpoint):
    def strip(state):
        if all(k.startswith("module.") for k in state):
            return {k[7:]: v for k, v in state.items()}
        if any(k.startswith("module.") for k in state):
            raise ValueError("mixed DataParallel prefixes")
        return state
    if "state_dict_model" in checkpoint:
        return strip(checkpoint["state_dict_model"]), strip(checkpoint["state_dict_classifier"])
    return strip(checkpoint["backbone"]), strip(checkpoint["classifier"])


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(array_hash(tensor.detach().cpu().contiguous().numpy()).encode())
    return h.hexdigest()


def build_model(pair_id, checkpoint):
    spec = PAIRS[pair_id]
    p = ROOT / spec["source"] / "models/resnet_cifar.py"
    module_spec = importlib.util.spec_from_file_location("screen_original_" + pair_id, p)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    bb = module.ResNet_fe(module.BasicBlock, [5, 5, 5], feature_tuning=spec["pif"])
    fc = module.Classifier(feat_in=64, num_classes=100)
    b, c = states(checkpoint)
    if any(".shortcut.0.weight" in k for k in b):
        raise ValueError("expected the locked Option-A architecture")
    bb.load_state_dict(b, strict=True)
    fc.load_state_dict(c, strict=True)
    bb.requires_grad_(False).eval()
    fc.requires_grad_(False).eval()
    return bb, fc


def normalization(source):
    """Extract only literal Normalize arguments; never construct CIFAR100_LT."""
    tree = ast.parse((ROOT / source / "datasets/cifar100.py").read_text(encoding="utf-8-sig"))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "eval_transform" for t in n.targets)]
    assert len(assignments) == 1
    transforms = assignments[0].value.args[0].elts
    assert [n.func.attr for n in transforms] == ["ToTensor", "Normalize"]
    return [list(ast.literal_eval(x)) for x in transforms[1].args]


def prepare():
    import torch
    meta = json.loads((OUT / "CHECKPOINT_METADATA_NUMPY_COMPAT.json").read_text(encoding="utf-8"))
    infos = {r["id"]: r for r in meta["checkpoints"]}
    registry, pair_checks, sources = [], {}, {}
    for pair_id, spec in PAIRS.items():
        stage1 = load_checkpoint(infos[spec["stage1"]])
        s1b, _ = states(stage1)
        assert int(stage1["epoch"]) == spec["stage1_epoch"]
        loaded = {}
        for model in (spec["reference"], spec["candidate"]):
            ck = load_checkpoint(infos[model])
            assert int(ck["epoch"]) == 10
            if spec["pif"]:
                assert ck["uses_pif"] is True and ck["method"] == "pi_recipient_only"
            bb, fc = build_model(pair_id, ck)
            b, _ = states(ck)
            parameter_keys = [k for k in b if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
            assert set(b) == set(s1b)
            assert all(torch.equal(b[k], s1b[k]) for k in parameter_keys)
            loaded[model] = b
            registry.append(dict(model=model, pair=pair_id, available=True,
                checkpoint_path=infos[model]["path"], checkpoint_sha256=infos[model]["sha256"],
                stage1_path=infos[spec["stage1"]]["path"], stage1_sha256=infos[spec["stage1"]]["sha256"],
                stage1_epoch=spec["stage1_epoch"], stage2_epoch=10, stage2_seed=spec["stage2_seed"],
                stage2_type="no_fusion" if model == spec["reference"] else ("local_ratio_only" if spec["pif"] else "H2T_batch_shared"),
                architecture="ResNet32 Option-A + " + ("historical 25-parameter PIF + " if spec["pif"] else "") + "100-way FC",
                classes=100, model_source=spec["source"] + "/models/resnet_cifar.py",
                normalization=normalization(spec["source"]), strict_state_load=True,
                parameter_count=sum(p.numel() for p in bb.parameters()),
                pif_parameters=sum(p.numel() for n, p in bb.named_parameters() if n.startswith("conv1d.")),
                fc_parameters=sum(p.numel() for p in fc.parameters()),
                full_backbone_state_sha256=state_hash(b), classifier_state_sha256=state_hash(fc.state_dict()),
                all_parameters_match_stage1=True, author_full_PI_implementation_verified=False,
                selection="Stage-I historical best on official TEST; Stage-II fixed epoch10, no shift selection"))
        left, right = (loaded[spec[x]] for x in ("reference", "candidate"))
        equal = all(torch.equal(left[k], right[k]) for k in left)
        assert equal, "The paired evaluation backbones/BN must be identical"
        pair_checks[pair_id] = dict(reference=spec["reference"], candidate=spec["candidate"],
            same_stage1=True, same_eval_architecture=True, identical_final_backbone_and_BN=True,
            stage1_sha256=infos[spec["stage1"]]["sha256"],
            compared_state_tensors=len(left), classifier_only_difference=True)
        for rel in [spec["source"] + "/models/resnet_cifar.py", spec["source"] + "/datasets/cifar100.py"]:
            sources[rel] = sha(ROOT / rel)
    assert registry[0]["normalization"] == registry[-1]["normalization"]
    # Stage-I training subset provenance: both source wrappers use rand_number=0.
    for spec in PAIRS.values():
        tree = ast.parse((ROOT / spec["source"] / "datasets/cifar100.py").read_text(encoding="utf-8-sig"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "IMBALANCECIFAR100"]
        assert len(calls) == 1
        kwargs = {k.arg: k.value for k in calls[0].keywords}
        assert ast.literal_eval(kwargs["rand_number"]) == 0 and ast.literal_eval(kwargs["train"]) is True
    raw_path = ROOT / "H2T/data/cifar100/cifar-100-python/train"
    assert sha(raw_path) == "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"
    with raw_path.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    images = np.asarray(raw["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(raw["fine_labels"], dtype=np.int64)
    assert images.shape == (50000, 32, 32, 3) and np.array_equal(np.bincount(labels), np.full(100, 500))
    counts = np.array([int(500 * (.01 ** (c / 99))) for c in range(100)], dtype=np.int64)
    rng = np.random.RandomState(0)
    selected = []
    for c, n in enumerate(counts):
        ids = np.flatnonzero(labels == c)
        rng.shuffle(ids)
        selected.extend(ids[:n])
    selected = np.asarray(selected, dtype=np.int64)
    with np.load(SPLIT / "split_indices.npz", allow_pickle=False) as z:
        assert np.array_equal(z["train_raw_indices"], selected)
        ids = z["validation_raw_indices"].copy()
    assert len(ids) == len(np.unique(ids)) == 1980 and np.intersect1d(ids, selected).size == 0
    x, y = images[ids], labels[ids]
    for prefix, ii in [("train", selected), ("validation", ids)]:
        assert np.array_equal(np.load(SPLIT / (prefix + "_images.npy"), allow_pickle=False, mmap_mode="r"), images[ii])
        assert np.array_equal(np.load(SPLIT / (prefix + "_labels.npy"), allow_pickle=False), labels[ii])
    training_pixels = {hashlib.sha256(a.tobytes()).digest() for a in images[selected]}
    assert all(hashlib.sha256(a.tobytes()).digest() not in training_pixels for a in x)
    with (CACHE / "holdout_pixels.npz").open("xb") as f:
        np.savez_compressed(f, ids=ids, labels=y, images=x, class_counts=counts)
    conditions = {condition: np.stack([corrupt(a, i, condition) for a, i in zip(x, ids)]) for condition in CONDITIONS}
    with (CACHE / "condition_pixels.npz").open("xb") as f:
        np.savez_compressed(f, **conditions)
    write_json(OUT / "CHECKPOINT_AVAILABILITY.json", dict(models=registry, pairs=pair_checks,
               source_sha256=sources, reused_saved_model_definitions=True))
    with (OUT / "CHECKPOINT_AVAILABILITY.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(registry[0]), lineterminator="\n")
        w.writeheader(); w.writerows(registry)
    write_json(OUT / "DATA_ATTESTATION.json", dict(raw_train_path=str(raw_path.relative_to(ROOT)),
        raw_train_sha256=sha(raw_path), split_sha256=sha(SPLIT / "split_indices.npz"),
        source_training_counts=counts.tolist(), LT_images=len(selected), holdout_images=len(ids),
        observed_classes=int(np.unique(y).size), missing_classes=np.flatnonzero(np.bincount(y, minlength=100) == 0).tolist(),
        holdout_class_counts=np.bincount(y, minlength=100).tolist(), raw_id_overlap=0, exact_pixel_overlap=0,
        historical_train_cache_matches_reconstructed_subset=True, historical_validation_cache_matches_raw_pixels=True,
        holdout_status="reused TRAIN development holdout; not independent confirmation",
        ids_sha256=array_hash(ids), labels_sha256=array_hash(y), clean_pixels_sha256=array_hash(x),
        condition_pixel_sha256={k: array_hash(v) for k, v in conditions.items()},
        condition_cache_sha256=sha(CACHE / "condition_pixels.npz"), no_official_test_read=True,
        corruption_seed=SEED, noise_pixel_sigma=[.03, .08], noise_seed_unit="raw image ID",
        blur_kernel=3, blur_sigma=[.5, 1.0], blur_padding="reflect", pixel_quantization="clip[0,1], round-to-nearest uint8",
        stage1_configs="Both historical Stage-I CIFAR100-LT configs specify IF=.01/100 classes; source wrappers specify seed0"))
    write_json(OUT / "PRE_REGISTERED_SCREEN_RULES.json", dict(
        written_before_real_model_forward=True, comparisons=list(PAIRS), conditions=list(CONDITIONS),
        families=dict(noise=["noise_003", "noise_008"], blur=["blur_05", "blur_10"]),
        groups=["all", "many", "medium", "few"],
        primary_effect="DiD=(candidate-reference)_shift-(candidate-reference)_clean",
        minimum_effect_pp=.5, minimum_excess_net_loss=3, minimum_distinct_negative_DiD_images=3,
        candidate_scenario="Same pair and group must meet all thresholds at BOTH severities of one prespecified family",
        frequency_specific="Additionally group DiD <= Overall DiD - .5pp at both severities",
        inference_limit="screening signal only; one Stage-I and one Stage-II seed per pair, reused holdout, 12 frequency-specific family/pair checks",
        no_new_method=True, no_training=True, no_threshold_search=True))
    make_preview(x, y, ids, counts, conditions)
    print("PREPARE PASS: four strict-loaded models; two matched pairs; 1980 disjoint TRAIN holdout images.", flush=True)


def make_preview(x, y, ids, counts, conditions):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(SEED)
    indices = np.concatenate([rng.choice(np.flatnonzero(mask[y]), 3, replace=False)
        for mask in [counts > 100, (counts >= 20) & (counts <= 100), counts < 20]])
    canvas = Image.new("RGB", (5 * 160, 40 + len(indices) * 174), "white")
    draw = ImageDraw.Draw(canvas)
    for col, condition in enumerate(CONDITIONS):
        draw.text((col * 160 + 8, 8), condition, fill="black")
        for row, index in enumerate(indices):
            im = Image.fromarray(conditions[condition][index]).resize((144, 144), Image.Resampling.NEAREST)
            canvas.paste(im, (col * 160 + 8, 40 + row * 174))
            draw.text((col * 160 + 8, 184 + row * 174), f"raw {ids[index]} / class {y[index]}", fill="black")
    canvas.save(OUT / "VISUAL_CORRUPTION_AUDIT.png")
    write_json(OUT / "VISUAL_SAMPLE_IDS.json", dict(seed=SEED, raw_ids=ids[indices].tolist(),
        labels=y[indices].tolist(), selection="3 random images per frequency group, fixed before predictions"))


def infer():
    audit = json.loads((OUT / "VISUAL_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["approved_conditions"] == list(CONDITIONS) and audit["before_model_inference"] is True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torchvision
    from PIL import Image
    assert torch.cuda.is_available(), "GPU inference unavailable; do not silently change protocol"
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    availability = json.loads((OUT / "CHECKPOINT_AVAILABILITY.json").read_text(encoding="utf-8"))
    infos = {x["id"]: x for x in json.loads((OUT / "CHECKPOINT_METADATA_NUMPY_COMPAT.json").read_text(encoding="utf-8"))["checkpoints"]}
    for rel, expected in availability["source_sha256"].items():
        assert sha(ROOT / rel) == expected
    attestation = json.loads((OUT / "DATA_ATTESTATION.json").read_text(encoding="utf-8"))
    assert sha(CACHE / "condition_pixels.npz") == attestation["condition_cache_sha256"]
    with np.load(CACHE / "holdout_pixels.npz", allow_pickle=False) as z:
        ids, labels, counts = z["ids"], z["labels"], z["class_counts"]
    with np.load(CACHE / "condition_pixels.npz", allow_pickle=False) as z:
        pixels = {c: z[c] for c in CONDITIONS}
    runtime = dict(python=sys.executable, torch=torch.__version__, torchvision=torchvision.__version__,
        numpy=np.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        GPU=torch.cuda.get_device_name(0), total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        batch_size=32, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        screen_rules_sha256=sha(OUT / "PRE_REGISTERED_SCREEN_RULES.json"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        autocast=False, training=False, optimizer_created=False, backward_calls=0)
    records, common_hashes, predictions = [], {}, {}
    destination = OUT / "predictions"
    destination.mkdir(exist_ok=False)
    for row in availability["models"]:
        model = row["model"]
        bb, fc = build_model(row["pair"], load_checkpoint(infos[model]))
        assert state_hash(bb.state_dict()) == row["full_backbone_state_sha256"]
        bb, fc = bb.cuda().eval(), fc.cuda().eval()
        assert not any(isinstance(m, torch.nn.modules.dropout._DropoutNd)
                       for m in list(bb.modules()) + list(fc.modules()))
        assert not any(p.requires_grad for p in list(bb.parameters()) + list(fc.parameters()))
        mean, std = row["normalization"]
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
        before = [state_hash(bb.state_dict()), state_hash(fc.state_dict())]
        predictions[model] = {}
        for condition in CONDITIONS:
            assert not any(m.training for m in list(bb.modules()) + list(fc.modules()))
            torch.cuda.reset_peak_memory_stats()
            tensor_hash = hashlib.sha256()
            output = []
            with torch.inference_mode():
                for start in range(0, len(ids), 32):
                    batch = torch.stack([transform(Image.fromarray(a)) for a in pixels[condition][start:start+32]])
                    tensor_hash.update(batch.numpy().tobytes())
                    logits = fc(bb(batch.cuda()))
                    assert logits.shape == (len(batch), 100) and logits.dtype == torch.float32
                    assert torch.isfinite(logits).all()
                    output.append(logits.cpu().numpy())
            logits = np.concatenate(output)
            input_sha = tensor_hash.hexdigest()
            if condition in common_hashes:
                assert input_sha == common_hashes[condition]
            else:
                common_hashes[condition] = input_sha
            path = destination / f"{model}_{condition}.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, ids=ids, labels=labels, logits=logits)
            after = [state_hash(bb.state_dict()), state_hash(fc.state_dict())]
            assert before == after
            predictions[model][condition] = path.relative_to(TASK).as_posix()
            records.append(dict(model=model, condition=condition, shape=list(logits.shape), dtype=str(logits.dtype),
                all_finite=True, normalized_inputs_sha256=input_sha, checkpoint_sha256=row["checkpoint_sha256"],
                npz_sha256=sha(path), npz_bytes=path.stat().st_size, state_before=before, state_after=after,
                allocated_peak_bytes=torch.cuda.max_memory_allocated(), model_eval=True, BN_unchanged=True,
                dropout_modules=0, all_parameters_require_grad=False))
            print(f"INFER {model} {condition}: {logits.shape}, finite, paired inputs, frozen state", flush=True)
        assert sha(ROOT / row["checkpoint_path"]) == row["checkpoint_sha256"]
        del bb, fc
        torch.cuda.empty_cache()
    write_json(OUT / "INFERENCE_VALIDATION.json", dict(runtime=runtime, predictions=records,
               common_normalized_input_hashes=common_hashes, parameters_and_BN_unchanged=True))
    # Only now do all manifest-referenced prediction files exist.
    for pair_id, spec in PAIRS.items():
        manifest = dict(class_counts=counts.tolist(), reference=spec["reference"], candidate=spec["candidate"],
            conditions=list(CONDITIONS), methods={m: predictions[m] for m in (spec["reference"], spec["candidate"])},
            pair_provenance=availability["pairs"][pair_id], holdout_status=attestation["holdout_status"])
        name = "manifest.json" if pair_id == "h2t" else "manifest_pi_local.json"
        write_json(TASK / name, manifest)
    print("INFERENCE COMPLETE: 20 prediction NPZs; manifests created after all files existed.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["prepare", "infer"])
    args = p.parse_args()
    guard = install_guard()
    (prepare if args.phase == "prepare" else infer)()
    assert not any(guard.values())
    write_json(OUT / (args.phase.upper() + "_GUARD.json"), guard)


if __name__ == "__main__":
    main()
