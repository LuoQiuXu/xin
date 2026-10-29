"""Fixed donor audit plan, a small real smoke pass, then cached frozen features.

No training code is imported. All images are members of the original LT TRAIN.
The full pass reuses pilot features verbatim, never recomputing a recipient.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from coverage import ROOT, TASK, DEFAULT_OUT, SEED, guard_io, sha, write_json
from checkpoint import CHECKPOINT, EXPECTED_SHA256, load_models, preprocessing, state_hash


def array_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_npz(path, **arrays):
    assert all(np.asarray(v).dtype.kind in "iufb" for v in arrays.values())
    with path.open("xb") as f:
        np.savez_compressed(f, **arrays)


def draw_candidates(recipients, many, fine, mapping, k):
    same, different = [], []
    for rid in recipients:
        valid = many[(many != rid) & (fine[many] != fine[rid])]
        is_same = mapping[fine[valid]] == mapping[fine[rid]]
        pools = [valid[is_same], valid[~is_same]]
        if any(len(pool) < k for pool in pools):
            raise ValueError("Insufficient Many donors; never broaden frequency groups")
        for arm, (pool, target) in enumerate(zip(pools, (same, different))):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, int(rid), arm]))
            target.append(rng.choice(pool, k, replace=False))
    return np.asarray(same, dtype=np.int64), np.asarray(different, dtype=np.int64)


def raw_train():
    import pickle
    p = ROOT / "H2T/data/cifar100/cifar-100-python/train"
    assert sha(p) == "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"
    with p.open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    pixels = raw["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return pixels, np.asarray(raw["fine_labels"], dtype=np.int64)


def read_json(out, name):
    return json.loads((out / name).read_text(encoding="utf-8"))


def plan(out):
    scope = read_json(out, "SCOPE_RESOLUTION.json")
    rules = read_json(out, "PRE_ANALYSIS_PLAN.json")
    assert scope["approved"] and scope["limited_eligible_Few_images"] == 210
    with np.load(out / "primary_cache/feasibility_plan.npz", allow_pickle=False) as z:
        rec = z["eligible_Few_raw_ids"].copy()
        many, counts, mapping, lt = (z[k].copy() for k in ["Many_raw_ids", "LT_class_counts", "fine_to_coarse", "LT_raw_ids"])
    _, fine = raw_train()
    assert len(rec) == 210 and len(np.unique(rec)) == 210
    assert np.array_equal(np.unique(fine[rec]), scope["limited_eligible_Few_classes"])
    assert (counts[fine[rec]] < 20).all() and (counts[fine[many]] > 100).all()
    rec = np.random.default_rng(SEED).permutation(rec)
    k = rules["candidates_per_recipient_per_semantic_arm"]
    same, different = draw_candidates(rec, many, fine, mapping, k)
    assert k == 32 and same.shape == different.shape == (210, 32)
    all_ids = np.unique(np.concatenate([rec, same.ravel(), different.ravel()]))
    assert np.isin(all_ids, lt).all()
    pilot = rec[:rules["pilot_recipients"]]
    pilot_ids = np.unique(np.concatenate([pilot, same[:len(pilot)].ravel(), different[:len(pilot)].ravel()]))
    save_npz(out / "primary_cache/pair_plan.npz", recipients=rec, same_donors=same,
        different_donors=different, all_raw_ids=all_ids, pilot_recipients=pilot,
        pilot_raw_ids=pilot_ids, fine_to_coarse=mapping, class_counts=counts)
    write_json(out / "PAIR_PLAN.json", dict(
        seed=SEED, recipients=rec.tolist(), recipient_classes=np.unique(fine[rec]).tolist(),
        n_recipients=210, n_unique_feature_images=len(all_ids), candidates_per_arm=32,
        n_candidate_pairs=210*64, same_semantic_pairs=210*32, different_semantic_pairs=210*32,
        pilot_recipients=pilot.tolist(), pilot_unique_images=len(pilot_ids),
        pair_plan_sha256=sha(out / "primary_cache/pair_plan.npz"),
        pre_analysis_plan_sha256=sha(out / "PRE_ANALYSIS_PLAN.json"),
        candidate_plan_written_before_features=True, feature_outcomes_used_for_sampling=False,
        original_Few_Many_definitions_preserved=True))
    print(f"PLAN PASS: 210 fixed Few recipients, 32+32 Many candidates each, {len(all_ids)} unique LT images.", flush=True)


def extract(out, phase):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torchvision
    from PIL import Image
    assert torch.cuda.is_available()
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    metadata = read_json(out, "PAIR_PLAN.json")
    assert sha(out / "PRE_ANALYSIS_PLAN.json") == metadata["pre_analysis_plan_sha256"]
    assert sha(out / "primary_cache/pair_plan.npz") == metadata["pair_plan_sha256"]
    availability = read_json(out, "CHECKPOINT_AVAILABILITY.json")
    for path, expected in availability["source_sha256"].items():
        assert sha(ROOT / path) == expected
    with np.load(out / "primary_cache/pair_plan.npz", allow_pickle=False) as z:
        planned = {k: z[k].copy() for k in z.files}
    bb, fc, _ = load_models()
    assert state_hash(bb.state_dict()) == availability["backbone_state_sha256"]
    assert state_hash(fc.state_dict()) == availability["classifier_state_sha256"]
    bb, fc = bb.cuda().eval(), fc.cuda().eval()
    modules = list(bb.modules()) + list(fc.modules())
    assert not any(m.training or isinstance(m, torch.nn.modules.dropout._DropoutNd) for m in modules)
    assert not any(p.requires_grad for p in list(bb.parameters()) + list(fc.parameters()))
    prior_ids = np.array([], dtype=np.int64)
    prior_features = np.empty((0, 64), dtype=np.float32)
    if phase == "pilot":
        ids = planned["pilot_raw_ids"]
    else:
        pilot = read_json(out, "PILOT_VALIDATION.json")
        assert pilot["passed"] and sha(out / "primary_cache/pilot_feature_bank.npz") == pilot["feature_bank_sha256"]
        with np.load(out / "primary_cache/pilot_feature_bank.npz", allow_pickle=False) as z:
            prior_ids, prior_features = z["ids"].copy(), z["features"].copy()
        ids = np.setdiff1d(planned["all_raw_ids"], prior_ids)
    pixels, labels = raw_train()
    mean, std = preprocessing()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    before = [state_hash(bb.state_dict()), state_hash(fc.state_dict())]
    input_digest = hashlib.sha256()
    features, batch_shapes, numpy_gap = [], [], 0.
    numpy_argmax_disagreements = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(ids), 64):
            current = ids[start:start+64]
            cpu = torch.stack([transform(Image.fromarray(pixels[i])) for i in current])
            input_digest.update(cpu.numpy().tobytes())
            maps = bb(cpu.cuda())
            feat = torch.nn.functional.avg_pool2d(maps, maps.size(3)).flatten(1)
            logits = fc(maps)
            direct = torch.nn.functional.linear(feat, fc.fc.weight, fc.fc.bias)
            assert torch.equal(logits, direct), "Feature pooling does not exactly reproduce original head input"
            assert maps.shape == (len(current), 64, 8, 8) and feat.shape == (len(current), 64)
            assert feat.dtype == logits.dtype == torch.float32
            assert torch.isfinite(feat).all() and torch.isfinite(logits).all()
            a = feat.cpu().numpy()
            native = logits.cpu().numpy().astype(np.float64)
            w = fc.fc.weight.detach().cpu().numpy().astype(np.float64)
            b = fc.fc.bias.detach().cpu().numpy().astype(np.float64)
            double_logits = a.astype(np.float64) @ w.T + b
            numpy_gap = max(numpy_gap, float(np.max(np.abs(native-double_logits))))
            numpy_argmax_disagreements += int((native.argmax(1) != double_logits.argmax(1)).sum())
            features.append(a)
            batch_shapes.append(list(cpu.shape))
    new_features = np.concatenate(features)
    after = [state_hash(bb.state_dict()), state_hash(fc.state_dict())]
    assert before == after and sha(CHECKPOINT) == EXPECTED_SHA256
    combined_ids = np.concatenate([prior_ids, ids])
    combined_features = np.concatenate([prior_features, new_features])
    order = np.argsort(combined_ids)
    combined_ids, combined_features = combined_ids[order], combined_features[order]
    assert len(combined_ids) == len(np.unique(combined_ids))
    if phase == "features":
        assert np.array_equal(combined_ids, planned["all_raw_ids"])
        assert np.array_equal(combined_features[np.searchsorted(combined_ids, prior_ids)], prior_features)
    bank_name = "pilot_feature_bank.npz" if phase == "pilot" else "feature_bank.npz"
    save_npz(out / "primary_cache" / bank_name, ids=combined_ids,
             labels=labels[combined_ids], features=combined_features)
    record = dict(passed=True, phase=phase, checkpoint_sha256=EXPECTED_SHA256,
        fresh_forward_images=len(ids), cached_reused_images=len(prior_ids), feature_bank_images=len(combined_ids),
        feature_shape=list(combined_features.shape), dtype=str(combined_features.dtype), all_finite=True,
        feature_bank_sha256=sha(out / "primary_cache" / bank_name),
        feature_tensor_sha256=array_hash(combined_ids, combined_features), normalized_input_sha256=input_digest.hexdigest(),
        state_before=before, state_after=after, BN_and_parameters_unchanged=True,
        original_head_vs_pooled_feature_logits_exact=True,
        NumPy_float64_vs_GPU_float32_max_abs_logit_gap=numpy_gap,
        NumPy_float64_vs_GPU_float32_argmax_disagreements=numpy_argmax_disagreements,
        model_eval=True, dropout_modules=0, all_requires_grad=False, optimizer_created=False, backward_calls=0,
        runtime=dict(python=os.sys.executable, torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), GPU=torch.cuda.get_device_name(0),
            total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
            batch_size=64, cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), autocast=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated()),
        batch_shapes=batch_shapes, clean_accuracy_used_to_choose_recipients=False)
    write_json(out / ("PILOT_VALIDATION.json" if phase == "pilot" else "FEATURE_EXTRACTION.json"), record)
    if phase == "features":
        rec = np.repeat(planned["recipients"], 64)
        donor = np.concatenate([planned["same_donors"], planned["different_donors"]], axis=1).ravel()
        ri, di = np.searchsorted(combined_ids, rec), np.searchsorted(combined_ids, donor)
        assert np.array_equal(combined_ids[ri], rec) and np.array_equal(combined_ids[di], donor)
        counts = planned["class_counts"]
        assert (counts[labels[rec]] < 20).all() and (counts[labels[donor]] > 100).all()
        assert (rec != donor).all() and (labels[rec] != labels[donor]).all()
        mapping = planned["fine_to_coarse"]
        arms = (mapping[labels[rec]] == mapping[labels[donor]]).reshape(210, 64)
        assert arms[:, :32].all() and not arms[:, 32:].any()
        pairs = out / "primary_cache/pairs.npz"
        save_npz(pairs, rec_id=rec, rec_label=labels[rec], donor_id=donor, donor_label=labels[donor],
            rec_feat=combined_features[ri].astype(np.float64), donor_feat=combined_features[di].astype(np.float64),
            coarse_label=mapping, weight=fc.fc.weight.detach().cpu().numpy().astype(np.float64),
            bias=fc.fc.bias.detach().cpu().numpy().astype(np.float64))
        write_json(out / "PAIR_INPUT.json", dict(path=pairs.relative_to(TASK).as_posix(), sha256=sha(pairs),
            n_pairs=len(rec), n_recipients=210, feature_dimension=64, candidate_arms=[32, 32],
            all_numeric_no_pickle=True, feature_array_dtype="float64 exact cast from cached float32",
            frozen_head_dtype="float64 exact cast from original float32 weights/bias", donor_mass_fixed=.25,
            candidate_plan_sha256=metadata["pair_plan_sha256"], analysis_plan_sha256=metadata["pre_analysis_plan_sha256"],
            rec_ids_sha256=array_hash(rec), donor_ids_sha256=array_hash(donor),
            algorithm="Protocol scalar frozen-feature proxy only; no original H2T/PI-H2T algorithm or training changed"))
    print(f"{phase.upper()} PASS: {len(ids)} fresh LT images, {len(prior_ids)} reused exact cached features; frozen state and exact head-input parity.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["plan", "pilot", "features"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    out = a.output.resolve()
    assert out.is_relative_to(TASK)
    target = {"plan": "PAIR_PLAN.json", "pilot": "PILOT_VALIDATION.json", "features": "FEATURE_EXTRACTION.json"}[a.phase]
    if (out / target).exists():
        raise FileExistsError("Refuse to overwrite completed phase")
    guard = guard_io()
    (plan(out) if a.phase == "plan" else extract(out, a.phase))
    assert not any(guard.values())
    write_json(out / (a.phase.upper() + "_GUARD.json"), guard)


if __name__ == "__main__":
    main()
