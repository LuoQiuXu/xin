"""Create separate, byte-identical runtime and explicit control observers."""
from pathlib import Path
import argparse
import difflib
import json
import shutil

from bn_audit_helpers import sha, dump, protected_files

BASE = Path(__file__).resolve().parent


def replace(source, old, new):
    assert source.count(old) == 1, f'Expected exactly one anchor: {old[:100]}'
    return source.replace(old,new)


def create(suite_id):
    folder = BASE/'bn_audit_runs'/suite_id
    assert not folder.exists(), 'Never overwrite an existing audit'
    protected = protected_files(BASE)
    manifest = json.loads((BASE/'evidence/runtime_manifest.json').read_text(encoding='utf-8'))['files']
    assert all(sha(BASE/'runtime'/p) == h for p,h in manifest.items())
    folder.mkdir(parents=True)
    for rel in manifest:
        target = folder/'runtime'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/'runtime'/rel,target)
        assert sha(target) == manifest[rel]
    shutil.copy2(BASE/'bn_audit_helpers.py',folder/'bn_audit_helpers.py')
    shutil.copy2(__file__,folder/'create_bn_audit.py')
    dump(folder/'protected_original_before.json',protected)
    old = (BASE/'formal_stage2.py').read_text(encoding='utf-8')
    code = old
    end = code.index('from __future__')
    code = '"""Fixed-BN paired-input control; calls the original training and H2T functions.\nThis is a mechanism control experiment, separate from the official reproduction.\n"""\n'+code[end:]
    code = replace(code, 'from formal_stage1 import check_sources, dump, sha, GpuMonitor, MetricsHandler',
        'from bn_audit_helpers import check_sources as verify_sources, dump, sha, GpuMonitor, MetricsHandler, protected_files')
    code = replace(code, 'BASE = Path(__file__).resolve().parent\nRUNTIME = BASE / "runtime"',
        'SUITE = Path(__file__).resolve().parent\nBASE = SUITE.parent.parent\nRUNTIME = SUITE / "runtime"\nCHANNEL_SEED = 123\n\ndef check_sources():\n    return verify_sources(BASE, RUNTIME)')
    code = replace(code, '"name", f"cifar100_if100_stage2_{suite_id}_{arm}"]',
        '"name", f"cifar100_if100_stage2_{suite_id}_{arm}", "shift_bn", "False"]')
    code = replace(code,'not cfg.mixup and cfg.alpha is None and cfg.shift_bn',
        'not cfg.mixup and cfg.alpha is None and not cfg.shift_bn')
    code = replace(code, 'changed = [k for k in before if not torch.equal(before[k], after[k])]',
        'assert before.keys() == after.keys()\n    changed = [k for k in before if before[k].numpy().tobytes() != after[k].contiguous().numpy().tobytes()]')
    code = replace(code,'assert not folder.exists(), "Use a new suite output directory"',
        'assert folder == SUITE and not (folder/"preflight.json").exists(), "Use a new suite output directory"\n    assert protected_files(BASE) == json.loads((folder/"protected_original_before.json").read_text(encoding="utf-8"))')
    code = replace(code,'folder.mkdir(parents=True)', 'folder.mkdir(parents=True, exist_ok=True)')
    code = replace(code,'shutil.copy2(__file__, folder/"formal_stage2.py")\n    shutil.copy2(BASE/"formal_stage1.py", folder/"formal_stage1_helpers.py")',
        'assert Path(__file__).parent == folder')
    code = replace(code,'"launcher_sha256": sha(__file__), "helpers_sha256": sha(BASE/"formal_stage1.py"),',
        '"launcher_sha256": sha(__file__), "helpers_sha256": sha(SUITE/"bn_audit_helpers.py"),\n        "channel_seed": CHANNEL_SEED, "shift_bn": False, "is_official_reproduction": False,\n        "independent_runtime": str(RUNTIME), "core_source_modifications": [],\n        "control_differences_from_original": ["shift_bn=False", "isolated CPU channel RNG"],')
    code = replace(code,'assert sha(BASE/"formal_stage1.py") == preflight["helpers_sha256"]',
        'assert sha(SUITE/"bn_audit_helpers.py") == preflight["helpers_sha256"]\n    assert protected_files(BASE) == json.loads((folder/"protected_original_before.json").read_text(encoding="utf-8"))')
    code = replace(code,'initial = {}',
        'initial = {}\n    channel_generator = torch.Generator(device="cpu").manual_seed(CHANNEL_SEED)\n    status["channel_rng"] = {"seed": CHANNEL_SEED, "isolation": "torch.random.fork_rng(devices=[])",\n        "initial_state_sha256": hashlib.sha256(channel_generator.get_state().numpy().tobytes()).hexdigest()}\n    assert Path(original.__file__).resolve().is_relative_to(RUNTIME.resolve())\n    import methods\n    assert Path(methods.__file__).resolve().is_relative_to(RUNTIME.resolve())\n    status["imported_source_files"] = {"train_stage2": original.__file__, "methods": methods.__file__}')
    start = code.index('    def fuse(')
    stop = code.index('    class LoaderObserver:',start)
    code = code[:start]+'''    def fuse(x1, x2, rho=.3):
        assert arm == "h2t" and rho == .25 and not torch.is_grad_enabled()
        assert x1.shape[1:] == (64,8,8)
        before, donor = x1.clone(), x2.clone()
        ambient = torch.get_rng_state().clone()
        predictor = torch.Generator(device="cpu")
        predictor.set_state(channel_generator.get_state())
        chosen = torch.randperm(64, generator=predictor)[:16].to(x1.device)
        others = torch.tensor([i for i in range(64) if i not in chosen.cpu().tolist()],device=x1.device)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(channel_generator.get_state())
            value = real_fuse(x1,x2,rho)
            channel_generator.set_state(torch.get_rng_state())
        assert torch.equal(torch.get_rng_state(), ambient), "Channel RNG leaked into data/augmentation stream"
        assert torch.equal(channel_generator.get_state(),predictor.get_state()), "Unexpected channel RNG consumption"
        assert value.data_ptr() == x1.data_ptr() and torch.equal(value[:,chosen],donor[:,chosen])
        assert torch.equal(value[:,others],before[:,others]) and torch.equal(x2,donor)
        status["h2t_calls"] += 1
        ctx["channel_masks"].append(chosen.cpu().tolist())
        if status["h2t_calls"] == 1:
            status["fusion_verified"] = {"rho":rho,"replaced_channels":16,"retained_channels":48,
                "chosen_channels_first_call":chosen.cpu().tolist(),"feature_shape":list(value.shape),
                "same_mask_for_entire_batch":True,"inplace":True,"donor_unchanged":True,
                "ambient_cpu_rng_bitwise_unchanged_every_call":True,"checked_every_fusion_call":True}
        return value

'''+code[stop:]
    code = replace(code,'                if step < 2:\n                    item.update', '                if True:  # fingerprint every batch, including the last partial batch\n                    item.update')
    code = replace(code,'"backbone_requires_grad_parameter_count": sum(p.requires_grad for p in model.parameters())})',
        '"backbone_requires_grad_parameter_count": sum(p.requires_grad for p in model.parameters()),\n                "initial_bn_counters": {k:int(v) for k,v in initial["buffers"].items() if k.endswith("num_batches_tracked")},\n                "core_sources_equal_audited_original": True})')
    code = replace(code,'"loss_min": float("inf"), "loss_max": float("-inf"), "batch_fingerprints": [],',
        '"loss_min": float("inf"), "loss_max": float("-inf"), "batch_fingerprints": [], "channel_masks": [],')
    code = replace(code,'assert not torch.is_grad_enabled() and model.training',
        'assert not torch.is_grad_enabled() and not model.training\n            assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))')
    code = replace(code,'assert len(bn_increment) == 31 and all(v == ctx["backbone_calls"] for v in bn_increment.values())',
        'assert len(bn_increment) == 31 and all(v == 0 for v in bn_increment.values())\n        assert state_hash(bn_after) == state_hash(initial["buffers"]), "BN buffers changed at the byte level"\n        assert state_hash(model_after) == state_hash(initial["parameters"]), "Conv/BN affine parameters changed"')
    code = replace(code,'"batch_fingerprints": ctx["batch_fingerprints"], "rng_at_epoch_start": rng_start,',
        '"batch_fingerprints": ctx["batch_fingerprints"], "channel_masks": ctx["channel_masks"],\n            "backbone_parameters_bitwise_equal_stage1": True, "bn_buffers_bitwise_equal_stage1": True,\n            "backbone_parameters_sha256": state_hash(model_after), "bn_buffers_sha256": state_hash(bn_after),\n            "rng_at_epoch_start": rng_start,')
    code = replace(code,'assert all(torch.equal(v, snapshot(model, True)[k]) for k,v in before.items())',
        'assert state_hash(before) == state_hash(snapshot(model, True)) == state_hash(initial["buffers"])')
    code = replace(code,'assert status["h2t_calls"] == (0 if arm == "dr" else 850)',
        'assert status["h2t_calls"] == (0 if arm == "dr" else 850)\n                status["channel_rng"]["final_state_sha256"] = hashlib.sha256(channel_generator.get_state().numpy().tobytes()).hexdigest()\n                assert protected_files(BASE) == json.loads((folder/"protected_original_before.json").read_text(encoding="utf-8"))\n                status["original_artifacts_unchanged"] = True')
    code = replace(code,'folder = BASE/"stage2_runs"/args.suite_id',
        'folder = SUITE\n    assert SUITE.name == args.suite_id')
    (folder/'bn_audit_runner.py').write_text(code,encoding='utf-8')
    diff = ''.join(difflib.unified_diff(old.splitlines(keepends=True), code.splitlines(keepends=True),
        fromfile='original_observer/formal_stage2.py',tofile='control_observer/bn_audit_runner.py'))
    (folder/'control_observer.diff').write_text(diff,encoding='utf-8')
    eval_old = (BASE/'evaluate_stage2.py').read_text(encoding='utf-8')
    eval_code = replace(eval_old,'from formal_stage1 import dump, sha','from bn_audit_helpers import dump, sha')
    eval_code = replace(eval_code,'BASE = Path(__file__).resolve().parent\nRUNTIME = BASE/"runtime"',
        'SUITE = Path(__file__).resolve().parent\nBASE = SUITE.parent.parent\nRUNTIME = SUITE/"runtime"')
    eval_code = replace(eval_code,'"H2T", str(status["H2T"]), "name",',
        '"H2T", str(status["H2T"]), "shift_bn", "False", "name",')
    eval_code = replace(eval_code,'assert all(torch.equal(v,after[k]) for k,v in before.items())',
        'assert all(v.numpy().tobytes() == after[k].contiguous().numpy().tobytes() for k,v in before.items())')
    eval_code = replace(eval_code,'evaluate(BASE/"stage2_runs"/args.suite_id,args.arm,args.kind)',
        'assert SUITE.name == args.suite_id\n    evaluate(SUITE,args.arm,args.kind)')
    (folder/'evaluate_bn_audit.py').write_text(eval_code,encoding='utf-8')
    (folder/'evaluation_observer.diff').write_text(''.join(difflib.unified_diff(eval_old.splitlines(keepends=True),eval_code.splitlines(keepends=True),
        fromfile='original_observer/evaluate_stage2.py',tofile='control_observer/evaluate_bn_audit.py')),encoding='utf-8')
    assert protected_files(BASE) == protected
    print(f'ISOLATED AUDIT CREATED: {folder}; 48 byte-identical source files; {len(protected)} original evidence files protected',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-id',required=True)
    args = parser.parse_args()
    assert all(c.isalnum() or c in '_-' for c in args.suite_id)
    create(args.suite_id)
