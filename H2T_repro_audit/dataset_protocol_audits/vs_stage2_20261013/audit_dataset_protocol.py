"""Read-only provenance checks; never construct a model/dataset or evaluate test images."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import pickle
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np
import torch

sys.dont_write_bytecode = True
WORK = Path(r'D:\xiangmuxunlian\新')
BASE = WORK / 'H2T_repro_audit'
VS = BASE / 'variation_shift_stage2_runs/vs_stage2_seed123'
SELECTOR = BASE / 'donor_stage2_runs/selector_fixed_bn_20261012'
SMART = BASE / 'smart_channel_runs/smart_cov_20261013_seed123'
MECHANISM = BASE / 'variation_shift_runs/variation_shift_20261013_seed123'
FORMAL = BASE / 'formal_runs/formal_20261011_1429_r2'
DATA = WORK / 'H2T/data/cifar100'
TV_SOURCE = Path(r'D:\app\anaconda\envs\pytorch\Lib\site-packages\torchvision\datasets\cifar.py')
ARMS = ['Original-H2T', 'Uniform-PerSample', 'High-VS', 'Low-VS', 'Shuffled-VS']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_json(out, name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(out, name, rows):
    with (out / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def source_record(path):
    p = Path(path)
    return {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}


def official_subset(labels):
    # Execute only the two original pure NumPy methods, with raw IDs as data.
    # No torchvision Dataset constructor and no official eval loader are called.
    source = BASE / 'official/datasets/cifar100.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IMBALANCECIFAR100')
    names = ['get_img_num_per_cls', 'gen_imbalanced_data']
    funcs = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), 'exec'), ns)
    obj = SimpleNamespace(data=np.arange(len(labels), dtype=np.int64)[:, None], targets=labels.tolist())
    for name in names:
        setattr(obj, name, MethodType(ns[name], obj))
    counts = obj.get_img_num_per_cls(100, 'exp', .01)
    previous = np.random.get_state()
    try:
        np.random.seed(0)  # rand_number=0 in the actual author loader.
        obj.gen_imbalanced_data(counts)
    finally:
        np.random.set_state(previous)
    return obj.data[:, 0], np.array(obj.targets, dtype=np.int64), np.array(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.is_relative_to((BASE / 'dataset_protocol_audits').resolve()), 'Audit writes restricted to new audit outputs'
    out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    evidence = {'audited_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'scope': 'existing VS seed123 five arms; read-only checks; no training or model forward',
        'python': sys.executable, 'numpy': np.__version__, 'torch': torch.__version__,
        'test_pickle_deserialized': False, 'test_model_evaluation_executed': False,
        'model_created': False, 'training_started': False}

    print('Checking protected historical file hashes...', flush=True)
    expected_history = read(VS / 'historical_before.json')
    delivery = read(VS / 'delivery_manifest.json')
    expected_history.update({str(VS / name): value for name, value in delivery['files'].items()})
    extra = [VS / 'delivery_manifest.json', WORK / 'VS_STAGE2_RESULTS.md', TV_SOURCE]
    extra.extend(DATA / 'cifar-100-python' / name for name in ['train', 'test', 'meta'])
    extra.append(DATA / 'cifar-100-python.tar.gz')
    for p in extra:
        expected_history.setdefault(str(p), digest(p))
    before = {name: digest(name) if Path(name).is_file() else None for name in expected_history}
    discrepancies = [{'path': p, 'expected': expected_history[p], 'actual': before[p]}
                     for p in before if before[p] != expected_history[p]]
    write_json(out, 'historical_before.json', before)
    evidence['previous_manifests'] = {'older_files': len(read(VS / 'historical_before.json')),
        'VS_delivery_files': delivery['file_count'], 'previous_seal': delivery['sealed_at'],
        'preexisting_discrepancies': discrepancies}

    print('Verifying official raw files and all LT / validation IDs and pixels...', flush=True)
    # Extract expected CIFAR100 integrity values from installed torchvision SOURCE.
    tv_tree = ast.parse(TV_SOURCE.read_text(encoding='utf-8'))
    tv_class = next(n for n in tv_tree.body if isinstance(n, ast.ClassDef) and n.name == 'CIFAR100')
    constants = {}
    for n in tv_class.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            constants[n.targets[0].id] = ast.literal_eval(n.value)
    expected_md5 = {'train': constants['train_list'][0][1], 'test': constants['test_list'][0][1],
        'meta': constants['meta']['md5'], 'cifar-100-python.tar.gz': constants['tgz_md5']}
    integrity = {}
    for name, expected in expected_md5.items():
        p = DATA / name if name.endswith('.tar.gz') else DATA / constants['base_folder'] / name
        integrity[name] = {'path': str(p), 'bytes': p.stat().st_size, 'md5': digest(p, 'md5'),
            'expected_torchvision_md5': expected, 'sha256': digest(p)}
        integrity[name]['verified'] = integrity[name]['md5'] == expected
        if not name.endswith('.tar.gz'):
            assert integrity[name]['verified'], name
    # A damaged legacy download must be recorded, not silently repaired or
    # mistaken for the provenance of already verified extracted files.
    archive_scan = {'members_seen': [], 'complete': False}
    try:
        with tarfile.open(DATA / 'cifar-100-python.tar.gz', 'r:gz') as archive:
            for item in archive:
                archive_scan['members_seen'].append({'name': item.name, 'bytes': item.size})
        archive_scan['complete'] = True
    except (EOFError, OSError, tarfile.TarError) as err:
        archive_scan['error_type'] = type(err).__name__
        archive_scan['error'] = str(err)
    evidence['legacy_download_archive_scan'] = archive_scan
    evidence['data_integrity'] = integrity
    write_json(out, 'raw_file_integrity.json', {'files': integrity, 'archive_scan': archive_scan})
    evidence['torchvision_cifar100_constants'] = constants
    # Only TRAIN and meta are decoded; TEST is hashed as an opaque file above.
    with (DATA / constants['base_folder'] / 'train').open('rb') as f:
        raw = pickle.load(f, encoding='bytes')
    images = raw[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    labels = np.asarray(raw[b'fine_labels'], dtype=np.int64)
    with (DATA / constants['base_folder'] / 'meta').open('rb') as f:
        meta = pickle.load(f, encoding='bytes')
    class_names = [v.decode('utf-8') for v in meta[b'fine_label_names']]
    assert images.shape == (50000, 32, 32, 3)
    assert np.array_equal(np.bincount(labels), np.full(100, 500))
    split = np.load(VS / 'split_indices.npz', allow_pickle=False)
    ti, vi, ui = (split[k] for k in ['train_raw_indices', 'validation_raw_indices', 'unused_raw_indices'])
    original_ids, original_labels, counts = official_subset(labels)
    assert np.array_equal(original_ids, ti) and np.array_equal(original_labels, labels[ti])
    assert np.array_equal(ui, np.setdiff1d(np.arange(50000), ti))
    checks = {'ordered_LT_indices_equal_original_H2T': np.array_equal(original_ids, ti),
        'LT_labels_equal_original_H2T': np.array_equal(original_labels, labels[ti]),
        'train_pixels_match_original_TRAIN_indices': np.array_equal(images[ti], np.load(VS / 'train_images.npy')),
        'train_labels_match_original_TRAIN_indices': np.array_equal(labels[ti], np.load(VS / 'train_labels.npy')),
        'validation_pixels_match_original_TRAIN_indices': np.array_equal(images[vi], np.load(VS / 'validation_images.npy')),
        'validation_labels_match_original_TRAIN_indices': np.array_equal(labels[vi], np.load(VS / 'validation_labels.npy'))}
    assert all(checks.values())
    training_hashes = {hashlib.sha256(im.tobytes()).digest() for im in images[ti]}
    val_hashes = [hashlib.sha256(im.tobytes()).digest() for im in images[vi]]
    content_overlap = sum(h in training_hashes for h in val_hashes)
    overlap = int(np.intersect1d(ti, vi).size)
    assert overlap == 0 and content_overlap == 0
    assert np.unique(ti).size == 10847 and np.unique(vi).size == 1980
    # Reproduce the already locked unused-TRAIN split, without overwriting it.
    split_rng = np.random.default_rng(20261012)
    reconstructed_vi, excluded = [], []
    for c in range(1, 100):
        pool = ui[labels[ui] == c].copy()
        split_rng.shuffle(pool)
        seen = set()
        chosen = []
        for index in pool:
            h = hashlib.sha256(images[index].tobytes()).digest()
            if h in training_hashes or h in seen:
                excluded.append(int(index))
                continue
            seen.add(h)
            chosen.append(int(index))
            if len(chosen) == 20:
                break
        reconstructed_vi.extend(chosen)
    assert np.array_equal(reconstructed_vi, vi)
    val_counts = np.bincount(labels[vi], minlength=100)
    assert val_counts.tolist() == [0] + [20] * 99
    frozen_files = ['split_indices.npz', 'train_images.npy', 'train_labels.npy',
        'validation_images.npy', 'validation_labels.npy', 'validation_feature_maps.npy']
    copies = {n: {str(p): digest(p / n) for p in [SELECTOR, SMART, VS]} for n in frozen_files}
    assert all(len(set(v.values())) == 1 for v in copies.values())
    defs = {'Head': np.arange(36), 'OfficialMedium': np.arange(36, 71), 'Tail': np.arange(71, 100),
        'Many': np.flatnonzero(counts > 100),
        'Medium': np.flatnonzero((counts >= 20) & (counts <= 100)), 'Few': np.flatnonzero(counts < 20)}
    groups = {g: {'nominal_classes': len(ids), 'effective_classes': ids[val_counts[ids] > 0].tolist(),
                  'effective_class_count': int((val_counts[ids] > 0).sum()),
                  'samples': int(val_counts[ids].sum())} for g, ids in defs.items()}
    evidence['split'] = {'source_partition': 'official CIFAR-100 TRAIN (50000 rows)',
        'raw_train_shape': list(images.shape), 'LT_samples': len(ti), 'LT_classes': len(counts),
        'counts': counts.tolist(), 'IF_max_over_min': int(counts.max() / counts.min()),
        'imb_factor': .01, 'imb_type': 'exp', 'rand_number': 0, 'validation_samples': len(vi),
        'validation_classes': 99, 'validation_counts': val_counts.tolist(),
        'missing_classes': [0], 'missing_class_name': class_names[0], 'unused_train_samples': len(ui),
        'unused_not_validation_samples': len(ui) - len(vi), 'train_validation_index_overlap': overlap,
        'train_validation_pixel_hash_overlap': content_overlap, 'validation_unique_pixel_hashes': len(set(val_hashes)),
        'validation_split_reconstructed_exactly': True, 'split_seed': 20261012,
        'duplicate_exclusions': excluded, 'checks': {k: bool(v) for k, v in checks.items()},
        'train_ID_array_sha256': array_hash(ti), 'validation_ID_array_sha256': array_hash(vi),
        'LT_pixel_bytes_sha256': hashlib.sha256(images[ti].tobytes()).hexdigest(),
        'LT_label_bytes_sha256': hashlib.sha256(labels[ti].tobytes()).hexdigest(),
        'frozen_file_hashes': copies, 'groups': groups,
        'split_creation_time': read(SELECTOR / 'protocol.json')['created_before_training']}
    write_csv(out, 'dataset_class_counts.csv', [dict(class_id=c, class_name=class_names[c],
        official_train=500, LT=int(counts[c]), unused_train=500-int(counts[c]), validation=int(val_counts[c])) for c in range(100)])
    write_csv(out, 'raw_indices_verified.csv', [dict(split=s, position=i, partition='official_train',
        raw_index=int(index), class_id=int(labels[index]), original_filename=raw[b'filenames'][index].decode('utf-8'))
        for s, ids in [('LT_train', ti), ('validation', vi)] for i, index in enumerate(ids)])

    print('Checking all Stage II batch-index logs and saved validation predictions...', flush=True)
    plan = np.load(VS / 'sampling_and_masks.npz', allow_pickle=False)
    receivers, donors = plan['receivers'], plan['donors']
    assert receivers.shape == donors.shape == (10, 10847)
    assert int(receivers.min()) >= 0 and int(receivers.max()) < len(ti)
    assert int(donors.min()) >= 0 and int(donors.max()) < len(ti)
    assert all(np.array_equal(np.sort(d), np.arange(10847)) for d in donors)
    bs_ids, is_ids = ti[receivers], ti[donors]
    assert np.intersect1d(bs_ids, vi).size == 0 and np.intersect1d(is_ids, vi).size == 0
    row_checks = {}
    saved_prediction_rows, selections = [], {}
    for arm in ARMS:
        rows = csv_rows(VS / 'arms' / arm / 'batches.csv')
        assert len(rows) == 850
        for e in range(10):
            for b in range(85):
                r = rows[e * 85 + b]
                start, end = b * 128, min((b + 1) * 128, 10847)
                rec, don = receivers[e, start:end], donors[e, start:end]
                assert int(r['epoch']) == e+1 and int(r['batch']) == b and int(r['size']) == end-start
                assert r['indices_sha256'] == array_hash(rec, don, labels[ti][rec], labels[ti][don])
        row_checks[arm] = {'verified_batch_rows': len(rows), 'training_draws_per_epoch': 10847,
            'all_batch_index_hashes_match_saved_plan': True, 'val_ID_overlap_BS': 0, 'val_ID_overlap_IS': 0}
        accs = []
        for e in range(1, 11):
            pred_file = VS / 'arms' / arm / f'validation_epoch{e:02d}.npz'
            with np.load(pred_file, allow_pickle=False) as q:
                logits, predictions, y = q['logits'], q['predictions'], q['labels']
                assert logits.shape == (1980, 100) and np.array_equal(y, labels[vi])
                assert np.array_equal(predictions, logits.argmax(1)) and np.isfinite(logits).all()
                accuracy = float((predictions == y).mean() * 100)
                accs.append(accuracy)
                saved_prediction_rows.append(dict(arm=arm, epoch=e, samples=len(y), outputs=logits.shape[1],
                    Overall_from_saved_predictions=accuracy, class0_predictions=int((predictions == 0).sum()),
                    labels_equal_frozen_validation=True, sha256=digest(pred_file)))
        status = read(VS / 'arms' / arm / 'status.json')
        best = int(np.argmax(accs)) + 1
        assert best == status['best_epoch']
        assert abs(accs[best-1] - status['best_validation']['Overall']) < 1e-12
        selections[arm] = {'best_epoch': best, 'best_99class_validation_Overall': accs[best-1],
            'epoch10_99class_validation_Overall': accs[9], 'selection_reconstructed_from_saved_predictions': True}
    evidence['stage2_actual_data'] = {'plans_shape': list(receivers.shape), 'all_plan_indices_LT_only': True,
        'IS_each_epoch_is_full_LT_permutation': True, 'BS_unique_raw_images_across_10_epochs': int(np.unique(bs_ids).size),
        'IS_unique_raw_images_across_10_epochs': int(np.unique(is_ids).size),
        'verified_batch_rows_total': 4250, 'arms': row_checks,
        'verified_saved_validation_files': len(saved_prediction_rows), 'logits_shape_each': [1980, 100],
        'checkpoint_selection_rule': 'max validation Overall over epochs1..10; earliest tie',
        'primary_comparison_rule': read(VS / 'protocol.json')['primary'], 'selections': selections}
    write_csv(out, 'vs_saved_validation_verified.csv', saved_prediction_rows)
    stats = np.load(VS / 'class_feature_statistics.npz', allow_pickle=False)
    assert np.array_equal(stats['counts'], counts)
    gap_path = MECHANISM / 'feature_banks/eval_GAP_float64.npy'
    features = np.load(gap_path, mmap_mode='r', allow_pickle=False)
    assert features.shape == (10847, 64)
    assert np.array_equal(np.load(MECHANISM / 'train_labels.npy'), labels[ti])
    assert np.array_equal(np.load(MECHANISM / 'split_indices.npz')['train_raw_indices'], ti)
    m, v = [], []
    for c in range(100):
        x = features[labels[ti] == c]
        mean = x.mean(0, dtype=np.float64)
        m.append(mean)
        v.append(np.square(x - mean).mean(0, dtype=np.float64))
    assert np.array_equal(m, stats['mean_eval']) and np.array_equal(v, stats['variance_eval_ddof0'])
    assert digest(VS / 'class_feature_statistics.npz') == digest(MECHANISM / 'class_feature_statistics.npz')
    evidence['VS_statistics'] = {'frozen_statistics_path': str(VS / 'class_feature_statistics.npz'),
        'sha256': digest(VS / 'class_feature_statistics.npz'), 'counts_equal_LT': True,
        'cached_training_GAP_path': str(gap_path), 'cached_training_GAP_shape': list(features.shape),
        'means_and_variances_reconstructed_exactly_from_saved_LT_features': True,
        'new_feature_extraction_performed': False}

    print('Reading CPU-only checkpoint metadata and Stage I selection history...', flush=True)
    s1_path = Path(read(VS / 'preflight.json')['stage1_checkpoint']['path'])
    s1 = torch.load(s1_path, map_location='cpu', weights_only=False)
    ep_rows = csv_rows(FORMAL / 'epochs.csv')
    scores = [float(r['overall']) for r in ep_rows]
    best_e = int(np.argmax(scores)) + 1
    assert len(ep_rows) == 200 and best_e == s1['epoch'] == 198
    assert abs(float(s1['best_acc1']) - scores[best_e-1]) < 1e-6
    old_evaluation = read(FORMAL / 'best_evaluation.json')
    assert sum(old_evaluation['class_counts']) == 10000 and old_evaluation['class_counts'] == [100] * 100
    formal_preflight = read(FORMAL / 'preflight.json')
    snapshot = Path(read(FORMAL / 'status.json')['code_dir'])
    assert digest(snapshot / 'train_stage1.py') == formal_preflight['runtime_files']['train_stage1.py']
    assert digest(snapshot / 'datasets/cifar100.py') == digest(BASE / 'official/datasets/cifar100.py')
    evidence['stage1'] = {'checkpoint': source_record(s1_path), 'epoch': int(s1['epoch']),
        'best_acc1_in_checkpoint': float(s1['best_acc1']), 'selection_epoch_from_200_saved_log_rows': best_e,
        'official_test_Overall_from_existing_report': old_evaluation['metrics_percent']['overall'],
        'epochs_with_test_evaluation': len(ep_rows), 'evaluated_partition': 'official TEST, train=False',
        'saved_historical_test_samples': sum(old_evaluation['class_counts']),
        'best_selected_using_test_Overall': True, 'test_used_for_gradient_training': False,
        'historical_evaluation_only_read': True, 'actual_training_source_snapshot': str(snapshot),
        'training_snapshot_sha_verified_against_preflight': True,
        'LT_pixel_hash_matches_StageI_preflight': evidence['split']['LT_pixel_bytes_sha256'] == formal_preflight['subset_sha256']}
    assert evidence['stage1']['LT_pixel_hash_matches_StageI_preflight']
    write_csv(out, 'stage1_epoch_selection_verified.csv', ep_rows)
    checkpoint_rows = []
    for arm in ARMS:
        for kind in ['best', 'final']:
            p = VS / 'arms' / arm / 'checkpoints' / f'model_{kind}.pth.tar'
            q = torch.load(p, map_location='cpu', weights_only=False)
            assert q['StageI_sha256'] == digest(s1_path)
            assert q['split_sha256'] == digest(VS / 'split_indices.npz')
            assert q['validation']['samples'] == 1980 and q['validation']['missing_classes'] == [0]
            fc = next(t for k, t in q['state_dict_classifier'].items() if k.endswith('fc.weight'))
            assert tuple(fc.shape) == (100, 64)
            assert q['epoch'] == (selections[arm]['best_epoch'] if kind == 'best' else 10)
            assert all(t.cpu().contiguous().numpy().tobytes() ==
                       q['state_dict_model'][k].cpu().contiguous().numpy().tobytes()
                       for k, t in s1['state_dict_model'].items())
            checkpoint_rows.append(dict(arm=arm, kind=kind, checkpoint=str(p), sha256=digest(p),
                epoch=q['epoch'], validation_samples=q['validation']['samples'],
                validation_Overall=q['validation']['Overall'], classifier_outputs=int(fc.shape[0]),
                StageI_sha256=q['StageI_sha256'], split_sha256=q['split_sha256'], backbone_equals_StageI=True))
    write_csv(out, 'checkpoint_metadata_verified.csv', checkpoint_rows)
    evidence['stage2_checkpoints'] = {'count': len(checkpoint_rows), 'all_from_locked_StageI': True,
        'all_100_class_outputs_retained': True, 'all_validation_metadata_1980_missing0': True,
        'all_saved_backbone_states_bitwise_equal_StageI': True}
    source_paths = [BASE / 'official/datasets/cifar100.py', BASE / 'runtime/datasets/cifar100.py',
        snapshot / 'datasets/cifar100.py', snapshot / 'train_stage1.py', FORMAL / 'formal_stage1.py',
        FORMAL / 'epochs.csv', FORMAL / 'preflight.json', FORMAL / 'best_evaluation.json',
        SELECTOR / 'selector_stage2.py', SELECTOR / 'protocol.json',
        VS / 'common.py', VS / 'vs_stage2.py', VS / 'training.py', VS / 'reused_helpers.py',
        VS / 'protocol.json', VS / 'preflight.json', VS / 'full_console.log',
        MECHANISM / 'audit.py', MECHANISM / 'statistics_and_scores.py', TV_SOURCE]
    evidence['sources'] = [source_record(p) for p in source_paths]
    evidence['official_archive_provenance'] = {k: v for k, v in read(VS / 'source_manifest.json').items() if k != 'files'}
    evidence['VS_git_provenance'] = read(VS / 'git_training_provenance.json')
    write_json(out, 'source_records.json', evidence['sources'])
    print('Verifying protected history remained unchanged...', flush=True)
    after = {p: digest(p) if Path(p).is_file() else None for p in before}
    changed = [p for p in before if before[p] != after[p]]
    history = {'protected_files': len(before), 'changed_files': changed, 'unchanged': not changed,
        'older_and_VS_prior_manifest_discrepancies': discrepancies}
    write_json(out, 'historical_verification.json', history)
    evidence['history_preservation'] = history
    evidence['elapsed_seconds'] = time.perf_counter() - begin
    write_json(out, 'audit_evidence.json', evidence)
    print(json.dumps({'audit_output': str(out), 'protected_files': len(before), 'history_unchanged': not changed,
        'LT': len(ti), 'validation': len(vi), 'train_val_overlap': overlap,
        'LT_matches_official_all_indices': True, 'StageI_test_selected_epoch': best_e,
        'StageII_validation_records_verified': 50, 'StageII_batch_index_hashes_verified': 4250,
        'elapsed_seconds': evidence['elapsed_seconds']}, ensure_ascii=False), flush=True)
    assert not changed, changed


if __name__ == '__main__':
    main()
