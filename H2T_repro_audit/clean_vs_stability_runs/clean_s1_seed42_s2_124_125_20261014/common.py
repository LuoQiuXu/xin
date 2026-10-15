"""Reviewed Clean helper functions with independent ROOT; no Stage I training entry."""
from __future__ import annotations
import ast,contextlib,csv,hashlib,json,math,os,random,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
WORK=BASE.parent
ORIGIN=BASE/'clean_vs_runs/clean_s1_seed42_vs_seed123_20261013'
RUNTIME=ROOT/'runtime'
DATA=WORK/'H2T/data/cifar100'
N,BATCH,STEPS,EPOCHS=10847,128,85,10
SEEDS=[124,125]
ARMS=['Uniform-PerSample','Low-VS']
S1_SHA='7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88'
STAT_SHA='e81a4897a965d642f832bd6429e0e5a7d136725daaa069d9ded96f87ab64aef9'
_GUARD=None
_INTEGRITY=False


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def dump(p, obj):
    p = Path(p)
    assert p.resolve().is_relative_to(ROOT), f'Write outside new experiment: {p}'
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.writing')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)

def sha(p, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()

def state_hash(module):
    h = hashlib.sha256()
    for k, t in module.state_dict().items():
        h.update(k.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def csv_write(p, rows):
    assert rows and Path(p).resolve().is_relative_to(ROOT)
    with Path(p).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def guard_flush():
    if _GUARD is not None:
        dump(ROOT/f'test_guard_{_GUARD["phase"]}.json', _GUARD)

def install_test_guard(phase):
    global _GUARD
    assert _GUARD is None
    _GUARD = {'phase': phase, 'installed_at': now(), 'official_test_image_or_label_loads': 0,
        'allowed_integrity_open_count': 0, 'blocked_attempts': [], 'CIFAR100_trainFalse_calls': 0}
    forbidden = os.path.normcase(os.path.abspath(DATA/'cifar-100-python/test'))

    def deny(kind, path):
        _GUARD['blocked_attempts'].append({'at': now(), 'kind': kind, 'path': str(path)})
        guard_flush()
        raise RuntimeError(f'OFFICIAL TEST ACCESS PROHIBITED: {kind}: {path}')

    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
            if p == forbidden:
                if _INTEGRITY:
                    _GUARD['allowed_integrity_open_count'] += 1
                else:
                    deny('official test file open', p)
    sys.addaudithook(audit)
    real_init = torchvision.datasets.CIFAR100.__init__

    def protected_init(self, root, train=True, *args, **kwargs):
        if not train:
            _GUARD['CIFAR100_trainFalse_calls'] += 1
            deny('CIFAR100(train=False)', root)
        return real_init(self, root, train, *args, **kwargs)
    torchvision.datasets.CIFAR100.__init__ = protected_init
    guard_flush()

@contextlib.contextmanager
def integrity_only():
    global _INTEGRITY
    previous = _INTEGRITY
    _INTEGRITY = True
    try:
        yield
    finally:
        _INTEGRITY = previous

def guard_pass():
    assert _GUARD is not None and not _GUARD['blocked_attempts']
    assert _GUARD['official_test_image_or_label_loads'] == 0
    assert _GUARD['CIFAR100_trainFalse_calls'] == 0
    guard_flush()

def seed_all(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

def metrics(pred, labels):
    counts = np.bincount(labels, minlength=100)
    correct = np.bincount(labels[pred == labels], minlength=100)
    per = np.divide(correct*100., counts, out=np.full(100,np.nan), where=counts > 0)
    n = np.bincount(np.load(ROOT/'train_labels.npy',allow_pickle=False), minlength=100)
    groups = {'Head':np.arange(36), 'OfficialMedium':np.arange(36,71), 'Tail':np.arange(71,100),
        'Many':np.flatnonzero(n > 100), 'Medium':np.flatnonzero((n >= 20)&(n <= 100)), 'Few':np.flatnonzero(n < 20)}
    result = {'Overall':float((pred == labels).mean()*100), 'Macro':float(np.nanmean(per)),
        'samples':int(len(labels)), 'logits_classes':100, 'missing_classes':np.flatnonzero(counts == 0).tolist(),
        'class_counts':counts.tolist(), 'class_correct':correct.tolist(),
        'per_class_accuracy':[None if np.isnan(v) else float(v) for v in per],
        'predicted_missing_class_count':int(np.isin(pred,np.flatnonzero(counts == 0)).sum()), 'effective_classes':{}}
    for k, ids in groups.items():
        active = ids[counts[ids] > 0]
        result[k] = float(per[active].mean())
        result['effective_classes'][k] = active.tolist()
    return result

def evaluate(classifier, labels, *, backbone=None, loader=None, feature_maps=None):
    classifier.eval()
    if backbone is not None:
        backbone.eval()
    logits_all, total = [], 0.
    with torch.no_grad():
        if loader is not None:
            for images, targets, _ in loader:
                z = classifier(backbone(images.cuda(non_blocking=True)))
                assert z.shape[1] == 100 and torch.isfinite(z).all()
                total += float(torch.nn.functional.cross_entropy(z,targets.cuda(),reduction='sum'))
                logits_all.append(z.cpu().numpy())
        else:
            for start in range(0,len(labels),BATCH):
                x = torch.from_numpy(np.array(feature_maps[start:start+BATCH],copy=True)).cuda()
                z = classifier(x)
                assert z.shape[1] == 100 and torch.isfinite(z).all()
                y = torch.from_numpy(labels[start:start+BATCH].copy()).cuda()
                total += float(torch.nn.functional.cross_entropy(z,y,reduction='sum'))
                logits_all.append(z.cpu().numpy())
    logits = np.concatenate(logits_all)
    assert logits.shape == (len(labels),100)
    pred = logits.argmax(1)
    out = metrics(pred,labels)
    out['validation_CE'] = total/len(labels)
    return out, logits, pred

def check_frozen(bb, before):
    assert all(not m.training for m in bb.modules())
    assert all(not p.requires_grad and p.grad is None for p in bb.parameters())
    assert all(v.detach().cpu().contiguous().numpy().tobytes() == before[k].contiguous().numpy().tobytes()
               for k,v in bb.state_dict().items())

def check_history():
    before = read(ROOT/'historical_before.json')
    changed = [p for p,v in before.items() if not Path(p).is_file() or sha(p) != v]
    out = {'verified_at':now(), 'protected_files':len(before), 'changed_or_missing':changed}
    dump(ROOT/'historical_verification.json',out)
    assert not changed, changed
    return out
