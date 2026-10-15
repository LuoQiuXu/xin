"""Independent evidence utilities; importing this never imports the old runtime."""
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def check_sources(base, runtime):
    expected = json.loads((base/'evidence/runtime_manifest.json').read_text(encoding='utf-8'))['files']
    for root in [base/'runtime', runtime]:
        assert {p: sha(root/p) for p in expected} == expected, f'Source changed: {root}'
    source = json.loads((base/'evidence/source_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(base/'official'/f['path']) == f['sha256'] for f in source['files'])
    return expected


def protected_files(base):
    suite = base/'stage2_runs/official_20261011_1525'
    manifest = json.loads((suite/'delivery_manifest.json').read_text(encoding='utf-8'))
    paths = {str(suite/'delivery_manifest.json'): sha(suite/'delivery_manifest.json')}
    for rel, expected in manifest['files'].items():
        path = suite/rel
        assert sha(path) == expected['sha256']
        paths[str(path)] = expected['sha256']
    for arm, checkpoints in manifest['checkpoint_hashes'].items():
        status = json.loads((suite/arm/'status.json').read_text(encoding='utf-8'))
        for name, expected in checkpoints.items():
            path = Path(status['model_dir'])/name
            assert sha(path) == expected
            paths[str(path)] = expected
        code_dir = Path(status['code_dir'])
        expected_sources = json.loads((base/'evidence/runtime_manifest.json').read_text(encoding='utf-8'))['files']
        for rel, expected in expected_sources.items():
            path = code_dir/rel
            assert sha(path) == expected
            paths[str(path)] = expected
        log = Path(status['official_log'])
        paths[str(log)] = sha(log)
    for name in ['H2T_STAGE2_RESULTS.md', 'H2T_STAGE1_RESULTS.md', 'H2T_REPRO_AUDIT.md']:
        path = base.parent/name
        paths[str(path)] = sha(path)
    for name in ['formal_stage1.py','formal_stage2.py','evaluate_stage2.py','summarize_stage2.py']:
        path = base/name
        paths[str(path)] = sha(path)
    return paths


class GpuMonitor:
    def __init__(self, folder):
        self.path = folder/'gpu_samples.jsonl'
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.peak_mib = 0

    def work(self):
        with self.path.open('w', encoding='utf-8', buffering=1) as stream:
            while not self.stop.is_set():
                sample = {'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    values = [float(x.strip()) for x in result.stdout.strip().split(',')]
                    sample.update(dict(zip(['used_mib','total_mib','gpu_util_percent','temperature_c','power_w'],values)))
                    self.peak_mib = max(self.peak_mib, values[0])
                except Exception as exc:
                    sample['error'] = str(exc)
                stream.write(json.dumps(sample)+'\n')
                self.stop.wait(5)


class MetricsHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.metrics = None

    def emit(self, record):
        match = re.search(r'\* Acc@1 ([\d.]+)% Acc@5 ([\d.]+)% HAcc ([\d.]+)% MAcc ([\d.]+)% TAcc ([\d.]+)%',record.getMessage())
        if match:
            self.metrics = dict(zip(['overall','top5','head','medium','tail'],map(float,match.groups())))
