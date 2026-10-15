"""Generate exact diff against downloaded upstream archive; no experiments."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path(__file__).resolve().parent
archive = root.parents[1]/'tmp/H2T_current_main.zip'
out = root/'review'
out.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    source = {n[len('H2T-main/'):]: z.read(n) for n in z.namelist() if n.startswith('H2T-main/') and not n.endswith('/')}
    provenance = dict(url='https://codeload.github.com/Keke921/H2T/zip/refs/heads/main',
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      archive_commit_comment=z.comment.decode('utf-8'), date='2026-09-30')
files = ['models/resnet_cifar.py', 'utils/util.py', 'utils/logger.py', 'pi_h2t.py', 'train_pi_h2t.py', 'test_pi_h2t_smoke.py', 'pi_dry_run_audit.py',
         'config/cifar100/cifar100_imb001_stage1_ce_pif.yaml', 'config/cifar100/cifar100_imb001_stage2_ce_pi_h2t.yaml',
         'PI_H2T_REVIEW_ZH.md', 'build_review_bundle.py']
diff = []
for name in files:
    old = source.get(name, b'').decode('utf-8').splitlines(keepends=True)
    new = (root/name).read_text(encoding='utf-8').splitlines(keepends=True)
    diff.extend(difflib.unified_diff(old, new, fromfile='a/'+name if name in source else '/dev/null', tofile='b/'+name))
for name in ['methods.py','train_stage1.py','train_stage2.py','datasets/cifar100.py',
             'config/cifar100/cifar100_imb001_stage1_ce_mixup.yaml','config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml']:
    assert (root/name).read_bytes() == source[name], 'Legacy file unexpectedly changed: '+name
provenance['legacy_files_byte_identical'] = True
(out/'upstream_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
(out/'pi_h2t_vs_official_main.diff').write_text(''.join(diff), encoding='utf-8')
test = subprocess.run([sys.executable, 'test_pi_h2t_smoke.py'], cwd=root, capture_output=True, text=True)
(out/'smoke_test.txt').write_text(test.stdout+test.stderr, encoding='utf-8')
print(test.stdout)
if test.returncode:
    print(test.stderr)
    raise SystemExit(test.returncode)
print(json.dumps(provenance))
