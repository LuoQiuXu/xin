"""Small run protection/status helpers. All outputs belong to this new iteration."""
from pathlib import Path
import os,sys,json,traceback
import numpy as np
import torch,torchvision
from datetime import datetime,timezone,timedelta


def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def history_manifest(old,sha):
    result={}
    for current,dirs,files in os.walk(old):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__'}]
        for name in files:
            p=Path(current)/name;result[p.relative_to(old).as_posix()]=sha(p)
    return result


def protect(project_audits,here,events):
    base=str(Path(project_audits).resolve()).casefold().rstrip('\\/')
    permitted=str(Path(here).resolve()).casefold().rstrip('\\/')
    def historical(value):
        if not isinstance(value,(str,bytes,os.PathLike)):return False
        p=os.path.abspath(os.fsdecode(value)).casefold()
        return (p==base or p.startswith(base+os.sep)) and not (p==permitted or p.startswith(permitted+os.sep))
    def audit(event,args):
        deny=False
        if event=='open' and args and historical(args[0]):
            mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
            deny=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        elif event in ['os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink','os.chmod','os.utime']:
            deny=any(historical(a) for a in args[:2])
        if deny:
            events.append({'event':event,'path':str(args[0])})
            raise RuntimeError('Other experiment files are read-only: '+str(args[0]))
    sys.addaudithook(audit)


def environment():
    return {'python':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'GPU':torch.cuda.get_device_name(0),'total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}


def check_history(r):
    expected=json.loads((r.RESULTS/'historical_manifest.json').read_text(encoding='utf-8'))
    assert history_manifest(r.OLD,r.sha)==expected,'Historical experiment changed'
    return len(expected)


def source_hashes(r):
    return {name:r.sha(r.HERE/name) for name in ['ravs_policy.py','run_ravs_v1.py','run_safeguards.py','test_policy.py']}
