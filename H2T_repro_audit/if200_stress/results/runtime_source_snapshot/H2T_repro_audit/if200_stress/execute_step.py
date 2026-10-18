"""Run exactly one authorized protocol command, preserving console and status."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
OUT=HERE/'results'
PYTHON=Path(r'D:\app\minicondaaa\envs\xiaoju\python.exe')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify():
    lock=json.loads((OUT/'execution_source_lock.json').read_text(encoding='utf-8'))
    for name,digest in lock.items():
        assert sha(WORK/name)==digest,('Source hash changed',name)
    history=json.loads((OUT/'historical_manifest.json').read_text(encoding='utf-8'))
    for name,digest in history.items():
        assert sha(WORK/name)==digest,('Historical file changed',name)

def main():
    assert sys.executable.casefold()==str(PYTHON).casefold(),sys.executable
    assert len(sys.argv)>=2
    args=sys.argv[1:]
    if args==['policy']:
        child=[str(PYTHON),'-B','-u',str(HERE/'policy.py')];tag='01_policy'
    else:
        allowed=[['preflight'],['stage1'],['prepare'],['report']]
        allowed += [['train','--seed',str(s),'--arm',a] for s in (124,125) for a in ('DR-NoFusion','Original-BatchShared','Uniform-PerSample','LowVS-PerSample')]
        assert args in allowed,args
        child=[str(PYTHON),'-B','-u',str(HERE/'run_if200.py'),*args]
        tag='_'.join(args).replace('--','')
    registry_path=OUT/'execution_registry.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else []
    earlier=[r for r in registry if r['args']==args]
    if earlier or (OUT/'FAILED_STOP.json').exists():
        repair=json.loads((OUT/'preflight_repair.json').read_text(encoding='utf-8'))
        assert repair['preflight_syntax_only'] and not any(r['status']=='failed' and r['args']!=['preflight'] for r in registry)
        assert all(r['status']=='failed' for r in earlier),('Refuse repeat completed step',args)
        if earlier:assert args==['preflight'] and len(earlier)==1
        assert not (OUT/'FAILED_STOP_SECOND.json').exists()
    if earlier:tag += '_attempt2'
    verify()
    now=lambda:datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    rec={'args':args,'command':child,'started_at':now(),'status':'running'};registry.append(rec)
    registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
    logs=OUT/'console';logs.mkdir(exist_ok=True)
    start=time.perf_counter()
    try:
        print('COMMAND:',subprocess.list2cmdline(child),flush=True)
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8'
        with (logs/(tag+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(child,cwd=WORK,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        rec['exit_code']=code
        if code:raise RuntimeError(f'Protocol child failed: {args}, exit={code}; do not retry')
        verify();rec.update(status='completed',finished_at=now(),seconds=time.perf_counter()-start)
    except BaseException as e:
        rec.update(status='failed',finished_at=now(),seconds=time.perf_counter()-start,error=str(e))
        failure=OUT/('FAILED_STOP_SECOND.json' if (OUT/'FAILED_STOP.json').exists() else 'FAILED_STOP.json')
        failure.write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    finally:registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
    print('STEP COMPLETE:',args,'seconds',round(rec['seconds'],2),flush=True)
if __name__=='__main__':main()
