"""Run one preregistered command with read-only historical inputs and console evidence.
No changes to losses.py, run_pilot.py, or any reused training/policy helper.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,runpy,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
OUT=HERE/'results'
PYTHON=Path(r'D:\app\minicondaaa\envs\xiaoju\python.exe')
ARMS=('LowVS','LowVS-CleanCE','LowVS-TopCompetitor')
EVENTS=[]

def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    p=Path(p);assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
def verify():
    for filename in ('execution_source_lock.json','historical_manifest.json'):
        for relative,expected in json.loads((OUT/filename).read_text(encoding='utf-8')).items():
            assert sha(WORK/relative)==expected,(filename,'Changed source or historical input',relative)

def install():
    def pathof(v):return Path(os.fsdecode(v)).resolve() if isinstance(v,(str,bytes,os.PathLike)) else None
    def historical(v):
        p=pathof(v)
        return p is not None and p.is_relative_to(WORK) and not p.is_relative_to(OUT)
    def audit(event,args):
        if event=='open' and args:
            p=pathof(args[0])
            if p is not None and p.name.casefold()=='test' and p.parent.name.casefold()=='cifar-100-python':
                EVENTS.append({'event':event,'path':str(p),'reason':'official TEST forbidden'})
                raise RuntimeError('Official CIFAR100 TEST access forbidden: '+str(p))
            if historical(args[0]):
                mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
                write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
                if write:
                    EVENTS.append({'event':event,'path':str(args[0]),'reason':'historical write forbidden'})
                    raise RuntimeError('Historical project files are read-only: '+str(args[0]))
        if event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink','os.chmod','os.utime') and any(historical(v) for v in args[:2]):
            EVENTS.append({'event':event,'path':str(args[0]),'reason':'historical mutation forbidden'})
            raise RuntimeError('Historical project mutation forbidden: '+str(args[0]))
    sys.addaudithook(audit)

def child(script,args,tag):
    assert script in ('losses.py','run_pilot.py')
    install();sys.argv=[str(HERE/script),*args];started=time.perf_counter();completed=False;error=None;g={}
    try:
        verify();g=runpy.run_path(str(HERE/script),run_name='__main__');verify();assert not EVENTS;completed=True
    except BaseException:
        error=traceback.format_exc();raise
    finally:
        import torch,torchvision,numpy
        data={'completed':completed,'argv':sys.argv,'seconds':time.perf_counter()-started,'guard_events':EVENTS,
              'traceback':error,'python':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,
              'numpy':numpy.__version__,'cuda':torch.version.cuda}
        if torch.cuda.is_available():
            data.update(GPU=torch.cuda.get_device_name(0),peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                        peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
        if 'old' in g:data['reused_TEST_guard_events']=g['old'].AUDIT_EVENTS
        path=OUT/'guards'/(tag+'.json');assert not path.exists();dump(path,data)

def main():
    assert sys.executable.casefold()==str(PYTHON).casefold(),sys.executable
    if len(sys.argv)>1 and sys.argv[1]=='_child':
        child(sys.argv[2],sys.argv[3:-1],sys.argv[-1]);return
    args=sys.argv[1:]
    expected=[['losses'],['preflight']]+[['train','--seed',str(s),'--arm',a] for s in (124,125) for a in ARMS]+[['report']]
    if (OUT/'FAILED_STOP.json').exists():raise RuntimeError('A previous step failed: stop without retry')
    registry_path=OUT/'execution_registry.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else []
    assert len(registry)<len(expected) and args==expected[len(registry)],('Out of sequence',args)
    assert all(r['status']=='completed' for r in registry)
    verify();tag='_'.join(args).replace('--','');script='losses.py' if args==['losses'] else 'run_pilot.py'
    scriptargs=[] if script=='losses.py' else args
    command=[str(PYTHON),'-B','-u',str(HERE/'execute_step.py'),'_child',script,*scriptargs,tag]
    actual=[str(PYTHON),'-B','-u',str(HERE/script),*scriptargs]
    rec={'args':args,'actual_protocol_command':actual,'command_with_guards':command,'started_at':stamp(),'status':'running'}
    registry.append(rec);dump(registry_path,registry);started=time.perf_counter()
    try:
        logpath=OUT/'console'/(tag+'.log');logpath.parent.mkdir(exist_ok=True)
        print('COMMAND:',subprocess.list2cmdline(actual),flush=True)
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8'
        with logpath.open('x',encoding='utf-8') as log:
            p=subprocess.Popen(command,cwd=WORK,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=p.wait()
        rec['exit_code']=code
        if code:raise RuntimeError(f'Protocol command failed; stop: {args}, exit={code}')
        verify();rec.update(status='completed',finished_at=stamp(),seconds=time.perf_counter()-started)
    except BaseException:
        rec.update(status='failed',finished_at=stamp(),seconds=time.perf_counter()-started,traceback=traceback.format_exc())
        dump(OUT/'FAILED_STOP.json',rec);raise
    finally:dump(registry_path,registry)
    print('STEP COMPLETE',args,'seconds',round(rec['seconds'],2),flush=True)
if __name__=='__main__':main()
