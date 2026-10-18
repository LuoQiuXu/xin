"""Execution guards only; no model, sampling, mask, loss or optimizer changes."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
EVENTS=[]
INSTALLED=False

def install(work,results):
    global INSTALLED
    if INSTALLED:return
    INSTALLED=True
    work=Path(work).resolve();results=Path(results).resolve()
    def inside(path,base):
        return path==base or base in path.parents
    def pathof(value):
        if isinstance(value,(str,bytes,os.PathLike)):
            return Path(os.fsdecode(value)).resolve()
        return None
    def historical(value):
        p=pathof(value)
        return p is not None and inside(p,work) and not inside(p,results)
    def audit(event,args):
        if event=='open' and args:
            p=pathof(args[0])
            if p is not None and p.name.casefold()=='test' and p.parent.name.casefold()=='cifar-100-python':
                EVENTS.append({'event':event,'path':str(p),'reason':'official TEST forbidden'})
                raise RuntimeError('Official CIFAR100 TEST access forbidden: '+str(p))
            if historical(args[0]):
                mode=args[1] if len(args)>1 else None; flags=args[2] if len(args)>2 else 0
                write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
                if write:
                    EVENTS.append({'event':event,'path':str(args[0]),'reason':'historical write forbidden'})
                    raise RuntimeError('Historical project files are read-only: '+str(args[0]))
        if event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink','os.chmod','os.utime'):
            if any(historical(v) for v in args[:2]):
                EVENTS.append({'event':event,'path':str(args[0]),'reason':'historical mutation forbidden'})
                raise RuntimeError('Historical project mutation forbidden: '+str(args[0]))
    sys.addaudithook(audit)

def run_action(main,results):
    started=time.perf_counter();completed=False;error=None
    try:
        main();assert not EVENTS;completed=True
    except BaseException:
        error=traceback.format_exc();raise
    finally:
        import torch
        tag='_'.join(sys.argv[1:]).replace('--','')
        info={'completed':completed,'argv':sys.argv,'seconds':time.perf_counter()-started,'guard_events':EVENTS,
              'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'traceback':error}
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(0),peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                        peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
        p=Path(results)/'guards'/f'{tag}.json';p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:json.dump(info,f,ensure_ascii=False,indent=2)
