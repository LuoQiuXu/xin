"""Verbatim seed123 training helper functions, independent ROOT."""
from common import *

def tensor_hash(t):
    return array_hash(t.detach().cpu().numpy())

def augment(rec,don,generator,transform):
    images = np.load(ROOT/'train_images.npy',mmap_mode='r',allow_pickle=False)
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        ri,di = [],[]
        for r,d in zip(rec,don):
            di.append(transform(Image.fromarray(images[d])))
            ri.append(transform(Image.fromarray(images[r])))
        generator.set_state(torch.get_rng_state())
    assert torch.equal(before,torch.get_rng_state())
    return torch.stack(ri),torch.stack(di)

def fuse(receiver,donor,indices):
    i = torch.from_numpy(indices.astype(np.int64).copy()).cuda()
    mask = torch.zeros((len(indices),64),dtype=torch.bool,device='cuda')
    mask.scatter_(1,i,True)
    assert (mask.sum(1) == 16).all()
    assert np.all(np.diff(np.sort(indices,axis=1),axis=1) > 0)
    return torch.where(mask[:,:,None,None],donor,receiver),mask

def scheduler():
    tree = ast.parse((RUNTIME/'train_stage2.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'adjust_learning_rate')
    ns = {'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original_stage2_scheduler','exec'),ns)
    return ns['adjust_learning_rate']

def optimizer_for(cfg,cls):
    opt = torch.optim.SGD([{'params':cls.parameters()}],cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
    assert not opt.state
    assert {id(p) for group in opt.param_groups for p in group['params']} == {id(p) for p in cls.parameters()}
    return opt
