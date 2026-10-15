"""Fresh independent classifier/SGD initialization from the one locked Clean checkpoint."""
from common import *

PAIR_KEYS = ['epoch','batch','size','indices_sha256','images_sha256','features_sha256',
    'augmentation_state_after_sha256','ambient_cpu_rng_sha256','ambient_cuda_rng_sha256',
    'gumbel_sha256','online_gap_sha256','online_scores_sha256','all_two_masks_sha256']


def plan_dir(seed):
    assert seed in SEEDS
    return ROOT/'plans'/str(seed)


def outdir(seed,arm):
    assert seed in SEEDS and arm in ARMS
    return ROOT/f'{"uniform" if arm == ARMS[0] else "lowvs"}_seed{seed}'


def imports():
    sys.path.insert(0,str(RUNTIME))
    from utils import config
    from models.resnet_cifar import resnet32_fe,Classifier
    from loss import CrossEntropyLoss
    import methods
    cfg = config.clone();cfg.defrost()
    cfg.merge_from_file(str(ROOT/'stage2_original_effective.yaml'))
    cfg.name = ROOT.name
    cfg.data_path = str(DATA)
    cfg.resume = ''
    cfg.shift_bn = False
    assert cfg.backbone == 'resnet32_fe' and cfg.num_classes == 100
    assert cfg.batch_size == 128 and cfg.num_epochs == 10 and cfg.workers == 0
    assert cfg.lr == .1 and cfg.momentum == .9 and cfg.weight_decay == .0005
    assert not cfg.mixup and not cfg.deterministic and cfg.H2T == .25
    assert cfg.loss.type == 'CrossEntropyLoss' and not cfg.loss.CE.reweight_CE
    assert cfg.sampler.type == 'weighted sampler' and cfg.sampler.weighted_sampler.type == 'balance'
    assert cfg.sampler.dual_sample.enable and cfg.sampler.dual_sample.type == 'long-tailed'
    tree = ast.parse((RUNTIME/'datasets/cifar100.py').read_text(encoding='utf-8'))
    c = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'CIFAR100_LT')
    init = next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name == '__init__')
    tr = {}
    for n in ast.walk(init):
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['train_transform','eval_transform']:
            tr[n.targets[0].id] = eval(compile(ast.Expression(n.value),'official_transforms','eval'),{'transforms':transforms})
    assert len(tr) == 2
    return cfg,tr,resnet32_fe,Classifier,CrossEntropyLoss,methods


def frozen_models(seed):
    assert seed in SEEDS
    p = ROOT/'locked/clean_stage1_best.pth'
    assert sha(p) == S1_SHA and sha(ROOT/'class_feature_statistics.npz') == STAT_SHA
    cfg,tr,ResNet,Classifier,CE,methods = imports()
    seed_all(seed,False)
    q = torch.load(p,map_location='cpu',weights_only=False)
    assert q['epoch'] == 184 and q['seed'] == 42 and not q['TEST_used']
    assert q['selection_partition'] == 'official TRAIN holdout'
    bb = torch.nn.DataParallel(ResNet()).cuda()
    cls = torch.nn.DataParallel(Classifier(64,100)).cuda()
    bb.load_state_dict(q['state_dict_model'],strict=True)
    cls.load_state_dict(q['state_dict_classifier'],strict=True)
    bb.eval().requires_grad_(False)
    cls.train()
    assert not bb.module.feature_tuning and not hasattr(bb.module,'conv1d')
    assert sum(p.numel() for p in bb.parameters()) == 463504
    assert sum(p.numel() for p in cls.parameters()) == 6500
    assert all(v.detach().cpu().numpy().tobytes() == q['state_dict_model'][k].numpy().tobytes() for k,v in bb.state_dict().items())
    assert all(v.detach().cpu().numpy().tobytes() == q['state_dict_classifier'][k].numpy().tobytes() for k,v in cls.state_dict().items())
    return cfg,tr,bb,cls,CE,methods,p,q


def check_sources():
    lock = read(ROOT/'source_lock.json')
    for category in ['files','data_files']:
        for p,v in lock[category].items():
            assert sha(ROOT/p) == v, ('locked source or input changed',p)

