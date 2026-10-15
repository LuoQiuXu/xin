"""Synthetic unit smoke tests only: no dataset loading or experiment training."""
import json
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from models.resnet_cifar import resnet32_fe, resnet32_fe_pif, Classifier
from pi_h2t import pool, h2tf, fusion_ratio, freeze_extractor, save_ratio_stats
from train_pi_h2t import load_stage1, train_stage2
from utils import config
from loss import CrossEntropyLoss


def main():
    torch.manual_seed(123)
    torch.set_num_threads(2)
    model = resnet32_fe_pif()
    head = Classifier(64, 100)
    plain = resnet32_fe()
    assert sum(p.numel() for p in model.pif.parameters()) == 2
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in plain.parameters()) == 2
    plain.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('pif.')}, strict=True)
    x = torch.randn(3, 3, 32, 32)
    y = torch.tensor([0, 30, 99])
    model.eval(); plain.eval()
    raw, out = plain(x), model(x)
    a, b = model.pif.fuse.weight.flatten()
    assert out.shape == raw.shape == (3, 64, 8, 8)
    torch.testing.assert_close(out, a*(raw-raw.mean(1, keepdim=True))+b*raw)
    torch.testing.assert_close(head(out), head(pool(out)))
    model.train()
    F.cross_entropy(head(model(x)), y).backward()
    grad = model.pif.fuse.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and (grad.abs() > 0).all()
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    freeze_extractor(model)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        ft, fh = model(x), model(x.flip(0))
        fused, r = h2tf(ft, fh, y, head.fc.weight)
    assert fused.shape == (3, 64) and ((r >= 0) & (r <= 1)).all()
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    F.cross_entropy(head(fused), y).backward()
    assert all(p.grad is None for p in model.parameters())
    assert head.fc.weight.grad is not None and head.fc.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    f = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]])
    w = torch.tensor([[1., 0.]])
    torch.testing.assert_close(fusion_ratio(f, torch.zeros(3, dtype=torch.long), w), torch.tensor([0., 1., .5]))
    # Ratios must not depend on batch composition.
    torch.testing.assert_close(fusion_ratio(f[:1], torch.tensor([0]), w), torch.tensor([0.]))
    ablated, ra = h2tf(ft, fh, y, head.fc.weight, 'legacy_h2tv2_ablation')
    torch.testing.assert_close(ablated, ra[:,None]*pool(ft)+(1-ra[:,None])*pool(fh))
    for detach in (True, False):
        head.zero_grad(set_to_none=True)
        mixed, ratio = h2tf(ft, fh, y, head.fc.weight, ratio_detach=detach)
        assert ratio.requires_grad == (not detach)
        if not detach:
            ratio_gradient = torch.autograd.grad(ratio.sum(), head.fc.weight, retain_graph=True)[0]
            assert torch.isfinite(ratio_gradient).all() and ratio_gradient.abs().sum() > 0
        F.cross_entropy(head(mixed), y).backward()
        assert torch.isfinite(head.fc.weight.grad).all() and head.fc.weight.grad.abs().sum() > 0
    detached, _ = h2tf(ft, fh, y, head.fc.weight, ratio_detach=True)
    attached, _ = h2tf(ft, fh, y, head.fc.weight, ratio_detach=False)
    torch.testing.assert_close(detached, attached, rtol=0, atol=0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        torch.save({'state_dict_model':model.state_dict(), 'state_dict_classifier':head.state_dict()}, tmp/'pif.pt')
        load_stage1(tmp/'pif.pt', resnet32_fe_pif(), Classifier(64,100))
        torch.save({'state_dict_model':plain.state_dict(), 'state_dict_classifier':head.state_dict()}, tmp/'plain.pt')
        try:
            load_stage1(tmp/'plain.pt', resnet32_fe_pif(), Classifier(64,100))
            raise AssertionError('Non-PIF checkpoint accepted')
        except ValueError:
            pass
        stats = save_ratio_stats([r], tmp/'r.json', 1, 'paper_theoretical_bounds')
        assert sum(stats['histogram_counts']) == 3
    for file, epochs in [('cifar100_imb001_stage1_ce_pif.yaml', 200), ('cifar100_imb001_stage2_ce_pi_h2t.yaml', 10)]:
        cfg = config.clone()
        cfg.merge_from_file(str(Path(__file__).parent/'config/cifar100'/file))
        assert cfg.num_epochs == epochs and cfg.backbone == 'resnet32_fe_pif'
    cuda_integration = False
    if torch.cuda.is_available():
        gm = torch.nn.DataParallel(resnet32_fe_pif()).cuda()
        gh = torch.nn.DataParallel(Classifier(64, 100)).cuda()
        frozen = {k:v.clone() for k,v in gm.state_dict().items()}
        opt = torch.optim.SGD(gh.parameters(), lr=.01)
        values = train_stage2([(x, y, {'sample_image':x.flip(0)})], gm, gh, CrossEntropyLoss(cfg).cuda(), opt, cfg, 'paper_theoretical_bounds')
        assert len(values) == 1 and all(torch.equal(v, frozen[k]) for k,v in gm.state_dict().items())
        assert gh.module.fc.weight.grad is not None
        cuda_integration = True
    print(json.dumps(dict(status='PASS', pif_parameters=2, feature_shape=list(out.shape), h2tf_shape=list(fused.shape),
                         stage1_two_pif_gradients=True, stage2_extractor_and_bn_unchanged=True,
                         stage2_classifier_gradient=True, pooled_classifier_equivalence=True,
                         ratio_endpoints_and_batch_independence=True, strict_checkpoint_loading=True,
                         cuda_synthetic_stage2_integration=cuda_integration,
                         ratio_detach_true_backward=True, ratio_detach_false_backward=True,
                         ratio_false_gradient_to_classifier=True)))


if __name__ == '__main__':
    main()
