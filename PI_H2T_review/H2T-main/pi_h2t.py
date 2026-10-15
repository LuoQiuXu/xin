"""PI-H2T paper-form implementation; legacy methods.py is untouched."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from methods import H2Tv2


def pool(x):
    if x.ndim != 4:
        raise ValueError('H2TF input must be a feature map [B,C,H,W]')
    return F.avg_pool2d(x, kernel_size=x.shape[-2:]).flatten(1)


def fusion_ratio(features, labels, weight, mode='paper_theoretical_bounds', ratio_detach=True):
    """Current classifier w_y; r is RECIPIENT mass. Detach defaults to True."""
    if mode not in ('paper_theoretical_bounds', 'legacy_h2tv2_ablation'):
        raise ValueError(mode)
    cosine = F.cosine_similarity(features, weight[labels], dim=1).clamp(-1, 1)
    distance = 1 - cosine
    if mode == 'paper_theoretical_bounds':
        ratio = (distance / 2).clamp(0, 1)
    else:
        ratio = ((1 / (1 + distance) - .5) / .5 * .4 + .6)
    return ratio.detach() if ratio_detach else ratio


def h2tf(recipient_map, donor_map, labels, weight, mode='paper_theoretical_bounds', ratio_detach=True):
    recipient, donor = pool(recipient_map), pool(donor_map)
    if recipient.shape != donor.shape:
        raise ValueError('Recipient/donor shapes must match')
    r = fusion_ratio(recipient, labels, weight, mode, ratio_detach)
    if mode == 'legacy_h2tv2_ablation':
        # Exercise the unchanged upstream H2Tv2, not the formal default.
        dis = 1 - F.normalize(recipient, dim=1) @ F.normalize(weight.detach(), dim=1).T
        fused = H2Tv2()(recipient, donor, dis.clamp(0, 2), labels, labels)
    else:
        fused = r[:, None] * recipient + (1-r[:, None]) * donor
    return fused, r


def freeze_extractor(model):
    model.requires_grad_(False)
    model.eval()  # Freeze BN running statistics as requested, not just gradients.


def save_ratio_stats(values, path, epoch, mode):
    r = torch.cat(values).float().cpu()
    if r.numel() == 0 or not torch.isfinite(r).all():
        raise ValueError('No finite fusion ratios')
    counts = torch.histc(r, bins=20, min=0, max=1).long()
    result = dict(epoch=epoch, mode=mode, count=r.numel(), mean=r.mean().item(),
                  std=r.std(unbiased=False).item(), min=r.min().item(), max=r.max().item(),
                  histogram_edges=torch.linspace(0, 1, 21).tolist(), histogram_counts=counts.tolist())
    Path(path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
