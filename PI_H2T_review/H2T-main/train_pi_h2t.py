"""Review-first CIFAR100-LT/CE entry point; never called by legacy scripts."""
import argparse
import json
import logging
from pathlib import Path
import torch
from utils import config, update_config, SetRandomSeed
from datasets.cifar100 import CIFAR100_LT
from models.resnet_cifar import resnet32_fe_pif, Classifier
from loss import CrossEntropyLoss
from train_stage1 import train as train_stage1, adjust_learning_rate as lr_stage1
from train_stage2 import validate, adjust_learning_rate as lr_stage2
from pi_h2t import freeze_extractor, h2tf, save_ratio_stats


def load_stage1(path, model, classifier):
    if not Path(path).is_file():
        raise FileNotFoundError('A trained PIF Stage-I checkpoint is required: '+str(path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict_model']
    if not any('pif.fuse.weight' in k for k in state):
        raise ValueError('Refusing a non-PIF checkpoint')
    model.load_state_dict(state, strict=True)
    classifier.load_state_dict(checkpoint['state_dict_classifier'], strict=True)
    return checkpoint


def train_stage2(loader, model, classifier, criterion, optimizer, cfg, mode):
    freeze_extractor(model)
    classifier.train()
    ratios = []
    for images, labels, meta in loader:
        images, labels = images.cuda(), labels.cuda()
        donor = meta['sample_image'].cuda()
        with torch.no_grad():
            recipient_map, donor_map = model(images), model(donor)
        features, r = h2tf(recipient_map, donor_map, labels, classifier.module.fc.weight, mode,
                           ratio_detach=cfg.PI_H2T.ratio_detach)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(classifier(features), labels)
        loss.backward()
        optimizer.step()
        ratios.append(r.detach().cpu())
    return ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ratio-mode', choices=['paper_theoretical_bounds', 'legacy_h2tv2_ablation'], default='paper_theoretical_bounds')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dry-run-one-epoch', action='store_true', help='Stop after one real epoch; preserve the configured LR horizon')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    if config.dataset != 'cifar100' or config.backbone != 'resnet32_fe_pif' or config.loss.type != 'CrossEntropyLoss':
        raise ValueError('This entry point is restricted to CIFAR100 / PIF / CE')
    if config.distributed or config.multiprocessing_distributed:
        raise ValueError('Distributed PI runner is not validated; use single-process DataParallel')
    stage2 = config.mode == 'stage2'
    if config.mode not in ('stage1', 'stage2'):
        raise ValueError('Unknown training stage')
    if stage2 and (config.mixup or config.shift_bn or not config.sampler.dual_sample.enable):
        raise ValueError('Stage-II requires no MixUp, frozen BN and dual sampling')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('Official training/evaluation helpers require CUDA')
    SetRandomSeed(args.seed)
    model = torch.nn.DataParallel(resnet32_fe_pif()).cuda()
    classifier = torch.nn.DataParallel(Classifier(64, 100)).cuda()
    if stage2:
        load_stage1(config.resume, model, classifier)
        freeze_extractor(model)
    elif config.resume:
        raise ValueError('Stage-I starts from scratch; resume is deliberately unsupported')
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.output/'train.log', encoding='utf-8')])
    logger = logging.getLogger('pi_h2t')
    logger.info('ratio_detach=%s', config.PI_H2T.ratio_detach)
    (args.output/'config.yaml').write_text(config.dump(), encoding='utf-8')
    (args.output/'protocol.json').write_text(json.dumps(dict(seed=args.seed, ratio_mode=args.ratio_mode,
        ratio_reference='current_classifier', ratio_detach=config.PI_H2T.ratio_detach, pif_init='Conv1d default', stage2_bn='frozen',
        stage1_checkpoint=config.resume, evaluation='original clean validate; best and current saved'), indent=2), encoding='utf-8')
    dataset = CIFAR100_LT(config, False, root=config.data_path, imb_factor=config.imb_factor,
                          batch_size=config.batch_size, num_works=config.workers)
    criterion = CrossEntropyLoss(config=config, cls_num_list=dataset.cls_num_list).cuda()
    params = [{'params': classifier.parameters()}] if stage2 else [{'params': model.parameters()}, {'params': classifier.parameters()}]
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    audit = None
    if args.dry_run_one_epoch:
        from pi_dry_run_audit import DryRunAudit
        audit = DryRunAudit(model, classifier, criterion, stage2)
    best = -1.
    for epoch in range(1 if args.dry_run_one_epoch else config.num_epochs):
        (lr_stage2 if stage2 else lr_stage1)(optimizer, epoch, config)
        if stage2:
            values = train_stage2(dataset.train_instance, model, classifier, criterion, optimizer, config, args.ratio_mode)
            stats = save_ratio_stats(values, args.output/f'ratio_epoch_{epoch+1:03d}.json', epoch+1, args.ratio_mode)
            logger.info('r mean/std/min/max: %.6f %.6f %.6f %.6f', stats['mean'], stats['std'], stats['min'], stats['max'])
        else:
            train_stage1(dataset.train_instance, model, classifier, criterion, optimizer, epoch, config, logger)
        accuracy, ece = validate(dataset.eval, model, classifier, criterion, config, logger)
        score = float(accuracy)
        state = dict(epoch=epoch+1, state_dict_model=model.state_dict(), state_dict_classifier=classifier.state_dict(),
                     optimizer=optimizer.state_dict(), best_acc1=max(best, score), its_ece=float(ece), mode=config.mode,
                     ratio_mode=args.ratio_mode, seed=args.seed, config=config.dump())
        torch.save(state, args.output/'current.pth.tar')
        if score > best:
            best = score
            torch.save(state, args.output/'model_best.pth.tar')
        if audit is not None:
            audit.finish(args.output, model, classifier, score, float(ece), optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    main()
