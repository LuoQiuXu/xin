"""Read-only instrumentation for the explicitly requested real one-epoch dry run."""
import json
import torch


class DryRunAudit:
    def __init__(self, model, classifier, criterion, stage2):
        self.stage2 = stage2
        self.before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        self.head_before = {k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()}
        self.pif_grad = False
        self.head_grad = False
        self.losses = []
        self.handles = [criterion.register_forward_hook(self.loss_hook)]
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.handles.append(p.register_hook(self.gradient_hook('pif' if 'pif.' in name else 'backbone')))
        for p in classifier.parameters():
            self.handles.append(p.register_hook(self.gradient_hook('classifier')))

    def loss_hook(self, module, inputs, output):
        assert torch.isfinite(output).all(), 'Non-finite loss'
        assert torch.isfinite(inputs[0]).all(), 'Non-finite logits'
        self.losses.append(float(output.detach()))

    def gradient_hook(self, kind):
        def check(grad):
            assert torch.isfinite(grad).all(), 'Non-finite gradient'
            if kind == 'pif':
                self.pif_grad |= bool((grad != 0).all())
            if kind == 'classifier':
                self.head_grad |= bool((grad != 0).any())
        return check

    def finish(self, output, model, classifier, accuracy, ece, lr):
        after = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        pif_key = 'module.pif.fuse.weight'
        changed = [k for k in after if not torch.equal(after[k], self.before[k])]
        head_changed = any(not torch.equal(v.detach().cpu(), self.head_before[k]) for k,v in classifier.state_dict().items())
        checkpoint = torch.load(output/'current.pth.tar', map_location='cpu', weights_only=False)
        assert all(torch.equal(v, checkpoint['state_dict_model'][k]) for k,v in after.items())
        assert all(torch.equal(v.detach().cpu(), checkpoint['state_dict_classifier'][k]) for k,v in classifier.state_dict().items())
        assert all(torch.isfinite(v).all() for v in after.values())
        assert all(torch.isfinite(v).all() for v in classifier.state_dict().values())
        assert torch.isfinite(torch.tensor([accuracy, ece])).all()
        if self.stage2:
            assert not changed and all(p.grad is None for p in model.parameters())
        else:
            assert self.pif_grad and bool((after[pif_key] != self.before[pif_key]).all())
        assert head_changed and self.head_grad
        result = dict(status='PASS', stage=2 if self.stage2 else 1, epochs=checkpoint['epoch'],
            accuracy=accuracy, ece=ece, lr=lr, loss_min=min(self.losses), loss_max=max(self.losses),
            loss_evaluations=len(self.losses), pif_before=self.before[pif_key].flatten().tolist(),
            pif_after=after[pif_key].flatten().tolist(), pif_two_gradients_observed=self.pif_grad,
            extractor_changed_state_entries=len(changed), classifier_changed=head_changed,
            classifier_gradient_observed=self.head_grad, checkpoint_roundtrip_exact=True,
            no_nan_inf=True, clean_validation='original train_stage2.validate: PIF retained; no H2TF')
        (output/'dry_run_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        for h in self.handles:
            h.remove()
        print(json.dumps(result), flush=True)
