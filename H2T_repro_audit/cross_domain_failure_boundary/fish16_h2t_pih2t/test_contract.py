import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import models
from data import load_rows, safe_image_path, schedules, Views


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.plain = models.FishModel(False).eval()
        cls.pif = models.FishModel(True).eval()
        cls.rows = load_rows()

    def test_official_weight_identity_and_strict_compatibility(self):
        self.assertEqual(models.sha(models.ORIGINAL_WEIGHT), models.ORIGINAL_SHA)
        info = json.loads((models.CACHE / "ADAPTER_PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(info["dropped_only"], ["fc.weight", "fc.bias"])
        self.assertEqual(len(info["legacy_zero_bn_counters"]), 53)
        self.assertEqual(info["learned_or_floating_keys_filled"], [])
        self.assertTrue(info["strict_load"])
        self.assertEqual(models.sha(models.SAFE_WEIGHT), info["safe_sha256"])

    def test_common_backbone_and_classifier_initialization(self):
        for k, v in self.plain.extractor.backbone.state_dict().items():
            self.assertTrue(torch.equal(v, self.pif.extractor.backbone.state_dict()[k]), k)
        for k, v in self.plain.classifier.state_dict().items():
            self.assertTrue(torch.equal(v, self.pif.classifier.state_dict()[k]), k)
        self.assertEqual(sum(p.numel() for p in self.pif.parameters()) -
                         sum(p.numel() for p in self.plain.parameters()), 2)

    def test_original_backbone_output_shape(self):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            self.assertEqual(self.plain.extractor(x).shape, (2, 2048, 7, 7))
            self.assertEqual(self.pif.extractor(x).shape, (2, 2048, 7, 7))

    def test_pif_formula_two_parameters_and_permutation(self):
        pif = models.pif_source.PermutationInvariantFusion()
        self.assertEqual(sum(p.numel() for p in pif.parameters()), 2)
        with torch.no_grad():
            pif.fuse.weight.copy_(torch.tensor([[[.2], [.7]]]))
        x = torch.randn(2, 2048, 3, 3, requires_grad=True)
        expected = .2 * (x - x.mean(1, keepdim=True)) + .7 * x
        torch.testing.assert_close(pif(x), expected)
        perm = torch.randperm(2048)
        torch.testing.assert_close(pif(x[:, perm]), pif(x)[:, perm])
        pif(x).square().mean().backward()
        self.assertTrue(torch.isfinite(pif.fuse.weight.grad).all())
        self.assertGreater(pif.fuse.weight.grad.abs().sum().item(), 0)

    def test_original_h2t_shared_512_channels_and_unchanged_recipient(self):
        r = torch.ones(3, 2048, 2, 2)
        d = torch.full_like(r, 7)
        with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
            fused, selected = models.h2t_with_indices(r, d)
        self.assertEqual(len(selected), 512)
        self.assertEqual(len(set(selected.tolist())), 512)
        mask = torch.zeros(2048, dtype=torch.bool)
        mask[selected] = True
        self.assertTrue(torch.equal(fused[:, mask], d[:, mask]))
        self.assertTrue(torch.equal(fused[:, ~mask], r[:, ~mask]))
        self.assertTrue(torch.equal(r, torch.ones_like(r)))
        self.assertTrue(torch.equal(fused[0], fused[1]))

    def test_pi_ratio_recipient_donor_mass_and_detach(self):
        r = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]]).reshape(3, 2, 1, 1)
        d = torch.full_like(r, 3)
        classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
        labels = torch.zeros(3, dtype=torch.long)
        fused, ratio = models.stage2_features(r, d, labels, classifier, "pi")
        torch.testing.assert_close(ratio, torch.tensor([0., 1., .5]))
        torch.testing.assert_close(fused, ratio[:, None] * r.flatten(1) + (1-ratio[:, None]) * d.flatten(1))
        self.assertFalse(ratio.requires_grad)
        classifier(fused).square().mean().backward()
        self.assertGreater(classifier.weight.grad.abs().sum().item(), 0)

    def test_stage2_parameter_and_bn_freeze_with_classifier_update(self):
        model = copy.deepcopy(self.plain)
        model.freeze()
        self.assertFalse(model.training)
        before = models.state_sha(model.extractor)
        self.assertTrue(all(not p.requires_grad for p in model.extractor.parameters()))
        self.assertTrue(all(not m.training for m in model.extractor.modules() if isinstance(m, nn.BatchNorm2d)))
        with torch.no_grad():
            features = model.extractor(torch.randn(2, 3, 224, 224))
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=.01)
        weight_before = model.classifier.weight.detach().clone()
        nn.functional.cross_entropy(model.classifier(features.mean((2, 3))), torch.tensor([0, 1])).backward()
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(model.classifier.weight, weight_before))
        self.assertEqual(before, models.state_sha(model.extractor))
        self.assertTrue(all(p.grad is None for p in model.extractor.parameters()))

    def test_schedule_train_only_and_reproducible(self):
        first, second = schedules(self.rows), schedules(self.rows)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        train, val, stage1, recipients, donors = first
        self.assertEqual(set(stage1), set(train))
        self.assertTrue(set(recipients) <= set(train) and set(donors) <= set(train))
        self.assertFalse(set(train) & set(val))
        self.assertEqual({self.rows[int(i)]["target"] for i in recipients}, set(range(16)))

    def test_matched_real_recipient_view_and_rng_restoration(self):
        _, _, _, recipients, donors = schedules(self.rows)
        crt = Views(self.rows, recipients[:2], True)
        fused = Views(self.rows, recipients[:2], True, donors=donors[:2])
        state = torch.get_rng_state()
        for i in range(2):
            self.assertTrue(torch.equal(crt[i]["recipient"], fused[i]["recipient"]))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(crt[0]["recipient"].shape, (3, 224, 224))

    def test_path_escape_masks_and_excluded_classes_rejected(self):
        for path in ("../../escape.png", "mask_01/file.png", "fish_17/file.png"):
            with self.subTest(path=path), self.assertRaises((AssertionError, ValueError)):
                safe_image_path({"relative_path": path})


if __name__ == "__main__":
    unittest.main(verbosity=2)
