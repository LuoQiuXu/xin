"""CPU correctness tests; synthetic fixtures are not experiment observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from officehome_data import (
    ARMS, CSV_SHA256, DOMAINS, SEEDS, OfficeHomeImages, Row, cell_counts,
    paired_changes, prediction_metrics, read_protocol, sha256, validate_rows,
    validate_test_lock, write_json,
)
from officehome_domain_probe import (
    HERE, bn_state, conditional_donors, freeze_stage2, make_plan,
    replace_channels, stage1_mode, state_hash,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 65)

    def forward(self, x):
        return self.fc(torch.relu(self.bn1(self.conv1(x))).mean((2, 3)))


class OfficeHomeCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_pinned_csv_split_disjointness(self):
        rows = read_protocol(HERE / "protocol" / "OfficeHome.csv")
        sets = {s: {r.path for r in rows if r.split == s} for s in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(sets[a] & sets[b])
        self.assertEqual(sum(map(len, sets.values())), len(rows))
        self.assertTrue((cell_counts(rows, "val") == 5).all())

    def test_duplicate_overlap_rejected(self):
        one = Row(0, 0, 0, "Art/Alarm_Clock/one.jpg", "train")
        two = Row(1, 0, 0, one.path, "val")
        with self.assertRaisesRegex(ValueError, "Duplicate/overlapping"):
            validate_rows([one, two])

    def test_gap_substitution_commutes_and_copies_whole_channels(self):
        gen = torch.Generator().manual_seed(71)
        recipient = torch.randn(3, 512, 7, 7, generator=gen, dtype=torch.float64)
        donor = torch.randn(3, 512, 7, 7, generator=gen, dtype=torch.float64)
        mask = torch.randperm(512, generator=gen)[:128]
        original = recipient.clone()
        y = torch.tensor([2, 9, 17])
        label_before = y.clone()
        fused = replace_channels(recipient, donor, mask)
        torch.testing.assert_close(fused.mean((2, 3)),
                                   replace_channels(recipient.mean((2, 3)), donor.mean((2, 3)), mask),
                                   rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(fused[:, mask], donor[:, mask]))
        keep = torch.ones(512, dtype=torch.bool)
        keep[mask] = False
        self.assertTrue(torch.equal(fused[:, keep], recipient[:, keep]))
        self.assertTrue(torch.equal(original, recipient))
        self.assertTrue(torch.equal(y, label_before))
        self.assertEqual(len(mask.unique()), 128)
        # Same channel identity for every image, not per-sample selection.
        for b in range(3):
            self.assertTrue(torch.equal(fused[b, mask], donor[b, mask]))

    def test_domain_constraints_and_only_empty_pool_fallback(self):
        labels = np.array([0, 0, 1, 2])
        domains = np.array([0, 1, 2, 0])
        receiver = np.array([0, 0, 2, 3])
        global_ = np.array([2, 3, 3, 1])
        matched, cross, mf, cf = conditional_donors(
            labels, domains, receiver, global_, np.random.default_rng(2), np.random.default_rng(3))
        np.testing.assert_array_equal(labels[matched], labels[global_])
        np.testing.assert_array_equal(labels[cross], labels[global_])
        np.testing.assert_array_equal(mf, [True, False, True, False])
        np.testing.assert_array_equal(cf, [False, True, False, False])
        np.testing.assert_array_equal(matched[mf], global_[mf])
        np.testing.assert_array_equal(cross[cf], global_[cf])

    def test_plan_reproducibility_balance_class_pairing_and_shared_masks(self):
        rows = [Row(d * 65 + c, d, c, f"{DOMAINS[d]}/Class{c}/x.jpg", "train")
                for d in range(4) for c in range(65)]
        a, b = make_plan(rows, 42, epochs=2), make_plan(rows, 42, epochs=2)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        labels, domains = np.array([r.label for r in rows]), np.array([r.domain for r in rows])
        for e in range(2):
            counts = np.bincount(labels[a["receiver"][e]], minlength=65)
            self.assertLessEqual(counts.max() - counts.min(), 1)
            np.testing.assert_array_equal(labels[a["global"][e]], labels[a["matched"][e]])
            np.testing.assert_array_equal(labels[a["global"][e]], labels[a["cross"][e]])
            self.assertTrue((domains[a["matched"][e]] == domains[a["receiver"][e]]).all())
            self.assertTrue((domains[a["cross"][e]] != domains[a["receiver"][e]]).all())
        self.assertEqual(a["masks"].shape, (2, 3, 128))
        for mask in a["masks"].reshape(-1, 128):
            self.assertEqual(len(np.unique(mask)), 128)
            self.assertTrue(((mask >= 0) & (mask < 512)).all())

    def test_stage1_bn_running_fixed_affine_and_backbone_trainable(self):
        torch.manual_seed(12)
        model = TinyModel()
        stage1_mode(model)
        initial_bn = state_hash(bn_state(model))
        affine = model.bn1.weight.detach().clone()
        conv = model.conv1.weight.detach().clone()
        self.assertFalse(model.bn1.training)
        self.assertTrue(model.bn1.weight.requires_grad)
        opt = torch.optim.Adam(model.parameters(), lr=5e-5)
        loss = nn.functional.cross_entropy(model(torch.randn(4, 3, 8, 8)), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertIsNotNone(model.bn1.weight.grad)
        opt.step()
        self.assertEqual(initial_bn, state_hash(bn_state(model)))
        self.assertFalse(torch.equal(affine, model.bn1.weight))
        self.assertFalse(torch.equal(conv, model.conv1.weight))

    def test_stage2_only_classifier_updates(self):
        torch.manual_seed(21)
        model = TinyModel()
        freeze_stage2(model)
        initial = state_hash(model.state_dict())
        head = copy.deepcopy(model.fc)
        for p in head.parameters():
            p.requires_grad_(True)
        initial_head = state_hash(head.state_dict())
        optimizer = torch.optim.SGD(head.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        loss = nn.functional.cross_entropy(head(torch.randn(4, 4)), torch.tensor([4, 3, 2, 1]))
        loss.backward()
        optimizer.step()
        self.assertEqual(initial, state_hash(model.state_dict()))
        self.assertNotEqual(initial_head, state_hash(head.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_test_dataset_blocked_before_any_pixel_open(self):
        row = Row(0, 0, 0, "Art/Alarm_Clock/x.jpg", "test")
        with patch("officehome_data.Image.open") as image_open:
            with self.assertRaisesRegex(RuntimeError, "TEST LOCKED"):
                OfficeHomeImages(Path("."), [row], "test")
            image_open.assert_not_called()

    def test_test_lock_requires_all_12_and_detects_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for seed in SEEDS:
                for arm in ARMS:
                    p = root / f"{seed}_{arm}.pth"
                    p.write_bytes(b"CPU hash-lock unit fixture, not model weights")
                    entries.append({"seed": seed, "arm": arm, "epoch": 10,
                                    "path": p.name, "sha256": sha256(p)})
            lock = root / "lock.json"
            write_json(lock, {"csv_sha256": CSV_SHA256, "checkpoints": entries[:-1]})
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                validate_test_lock(lock)
            write_json(lock, {"csv_sha256": CSV_SHA256, "checkpoints": entries})
            self.assertEqual(len(validate_test_lock(lock)["checkpoints"]), 12)
            (root / entries[0]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                validate_test_lock(lock)

    def test_cell_group_metrics_and_paired_counts(self):
        rows = [Row(d * 65 + c, d, c, f"{DOMAINS[d]}/Class{c}/x.jpg", "val")
                for d in range(4) for c in range(65)]
        counts = np.tile(np.array([0, 20, 60] * 21 + [0, 60]), (4, 1))
        perfect = np.array([r.label for r in rows])
        imperfect = perfect.copy()
        imperfect[:3] = (imperfect[:3] + 1) % 65
        metrics, cells, domains = prediction_metrics(rows, perfect, counts)
        self.assertEqual(metrics["Overall"], 100)
        self.assertEqual(metrics["DomainClassMacro"], 100)
        self.assertEqual(len(cells), 260)
        self.assertEqual(len(domains), 4)
        pair = paired_changes(rows, perfect, imperfect, counts)
        self.assertEqual(pair[0]["gain"], 3)
        self.assertEqual(pair[0]["loss"], 0)
        self.assertAlmostEqual(pair[0]["delta_pp"], 300 / 260)
        self.assertEqual(sum(r["n"] for r in pair[1:]), 260)


if __name__ == "__main__":
    unittest.main(verbosity=2)
