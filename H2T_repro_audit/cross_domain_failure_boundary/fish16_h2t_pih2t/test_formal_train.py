import argparse
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import formal_train as f


def checkpoint_metadata(condition="A0", completed=90):
    return {"formal_run":True,"smoke_only":False,"condition":condition,
            "stage":f.stage_of(condition),"seed":42,"epochs_total":f.epoch_budget(condition),
            "epoch_completed":completed,"protocol_sha256":"protocol",
            "records":[{"epoch":i} for i in range(1,completed+1)],
            "scheduler":{"stage":f.stage_of(condition),"next_epoch":completed,"version":1},
            "model":{},"optimizer":{},"scaler":{},"rng":{}}


class FormalTests(unittest.TestCase):
    def test_stage1_fixed_warmup_and_cosine(self):
        values=[f.epoch_lr(1,i) for i in range(90)]
        for actual,expected in zip(values[:5],[.002,.004,.006,.008,.01]):
            self.assertAlmostEqual(actual,expected,places=15)
        self.assertLess(values[5],values[4])
        self.assertTrue(all(a>=b for a,b in zip(values[4:],values[5:])))
        self.assertEqual(values[-1],0.)
        with self.assertRaises(ValueError):f.epoch_lr(1,90)

    def test_stage2_fixed_cosine_first_epoch_and_budget(self):
        values=[f.epoch_lr(2,i) for i in range(10)]
        self.assertEqual(values[0],.01)
        self.assertAlmostEqual(values[-1],.005*(1+math.cos(9*math.pi/10)),places=15)
        self.assertGreater(values[-1],0.)
        with self.assertRaises(ValueError):f.epoch_lr(2,10)

    def test_scheduler_resume_cannot_repeat_or_skip(self):
        parameter=torch.nn.Parameter(torch.zeros(1))
        optimizer=torch.optim.SGD([parameter],lr=.01)
        schedule=f.EpochSchedule(1)
        for epoch in range(3):
            schedule.apply(optimizer,epoch);schedule.complete(epoch)
        restored=f.EpochSchedule(1);restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.apply(optimizer,3),f.epoch_lr(1,3))
        with self.assertRaises(ValueError):restored.apply(optimizer,2)
        with self.assertRaises(ValueError):restored.apply(optimizer,4)
        with self.assertRaises(ValueError):f.EpochSchedule(2).load_state_dict(schedule.state_dict())

    def test_formal_amp_policy_prevents_growth_during_one_condition(self):
        policy = f.FORMAL_AMP_POLICY
        self.assertEqual(policy["init_scale"], 1.0)
        self.assertEqual(policy["growth_interval"], 100000)
        self.assertGreater(policy["growth_interval"], policy["stage1_max_attempts"])
        self.assertGreater(policy["growth_interval"], policy["stage2_max_attempts"])
        self.assertFalse(policy["enabled"])
        self.assertEqual(policy["precision"], f.NUMERIC_MODE)
        self.assertFalse(policy["automatic_growth_during_condition"])
        # GradScaler construction is lazy; this checks its configuration without
        # executing a CUDA kernel or a GPU smoke run.
        with patch("torch.cuda.is_available", return_value=True):
            scaler = f.make_formal_grad_scaler()
        self.assertFalse(scaler.is_enabled())
        self.assertEqual(scaler.get_scale(), 1.0)

    def test_json_normalize_handles_tuple_list_resume_roundtrip(self):
        record = {
            "overflow_events": [{
                "nonfinite_parameters": ("extractor.pif.fuse.weight",),
                "nonfinite_positions": {"extractor.pif.fuse.weight": ((0, 1, 0),)},
            }]
        }
        restored = f.json_normalize(record)
        self.assertEqual(
            restored,
            {"overflow_events": [{
                "nonfinite_parameters": ["extractor.pif.fuse.weight"],
                "nonfinite_positions": {"extractor.pif.fuse.weight": [[0, 1, 0]]},
            }]},
        )
        self.assertEqual(restored, json.loads(json.dumps(record)))

    def test_routing_uses_one_final_source_per_pair(self):
        self.assertEqual([f.stage2_source(c) for c in ["A1","A2","B1","B2"]],["A0","A0","B0","B0"])
        self.assertEqual([f.ROUTES[c][1] for c in ["A1","A2","B1","B2"]],["crt","h2t","crt","pi"])
        with self.assertRaises(ValueError):f.stage2_source("A0")

    def test_final_source_rejects_smoke_wrong_seed_and_early_epoch(self):
        valid=checkpoint_metadata()
        f.validate_checkpoint_metadata(valid,"A0","protocol",final_source=True)
        for change in [{"smoke_only":True},{"formal_run":False},{"seed":43},{"condition":"B0"},{"protocol_sha256":"different"}]:
            invalid={**valid,**change}
            with self.assertRaises(ValueError):f.validate_checkpoint_metadata(invalid,"A0","protocol",True)
        with self.assertRaises(ValueError):f.validate_checkpoint_metadata(checkpoint_metadata(completed=89),"A0","protocol",True)

    def test_resume_rejects_missing_or_repeated_epoch_history(self):
        cp=checkpoint_metadata(completed=2)
        f.validate_checkpoint_metadata(cp,"A0","protocol")
        cp["records"]=[{"epoch":1},{"epoch":1}]
        with self.assertRaises(ValueError):f.validate_checkpoint_metadata(cp,"A0","protocol")
        cp=checkpoint_metadata(completed=2);cp["scheduler"]["next_epoch"]=1
        with self.assertRaises(ValueError):f.validate_checkpoint_metadata(cp,"A0","protocol")
        del cp["rng"]
        with self.assertRaises(ValueError):f.validate_checkpoint_metadata(cp,"A0","protocol")

    def test_source_file_hash_checked_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"source.pth"
            f.atomic_checkpoint(path,{"formal_run":True,"seed":42})
            expected=f.models.sha(path)
            self.assertEqual(f.checked_checkpoint(path,expected)["seed"],42)
            with self.assertRaises(ValueError):f.checked_checkpoint(path,"0"*64)

    def test_paired_hash_mismatch_is_fatal(self):
        f.verify_pair_sources({"A1":"same","A2":"same","B1":"other","B2":"other"})
        with self.assertRaises(ValueError):f.verify_pair_sources({"A1":"one","A2":"two"})
        with self.assertRaises(ValueError):f.verify_pair_sources({"B1":"one","B2":"two"})

    def test_test_split_is_rejected_before_any_image_open(self):
        rows=[{"split_primary":"train"},{"split_primary":"val"},{"split_primary":"test"}]
        f.require_split(rows,[0],"train");f.require_split(rows,[1],"val")
        for indices,split in [([2],"train"),([2],"val"),([2],"test"),([1],"train")]:
            with self.assertRaises(ValueError):f.require_split(rows,indices,split)
        with patch("PIL.Image.open") as opened:
            with self.assertRaises(ValueError):f.SafeViews(rows,[2],True)
            with self.assertRaises(ValueError):f.SafeViews(rows,[1],False,donors=[0])
            opened.assert_not_called()

    def test_rng_checkpoint_roundtrip_is_weights_only_safe(self):
        before=f.capture_rng(cuda=False)
        try:
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            state=f.capture_rng(cuda=False)
            expected=(random.random(),np.random.rand(),torch.rand(4))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"rng.pth";f.atomic_checkpoint(path,{"rng":state})
                saved=torch.load(path,map_location="cpu",weights_only=True)["rng"]
            f.restore_rng(saved)
            self.assertEqual(random.random(),expected[0])
            self.assertEqual(np.random.rand(),expected[1])
            self.assertTrue(torch.equal(torch.rand(4),expected[2]))
        finally:f.restore_rng(before)

    def test_initial_resume_checkpoint_is_valid_but_not_stage2_source(self):
        cp=checkpoint_metadata(completed=0)
        f.validate_checkpoint_metadata(cp,"A0","protocol")
        with self.assertRaises(ValueError):f.validate_checkpoint_metadata(cp,"A0","protocol",True)

    def test_new_run_refuses_existing_output_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"sentinel.txt").write_text("preserve",encoding="utf-8")
            args=argparse.Namespace(output_dir=directory,resume=False,condition="all",source_checkpoint=None,source_sha256=None,cpu_validation_record=None)
            with self.assertRaises(FileExistsError):f.FormalRun(args).execute()
            self.assertEqual([p.name for p in root.iterdir()],["sentinel.txt"])
            self.assertEqual((root/"sentinel.txt").read_text(encoding="utf-8"),"preserve")


if __name__=="__main__":unittest.main(verbosity=2)
