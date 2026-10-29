"""CPU checks of author recipe boundaries and actual observed-label adapter."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from runtime import OFFICIAL, OUT, ObservedDataset, config, function, module
from evaluate import group_masks


class TrainingContracts(unittest.TestCase):
    def test_original_channel_replacement_is_batch_shared(self):
        author=module(OFFICIAL / "methods.py","test_original_h2t")
        receiver=torch.arange(2*64*8*8).reshape(2,64,8,8).float()
        donor=receiver+100000
        before=receiver.clone()
        # Only the CUDA boundary is mapped to CPU for this CPU-only contract test.
        with patch.object(torch.Tensor,"cuda",lambda x,*a,**kw:x):
            result=author.H2T(receiver,donor,rho=.25)
        changed=(result!=before).flatten(2).all(2)
        self.assertEqual(changed.sum(1).tolist(),[16,16])
        self.assertTrue(torch.equal(changed[0],changed[1]))
        self.assertTrue(torch.equal(result[changed],donor[changed]))
        self.assertTrue(torch.equal(result[~changed],before[~changed]))

    def test_original_learning_rate_boundaries(self):
        schedule=function(OFFICIAL / "train_stage1.py","adjust_learning_rate",{"math":math})
        optimizer=SimpleNamespace(param_groups=[{},{}]);cfg=config(1)
        for epoch,expected in [(0,.02),(4,.1),(159,.1),(160,.01),(179,.01),(180,.001),(199,.001)]:
            schedule(optimizer,epoch,cfg)
            self.assertTrue(all(abs(g["lr"]-expected)<1e-12 for g in optimizer.param_groups))
        schedule=function(OFFICIAL / "train_stage2.py","adjust_learning_rate",{"math":math})
        optimizer=SimpleNamespace(param_groups=[{}]);cfg=config(2)
        schedule(optimizer,0,cfg);self.assertAlmostEqual(optimizer.param_groups[0]["lr"],.02)
        schedule(optimizer,9,cfg)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],.01*(1+math.cos(.9*3.1415926535)))

    def test_actual_noisy_sampler_class_pools_use_observed_labels(self):
        ds=ObservedDataset("noisy",config(2))
        with np.load(OUT / "protocol/noisy_train.npz",allow_pickle=False) as z:observed=z["labels"]
        self.assertTrue(np.array_equal(ds.targets,observed))
        self.assertEqual(ds.cls_num_list,np.bincount(observed,minlength=100).tolist())
        for c,indices in ds.class_dict.items():
            self.assertGreater(len(indices),0)
            self.assertTrue((observed[indices]==c).all())
        self.assertEqual((min(ds.cls_num_list),max(ds.cls_num_list)),(14,445))

    def test_benchmark_group_boundaries(self):
        masks=group_masks(np.arange(4),np.array([101,100,20,19]))
        self.assertEqual({k:np.flatnonzero(v).tolist() for k,v in masks.items()},
                         dict(Overall=[0,1,2,3],Many=[0],Medium=[1,2],Few=[3]))


if __name__=="__main__":unittest.main()
