"""CPU-only sanity tests for the pre-registered receiver ablation."""
import unittest
import numpy as np
from policy import ARMS, choose_active


class ReceiverPolicyTests(unittest.TestCase):
    def test_original_and_dr(self):
        labels=np.array([0,35,36,70,71,95,99], dtype=np.int64)
        self.assertEqual(ARMS,("DR-NoFusion","Original-BatchShared","TailOff","RandomOffMatched"))
        self.assertEqual(choose_active(labels,"DR-NoFusion",124,0,0).sum(),0)
        self.assertEqual(choose_active(labels,"Original-BatchShared",124,0,0).sum(),len(labels))

    def test_tail_only_excluded(self):
        labels=np.array([0,35,36,70,71,95,99],dtype=np.int64)
        self.assertEqual(choose_active(labels,"TailOff",124,0,0).tolist(),
                         [True,True,True,True,False,False,False])

    def test_exact_dose_and_determinism(self):
        labels=np.r_[np.arange(100),np.repeat(99,12)]
        for seed in (124,125):
            for epoch in (0,1,9):
                for batch in (0,1,74):
                    tailoff=choose_active(labels,"TailOff",seed,epoch,batch)
                    randomoff=choose_active(labels,"RandomOffMatched",seed,epoch,batch)
                    again=choose_active(labels,"RandomOffMatched",seed,epoch,batch)
                    self.assertEqual(int(tailoff.sum()),int(randomoff.sum()))
                    self.assertTrue(np.array_equal(randomoff,again))
                    self.assertEqual(randomoff.dtype,np.bool_)

    def test_does_not_mutate_global_numpy_rng(self):
        labels=np.r_[np.arange(100),np.arange(100)]
        np.random.seed(25); reference=np.random.random(5)
        np.random.seed(25)
        choose_active(labels,"RandomOffMatched",124,4,19)
        self.assertTrue(np.array_equal(np.random.random(5), reference))

    def test_edge_cases(self):
        labels=np.repeat(99,16)
        self.assertEqual(choose_active(labels,"TailOff",124,0,0).sum(),0)
        self.assertEqual(choose_active(labels,"RandomOffMatched",124,0,0).sum(),0)
        labels=np.repeat(2,16)
        self.assertEqual(choose_active(labels,"TailOff",124,0,0).sum(),16)
        self.assertEqual(choose_active(labels,"RandomOffMatched",124,0,0).sum(),16)
        with self.assertRaises(ValueError):
            choose_active(np.array([-1]),"TailOff",124,0,0)
        with self.assertRaises(ValueError):
            choose_active(np.array([0,101]),"TailOff",124,0,0)
        with self.assertRaises(ValueError):
            choose_active(np.array([0]),"Undefined",124,0,0)

if __name__=="__main__":
    unittest.main()
