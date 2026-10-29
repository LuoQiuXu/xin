"""CPU-only contract tests. Synthetic data are never scientific experiment output."""
import unittest
import numpy as np
from prepare_protocol import build
from compare_outcomes import compare

class Contracts(unittest.TestCase):
    def setUp(self):
        self.original=np.repeat(np.arange(100,dtype=np.int64),500)
        self.tr=np.concatenate([np.arange(500*c,500*c+int(500*(0.01**(c/99)))) for c in range(100)])
        self.va=np.array([49999,49998],dtype=np.int64)

    def test_valid_audited_subset(self):
        noisy=self.original.copy();noisy[0]=1
        arrays,audit=build(self.original,self.original,noisy,self.tr,self.va)
        self.assertEqual(audit["lt_train_images"],10847)
        self.assertEqual(audit["lt_noise_rate"] if "lt_noise_rate" in audit else audit["lt_human_annotation_noise_rate"],1/10847)
        self.assertTrue(np.array_equal(arrays["clean_train"]["ids"],arrays["noisy_train"]["ids"]))

    def test_image_order_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            build(self.original,np.roll(self.original,1),self.original,self.tr,self.va)

    def test_train_val_overlap_rejected(self):
        with self.assertRaises(ValueError):
            build(self.original,self.original,self.original,self.tr,self.tr[:2])

    def test_invalid_results_rejected(self):
        with self.assertRaises(ValueError):compare([])

    def test_paired_interaction(self):
        rows=[]
        for regime,acc in (("clean",50),("noisy",39)):
            for method,delta in (("no_fusion",0),("h2t",2 if regime=="clean" else -2)):
                for group in ("Overall","Many","Medium","Few"):
                    rows.append({"stage1_seed":42,"stage2_seed":124,"regime":regime,"method":method,
                                 "group":group,"accuracy_pct":acc+delta,"n":20,
                                 "stage1_sha":"stage1_"+regime,"val_ids_sha":"same","epoch":10})
        result=compare(rows)
        self.assertEqual(result[-1]["noise_interaction_pp"],-4)
        rows[-1]["stage1_sha"]="wrong"
        with self.assertRaises(ValueError):compare(rows)

if __name__=="__main__":unittest.main()
