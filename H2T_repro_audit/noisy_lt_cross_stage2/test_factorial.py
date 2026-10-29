"""CPU-only synthetic test of factorial math/provenance; no GPU or old weights."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_factorial import assess, GROUPS

class FactorialContract(unittest.TestCase):
    def fixture(self):
        diagonal=[]; crossed=[]
        for s in (124,125):
            for rep,lab in (("clean","clean"),("clean","noisy"),("noisy","clean"),("noisy","noisy")):
                for group,n in (("Overall",1980),("Many",680),("Medium",700),("Few",600)):
                    for method in ("no_fusion","h2t"):
                        # D(clean,clean)=4; D(clean,noisy)=1;
                        # D(noisy,clean)=3; D(noisy,noisy)=0.5
                        d={("clean","clean"):4,("clean","noisy"):1,
                           ("noisy","clean"):3,("noisy","noisy"):0.5}[rep,lab]
                        accuracy=30+(d if method=="h2t" else 0)
                        if rep==lab:
                            diagonal.append(dict(stage2_seed=str(s),regime=rep,method=method,group=group,
                                accuracy_pct=str(accuracy),n=str(n),epoch="10",val_ids_sha="same",
                                stage1_sha="stage_"+rep))
                        else:
                            crossed.append(dict(stage2_seed=str(s),representation_regime=rep,
                                label_regime=lab,method=method,group=group,n=str(n),
                                accuracy_pct=str(accuracy),source_checkpoint_sha="stage_"+rep))
        return diagonal,crossed

    def test_effects(self):
        a,b=self.fixture()
        _,effects,summary=assess(a,b)
        few=effects["Few"]
        self.assertEqual(len(few),2)
        self.assertAlmostEqual(few[0]["representation_effect_pp"], -0.75)
        self.assertAlmostEqual(few[0]["observed_stage2_label_sampling_effect_pp"],-2.75)
        self.assertAlmostEqual(few[0]["total_diagonal_change_pp"],-3.5)
        self.assertAlmostEqual(few[0]["factorial_interaction_pp"], 0.5)
        self.assertEqual(summary["groups"]["Few"]["mean_total_diagonal_change_pp"],-3.5)

    def test_missing_arm_rejected(self):
        a,b=self.fixture()
        with self.assertRaises(ValueError):assess(a,b[1:])

    def test_wrong_source_rejected(self):
        a,b=self.fixture()
        b[0]["source_checkpoint_sha"]="not_same_parent"
        with self.assertRaises(ValueError):assess(a,b)

    def test_eval_denominator_rejected(self):
        a,b=self.fixture()
        b[0]["n"]="599" if b[0]["group"]=="Few" else "31"
        with self.assertRaises(ValueError):assess(a,b)

if __name__=="__main__":
    unittest.main()
