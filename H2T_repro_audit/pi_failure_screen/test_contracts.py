"""CPU-only tests for actual inference/analysis contracts; no checkpoints or CIFAR."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from inference import corrupt
from screen import load
from summarize import screen_record

TASK = Path(__file__).resolve().parent


class Contracts(unittest.TestCase):
    def setUp(self):
        temp_root = TASK / "results/primary_cache"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temp_root)
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_screen_requires_two_severities_and_multiple_images(self):
        rules = dict(families={"noise": ["low", "high"]}, minimum_effect_pp=.5,
                     minimum_excess_net_loss=3, minimum_distinct_negative_DiD_images=3)
        def row(did, net, negative):
            return dict(n=600, difference_in_differences_pp=did,
                        excess_net_correct=net, negative_DiD_images=negative)
        by = {("low", "few"): row(-1, -6, 8), ("high", "few"): row(-2, -12, 15),
              ("low", "all"): row(0, 0, 0), ("high", "all"): row(0, 0, 0)}
        self.assertTrue(screen_record("toy", "noise", "few", by, rules)["frequency_specific_candidate"])
        by["high", "few"] = row(.5, 3, 8)
        self.assertFalse(screen_record("toy", "noise", "few", by, rules)["within_checkpoint_candidate"])
        by["high", "few"] = row(-1, -2, 1)
        self.assertFalse(screen_record("toy", "noise", "few", by, rules)["within_checkpoint_candidate"])

    def npz(self, **changes):
        y = np.array([0, 1, 2], dtype=np.int64)
        d = dict(ids=np.arange(3, dtype=np.int64), labels=y, logits=np.eye(3, dtype=np.float32))
        d.update(changes)
        p = self.path / "prediction.npz"
        np.savez_compressed(p, **d)
        return p

    def test_noise_is_raw_id_specific_and_order_independent(self):
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        first = {i:corrupt(image,i,"noise_003") for i in [13,5,21]}
        second = {i:corrupt(image,i,"noise_003") for i in [21,13,5]}
        for i in first:
            np.testing.assert_array_equal(first[i],second[i])
        self.assertFalse(np.array_equal(first[13],first[5]))
        self.assertGreater(np.abs(corrupt(image,13,"noise_008").astype(float)-127).mean(),
                           np.abs(first[13].astype(float)-127).mean())

    def test_clean_and_corruption_leave_source_unchanged(self):
        image=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
        before=image.copy()
        for condition in ["clean","noise_003","noise_008","blur_05","blur_10"]:
            output=corrupt(image,8,condition)
            self.assertEqual(output.dtype,np.uint8)
            self.assertFalse(np.shares_memory(image,output))
            np.testing.assert_array_equal(image,before)

    def test_blur_is_three_by_three_and_channel_independent(self):
        image=np.zeros((32,32,3),dtype=np.uint8);image[16,16,0]=255
        for condition in ["blur_05","blur_10"]:
            output=corrupt(image,0,condition)
            self.assertEqual(np.count_nonzero(output[:,:,0]),9)
            self.assertFalse(output[:,:,1:].any())
            self.assertFalse(output[:15].any())
            self.assertFalse(output[18:].any())
            constant=np.full_like(image,83)
            np.testing.assert_array_equal(corrupt(constant,0,condition),constant)

    def test_corruption_rejects_nonpixel_input(self):
        with self.assertRaises(ValueError):corrupt(np.zeros((32,32,3)),1,"clean")
        with self.assertRaises(ValueError):corrupt(np.zeros((32,32,3),np.uint8),1,"unknown")

    def test_labels_must_be_integer(self):
        with self.assertRaises(ValueError):load(self.npz(labels=np.array([0,.5,2])),3)

    def test_ids_must_be_unique_and_typed(self):
        with self.assertRaises(ValueError):load(self.npz(ids=np.array([0,0,2])),3)
        with self.assertRaises(ValueError):load(self.npz(ids=np.array([0.,1.,2.])),3)

    def test_nonfinite_and_shape_checks(self):
        logits=np.eye(3);logits[0,0]=np.nan
        with self.assertRaises(ValueError):load(self.npz(logits=logits),3)
        with self.assertRaises(ValueError):load(self.npz(logits=np.zeros((3,4))),3)

    def test_sort_preserves_label_prediction_alignment(self):
        ids,y,pred=load(self.npz(ids=np.array([2,0,1])),3)
        np.testing.assert_array_equal(ids,[0,1,2])
        np.testing.assert_array_equal(y,[1,2,0])
        np.testing.assert_array_equal(pred,y)

    def test_difference_in_differences_and_no_overwrite(self):
        y=np.array([0,0,1,1,2,2],dtype=np.int64);ids=np.arange(6)
        flags={"ref":{"clean":[1,1,1,1,1,0],"noise_003":[1,1,1,1,0,0]},
               "cand":{"clean":[1,1,1,1,1,1],"noise_003":[1,1,1,0,0,0]}}
        methods={}
        for method,conditions in flags.items():
            methods[method]={}
            for cond,correct in conditions.items():
                pred=np.where(correct,y,(y+1)%3)
                logits=np.zeros((6,3));logits[np.arange(6),pred]=3
                p=self.path/f"{method}_{cond}.npz"
                np.savez_compressed(p,ids=ids,labels=y,logits=logits)
                methods[method][cond]=p.name
        manifest=self.path/"manifest.json"
        manifest.write_text(json.dumps(dict(class_counts=[500,40,5],reference="ref",candidate="cand",
            conditions=["clean","noise_003"],methods=methods)),encoding="utf-8")
        output=self.path/"analysis"
        cmd=[sys.executable,"-B",str(TASK/"screen.py"),"--manifest",str(manifest),"--output",str(output)]
        r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
        with (output/"difference_in_differences.csv").open(encoding="utf-8",newline="") as f:
            rows=list(csv.DictReader(f))
        row=next(r for r in rows if r["condition"]=="noise_003" and r["group"]=="all")
        self.assertAlmostEqual(float(row["difference_in_differences_pp"]),-100/3)
        self.assertEqual(int(row["excess_net_correct"]),-2)
        self.assertEqual(int(row["negative_DiD_images"]),2)
        before=(output/"metrics.csv").read_bytes()
        repeat=subprocess.run(cmd,capture_output=True,text=True)
        self.assertNotEqual(repeat.returncode,0)
        self.assertEqual((output/"metrics.csv").read_bytes(),before)


if __name__ == "__main__":
    unittest.main()
