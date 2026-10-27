"""CPU-only protocol regressions; no CIFAR data or CUDA calls."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import run_reproduction as runner


class ProtocolTests(unittest.TestCase):
    def test_stage1_lr_exact_original_boundaries(self):
        path=runner.ORIGINAL/"train_stage1.py"
        tree=ast.parse(path.read_text(encoding="utf-8"))
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="adjust_learning_rate")
        # Only the audited numeric schedule function executes, with a fake
        # optimizer and literal config; the original training module is not run.
        namespace={}
        exec(compile(ast.Module(body=[func],type_ignores=[]),str(path),"exec"),namespace)
        config=SimpleNamespace(lr=.1,lr_scheduler=SimpleNamespace(type="multistep",lr_step=[160,180]))
        for ep in [0,1,2,3,4,5,158,159,160,178,179,180,199]:
            optimizer=SimpleNamespace(param_groups=[{},{}])
            namespace["adjust_learning_rate"](optimizer,ep,config)
            self.assertAlmostEqual(runner.stage1_lr(ep),optimizer.param_groups[0]["lr"],places=15)
        self.assertAlmostEqual(runner.stage1_lr(159),.1)
        self.assertAlmostEqual(runner.stage1_lr(160),.01)
        self.assertAlmostEqual(runner.stage1_lr(179),.01)
        self.assertAlmostEqual(runner.stage1_lr(180),.001)

    def test_stage2_lr_horizon_is_locked(self):
        self.assertAlmostEqual(.2*runner.cosine_lr(.1,0,10),.02)
        self.assertGreater(.2*runner.cosine_lr(.1,9,10),0)
        self.assertEqual(runner.BATCH,128)
        self.assertEqual((runner.STAGE1_EPOCHS,runner.STAGE2_EPOCHS),(200,10))

    def test_checkpoint_sha_checked_before_deserialization(self):
        runner.CHECKS.mkdir(parents=True,exist_ok=True)
        self.assertTrue(runner.CHECKS.resolve().is_relative_to(runner.HERE.resolve()))
        with TemporaryDirectory(dir=runner.CHECKS) as folder:
            run=Path(folder)
            checkpoint=run/"stage1_final_epoch200.pth"
            checkpoint.write_bytes(b"cpu-test-placeholder-not-a-torch-checkpoint")
            expected=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (run/"stage1_summary.json").write_text(json.dumps({"checkpoint_file_sha256":expected}),encoding="utf-8")
            self.assertEqual(runner.verify_checkpoint_digest(run),expected)
            checkpoint.write_bytes(b"tampered-placeholder")
            with self.assertRaisesRegex(RuntimeError,"SHA256"):
                runner.verify_checkpoint_digest(run)

    def test_checkpoint_requires_completed_summary(self):
        runner.CHECKS.mkdir(parents=True,exist_ok=True)
        self.assertTrue(runner.CHECKS.resolve().is_relative_to(runner.HERE.resolve()))
        with TemporaryDirectory(dir=runner.CHECKS) as folder:
            with self.assertRaises(FileNotFoundError):runner.verify_checkpoint_digest(Path(folder))


if __name__=="__main__":unittest.main()
