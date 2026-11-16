"""Regression checks for the sampler, metrics and sealed evaluation entry point."""
from pathlib import Path
from types import SimpleNamespace
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cuda_statistics import initialize_cuda
from evaluate_final import classification_metrics, evaluate
from experiment_common import PROTOCOL, REPO, read_json, record_run
from prepare_experiment import official_plan
from train_stage1 import MAX_SMOKE_AMP_RETRIES, epoch_lr, smoke_amp_update


class PipelineTests(unittest.TestCase):
    def test_sampler_preserves_rng_and_continues_between_epochs(self):
        sampler = read_json(PROTOCOL)["official_sampler"]
        source = REPO / sampler["path"]
        labels = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
        initial = random.getstate()
        plan = official_plan(source, sampler["sha256"], labels, 123, 3)
        self.assertEqual(random.getstate(), initial)
        self.assertEqual(plan.shape, (3, len(labels)))
        self.assertEqual(plan.dtype, np.int32)
        self.assertTrue(np.array_equal(plan, official_plan(source, sampler["sha256"], labels, 123, 3)))
        self.assertFalse(np.array_equal(plan[0], plan[1]))
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            official_plan(source, "0" * 64, labels, 123, 3)

    def test_warmup_and_cosine_endpoints(self):
        self.assertAlmostEqual(epoch_lr(1, 90, 0.01), 0.002)
        self.assertAlmostEqual(epoch_lr(5, 90, 0.01), 0.01)
        self.assertLess(epoch_lr(6, 90, 0.01), 0.01)
        self.assertEqual(epoch_lr(90, 90, 0.01), 0.0)

    def test_metrics_use_official_train_bins_and_macro_accuracy(self):
        counts = np.array([101, 100, 20, 19], dtype=np.int64)
        y = np.array([0, 0, 1, 1, 1, 2, 3, 3], dtype=np.int64)
        prediction = np.array([0, 1, 1, 1, 1, 3, 3, 3], dtype=np.int64)
        result, support, correct, _ = classification_metrics(prediction, y, counts)
        self.assertEqual(result, {"Overall": 75.0, "Many": 50.0, "Medium": 50.0, "Few": 100.0})
        self.assertEqual(support.tolist(), [2, 3, 1, 2])
        self.assertEqual(correct.tolist(), [1, 3, 0, 2])

    def test_final_evaluation_refuses_before_any_test_loader_or_output(self):
        exp = SimpleNamespace(
            protocol={"stage2": {"epochs": 30}}, arms=("D0", "D1", "D2"),
            jobs=lambda: iter([("imagenet_lt", 123)]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "final"
            run = Path(tmp) / "missing_stage2"
            with (patch("evaluate_final.FINAL", final),
                  patch("evaluate_final.run_path", return_value=run),
                  patch("evaluate_final.official_test_dataset") as loader):
                with self.assertRaisesRegex(RuntimeError, "Official TEST remains sealed"):
                    evaluate(exp, {}, torch.device("cpu"))
            loader.assert_not_called()
            self.assertFalse(final.exists())

    def test_gpu_run_initializes_cuda_before_resetting_peak_stats(self):
        steps = []
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "init", side_effect=lambda: steps.append("initialize")),
                patch.object(torch.cuda, "device_count", return_value=2),
                patch.object(torch.cuda, "current_device", side_effect=[0, 1]),
                patch.object(torch.cuda, "set_device", side_effect=lambda i: steps.append(("select", i))),
                patch.object(torch.cuda, "get_device_name", return_value="test GPU"),
                patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=1024)),
                patch.object(torch.cuda, "synchronize", side_effect=lambda i: steps.append(("sync", i))),
                patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=lambda i: steps.append(("reset", i))),
                patch.object(torch.cuda, "max_memory_allocated", return_value=128) as allocated,
                patch.object(torch.cuda, "max_memory_reserved", return_value=256) as reserved,
            ):
                with record_run(directory, {"phase": "regression"}, device=torch.device("cuda", 1)):
                    steps.append("run")
            self.assertEqual(steps, ["initialize", ("select", 1), ("sync", 1), ("reset", 1), "run"])
            allocated.assert_called_once_with(1)
            reserved.assert_called_once_with(1)
            status = read_json(directory / "status.json")
            self.assertEqual(status["peak_cuda_allocated_bytes"], 128)
            self.assertEqual(status["cuda_diagnostics"]["device_index"], 1)

    def test_gpu_startup_failure_is_recorded_before_entering_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=1),
                patch.object(torch.cuda, "init", side_effect=RuntimeError("device initialization failed")),
                patch.object(torch.cuda, "reset_peak_memory_stats") as reset,
            ):
                with self.assertRaisesRegex(RuntimeError, "device initialization failed"):
                    with record_run(directory, {"phase": "regression"}, device=torch.device("cuda", 0)):
                        self.fail("A failed initialization must stop the run")
            reset.assert_not_called()
            status = read_json(directory / "status.json")
            self.assertEqual(status["state"], "failed")
            self.assertIn("device initialization failed", status["error"])
            self.assertNotIn("peak_cuda_allocated_bytes", status)
            self.assertIn("initialize CUDA diagnostics", (directory / "run.log").read_text(encoding="utf-8"))

    def test_cuda_without_index_resolves_the_actual_current_device(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "init"),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "current_device", return_value=1),
            patch.object(torch.cuda, "set_device") as selected,
            patch.object(torch.cuda, "get_device_name", return_value="test GPU"),
            patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=1024)),
        ):
            stats = initialize_cuda(torch.device("cuda"))
        self.assertEqual(stats.device, torch.device("cuda", 1))
        self.assertIsNone(stats.diagnostics["requested_device_index"])
        selected.assert_called_once_with(1)

    def test_cuda_rejects_out_of_range_index_before_selection(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "init"),
            patch.object(torch.cuda, "device_count", return_value=1),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "set_device") as selected,
        ):
            with self.assertRaisesRegex(ValueError, "outside") as error:
                initialize_cuda(torch.device("cuda", 2))
        selected.assert_not_called()
        self.assertIn('"device_count": 1', error.exception.__notes__[0])
        self.assertIn('"device_index": 2', error.exception.__notes__[0])


class Stage1AmpSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        self.classifier = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.SGD([*self.encoder.parameters(), *self.classifier.parameters()], lr=0.01)
        self.scaler = torch.amp.GradScaler("cpu")
        self.inputs = torch.randn(4, 4)
        self.labels = torch.tensor([0, 1, 0, 1])
        self.attempts = []

    def forward(self):
        features = self.encoder(self.inputs)
        logits = self.classifier(features)
        return torch.nn.functional.cross_entropy(logits, self.labels), features, logits

    def run_batch(self, forward=None):
        return smoke_amp_update(
            self.encoder, self.classifier, self.optimizer, self.scaler, forward or self.forward,
            lambda summary, *_: self.attempts.append(summary),
        )

    def test_stage1_amp_retry_requires_real_skip_and_finite_update(self):
        calls = 0

        def overflow_once(grad):
            nonlocal calls
            calls += 1
            return torch.full_like(grad, float("inf")) if calls == 1 else grad

        handle = self.encoder[0].weight.register_hook(overflow_once)
        try:
            _, _, _, final = self.run_batch()
        finally:
            handle.remove()
        first, second = self.attempts
        self.assertEqual(first["outcome"], "gradscaler_skipped_nonfinite")
        self.assertEqual(first["unscaled_gradient_state"]["nonfinite_names"], ["encoder.0.weight"])
        self.assertFalse(first["optimizer_step"])
        self.assertEqual(first["update"]["changed_parameter_tensors"], 0)
        self.assertLess(first["scale_after"], first["scale_before"])
        self.assertEqual(second["scale_before"], first["scale_after"])
        self.assertEqual(final["outcome"], "finite_optimizer_update")
        self.assertTrue(final["optimizer_step"])
        self.assertGreater(final["update"]["changed_parameter_tensors"], 0)
        self.assertEqual(self.encoder[1].num_batches_tracked.item(), 1)

    def test_stage1_missing_gradient_is_terminal_without_backoff(self):
        def disconnected():
            features = self.encoder(self.inputs)
            logits = self.classifier(features)
            return features.square().mean(), features, logits

        with self.assertRaisesRegex(RuntimeError, "Missing Stage-I gradient"):
            self.run_batch(disconnected)
        self.assertEqual(len(self.attempts), 1)
        first = self.attempts[0]
        self.assertEqual(first["unscaled_gradient_state"]["missing_names"],
                         ["classifier.weight", "classifier.bias"])
        self.assertEqual(first["unscaled_gradient_state"]["nonfinite_names"], [])
        self.assertFalse(first["gradscaler_update_called"])
        self.assertFalse(first["optimizer_step"])
        self.assertEqual(first["scale_after"], first["scale_before"])

    def test_stage1_persistent_inf_stops_after_eight_retries(self):
        handle = self.encoder[0].weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with self.assertRaisesRegex(RuntimeError, "exhausted 8 GradScaler retries"):
                self.run_batch()
        finally:
            handle.remove()
        self.assertEqual(len(self.attempts), MAX_SMOKE_AMP_RETRIES + 1)
        self.assertTrue(all(a["gradscaler_update_called"] and not a["optimizer_step"] for a in self.attempts))
        self.assertTrue(all(a["update"]["changed_parameter_tensors"] == 0 for a in self.attempts))


if __name__ == "__main__":
    unittest.main()
