import unittest

import torch
from torch import nn

from amp_recovery import (
    GradientContractError,
    OptimizerStepCounter,
    guarded_scaler_step,
    parameter_sha,
)


class ToyPIFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = nn.Module()
        self.extractor.pif = nn.Module()
        self.extractor.pif.fuse = nn.Conv1d(2, 1, 1, bias=False)
        self.other = nn.Parameter(torch.tensor([1.0]))


class FakeScaler:
    """CPU-only GradScaler contract double; GPU integration is tested by Gate 1."""

    def __init__(self, scale=65536.0, enabled=True):
        self.scale = float(scale)
        self.enabled = enabled
        self.step_calls = 0
        self.update_calls = 0
        self._found_inf = False

    def is_enabled(self):
        return self.enabled

    def get_scale(self):
        return self.scale

    def step(self, optimizer):
        self.step_calls += 1
        self._found_inf = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        if not self._found_inf:
            optimizer.step()

    def update(self):
        self.update_calls += 1
        if self._found_inf:
            self.scale *= 0.5
        self._found_inf = False


def make_case():
    model = ToyPIFModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    counter = OptimizerStepCounter(optimizer)
    named = list(model.named_parameters())
    for _, parameter in named:
        parameter.grad = torch.ones_like(parameter)
    return model, optimizer, counter, named


class AmpRecoveryTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "counter"):
            self.counter.close()

    def test_exact_pif_b_inf_is_skipped_and_scale_is_reduced(self):
        model, optimizer, self.counter, named = make_case()
        model_before = parameter_sha(named)
        model.extractor.pif.fuse.weight.grad[0, 1, 0] = float("inf")
        scaler = FakeScaler()

        result = guarded_scaler_step(
            optimizer=optimizer,
            scaler=scaler,
            named_parameters=named,
            step_counter=self.counter,
            allow_pif_b_overflow=True,
        )

        self.assertTrue(result.skipped)
        self.assertEqual(result.kind, "recoverable_pif_b_overflow_skip")
        self.assertEqual(result.inf_count, 1)
        self.assertEqual(result.nan_count, 0)
        self.assertEqual(result.scale_before, 65536.0)
        self.assertEqual(result.scale_after, 32768.0)
        self.assertEqual(self.counter.count, 0)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)
        self.assertEqual(parameter_sha(named), model_before)

    def test_finite_gradients_execute_exactly_one_optimizer_step(self):
        model, optimizer, self.counter, named = make_case()
        before = parameter_sha(named)
        scaler = FakeScaler()

        result = guarded_scaler_step(
            optimizer=optimizer,
            scaler=scaler,
            named_parameters=named,
            step_counter=self.counter,
        )

        self.assertFalse(result.skipped)
        self.assertEqual(result.kind, "optimizer_step")
        self.assertEqual(self.counter.count, 1)
        self.assertEqual(scaler.step_calls, 1)
        self.assertEqual(scaler.update_calls, 1)
        self.assertNotEqual(parameter_sha(named), before)

    def test_exact_pif_inf_is_fatal_without_explicit_b0_permission(self):
        model, optimizer, self.counter, named = make_case()
        model.extractor.pif.fuse.weight.grad[0, 1, 0] = float("inf")
        scaler = FakeScaler()

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=False,
            )
        self.assertEqual(scaler.step_calls, 0)
        self.assertEqual(self.counter.count, 0)

    def test_nan_is_never_recoverable(self):
        model, optimizer, self.counter, named = make_case()
        model.extractor.pif.fuse.weight.grad[0, 1, 0] = float("nan")
        scaler = FakeScaler()

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=True,
            )
        self.assertEqual(scaler.step_calls, 0)
        self.assertEqual(self.counter.count, 0)

    def test_inf_in_any_other_parameter_is_fatal(self):
        model, optimizer, self.counter, named = make_case()
        model.other.grad[0] = float("inf")
        scaler = FakeScaler()

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=True,
            )
        self.assertEqual(scaler.step_calls, 0)

    def test_inf_at_wrong_pif_coordinate_is_fatal(self):
        model, optimizer, self.counter, named = make_case()
        model.extractor.pif.fuse.weight.grad[0, 0, 0] = float("inf")
        scaler = FakeScaler()

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=True,
            )
        self.assertEqual(scaler.step_calls, 0)

    def test_multiple_inf_values_are_fatal(self):
        model, optimizer, self.counter, named = make_case()
        model.extractor.pif.fuse.weight.grad[0, 0, 0] = float("inf")
        model.extractor.pif.fuse.weight.grad[0, 1, 0] = float("inf")
        scaler = FakeScaler()

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=True,
            )
        self.assertEqual(scaler.step_calls, 0)

    def test_missing_gradient_is_fatal(self):
        model, optimizer, self.counter, named = make_case()
        model.other.grad = None
        scaler = FakeScaler()

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=True,
            )
        self.assertEqual(scaler.step_calls, 0)

    def test_overflow_requires_enabled_scaler(self):
        model, optimizer, self.counter, named = make_case()
        model.extractor.pif.fuse.weight.grad[0, 1, 0] = float("inf")
        scaler = FakeScaler(enabled=False)

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=True,
            )
        self.assertEqual(scaler.step_calls, 0)

    def test_consecutive_skip_limit_is_fail_closed(self):
        model, optimizer, self.counter, named = make_case()
        model.extractor.pif.fuse.weight.grad[0, 1, 0] = float("inf")
        scaler = FakeScaler()

        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=scaler,
                named_parameters=named,
                step_counter=self.counter,
                allow_pif_b_overflow=True,
                consecutive_overflow_skips=2,
                max_consecutive_overflow_skips=2,
            )
        self.assertEqual(scaler.step_calls, 0)

    def test_optimizer_parameter_set_must_match_audit(self):
        model, optimizer, self.counter, named = make_case()
        with self.assertRaises(GradientContractError):
            guarded_scaler_step(
                optimizer=optimizer,
                scaler=FakeScaler(),
                named_parameters=named[:-1],
                step_counter=self.counter,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
