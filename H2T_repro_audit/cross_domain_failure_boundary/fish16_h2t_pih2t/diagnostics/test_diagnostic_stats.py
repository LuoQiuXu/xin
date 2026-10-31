import unittest
import torch
from torch import nn
from diagnose_b0 import tensor_stats, gradient_records, gradient_groups, cpu_tree


class DiagnosticStatsTests(unittest.TestCase):
    def test_finite_statistics_and_l2_use_float64(self):
        s = tensor_stats(torch.tensor([3., -4.]))
        self.assertTrue(s["finite"])
        self.assertEqual(s["max_abs"], 4.)
        self.assertEqual(s["l2_norm"], 5.)
        self.assertEqual(s["nan_count"], 0)
        self.assertGreater(tensor_stats(torch.tensor([1e30]))["l2_norm"], 1e29)

    def test_missing_is_distinct_from_finite_zero(self):
        missing, zero = tensor_stats(None), tensor_stats(torch.zeros(2))
        self.assertTrue(missing["is_none"])
        self.assertIsNone(missing["finite"])
        self.assertFalse(zero["is_none"])
        self.assertTrue(zero["finite"])
        self.assertEqual(zero["l2_norm"], 0.)

    def test_nan_inf_not_hidden_by_finite_substatistics(self):
        s = tensor_stats(torch.tensor([3., 4., float('nan'), float('inf')]))
        self.assertFalse(s["finite"])
        self.assertEqual(s["nan_count"], 1)
        self.assertEqual(s["inf_count"], 1)
        self.assertEqual(s["l2_norm"], "NaN")
        self.assertEqual(s["finite_l2_norm"], 5.)
        self.assertEqual(tensor_stats(torch.tensor([float('inf')]))["l2_norm"], "Inf")

    def test_parameter_group_report_distinguishes_missing_and_inf(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(2, 1)
                self.extractor = nn.Module()
                self.extractor.pif = nn.Linear(2, 1, bias=False)
        model = Dummy()
        model.extractor.pif.weight.grad = torch.tensor([[float('inf'), 1.]])
        model.classifier.weight.grad = torch.ones_like(model.classifier.weight)
        groups = gradient_groups(gradient_records(model, "unit", 0, "after_unscale"))
        self.assertEqual(groups["PIF"]["inf_count"], 1)
        self.assertEqual(groups["classifier"]["missing_parameters"], ["classifier.bias"])
        self.assertFalse(groups["PIF"]["all_present_and_finite"])

    def test_snapshot_is_a_copy(self):
        a = torch.tensor([1., 2.])
        snapshot = cpu_tree({"a": a, "list": [a]})
        a.zero_()
        self.assertTrue(torch.equal(snapshot["a"], torch.tensor([1., 2.])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
