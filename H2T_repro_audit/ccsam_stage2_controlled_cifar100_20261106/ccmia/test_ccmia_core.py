import unittest

import torch
import torch.nn.functional as F

from ccmia_core import (
    SGDInfluenceTracker,
    clean_weighted_ce_gradient,
    decompose_perturbation_gradient_by_source,
    max_abs_and_relative_error,
    project_gradient_to_fixed_margin,
    project_source_gradients_to_fixed_margin,
    project_source_parameter_delta_to_fixed_margin,
)


class CCMIACoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261117)
        self.device = torch.device("cpu")
        self.num_classes = 5
        self.feature_dim = 7
        self.batch = 19
        self.features = torch.randn(self.batch, self.feature_dim, device=self.device)
        self.labels = torch.randint(0, self.num_classes, (self.batch,), device=self.device)
        self.weights = torch.linspace(0.4, 1.7, self.num_classes, device=self.device)
        self.delta = 0.08 * torch.randn_like(self.features)
        self.alpha = 0.23
        self.head = torch.nn.Linear(self.feature_dim, self.num_classes).to(self.device)

    def _c2_loss(self, head, features):
        logits = head(features)
        ce = F.cross_entropy(logits, self.labels, reduction="none")
        return self.alpha * (ce * self.weights[self.labels]).mean() + (1.0 - self.alpha) * ce.mean()

    def _c3_loss(self, head, features):
        logits_clean = head(features)
        logits_adv = head(features + self.delta)
        clean = F.cross_entropy(logits_clean, self.labels, reduction="none")
        adv = F.cross_entropy(logits_adv, self.labels, reduction="none")
        return self.alpha * (adv * self.weights[self.labels]).mean() + (1.0 - self.alpha) * clean.mean()

    def test_clean_weighted_gradient_matches_autograd(self):
        head = self.head
        loss = self._c2_loss(head, self.features)
        grad_w, grad_b = torch.autograd.grad(loss, (head.weight, head.bias))
        analytic_w, analytic_b = clean_weighted_ce_gradient(
            self.features, self.labels, head(self.features).detach(), self.weights, self.alpha
        )
        self.assertTrue(torch.allclose(grad_w, analytic_w, rtol=1e-5, atol=1e-7))
        self.assertTrue(torch.allclose(grad_b, analytic_b, rtol=1e-5, atol=1e-7))

    def test_source_decomposition_matches_same_state_autograd_difference(self):
        head = self.head
        train_counts = torch.tensor([500, 150, 80, 10, 5], dtype=torch.long)
        many_input = train_counts[self.labels] > 100
        masked_delta = self.delta.masked_fill(many_input.unsqueeze(1), 0.0)
        logits_clean = head(self.features)
        logits_adv = head(self.features + masked_delta)
        loss3 = self.alpha * (
            F.cross_entropy(logits_adv, self.labels, reduction="none") * self.weights[self.labels]
        ).mean() + (1.0 - self.alpha) * F.cross_entropy(logits_clean, self.labels, reduction="none").mean()
        grad3_w, grad3_b = torch.autograd.grad(loss3, (head.weight, head.bias))

        source_w, source_b = decompose_perturbation_gradient_by_source(
            self.features, self.labels, masked_delta, logits_clean.detach(), logits_adv.detach(),
            self.weights, self.alpha
        )
        direct_w = source_w.sum(dim=0)
        direct_b = source_b.sum(dim=0)
        clean_w, clean_b = clean_weighted_ce_gradient(
            self.features, self.labels, logits_clean.detach(), self.weights, self.alpha
        )
        self.assertTrue(torch.allclose(grad3_w - clean_w, direct_w, rtol=1e-5, atol=1e-7))
        self.assertTrue(torch.allclose(grad3_b - clean_b, direct_b, rtol=1e-5, atol=1e-7))
        many_sources = torch.tensor([True, True, False, False, False])
        self.assertTrue(torch.equal(source_w[many_sources], torch.zeros_like(source_w[many_sources])))
        self.assertTrue(torch.equal(source_b[many_sources], torch.zeros_like(source_b[many_sources])))

    def test_source_margin_projection_is_additive(self):
        source_w, source_b = decompose_perturbation_gradient_by_source(
            self.features, self.labels, self.delta, self.head(self.features).detach(),
            self.head(self.features + self.delta).detach(), self.weights, self.alpha
        )
        val_features = torch.randn(13, self.feature_dim)
        val_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
        fixed_competitors = torch.tensor([2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2], dtype=torch.long)
        many_mask = torch.ones(13, dtype=torch.bool)

        by_source = project_source_gradients_to_fixed_margin(
            source_w, source_b, val_features, val_labels, fixed_competitors,
            many_mask, learning_rate=0.01
        )
        total = project_gradient_to_fixed_margin(
            source_w.sum(dim=0), source_b.sum(dim=0), val_features, val_labels,
            fixed_competitors, many_mask, learning_rate=0.01
        )
        self.assertTrue(torch.allclose(by_source["margin_step"].sum(0), total["margin_step"], rtol=1e-5, atol=1e-7))

    def test_sgd_component_tracker_reconstructs_paired_optimizer(self):
        torch.manual_seed(99)
        head2 = torch.nn.Linear(self.feature_dim, self.num_classes)
        head3 = torch.nn.Linear(self.feature_dim, self.num_classes)
        head3.load_state_dict(head2.state_dict())
        opt2 = torch.optim.SGD(head2.parameters(), lr=0.013, momentum=0.9, weight_decay=0.002)
        opt3 = torch.optim.SGD(head3.parameters(), lr=0.013, momentum=0.9, weight_decay=0.002)
        tracker = SGDInfluenceTracker(
            self.num_classes, self.feature_dim, torch.device("cpu"),
            momentum=0.9, learning_rate=0.013, weight_decay=0.002
        )

        for step in range(4):
            x = torch.randn(self.batch, self.feature_dim)
            y = torch.randint(0, self.num_classes, (self.batch,))
            d = 0.04 * torch.randn_like(x)
            w = self.weights
            z2 = head2(x)
            z3 = head3(x)
            clean2 = F.cross_entropy(z2, y, reduction="none")
            clean3 = F.cross_entropy(z3, y, reduction="none")
            adv3 = F.cross_entropy(head3(x + d), y, reduction="none")
            loss2 = self.alpha * (clean2 * w[y]).mean() + (1.0 - self.alpha) * clean2.mean()
            loss3 = self.alpha * (adv3 * w[y]).mean() + (1.0 - self.alpha) * clean3.mean()

            source_w, source_b = decompose_perturbation_gradient_by_source(
                x, y, d, z3.detach(), head3(x + d).detach(), w, self.alpha
            )
            clean3_w, clean3_b = clean_weighted_ce_gradient(x, y, z3.detach(), w, self.alpha)
            clean2_w, clean2_b = clean_weighted_ce_gradient(x, y, z2.detach(), w, self.alpha)
            state_w, state_b = clean3_w - clean2_w, clean3_b - clean2_b
            before_dw = head3.weight.detach() - head2.weight.detach()
            before_db = head3.bias.detach() - head2.bias.detach()

            opt2.zero_grad(set_to_none=True)
            opt3.zero_grad(set_to_none=True)
            loss2.backward()
            loss3.backward()
            direct_sum_w, direct_sum_b = source_w.sum(0), source_b.sum(0)
            self.assertTrue(torch.allclose(head3.weight.grad - clean3_w, direct_sum_w, rtol=1e-5, atol=1e-7))
            self.assertTrue(torch.allclose(head3.bias.grad - clean3_b, direct_sum_b, rtol=1e-5, atol=1e-7))
            self.assertTrue(torch.allclose(head2.weight.grad, clean2_w, rtol=1e-5, atol=1e-7))
            self.assertTrue(torch.allclose(head2.bias.grad, clean2_b, rtol=1e-5, atol=1e-7))

            tracker.advance(source_w, source_b, state_w, state_b, before_dw, before_db)
            opt2.step()
            opt3.step()

            components = tracker.parameter_components()
            predicted_w = components["direct_weight"] + components["state_weight"] + components["decay_weight"]
            predicted_b = components["direct_bias"] + components["state_bias"] + components["decay_bias"]
            actual_w = head3.weight.detach() - head2.weight.detach()
            actual_b = head3.bias.detach() - head2.bias.detach()
            max_w, rel_w = max_abs_and_relative_error(actual_w, predicted_w)
            max_b, rel_b = max_abs_and_relative_error(actual_b, predicted_b)
            self.assertLess(max_w, 2e-6)
            self.assertLess(max_b, 2e-6)
            self.assertLess(rel_w, 2e-5)
            self.assertLess(rel_b, 2e-5)

            buffer_components = tracker.buffer_components()
            predicted_buffer_w = (
                buffer_components["direct_weight"]
                + buffer_components["state_weight"]
                + buffer_components["decay_weight"]
            )
            predicted_buffer_b = (
                buffer_components["direct_bias"]
                + buffer_components["state_bias"]
                + buffer_components["decay_bias"]
            )
            actual_buffer_w = (
                opt3.state[head3.weight]["momentum_buffer"]
                - opt2.state[head2.weight]["momentum_buffer"]
            )
            actual_buffer_b = (
                opt3.state[head3.bias]["momentum_buffer"]
                - opt2.state[head2.bias]["momentum_buffer"]
            )
            self.assertLess(max_abs_and_relative_error(actual_buffer_w, predicted_buffer_w)[0], 2e-6)
            self.assertLess(max_abs_and_relative_error(actual_buffer_b, predicted_buffer_b)[0], 2e-6)

        # Parameter-component projections accept FP32 validation features and FP64 trackers.
        validation_features = torch.randn(6, self.feature_dim)
        validation_labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        fixed_competitors = torch.tensor([2, 3, 4, 2, 3, 4], dtype=torch.long)
        many_mask = torch.ones(6, dtype=torch.bool)
        projected = project_source_parameter_delta_to_fixed_margin(
            tracker.direct_parameter_weight, tracker.direct_parameter_bias,
            validation_features, validation_labels, fixed_competitors, many_mask
        )
        self.assertEqual(tuple(projected["margin_delta"].shape), (self.num_classes, 6))

    def test_nonfinite_inputs_fail_closed(self):
        bad = self.features.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            clean_weighted_ce_gradient(
                bad, self.labels, self.head(bad).detach(), self.weights, self.alpha
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
