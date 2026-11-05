import unittest

import numpy as np
import torch
import torch.nn.functional as F

from feature_geometry import (
    estimate_class_subspaces,
    make_boundary_targets,
    sample_isotropic_directions,
    sample_subspace_directions,
    scale_directions,
)


class FeatureGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261105)

    def test_class_subspaces_are_orthonormal_and_finite(self):
        num_classes, dim, samples_per_class, rank = 4, 16, 12, 5
        labels = np.repeat(np.arange(num_classes), samples_per_class)
        base = self.rng.normal(size=(num_classes, dim))
        view_a = base[labels] + self.rng.normal(scale=0.3, size=(len(labels), dim))
        view_b = base[labels] + self.rng.normal(scale=0.3, size=(len(labels), dim))
        basis, eigenvalues, diagnostics = estimate_class_subspaces(
            view_a,
            view_b,
            labels,
            num_classes=num_classes,
            rank=rank,
            shrinkage_tau=8.0,
        )
        self.assertEqual(basis.shape, (num_classes, dim, rank))
        self.assertEqual(eigenvalues.shape, (num_classes, rank))
        self.assertTrue(np.isfinite(basis).all())
        self.assertTrue(np.isfinite(eigenvalues).all())
        self.assertTrue((eigenvalues >= 0).all())
        gram = np.einsum("cdk,cdl->ckl", basis, basis)
        self.assertTrue(np.allclose(gram, np.eye(rank)[None], atol=2e-4))
        self.assertLess(diagnostics["max_basis_orthogonality_error"], 2e-4)

    def test_boundary_perturbation_reduces_but_does_not_cross_margin(self):
        num_classes, dim, rank = 4, 16, 4
        basis = np.zeros((num_classes, dim, rank), dtype=np.float32)
        basis[:, :rank, :] = np.eye(rank, dtype=np.float32)[None]
        weight = np.zeros((num_classes, dim), dtype=np.float64)
        weight[:, :num_classes] = np.eye(num_classes)
        bias = np.zeros(num_classes, dtype=np.float64)
        labels = np.array([2, 3, 2, 3], dtype=np.int64)
        class_counts = np.array([100, 100, 5, 5], dtype=np.int64)
        features = np.zeros((len(labels), dim), dtype=np.float64)
        features[np.arange(len(labels)), labels] = 2.0
        features[:, 0] += 0.05
        features[:, 1] += 0.02

        result = make_boundary_targets(
            features,
            labels,
            weight,
            bias,
            basis,
            class_counts,
            alpha=0.5,
            max_norm_ratio=0.25,
            min_alignment=0.05,
        )
        eligible = result["eligible"].astype(bool)
        self.assertTrue(eligible.all())
        self.assertTrue((result["margin_after"][eligible] > 0).all())
        self.assertTrue((result["margin_reduction"][eligible] > 0).all())
        self.assertTrue(
            np.allclose(
                np.linalg.norm(result["delta"], axis=1),
                result["radius"],
                atol=2e-6,
            )
        )
        self.assertTrue(
            np.allclose(
                result["margin"] - result["margin_after"],
                result["margin_reduction"],
                atol=2e-6,
            )
        )
        self.assertTrue((result["tail"] == 1).all())

    def test_isotropic_and_subspace_directions_are_unit_norm(self):
        num_classes, dim, rank = 5, 20, 6
        labels = np.repeat(np.arange(num_classes), 7)
        a = self.rng.normal(size=(len(labels), dim))
        b = self.rng.normal(size=(len(labels), dim))
        basis, eigenvalues, _ = estimate_class_subspaces(
            a, b, labels, num_classes=num_classes, rank=rank
        )
        gaussian = sample_isotropic_directions(self.rng, len(labels), dim)
        structured = sample_subspace_directions(
            self.rng, labels, basis, eigenvalues
        )
        self.assertTrue(np.allclose(np.linalg.norm(gaussian, axis=1), 1.0, atol=1e-10))
        self.assertTrue(np.allclose(np.linalg.norm(structured, axis=1), 1.0, atol=1e-6))
        projected = np.einsum(
            "ndk,nk->nd",
            basis[labels],
            np.einsum("ndk,nd->nk", basis[labels], structured),
        )
        self.assertTrue(np.allclose(projected, structured, atol=2e-5))

    def test_all_generated_arms_can_share_exact_norm_budget(self):
        count, dim = 32, 64
        radius = self.rng.uniform(0.0, 0.2, size=count)
        gaussian = sample_isotropic_directions(self.rng, count, dim)
        gaussian_delta = scale_directions(gaussian, radius)
        labels = np.zeros(count, dtype=np.int64)
        basis = np.zeros((1, dim, 8), dtype=np.float32)
        basis[0, :8, :] = np.eye(8, dtype=np.float32)
        eigenvalues = np.ones((1, 8), dtype=np.float32)
        structured = sample_subspace_directions(
            self.rng, labels, basis, eigenvalues
        )
        structured_delta = scale_directions(structured, radius)
        self.assertTrue(
            np.allclose(np.linalg.norm(gaussian_delta, axis=1), radius, atol=2e-6)
        )
        self.assertTrue(
            np.allclose(np.linalg.norm(structured_delta, axis=1), radius, atol=2e-6)
        )

    def test_h2t_channel_replacement_commutes_with_global_average_pooling(self):
        generator = torch.Generator(device="cpu").manual_seed(20261105)
        receiver = torch.randn((3, 64, 8, 8), generator=generator)
        donor = torch.randn((3, 64, 8, 8), generator=generator)
        selected = torch.randperm(64, generator=generator)[:16]
        mask = torch.zeros(64, dtype=torch.bool)
        mask[selected] = True
        fused_map = torch.where(mask[None, :, None, None], donor, receiver)
        pooled_after = F.avg_pool2d(fused_map, 8).flatten(1)
        receiver_gap = F.avg_pool2d(receiver, 8).flatten(1)
        donor_gap = F.avg_pool2d(donor, 8).flatten(1)
        fused_gap = torch.where(mask[None, :], donor_gap, receiver_gap)
        self.assertTrue(torch.allclose(pooled_after, fused_gap, atol=1e-7, rtol=1e-7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
