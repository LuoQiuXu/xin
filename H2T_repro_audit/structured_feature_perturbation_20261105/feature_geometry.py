"""Low-rank augmentation subspaces and boundary-directed feature perturbations.

All geometry is estimated from TRAIN features only. The module is NumPy-only so
its mathematical behavior can be unit-tested independently of the training loop.
"""
from __future__ import annotations

import numpy as np


EPS = 1e-12


def _matrix(value, name, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite rank-2 matrix")
    return array


def _canonicalize_eigenvectors(vectors):
    """Fix arbitrary eigenvector signs for reproducible serialized bases."""
    result = np.asarray(vectors, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def estimate_class_subspaces(
    view_a,
    view_b,
    labels,
    *,
    num_classes=100,
    rank=8,
    shrinkage_tau=20.0,
):
    """Estimate class-conditional low-rank subspaces from paired TRAIN views.

    For each class, covariance is computed from centered within-image view
    differences d_i = z_i(view_a) - z_i(view_b). Small classes are shrunk
    toward an equal-class-weighted global within-class covariance:
        C_c^shrunk = n_c/(n_c+tau) C_c + tau/(n_c+tau) C_global.
    """
    a = _matrix(view_a, "view_a")
    b = _matrix(view_b, "view_b")
    y = np.asarray(labels, dtype=np.int64)
    if a.shape != b.shape or y.shape != (a.shape[0],):
        raise ValueError("paired views and labels have incompatible shapes")
    if num_classes <= 1 or rank <= 0 or rank > a.shape[1]:
        raise ValueError("invalid num_classes or subspace rank")
    if shrinkage_tau < 0:
        raise ValueError("shrinkage_tau must be nonnegative")
    if y.size == 0 or y.min() < 0 or y.max() >= num_classes:
        raise ValueError("labels must be nonempty and lie in [0, num_classes)")

    counts = np.bincount(y, minlength=num_classes).astype(np.int64)
    if np.any(counts < 2):
        missing = np.flatnonzero(counts < 2).tolist()
        raise ValueError(f"each class needs at least two paired views; invalid classes={missing}")

    differences = a - b
    class_covariances = []
    for class_id in range(num_classes):
        values = differences[y == class_id]
        centered = values - values.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / float(max(len(values) - 1, 1))
        class_covariances.append((covariance + covariance.T) * 0.5)

    # Equal class weighting prevents head classes from dominating the shared prior.
    global_covariance = np.mean(np.stack(class_covariances, axis=0), axis=0)
    global_values, global_vectors = np.linalg.eigh(global_covariance)
    global_order = np.argsort(global_values)[::-1][:rank]
    global_values = np.maximum(global_values[global_order], 0.0)
    global_vectors = _canonicalize_eigenvectors(global_vectors[:, global_order])

    basis = np.empty((num_classes, a.shape[1], rank), dtype=np.float32)
    eigenvalues = np.empty((num_classes, rank), dtype=np.float32)
    shrinkage = counts.astype(np.float64) / (counts.astype(np.float64) + shrinkage_tau)
    trace_before = np.empty(num_classes, dtype=np.float64)
    trace_after = np.empty(num_classes, dtype=np.float64)

    for class_id, covariance in enumerate(class_covariances):
        weight = float(shrinkage[class_id])
        shrunk = weight * covariance + (1.0 - weight) * global_covariance
        shrunk = (shrunk + shrunk.T) * 0.5
        values, vectors = np.linalg.eigh(shrunk)
        order = np.argsort(values)[::-1][:rank]
        values = np.maximum(values[order], 0.0)
        vectors = _canonicalize_eigenvectors(vectors[:, order])

        if float(values.sum()) <= EPS:
            values = global_values.copy()
            vectors = global_vectors.copy()

        basis[class_id] = vectors.astype(np.float32)
        eigenvalues[class_id] = values.astype(np.float32)
        trace_before[class_id] = float(np.trace(covariance))
        trace_after[class_id] = float(np.trace(shrunk))

    gram = np.einsum("cdk,cdl->ckl", basis, basis)
    identity = np.eye(rank, dtype=np.float32)[None, :, :]
    max_orthogonality_error = float(np.max(np.abs(gram - identity)))
    if max_orthogonality_error > 2e-4:
        raise AssertionError(f"subspace basis is not orthonormal: {max_orthogonality_error}")

    diagnostics = {
        "num_classes": int(num_classes),
        "feature_dim": int(a.shape[1]),
        "rank": int(rank),
        "shrinkage_tau": float(shrinkage_tau),
        "class_counts": counts.tolist(),
        "shrinkage_weights": shrinkage.tolist(),
        "global_covariance_trace": float(np.trace(global_covariance)),
        "mean_class_covariance_trace": float(trace_before.mean()),
        "mean_shrunk_covariance_trace": float(trace_after.mean()),
        "max_basis_orthogonality_error": max_orthogonality_error,
        "paired_view_difference_mean_norm": float(np.linalg.norm(differences.mean(axis=0))),
        "paired_view_difference_mean_sample_norm": float(np.linalg.norm(differences, axis=1).mean()),
    }
    return basis, eigenvalues, diagnostics


def make_boundary_targets(
    features,
    labels,
    classifier_weight,
    classifier_bias,
    basis,
    class_counts,
    *,
    alpha=0.5,
    max_norm_ratio=0.25,
    min_alignment=0.05,
    eps=EPS,
):
    """Move eligible tail features partway toward their nearest fixed boundary.

    The competitor is selected by the frozen Stage-I classifier. For g=w_y-w_j,
    project g into the class augmentation subspace P_c g and move in direction
    -P_c g. The radius is min(alpha * margin / ||P_c g||,
    max_norm_ratio * ||z||), so the selected pairwise margin is not crossed.
    """
    z = _matrix(features, "features")
    y = np.asarray(labels, dtype=np.int64)
    weight = _matrix(classifier_weight, "classifier_weight")
    bias = np.asarray(classifier_bias, dtype=np.float64)
    u_all = np.asarray(basis, dtype=np.float64)
    counts = np.asarray(class_counts, dtype=np.int64)

    n, dim = z.shape
    num_classes = weight.shape[0]
    if weight.shape != (num_classes, dim) or bias.shape != (num_classes,):
        raise ValueError("classifier dimensions do not match features")
    if y.shape != (n,) or counts.shape != (num_classes,):
        raise ValueError("labels or class_counts have incompatible shapes")
    if u_all.ndim != 3 or u_all.shape[:2] != (num_classes, dim):
        raise ValueError("basis must have shape [num_classes, feature_dim, rank]")
    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError("labels out of range")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    if max_norm_ratio <= 0 or min_alignment < 0:
        raise ValueError("invalid perturbation cap or alignment threshold")

    logits = z @ weight.T + bias[None, :]
    competitors_logits = logits.copy()
    rows = np.arange(n)
    competitors_logits[rows, y] = -np.inf
    competitor = competitors_logits.argmax(axis=1)
    margin = logits[rows, y] - logits[rows, competitor]
    normal = weight[y] - weight[competitor]

    class_basis = u_all[y]  # [N, D, K]
    coefficients = np.einsum("ndk,nd->nk", class_basis, normal)
    projected_normal = np.einsum("ndk,nk->nd", class_basis, coefficients)
    projected_norm = np.linalg.norm(projected_normal, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    alignment = projected_norm / np.maximum(normal_norm, eps)
    feature_norm = np.linalg.norm(z, axis=1)

    is_tail = counts[y] < 20
    eligible = (
        is_tail
        & (margin > 0.0)
        & (alignment >= min_alignment)
        & (projected_norm > eps)
        & (feature_norm > eps)
    )

    boundary_distance = np.divide(
        margin,
        projected_norm,
        out=np.zeros_like(margin),
        where=projected_norm > eps,
    )
    unconstrained_radius = alpha * boundary_distance
    norm_cap = max_norm_ratio * feature_norm
    radius = np.minimum(unconstrained_radius, norm_cap)
    radius[~eligible] = 0.0

    unit_direction = np.divide(
        projected_normal,
        np.maximum(projected_norm[:, None], eps),
        out=np.zeros_like(projected_normal),
        where=projected_norm[:, None] > eps,
    )
    delta = -radius[:, None] * unit_direction
    margin_delta = np.einsum("nd,nd->n", normal, delta)
    margin_after = margin + margin_delta
    clipped = eligible & (norm_cap < unconstrained_radius)

    expected_reduction = np.minimum(alpha * margin, norm_cap * projected_norm)
    if eligible.any():
        if not np.allclose(
            -margin_delta[eligible],
            expected_reduction[eligible],
            atol=2e-7,
            rtol=2e-6,
        ):
            raise AssertionError("boundary margin reduction does not match the construction")
        if np.any(margin_after[eligible] < -2e-7):
            raise AssertionError("boundary perturbation crossed the selected competitor boundary")
    if np.any(radius < -eps) or not np.isfinite(delta).all():
        raise AssertionError("invalid boundary perturbation")

    return {
        "delta": delta.astype(np.float32),
        "radius": radius.astype(np.float32),
        "margin": margin.astype(np.float32),
        "margin_after": margin_after.astype(np.float32),
        "margin_reduction": (-margin_delta).astype(np.float32),
        "alignment": alignment.astype(np.float32),
        "competitor": competitor.astype(np.int64),
        "eligible": eligible.astype(np.uint8),
        "clipped": clipped.astype(np.uint8),
        "boundary_distance": boundary_distance.astype(np.float32),
        "tail": is_tail.astype(np.uint8),
    }


def sample_isotropic_directions(rng, count, dim):
    """Sample unit directions uniformly from an isotropic Gaussian."""
    values = rng.standard_normal((count, dim))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise AssertionError("degenerate isotropic direction")
    return values / norms


def sample_subspace_directions(rng, labels, basis, eigenvalues):
    """Sample augmentation-structured directions in each label's learned subspace."""
    y = np.asarray(labels, dtype=np.int64)
    u = np.asarray(basis, dtype=np.float64)[y]
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64)[y], 0.0)
    coefficients = rng.standard_normal(values.shape) * np.sqrt(values)
    directions = np.einsum("ndk,nk->nd", u, coefficients)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)

    degenerate = norms[:, 0] <= EPS
    if degenerate.any():
        fallback_coefficients = rng.standard_normal(values[degenerate].shape)
        fallback = np.einsum(
            "ndk,nk->nd",
            u[degenerate],
            fallback_coefficients,
        )
        fallback_norms = np.linalg.norm(fallback, axis=1, keepdims=True)
        fallback_degenerate = fallback_norms[:, 0] <= EPS
        if fallback_degenerate.any():
            fallback[fallback_degenerate] = u[degenerate][fallback_degenerate, :, 0]
            fallback_norms = np.linalg.norm(fallback, axis=1, keepdims=True)
        directions[degenerate] = fallback
        norms[degenerate] = fallback_norms

    if np.any(norms <= EPS):
        raise AssertionError("degenerate subspace direction")
    return directions / norms


def scale_directions(directions, radii):
    """Apply a per-example norm budget to unit directions."""
    direction = _matrix(directions, "directions")
    radius = np.asarray(radii, dtype=np.float64)
    if radius.shape != (direction.shape[0],) or not np.isfinite(radius).all():
        raise ValueError("radii must be a finite vector matching directions")
    if np.any(radius < 0):
        raise ValueError("radii must be nonnegative")
    norms = np.linalg.norm(direction, axis=1)
    unit = np.divide(
        direction,
        np.maximum(norms[:, None], EPS),
        out=np.zeros_like(direction),
        where=norms[:, None] > EPS,
    )
    return (unit * radius[:, None]).astype(np.float32)
