"""Paired GAP-norm control for H2T Low-VS versus Uniform.

Both strategies first make exactly the historical 16-channel hard replacement.
For norm-matched controls, move both perturbations toward the receiver so that
their pooled L2 norms equal min(original_uniform_norm, original_low_norm).
No trainable policy, no logits, no validation/test signal.
"""
from __future__ import annotations
import torch

ARMS = ("Uniform-Full", "LowVS-Full", "Uniform-Matched", "LowVS-Matched")
EPS = 1e-12

@torch.no_grad()
def paired_norm_control(receiver: torch.Tensor, donor: torch.Tensor,
                        uniform_mask: torch.Tensor, low_mask: torch.Tensor):
    assert receiver.shape == donor.shape and receiver.ndim == 4
    assert receiver.shape[1] == 64 and uniform_mask.shape == low_mask.shape == (len(receiver), 64)
    assert uniform_mask.dtype == low_mask.dtype == torch.bool
    assert (uniform_mask.sum(1) == 16).all() and (low_mask.sum(1) == 16).all()
    assert torch.isfinite(receiver).all() and torch.isfinite(donor).all()

    diff = donor - receiver
    gap_diff = diff.mean(dim=(-1, -2))
    u = uniform_mask.to(gap_diff.dtype)
    l = low_mask.to(gap_diff.dtype)
    nu = torch.linalg.vector_norm(gap_diff * u, dim=1)
    nl = torch.linalg.vector_norm(gap_diff * l, dim=1)
    target = torch.minimum(nu, nl)
    au = torch.where(nu > EPS, target / nu.clamp_min(EPS), torch.zeros_like(nu)).clamp(0, 1)
    al = torch.where(nl > EPS, target / nl.clamp_min(EPS), torch.zeros_like(nl)).clamp(0, 1)

    uniform_full = torch.where(uniform_mask[..., None, None], donor, receiver)
    low_full = torch.where(low_mask[..., None, None], donor, receiver)
    matched_u = receiver + au[:, None, None, None] * (uniform_full - receiver)
    matched_l = receiver + al[:, None, None, None] * (low_full - receiver)
    actual_u = torch.linalg.vector_norm((matched_u - receiver).mean(dim=(-1, -2)), dim=1)
    actual_l = torch.linalg.vector_norm((matched_l - receiver).mean(dim=(-1, -2)), dim=1)
    tolerance = 1e-5 + 1e-4 * target
    if not (torch.all((actual_u - actual_l).abs() <= tolerance) and
            torch.all((actual_u - target).abs() <= tolerance)):
        raise AssertionError("Matched GAP L2 mismatch (float32 tolerance)")
    for name, alpha in (("uniform", au), ("lowvs", al)):
        assert torch.isfinite(alpha).all() and (alpha >= 0).all() and (alpha <= 1).all(), name
    return {
        "Uniform-Full": uniform_full,
        "LowVS-Full": low_full,
        "Uniform-Matched": matched_u,
        "LowVS-Matched": matched_l,
    }, {
        "original_norm_uniform": nu,
        "original_norm_lowvs": nl,
        "target_norm": target,
        "matched_norm_uniform": actual_u,
        "matched_norm_lowvs": actual_l,
        "alpha_uniform": au,
        "alpha_lowvs": al,
        "zero_norm_pairs": ((nu <= EPS) | (nl <= EPS)).to(torch.int32),
    }

def test_policy():
    torch.manual_seed(20261016)
    b = 20
    receiver = torch.randn(b, 64, 8, 8)
    donor = torch.randn_like(receiver)
    u = torch.zeros(b, 64, dtype=torch.bool)
    l = torch.zeros_like(u)
    for i in range(b):
        u[i, torch.randperm(64)[:16]] = True
        l[i, torch.randperm(64)[:16]] = True
    output, diag = paired_norm_control(receiver, donor, u, l)
    assert len(output) == len(ARMS)
    assert torch.equal(output["Uniform-Full"][u[..., None, None].expand_as(receiver)],
                       donor[u[..., None, None].expand_as(receiver)])
    assert torch.equal(output["LowVS-Full"][~l[..., None, None].expand_as(receiver)],
                       receiver[~l[..., None, None].expand_as(receiver)])
    assert torch.allclose(diag["matched_norm_uniform"], diag["matched_norm_lowvs"], rtol=1e-4, atol=1e-5)
    assert ((diag["alpha_uniform"] <= 1) & (diag["alpha_lowvs"] <= 1)).all()
    eq, q = paired_norm_control(receiver, donor, u, u)
    assert torch.equal(eq["Uniform-Full"], eq["LowVS-Full"])
    assert torch.allclose(eq["Uniform-Matched"], eq["LowVS-Matched"])
    eq, q = paired_norm_control(receiver, receiver, u, l)
    assert all(torch.isfinite(x).all() for x in eq.values())
    assert q["zero_norm_pairs"].sum().item() == b
    print("PASS: 16-channel identity, finite zero norm, exact full endpoints, paired GAP norms")

if __name__ == "__main__":
    test_policy()
