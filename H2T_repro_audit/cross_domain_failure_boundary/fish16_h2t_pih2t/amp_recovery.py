"""Fail-closed handling for the one diagnosed PIF AMP overflow.

This module does not clamp, replace, or otherwise alter gradients.  It permits
GradScaler's native skip/backoff path only when the sole non-finite value is
the already-diagnosed positive Inf at PIF coefficient b at [0, 1, 0].
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch

PIF_B_PARAMETER = "extractor.pif.fuse.weight"
PIF_B_SHAPE = (1, 2, 1)
PIF_B_INDEX = (0, 1, 0)


class GradientContractError(RuntimeError):
    """A gradient state is outside the explicitly audited AMP-overflow case."""


@dataclass(frozen=True)
class StepOutcome:
    kind: str
    scale_before: float
    scale_after: float
    optimizer_steps_before: int
    optimizer_steps_after: int
    nan_count: int
    inf_count: int
    nonfinite_parameters: tuple
    nonfinite_positions: dict
    parameter_sha_before: str | None = None
    parameter_sha_after: str | None = None

    @property
    def skipped(self):
        return self.kind == "recoverable_pif_b_overflow_skip"

    def as_dict(self):
        return asdict(self)


class OptimizerStepCounter:
    """Count actual optimizer.step() calls, not GradScaler.step() attempts."""

    def __init__(self, optimizer):
        register = getattr(optimizer, "register_step_post_hook", None)
        if not callable(register):
            raise RuntimeError(
                "This PyTorch build must support Optimizer.register_step_post_hook "
                "to verify GradScaler skipped/actual optimizer updates."
            )
        self.count = 0
        self._closed = False
        self._handle = register(self._after_step)

    def _after_step(self, optimizer, args, kwargs):
        self.count += 1

    def close(self):
        if not self._closed:
            self._handle.remove()
            self._closed = True


def make_grad_scaler():
    """Spell out the historical PyTorch defaults so the numeric protocol is locked."""
    if not torch.cuda.is_available():
        raise RuntimeError("The Fish16 AMP recovery gate requires CUDA.")
    return torch.amp.GradScaler(
        "cuda",
        init_scale=65536.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
    )


def parameter_sha(named_parameters):
    """Hash parameters only; BN buffers are intentionally excluded."""
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_optimizer_parameter_set(optimizer, named_parameters):
    names = [name for name, _ in named_parameters]
    if len(names) != len(set(names)):
        raise GradientContractError("Duplicate parameter names in gradient audit.")
    parameter_ids = [id(parameter) for _, parameter in named_parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise GradientContractError("Duplicate parameter objects in gradient audit.")
    if any(not parameter.requires_grad for _, parameter in named_parameters):
        raise GradientContractError("Frozen parameter was passed to the optimizer audit.")

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if set(parameter_ids) != optimizer_ids:
        raise GradientContractError(
            "Gradient audit parameter set does not exactly match optimizer parameter groups."
        )
    if not named_parameters:
        raise GradientContractError("No optimizer parameters were supplied.")


def _inspect_gradients(named_parameters):
    missing = [name for name, parameter in named_parameters if parameter.grad is None]
    if missing:
        raise GradientContractError(
            "Missing gradients are never recoverable: " + ", ".join(missing[:20])
        )

    nan_count = 0
    inf_count = 0
    nonfinite_positions = {}
    nonfinite_parameters = []
    any_nonzero = False

    for name, parameter in named_parameters:
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        if torch.count_nonzero(gradient).item() > 0:
            any_nonzero = True

        nans = torch.isnan(gradient)
        infs = torch.isinf(gradient)
        n_nan = int(nans.sum().item())
        positions = torch.nonzero(infs, as_tuple=False).cpu().tolist()
        n_inf = len(positions)
        nan_count += n_nan
        inf_count += n_inf

        if n_nan or n_inf:
            nonfinite_parameters.append(name)
            nonfinite_positions[name] = positions
            if n_nan:
                nonfinite_positions[name + "::nan_count"] = n_nan

    if not any_nonzero:
        raise GradientContractError("All optimizer gradients are zero.")

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_parameters": tuple(sorted(nonfinite_parameters)),
        "nonfinite_positions": nonfinite_positions,
    }


def _is_the_one_audited_pif_overflow(named_parameters, diagnostics):
    if diagnostics["nan_count"] != 0:
        return False
    if diagnostics["inf_count"] != 1:
        return False
    if diagnostics["nonfinite_parameters"] != (PIF_B_PARAMETER,):
        return False

    parameter = dict(named_parameters).get(PIF_B_PARAMETER)
    if parameter is None or tuple(parameter.shape) != PIF_B_SHAPE:
        return False
    positions = diagnostics["nonfinite_positions"].get(PIF_B_PARAMETER, [])
    if positions != [list(PIF_B_INDEX)]:
        return False
    # The archived exact replay is +Inf at b. Do not broaden this exception
    # to a sign-flipped or otherwise different non-finite event.
    return bool(torch.isposinf(parameter.grad[PIF_B_INDEX]).item())


def guarded_scaler_step(
    *,
    optimizer,
    scaler,
    named_parameters,
    step_counter,
    allow_pif_b_overflow=False,
    consecutive_overflow_skips=0,
    max_consecutive_overflow_skips=2,
):
    """Run one optimizer attempt after scaler.unscale_(optimizer).

    Finite gradients follow the ordinary GradScaler step/update path.
    The only permitted non-finite case is one positive Inf at PIF fuse.weight[0,1,0].
    In that case GradScaler must skip the optimizer update and lower its scale.
    NaNs, missing gradients, any other Inf, or an unverified skip are fatal.
    """
    _validate_optimizer_parameter_set(optimizer, named_parameters)
    diagnostics = _inspect_gradients(named_parameters)
    has_nonfinite = diagnostics["inf_count"] > 0 or diagnostics["nan_count"] > 0

    if has_nonfinite:
        if not allow_pif_b_overflow:
            raise GradientContractError(
                "Non-finite gradients are not permitted for this condition: "
                + ", ".join(diagnostics["nonfinite_parameters"])
            )
        if not _is_the_one_audited_pif_overflow(named_parameters, diagnostics):
            raise GradientContractError(
                "Non-finite gradients do not match the single audited PIF b overflow."
            )
        if not callable(getattr(scaler, "is_enabled", None)) or not scaler.is_enabled():
            raise GradientContractError(
                "The audited PIF overflow can only be handled by an enabled GradScaler."
            )
        if consecutive_overflow_skips >= max_consecutive_overflow_skips:
            raise GradientContractError(
                "Maximum consecutive recoverable AMP overflow skips exceeded."
            )

    scale_before = float(scaler.get_scale())
    if not math.isfinite(scale_before) or scale_before <= 0:
        raise GradientContractError(f"Invalid GradScaler scale before step: {scale_before}")

    steps_before = int(step_counter.count)
    parameter_before = parameter_sha(named_parameters) if has_nonfinite else None

    # unscale_ must already have been called by the training loop.  GradScaler
    # owns the overflow decision and performs the actual skip/backoff.
    scaler.step(optimizer)
    scaler.update()

    steps_after = int(step_counter.count)
    scale_after = float(scaler.get_scale())
    if not math.isfinite(scale_after) or scale_after <= 0:
        raise GradientContractError(f"Invalid GradScaler scale after step: {scale_after}")

    if has_nonfinite:
        parameter_after = parameter_sha(named_parameters)
        if steps_after != steps_before:
            raise GradientContractError(
                "GradScaler called optimizer.step() despite the diagnosed Inf."
            )
        if not scale_after < scale_before:
            raise GradientContractError(
                "GradScaler did not lower its scale after the diagnosed Inf."
            )
        if parameter_after != parameter_before:
            raise GradientContractError(
                "Model parameters changed during a supposedly skipped optimizer step."
            )
        return StepOutcome(
            kind="recoverable_pif_b_overflow_skip",
            scale_before=scale_before,
            scale_after=scale_after,
            optimizer_steps_before=steps_before,
            optimizer_steps_after=steps_after,
            nan_count=diagnostics["nan_count"],
            inf_count=diagnostics["inf_count"],
            nonfinite_parameters=diagnostics["nonfinite_parameters"],
            nonfinite_positions=diagnostics["nonfinite_positions"],
            parameter_sha_before=parameter_before,
            parameter_sha_after=parameter_after,
        )

    if steps_after != steps_before + 1:
        raise GradientContractError(
            "GradScaler skipped an optimizer update although the gradient audit was finite."
        )
    if scale_after < scale_before:
        raise GradientContractError(
            "GradScaler lowered its scale although the gradient audit found no non-finite values."
        )

    return StepOutcome(
        kind="optimizer_step",
        scale_before=scale_before,
        scale_after=scale_after,
        optimizer_steps_before=steps_before,
        optimizer_steps_after=steps_after,
        nan_count=0,
        inf_count=0,
        nonfinite_parameters=(),
        nonfinite_positions={},
    )
