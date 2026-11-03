"""Gradient auditing and exact optimizer-step accounting.

The narrow PIF AMP-overflow diagnostic path is retained for historical,
disposable forensic tools. The clean-launch Smoke and formal runner use full
FP32, construct a disabled/no-op GradScaler, and never permit that recovery:
all non-finite gradients are fatal. No gradient is clamped or replaced.
"""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch

PIF_B_PARAMETER = "extractor.pif.fuse.weight"
PIF_B_SHAPE = (1, 2, 1)
PIF_B_INDEX = (0, 1, 0)


class GradientContractError(RuntimeError):
    """A gradient state is outside the explicitly audited AMP-overflow case.

    diagnostics is a small, JSON-serializable description of the rejected state.
    Attaching it never changes the fail-closed decision.
    """
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


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
    """No-op scaler adapter; Fish16 clean-launch uses full FP32."""
    if not torch.cuda.is_available():
        raise RuntimeError("The Fish16 runner requires CUDA.")
    return torch.amp.GradScaler("cuda", init_scale=1.0, enabled=False)


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
            "Missing gradients are never recoverable: " + ", ".join(missing[:20]),
            diagnostics={
                "missing_parameters": missing,
                "nan_count": 0,
                "inf_count": 0,
                "nonfinite_parameters": [],
                "nonfinite_positions": {},
                "nonfinite_details": [],
                "scaler_step_called": False,
                "scaler_update_called": False,
            },
        )

    nan_count = 0
    inf_count = 0
    nonfinite_positions = {}
    nonfinite_parameters = []
    nonfinite_details = []
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
        n_inf = int(infs.sum().item())
        positions = torch.nonzero(infs, as_tuple=False)[:32].cpu().tolist()
        nan_count += n_nan
        inf_count += n_inf

        if n_nan or n_inf:
            positive_inf_count = int(torch.isposinf(gradient).sum().item())
            negative_inf_count = int(torch.isneginf(gradient).sum().item())
            nan_positions = torch.nonzero(nans, as_tuple=False)[:32].cpu().tolist()
            nonfinite_parameters.append(name)
            nonfinite_positions[name] = positions
            if n_nan:
                nonfinite_positions[name + "::nan_count"] = n_nan
            nonfinite_details.append({
                "name": name,
                "shape": list(parameter.shape),
                "gradient_dtype": str(gradient.dtype),
                "nan_count": n_nan,
                "positive_inf_count": positive_inf_count,
                "negative_inf_count": negative_inf_count,
                "inf_count": n_inf,
                "inf_positions": positions[:32],
                "inf_positions_truncated": n_inf > 32,
                "nan_positions": nan_positions[:32],
                "nan_positions_truncated": n_nan > 32,
            })

    if not any_nonzero:
        raise GradientContractError(
            "All optimizer gradients are zero.",
            diagnostics={
                "missing_parameters": [],
                "nan_count": nan_count,
                "inf_count": inf_count,
                "nonfinite_parameters": sorted(nonfinite_parameters),
                "nonfinite_positions": nonfinite_positions,
                "nonfinite_details": nonfinite_details,
                "scaler_step_called": False,
                "scaler_update_called": False,
            },
        )

    return {
        "missing_parameters": [],
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_parameters": tuple(sorted(nonfinite_parameters)),
        "nonfinite_positions": nonfinite_positions,
        "nonfinite_details": nonfinite_details,
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
    # GradScaler's overflow detector is sign-agnostic: both +Inf and -Inf
    # are non-finite. Keep the exception narrow in parameter, coordinate,
    # count, and absence of NaNs; do not accept any other non-finite pattern.
    return bool(torch.isinf(parameter.grad[PIF_B_INDEX]).item())


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
    The only permitted non-finite case is one +Inf or -Inf at
    PIF fuse.weight[0,1,0]. In that case GradScaler must skip the optimizer
    update and lower its scale.
    NaNs, missing gradients, any other Inf, or an unverified skip are fatal.
    """
    _validate_optimizer_parameter_set(optimizer, named_parameters)
    diagnostics = _inspect_gradients(named_parameters)
    has_nonfinite = diagnostics["inf_count"] > 0 or diagnostics["nan_count"] > 0

    if has_nonfinite:
        if not allow_pif_b_overflow:
            raise GradientContractError(
                "Non-finite gradients are not permitted for this condition: "
                + ", ".join(diagnostics["nonfinite_parameters"]),
                diagnostics={**diagnostics, "scaler_step_called": False, "scaler_update_called": False},
            )
        if not _is_the_one_audited_pif_overflow(named_parameters, diagnostics):
            raise GradientContractError(
                "Non-finite gradients do not match the single audited PIF b overflow.",
                diagnostics={
                    **diagnostics,
                    "expected_event": {
                        "parameter": PIF_B_PARAMETER,
                        "shape": list(PIF_B_SHAPE),
                        "index": list(PIF_B_INDEX),
                        "sign": "positive_or_negative_inf",
                        "nan_count": 0,
                        "inf_count": 1,
                    },
                    "allow_pif_b_overflow": True,
                    "matches_audited_event": False,
                    "scaler_step_called": False,
                    "scaler_update_called": False,
                },
            )
        if not callable(getattr(scaler, "is_enabled", None)) or not scaler.is_enabled():
            raise GradientContractError(
                "The audited PIF overflow can only be handled by an enabled GradScaler.",
                diagnostics={**diagnostics, "scaler_step_called": False, "scaler_update_called": False},
            )
        if consecutive_overflow_skips >= max_consecutive_overflow_skips:
            raise GradientContractError(
                "Maximum consecutive recoverable AMP overflow skips exceeded.",
                diagnostics={**diagnostics, "scaler_step_called": False, "scaler_update_called": False},
            )

    scale_before = float(scaler.get_scale())
    if not math.isfinite(scale_before) or scale_before <= 0:
        raise GradientContractError(
            f"Invalid GradScaler scale before step: {scale_before}",
            diagnostics={**diagnostics, "scaler_step_called": False, "scaler_update_called": False},
        )

    steps_before = int(step_counter.count)
    parameter_before = parameter_sha(named_parameters) if has_nonfinite else None

    # unscale_ must already have been called by the training loop.  GradScaler
    # owns the overflow decision and performs the actual skip/backoff.
    scaler.step(optimizer)
    scaler.update()

    steps_after = int(step_counter.count)
    scale_after = float(scaler.get_scale())
    if not math.isfinite(scale_after) or scale_after <= 0:
        raise GradientContractError(
            f"Invalid GradScaler scale after step: {scale_after}",
            diagnostics={**diagnostics, "scaler_step_called": True, "scaler_update_called": True},
        )

    if has_nonfinite:
        parameter_after = parameter_sha(named_parameters)
        if steps_after != steps_before:
            raise GradientContractError(
                "GradScaler called optimizer.step() despite the diagnosed Inf.",
                diagnostics={**diagnostics, "scaler_step_called": True, "scaler_update_called": True},
            )
        if not scale_after < scale_before:
            raise GradientContractError(
                "GradScaler did not lower its scale after the diagnosed Inf.",
                diagnostics={**diagnostics, "scaler_step_called": True, "scaler_update_called": True},
            )
        if parameter_after != parameter_before:
            raise GradientContractError(
                "Model parameters changed during a supposedly skipped optimizer step.",
                diagnostics={**diagnostics, "scaler_step_called": True, "scaler_update_called": True},
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
            "GradScaler skipped an optimizer update although the gradient audit was finite.",
            diagnostics={**diagnostics, "scaler_step_called": True, "scaler_update_called": True},
        )
    if scale_after < scale_before:
        raise GradientContractError(
            "GradScaler lowered its scale although the gradient audit found no non-finite values.",
            diagnostics={**diagnostics, "scaler_step_called": True, "scaler_update_called": True},
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
