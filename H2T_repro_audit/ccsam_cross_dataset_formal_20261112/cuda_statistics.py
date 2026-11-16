"""Select the workload CUDA device and record memory statistics on that device."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

DEFAULT_CUDA_DEVICE = torch.device("cuda", 0)


def note_diagnostics(error: BaseException, operation: str, diagnostics: dict) -> None:
    error.add_note(f"{operation} CUDA diagnostics: {json.dumps(diagnostics, sort_keys=True)}")


@dataclass(frozen=True)
class CudaStatistics:
    index: int
    diagnostics: dict

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.index)

    def synchronize(self) -> None:
        try:
            torch.cuda.synchronize(self.index)
        except Exception as exc:
            note_diagnostics(exc, "synchronize", self.diagnostics)
            raise

    def reset_peak(self) -> None:
        self.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats(self.index)
        except Exception as exc:
            note_diagnostics(exc, "reset peak memory statistics", self.diagnostics)
            raise

    def peak(self) -> dict[str, int]:
        try:
            return {
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.index),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.index),
            }
        except Exception as exc:
            note_diagnostics(exc, "read peak memory statistics", self.diagnostics)
            raise


def initialize_cuda(device: torch.device, diagnostics: dict | None = None) -> CudaStatistics:
    """Initialize PyTorch CUDA and explicitly select the workload device."""
    if diagnostics is None:
        diagnostics = {}
    requested = torch.device(device)
    diagnostics.update({
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_device": str(requested),
        "requested_device_index": requested.index,
        "device": None,
        "device_index": None,
        "cuda_available": None,
        "device_count": None,
        "initial_current_device": None,
        "current_device": None,
        "gpu": None,
        "gpu_memory_bytes": None,
    })
    try:
        available = torch.cuda.is_available()
        diagnostics["cuda_available"] = available
        if not available:
            raise RuntimeError("CUDA is unavailable in the requested Python environment")
        if requested.type != "cuda":
            raise ValueError(f"Expected a CUDA workload device, received {requested}")
        torch.cuda.init()
        count = torch.cuda.device_count()
        diagnostics["device_count"] = count
        if count < 1:
            raise RuntimeError("PyTorch CUDA initialized with no visible devices")
        current = torch.cuda.current_device()
        index = current if requested.index is None else requested.index
        diagnostics.update(initial_current_device=current, current_device=current, device_index=index)
        if not 0 <= index < count:
            raise ValueError(f"CUDA device index {index} is outside [0, {count})")
        torch.cuda.set_device(index)
        diagnostics["current_device"] = torch.cuda.current_device()
        if diagnostics["current_device"] != index:
            raise RuntimeError(f"CUDA selected device {diagnostics['current_device']}; requested {index}")
        diagnostics.update(device=f"cuda:{index}", gpu=torch.cuda.get_device_name(index),
                           gpu_memory_bytes=torch.cuda.get_device_properties(index).total_memory)
    except Exception as exc:
        if diagnostics["device_count"] is None:
            try:
                diagnostics["device_count"] = torch.cuda.device_count()
            except Exception as count_error:
                diagnostics["device_count_error"] = f"{type(count_error).__name__}: {count_error}"
        note_diagnostics(exc, "initialize", diagnostics)
        raise
    return CudaStatistics(index, diagnostics)


def check_cuda_memory(expected_gpu: str, diagnostics: dict) -> dict:
    """Exercise the real CUDA allocator without loading an experiment dataset."""
    stats = initialize_cuda(DEFAULT_CUDA_DEVICE, diagnostics)
    if expected_gpu.casefold() not in diagnostics["gpu"].casefold():
        error = RuntimeError(f"Expected GPU {expected_gpu}; detected {diagnostics['gpu']}")
        note_diagnostics(error, "verify device identity", diagnostics)
        raise error

    first = torch.empty((256,), dtype=torch.float32, device=stats.device)
    stats.reset_peak()
    second = torch.empty((512,), dtype=torch.float32, device=stats.device)
    stats.synchronize()
    peak = stats.peak()
    tensor_bytes = [tensor.numel() * tensor.element_size() for tensor in (first, second)]
    if not (peak["peak_cuda_reserved_bytes"] >= peak["peak_cuda_allocated_bytes"] >= sum(tensor_bytes)):
        error = RuntimeError(f"Unexpected CUDA peak statistics for {tensor_bytes} tensor bytes: {peak}")
        note_diagnostics(error, "verify peak memory statistics", diagnostics)
        raise error
    return {"passed": True, "tensor_bytes": tensor_bytes, "cuda": diagnostics, **peak}
