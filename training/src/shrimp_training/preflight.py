# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed Windows/NVIDIA preflight for the isolated training environment."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

_REQUIRED_MEMORY_MIB = 5120
_REQUIRED_CUDA_RUNTIME = "11.8"


class PreflightError(RuntimeError):
    """The machine or Python environment is unsafe to start training."""


@dataclass(frozen=True, slots=True)
class TorchProbe:
    torch_version: str
    cuda_available: bool
    cuda_runtime: str | None
    device_count: int
    device_name: str | None
    device_memory_mib: int | None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    system: str
    python_version: str
    gpu_name: str
    memory_mib: int
    driver_version: str
    torch_version: str
    cuda_runtime: str


def _parse_smi(csv_text: str) -> tuple[str, int, str]:
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PreflightError("nvidia-smi must report exactly one selected NVIDIA GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 3 or not fields[0] or not fields[2]:
        raise PreflightError("nvidia-smi returned malformed GPU data")
    try:
        memory_mib = int(fields[1])
    except ValueError as exc:
        raise PreflightError("nvidia-smi returned malformed GPU memory") from exc
    return fields[0], memory_mib, fields[2]


def assess_preflight(
    *,
    system: str,
    python_version: tuple[int, int, int],
    nvidia_smi_csv: str,
    torch_probe: TorchProbe,
) -> PreflightReport:
    """Assess captured environment facts without executing third-party code."""
    if system != "Windows":
        raise PreflightError("this handoff requires Windows")
    if python_version[:2] != (3, 11):
        raise PreflightError("the training environment must use Python 3.11")
    gpu_name, smi_memory, driver = _parse_smi(nvidia_smi_csv)
    if not torch_probe.cuda_available or torch_probe.device_count < 1:
        raise PreflightError("Torch cannot access a CUDA device")
    if torch_probe.cuda_runtime != _REQUIRED_CUDA_RUNTIME:
        raise PreflightError(
            f"Torch must use bundled CUDA {_REQUIRED_CUDA_RUNTIME}, found "
            f"{torch_probe.cuda_runtime!r}"
        )
    if torch_probe.device_memory_mib is None:
        raise PreflightError("Torch did not report CUDA device memory")
    usable_memory = min(smi_memory, torch_probe.device_memory_mib)
    if usable_memory < _REQUIRED_MEMORY_MIB:
        raise PreflightError(f"training requires at least {_REQUIRED_MEMORY_MIB} MiB GPU memory")
    return PreflightReport(
        system=system,
        python_version=".".join(str(value) for value in python_version),
        gpu_name=gpu_name,
        memory_mib=usable_memory,
        driver_version=driver,
        torch_version=torch_probe.torch_version,
        cuda_runtime=_REQUIRED_CUDA_RUNTIME,
    )


def _probe_torch(torch: Any) -> TorchProbe:
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    properties = torch.cuda.get_device_properties(0) if count else None
    return TorchProbe(
        torch_version=str(torch.__version__),
        cuda_available=available,
        cuda_runtime=str(torch.version.cuda) if torch.version.cuda is not None else None,
        device_count=count,
        device_name=str(properties.name) if properties is not None else None,
        device_memory_mib=int(properties.total_memory // (1024 * 1024))
        if properties is not None
        else None,
    )


def run_preflight() -> PreflightReport:
    """Probe the active Windows environment and return a validated report."""
    if platform.system() != "Windows":
        raise PreflightError("this handoff requires Windows")
    windows_root = os.environ.get("WINDIR")
    if windows_root is None or not Path(windows_root).is_absolute():
        raise PreflightError("WINDIR is absent or invalid")
    nvidia_smi = (Path(windows_root) / "System32" / "nvidia-smi.exe").resolve()
    if not nvidia_smi.is_file():
        raise PreflightError(
            f"nvidia-smi is absent from the Windows system directory: {nvidia_smi}"
        )
    try:
        completed = subprocess.run(  # noqa: S603 -- absolute checked Windows system path
            [
                str(nvidia_smi),
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError("nvidia-smi could not query GPU 0") from exc
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise PreflightError("Torch is not installed in the training environment") from exc
    return assess_preflight(
        system=platform.system(),
        python_version=(sys.version_info.major, sys.version_info.minor, sys.version_info.micro),
        nvidia_smi_csv=completed.stdout,
        torch_probe=_probe_torch(torch),
    )
