from __future__ import annotations

import pytest

from shrimp_training.preflight import PreflightError, TorchProbe, assess_preflight


def _torch(**overrides: object) -> TorchProbe:
    values: dict[str, object] = {
        "torch_version": "2.5.1+cu118",
        "cuda_available": True,
        "cuda_runtime": "11.8",
        "device_count": 1,
        "device_name": "NVIDIA Generic 6GB Test GPU",
        "device_memory_mib": 6144,
    }
    values.update(overrides)
    return TorchProbe(**values)  # type: ignore[arg-type]


def test_assess_preflight_accepts_supported_windows_cuda_environment() -> None:
    report = assess_preflight(
        system="Windows",
        python_version=(3, 11, 9),
        nvidia_smi_csv="NVIDIA Generic 6GB Test GPU, 6144, 530.30\n",
        torch_probe=_torch(),
    )

    assert report.gpu_name == "NVIDIA Generic 6GB Test GPU"
    assert report.driver_version == "530.30"
    assert report.memory_mib == 6144
    assert report.cuda_runtime == "11.8"


@pytest.mark.parametrize(
    ("python_version", "smi", "torch_probe", "message"),
    [
        ((3, 13, 1), "GPU, 6144, 530.30", _torch(), "Python 3.11"),
        ((3, 11, 9), "GPU, 4096, 530.30", _torch(device_memory_mib=4096), "5120 MiB"),
        ((3, 11, 9), "GPU, 6144, 530.30", _torch(cuda_available=False), "CUDA"),
        ((3, 11, 9), "GPU, 6144, 530.30", _torch(cuda_runtime="12.4"), "11.8"),
        ((3, 11, 9), "malformed", _torch(), "nvidia-smi"),
    ],
)
def test_assess_preflight_fails_closed_on_incompatible_environment(
    python_version: tuple[int, int, int],
    smi: str,
    torch_probe: TorchProbe,
    message: str,
) -> None:
    with pytest.raises(PreflightError, match=message):
        assess_preflight(
            system="Windows",
            python_version=python_version,
            nvidia_smi_csv=smi,
            torch_probe=torch_probe,
        )
