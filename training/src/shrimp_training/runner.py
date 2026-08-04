# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ultralytics training boundary with deterministic CUDA OOM fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from shrimp_training.config import TrainingProfile


class TrainingError(RuntimeError):
    """Training could not produce a trustworthy best checkpoint."""


class TrainResult(Protocol):
    save_dir: Path | str


class TrainableModel(Protocol):
    def train(self, **kwargs: Any) -> TrainResult: ...


ModelFactory = Callable[[Path], TrainableModel]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    best_checkpoint: Path
    batch_size: int


def _default_factory(weights: Path) -> TrainableModel:
    module = import_module("ultralytics")
    constructor = getattr(module, "YOLO", None)
    if constructor is None:
        raise TrainingError("the installed ultralytics package exposes no YOLO constructor")
    return cast(TrainableModel, constructor(str(weights)))


def _is_cuda_oom(exc: RuntimeError) -> bool:
    lowered = str(exc).casefold()
    return "cuda" in lowered and "out of memory" in lowered


def train_with_fallback(
    profile: TrainingProfile,
    initial_weights: Path,
    dataset_yaml: Path,
    output: Path,
    *,
    model_factory: ModelFactory = _default_factory,
) -> TrainingResult:
    """Train from clean weights, reducing batch size only after a CUDA OOM."""
    if not initial_weights.is_file():
        raise TrainingError(f"initial weights do not exist: {initial_weights}")
    if not dataset_yaml.is_file():
        raise TrainingError(f"dataset descriptor does not exist: {dataset_yaml}")
    output.mkdir(parents=True, exist_ok=True)
    last_oom: RuntimeError | None = None
    for batch_size in profile.batch_fallback:
        name = f"{profile.profile_name}-batch-{batch_size}"
        model = model_factory(initial_weights)
        try:
            result = model.train(
                data=str(dataset_yaml),
                imgsz=profile.image_size,
                epochs=profile.epochs,
                patience=profile.patience,
                batch=batch_size,
                workers=profile.workers,
                device=profile.device,
                amp=profile.amp,
                deterministic=profile.deterministic,
                seed=profile.seed,
                cache=profile.cache,
                project=str(output),
                name=name,
                exist_ok=False,
                pretrained=True,
                val=True,
                plots=True,
                close_mosaic=10,
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            last_oom = exc
            continue
        run_dir = Path(result.save_dir).resolve()
        best_checkpoint = run_dir / "weights" / "best.pt"
        if not best_checkpoint.is_file():
            raise TrainingError(f"training returned without a best checkpoint: {best_checkpoint}")
        return TrainingResult(
            run_dir=run_dir,
            best_checkpoint=best_checkpoint,
            batch_size=batch_size,
        )
    raise TrainingError(
        f"CUDA ran out of memory at every configured batch size {profile.batch_fallback}"
    ) from last_oom
