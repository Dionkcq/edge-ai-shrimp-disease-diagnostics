# SPDX-License-Identifier: AGPL-3.0-or-later
"""From-scratch-model training boundary with deterministic CUDA OOM fallback.

Training starts from a fresh, seeded random init -- ``model.YOLO`` has no
pretrained-weights loading path, so there is no external checkpoint to pin or
verify here (contrast with the pinned Ultralytics base checkpoint this module used
to require).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from shrimp_training.adapter import CustomYoloModel
from shrimp_training.anchors import AnchorSet
from shrimp_training.config import TrainingProfile


class TrainingError(RuntimeError):
    """Training could not produce a trustworthy best checkpoint."""


class TrainResult(Protocol):
    save_dir: Path | str


class TrainableModel(Protocol):
    def train(self, **kwargs: Any) -> TrainResult: ...


ModelFactory = Callable[[], TrainableModel]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    best_checkpoint: Path
    batch_size: int


def _default_factory() -> TrainableModel:
    return cast(TrainableModel, CustomYoloModel())


def _is_cuda_oom(exc: RuntimeError) -> bool:
    lowered = str(exc).casefold()
    return "cuda" in lowered and "out of memory" in lowered


def train_with_fallback(
    profile: TrainingProfile,
    dataset_root: Path,
    output: Path,
    *,
    anchors: AnchorSet,
    model_factory: ModelFactory = _default_factory,
) -> TrainingResult:
    """Train from a fresh random init, reducing batch size only after a CUDA OOM."""
    if not dataset_root.is_dir():
        raise TrainingError(f"prepared dataset root does not exist: {dataset_root}")
    output.mkdir(parents=True, exist_ok=True)
    last_oom: RuntimeError | None = None
    for batch_size in profile.batch_fallback:
        name = f"{profile.profile_name}-batch-{batch_size}"
        model = model_factory()
        try:
            result = model.train(
                data=str(dataset_root),
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
                learning_rate=profile.learning_rate,
                weight_decay=profile.weight_decay,
                anchors=anchors,
                project=str(output),
                name=name,
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
