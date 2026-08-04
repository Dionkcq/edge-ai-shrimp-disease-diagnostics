# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adapter satisfying the ``TrainableModel``/``ArtifactModel`` protocols already
defined in ``runner.py``/``artifacts.py``.

Ultralytics' own ``YOLO`` class does double duty as both a trainer and an
evaluator/exporter; ``CustomYoloModel`` plays the same role for the from-scratch
architecture, which is what lets the rest of this package's orchestration -- the
CUDA-OOM batch fallback in ``runner.py``, the evaluation/parity/bundle machinery in
``artifacts.py`` -- keep working unmodified against a different trainer.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
import onnx.helper
import onnxruntime as ort
import torch
from torch.amp import GradScaler  # type: ignore[attr-defined]  # torch 2.5.1 stub gap
from torch.utils.data import DataLoader

from shrimp_training.anchors import ANCHORS_PER_SCALE, AnchorSet, anchor_set_to_document
from shrimp_training.data import ShrimpDetectionDataset, collate_fn
from shrimp_training.dataset import validate_prepared_dataset
from shrimp_training.evaluate import evaluate
from shrimp_training.loss import YoloLoss
from shrimp_training.model import STRIDES, YOLO

NUM_CLASSES = 2
CLASS_NAMES = {0: "dark_gill", 1: "white_spot"}


class AdapterError(RuntimeError):
    """The custom model adapter could not complete a train/val/export call."""


def _resolve_device(value: int | str) -> torch.device:
    if isinstance(value, str):
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{value}")
    return torch.device("cpu")


def _require(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs:
        raise AdapterError(f"missing required argument: {key}")
    return kwargs[key]


def _anchors_from_kwargs(kwargs: dict[str, Any], fallback: AnchorSet | None) -> AnchorSet:
    anchors = kwargs.get("anchors", fallback)
    if not isinstance(anchors, AnchorSet):
        raise AdapterError(
            "anchors=<AnchorSet> is required (pass explicitly, or load a checkpoint "
            "that already carries them)"
        )
    return anchors


class _FlattenHead(torch.nn.Module):
    """Wraps the 3 raw per-scale heads into one static ``(1, 5+nc, anchors)`` tensor
    for ONNX export -- the ``CUSTOM_YOLO_ANCHOR_V1`` runtime contract.

    Flattening order is load-bearing: anchor-major, then row-major over the grid,
    scale-major across the 3 scales (matching ``model.YOLO``'s own
    ``(B, na, H, W, no)`` layout and ``STRIDES`` order). The backend's decoder
    builds its anchor grid in this exact same order.
    """

    def __init__(self, model: YOLO) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scales = self.model(x)
        flattened = []
        for scale in scales:
            batch, num_anchors, height, width, channels = scale.shape
            flattened.append(scale.reshape(batch, num_anchors * height * width, channels))
        merged = torch.cat(flattened, dim=1)  # (B, total_anchors, 5+nc)
        return merged.permute(0, 2, 1)  # (B, 5+nc, total_anchors)


class _OnnxEvalModel:
    """Callable wrapper making an ONNX session look like an ``nn.Module`` forward
    for ``evaluate.evaluate``: input a batch, get back the 3-scale output list."""

    def __init__(self, session: ort.InferenceSession, layout: list[tuple[int, int, int]]) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._layout = layout

    def eval(self) -> _OnnxEvalModel:
        return self

    def __call__(self, images: torch.Tensor) -> list[torch.Tensor]:
        batch = images.detach().cpu().numpy().astype(np.float32)
        raw = self._session.run(None, {self._input_name: batch})[0]
        raw_tensor = torch.from_numpy(np.asarray(raw)).permute(0, 2, 1)  # (B, anchors, no)
        scales: list[torch.Tensor] = []
        offset = 0
        for num_anchors, height, width in self._layout:
            count = num_anchors * height * width
            chunk = raw_tensor[:, offset : offset + count, :]
            scales.append(chunk.reshape(raw_tensor.shape[0], num_anchors, height, width, -1))
            offset += count
        return scales


class CustomYoloModel:
    """Constructed fresh, from a ``.pt`` checkpoint, or from a ``.onnx`` artifact."""

    def __init__(self, weights: Path | None = None) -> None:
        self._model: YOLO | None = None
        self._onnx_session: ort.InferenceSession | None = None
        self._checkpoint_anchors: AnchorSet | None = None

        if weights is None:
            self._model = YOLO(num_classes=NUM_CLASSES, num_anchors=ANCHORS_PER_SCALE)
        elif weights.suffix.casefold() == ".onnx":
            self._onnx_session = ort.InferenceSession(
                str(weights), providers=["CPUExecutionProvider"]
            )
        else:
            try:
                checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise AdapterError(f"cannot load checkpoint: {weights}") from exc
            model = YOLO(
                num_classes=int(checkpoint.get("num_classes", NUM_CLASSES)),
                num_anchors=int(checkpoint.get("num_anchors", ANCHORS_PER_SCALE)),
            )
            model.load_state_dict(checkpoint["state_dict"])
            self._model = model
            anchor_document = checkpoint["anchors"]
            self._checkpoint_anchors = AnchorSet(
                boxes=tuple(tuple(pair) for pair in anchor_document["boxes"]),
                strides=tuple(anchor_document["strides"]),
                seed=int(anchor_document["seed"]),
                source_box_count=int(anchor_document["source_box_count"]),
            )

    def _save_checkpoint(self, model: YOLO, anchors: AnchorSet, path: Path) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": model.nc,
                "num_anchors": model.na,
                "anchors": anchor_set_to_document(anchors),
            },
            path,
        )

    def train(self, **kwargs: Any) -> Any:
        if self._model is None:
            raise AdapterError("train() requires a fresh or checkpoint-loaded torch model")
        data_root = Path(_require(kwargs, "data"))
        image_size = int(_require(kwargs, "imgsz"))
        epochs = int(_require(kwargs, "epochs"))
        patience = int(_require(kwargs, "patience"))
        batch_size = int(_require(kwargs, "batch"))
        workers = int(_require(kwargs, "workers"))
        amp = bool(_require(kwargs, "amp"))
        deterministic = bool(_require(kwargs, "deterministic"))
        seed = int(_require(kwargs, "seed"))
        learning_rate = float(_require(kwargs, "learning_rate"))
        weight_decay = float(_require(kwargs, "weight_decay"))
        project = Path(_require(kwargs, "project"))
        name = str(_require(kwargs, "name"))
        device = _resolve_device(kwargs.get("device", 0))
        anchors = _anchors_from_kwargs(kwargs, self._checkpoint_anchors)

        if deterministic:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        prepared = validate_prepared_dataset(data_root)
        train_dataset = ShrimpDetectionDataset(
            prepared, "train", image_size=image_size, augment=True, seed=seed
        )
        validation_dataset = ShrimpDetectionDataset(
            prepared, "validation", image_size=image_size, augment=False, seed=seed
        )
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
        )

        model = self._model.to(device)
        loss_fn = YoloLoss(anchors, NUM_CLASSES).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        use_amp = amp and device.type == "cuda"
        scaler = GradScaler(device.type, enabled=use_amp)

        run_dir = project / name
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=False)

        # A fresh random-init network has a sharp early loss surface: the base
        # learning rate applied from step 1 reliably diverges (observed directly:
        # loss went 1918 -> 40613 -> 1955406 across the first 3 real batches
        # without warmup). Linear LR warmup over the first few hundred iterations
        # is the standard fix and is applied before the per-epoch cosine schedule
        # takes over. Gradient-norm clipping is also tightened for the same reason.
        warmup_iterations = min(500, len(loader) * epochs)
        global_step = 0
        best_metric = -1.0
        best_epoch = -1
        for epoch in range(epochs):
            model.train()
            for batch_images, batch_targets in loader:
                if warmup_iterations and global_step < warmup_iterations:
                    warmup_lr = learning_rate * (global_step + 1) / warmup_iterations
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr
                images = batch_images.to(device)
                targets = batch_targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    losses = loss_fn(outputs, targets)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    raise AdapterError(f"non-finite training loss at epoch {epoch}")
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                global_step += 1
            scheduler.step()

            result = evaluate(
                model, validation_dataset, anchors, num_classes=NUM_CLASSES, device=device
            )
            metric = float(result.results_dict["metrics/mAP50-95(B)"])
            self._save_checkpoint(model, anchors, weights_dir / "last.pt")
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                self._save_checkpoint(model, anchors, weights_dir / "best.pt")
            elif patience and epoch - best_epoch >= patience:
                break

        if not (weights_dir / "best.pt").is_file():
            self._save_checkpoint(model, anchors, weights_dir / "best.pt")
        return SimpleNamespace(save_dir=run_dir)

    def val(self, **kwargs: Any) -> Any:
        data_root = Path(_require(kwargs, "data"))
        split = str(kwargs.get("split", "validation"))
        partition = "test" if split == "test" else "validation"
        image_size = int(kwargs.get("imgsz", 640))
        device = _resolve_device(kwargs.get("device", "cpu" if self._onnx_session else 0))
        anchors = _anchors_from_kwargs(kwargs, self._checkpoint_anchors)

        prepared = validate_prepared_dataset(data_root)
        dataset = ShrimpDetectionDataset(prepared, partition, image_size=image_size, augment=False)

        if self._onnx_session is not None:
            layout = [
                (ANCHORS_PER_SCALE, image_size // stride, image_size // stride)
                for stride in STRIDES
            ]
            model: Any = _OnnxEvalModel(self._onnx_session, layout)
        else:
            if self._model is None:
                raise AdapterError("no model is available to evaluate")
            model = self._model.to(device)
        return evaluate(model, dataset, anchors, num_classes=NUM_CLASSES, device=device)

    def export(self, **kwargs: Any) -> Path:
        if self._model is None:
            raise AdapterError("export requires a torch checkpoint, not an ONNX artifact")
        image_size = int(_require(kwargs, "imgsz"))
        opset = int(kwargs.get("opset", 17))
        destination = Path(_require(kwargs, "destination"))
        if destination.exists():
            raise AdapterError(f"refusing to overwrite existing export target: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        wrapper = _FlattenHead(self._model).eval()
        dummy = torch.zeros(1, 3, image_size, image_size)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(temporary),
            input_names=["images"],
            output_names=["output"],
            opset_version=opset,
            dynamic_axes=None,
            do_constant_folding=True,
        )
        onnx_model = onnx.load(str(temporary))
        onnx.helper.set_model_props(
            onnx_model,
            {"names": repr(CLASS_NAMES), "task": "detect", "imgsz": repr(image_size)},
        )
        onnx.save(onnx_model, str(temporary))
        temporary.replace(destination)
        return destination
