from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shrimp_training.config import ProfileError, load_profile
from shrimp_training.runner import TrainingError, train_with_fallback


def _profile(path: Path, **overrides: object) -> Path:
    document: dict[str, object] = {
        "schema_version": "1.0.0",
        "profile_name": "compact-nvidia-6gb",
        "image_size": 640,
        "epochs": 100,
        "patience": 20,
        "batch_fallback": [4, 2, 1],
        "workers": 4,
        "device": 0,
        "amp": True,
        "deterministic": True,
        "seed": 20260730,
        "cache": False,
    }
    document.update(overrides)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_load_profile_accepts_generic_6gb_settings_and_rejects_unknowns(tmp_path: Path) -> None:
    profile = load_profile(_profile(tmp_path / "profile.json"))

    assert profile.image_size == 640
    assert profile.batch_fallback == (4, 2, 1)
    assert profile.seed == 20260730

    with pytest.raises(ProfileError):
        load_profile(_profile(tmp_path / "unknown.json", personal_gpu="private"))
    with pytest.raises(ProfileError):
        load_profile(_profile(tmp_path / "bad-batch.json", batch_fallback=[4, 4, 0]))


class _FakeResult:
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir


class _FakeModel:
    def __init__(self, attempts: list[dict[str, Any]], output: Path, *, unrelated: bool) -> None:
        self._attempts = attempts
        self._output = output
        self._unrelated = unrelated

    def train(self, **kwargs: Any) -> _FakeResult:
        self._attempts.append(kwargs)
        if self._unrelated:
            raise RuntimeError("dataset parser failed")
        if kwargs["batch"] == 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate 64 MiB")
        save_dir = self._output / str(kwargs["name"])
        weights = save_dir / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"checkpoint")
        return _FakeResult(save_dir)


def test_train_retries_only_cuda_oom_and_returns_existing_best_checkpoint(tmp_path: Path) -> None:
    profile = load_profile(_profile(tmp_path / "profile.json"))
    weights = tmp_path / "initial.pt"
    weights.write_bytes(b"initial")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}\n", encoding="utf-8")
    output = tmp_path / "runs"
    attempts: list[dict[str, Any]] = []

    result = train_with_fallback(
        profile,
        weights,
        dataset,
        output,
        model_factory=lambda _: _FakeModel(attempts, output, unrelated=False),
    )

    assert [attempt["batch"] for attempt in attempts] == [4, 2]
    assert result.batch_size == 2
    assert result.best_checkpoint.read_bytes() == b"checkpoint"
    assert all(attempt["imgsz"] == 640 for attempt in attempts)
    assert all(attempt["deterministic"] is True for attempt in attempts)
    assert all(attempt["pretrained"] is True for attempt in attempts)


def test_train_does_not_hide_non_oom_errors(tmp_path: Path) -> None:
    profile = load_profile(_profile(tmp_path / "profile.json"))
    weights = tmp_path / "initial.pt"
    weights.write_bytes(b"initial")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dataset parser failed"):
        train_with_fallback(
            profile,
            weights,
            dataset,
            tmp_path / "runs",
            model_factory=lambda _: _FakeModel([], tmp_path / "runs", unrelated=True),
        )

    weights.unlink()
    with pytest.raises(TrainingError, match="initial weights"):
        train_with_fallback(
            profile,
            weights,
            dataset,
            tmp_path / "runs-2",
            model_factory=lambda _: _FakeModel([], tmp_path / "runs-2", unrelated=False),
        )
