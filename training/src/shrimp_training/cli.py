# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line orchestration for preflight, train, evaluate, export, and bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shrimp_training.acceptance import validate_mapping_acceptance
from shrimp_training.artifacts import (
    compare_parity,
    evaluate_artifact,
    export_static_onnx,
    validate_onnx_contract,
)
from shrimp_training.bundle import (
    build_return_bundle,
    bundle_manifest_sha256,
    verify_return_bundle,
)
from shrimp_training.config import load_profile
from shrimp_training.dataset import validate_prepared_dataset, write_ultralytics_dataset
from shrimp_training.preflight import run_preflight
from shrimp_training.runner import train_with_fallback

BASE_WEIGHTS_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
_TOOLCHAIN_PACKAGES = ("ultralytics", "torch", "torchvision", "onnx", "onnxruntime", "numpy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _environment_record(
    base_weights: Path,
    prepared_manifest_sha256: str,
    dataset_inventory_sha256: str,
    training_lock: Path,
) -> dict[str, Any]:
    packages = {name: importlib.metadata.version(name) for name in _TOOLCHAIN_PACKAGES}
    return {
        "schema_version": "1.0.0",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "base_weights": {
            "filename": base_weights.name,
            "sha256": _sha256(base_weights),
        },
        "training_lock_sha256": _sha256(training_lock),
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "dataset_inventory_sha256": dataset_inventory_sha256,
    }


def _run_preflight_command(args: argparse.Namespace) -> int:
    report = run_preflight()
    _write_json(args.output, {"schema_version": "1.0.0", **asdict(report)})
    print(json.dumps({"status": "READY", "report": str(args.output)}))
    return 0


def _run_all(args: argparse.Namespace) -> int:
    if _sha256(args.initial_weights) != BASE_WEIGHTS_SHA256:
        raise RuntimeError(
            "initial yolo11n.pt SHA-256 does not match the pinned v8.4.0 base artifact"
        )
    prepared = validate_prepared_dataset(args.dataset_root)
    acceptance = validate_mapping_acceptance(args.mapping_acceptance, prepared.manifest)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    records = args.work_dir / "records"
    preflight_path = _write_json(
        records / "preflight.json",
        {"schema_version": "1.0.0", **asdict(run_preflight())},
    )
    profile = load_profile(args.profile)
    dataset_yaml = write_ultralytics_dataset(prepared, args.work_dir / "dataset.yaml")
    training = train_with_fallback(
        profile,
        args.initial_weights,
        dataset_yaml,
        args.work_dir / "runs",
    )
    pytorch_evaluation = records / "evaluation-pytorch.json"
    evaluate_artifact(
        training.best_checkpoint,
        dataset_yaml,
        profile,
        pytorch_evaluation,
        prepared_root=prepared.root,
    )
    model_onnx = export_static_onnx(
        training.best_checkpoint,
        profile,
        args.work_dir / "model.onnx",
    )
    validate_onnx_contract(model_onnx, profile.image_size)
    onnx_evaluation = records / "evaluation-onnx.json"
    evaluate_artifact(
        model_onnx,
        dataset_yaml,
        profile,
        onnx_evaluation,
        prepared_root=prepared.root,
    )
    parity = records / "parity.json"
    compare_parity(
        pytorch_evaluation,
        onnx_evaluation,
        parity,
        tolerance=args.parity_tolerance,
    )
    environment = _write_json(
        records / "environment.json",
        _environment_record(
            args.initial_weights,
            prepared.manifest_sha256,
            prepared.inventory_sha256,
            Path(__file__).resolve().parents[2] / "uv.lock",
        ),
    )
    toolchain = (
        f"ultralytics {importlib.metadata.version('ultralytics')} / "
        f"torch {importlib.metadata.version('torch')}"
    )
    bundle = build_return_bundle(
        args.bundle,
        {
            "model/model.onnx": model_onnx,
            "records/evaluation-pytorch.json": pytorch_evaluation,
            "records/evaluation-onnx.json": onnx_evaluation,
            "records/parity.json": parity,
            "records/prepared-manifest.json": prepared.manifest,
            "records/mapping-acceptance.json": args.mapping_acceptance,
            "records/evidence-report.json": acceptance.evidence_report,
            "records/training-profile.json": args.profile,
            "records/preflight.json": preflight_path,
            "records/environment.json": environment,
        },
        model_id=args.model_id,
        version=args.version,
        input_size=profile.image_size,
        toolchain=toolchain,
    )
    manifest_digest = bundle_manifest_sha256(bundle)
    verify_return_bundle(bundle, expected_manifest_sha256=manifest_digest)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "bundle": str(bundle),
                "bundle_manifest_sha256": manifest_digest,
                "best_checkpoint": str(training.best_checkpoint),
                "batch_size": training.batch_size,
            }
        )
    )
    return 0


def _verify_bundle_command(args: argparse.Namespace) -> int:
    manifest = verify_return_bundle(
        args.bundle,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps({"status": "VERIFIED", "manifest": manifest}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shrimp-train")
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="verify Windows, NVIDIA, and Torch")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(handler=_run_preflight_command)

    run_all = commands.add_parser("run-all", help="train, evaluate, export, and bundle")
    run_all.add_argument("--dataset-root", type=Path, required=True)
    run_all.add_argument("--mapping-acceptance", type=Path, required=True)
    run_all.add_argument("--initial-weights", type=Path, required=True)
    run_all.add_argument("--profile", type=Path, required=True)
    run_all.add_argument("--work-dir", type=Path, required=True)
    run_all.add_argument("--bundle", type=Path, required=True)
    run_all.add_argument("--model-id", default="shrimp-marker-yolo11n")
    run_all.add_argument("--version", required=True)
    run_all.add_argument("--parity-tolerance", type=float, default=0.01)
    run_all.set_defaults(handler=_run_all)

    verify = commands.add_parser("verify-bundle", help="verify a returned model bundle")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--expected-manifest-sha256", required=True)
    verify.set_defaults(handler=_verify_bundle_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
