"""Cross-platform one-command runtime bootstrap for the screening application."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import webbrowser
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LauncherError(RuntimeError):
    """A user-actionable launcher or model-bundle error."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    provider: str
    model_path: Path | None
    registry_path: Path | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherError(f"Could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"{label} must contain a JSON object: {path}")
    return value


def _normalise_entry(raw: dict[str, Any], model_path: Path) -> dict[str, Any]:
    required = {
        "model_id",
        "version",
        "filename",
        "input_size",
        "class_names",
        "opset",
        "output_layout",
        "anchors",
        "dataset_mapping_status",
        "artifact_license",
        "training_toolchain",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise LauncherError(
            "The model manifest is missing " + ", ".join(missing) + ". "
            "Use the official validated model bundle."
        )
    if raw["filename"] != model_path.name:
        raise LauncherError(
            f"Model manifest names {raw['filename']!r}, but the file is {model_path.name!r}."
        )
    digest = _sha256(model_path)
    declared = raw.get("sha256")
    if declared is not None and str(declared).casefold() != digest:
        raise LauncherError(
            "Model SHA-256 does not match the manifest. Re-download the official model bundle."
        )
    entry = dict(raw)
    entry["sha256"] = digest
    return entry


def _write_registry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "models": [entry]}, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_zip_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        member = Path(info.filename)
        if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
            raise LauncherError("The model bundle contains an unsafe archive path.")
        if info.external_attr >> 16 & 0o170000 == 0o120000:
            raise LauncherError("The model bundle contains an unsafe symlink.")


def _install_zip_bundle(bundle: Path, runtime_dir: Path) -> tuple[Path, Path]:
    extracted = runtime_dir / "bundle"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    try:
        with zipfile.ZipFile(bundle) as archive:
            _safe_zip_members(archive)
            names = set(archive.namelist())
            model_member = "model/model.onnx"
            registry_member = "registry-entry.json"
            if {model_member, registry_member} - names:
                raise LauncherError(
                    "The model ZIP must contain model/model.onnx and registry-entry.json."
                )
            archive.extractall(extracted)
    except (OSError, zipfile.BadZipFile) as exc:
        raise LauncherError(f"Could not read model bundle: {bundle}") from exc

    source_model = extracted / "model" / "model.onnx"
    model_path = runtime_dir / "model.onnx"
    shutil.copyfile(source_model, model_path)
    raw_entry = _load_object(extracted / "registry-entry.json", "model registry entry")
    entry = _normalise_entry(raw_entry, model_path)
    registry_path = runtime_dir / "registry.json"
    _write_registry(registry_path, entry)
    return model_path, registry_path


def prepare_runtime(
    root: Path,
    model_dir: Path,
    runtime_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> RuntimeConfig:
    """Discover a model or prepare the safe no-model runtime without editing .env."""
    if not model_dir.is_dir():
        return RuntimeConfig("unavailable", None, None)

    runtime_dir.mkdir(parents=True, exist_ok=True)
    bundles = sorted(model_dir.glob("*.zip"))
    if len(bundles) > 1:
        raise LauncherError("Put exactly one official model ZIP in the model folder.")
    if bundles:
        model_path, registry_path = _install_zip_bundle(bundles[0], runtime_dir)
        return RuntimeConfig("onnx", model_path, registry_path)

    models = sorted(model_dir.glob("*.onnx"))
    if not models:
        return RuntimeConfig("unavailable", None, None)
    if len(models) != 1:
        raise LauncherError("Put exactly one ONNX file in the model folder.")
    manifest_path = model_dir / "model-manifest.json"
    if manifest_path.is_file():
        raw_entry = _load_object(manifest_path, "model manifest")
    elif env is not None:
        raw_entry = _extract_manifest(root, models[0], env)
    else:
        raise LauncherError(
            "An ONNX file was found, but automatic metadata extraction was not requested."
        )
    entry = _normalise_entry(raw_entry, models[0])
    registry_path = runtime_dir / "registry.json"
    _write_registry(registry_path, entry)
    return RuntimeConfig("onnx", models[0], registry_path)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)  # noqa: S603
    except FileNotFoundError as exc:
        executable = command[0]
        raise LauncherError(
            f"{executable!r} was not found. Install the prerequisites listed in README.md."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise LauncherError(
            f"Command failed with exit code {exc.returncode}: {command[0]}"
        ) from exc


def _run_capture(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    print("+", " ".join(command), flush=True)
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise LauncherError(
            f"{command[0]!r} was not found. Install the prerequisites listed in README.md."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()[-1:]
        suffix = f" ({detail[0]})" if detail else ""
        raise LauncherError(
            f"Command failed with exit code {exc.returncode}: {command[0]}{suffix}"
        ) from exc
    return completed.stdout


def _extract_manifest(root: Path, model_path: Path, env: dict[str, str]) -> dict[str, Any]:
    output = _run_capture(
        ["uv", "run", "python", "-m", "scripts.model_metadata", str(model_path)],
        cwd=root,
        env=env,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise LauncherError("ONNX metadata extractor returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LauncherError("ONNX metadata extractor did not return a model entry")
    return value


def _ensure_frontend(root: Path, env: dict[str, str]) -> None:
    frontend = root / "frontend"
    if not (frontend / "node_modules").is_dir():
        _run(["npm", "ci"], cwd=frontend, env=env)
    if not (frontend / "dist" / "index.html").is_file():
        _run(["npm", "run", "build"], cwd=frontend, env=env)


def run_application(root: Path, *, host: str, port: int, rebuild: bool, open_browser: bool) -> int:
    runtime_dir = root / ".runtime"
    model_dir = root / "model"
    env = os.environ.copy()
    env.update(
        {
            "SHRIMP_REPO_ROOT": str(root),
            "SHRIMP_ENV": "dev",
        }
    )
    config = prepare_runtime(root, model_dir, runtime_dir, env=env)
    env["SHRIMP_PROVIDER"] = config.provider
    if config.model_path is not None and config.registry_path is not None:
        env["SHRIMP_ONNX_MODEL_PATH"] = str(config.model_path)
        env["SHRIMP_MODEL_REGISTRY_PATH"] = str(config.registry_path)
    else:
        env.pop("SHRIMP_ONNX_MODEL_PATH", None)
        env.pop("SHRIMP_MODEL_REGISTRY_PATH", None)

    frontend_dist = root / "frontend" / "dist" / "index.html"
    if rebuild or not frontend_dist.is_file():
        _ensure_frontend(root, env)

    command = [
        "uv",
        "run",
        "uvicorn",
        "shrimp_screening.main:create_default_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"Model provider: {config.provider}", flush=True)
    if config.model_path is not None:
        print(f"Model: {config.model_path}", flush=True)
    print(f"Open http://{host}:{port}", flush=True)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        _run(command, cwd=root, env=env)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the frontend before starting"
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        return run_application(
            root,
            host=args.host,
            port=args.port,
            rebuild=args.rebuild,
            open_browser=not args.no_browser,
        )
    except LauncherError as exc:
        print(f"\nStartup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
