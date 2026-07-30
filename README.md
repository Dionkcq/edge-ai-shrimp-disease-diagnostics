# Edge AI for Sustainable Shrimp Disease Diagnostics

An offline, appearance-based shrimp screening demonstrator for a phone and laptop on the same private LAN or hotspot. The phone captures and displays; a laptop-hosted FastAPI service performs intake checks and, only when a validated artifact is installed, local ONNX inference.

> **Educational screening only.** This software does not confirm a pathogen or disease, determine that an animal is healthy or disease-free, replace laboratory testing, or prescribe medication, antibiotics, or chemical dosing.

## Scope

The demonstrator screens one photograph for two visible appearances:

- dark-gill-like regions;
- white-spot-like regions.

It does not support EMS/AHPND. A clean checkout contains no trained weights or production ONNX artifact, so the service deliberately returns `UNABLE_TO_ASSESS` with `MODEL_UNAVAILABLE`.

The response vocabulary is fixed:

```text
GILL_DARKENING_MARKER_DETECTED
WHITE_SPOT_MARKER_DETECTED
MULTIPLE_TARGET_MARKERS_DETECTED
NO_TARGET_MARKER_DETECTED
UNABLE_TO_ASSESS
```

`NO_TARGET_MARKER_DETECTED` means only that the two target appearances were not retained in that photograph. It is not a health assessment.

## Runtime flow

```text
Phone or laptop selects one JPEG/PNG
→ FastAPI streams and bounds the multipart body
→ image format, dimensions and quality are checked
→ optional local ONNX detector runs on the laptop
→ confidence and abstention policy chooses one stable result
→ the browser shows normalized boxes and cited local guidance
```

The frontend and API share one origin. Runtime assets, policies, guidance and model metadata are local. Uploaded image bytes remain in memory and are discarded after the request.

## Current status

| Area | Status |
|---|---|
| Secure FastAPI contracts and image intake | Implemented and tested |
| Deterministic, fail-closed dataset preparation | Implemented and tested |
| Responsive React interface | Implemented and tested on desktop/mobile Chromium |
| Same-origin production serving | Implemented and integration-tested |
| LikeC4 architecture site | Implemented and locally runnable; current private-repository plan blocks Pages |
| Training pipeline | Not implemented |
| Trained or validated model weights | Not included |
| Accuracy, calibration, latency and parity measurements | Not available |
| Guidance review | Literature-reviewed; not expert-reviewed |

See [`docs/KNOWN_GAPS.md`](docs/KNOWN_GAPS.md) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before interpreting any result.

## Repository layout

```text
backend/       FastAPI service, contracts, intake, policies and inference providers
frontend/      React/Vite interface, unit tests and Playwright tests
pipeline/      AGPL dataset audit, evidence and preparation tooling
contracts/     Generated JSON schemas shared across boundaries
policy/        Versioned quality and decision policies
guidance/      Cited, non-generative educational guidance
architecture/  LikeC4 model and publishable architecture site
scripts/       Repository, licensing and release checks
datasets/      Provenance records and non-accepting mapping template
models/        Empty registry and model-card requirements; no weights
```

Raw archives, processed data, acceptance records, generated artifacts, experiment runs and model weights are ignored and rejected by repository policy.

## Run locally

### Prerequisites

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 24 or newer
- npm

### Install and verify

```bash
uv sync --locked --all-packages --all-groups
uv run pytest
uv run ruff check backend pipeline scripts
uv run ruff format --check backend pipeline scripts
uv run mypy backend/src pipeline/src scripts

cd frontend
npm ci
npm run check
npm run test:e2e
```

### Build the interface and start FastAPI

```bash
cd frontend
npm run build
cd ..
uv run uvicorn shrimp_screening.main:create_default_app \
  --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. To use a phone, bind only on a trusted private LAN/hotspot interface and apply the host firewall policy described in the deployment documentation. Internet access is not required.

The default service is intentionally unavailable for model inference. Installing an arbitrary ONNX file is insufficient: the registry SHA-256, class metadata, tensor shapes and export contract must all validate at startup.

## Dataset preparation

The source-folder class namespaces are independent. The combined-folder interpretation remains provisional and publisher-unconfirmed:

```text
BG/0      → global 0, dark-gill appearance
WSSV/0    → global 1, white-spot appearance
WSSV_BG/0 → global 0, dark-gill appearance (provisional)
WSSV_BG/1 → global 1, white-spot appearance (provisional)
```

Preparation fails closed until a real reviewer inspects at least 60 generated overlays, accepts the provisional semantics and annotation-convention drift, and records the exact evidence-report SHA-256. The checked-in example is deliberately non-accepting. No boxes are fabricated; Healthy images receive empty YOLO label files.

See [`datasets/README.md`](datasets/README.md) and [`datasets/DATASET_REGISTRY.md`](datasets/DATASET_REGISTRY.md).

## Architecture documentation

```bash
cd architecture
npm ci
npm run format:check
npm run validate
npm run build
npm run check:site
```

GitHub Pages would publish only `architecture/dist`, not the screening application, raw data, acceptance records or model artifacts. The current GitHub plan does not support Pages for this private repository, so there is no public LikeC4 URL. Run `npm run dev -- --listen 127.0.0.1 --port 5173` from `architecture/` and open <http://127.0.0.1:5173>. The checked-in workflow becomes usable if the repository plan/settings later support GitHub Actions Pages.

## Team

OIP Group One: Dion, Johnathan, Lambert and Bryan.

Development and review conventions are in [`CONTRIBUTING.md`](CONTRIBUTING.md). Report security issues privately as described in [`SECURITY.md`](SECURITY.md).

## Sources

- [ShrimpDiseaseImageBD Version 3](https://data.mendeley.com/datasets/jhrtdj9txm/3), CC BY 4.0
- [TigerShrimpBD Version 1](https://data.mendeley.com/datasets/9dj4sk5d55/1), CC BY 4.0
- [ShrimpDiseaseImageBD companion paper](https://doi.org/10.1016/j.dib.2025.111553)

## Licensing

This repository is **not uniformly MIT-licensed**. Runtime/backend/frontend and original documentation use the licenses declared for their trees; `pipeline/` is AGPL-3.0-or-later because it is the training/data-tooling boundary. Datasets and publications retain their own licenses and attribution requirements.

Read [`LICENSING.md`](LICENSING.md) for the authoritative per-tree map and dependency boundary.
