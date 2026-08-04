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
| Isolated training/export workflow | Implemented and orchestration-tested; no real run completed |
| Trained or validated model weights | Not included |
| Accuracy, calibration, latency and parity measurements | Not available |
| Guidance review | Literature-reviewed; not expert-reviewed |

The declared limitations of any result are served with it, in the `limitations[]` array
of every response. They are defined in
[`backend/src/shrimp_screening/limitations.py`](backend/src/shrimp_screening/limitations.py).

## Repository layout

```text
backend/src/shrimp_server/     HTTP surface: routers, middleware, problem rendering
backend/src/shrimp_screening/  Domain: detection, intake, policy, guidance, LLM advice
backend/tests/                 Unit, integration and security tests
frontend/                      React/Vite interface, unit tests and Playwright tests
data/                          Every JSON the runtime and the pipeline read
model/                         Everything that produces the detector (AGPL, never runtime)
  pipeline/                    Dataset audit, evidence and fail-closed preparation
  training/                    Separately locked Python 3.11 trainer and ONNX export
```

### `data/`

One folder, flat, holding everything that is data rather than code:

```text
quality_policy_v1.json         Capture-quality thresholds     -> loaded at startup
decision_policy_v1.json        Score/abstention thresholds    -> loaded at startup
guidance_v1.json               Cited educational guidance     -> loaded at startup
model_registry.json            SHA-256 allowlist of models    -> loaded at startup
dataset_manifest.json          Source-dataset provenance      -> pipeline
mapping_acceptance.example.json  Non-accepting gate template  -> pipeline
raw/                           Source archives (gitignored, never committed)
```

The first three are the **root markers**: `repository_root()` locates the repository by
finding `data/quality_policy_v1.json`, `data/guidance_v1.json` and
`data/model_registry.json` together, and fails loudly rather than falling back to the
working directory. Renaming any of them means updating
[`paths.py`](backend/src/shrimp_screening/paths.py).

`shrimp_server` imports `shrimp_screening`, never the reverse — so the domain stays
usable from a script or a notebook without pulling in an HTTP stack. The direction is
enforced by `backend/tests/security/test_dependency_boundary.py`.

Raw archives, processed data, acceptance records, generated artifacts, experiment runs and model weights are gitignored and never committed.

### Two Python environments, deliberately

`.venv` is Python 3.13 and holds the runtime; `model/training/.venv` is Python 3.11 and
holds PyTorch. They cannot be merged, for two independent reasons: the interpreters differ,
and `model/training/` is excluded from the uv workspace so that AGPL PyTorch/Ultralytics
never enter the served application's resolved dependency graph.

That boundary is asserted by
[`backend/tests/security/test_dependency_boundary.py`](backend/tests/security/test_dependency_boundary.py),
which checks declared dependencies, the resolved `uv.lock`, static imports, and the real
import graph of a booted app.

`model/pipeline/` is different again: it is Python 3.13 and *is* a workspace member, so it
installs into the root `.venv` alongside the backend. It sits under `model/` because it is
part of producing the detector, not because it shares the trainer's environment — which is
why `model/` is a plain container directory rather than a project of its own.

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
uv run ruff check backend model
uv run ruff format --check backend model
uv run mypy

cd frontend
npm ci
npm run check
npm run test:e2e
```

### Configure

Optional. Every setting has a safe default, so running with no configuration at all is
a supported state — it is the fail-closed one.

To change something, create a `.env` in the repository root (it is gitignored) or export
the variable. Every field is prefixed `SHRIMP_` and each one is documented, with the
reasoning for its default, in
[`backend/src/shrimp_screening/settings.py`](backend/src/shrimp_screening/settings.py):

```bash
SHRIMP_PROVIDER=fixture        # unavailable (default) | fixture | onnx
SHRIMP_ONNX_MODEL_PATH=...     # only loaded if its sha256 is in data/model_registry.json
SHRIMP_LLM_ENABLED=true        # opt in to local Ollama advice; off by default
```

### Build the interface and start FastAPI

```bash
cd frontend
npm run build
cd ..
uv run uvicorn shrimp_server.main:create_default_app \
  --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. To use a phone, bind only on a trusted private LAN/hotspot interface and apply the host firewall policy described in the deployment documentation. Internet access is not required.

The default service is intentionally unavailable for model inference. Installing an arbitrary ONNX file is insufficient: the registry SHA-256, class metadata, tensor shapes and export contract must all validate at startup.

## Optional: local generated advice

`GET /api/v1/advice/{decision}` expands the cited guidance above into a longer,
farmer-facing explanation and action plan, using a local
[Ollama](https://ollama.com/) model. It is entirely optional, off by default, and
never consulted for the screening decision itself:

- Disabled unless `SHRIMP_LLM_ENABLED=true`, in which case the endpoint answers
  `404` exactly as if it did not exist.
- Talks only to a local Ollama server (`SHRIMP_LLM_BASE_URL`, default
  `http://127.0.0.1:11434`) running `SHRIMP_LLM_MODEL` (default
  `qwen2.5:7b-instruct-q4_0`). No photograph, decision or pond record ever
  leaves the machine.
- Every response is grounded in the same cited guidance text as
  `GET /api/v1/guidance/{decision}` and is labelled `AI_GENERATED_NOT_REVIEWED`.
  It is scanned with the same lexicon the guidance corpus itself must pass
  (`backend/src/shrimp_screening/guidance/lexicon.py`) before it can reach a
  response: a claim of health, a diagnosis, or a named medication/dose fails
  closed as `503 ADVICE_UNAVAILABLE` rather than reaching the client.
- If Ollama is not running, unreachable, or the model is not pulled, the
  endpoint answers `503 ADVICE_UNAVAILABLE` with `Retry-After`, the same way the
  rest of this API fails explicitly rather than degrading silently.

`GET /api/v1/meta` reports `advice_available`, so a client knows whether the
feature exists on this build instead of discovering it by asking and failing.

In the interface, advice appears as an opt-in panel below the reviewed guidance,
never in place of it:

- The panel is rendered only when `advice_available` is true, and nothing is
  requested until a person presses **Generate explanation**. Generation runs on a
  local model and takes seconds to tens of seconds, so it is never a side effect
  of screening a photograph.
- The `AI GENERATED · NOT REVIEWED` banner and `review_note` precede the
  generated text in document order. The client validates `review_status`,
  `review_note` and `provider` as strictly as the content itself and refuses a
  body that arrives without them, so there is no state in which unreviewed text
  renders undisclosed.
- Generated text belongs to exactly one screening result and is dropped when a
  new photograph is chosen or submitted.
- A failed generation is confined to this panel and reported there; the cited
  guidance above it is unaffected.

To use it locally: install Ollama, `ollama pull qwen2.5:7b-instruct-q4_0`, then
set `SHRIMP_LLM_ENABLED=true` before starting the FastAPI service.

## Dataset preparation

The source-folder class namespaces are independent. The combined-folder interpretation remains provisional and publisher-unconfirmed:

```text
BG/0      → global 0, dark-gill appearance
WSSV/0    → global 1, white-spot appearance
WSSV_BG/0 → global 0, dark-gill appearance (provisional)
WSSV_BG/1 → global 1, white-spot appearance (provisional)
```

Preparation fails closed until a real reviewer inspects at least 60 generated overlays, accepts the provisional semantics and annotation-convention drift, and records the exact evidence-report SHA-256. The checked-in example is deliberately non-accepting. No boxes are fabricated; Healthy images receive empty YOLO label files.

## Model training and export

Training is isolated from the Python 3.13 runtime workspace in a separately locked
Python 3.11 AGPL project. It validates the prepared dataset, trains with a generic
6 GB profile and CUDA-memory fallback, evaluates the locked test split, exports a
static ONNX graph, checks PyTorch/ONNX parity, and creates a checksummed private
return bundle. No weights or generated training records are committed.

## Team

OIP Group One: Dion, Johnathan, Lambert and Bryan.

## Sources

- [ShrimpDiseaseImageBD Version 3](https://data.mendeley.com/datasets/jhrtdj9txm/3), CC BY 4.0
- [TigerShrimpBD Version 1](https://data.mendeley.com/datasets/9dj4sk5d55/1), CC BY 4.0
- [ShrimpDiseaseImageBD companion paper](https://doi.org/10.1016/j.dib.2025.111553)

## Licensing

This repository carries **no root licence file**. Note that `backend/pyproject.toml` still
declares `license = "MIT"` in its package metadata, so the two disagree about the terms —
resolve that deliberately, in whichever direction you intend.

`model/pipeline/` and `model/training/` are the exception and remain **AGPL-3.0-or-later**: they link
the Ultralytics/PyTorch toolchain, so that licence is not optional. Each keeps its own
`LICENSE.AGPL`, and the AGPL requires that text to travel with the code — do not delete
those two files. The boundary is enforced by the backend security tests, which assert that
nothing AGPL reaches the served application.

Datasets and publications retain their own licences and attribution requirements.
