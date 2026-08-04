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
| LikeC4 architecture site | Deployed and verified on GitHub Pages |
| Isolated model/training/export workflow | Implemented and orchestration-tested; no real run completed |
| Trained or validated model weights | Not included |
| Accuracy, calibration, latency and parity measurements | Not available |
| Guidance review | Literature-reviewed; not expert-reviewed |

See [`docs/KNOWN_GAPS.md`](docs/KNOWN_GAPS.md) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before interpreting any result.

## Repository layout

```text
backend/       FastAPI service, contracts, intake, policies and inference providers
frontend/      React/Vite interface, unit tests and Playwright tests
model/pipeline/      AGPL dataset audit, evidence and preparation tooling
model/training/      Separately locked Python 3.11 AGPL model/training/export tooling
contracts/     Generated JSON schemas shared across boundaries
policy/        Versioned quality and decision policies
guidance/      Cited, non-generative educational guidance
architecture/  LikeC4 model and publishable architecture site
scripts/       Repository, licensing and release checks
datasets/      Provenance records and non-accepting mapping template
models/        Empty registry and model-card requirements; no weights
```

Raw archives, processed data, acceptance records, generated artifacts, experiment runs and model weights are ignored and rejected by repository policy.

## Run the application with one command

### Prerequisites

Install these once on the laptop that will run the application:

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 24 or newer
- npm

### Start

From the repository root, run:

```bash
python run.py
```

On Windows, either `py run.py` or `python run.py` is supported. The launcher:

1. installs the locked Python and frontend dependencies when they are missing;
2. builds the frontend when `frontend/dist/index.html` is absent;
3. searches `model/` for one ONNX model or official model ZIP;
4. reads embedded ONNX metadata when no sidecar file is present;
5. extracts and verifies the model SHA-256 and output contract automatically;
6. generates ignored runtime state under `.runtime/`;
7. starts FastAPI and serves the built React interface from the same origin;
8. opens `http://127.0.0.1:8000` in a browser.

A clean checkout has no model, so the launcher starts in the safe `unavailable`
state. It displays the application, but screening returns `UNABLE_TO_ASSESS /
MODEL_UNAVAILABLE`. To use a trained model, put one validated bundle in `model/`
and run the same command again:

```text
model/
└── shrimp-model-v1.zip
    ├── model/model.onnx
    └── registry-entry.json
```

Do not edit `.env`, calculate a hash, or edit `models/registry.json` for an
ONNX file that contains the required metadata. The launcher performs those
steps in `.runtime/`, which is ignored by Git. If the model has a custom
anchor-based output, its anchors must be embedded in the ONNX metadata; models
without a supported self-describing output contract are rejected.

Useful options:

```bash
python run.py --rebuild       # force a frontend rebuild
python run.py --no-browser    # start without opening a browser
python run.py --host 0.0.0.0  # trusted private LAN/hotspot only
```

Open `http://127.0.0.1:8000`. To use a phone, bind only on a trusted private
LAN/hotspot interface and apply the host firewall policy described in the
deployment documentation. Internet access is not required.

The launcher keeps the existing fail-closed model contract: a malformed,
renamed, unregistered or incompatible model stops startup with an actionable
error instead of silently producing detections.

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

In the interface, advice appears as an opt-in panel below the cited local guidance,
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

### Pond-side chat grounding

`POST /api/v1/chat` may use the local Qwen model to request the screening tool; it
does not make detector output or guidance authoritative. An attached image is
screened before the assistant may answer about it, even when the model omits the
tool call. The tool result includes the matching cited local guidance item, its
review status and source titles.

Every image-result reply is then constructed deterministically from that tool
payload. There is no second free-form model pass that can replace the detector
result, invent pond actions or drift into human healthcare. When there is no screening
result, chat always fails closed to an upload prompt instead of returning model prose.
Escalation is always framed as a qualified aquatic-animal health professional. The
LLM does not author a user-facing pond-side answer, decide a screening result or
prescribe treatment.

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

## Model training and export

Training is isolated from the Python 3.13 runtime workspace in a separately locked
Python 3.11 AGPL project. It validates the prepared dataset, trains with a generic
6 GB profile and CUDA-memory fallback, evaluates the locked test split, exports a
static ONNX graph, checks PyTorch/ONNX parity, and creates a checksummed private
return bundle. No weights or generated training records are committed.

See [`model/training/README.md`](model/training/README.md) for the generic reproducible workflow.

## Architecture documentation

```bash
cd architecture
npm ci
npm run format:check
npm run validate
npm run build
npm run check:site
```

GitHub Pages publishes only the validated `architecture/dist` artifact—not the screening application, raw data, acceptance records or model artifacts. Open the verified site at <https://dionkcq.github.io/edge-ai-shrimp-disease-diagnostics/>. For local development, run `npm run dev -- --listen 127.0.0.1 --port 5173` from `architecture/` and open <http://127.0.0.1:5173>.

## Team

OIP Group One: Dion, Johnathan, Lambert and Bryan.

Development and review conventions are in [`CONTRIBUTING.md`](CONTRIBUTING.md). Report security issues privately as described in [`SECURITY.md`](SECURITY.md).

## Sources

- [ShrimpDiseaseImageBD Version 3](https://data.mendeley.com/datasets/jhrtdj9txm/3), CC BY 4.0
- [TigerShrimpBD Version 1](https://data.mendeley.com/datasets/9dj4sk5d55/1), CC BY 4.0
- [ShrimpDiseaseImageBD companion paper](https://doi.org/10.1016/j.dib.2025.111553)

## Licensing

This repository is **not uniformly MIT-licensed**. Runtime/backend/frontend and original documentation use the licenses declared for their trees; `model/pipeline/` is AGPL-3.0-or-later because it is the model/training/data-tooling boundary. Datasets and publications retain their own licenses and attribution requirements.

Read [`LICENSING.md`](LICENSING.md) for the authoritative per-tree map and dependency boundary.
