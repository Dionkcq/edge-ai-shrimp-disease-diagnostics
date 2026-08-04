# AI Agent Handoff — Shrimp Disease Screening Project

**Repository:** `Dionkcq/edge-ai-shrimp-disease-diagnostics`
**Target branch:** `main`
**Verified remote commit:** `93b0f3cf747e1331156147fcfe4230af0384e664`
**Latest commits:**
- `93b0f3c` — organize model tooling under `model/`
- `8b5b3f7` — restore pretrained YOLO training workflow
- `fbc2d9c` — force CNN screening for attached chat images
- `3ab49d7` — normalize ONNX exporter metadata
- `8eb6d2c` — auto-discover ONNX model metadata
- `308b603` — add one-command application launcher

## Project purpose

This is an offline, appearance-based shrimp screening demonstrator. It screens photographs for two visible markers:

```text
0 = dark_gill
1 = white_spot
```

It is **not** a pathogen-confirming diagnostic system, health certificate, laboratory replacement, medication recommender, or disease-free guarantee.

The application must fail closed. Missing, unsupported, ambiguous, or unregistered detector artifacts must produce unavailable/abstain behavior such as:

```text
MODEL_UNAVAILABLE
UNABLE_TO_ASSESS
```

Never fabricate detections, class mappings, confidence semantics, preprocessing, anchors, or model metadata.

## Runtime architecture

```text
User image
  ↓
FastAPI image intake and security bounds
  ↓
Image decode + EXIF normalization + quality assessment
  ↓
Configured detector provider
  ↓
ONNX inference and output decoding
  ↓
Deterministic screening policy
  ↓
Cited guidance lookup
  ↓
Optional local Ollama explanation/chat
  ↓
React frontend
```

Main runtime files:

```text
run.py                                      Cross-platform launcher
scripts/app_launcher.py                     Model discovery and runtime setup
scripts/model_metadata.py                   Embedded ONNX metadata extraction
backend/src/shrimp_screening/main.py       FastAPI application factory
backend/src/shrimp_screening/api/           HTTP routes, errors, middleware
backend/src/shrimp_screening/detection/     Providers, ONNX validation, decoding, NMS
backend/src/shrimp_screening/policy/        Deterministic decision and quality policies
backend/src/shrimp_screening/guidance/      Reviewed, cited guidance corpus loader
backend/src/shrimp_screening/ai/            Chat agent, bounded memory, CNN tool
backend/src/shrimp_screening/llm/           Optional local Ollama client
frontend/                                   React/Vite UI
```

## Native runtime workflow

From the repository root:

```bash
python run.py
```

The launcher:

1. resolves the repository root from `run.py`;
2. discovers `model/model.onnx` or an official private model bundle;
3. extracts embedded metadata where possible;
4. computes the model SHA-256;
5. creates ignored disposable state in `.runtime/`;
6. generates `.runtime/registry.json`;
7. starts the application.

No model is committed to Git. With no model, the app may start but readiness/screening must remain unavailable.

Useful verification:

```bash
python run.py --no-browser --port 18000
```

Then check:

```text
GET /livez  → 200
GET /readyz → 503 with MODEL_UNAVAILABLE when no valid model exists
```

Do not manually edit `.runtime/registry.json`, calculate hashes for users, or commit `.env`, model files, datasets, checkpoints, runs, or private bundles.

## Conversational agent

The LLM has one declared tool:

```text
screen_shrimp_image
```

The tool runs image quality checks, the configured detector, and the deterministic screening policy. The LLM may request the tool, but it does not author user-facing pond-side replies or decide the screening result.

The tool result also carries the matching cited local guidance item, its explicit
review status and source titles. Every image-result reply is constructed
deterministically from that payload; the backend does not ask the model for a second
free-form answer. This prevents generic human-health language, invented pond actions
and unsupported diagnosis/treatment content from reaching the user. Refer users to
a qualified aquatic-animal health professional, never a doctor or healthcare
professional. Every no-image turn fails closed to an upload prompt; no free-form
model reply crosses the pond-side chat boundary.

A chat turn with an attached image normally stores three bounded records:

```text
user → tool result → assistant
```

Memory is process-local and per-session:

```python
deque(maxlen=12)
```

It includes user, tool, and assistant records. It is not persistent memory and is not RAG. There is no embedding database, vector retrieval, document chunking, or semantic RAG pipeline.

Guidance is a deterministic lookup from `guidance/guidance_v1.json`, with validation and citations.

## Important constraints

- Keep detector screening deterministic and fail closed.
- Keep Ollama optional; basic startup must not require it.
- Do not put private model artifacts, datasets, secrets, credentials, or tokens in Git.
- Preserve the ONNX runtime contract and metadata validation.
- Preserve API schemas in `contracts/` and schema-drift tests.
- Preserve CI, licensing, security, dataset provenance, and architecture documentation.
- Do not reintroduce the old from-scratch custom detector.

## Current test commands

```bash
uv run pytest backend/tests model/pipeline/tests scripts/tests -q
PYTHONPATH=model/training/src python -m pytest model/training/tests -q
python -m ruff check backend model/pipeline model/training scripts
python -m ruff format --check backend model/pipeline model/training scripts
uv lock --check
uv run python scripts/check_repository_policy.py
uv run python scripts/check_no_agpl_in_runtime.py
```

Frontend checks, when frontend code changes:

```bash
cd frontend
npm run format:check
npm run lint
npm run typecheck
npm run test
npm run build
npm run check:offline
```

## Safe change protocol

Before modifying a function, inspect its callers and tests. Before changing a path, search the whole repository, CI workflows, lockfiles, scripts, tests, docs, and ignore rules. After changes, run focused tests, affected suites, lint/format, policy checks, and `git diff --check`.

For a cleanup branch based on an older commit, compare ancestry first and integrate selectively. Do not merge deletions blindly.
