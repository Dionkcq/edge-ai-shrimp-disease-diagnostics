# Architecture documentation

This directory is the self-contained source for the project's LikeC4 site. It is
safe to publish: it documents interfaces and boundaries, not private plans,
source datasets, model weights, secrets, or runtime application assets.

## What the model says

The system is an **offline visible-marker screening aid**, not a medical or
veterinary diagnostic system. The default clean-checkout state has no trained
model installed. In that state readiness is false and screening abstains with
`UNABLE_TO_ASSESS / MODEL_UNAVAILABLE`.

The views move from context to implementation detail:

1. **System context** — people, the offline system, and external escalation paths.
2. **Containers** — browser UI, FastAPI service, policies, guidance and optional ONNX artifact.
3. **Screening components** — bounded intake through quality, inference, policy and contract assembly.
4. **Screening sequence** — request flow and explicit failure/abstention branches.
5. **Decision and abstention** — the five-state decision vocabulary and reasons for declining.
6. **Security and trust boundaries** — untrusted image input, ephemeral pixels, local-only processing and reviewed content.
7. **Data and training boundary** — development-only, AGPL-governed tooling producing a replaceable ONNX contract.
8. **Offline deployment** — phone and laptop on a local hotspot with no runtime internet dependency.

## Local commands

```bash
npm ci
npm run format:check
npm run validate
npm run build
npm run check:site
```

The application runtime is intentionally started from the repository root with one
command:

```bash
python run.py
```

The launcher discovers an out-of-band model bundle, validates its hash and
contract, generates ignored runtime registry state, builds the frontend and
starts the same-origin API/UI process. The diagram's model artifact remains
optional and the clean checkout still abstains safely.

For the interactive architecture site:

```bash
npm run dev -- --listen 127.0.0.1 --port 5173
```

Open <http://127.0.0.1:5173>. To view it from another device on a trusted private hotspot, use `--listen 0.0.0.0` and the laptop's private IP; do not expose the development server to an untrusted network.

The production build uses `/edge-ai-shrimp-disease-diagnostics/`, the project
path for `Dionkcq/edge-ai-shrimp-disease-diagnostics`, and hash navigation so
shared diagram links do not require server-side rewrites.

## Publication boundary

The public architecture site is deployed through the manually dispatched `.github/workflows/architecture-pages.yml` workflow. The workflow validates and scans output before deployment:

<https://dionkcq.github.io/edge-ai-shrimp-disease-diagnostics/>
