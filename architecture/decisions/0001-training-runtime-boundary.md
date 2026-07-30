# ADR 0001: Keep training outside the served runtime

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

Training depends on a separate toolchain and has different licensing, hardware,
and data-handling concerns from offline screening. The served path needs to remain
small, auditable and usable without internet access.

## Decision

The development-only pipeline may read approved local datasets and may produce an
ONNX artifact plus registry metadata. The runtime consumes only that versioned
artifact contract. It never imports the training package and never reads source
or prepared datasets.

No model artifact is currently registered; this architecture does not imply that
training has happened.

## Consequences

- The trainer can be replaced without changing the screening API.
- AGPL-governed training code remains outside the MIT runtime boundary.
- Mapping acceptance, annotation-convention acknowledgement, grouped splitting,
  evaluation and parity checks remain explicit preconditions to a usable model.
