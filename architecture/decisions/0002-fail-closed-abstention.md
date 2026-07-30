# ADR 0002: Abstain rather than infer a reassuring answer

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

A missing model, rejected image, weak signal or inference failure must not be
mistaken for a negative screening result. A negative result must also not be
presented as evidence that an animal is healthy or disease-free.

## Decision

The public contract has exactly five decision values. Four describe visible-marker
screening outcomes; `UNABLE_TO_ASSESS` absorbs every decline path. A separate
abstention reason records `MODEL_UNAVAILABLE`, `IMAGE_QUALITY_REJECTED`,
`LOW_CONFIDENCE` or `INFERENCE_FAILED`.

`/readyz` returns 503 when no detector is loaded. A clean checkout contains no
trained weights and therefore follows this unavailable path.

## Consequences

- Missing capability cannot silently render as `NO_TARGET_MARKER_DETECTED`.
- Clients must display notices and limitations next to the result.
- Scores are shown as discrete bands, not as uncalibrated probability claims.
