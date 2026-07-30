# Shrimp visible-marker detector — model card

**Status:** No trained model exists as of 2026-07-30.

The runtime contract and fail-closed data-preparation tooling are implemented. A training pipeline, trained weights and production ONNX artifact do not exist. Every accuracy, latency, parity and calibration field remains **NOT MEASURED**. Fixture-provider output is synthetic demonstration data and must never be presented as model performance.

## Intended task

Localize two visible appearances in a shrimp photograph: dark-gill-like regions and white-spot-like regions. These observations are educational screening signals, not pathogen confirmation.

## Open validation gates

- Combined-folder class order remains author-unconfirmed.
- Annotation conventions differ between source folders.
- Median white-spot boxes may be only about 11 pixels at 640-pixel model input.
- No out-of-distribution or shrimp-presence gate exists.
- No specimen-grouped training, ONNX export, parity test, benchmark or field validation has been completed.
