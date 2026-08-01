# Shrimp visible-marker detector — model card

**Status:** A trained demonstration artifact exists as of 2026-08-01: `shrimp-marker-yolo11n` v1.0.0.

The weights are **not in this repository** and never will be — they are transferred out of band. A
clean checkout still contains no model and still reports `MODEL_UNAVAILABLE`, which is the correct
state. See [`docs/RUN_WITH_A_MODEL.md`](../docs/RUN_WITH_A_MODEL.md) to install it locally.

**It is not fit for any diagnostic use.** Measured test mAP50 is **0.095**, meaning the detector misses the large majority of markers it is shown. It is registered so the end-to-end runtime path (image intake → quality gate → ONNX inference → decision policy → response) can be exercised and demonstrated. Every figure below is measured, not estimated. Fixture-provider output remains synthetic demonstration data and must never be presented as model performance.

## Intended task

Localize two visible appearances in a shrimp photograph: dark-gill-like regions and white-spot-like regions. These observations are educational screening signals, not pathogen confirmation.

## Artifact

| Field | Value |
|---|---|
| Model id / version | `shrimp-marker-yolo11n` 1.0.0 |
| Architecture | YOLO11n (181 layers, ~2.62 M parameters) |
| Upstream checkpoint | `yolo11n.pt`, sha256 `0ebbc80d…44ee1` |
| ONNX opset | 17 |
| Input | `images`, static `[1, 3, 640, 640]` |
| Output | `output0`, `[1, 6, 8400]` — layout `ultralytics_v8_detect_v1` |
| Artifact sha256 | `31a0af8f041ac349cb9ddb8b88dbc1487cdc9255dabee2d2875ef1e93f7a4899` |
| Artifact licence | AGPL-3.0-or-later (Ultralytics-derived) |
| Training toolchain | ultralytics 8.4.112 / torch 2.5.1+cu118 |

## Training data

ShrimpDiseaseImageBD v3 only. TigerShrimpBD v1 was not obtained and is out of scope for this version.

1,149 canonical images after de-duplication (1,895 entries, 746 duplicates removed), split by specimen group with seed `20260730` into train 812 / validation 166 / test 171. All photographs of one specimen stay within a single split.

Prepared-manifest sha256 `fb15ddbc…2d0d8`.

## Measured performance — test split, ONNX runtime

171 images, 799 instances.

| Metric | All | dark_gill | white_spot |
|---|---:|---:|---:|
| mAP50 | **0.0951** | 0.115 | 0.075 |
| mAP50-95 | **0.0255** | 0.0322 | 0.0188 |
| Precision | 0.169 | 0.181 | 0.157 |
| Recall | 0.191 | 0.233 | 0.150 |

Instances: dark_gill 146, white_spot 653. The rarer class scores higher, so the gap is not explained by class support alone.

Recall falls away sharply as confidence rises — 0.321 at 0.05, 0.132 at 0.20, 0.021 at 0.50 — so nearly every detection this model produces is low-confidence.

Training: 216 epochs (early-stopped from a 300 ceiling, patience 20), best validation mAP50 0.1159 at epoch 196, batch 4 at 640 px on an RTX 3060 Laptop 6 GB, roughly 73 minutes.

Inference: 24.0 ms per image at batch 1 on the training GPU. **Not** benchmarked on the demonstration laptop's target runtime.

## PyTorch / ONNX parity

Passed. Maximum metric delta **0.006849** against a tolerance of 0.010, across 16 comparisons: threshold-independent mAP50, mAP50-95 and per-class mAP50-95; recall at confidences 0.05–0.50; precision at 0.05–0.20 only, because precision's denominator collapses at higher confidences and the metric's resolution there is coarser than the tolerance.

## Calibration

**NOT MEASURED.** No reliability diagram, temperature scaling or confidence calibration has been performed. Confidence is reported only as discrete bands, never as a percentage, because no calibrated probability exists.

## Class mapping — provisional and unconfirmed

`dataset_mapping_status` is `PROVISIONAL_UNCONFIRMED` and ships in every response.

The mapping `0 → dark_gill`, `1 → white_spot` was accepted by a human reviewer (Dion, 2026-07-31) against 60 rendered overlays, evidence report sha256 `bdabec9d…d77af`. The reviewer records that they **are not a trained shrimp pathologist**; that class 0 read as dark-gill and class 1 as white-spot across the sample with no systematic inversion observed; and that **on some images the spots were subtle and hard to judge confidently**. The dataset authors have never confirmed the combined-folder class order.

If that mapping is wrong, every label this model emits is wrong in the same direction.

## Open validation gates

- Combined-folder class order remains author-unconfirmed.
- Annotation conventions differ between source folders.
- Median white-spot boxes may be only about 11 pixels at 640-pixel model input; markers measure roughly 13–32 px, which is small relative to the input resolution and untested at higher input sizes.
- No out-of-distribution or shrimp-presence gate exists — the model will emit boxes on photographs containing no shrimp at all.
- No field validation, no demonstration-laptop latency benchmark, no calibration.
- 812 training images may simply be too few for this task.
