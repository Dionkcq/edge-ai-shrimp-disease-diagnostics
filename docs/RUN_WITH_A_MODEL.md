# Running the demonstration with a trained model

A clean checkout has no model and reports `UNABLE_TO_ASSESS` / `MODEL_UNAVAILABLE`. That is
deliberate, not a bug: weights are never committed, and `models/registry.json` ships empty so an
unvouched artifact cannot be executed.

This page is for running the application end to end once someone hands you the trained ONNX file.

## 1. Obtain the model file

The `.onnx` file is **not in Git** — `models/.gitignore` excludes it, and the repository policy check
enforces that the committed registry declares no models. Ask the maintainer for:

- `shrimp-marker-yolo11n-1.0.0.onnx` (about 10 MB)
- its SHA-256 digest

Transfer it directly (USB, private file transfer). Do not commit it, and do not attach it to an
issue or pull request.

## 2. Install and verify it

Put the file in `models/`, then confirm the digest matches what you were given:

```powershell
Get-FileHash models\shrimp-marker-yolo11n-1.0.0.onnx -Algorithm SHA256
```

```bash
sha256sum models/shrimp-marker-yolo11n-1.0.0.onnx
```

If it does not match, stop. The application will refuse the file anyway.

## 3. Register it locally

Replace the contents of `models/registry.json` with the entry below, substituting the digest you
just verified.

```json
{
  "schema_version": "1.0.0",
  "models": [
    {
      "model_id": "shrimp-marker-yolo11n",
      "version": "1.0.0",
      "filename": "shrimp-marker-yolo11n-1.0.0.onnx",
      "sha256": "PASTE_THE_VERIFIED_SHA256_HERE",
      "input_size": 640,
      "class_names": { "0": "dark_gill", "1": "white_spot" },
      "opset": 17,
      "output_layout": "ultralytics_v8_detect_v1",
      "dataset_mapping_status": "PROVISIONAL_UNCONFIRMED",
      "artifact_license": "AGPL-3.0-or-later",
      "training_toolchain": "ultralytics 8.4.112 / torch 2.5.1+cu118"
    }
  ]
}
```

> **Keep this change local. Do not commit it.**
>
> `scripts/check_repository_policy.py` requires the committed registry to be empty, so a populated
> `registry.json` fails that check and three of its tests plus
> `backend/tests/unit/test_registry.py::test_the_committed_registry_parses_and_vouches_for_nothing`
> — 4 failures in total, all from this one cause. Restore it before committing anything:
>
> ```bash
> git checkout models/registry.json
> ```

## 4. Run it

Two terminals.

```powershell
# backend
$env:SHRIMP_PROVIDER="onnx"
$env:SHRIMP_ONNX_MODEL_PATH="$PWD\models\shrimp-marker-yolo11n-1.0.0.onnx"
$env:SHRIMP_ENV="demo"
uv run uvicorn shrimp_screening.main:create_app --factory --host 127.0.0.1 --port 8000
```

```bash
# frontend
cd frontend
npm ci      # first time only
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api`, `/livez` and `/readyz` to port 8000.

Check the backend picked up the model:

```bash
curl http://127.0.0.1:5173/readyz
# {"status":"ready","provider":"onnx","model_available":true,"reason":null}
```

`SHRIMP_ENV=demo` is required. A `demo` or `production` build refuses to start on anything but a real
ONNX provider — there is no silent fallback to synthetic fixture output.

## 5. What to expect

**The model is poor and the demonstration will show that.** Measured test mAP50 is 0.095. Most
images return `NO_TARGET_MARKER_DETECTED` or `UNABLE_TO_ASSESS`; detections that do appear are mostly
in the `LOW` confidence band, because recall is 0.32 at confidence 0.05 but only 0.02 at 0.50.
`white_spot` performs worse than `dark_gill`.

Every response carries `DATASET_CLASS_MAPPING_UNCONFIRMED` and `THRESHOLDS_UNCALIBRATED`, plus the
applicable `limitations` references. Those are product-visible facts and must not be hidden or
filtered in any interface built on this API.

This demonstrates that the pipeline works end to end. It demonstrates nothing about diagnostic
ability. See [`models/MODEL_CARD.md`](../models/MODEL_CARD.md) for the full measured results and the
open validation gates.
