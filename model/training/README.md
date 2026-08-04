# Isolated model training

This directory is the separately locked **AGPL-3.0-or-later** training and export boundary. It is intentionally excluded from the Python 3.13 runtime workspace and has its own Python 3.11 `uv.lock`.

The workflow consumes a private, mapping-accepted prepared dataset; trains a compact Ultralytics detector; evaluates the locked test split; exports static ONNX opset 17; verifies the runtime tensor and metadata contract; compares PyTorch and ONNX metrics; and creates a checksummed private return bundle.

No dataset, acceptance record, hardware report, run, checkpoint, ONNX model, or return bundle belongs in Git. Put all such files under the ignored `private/` directory.

## Environment

- Python 3.11 only
- Windows with one NVIDIA CUDA-capable GPU
- PyTorch `2.5.1+cu118` from the CUDA 11.8 wheel index
- Ultralytics `8.4.112`
- Generic profile: `configs/compact-nvidia-6gb.json`

The PyTorch wheel contains the CUDA runtime. A separate CUDA Toolkit installation is not required.

```powershell
uv python install 3.11
uv sync --project model/training --group test --frozen
uv run --project model/training pytest model/training/tests -q
```

## Pinned base artifact

Download the public Ultralytics `v8.4.0` nano checkpoint into `private/`:

```powershell
New-Item -ItemType Directory -Force private | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt" `
  -OutFile "private/yolo11n.pt"
(Get-FileHash "private/yolo11n.pt" -Algorithm SHA256).Hash.ToLower()
```

Required SHA-256:

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

`run-all` refuses any other initial checkpoint.

## Preflight

```powershell
uv run --project model/training shrimp-train preflight `
  --output private/preflight.json
```

Preflight fails unless the active interpreter is Python 3.11, Windows can query GPU 0 through the system `nvidia-smi.exe`, Torch sees CUDA, its bundled runtime is CUDA 11.8, and the device reports at least 5120 MiB.

## Train, evaluate, export, and bundle

First prepare the dataset with `shrimp-pipeline prepare`; that command remains fail-closed until a real mapping-acceptance record matches reviewed overlay evidence.

Then run:

```powershell
uv run --project model/training shrimp-train run-all `
  --dataset-root private/prepared `
  --mapping-acceptance private/mapping-acceptance.json `
  --initial-weights private/yolo11n.pt `
  --profile model/training/configs/compact-nvidia-6gb.json `
  --work-dir private/run-v1 `
  --bundle private/shrimp-model-v1.zip `
  --version 1.0.0
```

The generic 6 GB profile starts at batch 4 and retries only CUDA out-of-memory failures at batch 2 and then batch 1. Other errors are never hidden or converted into retries.

The command succeeds only after:

1. all prepared image and label hashes match the manifest;
2. no specimen crosses train, validation, or test partitions;
3. class order is exactly `0=dark_gill`, `1=white_spot`;
4. a best PyTorch checkpoint exists;
5. locked-test metrics are finite and include both classes;
6. the ONNX graph has static input `(1,3,640,640)` and output `(1,6,8400)`;
7. ONNX metadata matches the runtime class and task contract;
8. PyTorch and ONNX test metrics differ by no more than `0.01`;
9. the return bundle passes path, size, schema, SHA-256, and registry checks.

The PyTorch `best.pt` checkpoint remains only in the ignored laptop work directory.
It is never placed in the return bundle and must never be loaded by the MIT runtime.
Only the validated ONNX artifact crosses that boundary.

`run-all` prints `bundle_manifest_sha256`. Record that value separately from the
ZIP transfer. Verify a transferred bundle without extracting it:

```powershell
uv run --project model/training shrimp-train verify-bundle `
  private/shrimp-model-v1.zip `
  --expected-manifest-sha256 <separately-recorded-64-hex-digest>
```

The external digest anchors verification to the expected manifest. Internal hashes
alone detect corruption but are not an authenticity mechanism, because an attacker
could replace both a file and its internal checksum.

A completed bundle is evidence of a technically valid export—not evidence that the detector is accurate, calibrated, safe for diagnosis, or suitable for production. Those claims require review of the measured results and model card.
