# AI Agent Handoff — Training and ONNX Deployment

## Critical architecture decision

The active training implementation uses **Ultralytics YOLO11n transfer learning**. It does not train a custom CNN from scratch.

Active files:

```text
model/training/src/shrimp_training/runner.py
model/training/src/shrimp_training/cli.py
model/training/src/shrimp_training/artifacts.py
model/training/src/shrimp_training/bundle.py
model/training/src/shrimp_training/config.py
model/training/src/shrimp_training/dataset.py
model/training/configs/compact-nvidia-6gb.json
model/training/pyproject.toml
model/training/uv.lock
```

The old custom from-scratch modules are intentionally not present in the current active trainer. Do not recreate or select them.

## Transfer-learning flow

```text
private/yolo11n.pt
        ↓
YOLO(str(initial_weights))
        ↓
Ultralytics YOLO11n architecture
        ↓
pretrained=True training call
        ↓
Fine-tuned two-class detector
        ↓
best.pt
        ↓
Static ONNX opset 17 export
        ↓
ONNX contract validation and PT/ONNX parity
        ↓
Private model bundle
```

The code does not add custom convolutional layers. It reuses the pretrained YOLO11n CNN backbone, neck, and compatible detection layers. The final class-output portion is adapted for the project classes:

```text
0 = dark_gill
1 = white_spot
```

When the original checkpoint has a different class count, incompatible class-output tensors are reinitialized by the framework while compatible pretrained weights are transferred. This is normal transfer learning; the whole network does not start randomly.

The runner must continue to contain both of these semantics:

```python
constructor(str(weights))
pretrained=True
```

A test in `model/training/tests/test_config_runner.py` asserts that the training call passes `pretrained=True`.

## Training environment

The trainer is a separately locked Python 3.11 AGPL project and is excluded from the runtime workspace. Ultralytics is pinned in `model/training/pyproject.toml`.

Prepare the private environment:

```powershell
uv python install 3.11
uv sync --project model/training --group test --frozen
uv run --project model/training pytest model/training/tests -q
```

Download the private base checkpoint:

```powershell
New-Item -ItemType Directory -Force private | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt" `
  -OutFile "private/yolo11n.pt"
(Get-FileHash "private/yolo11n.pt" -Algorithm SHA256).Hash.ToLower()
```

Expected pinned SHA-256:

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

The full workflow is:

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

The 6 GB profile starts with batch size 4 and retries only CUDA OOM at batch sizes 2 and 1. It must not hide other failures.

## Required training gates

The workflow validates:

- dataset and mapping acceptance;
- train/validation/test separation and specimen leakage;
- class order `0=dark_gill`, `1=white_spot`;
- best PyTorch checkpoint existence;
- locked-test metrics;
- static ONNX input `(1,3,640,640)`;
- expected output layout and metadata;
- PyTorch/ONNX parity;
- private bundle path, size, schema, hash, and registry invariants.

Do not claim the model is accurate merely because export succeeds. Evaluation should include per-class precision/recall/F1, mAP@0.50, mAP@0.50:0.95, IoU sweeps, confidence-threshold analysis, false-positive/false-negative review, image-level screening metrics, robustness, calibration, and ONNX parity.

## Deployment artifact

The simple runtime layout is:

```text
model/model.onnx
```

A model crosses from the private training area to runtime only after ONNX export and contract validation. Keep `.pt`, `.onnx`, datasets, run directories, and bundles private and ignored.

The runtime launcher extracts metadata rather than guessing. Unsupported or ambiguous metadata/output semantics must fail closed.

## Common mistakes to avoid

- Do not run a custom from-scratch model.
- Do not remove `pretrained=True`.
- Do not manually edit a runtime registry.
- Do not assume a filename establishes class order or preprocessing.
- Do not copy `best.pt` into the runtime application.
- Do not commit `private/`, `.runtime/`, weights, or dataset files.
- Do not replace the transfer-learning trainer with `model.py`/`adapter.py` from the old `bert`/`bryan` architecture.
