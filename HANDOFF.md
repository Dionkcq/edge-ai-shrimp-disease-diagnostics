# Stronger-Laptop Handoff

**Project:** Edge AI for Sustainable Shrimp Disease Diagnostics

**Purpose:** Move the verified project to Dion's stronger laptop for data review and, after the training workflow is added, model training and ONNX export.

**Status:** The application and pre-training data foundation are verified. **A real trainer is not implemented yet. Do not start training from this revision.**

## 1. Source of truth

| Item | Canonical source |
|---|---|
| Repository | `Dionkcq/edge-ai-shrimp-disease-diagnostics` |
| Integration branch | `feat/end-to-end-platform` until PR #1 is merged; use `main` after merge |
| Pull request | <https://github.com/Dionkcq/edge-ai-shrimp-disease-diagnostics/pull/1> |
| Raw datasets | Dion's private copies; never Git |
| Prepared datasets | Local generated output; never Git |
| Model weights and experiment runs | Local artifacts; never Git |
| Runtime model trust record | `models/registry.json` |
| Model limitations and metrics | `models/MODEL_CARD.md` |

The repository intentionally starts with an empty model registry. The application must report `MODEL_UNAVAILABLE` until a reviewed ONNX file and matching metadata are installed.

## 2. Clone or update the repository

### Recommended: GitHub CLI

```bash
gh auth login
gh repo clone Dionkcq/edge-ai-shrimp-disease-diagnostics
cd edge-ai-shrimp-disease-diagnostics
git switch feat/end-to-end-platform
git pull --ff-only
```

After PR #1 is merged, use:

```bash
git switch main
git pull --ff-only origin main
```

If the repository already exists, do not clone another copy:

```bash
cd edge-ai-shrimp-disease-diagnostics
git fetch origin --prune
git switch feat/end-to-end-platform
git pull --ff-only
```

Confirm the worktree is clean:

```bash
git status --short --branch
```

Do not continue if source files show unexplained local modifications.

## 3. Record the stronger laptop's hardware

Save the following output as `artifacts/hardware/hardware-report.txt`. The `artifacts/` tree is ignored by Git.

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force artifacts/hardware | Out-Null
$report = @()
$report += "=== OS ==="
$report += (Get-CimInstance Win32_OperatingSystem | Format-List Caption,Version,OSArchitecture | Out-String)
$report += "=== CPU ==="
$report += (Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors | Out-String)
$report += "=== RAM ==="
$report += (Get-CimInstance Win32_ComputerSystem | Format-List TotalPhysicalMemory | Out-String)
$report += "=== NVIDIA ==="
$report += (& nvidia-smi 2>&1 | Out-String)
$report += "=== PYTHON ==="
$report += (& python --version 2>&1 | Out-String)
$report += "=== UV ==="
$report += (& uv --version 2>&1 | Out-String)
$report | Set-Content artifacts/hardware/hardware-report.txt
Get-Content artifacts/hardware/hardware-report.txt
```

If `nvidia-smi` is unavailable, include the exact error; do not install a random CUDA version yet.

### Linux

```bash
mkdir -p artifacts/hardware
{
  echo '=== OS ==='; cat /etc/os-release
  echo '=== KERNEL ==='; uname -a
  echo '=== CPU ==='; lscpu
  echo '=== RAM ==='; free -h
  echo '=== NVIDIA ==='; nvidia-smi 2>&1 || true
  echo '=== PYTHON ==='; python --version 2>&1 || true
  echo '=== UV ==='; uv --version 2>&1 || true
} > artifacts/hardware/hardware-report.txt
cat artifacts/hardware/hardware-report.txt
```

Send this report back before the training environment is frozen. GPU model, VRAM, operating system and driver support determine the batch size and dependency choice.

## 4. Install the verified application environment

Required baseline tools:

- Python 3.13
- `uv`
- Node.js 24
- npm
- Git

From the repository root:

```bash
uv sync --locked --all-packages --all-groups
```

Install frontend dependencies:

```bash
cd frontend
npm ci
cd ..
```

The future training environment will be separate from this runtime workspace because its licensing and GPU dependencies must not enter the MIT runtime lockfile.

## 5. Run baseline verification

### Python and repository gates

```bash
uv run ruff format --check backend pipeline scripts
uv run ruff check backend pipeline scripts
uv run mypy backend/src pipeline/src scripts
uv run pytest -q
uv run python scripts/check_repository_policy.py
uv run python scripts/check_no_agpl_in_runtime.py
```

### Frontend gates

```bash
cd frontend
npm run check
npx playwright install --with-deps chromium
npm run test:e2e
cd ..
```

Expected baseline behavior:

- all gates pass;
- desktop and mobile browser tests pass;
- the default service reports model unavailable;
- no result is represented as real model inference.

## 6. Place the private datasets

Create the ignored directory:

```bash
mkdir -p datasets/raw
```

Place these archives there without renaming them:

```text
datasets/raw/ShrimpDiseaseImageBD_v3.zip
datasets/raw/TigerShrimpBD_v1.zip
```

Do not stage or commit either archive. Verify:

```bash
git status --short
```

Neither archive should appear because `datasets/raw/` is ignored.

## 7. Audit the archives

```bash
mkdir -p artifacts/audit
uv run shrimp-pipeline audit \
  datasets/raw/ShrimpDiseaseImageBD_v3.zip \
  datasets/raw/TigerShrimpBD_v1.zip \
  --output artifacts/audit/archive-audit.json
```

Known reference counts from the verified source copies:

| Archive | Image entries | Unique hashes | Duplicate entries | Inconsistent duplicate groups |
|---|---:|---:|---:|---:|
| ShrimpDiseaseImageBD v3 | 1,895 | 1,149 | 746 | 0 |
| TigerShrimpBD v1 | 4,575 | 4,575 | 0 | 0 |

Stop and report the audit if these values differ. Do not silently train on a different archive revision.

## 8. Generate mapping evidence

The primary archive uses independent class-number namespaces. The proposed global mapping is:

```text
BG/0      → 0, dark-gill appearance
WSSV/0    → 1, white-spot appearance
WSSV_BG/0 → 0, dark-gill appearance — provisional
WSSV_BG/1 → 1, white-spot appearance — provisional
```

Generate at least 60 review overlays:

```bash
uv run shrimp-pipeline evidence \
  datasets/raw/ShrimpDiseaseImageBD_v3.zip \
  artifacts/evidence/mapping-review \
  --minimum-overlays 60 \
  --seed 20260730
```

A human must inspect the generated overlays before preparation. Do not copy `datasets/mapping_acceptance.example.json` and change only its status. Acceptance must reference the exact evidence-report SHA-256 and record who reviewed what.

Only after genuine acceptance should a local, ignored file exist at:

```text
datasets/mapping_acceptance.json
```

Then preparation can run:

```bash
uv run shrimp-pipeline prepare \
  datasets/raw/ShrimpDiseaseImageBD_v3.zip \
  datasets/processed/primary \
  --acceptance datasets/mapping_acceptance.json \
  --seed 20260730
```

Preparation must prove that all photographs of one shrimp remain in the same split.

## 9. Training stop gate

At this revision:

```bash
uv run shrimp-pipeline train
```

must return:

```text
UNAVAILABLE
```

That is intentional. **Do not invent a training command or install Ultralytics into the runtime workspace.** Wait for the separately verified training-handoff revision.

The next revision must provide:

1. a pinned, separately licensed training environment;
2. a hardware-aware configuration;
3. grouped training, validation and test manifests;
4. compact-detector fine-tuning and checkpoint resumption;
5. evaluation and calibration outputs;
6. ONNX export with the required opset and output layout;
7. PyTorch-versus-ONNX parity tests;
8. a deterministic return-bundle command.

## 10. What to return after training becomes available

Do not send only `best.onnx`. Return one bundle containing:

```text
model.onnx
model.sha256
registry-entry.json
training-config.yaml
dataset-manifest.json
mapping-acceptance.json
evaluation.json
calibration.json
parity.json
environment.txt
README.txt
```

The bundle must state:

- model architecture and upstream checkpoint;
- training-toolchain version and artifact licence;
- ONNX opset, input size, class index order and output layout;
- source archive hashes and prepared-manifest hash;
- grouped split seed and specimen counts;
- validation/test metrics and confidence intervals;
- calibration and abstention policy inputs;
- PyTorch/ONNX parity tolerances and results;
- GPU, driver and training duration.

Keep the bundle outside Git. Transfer it directly back to the integration machine.

## 11. Integration after the model is returned

The returned model will not be trusted automatically. Integration must:

1. verify every bundle checksum;
2. inspect the training and evaluation evidence;
3. copy the ONNX artifact into the ignored `models/` location;
4. add its exact metadata and SHA-256 to `models/registry.json`;
5. update `models/MODEL_CARD.md` with measured—not estimated—results;
6. run ONNX provider, parity, screening API and browser tests;
7. benchmark latency on the actual demonstration laptop;
8. test with networking disabled;
9. preserve abstention and educational-only wording.

A successful model load is not proof of useful accuracy. Release requires the evidence above.
