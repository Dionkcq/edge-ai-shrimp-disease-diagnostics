# AI Agent Handoff — `bert2` Cleanup Review and Integration Decision

## Branch reviewed

The remote branch is unusual because its GitHub name is `origin/bert2`; the local remote-tracking ref is:

```text
origin/origin/bert2
```

Branch commit:

```text
932f71a9968971476058561249764f1713044f09
```

It was based on the older `b5fcd78` (`origin/bryan`), not on the current working `main`.

Current `main` after selective integration:

```text
93b0f3cf747e1331156147fcfe4230af0384e664
```

## Why a wholesale merge was rejected

`bert2` was not a pure cleanup of current main. It also changed architecture and deleted important safeguards.

### Detrimental or unsafe changes

1. It removed `run.py` and `scripts/app_launcher.py`, breaking the one-command launcher and automatic ONNX metadata/runtime-registry setup.
2. It was based before the pretrained YOLO11n restoration and would have reintroduced the older from-scratch trainer.
3. It deleted GitHub Actions workflows.
4. It deleted `LICENSE`, `LICENSING.md`, and `SECURITY.md`.
5. It deleted API contracts and schema-drift tests.
6. It deleted repository policy checks and parts of security/dependency-boundary verification.
7. It deleted architecture source files and publication checks.
8. It deleted dataset provenance and documentation.
9. It moved runtime data into `data/`, requiring coordinated launcher/path/registry changes that were not safe to accept blindly.
10. It renamed the HTTP package to `shrimp_server`, which is a potentially good layering idea but was not compatible with the latest current-main runtime without additional migration work.

## Selective integration performed

The useful organizational idea was adopted:

```text
model/pipeline/
model/training/
```

The following were moved from current main without replacing their contents with `bert2` versions:

```text
pipeline/  → model/pipeline/
training/  → model/training/
```

The current main transfer-learning trainer was preserved. All references were updated in:

- root `pyproject.toml` and `uv.lock`;
- `.gitignore`;
- GitHub Actions workflow paths and Windows training job;
- licensing documentation;
- repository policy checks;
- AGPL/runtime boundary checks;
- pipeline/training tests;
- training README and commands;
- backend documentation references.

The following current-main surfaces were deliberately retained:

```text
run.py
scripts/app_launcher.py
scripts/model_metadata.py
backend/src/shrimp_screening/
contracts/
policy/
guidance/
models/
architecture/
.github/workflows/
LICENSE
LICENSING.md
SECURITY.md
```

## Verification performed

Passed after integration:

```bash
uv run pytest backend/tests model/pipeline/tests scripts/tests -q
PYTHONPATH=model/training/src python -m pytest model/training/tests -q
python -m ruff check backend model/pipeline model/training scripts
python -m ruff format --check backend model/pipeline model/training scripts
uv lock --check
python scripts/check_repository_policy.py
uv run python scripts/check_no_agpl_in_runtime.py
```

GitHub workflow YAML files were parsed successfully, and the remote branch was verified to point at the final commit.

## Guidance for future agents

If asked to merge `bert2` again:

- do not merge it wholesale;
- compare merge-base and current main first;
- preserve launcher, transfer-learning trainer, contracts, legal files, CI, security checks, and private-artifact rules;
- accept organizational moves only after updating every reference, lockfile, CI job, test fixture, and documentation path;
- verify that `model/training/src/shrimp_training/runner.py` still loads the supplied checkpoint and passes `pretrained=True`.
