# Local model folder

Put one official validated model bundle (`.zip`) in this folder, then run the
repository launcher from the project root:

```bash
python run.py
```

The launcher extracts the bundle into ignored `.runtime/` state, verifies the
model SHA-256 and registry metadata, builds the frontend when necessary, and
starts the API and browser UI together. Do not edit `.env` or
`models/registry.json` for an official bundle.

The bundle must contain:

```text
model/model.onnx
registry-entry.json
```

Raw `.onnx` files are accepted only when accompanied by `model-manifest.json`.
Model weights and bundles are ignored by Git and must never be committed.
