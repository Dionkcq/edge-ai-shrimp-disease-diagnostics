# Local model folder

Put one ONNX model in this folder, then run the repository launcher from the
project root:

```bash
python run.py
```

The simplest supported layout is:

```text
model/
└── model.onnx
```

The launcher reads the ONNX graph and embedded metadata, calculates its SHA-256,
and generates ignored runtime registry state. It supports standard YOLO detect
exports when the file includes class names, task, license and compatible graph
metadata. It refuses unsupported or incomplete models rather than guessing how
to decode their outputs.

Official bundles with an explicit registry entry remain supported:

```text
model/model-bundle.zip
├── model/model.onnx
└── registry-entry.json
```

Model weights and bundles are ignored by Git and must never be committed.
