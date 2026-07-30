# Dataset preparation boundary

Raw archives and derived images are local-only. The repository commits provenance, hashes, split manifests and code—not photographs or labels.

`shrimp-pipeline audit` is read-only. `shrimp-pipeline evidence` creates a deterministic, hashable set of source-box overlays but does **not** record human acceptance. `shrimp-pipeline prepare` refuses to proceed unless a human copies `mapping_acceptance.example.json` to the ignored `mapping_acceptance.json`, reviews at least 60 stratified overlays, replaces every placeholder, records the SHA-256 of the exact evidence report bytes, and explicitly acknowledges both the provisional class-order evidence and the annotation-convention mismatch. The gate rejects unknown fields, copied placeholders, non-ISO dates, mismatched evidence hashes, and non-Boolean acknowledgements. This repository does not ship an acceptance.

Healthy photographs are negatives with empty YOLO label files. All views of one specimen must remain in one partition. Source augmentations are excluded from validation and test partitions.
