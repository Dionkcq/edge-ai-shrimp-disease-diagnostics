# Contributing

## Branch and review workflow

1. Pull the latest `main` branch.
2. Create a short branch such as `feat/data-audit`, `fix/split-leakage` or `docs/model-card`.
3. Keep the change focused. Do not mix unrelated cleanup into the same pull request.
4. Add or update tests when code behaviour changes.
5. Run the available checks locally.
6. Open a pull request and ask at least one teammate to review it.
7. Resolve comments before merging. Preserve meaningful commits; squash only noisy work-in-progress history.

## Commit examples

```text
feat: add image-quality gate
fix: prevent specimen leakage across dataset splits
test: add offline inference smoke test
docs: record dataset license and provenance
```

## Pull-request expectations

A pull request should state:

- what changed and why;
- how it was tested;
- any effect on datasets, metrics, model behaviour or safety wording;
- screenshots or sample output when the user interface changes;
- limitations that remain.

Do not claim a model improvement from training metrics alone. Include the relevant untouched or grouped evaluation result and explain the comparison.

## Data rules

- Keep raw images, processed datasets and trained weights out of Git.
- Record every source, version, license and checksum in the dataset registry.
- Group all photographs of one shrimp in the same data split.
- Create augmentation only inside the training split.
- Preserve an untouched evaluation set.
- Do not add a disease label that the source data does not support.

## Safety rules

- Describe predictions as visible-marker screening, not laboratory confirmation.
- Support an `UNABLE_TO_ASSESS` outcome.
- Explain that `NO_TARGET_MARKER_DETECTED` does not prove that a shrimp is healthy.
- Do not provide medication, antibiotic or chemical dosing advice.
- Keep guidance cited, reviewable and versioned.
- Escalate unsupported, severe or uncertain cases.
- Keep credentials, personal data and sensitive location metadata out of the repository.

## Secrets and large files

Never commit `.env` files, API keys, OAuth tokens, personal credentials, raw datasets, generated runs or model weights. If one is committed accidentally, notify Dion immediately so it can be removed and rotated where necessary.
