# Edge AI for Sustainable Shrimp Disease Diagnostics

[![Project status](https://img.shields.io/badge/status-foundation%20and%20data%20audit-2563eb)](#project-status)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#development)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)
[![Data policy](https://img.shields.io/badge/datasets-not%20stored%20in%20Git-orange)](#data-and-model-artifacts)

An offline computer-vision demonstrator for screening visible shrimp-health markers at pond side. The project combines a compact edge model with a cautious, locally stored guidance layer.

> This is an educational screening and decision-support project. It does not provide a laboratory-confirmed diagnosis and must not be used to prescribe medication, antibiotics or chemical dosing.

## Team

OIP Group One:

- Dion
- Johnathan
- Lambert
- Bryan

## Problem

Shrimp disease can spread quickly. Small and remote farms may not have reliable internet, a nearby laboratory or immediate access to an aquatic-health specialist. The project explores whether a farmer can photograph one shrimp and receive a useful offline screening result without pretending that an image can confirm a pathogen.

## Intended system

```text
Photograph one shrimp
        ↓
Check that the photograph is usable
        ↓
Run a compact model on the local device
        ↓
Return a visual-marker result or refuse uncertain cases
        ↓
Explain the result using reviewed information stored on the device
        ↓
Escalate unsupported, severe or uncertain cases
```

The first supported visual markers are white-spot appearance and dark-gill appearance. EMS/AHPND is outside the current scope because the verified pond-side datasets do not include that class.

## Safety boundary

The system must:

- say when it cannot assess an image;
- separate visible markers from confirmed disease;
- explain that a negative marker result does not prove that a shrimp is healthy;
- keep guidance educational and source-backed;
- avoid medication, antibiotic and chemical dosing advice;
- recommend expert or laboratory confirmation when appropriate.

## Project status

The repository currently contains the project foundation and verified data records. Two source archives have been checked for licensing and integrity. Model development begins after the team confirms the course requirements, target device and evaluation rules.

| Area | Status |
|---|---|
| Repository and collaboration rules | Ready |
| Dataset provenance and checksums | Verified |
| Specimen-aware split requirement | Defined |
| Target hardware | Awaiting lecturer confirmation |
| Model training | Not started |
| Offline application | Not started |
| Domain review of guidance | Not started |

## Repository layout

```text
.github/               Pull-request, issue and repository checks
CONTRIBUTING.md        Team workflow and review rules
SECURITY.md            Private reporting and safety policy
LICENSE                License for repository code and original documentation
datasets/              Dataset registry and reproducibility metadata
docs/                  Project brief and technical documentation
models/                 Model cards only; large weights stay outside Git
notebooks/              Exploration and training experiments
src/                    Reusable training, evaluation and inference code
tests/                  Automated checks
```

Git does not preserve empty directories, so some development folders will appear after their first file is added.

## Data and model artifacts

Raw datasets, processed copies, trained weights, experiment runs and generated artifacts are excluded from Git. The team keeps shared archives in Google Drive and records the following information here:

- official source and DOI;
- exact dataset version and license;
- archive checksum;
- known limitations;
- rules that prevent train/test leakage.

See [`datasets/DATASET_REGISTRY.md`](datasets/DATASET_REGISTRY.md) and [`datasets/source-notes/dataset_manifest.json`](datasets/source-notes/dataset_manifest.json).

The dataset licenses apply to the datasets themselves. The repository's MIT license does not replace or override those licenses.

## Development

Python 3.9 or newer is required by the course materials.

```bash
git clone git@github.com:Dionkcq/edge-ai-shrimp-disease-diagnostics.git
cd edge-ai-shrimp-disease-diagnostics
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Dependencies and executable commands will be added when the team approves the technical stack. Do not invent a local setup by copying unreviewed package lists into the repository.

## Team workflow

1. Branch from `main` using a short name such as `feat/data-audit`.
2. Keep each change focused and add tests where code behaviour changes.
3. Open a pull request and request at least one teammate review.
4. Do not commit raw images, trained weights, credentials or personal data.
5. Merge only after the required checks pass.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for commit conventions and review expectations.

## Sources

- [Original project brief](docs/project-brief.md)
- [ShrimpDiseaseImageBD Version 3](https://data.mendeley.com/datasets/jhrtdj9txm/3), CC BY 4.0
- [TigerShrimpBD Version 1](https://data.mendeley.com/datasets/9dj4sk5d55/1), CC BY 4.0
- [ShrimpDiseaseImageBD companion paper](https://doi.org/10.1016/j.dib.2025.111553)

## License

Original code and documentation in this repository are licensed under the [MIT License](LICENSE). Third-party datasets and publications retain their own licenses and attribution requirements.
