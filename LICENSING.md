# Licensing

This repository is **not uniformly licensed**, and it cannot be. The served
application is MIT. The intended training toolchain (Ultralytics YOLO) is
AGPL-3.0-or-later. A single file cannot honestly claim both, so the two are kept in
separate trees with an enforced boundary between them.

`scripts/check_no_agpl_in_runtime.py` is the enforcement. This file is the
explanation.

---

## Per-tree licence map

| Tree | Licence | Notes |
|---|---|---|
| `backend/` | MIT | The served runtime. Must never resolve or import an AGPL distribution. |
| `contracts/`, `policy/`, `guidance/`, `docs/`, `scripts/` | MIT | Text, JSON and tooling with no AGPL dependency. |
| `pipeline/` | **AGPL-3.0-or-later** | Development-time data and training tooling. Never imported by `backend/`. |
| `datasets/` (contents) | see `datasets/DATASET_REGISTRY.md` | Source archives are third-party CC BY 4.0. Not redistributed here. |
| Model weights (`models/*.onnx`, `*.pt`) | **AGPL-3.0-or-later if produced by Ultralytics** | None exist in this repository. See below. |

The root `LICENSE` (MIT) is scoped to the MIT trees above. `pipeline/LICENSE.AGPL`
carries the AGPL text for that tree.

---

## Why the boundary is drawn where it is

**AGPL §13, the network clause, is not triggered.** It applies to running *your
modified version* of the AGPL program so that users interact with it remotely. We
do not modify Ultralytics, and the served application never imports it. The
screening service could be exposed on a network without §13 attaching.

**AGPL §5, combined works, is the real exposure.** `pipeline/` imports Ultralytics.
Conveyed together with the rest of the repository, that is arguably a combined
work. The mitigation is separation, not a claim that the problem does not exist:

- `pipeline/` is its own distribution package with its own licence file;
- every module that imports Ultralytics carries
  `SPDX-License-Identifier: AGPL-3.0-or-later`;
- `backend/` declares no dependency on it and cannot import it;
- the boundary is asserted by a test and by a CI check, not by a convention.

**Ultralytics is not declared as a dependency anywhere in the lockfile.** This is
deliberate and slightly inconvenient. Declaring it — even under an optional extra
in `pipeline/` — would resolve an AGPL distribution into the same `uv.lock` that
resolves the MIT runtime, which is exactly the coupling the boundary exists to
prevent. No training implementation or install path exists yet;
`shrimp-pipeline train` therefore reports `UNAVAILABLE` rather than implying that
the separately licensed toolchain is ready.

---

## Weights

Ultralytics asserts that the AGPL extends to models trained with their code. That
position is legally contested, but fine-tuning would start from `yolo11n.pt`, which
is itself an Ultralytics AGPL artifact, and that strengthens their claim rather
than weakening it.

**This project therefore does not claim MIT over any weights it might later
produce.** When weights exist, `models/registry.json` must record
`artifact_license: "AGPL-3.0-or-later"` and the training toolchain and version.

As of 2026-07-30 no weights exist, `models/registry.json` declares zero models, and
`scripts/check_repository_policy.py` fails the build if that stops being true
without the registry being updated.

### The escape hatch, and its honest cost

The backend depends on an **ONNX output contract**
(`models/registry.json` → `output_layout`), not on Ultralytics. If the AGPL
position becomes unacceptable, the replacement is a new trainer in `pipeline/` plus
one registry field; the runtime does not change.

The cost is real and should not be glossed over: the obvious BSD-3 alternative
(`torchvision` SSDlite320-MobileNetV3) is genuinely weak on the ~11 px targets that
dominate this dataset. It is a *licence* fallback, not a performance-equivalent one.

---

## Dataset attribution (CC BY 4.0)

Both source datasets are CC BY 4.0. That licence requires attribution **on
distribution and on adaptations**, which includes any converted or re-split
derivative this project produces. Attribution is therefore a product obligation,
not a README footnote:

- `datasets/DATASET_REGISTRY.md` carries the full citation, DOI and licence;
- the modifications made must be stated alongside it (class-ID remapping,
  specimen-aware re-splitting, exclusion of pre-augmented images);
- a later frontend slice must surface this in an in-app **Data sources** panel.

Neither archive is redistributed in this repository. `datasets/raw/` is gitignored
and the repository-policy check fails if any archive becomes tracked.

---

## Third-party runtime dependencies

All runtime dependencies are permissively licensed. The direct runtime set includes
FastAPI and Pydantic (MIT), Starlette and Uvicorn (BSD-3-Clause),
python-multipart (Apache-2.0), Pillow (MIT-CMU), NumPy (BSD-3-Clause and bundled
permissive components), and ONNX Runtime (MIT). `httpx2` (BSD-3-Clause) is test-only.
The lockfile is the authoritative version record; dependency licences are checked
from installed package metadata during release review.

`opencv` is deliberately excluded from the runtime — the variance-of-Laplacian,
luminance and letterbox operations are a few dozen lines of NumPy — and
`check_no_agpl_in_runtime.py` asserts it cannot return through a transitive
dependency.
