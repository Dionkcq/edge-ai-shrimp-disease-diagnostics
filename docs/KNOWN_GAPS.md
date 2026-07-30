# Known gaps

Every entry states an owner, a status, and **what would change our mind** — the
specific observation that would let the gap be closed or downgraded. A gap without
that last field tends to stay open forever because nobody knows what "done" is.

Distinct from `docs/LIMITATIONS.md`: that file is the machine-readable generator for
the `limitations[]` array in API responses and describes *permanent* properties of
the product. This file tracks *work not yet done*.

Last reviewed: 2026-07-30.

---

## GAP-01 — No trained model exists

**Owner:** Dion · **Status:** open, expected for this cycle

There are no weights. `models/registry.json` declares zero models, `/readyz`
returns 503 on a clean checkout, and every screening returns `UNABLE_TO_ASSESS` /
`MODEL_UNAVAILABLE`. The runtime contract and fail-closed data-preparation tools are
implemented. Training, model selection and export are not. No accuracy, latency or
calibration figure anywhere in this repository has been measured.

**What would change our mind:** a completed training run whose ONNX export passes
the contract assertions in `detection/onnx_provider.py`, with per-size recall and
grouped-CV confidence intervals recorded in `models/MODEL_CARD.md`.

---

## GAP-02 — Small-target detection may not work at 640 px input

**Owner:** Dion · **Status:** open, unquantified — highest technical risk

The median white-spot box in the WSSV-only folder is ~0.00031 of a 2048² frame,
which is **≈11 px at 640×640 input**. A nano detector's finest head is stride 8, so
the median target occupies 1–2 cells. This is the single largest technical risk in
the project and it is not addressed by any code currently written.

**What would change our mind:** a per-size recall curve from a real training run,
comparing 640 against 960/1024, scored at image level on held-out specimen groups.
If recall in the smallest size bucket is unusable at 640 and acceptable at 1024,
the input size changes and the latency budget is renegotiated.

---

## GAP-03 — Combined-folder class mapping is author-unconfirmed

**Owner:** Dion · **Status:** open, blocked on a third party

The `4. WSSV_BG` folder's class order is inferred, not documented. Four signals
concord on `0=dark_gill, 1=white_spot`: boxes per file (1.89 vs 7.32), median
box-vs-ring luminance Δ (−18.04 vs −1.98), share of boxes brighter than surroundings
(3.9% vs 43.5%), and within-image centroid spread (0.037 tight vs 0.101 dispersed).
**One signal does not corroborate:** median box area is 0.00066 vs 0.00067 —
indistinguishable.

The archive contains no `data.yaml`, no names file and no author statement. Its own
`Readme.docx` calls the folder `BG_WSSV` while the directory is `4. WSSV_BG`, which
lowers confidence in every undocumented aspect of the archive.

**What would change our mind:** a written statement from the dataset authors. Until
then `datasets/mapping_acceptance.json` must be signed by a human reviewer with
`author_confirmed: false` recorded truthfully, and every API response carries
`dataset_mapping_status: PROVISIONAL_UNCONFIRMED`.

---

## GAP-04 — Annotation conventions differ between source folders

**Owner:** Dion · **Status:** open, unquantified

BG-only median box area is 0.00185 of the frame (≈88 px square at 2048²); combined
folder class-0 median is 0.00066 (≈53 px). The same nominal target is drawn ~1.7×
smaller in one folder than the other. The folders were annotated under **different
conventions**, not merely different class-ID scopes. Pooling them injects systematic
label noise that no mapping choice can remove.

This is arguably a larger hazard than GAP-03 and is guarded by a separate gate:
`annotation_convention_acknowledged` must be `true` in the acceptance record.

**What would change our mind:** per-folder metrics (never only pooled) from a real
evaluation, showing whether a model trained on the pooled corpus degrades on either
folder's convention. If it does, the folders need per-source calibration or one of
them is dropped.

---

## GAP-05 — No out-of-distribution or shrimp-presence gate

**Owner:** Dion · **Status:** open, mitigation only

The system will confidently score a photograph of a hand, a net or a bucket. The
only mitigation is a capture guide frame, which is advisory. This is recorded as a
permanent limitation (`lim-no-ood-gate`) as well as a gap, because a real fix is
out of scope this cycle.

**What would change our mind:** a shrimp-presence classifier or a coarse
region-of-interest stage whose false-accept rate on non-shrimp photographs is
measured, not assumed.

---

## GAP-06 — Every threshold is uncalibrated

**Owner:** Dion · **Status:** open by construction

`policy/decision_policy_v1.json` and `policy/quality_policy_v1.json` both carry
`"status": "UNCALIBRATED"`, and the API emits a `THRESHOLDS_UNCALIBRATED` notice.
The quality thresholds were chosen to be conservative on synthetic images; the
score thresholds have never seen a real score distribution.

**What would change our mind:** a score distribution from a trained model on a
held-out specimen group, and a chosen operating point with its
sensitivity/specificity trade-off stated explicitly.

---

## GAP-07 — Statistical power is low

**Owner:** Dion · **Status:** open, structural

416 specimens (Healthy 142 / BG 75 / WSSV 118 / WSSV_BG 81), of which only 81 are
in the combined class. Any metric computed from this corpus needs specimen-grouped
cross-validation with confidence intervals, and the CI **width** must be quoted
rather than the point estimate.

**What would change our mind:** nothing available to this project. More data would;
none is obtainable. The response is honest reporting, not a fix.

---

## GAP-08 — ONNX parity is untested

**Owner:** Dion · **Status:** open, blocked on GAP-01

No torch-vs-onnxruntime parity harness exists because there is no training or
export implementation and nothing to compare. The decode path is verified by
round-trip property tests and by external letterbox/NMS goldens, which prove
*internal consistency* — they cannot prove that our assumptions about Ultralytics'
actual output layout are correct.

**What would change our mind:** one real `(1, 6, 8400)` tensor from a genuine
export, committed as an `.npz` anchor, plus a passing parity run within tolerance.

---

## GAP-09 — Training hardware is unresolved

**Owner:** Dion · **Status:** open, decision needed

YOLO11n at 640, batch 8, ~900 images on a 2-core 15 W CPU is roughly 20–45 min per
epoch, i.e. 30–75 hours for 100 epochs, on the same laptop that must run the demo.

**What would change our mind:** a decision. The recommendation is a free
Colab/Kaggle GPU for training only, stated openly — *deployment* is what must be
offline, and pretending otherwise while burning 75 hours of laptop time helps
nobody. Whichever is chosen goes in the model card.

---

## GAP-10 — Guidance corpus is literature-reviewed, not expert-reviewed

**Owner:** Dion · **Status:** open, disclosed

`guidance/guidance_v1.json` carries `review_status` and every item carries a
`source_id` and citation. No qualified aquatic-animal health professional has
reviewed it. The API exposes the review status through `/api/v1/meta` so a client
cannot present it as expert-reviewed guidance.

**What would change our mind:** a named reviewer with relevant qualifications
signing off, recorded per item.

---

## GAP-11 — Frontend and architecture documentation

**Owner:** Dion · **Status:** closed in implementation, deployment pending

The responsive React/Vite frontend and LikeC4 architecture site now exist and are
covered by contract, component, browser, accessibility, offline and static-site
checks. GitHub Pages deployment remains a release operation rather than a product
gap.

**What changed our mind:** successful local frontend production builds, desktop and
mobile Playwright runs, LikeC4 validation and a static-site integrity check.

---

## GAP-12 — Inference execution has no enforceable deadline

**Owner:** Dion · **Status:** open, blocked on the production model architecture

`queue_wait_timeout_seconds` bounds how long a request can wait for the single
inference slot. It does **not** stop a detector after execution begins. The current
ONNX call runs in a Python worker thread, and Python cannot safely terminate a
stuck native inference call. Releasing the semaphore while that thread continued
would permit uncontrolled overlapping inference, so the service deliberately does
not claim an execution timeout.

**What would change our mind:** run production inference in a supervised worker
process that can be terminated and replaced safely, with tests proving that a
forced deadline leaves no orphan process, does not release capacity early, and
returns a stable abstention or service error.
