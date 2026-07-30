# Limitations

This file is the **generator** for the `limitations[]` array in every screening
response, not a commentary on it. `backend/src/shrimp_screening/limitations.py`
parses the headings below, and a contract test asserts that every identifier the
API emits is defined here. Adding a limitation to the product means adding it
here first.

Format, which the parser depends on:

- each limitation is an `## ` heading whose text is its stable identifier;
- the first body line is `**Applies to:** ` followed by `all`, or a comma-separated
  list of `Decision` enum members;
- the remaining prose is the human-readable text.

---

## lim-not-diagnostic

**Applies to:** all

This system reports visible appearances in a photograph. It does not identify a
pathogen, does not establish a cause and is not a diagnosis. Any result that
matters should be confirmed by a qualified aquatic-animal health professional.

## lim-no-lab-confirmation

**Applies to:** all

No laboratory method is involved. Confirmation of white spot syndrome virus
requires molecular or histological testing that an image cannot substitute for.

## lim-two-markers-only

**Applies to:** all

Only two visible appearances are in scope: white-spot-like regions and
dark-gill-like regions. Every other condition, including EMS/AHPND, yellow head
and tail discoloration, is outside the intended screening scope and will not be reported even
when present.

## lim-mapping-provisional

**Applies to:** all

The class order of the combined-folder annotations in the source dataset was
inferred from image evidence, not confirmed by the dataset authors. Responses
carry `model.dataset_mapping_status` for as long as that remains true.

## lim-no-ood-gate

**Applies to:** all

There is no check that the photograph contains a shrimp at all. The system will
score a photograph of a hand, a net or a bucket and report an appearance-based
result for it.

## lim-uncalibrated-thresholds

**Applies to:** all

Every score threshold and quality threshold in `policy/` is an unfitted starting
point. No threshold in this repository has been calibrated against ground truth,
because no trained model exists.

## lim-negative-is-not-health

**Applies to:** NO_TARGET_MARKER_DETECTED

A result of "no target marker detected" does not mean the shrimp is healthy or
disease-free. It means the two appearances in scope were not found in this one
photograph, at this one angle, under this lighting.

## lim-small-target-scale

**Applies to:** WHITE_SPOT_MARKER_DETECTED, MULTIPLE_TARGET_MARKERS_DETECTED

White-spot targets in the source data have a median size of roughly eleven pixels
at the model's input resolution, which is at the edge of what a compact detector
can resolve. Both missed spots and spurious ones are expected until this is
measured.

## lim-single-photograph

**Applies to:** GILL_DARKENING_MARKER_DETECTED, WHITE_SPOT_MARKER_DETECTED, MULTIPLE_TARGET_MARKERS_DETECTED

One photograph of one animal is not a pond assessment. Shell texture, reflections,
debris and shadow all resemble the appearances being screened for.

## lim-no-model-installed

**Applies to:** UNABLE_TO_ASSESS

When no screening model is installed, the service still validates and measures the
photograph but performs no inference at all. Nothing about the animal is being
assessed in that state.
