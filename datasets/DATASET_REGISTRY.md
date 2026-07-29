# Dataset Registry

Every dataset used by this project must have an entry here before training.

## Primary candidate: ShrimpDiseaseImageBD

- **Official repository:** https://data.mendeley.com/datasets/jhrtdj9txm/3
- **Dataset version:** Version 3, published 22 January 2025; DOI `10.17632/jhrtdj9txm.3`
- **Companion paper:** https://doi.org/10.1016/j.dib.2025.111553
- **Paper mirror:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12048804/
- **Task support:** image classification and YOLO-format disease-region detection
- **Original images:** 1,149 JPG images at 2048 × 2048
- **Underlying shrimp specimens:** 416
- **Classes:** Healthy (403 images), Black Gill (198), WSSV (328), Black Gill + WSSV (220)
- **Collection:** Bangladesh farms/markets under expert supervision; smartphone capture
- **License:** CC BY 4.0 (verified on the Mendeley Data Version 3 page). Attribution is required.
- **Known limitations:** Bangladesh-only source domain; no temporal progression; multiple images per individual shrimp; no EMS/AHPND class.
- **Required split rule:** group by individual shrimp if specimen identity is recoverable. Never let near-duplicate views of the same shrimp cross train/validation/test splits.
- **Status:** downloaded and ZIP-integrity verified; SHA-256 recorded in `datasets/source-notes/dataset_manifest.json`.

## Secondary candidate: TigerShrimpBD

- **Official repository:** https://data.mendeley.com/datasets/9dj4sk5d55/1
- **Dataset version:** Version 1, published 26 February 2025; DOI `10.17632/9dj4sk5d55.1`
- **Task support:** tiger-shrimp image classification
- **Images:** 1,001 original images; 3,574 total after augmentation
- **Classes:** Healthy, Black Gill, Yellow Head, WSSV
- **Important rule:** obtain original images and perform augmentation only inside the training split. Do not randomly split the already-augmented pool, because variants of one source image could leak into test data.
- **License:** CC BY 4.0 (verified on the Mendeley Data page). Attribution is required.
- **Status:** downloaded and ZIP-integrity verified; use as a secondary/generalization source after duplicate analysis.

## Not selected as a primary source

### Kaggle mirror of ShrimpDiseaseImageBD

https://www.kaggle.com/datasets/pritamroy24mcb1016/shrimp-disease-image-dataset-for-detection-models

This appears to mirror the Mendeley dataset. The Kaggle page shows inconsistent license signals, so the official Mendeley repository and companion paper should remain authoritative.

### Hugging Face BD Fish & Shrimp Disease Dataset

https://huggingface.co/datasets/Saon110/bd-fish-disease-dataset

Contains 5,887 fish/shrimp images and four shrimp categories, but access is gated and it appears to curate/augment other datasets. License is CC BY-NC-SA 4.0. It must not be merged into the primary dataset until duplicate/provenance checks are complete.

### Small Roboflow datasets

Potentially useful only for external stress testing. Their small sizes and unclear relationship to larger sources make them unsuitable as the primary training corpus.

## Critical scope gap

The assignment mentions EMS/AHPND, but the main public pond-side image datasets found so far do not contain EMS/AHPND. A 2025 paper reports a 1,357-image AHPND/EHP/HPV/Normal dataset, but these are 100× hepatopancreatic histopathology fields—not ordinary pond-side photos—and its data-availability statement says the dataset is **not publicly available** because of institutional restrictions. The approved system must not claim EMS/AHPND image detection. Treat EMS/AHPND as a future extension requiring a suitable licensed dataset, different imaging hardware/workflow, and laboratory/expert validation.
