# Dental Diagnostic AI Platform

An end-to-end diagnostic pipeline for panoramic dental X-rays: tooth/finding
detection, pathology segmentation, severity grading, explainability, and a
deployed web app — not a single-model Kaggle notebook.

## Why this project is different from a typical student submission
Most public notebooks on this dataset (including the one this project started
from) stop at "train YOLO, print precision/recall." This project instead:

- **Fixes a real bug** in the reference evaluation code (an IoU calculation
  that always returned zero union area — see `evaluation/metrics.py`) and
  replaces it with proper IoU-matched, per-class precision/recall/F1 and a
  confusion matrix.
- **Uses the segmentation masks the dataset actually ships** (COCO-format
  annotations under `COCO/COCO/annotations/`) which the reference notebook
  never touched — training a real U-Net instead of only bounding boxes.
- **Adds a severity-grading stage** that turns raw model output into
  something a clinician-facing report can use (mild/moderate/severe), using
  interpretable engineered features rather than a black-box third model.
- **Adds Grad-CAM explainability** so predictions come with visual evidence,
  not just a confidence score.
- **Ships as a product**: FastAPI backend + Streamlit frontend + generated
  PDF report, deployable from a single Colab session via ngrok, or to
  Hugging Face Spaces / Render for a permanent demo link.

## Architecture

```
Kaggle dataset
      │
      ▼
Preprocessing (CLAHE, denoise, letterbox + label remap)
      │
      ▼
Augmentation (rotation/brightness/noise; flips off by default —
              left/right jaw side is clinically meaningful)
      │
      ├──────────────┐
      ▼              ▼
Stage 1: YOLO   Stage 2: U-Net (COCO masks)
tooth/finding   pathology segmentation
localization    (caries, bone loss, lesions)
      │              │
      └──────┬───────┘
             ▼
   Stage 3: Severity grading
   (engineered features → GBM classifier
    or rule-based heuristic fallback)
             │
             ▼
   Explainability (Grad-CAM overlays)
             │
             ▼
   FastAPI (/predict, /report) ──► Streamlit app ──► PDF report
```

## Repo layout
```
data/            kaggle_download.py, preprocess.py, augment.py
models/
  detection/     YOLO training (train.py)
  segmentation/  U-Net (unet.py), COCO dataset loader (dataset.py), train.py
  severity/      feature extraction + GBM/heuristic grading
evaluation/      fixed IoU matcher, per-class metrics, mAP, confusion matrix
explainability/  Grad-CAM for both YOLO and U-Net
api/             FastAPI app + model loader
app/             Streamlit UI
reports/         PDF report generator (ReportLab)
notebooks/       01_full_pipeline_colab.ipynb — run everything end to end
```

## Quickstart (Google Colab)
1. Open `notebooks/01_full_pipeline_colab.ipynb` in Colab, GPU runtime.
2. Run cells top to bottom: clone → Kaggle download → preprocess → augment →
   train detector → train segmenter → evaluate → Grad-CAM demo → deploy.
3. Grab the ngrok Streamlit URL printed at the end and open it.

## Quickstart (local)
```bash
git clone <your-repo-url> && cd dental-ai-diagnostics
pip install -r requirements.txt

# after training (or download pretrained weights into runs/):
uvicorn api.main:app --reload &
streamlit run app/streamlit_app.py
```

## Results
_Fill in after training on your machine — this table is what recruiters look
for first, so keep it current:_

| Metric | Value |
|---|---|
| Detector mAP@50 | 0.410 |
| Detector mAP@50-95 | 0.214 |
| Detector macro precision / recall (fixed-IoU eval, thresh=0.5) | 0.432 / 0.428 |
| Caries precision / recall | 0.6562 / 0.1583 |
| Segmentation mean Dice | TBD |
| Inference latency (CPU) | TBD |
| Inference latency (GPU) | TBD |
| Detector epochs completed | 22 |

**Per-class detection performance** (fixed IoU-matched eval):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Caries | 0.6562 | 0.1583 | 0.2551 |
| Crown | 0.646 | 0.8911 | 0.749 |
| Filling | 0.5071 | 0.4352 | 0.4684 |
| Implant | 0.7656 | 0.8989 | 0.8269 |
| Malaligned | 0.0 | 0.0 | 0.0 |
| Mandibular Canal | 0.2524 | 0.65 | 0.3636 |
| Missing teeth | 0.6791 | 0.5385 | 0.6007 |
| Periapical lesion | 0.5093 | 0.0977 | 0.1639 |
| Retained root | 0.0 | 0.0 | 0.0 |
| Root Canal Treatment | 0.4912 | 0.5773 | 0.5308 |
| Root Piece | 0.4868 | 0.5993 | 0.5372 |
| croen | 0.0 | 0.0 | 0.0 |
| impacted tooth | 0.8343 | 0.9373 | 0.8828 |
| maxillary sinus | 0.2273 | 0.2083 | 0.2174 |

## Known dataset caveats worth mentioning in interviews
- 18 detection classes are heavily imbalanced (e.g. "Caries" vastly
  outnumbers "TAD" or "Supra Eruption") — see the class-distribution printout
  in `models/detection/train.py`; addressed via Dice+CE loss on the
  segmentation side and is a good place to discuss focal loss / oversampling
  if pushed further.
- Left/right flip augmentation is intentionally disabled by default since
  tooth siding (FDI numbering) is clinically meaningful; flipping without
  remapping labels would silently corrupt them.

## Current status / known gaps (updated after first full training run)
- **Caries recall is currently 0.16** (misses ~84% of actual caries) despite
  reasonable precision (0.66). This is the metric that matters most for a
  cavity-detection product and is not yet production-usable. Mitigation in
  progress: `data/oversample_rare_classes.py` oversamples caries-containing
  training images (ultralytics YOLOv8's public API doesn't expose per-class
  loss weighting, so sampling frequency is the practical lever). Re-train
  and re-run `evaluation/generate_report.py` after applying it.
- **The U-Net segmentation stage previously collapsed to all-background**
  (dead/negative logits, 0% positive Grad-CAM activation). Root cause: the
  training dataloader resized whole panoramic X-rays to 256x256 while
  inference crops individual findings first — the model was trained on a
  different input distribution than it was asked to predict on. Fixed in
  `models/segmentation/dataset.py` (per-finding crops matching inference)
  and `models/segmentation/train.py` (class-weighted loss + a mandatory
  pre-training sanity check that aborts if masks are empty). Re-train with
  the fixed dataset and run `models/segmentation/evaluate.py` to get a real
  Dice score before trusting segmentation output.
- Detector mAP@50 (0.41) and mAP@50-95 (0.214) are from a run that stopped
  at epoch 22/30 (early stopping or interruption) - a longer/tuned run
  would likely improve these.
- Latency has not yet been benchmarked; run `evaluation/benchmark_latency.py`
  on your target deployment hardware and re-run `generate_report.py` to
  fill that row in with a real, hardware-specific number.
- The "mAP@50-95" from the custom `evaluation/metrics.py::compute_map()` is
  a simplified proxy (mean of macro-precision across IoU thresholds), not a
  full COCO-style precision/recall-curve integration — the mAP figures in
  the table above are ultralytics' own validation metrics instead, which
  are the standard implementation.

## Disclaimer
This is a research/portfolio project, not a certified medical device. The
generated report explicitly states it is meant to support, not replace,
review by a licensed dental professional.
