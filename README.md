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
| Detector mAP@50 | TBD |
| Detector mAP@50-95 | TBD |
| Segmentation mean Dice | TBD |
| Macro precision / recall (detector) | TBD |
| Inference latency (CPU / GPU) | TBD |

## Known dataset caveats worth mentioning in interviews
- 18 detection classes are heavily imbalanced (e.g. "Caries" vastly
  outnumbers "TAD" or "Supra Eruption") — see the class-distribution printout
  in `models/detection/train.py`; addressed via Dice+CE loss on the
  segmentation side and is a good place to discuss focal loss / oversampling
  if pushed further.
- Left/right flip augmentation is intentionally disabled by default since
  tooth siding (FDI numbering) is clinically meaningful; flipping without
  remapping labels would silently corrupt them.

## Disclaimer
This is a research/portfolio project, not a certified medical device. The
generated report explicitly states it is meant to support, not replace,
review by a licensed dental professional.
