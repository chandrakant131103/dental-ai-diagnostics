"""
FastAPI serving layer for the dental diagnostic pipeline.

Endpoints:
  POST /predict  - image in, structured JSON findings out
  POST /report    - same input, returns a downloadable PDF report
  GET  /health    - liveness check

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
(In Colab, expose with ngrok or pyngrok - see notebooks/02_deploy_demo.ipynb)
"""
import io
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.model_loader import PipelineModels, DEVICE
from models.severity.severity_classifier import extract_features, heuristic_grade
from reports.pdf_generator import generate_pdf_report

app = FastAPI(title="Dental Diagnostic AI API", version="1.0")


class Finding(BaseModel):
    tooth_box: List[float]        # [x1, y1, x2, y2]
    finding_class: str
    detector_confidence: float
    lesion_area_ratio: float
    severity: str


class PredictResponse(BaseModel):
    num_teeth_detected: int
    findings: List[Finding]


def read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image.")
    return img


def run_pipeline(image_bgr: np.ndarray) -> List[Finding]:
    models = PipelineModels.get()
    findings = []

    det_results = models.detector.predict(source=image_bgr, conf=0.25, verbose=False)[0]
    boxes = det_results.boxes.xyxy.cpu().numpy()
    confs = det_results.boxes.conf.cpu().numpy()
    class_ids = det_results.boxes.cls.cpu().numpy().astype(int)
    names = det_results.names

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    for box, conf, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = gray[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (256, 256))
        crop_tensor = torch.from_numpy(crop_resized).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            seg_logits = models.segmenter(crop_tensor)
            seg_mask = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()

        lesion_mask = (seg_mask > 0).astype(np.uint8)
        feats = extract_features(crop_resized, lesion_mask, float(conf))

        if models.severity is not None:
            severity = models.severity.predict([feats])[0]
        else:
            severity = heuristic_grade(feats)

        findings.append(Finding(
            tooth_box=[float(x1), float(y1), float(x2), float(y2)],
            finding_class=names.get(cls_id, str(cls_id)),
            detector_confidence=float(conf),
            lesion_area_ratio=float(feats.lesion_area_ratio),
            severity=severity,
        ))

    return findings


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = read_image(contents)
    findings = run_pipeline(image)
    return PredictResponse(num_teeth_detected=len(findings), findings=findings)


@app.post("/report")
async def report(file: UploadFile = File(...), patient_name: str = "N/A"):
    contents = await file.read()
    image = read_image(contents)
    findings = run_pipeline(image)
    pdf_bytes = generate_pdf_report(image, findings, patient_name=patient_name)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=dental_report.pdf"},
    )
