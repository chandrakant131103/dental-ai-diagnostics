"""
Loads all three pipeline stages once at API startup and caches them, so
/predict requests don't reload weights from disk every call.
"""
import os
from pathlib import Path

import torch
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.segmentation.unet import UNet
from models.severity.severity_classifier import SeverityClassifier

DETECTOR_WEIGHTS = os.environ.get("DETECTOR_WEIGHTS", "runs/detect/tooth_localization/weights/best.pt")
SEGMENTER_WEIGHTS = os.environ.get("SEGMENTER_WEIGHTS", "runs/segment/unet_best.pt")
SEVERITY_WEIGHTS = os.environ.get("SEVERITY_WEIGHTS", "runs/severity/severity_model.joblib")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PipelineModels:
    _instance = None

    def __init__(self):
        print(f"[model_loader] Loading detector from {DETECTOR_WEIGHTS}")
        self.detector = YOLO(DETECTOR_WEIGHTS)

        print(f"[model_loader] Loading segmenter from {SEGMENTER_WEIGHTS}")
        ckpt = torch.load(SEGMENTER_WEIGHTS, map_location=DEVICE)
        self.segmenter = UNet(n_channels=1, n_classes=ckpt["num_classes"])
        self.segmenter.load_state_dict(ckpt["model_state"])
        self.segmenter.to(DEVICE).eval()
        self.seg_class_ids = ckpt["class_ids"]

        self.severity = None
        if Path(SEVERITY_WEIGHTS).exists():
            print(f"[model_loader] Loading severity classifier from {SEVERITY_WEIGHTS}")
            self.severity = SeverityClassifier.load(SEVERITY_WEIGHTS)
        else:
            print("[model_loader] No trained severity classifier found - falling back to heuristic_grade()")

    @classmethod
    def get(cls) -> "PipelineModels":
        if cls._instance is None:
            cls._instance = PipelineModels()
        return cls._instance
