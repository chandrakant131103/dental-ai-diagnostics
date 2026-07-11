"""
Stage 3: Severity grading.

Rather than training a third deep model (overkill and hard to justify with
limited labeled severity data), this stage engineers clinically-motivated
features from the Stage 2 mask + Stage 1 detection and feeds them into a
lightweight, fully explainable gradient-boosted classifier. This is a
deliberate architecture choice you should defend in interviews: not every
stage of a pipeline needs to be a neural net.

Features per finding:
  - lesion_area_ratio: mask pixel area / tooth crop area
  - mean_intensity_drop: how much darker the lesion region is vs surrounding
    tooth (radiolucency depth correlates with decay/bone-loss severity)
  - boundary_irregularity: perimeter^2 / area (irregular borders often signal
    more advanced/aggressive lesions)
  - detector_confidence: Stage 1 YOLO confidence, as a soft prior

Usage:
    from severity_classifier import extract_features, SeverityClassifier
    feats = extract_features(tooth_crop_gray, lesion_mask, yolo_conf)
    clf = SeverityClassifier.load("severity_model.joblib")
    grade = clf.predict([feats])  # 0=mild, 1=moderate, 2=severe
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class LesionFeatures:
    lesion_area_ratio: float
    mean_intensity_drop: float
    boundary_irregularity: float
    detector_confidence: float


def extract_features(tooth_crop_gray: np.ndarray, lesion_mask: np.ndarray, detector_conf: float) -> LesionFeatures:
    tooth_area = tooth_crop_gray.shape[0] * tooth_crop_gray.shape[1]
    lesion_area = int(lesion_mask.sum())
    area_ratio = lesion_area / max(tooth_area, 1)

    if lesion_area > 0:
        lesion_mean = tooth_crop_gray[lesion_mask.astype(bool)].mean()
        healthy_mean = tooth_crop_gray[~lesion_mask.astype(bool)].mean() if lesion_area < tooth_area else lesion_mean
        intensity_drop = max(0.0, float(healthy_mean - lesion_mean)) / 255.0
    else:
        intensity_drop = 0.0

    mask_u8 = (lesion_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(c, True)
        area = max(cv2.contourArea(c), 1.0)
        irregularity = (perimeter ** 2) / (4 * np.pi * area)  # 1.0 = perfect circle, higher = irregular
    else:
        irregularity = 0.0

    return LesionFeatures(
        lesion_area_ratio=area_ratio,
        mean_intensity_drop=intensity_drop,
        boundary_irregularity=float(irregularity),
        detector_confidence=float(detector_conf),
    )


class SeverityClassifier:
    """Thin wrapper around sklearn's GradientBoostingClassifier so severity
    logic lives in one obviously-inspectable place."""

    LABELS = {0: "mild", 1: "moderate", 2: "severe"}

    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05)
        self.is_fitted = False

    def fit(self, features: list, labels: list):
        X = np.array([list(asdict(f).values()) for f in features])
        y = np.array(labels)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, features: list):
        X = np.array([list(asdict(f).values()) if isinstance(f, LesionFeatures) else f for f in features])
        preds = self.model.predict(X)
        return [self.LABELS[p] for p in preds]

    def predict_proba(self, features: list):
        X = np.array([list(asdict(f).values()) if isinstance(f, LesionFeatures) else f for f in features])
        return self.model.predict_proba(X)

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        obj = cls()
        obj.model = joblib.load(path)
        obj.is_fitted = True
        return obj


def heuristic_grade(features: LesionFeatures) -> str:
    """Fallback rule-based grading for when you don't yet have enough labeled
    severity data to train the classifier - use this first, replace with
    SeverityClassifier once you've hand-labeled ~150-200 findings."""
    score = (
        features.lesion_area_ratio * 40
        + features.mean_intensity_drop * 30
        + min(features.boundary_irregularity / 3.0, 1.0) * 20
        + features.detector_confidence * 10
    )
    if score < 20:
        return "mild"
    elif score < 45:
        return "moderate"
    return "severe"
