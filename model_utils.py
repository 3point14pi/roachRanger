# model_utils.py
import os
from ultralytics import YOLO

_model = None

def load_yolo_model():
    global _model
    if _model is None:
        weights = os.getenv("YOLO_MODEL_PATH", "cockroach_detection.pt")
        _model = YOLO(weights)
    return _model
