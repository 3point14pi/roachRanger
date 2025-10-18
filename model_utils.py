# model_utils.py
import os
from ultralytics import YOLO

# Lazy, singleton model loader
_model = None

def load_yolo_model():
    """
    Loads YOLO weights once and returns the model.
    Uses YOLO_MODEL_PATH env var or falls back to 'cockroach_detection.pt'.
    """
    global _model
    if _model is None:
        weights = os.getenv("YOLO_MODEL_PATH", "cockroach_detection.pt")
        _model = YOLO(weights)
        # Optional CUDA speedups (applied by app.py when available)
    return _model
