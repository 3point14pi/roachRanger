# app.py
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# --- YOLO (cached so it loads once) ---
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    # put your weight path here (keep it small, e.g., yolov8n.pt)
    return YOLO("models/yolov8n.pt")

model = load_model()

st.set_page_config(page_title="Cockroach Live", layout="wide")
st.title("Cockroach Detection — Live")

# UI controls
device_index = st.number_input("Webcam index", min_value=0, value=0, step=1,
                               help="If you have multiple cameras, try 0, 1, 2…")
conf = st.slider("Confidence", 0.1, 0.9, 0.5, 0.05)
fps_limit = st.slider("Max FPS", 1, 30, 10)
start = st.toggle("Start webcam", value=False, key="run_toggle")

# Display placeholder
frame_ph = st.empty()
status_ph = st.empty()

def _open_cap(idx: int):
    cap = cv2.VideoCapture(idx)
    # Optional: request a resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

if start:
    cap = _open_cap(int(device_index))
    if not cap.isOpened():
        st.error("Could not open webcam. Try a different index (0/1/2) and check permissions.")
    else:
        try:
            status_ph.info("Streaming… toggle Off to stop.")
            delay = 1.0 / max(1, fps_limit)
            while st.session_state.get("run_toggle", False):
                ok, frame_bgr = cap.read()
                if not ok:
                    status_ph.error("Camera read failed. Is the device in use by another app?")
                    break

                # YOLO expects RGB or BGR ndarray; Ultralytics handles BGR fine
                results = model.predict(frame_bgr, conf=conf, verbose=False)
                plotted_bgr = results[0].plot()  # returns BGR ndarray

                # Show
                frame_ph.image(plotted_bgr, channels="BGR", use_column_width=True)

                # Simple FPS cap
                time.sleep(delay)
        finally:
            cap.release()
            status_ph.info("Camera released.")
