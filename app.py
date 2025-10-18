# --- Add this to your Streamlit app (same file) ---
import streamlit as st
import cv2
import numpy as np
import time

# reuse your existing model loader
from model_utils import load_yolo_model
yolo_model = yolo_model if 'yolo_model' in globals() else load_yolo_model()

st.subheader("Live Camera (1 frame every 5 seconds)")

# Controls
colA, colB = st.columns(2)
start = colA.toggle("Start live capture", value=False, key="live_toggle")
interval_sec = colB.number_input("Interval (seconds)", min_value=1, max_value=30, value=5, step=1)

# This placeholder will show the latest annotated frame
frame_placeholder = st.empty()
status_placeholder = st.empty()

# Use autorefresh to trigger a rerun every N ms while running
if start:
    # Autorefresh triggers a rerun every `interval_sec` seconds
    st.autorefresh(interval=interval_sec * 1000, key="live_autorefresh", limit=None)

    # Capture exactly ONE frame per rerun
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        st.error("Could not open camera. Check permissions or try a different device index.", icon="🚨")
    else:
        ok, frame_bgr = cap.read()
        cap.release()

        if not ok or frame_bgr is None:
            st.error("Failed to read a frame from the camera.", icon="🚨")
        else:
            # Convert BGR (OpenCV) -> RGB (YOLO/Plot convenience)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run YOLO on the single frame; no files saved
            with st.spinner("Detecting..."):
                results = yolo_model.predict(
                    source=frame_rgb,  # numpy array input
                    conf=0.5,
                    verbose=False,
                    save=False
                )
                # Annotated image as numpy array
                annotated = results[0].plot()  # RGB array

            # Show original + annotated side-by-side
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Original frame")
                frame_placeholder.image(frame_rgb, use_column_width=True)
            with c2:
                st.caption("Detections")
                st.image(annotated, use_column_width=True)

            status_placeholder.success(f"Captured and processed one frame at {time.strftime('%H:%M:%S')}")

else:
    st.info("Toggle ‘Start live capture’ to begin sampling one frame every few seconds.")
