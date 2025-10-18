import streamlit as st
import cv2
import time
from streamlit_autorefresh import st_autorefresh

# reuse your model loader
from model_utils import load_yolo_model
yolo_model = yolo_model if 'yolo_model' in globals() else load_yolo_model()

st.subheader("Live Camera (capture 1 frame every N seconds)")

# UI controls
colA, colB, colC = st.columns(3)
start = colA.toggle("Start", value=False, key="live_toggle")
interval_sec = colB.number_input("Interval (sec)", min_value=1, max_value=30, value=5, step=1)
cam_index = colC.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)

# placeholders for images and status
orig_ph = st.empty()
det_ph  = st.empty()
msg_ph  = st.empty()

def capture_one_frame(index:int=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None, "Could not open camera (index {}).".format(index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None, "Failed to read a frame."
    return frame_bgr, None

if start:
    # trigger a re-run automatically every interval
    st_autorefresh(interval=interval_sec * 1000, key="live_autorefresh")
    # on each rerun, take exactly ONE frame, run YOLO, and display
    frame_bgr, err = capture_one_frame(cam_index)
    if err:
        st.error(err, icon="🚨")
    else:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with st.spinner("Detecting..."):
            results = yolo_model.predict(frame_rgb, conf=0.5, save=False, verbose=False)
            annotated = results[0].plot()  # RGB array

        col1, col2 = st.columns(2)
        with col1:
            col1.caption("Original")
            orig_ph.image(frame_rgb, use_column_width=True)
        with col2:
            col2.caption("Detections")
            det_ph.image(annotated, use_column_width=True)

        msg_ph.success(f"Captured at {time.strftime('%H:%M:%S')}")
else:
    st.info("Toggle ‘Start’ to begin sampling a single frame every few seconds.")
