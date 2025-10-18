# app.py
import time
import numpy as np
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from model_utils import load_yolo_model

# --- Streamlit page setup ---
st.set_page_config(page_title="Live Cockroach Detection", layout="wide")
st.title("🐞 Live Cockroach Detection (WebRTC, YOLO)")

# --- Sidebar controls (affect speed/FPS) ---
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence", 0.1, 0.95, 0.55, 0.05)
imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640], index=1)
resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 0.75, 0.05)
draw_mode = st.sidebar.selectbox("Drawing", ["Fast boxes", "Ultralytics .plot()"], index=0)
show_fps = st.sidebar.toggle("Show FPS overlay", True)
request_fps = st.sidebar.selectbox("Requested camera FPS", [15, 24, 30, 60], index=2)
preview_width = st.sidebar.slider("Preview width (%)", 30, 100, 50, 5)

st.sidebar.caption("Tip: smaller imgsz / higher conf / pre-resize ↑ = faster FPS")

# keep OpenCV from oversubscribing tiny CPUs (feel free to comment out on big servers)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# --- Lightweight box drawer (faster than .plot()) ---
def fast_draw(img_bgr, boxes):
    if boxes is None or len(boxes) == 0:
        return img_bgr
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_bgr

class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        # Load once
        self.model = load_yolo_model()

        # Attempt GPU + half precision if available
        self.device = "cpu"
        self.use_half = False
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda")
                torch.backends.cudnn.benchmark = True  # autotune conv kernels
                self.device = 0  # CUDA device index
                self.use_half = True  # enables half precision
        except Exception:
            pass

        # Runtime-updatable settings
        self.conf = 0.55
        self.imgsz = 480
        self.resize_factor = 0.75
        self.draw_mode = "Fast boxes"
        self.show_fps = True

        # FPS meter
        self._t0 = time.time()
        self._cnt = 0
        self._fps = 0.0

    def recv(self, frame):
        # Get frame (BGR)
        img_bgr = frame.to_ndarray(format="bgr24")

        # Optional pre-resize BEFORE inference (saves compute)
        if 0.4 <= self.resize_factor < 1.0:
            img_bgr = cv2.resize(img_bgr, (0, 0),
                                 fx=self.resize_factor, fy=self.resize_factor,
                                 interpolation=cv2.INTER_AREA)

        # Inference
        results = self.model.predict(
            img_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device if self.device != "cpu" else None,
            half=self.use_half,
            verbose=False,
        )
        r = results[0]

        # Draw detections
        if self.draw_mode == "Fast boxes":
            out = fast_draw(img_bgr.copy(), r.boxes)
        else:
            # Ultralytics visualizer (nicer, slower)
            out = r.plot()

        # FPS meter
        if self.show_fps:
            self._cnt += 1
            if self._cnt >= 15:
                t1 = time.time()
                self._fps = self._cnt / (t1 - self._t0)
                self._t0, self._cnt = t1, 0
            cv2.putText(out, f"{self._fps:.1f} FPS", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# --- WebRTC streamer (browser camera) ---
ctx = webrtc_streamer(
    key="yolo-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": int(request_fps), "max": 60},
        },
        "audio": False,
    },
    video_processor_factory=YOLOProcessor,
    async_processing=True,  # don't block capture thread -> higher FPS
    video_html_attrs={
        "style": {"width": f"{preview_width}%", "margin": "auto"},
        "controls": False,
        "autoPlay": True,
        "muted": True,
        "playsInline": True,
    },
)

# --- Live-update processor settings from the sidebar ---
if ctx and ctx.video_processor:
    vp = ctx.video_processor
    vp.conf = conf
    vp.imgsz = int(imgsz)
    vp.resize_factor = float(resize_factor)
    vp.draw_mode = draw_mode
    vp.show_fps = bool(show_fps)

# Helpful notes
st.caption(
    "If FPS is low: reduce inference size to 320, increase confidence to 0.7, "
    "and raise pre-resize (e.g., 0.6). A CUDA GPU with half precision is fastest."
)
