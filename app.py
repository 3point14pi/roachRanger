# app.py
import time
import threading
import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Your own util; must return a preloaded ultralytics YOLO model
from model_utils import load_yolo_model

# ───────────────── Page setup ─────────────────
st.set_page_config(page_title="Live Cockroach Detection", layout="wide")
st.title("🐞 Live Cockroach Detection (WebRTC + YOLO)")

# ──────── Sidebar controls ────────
st.sidebar.header("Controls")

hq = st.sidebar.toggle("High-Quality mode (GPU recommended)", value=False)

preview_width = st.sidebar.slider("Preview width (%)", 30, 100, 60, 5)
req_fps = st.sidebar.selectbox("Requested camera FPS", [15, 24, 30, 60], index=2)

if hq:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [480, 640, 800], index=1)
    resize_factor = 1.0  # no pre-resize in HQ
    draw_mode = st.sidebar.selectbox("Drawing", ["Fast boxes", "Ultralytics .plot()"], index=0)
else:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.60, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640], index=1)
    resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 0.75, 0.05)
    draw_mode = st.sidebar.selectbox("Drawing", ["Fast boxes", "Ultralytics .plot()"], index=0)

show_fps = st.sidebar.toggle("Show FPS overlay", True)

st.sidebar.caption(
    "Speed tips: pick Fast boxes, smaller imgsz (320/480), pre-resize ~0.5–0.75, and higher conf."
)

# Keep OpenCV threads sane on small CPUs
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass


# ───────────── Video processor ─────────────
class YOLOProcessor(VideoProcessorBase):
    def __init__(self, init_conf, init_imgsz, init_resize, init_draw, show_fps, hq):
        # Load once
        self.model = load_yolo_model()

        # Device / half / fuse
        self.device = "cpu"
        self.use_half = False
        try:
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self.model.to("cuda")
                torch.backends.cudnn.benchmark = True
                self.device = 0
                # fuse conv+bn for speed; harmless if already fused
                try:
                    self.model.fuse()
                except Exception:
                    pass
                # half precision if supported
                try:
                    self.model.model.half()
                    self.use_half = True
                except Exception:
                    self.use_half = False
            else:
                self.torch = torch
        except Exception:
            self.torch = None  # ultralytics still uses torch; this is mostly defensive

        # Runtime state
        self.conf = float(init_conf)
        self.imgsz = int(init_imgsz)
        self.resize_factor = float(init_resize)
        self.draw_mode = init_draw
        self.show_fps = bool(show_fps)
        self.hq = bool(hq)

        # FPS meter
        self._t0, self._cnt, self._fps = time.time(), 0, 0.0

        # Async inference state
        self._lock = threading.Lock()
        self._last_boxes = None
        self._infer_busy = False
        # infer every Nth frame (tune); 2 for std, 1 for HQ
        self._frame_skip = 1 if self.hq else 2
        self._frame_id = 0

    def _rescale_boxes(self, boxes, fx, fy):
        """Rescale xyxy in-place from resized image back to original frame."""
        if boxes is None or len(boxes) == 0 or (fx == 1.0 and fy == 1.0):
            return boxes
        # boxes.xyxy is an Nx4 tensor; do it safely without extra allocs
        try:
            xyxy = boxes.xyxy
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]] / fx
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]] / fy
        except Exception:
            pass
        return boxes

    def _draw_fast(self, img_bgr, boxes):
        """Minimal, fast rectangle draw without labels."""
        if boxes is None or len(boxes) == 0:
            return img_bgr
        # Iterate once; avoid per-box heavy effects
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (50, 220, 50), 2, lineType=cv2.LINE_AA)
        return img_bgr

    def _predict_one(self, img_bgr):
        # Optional pre-resize BEFORE inference
        if 0.4 <= self.resize_factor < 1.0:
            img_in = cv2.resize(
                img_bgr, (0, 0),
                fx=self.resize_factor, fy=self.resize_factor,
                interpolation=cv2.INTER_AREA
            )
            fx = fy = self.resize_factor
        else:
            img_in = img_bgr
            fx = fy = 1.0

        # Inference (no grad)
        kwargs = dict(
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device != "cpu":
            kwargs["device"] = self.device
            kwargs["half"] = self.use_half

        results = self.model.predict(img_in, **kwargs)
        r = results[0]
        boxes = r.boxes

        # Rescale boxes back to original frame space if we resized input
        self._rescale_boxes(boxes, fx, fy)

        # Cache
        with self._lock:
            self._last_boxes = boxes

    def _maybe_start_infer(self, frame_bgr):
        if self._infer_busy:
            return
        self._infer_busy = True

        def _task(img_copy):
            try:
                self._predict_one(img_copy)
            finally:
                self._infer_busy = False

        # Use a daemon thread; drop late frames by always using the latest image copy
        threading.Thread(target=_task, args=(frame_bgr.copy(),), daemon=True).start()

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        self._frame_id += 1

        # Start async inference every Nth frame
        if (self._frame_id % self._frame_skip) == 0:
            self._maybe_start_infer(img_bgr)

        # Draw using last available detections (may be 1–2 frames old)
        with self._lock:
            boxes = self._last_boxes

        if self.draw_mode.startswith("Ultra"):
            # Avoid re-running Ultralytics .plot() (it would re-infer).
            # Keep fast draw for speed.
            out = self._draw_fast(img_bgr.copy(), boxes)
        else:
            out = self._draw_fast(img_bgr.copy(), boxes)

        # FPS overlay
        if self.show_fps:
            self._cnt += 1
            if self._cnt >= 15:
                t1 = time.time()
                self._fps = self._cnt / (t1 - self._t0)
                self._t0, self._cnt = t1, 0
            cv2.putText(out, f"{self._fps:.1f} FPS", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ───────── Capture constraints ─────────
capture_constraints = {
    "video": {
        # HQ asks for HD; Standard asks for SD-ish
        "width":  {"ideal": 1280, "min": 960} if hq else {"ideal": 640},
        "height": {"ideal": 720,  "min": 540} if hq else {"ideal": 480},
        "frameRate": {"ideal": int(req_fps), "max": 60},
        # "facingMode": {"ideal": "environment"},  # uncomment for rear cam on mobile
    },
    "audio": False,
}

# ───────── WebRTC streamer ─────────
ctx = webrtc_streamer(
    key="yolo-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints=capture_constraints,
    video_processor_factory=lambda: YOLOProcessor(
        init_conf=conf,
        init_imgsz=imgsz,
        init_resize=resize_factor,
        init_draw=draw_mode,
        show_fps=show_fps,
        hq=hq,
    ),
    async_processing=True,  # let streamlit-webrtc pipeline run async
    video_html_attrs={
        "style": {"width": f"{preview_width}%", "margin": "auto"},
        "controls": False,
        "autoPlay": True,
        "muted": True,
        "playsInline": True,
    },
)

# Live-update processor from sidebar
if ctx and ctx.video_processor:
    vp = ctx.video_processor
    vp.conf = float(conf)
    vp.imgsz = int(imgsz)
    vp.resize_factor = float(resize_factor)
    vp.draw_mode = draw_mode
    vp.show_fps = bool(show_fps)
