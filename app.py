# app.py — Live Cockroach Detection (WebRTC + YOLO) — fixed det~0.0fps + smarter skip
# -----------------------------------------------------------------------------
# Key fixes:
#  • One-time warmup inference to avoid first-call 0.0 FPS
#  • EMA smoothing of detection FPS (no more sticky 0.0)
#  • Smarter adaptive frame skipping (1–8) based on measured det time
#  • Optional pre-resize for inference; small default imgsz on CPU
#  • Back-camera selection on mobile via facingMode
#  • CPU thread limiting for OpenCV and PyTorch
#  • Robust trackers to keep high box refresh rate between detections
# -----------------------------------------------------------------------------

import time
import threading
from typing import List, Tuple

import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Your util that returns a preloaded ultralytics YOLO model (e.g., YOLO("yolov8n.pt"))
from model_utils import load_yolo_model

# ───────────────────────────────────────────────────────────────────────────────
# Page & Sidebar
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Cockroach Detection", layout="wide")
st.title("🐞 Live Cockroach Detection (WebRTC + YOLO, Adaptive + Tracking)")

st.sidebar.header("Controls")

# Camera
camera_mode = st.sidebar.selectbox("Camera", ["Back (environment)", "Front (user)"], index=0)
preview_width = st.sidebar.slider("Preview width (%)", 30, 100, 65, 5)
req_fps = st.sidebar.selectbox("Requested camera FPS", [15, 24, 30, 60], index=2)

# Speed/quality
hq = st.sidebar.toggle("High-Quality mode (GPU recommended)", value=False)
draw_labels = st.sidebar.toggle("Show labels", value=False)

if hq:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [480, 640, 800], index=1)
    resize_factor = 1.0  # no pre-resize in HQ
else:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.65, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640], index=0)
    resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 0.7, 0.05)

show_fps = st.sidebar.toggle("Show FPS overlay", True)

st.sidebar.caption(
    "Tips: Use a small model (e.g., yolov8n), imgsz 320–480, pre-resize ~0.5–0.75, and higher conf (0.65+)."
)

# Keep OpenCV light on small CPUs
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _make_tracker():
    # Prefer KCF → CSRT → MOSSE (fastest). Availability varies by build.
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    # Fallback to legacy API if needed
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except Exception:
        return None


def _draw_fast(img, boxes_xyxy: List[Tuple[int, int, int, int]], labels=None):
    if not boxes_xyxy:
        return img
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        cv2.rectangle(img, (x1, y1), (x2, y2), (50, 220, 50), 2, lineType=cv2.LINE_AA)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(img, labels[i], (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


# ───────────────────────────────────────────────────────────────────────────────
# Video Processor
# ───────────────────────────────────────────────────────────────────────────────
class YOLOProcessor(VideoProcessorBase):
    """
    Best-practice pipeline for Streamlit WebRTC:
    - Warmup inference to avoid first-call stalls
    - Adaptive inference cadence based on measured model time & requested FPS
    - Latest-frame dropping to prevent backpressure
    - Optional GPU half precision + layer fusion
    - Multi-object tracking between detections for smooth box updates
    """

    def __init__(self, init_conf, init_imgsz, init_resize, show_fps, draw_labels, hq, req_fps):
        # Model
        self.model = load_yolo_model()
        self.conf = float(init_conf)
        self.imgsz = int(init_imgsz)
        self.resize_factor = float(init_resize)
        self.show_fps = bool(show_fps)
        self.draw_labels = bool(draw_labels)
        self.hq = bool(hq)
        self.req_fps = int(req_fps)

        # Device / half / fuse
        self.device = "cpu"
        self.use_half = False
        self._ema_det = None  # EMA of det time

        try:
            import torch
            self.torch = torch
            # Limit CPU threads on small boxes
            try:
                self.torch.set_num_threads(1)
            except Exception:
                pass

            if torch.cuda.is_available():
                self.model.to("cuda")
                torch.backends.cudnn.benchmark = True
                self.device = 0
                try:
                    self.model.fuse()  # conv+bn fusion
                except Exception:
                    pass
                try:
                    # Not all wrappers expose .model.half(); ignore if missing
                    if hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                        self.model.model.half()
                        self.use_half = True
                except Exception:
                    self.use_half = False
        except Exception:
            self.torch = None

        # Async inference state
        self._lock = threading.Lock()
        self._infer_busy = False
        self._last_det_time = 0.05  # seconds; seed ~20 FPS
        self._target_det_fps = 20 if self.use_half else (12 if self.hq else 10)
        self._frame_skip = 2  # start slightly conservative
        self._frame_id = 0

        # Last detections (xyxy, labels)
        self._last_boxes_xyxy: List[Tuple[int, int, int, int]] = []
        self._last_labels: List[str] = []

        # Tracking between detections
        self._trackers = []  # list of cv2 trackers
        self._track_labels = []

        # FPS meter (render)
        self._t0, self._cnt, self._fps = time.time(), 0, 0.0

        # One-time warmup to avoid det~0.0 due to first-call latency
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
            if self.device != "cpu":
                kwargs["device"] = self.device
                kwargs["half"] = self.use_half
            _ = self.model.predict(dummy, **kwargs)
            self._last_det_time = 0.05
            self._ema_det = self._last_det_time
        except Exception:
            # If warmup fails, keep defaults; first live call will set times
            pass

    def _adapt_frame_skip(self):
        """Adapt how often we run YOLO to reach target detection FPS."""
        # Backoff more aggressively on weak hardware; cap skip to keep UX
        det_time = max(self._last_det_time, 1e-3)
        target_dt = 0.12 if self.hq else 0.14  # ~8–7 det/s targets
        # Desired number of frames to wait between detections based on timing
        desired = int(round((det_time / target_dt) * max(1, self.req_fps or 15)))
        self._frame_skip = int(max(1, min(8, desired)))

    def _rescale_boxes_inplace(self, boxes, fx, fy):
        if boxes is None or len(boxes) == 0 or (fx == 1.0 and fy == 1.0):
            return boxes
        xyxy = boxes.xyxy  # tensor Nx4
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] / fx
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] / fy
        return boxes

    def _boxes_to_int_xyxy_and_labels(self, boxes, names):
        xyxy_list, labels = [], []
        if boxes is None or len(boxes) == 0:
            return xyxy_list, labels
        for i in range(len(boxes)):
            b = boxes[i]
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls = int(b.cls[0]) if b.cls is not None else -1
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            lab = f"{names[cls]} {conf:.2f}" if (cls >= 0 and names and self.draw_labels) else None
            xyxy_list.append((x1, y1, x2, y2))
            labels.append(lab)
        return xyxy_list, labels

    def _(self):
        # niche trick to avoid accidental name shadowing in some REPLs
        return None

    def _start_trackers(self, img, boxes_xyxy, labels):
        self._trackers, self._track_labels = [], []
        for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
            trk = _make_tracker()
            if trk is None:
                continue
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            ok = trk.init(img, (x1, y1, w, h))
            if ok:
                self._trackers.append(trk)
                self._track_labels.append(lab)

    def _update_trackers(self, img, frame_w, frame_h):
        boxes_xyxy = []
        alive_trackers = []
        alive_labels = []
        for trk, lab in zip(self._trackers, self._track_labels):
            ok, box = trk.update(img)
            if not ok:
                continue
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, frame_w, frame_h)
            # reject too tiny boxes (tracker drift)
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            boxes_xyxy.append((x1, y1, x2, y2))
            alive_trackers.append(trk)
            alive_labels.append(lab)
        self._trackers = alive_trackers
        self._track_labels = alive_labels
        return boxes_xyxy, alive_labels

    def _predict_once(self, img_bgr):
        # Optional pre-resize for inference only
        if 0.4 <= self.resize_factor < 1.0:
            img_in = cv2.resize(img_bgr, (0, 0),
                                fx=self.resize_factor, fy=self.resize_factor,
                                interpolation=cv2.INTER_AREA)
            fx = fy = self.resize_factor
        else:
            img_in = img_bgr
            fx = fy = 1.0

        kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device != "cpu":
            kwargs["device"] = self.device
            kwargs["half"] = self.use_half

        t0 = time.time()
        results = self.model.predict(img_in, **kwargs)
        dt = max(1e-6, time.time() - t0)
        self._last_det_time = dt
        # Update EMA of detection time (smooths meter)
        self._ema_det = (0.9 * (self._ema_det if self._ema_det else dt)) + 0.1 * dt

        r = results[0]
        boxes = r.boxes
        names = getattr(getattr(self.model, "names", None), "values", None) or getattr(self.model, "names", None)

        self._rescale_boxes_inplace(boxes, fx, fy)
        boxes_xyxy, labels = self._boxes_to_int_xyxy_and_labels(boxes, names)
        return boxes_xyxy, labels

    def _maybe_start_infer(self, img_bgr):
        if self._infer_busy:
            return
        self._infer_busy = True

        def _task(frame_copy):
            try:
                boxes_xyxy, labels = self._predict_once(frame_copy)
                with self._lock:
                    self._last_boxes_xyxy = boxes_xyxy
                    self._last_labels = labels
                # Rebuild trackers on each fresh detection
                self._start_trackers(frame_copy, boxes_xyxy, labels)
                # Adapt cadence to match hardware
                self._adapt_frame_skip()
            finally:
                self._infer_busy = False

        threading.Thread(target=_task, args=(img_bgr.copy(),), daemon=True).start()

    # ───────────── main callback ─────────────
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self._frame_id += 1
        h, w = img.shape[:2]

        # Start adaptive detection at cadence (every Nth frame)
        if (self._frame_id % self._frame_skip) == 0:
            self._maybe_start_infer(img)

        # Between detections: track for high box refresh rate
        boxes_xyxy, labels = self._update_trackers(img, w, h)

        # If no trackers (first frames), draw last detections if any
        if not boxes_xyxy:
            with self._lock:
                boxes_xyxy = list(self._last_boxes_xyxy)
                labels = list(self._last_labels)

        out = _draw_fast(img.copy(), boxes_xyxy, labels if self.draw_labels else None)

        # FPS overlay (render FPS + smoothed detection FPS)
        if self.show_fps:
            self._cnt += 1
            if self._cnt >= 15:
                t1 = time.time()
                self._fps = self._cnt / (t1 - self._t0)
                self._t0, self._cnt = t1, 0
            det_fps_now = 1.0 / max(self._ema_det if self._ema_det else self._last_det_time, 1e-6)
            warming = (self._frame_id < 45 and det_fps_now < 1.0)
            label = "warming up…" if warming else f"det~{det_fps_now:.1f} FPS"
            cv2.putText(out, f"{self._fps:.1f} FPS  {label}  skip:{self._frame_skip}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ───────────────────────────────────────────────────────────────────────────────
# WebRTC capture constraints (request back camera on mobile)
# ───────────────────────────────────────────────────────────────────────────────
facing = "environment" if camera_mode.startswith("Back") else "user"

capture_constraints = {
    "video": {
        "width":  {"ideal": 1280, "min": 960} if hq else {"ideal": 640},
        "height": {"ideal": 720,  "min": 540} if hq else {"ideal": 480},
        "frameRate": {"ideal": int(req_fps), "max": 60},
        "facingMode": {"ideal": facing},  # back/front camera on phones
    },
    "audio": False,
}

# ───────────────────────────────────────────────────────────────────────────────
# Streamer
# ───────────────────────────────────────────────────────────────────────────────
ctx = webrtc_streamer(
    key="yolo-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints=capture_constraints,
    video_processor_factory=lambda: YOLOProcessor(
        init_conf=conf,
        init_imgsz=imgsz,
        init_resize=resize_factor,
        show_fps=show_fps,
        draw_labels=draw_labels,
        hq=hq,
        req_fps=req_fps,
    ),
    async_processing=True,
    video_html_attrs={
        "style": {"width": f"{preview_width}%", "margin": "auto"},
        "controls": False,
        "autoPlay": True,
        "muted": True,
        "playsInline": True,
    },
)

# Live updates from sidebar
if ctx and ctx.video_processor:
    vp = ctx.video_processor
    vp.conf = float(conf)
    vp.imgsz = int(imgsz)
    vp.resize_factor = float(resize_factor)
    vp.draw_labels = bool(draw_labels)
    vp.hq = bool(hq)
    vp.req_fps = int(req_fps)
