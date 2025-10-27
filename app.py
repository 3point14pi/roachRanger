# app.py
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
    imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640], index=1)
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
    - Adaptive inference cadence based on measured model time & requested FPS
    - Latest-frame dropping to prevent backpressure (no 1 FPS collapse)
    - Optional GPU half precision + layer fusion
    - Multi-object tracking (OpenCV) between detections for smooth box updates
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
        try:
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self.model.to("cuda")
                torch.backends.cudnn.benchmark = True
                self.device = 0
                try:
                    self.model.fuse()  # conv+bn fusion
                except Exception:
                    pass
                try:
                    self.model.model.half()
                    self.use_half = True
                except Exception:
                    self.use_half = False
            else:
                self.torch = torch
        except Exception:
            self.torch = None

        # Async inference state
        self._lock = threading.Lock()
        self._infer_busy = False
        self._last_det_time = 0.05  # seconds; seed ~20 FPS
        self._target_det_fps = 20 if self.use_half else (12 if self.hq else 10)
        self._frame_skip = 1  # will adapt on the fly
        self._frame_id = 0

        # Last detections (xyxy, labels)
        self._last_boxes_xyxy: List[Tuple[int, int, int, int]] = []
        self._last_labels: List[str] = []

        # Tracking between detections
        self._trackers = []  # list of cv2 trackers
        self._track_labels = []

        # FPS meter (render)
        self._t0, self._cnt, self._fps = time.time(), 0, 0.0

    def _adapt_frame_skip(self):
        """Adapt how often we run YOLO to reach target detection FPS."""
        if self.req_fps <= 0:
            self._frame_skip = 1
            return
        # If detections take longer, increase skip; shorter → reduce skip
        # aim: det_fps ~= target_det_fps
        est_det_fps = 1.0 / max(self._last_det_time, 1e-6)
        ratio = max(0.5, min(2.0, self._target_det_fps / max(1e-6, est_det_fps)))
        desired_det_every = max(1, int(round(self.req_fps / max(1.0, self._target_det_fps * ratio))))
        self._frame_skip = max(1, desired_det_every)

    def _rescale_boxes_inplace(self, boxes, fx, fy):
        if boxes is None or len(boxes) == 0 or (fx == 1.0 and fy == 1.0):
            return boxes
        xyxy = boxes.xyxy  # tensor Nx4
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] / fx
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] / fy
        return boxes

    # ---------- CHANGED: robust extraction, same signature ----------
    def _boxes_to_int_xyxy_and_labels(self, boxes, names):
        """
        Robustly convert Ultralytics Boxes -> (list[int xyxy], list[str|None] labels).
        Keeps the same signature and return type as your original function.
        """
        if boxes is None or len(boxes) == 0:
            return [], []

        # Get tensors/arrays
        xyxy = getattr(boxes, "xyxy", None)
        cls = getattr(boxes, "cls", None)
        conf = getattr(boxes, "conf", None)

        if xyxy is None:
            return [], []

        # to numpy safely
        try:
            xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.array(xyxy)
        except Exception:
            return [], []

        out_boxes, out_labels = [], []
        n = xyxy_np.shape[0]

        # Normalize names (support dict or list)
        norm_names = names
        if isinstance(norm_names, dict):
            try:
                max_k = max(norm_names.keys())
                if all(k in norm_names for k in range(max_k + 1)):
                    norm_names = [norm_names[k] for k in range(max_k + 1)]
            except Exception:
                pass

        for i in range(n):
            x1, y1, x2, y2 = map(int, xyxy_np[i].tolist())
            out_boxes.append((x1, y1, x2, y2))

            label = None
            if self.draw_labels and cls is not None and conf is not None:
                try:
                    c = int(cls[i].item() if hasattr(cls[i], "item") else cls[i])
                    p = float(conf[i].item() if hasattr(conf[i], "item") else conf[i])
                    if isinstance(norm_names, (list, tuple)) and 0 <= c < len(norm_names):
                        label = f"{norm_names[c]} {p:.2f}"
                    elif isinstance(norm_names, dict) and c in norm_names:
                        label = f"{norm_names[c]} {p:.2f}"
                    else:
                        label = f"id{c} {p:.2f}"
                except Exception:
                    label = None
            out_labels.append(label)

        return out_boxes, out_labels
    # ---------- end change ----------

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

    # ---------- CHANGED: names access only; rest same ----------
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
        self._last_det_time = max(1e-6, time.time() - t0)

        r = results[0]
        boxes = r.boxes

        # Use the model's names directly (handles dict or list)
        names = getattr(self.model, "names", None)

        # Keep your original rescale path
        self._rescale_boxes_inplace(boxes, fx, fy)
        boxes_xyxy, labels = self._boxes_to_int_xyxy_and_labels(boxes, names)
        return boxes_xyxy, labels
    # ---------- end change ----------

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

        # FPS overlay (render FPS, not detection FPS)
        if self.show_fps:
            self._cnt += 1
            if self._cnt >= 15:
                t1 = time.time()
                self._fps = self._cnt / (t1 - self._t0)
                self._t0, self._cnt = t1, 0
            cv2.putText(out, f"{self._fps:.1f} FPS  det~{1.0/max(self._last_det_time,1e-6):.1f} FPS  skip:{self._frame_skip}",
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
