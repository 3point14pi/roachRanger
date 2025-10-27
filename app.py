# app.py
# Streamlit WebRTC + Ultralytics YOLO live detection with adaptive cadence & optional tracking
import time
import threading
from typing import List, Tuple, Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# You must provide this util. Typical impl: from ultralytics import YOLO; return YOLO("yolov8n.pt")
from model_utils import load_yolo_model


# ─────────────────────────── Page / Sidebar ───────────────────────────
st.set_page_config(page_title="Live Cockroach Detection", layout="wide")
st.title("🐞 Live Cockroach Detection (WebRTC + YOLO, Adaptive + Tracking)")

st.sidebar.header("Controls")

camera_mode = st.sidebar.selectbox("Camera", ["Back (environment)", "Front (user)"], index=0)
preview_width = st.sidebar.slider("Preview width (%)", 30, 100, 70, 5)
req_fps = st.sidebar.selectbox("Requested camera FPS", [15, 24, 30, 60], index=2)

hq = st.sidebar.toggle("High-Quality mode (GPU recommended)", value=False)
draw_labels = st.sidebar.toggle("Show labels", value=True)

if hq:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [480, 640, 800], index=1)
    resize_factor = 1.0
else:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.65, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640], index=1)
    resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 0.7, 0.05)

use_tracking = st.sidebar.toggle("Use tracker between detections", value=True)
show_fps = st.sidebar.toggle("Show FPS overlay", True)

st.sidebar.caption("Tip: try yolov8n, imgsz 320–480, pre-resize ~0.5–0.75, and higher conf (≥0.65) on CPU.")

# Keep OpenCV light on smaller CPUs
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass


# ─────────────────────────── Helpers ───────────────────────────
def make_tracker():
    # Try modern constructors
    for ctor in ("TrackerKCF_create", "TrackerCSRT_create", "TrackerMOSSE_create"):
        if hasattr(cv2, ctor):
            return getattr(cv2, ctor)()
    # Legacy fallback
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except Exception:
        return None


def draw_boxes(img: np.ndarray, boxes_xyxy: List[Tuple[int, int, int, int]], labels: Optional[List[Optional[str]]] = None):
    if not boxes_xyxy:
        return img
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        cv2.rectangle(img, (x1, y1), (x2, y2), (50, 220, 50), 2, lineType=cv2.LINE_AA)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(img, labels[i], (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def clip_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


# ─────────────────────────── Video Processor ───────────────────────────
class YOLOProcessor(VideoProcessorBase):
    """
    Streamlit WebRTC video processor:
      • Adaptive YOLO inference cadence based on model runtime & requested FPS
      • Frame dropping to avoid backpressure
      • Optional GPU (half precision + fuse) when available
      • Optional OpenCV tracking between detections for smoother box updates
    """
    def __init__(self, init_conf, init_imgsz, init_resize, show_fps, draw_labels, hq, req_fps, use_tracking):
        # Load model
        self.model = load_yolo_model()
        self.conf = float(init_conf)
        self.imgsz = int(init_imgsz)
        self.resize_factor = float(init_resize)
        self.show_fps = bool(show_fps)
        self.draw_labels = bool(draw_labels)
        self.hq = bool(hq)
        self.req_fps = int(req_fps)
        self.use_tracking = bool(use_tracking)

        # Device / half / fuse (Ultralytics-friendly)
        self.device = "cpu"
        self.use_half = False
        self.torch = None
        try:
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self.model.to("cuda")
                torch.backends.cudnn.benchmark = True
                self.device = 0
                try:
                    self.model.fuse()
                except Exception:
                    pass
                try:
                    # Some models expose .model, others handle half via arg; both are fine
                    self.model.model.half()  # type: ignore[attr-defined]
                    self.use_half = True
                except Exception:
                    self.use_half = False
        except Exception:
            pass

        # Async inference state
        self._lock = threading.Lock()
        self._infer_busy = False
        self._last_det_time = 0.05  # seed ~20 det FPS
        self._target_det_fps = 20 if self.use_half else (12 if self.hq else 10)
        self._frame_skip = 1
        self._frame_id = 0

        # Last detections
        self._last_boxes_xyxy: List[Tuple[int, int, int, int]] = []
        self._last_labels: List[Optional[str]] = []

        # Trackers
        self._trackers = []
        self._track_labels: List[Optional[str]] = []

        # FPS meter (render FPS)
        self._t0, self._cnt, self._fps = time.time(), 0, 0.0

        # Last error (rendered on overlay to aid debugging)
        self._last_err: Optional[str] = None

    # ── cadence control ──
    def _adapt_frame_skip(self):
        if self.req_fps <= 0:
            self._frame_skip = 1
            return
        det_fps = 1.0 / max(self._last_det_time, 1e-6)
        if det_fps <= 0:
            self._frame_skip = max(1, int(self.req_fps / max(1.0, self._target_det_fps)))
            return
        ratio = self._target_det_fps / det_fps
        desired_every = int(round(self.req_fps / max(1.0, self._target_det_fps * max(0.5, min(2.0, ratio)))))
        self._frame_skip = max(1, desired_every)

    # ── tensor-safe box extraction ──
    def _boxes_to_xyxy_and_labels(self, boxes, names):
        """Return ([(x1,y1,x2,y2), ...], [label_or_None, ...]) in image coordinates."""
        xyxy_list: List[Tuple[int, int, int, int]] = []
        labels: List[Optional[str]] = []
        if boxes is None or len(boxes) == 0:
            return xyxy_list, labels

        # Ultralytics Boxes fields are torch tensors; convert once
        xyxy = boxes.xyxy
        cls_t = boxes.cls
        conf_t = boxes.conf

        # Handle numpy vs torch gracefully
        def to_np(t):
            if t is None:
                return None
            return t.detach().cpu().numpy() if hasattr(t, "device") else np.asarray(t)

        xyxy_np = to_np(xyxy)
        cls_np = to_np(cls_t)
        conf_np = to_np(conf_t)

        for i in range(int(xyxy_np.shape[0])):
            x1, y1, x2, y2 = [int(v) for v in xyxy_np[i].tolist()]
            label = None
            if self.draw_labels and cls_np is not None and conf_np is not None and names is not None:
                ci = int(cls_np[i])
                cf = float(conf_np[i])
                name = names[ci] if isinstance(names, (dict, list)) else str(ci)
                label = f"{name} {cf:.2f}"
            xyxy_list.append((x1, y1, x2, y2))
            labels.append(label)
        return xyxy_list, labels

    def _start_trackers(self, img_bgr, boxes_xyxy, labels):
        self._trackers, self._track_labels = [], []
        if not self.use_tracking:
            return
        for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
            trk = make_tracker()
            if trk is None:
                continue
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            if trk.init(img_bgr, (x1, y1, w, h)):
                self._trackers.append(trk)
                self._track_labels.append(lab)

    def _update_trackers(self, img_bgr, frame_w, frame_h):
        if not self.use_tracking or not self._trackers:
            return [], []
        new_boxes, alive_trackers, alive_labels = [], [], []
        for trk, lab in zip(self._trackers, self._track_labels):
            ok, box = trk.update(img_bgr)
            if not ok:
                continue
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            x1, y1, x2, y2 = clip_xyxy(x1, y1, x2, y2, frame_w, frame_h)
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            new_boxes.append((x1, y1, x2, y2))
            alive_trackers.append(trk)
            alive_labels.append(lab)
        self._trackers, self._track_labels = alive_trackers, alive_labels
        return new_boxes, alive_labels

    # ── one detection pass ──
    def _predict_once(self, img_bgr):
        fx = fy = 1.0
        img_in = img_bgr
        if 0.4 <= self.resize_factor < 1.0:
            img_in = cv2.resize(img_bgr, (0, 0), fx=self.resize_factor, fy=self.resize_factor,
                                interpolation=cv2.INTER_AREA)
            fx = fy = self.resize_factor

        kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device != "cpu":
            kwargs["device"] = self.device
            kwargs["half"] = self.use_half

        t0 = time.time()
        results = self.model.predict(img_in, **kwargs)
        self._last_det_time = max(1e-6, time.time() - t0)

        r = results[0]
        boxes = r.boxes
        # Ultralytics stores names as list or dict on model.names
        names = getattr(self.model, "names", None)

        boxes_xyxy, labels = self._boxes_to_xyxy_and_labels(boxes, names)

        # rescale to the original frame size when we pre-resized
        if fx != 1.0 or fy != 1.0:
            boxes_xyxy = [(int(x1 / fx), int(y1 / fy), int(x2 / fx), int(y2 / fy)) for (x1, y1, x2, y2) in boxes_xyxy]

        return boxes_xyxy, labels

    def _maybe_start_infer(self, frame_bgr):
        if self._infer_busy:
            return
        self._infer_busy = True

        def _task(bgr_copy):
            try:
                boxes_xyxy, labels = self._predict_once(bgr_copy)
                with self._lock:
                    self._last_boxes_xyxy = boxes_xyxy
                    self._last_labels = labels
                self._start_trackers(bgr_copy, boxes_xyxy, labels)
                self._adapt_frame_skip()
                self._last_err = None
            except Exception as e:
                # Record an error so we can show it on the overlay — helps debugging
                self._last_err = f"{type(e).__name__}: {e}"
            finally:
                self._infer_busy = False

        threading.Thread(target=_task, args=(frame_bgr.copy(),), daemon=True).start()

    # ── main callback ──
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self._frame_id += 1
        h, w = img.shape[:2]

        if (self._frame_id % self._frame_skip) == 0:
            self._maybe_start_infer(img)

        boxes_xyxy, labels = self._update_trackers(img, w, h)

        if not boxes_xyxy:
            with self._lock:
                boxes_xyxy = list(self._last_boxes_xyxy)
                labels = list(self._last_labels)

        out = draw_boxes(img.copy(), boxes_xyxy, labels if self.draw_labels else None)

        # FPS overlay
        if self.show_fps:
            self._cnt += 1
            if self._cnt >= 15:
                t1 = time.time()
                self._fps = self._cnt / (t1 - self._t0)
                self._t0, self._cnt = t1, 0
            overlay = f"{self._fps:.1f} FPS  det~{1.0 / max(self._last_det_time, 1e-6):.1f} FPS  skip:{self._frame_skip}"
            cv2.putText(out, overlay, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if self._last_err:
                cv2.putText(out, self._last_err[:80], (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (64, 200, 255), 2)

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ─────────────────────────── WebRTC Constraints ───────────────────────────
facing = "environment" if camera_mode.startswith("Back") else "user"
capture_constraints = {
    "video": {
        "width": {"ideal": 1280, "min": 960} if hq else {"ideal": 640},
        "height": {"ideal": 720, "min": 540} if hq else {"ideal": 480},
        "frameRate": {"ideal": int(req_fps), "max": 60},
        "facingMode": {"ideal": facing},  # pick back/front cam on phones
    },
    "audio": False,
}

# ─────────────────────────── Streamer ───────────────────────────
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
        use_tracking=use_tracking,
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

# Live updates while running
if ctx and ctx.video_processor:
    vp = ctx.video_processor
    vp.conf = float(conf)
    vp.imgsz = int(imgsz)
    vp.resize_factor = float(resize_factor)
    vp.draw_labels = bool(draw_labels)
    vp.hq = bool(hq)
    vp.req_fps = int(req_fps)
    vp.use_tracking = bool(use_tracking)
