# app.py
import time
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from model_utils import load_yolo_model

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Cockroach Detection", layout="wide")
st.title("🐞 Live Cockroach Detection (WebRTC + YOLO)")

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("Controls")

# Quality vs speed master switch
hq = st.sidebar.toggle("High-Quality mode (GPU recommended)", value=False)

# Capture / preview
preview_width = st.sidebar.slider("Preview width (%)", 30, 100, 60, 5)
req_fps = st.sidebar.selectbox("Requested camera FPS", [15, 24, 30, 60], index=2)

# Inference settings
if hq:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [480, 640, 800], index=1)
    resize_factor = 1.0  # no pre-resize in HQ
    draw_mode = st.sidebar.selectbox("Drawing", ["Ultralytics .plot()", "Fast boxes"], index=0)
else:
    conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.60, 0.05)
    imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640], index=1)
    resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 0.75, 0.05)
    draw_mode = st.sidebar.selectbox("Drawing", ["Fast boxes", "Ultralytics .plot()"], index=0)

show_fps = st.sidebar.toggle("Show FPS overlay", True)
apply_filters = st.sidebar.toggle("Denoise + sharpen (visual)", value=hq)

st.sidebar.caption(
    "Tip: Smaller imgsz / higher conf / pre-resize ↑ = faster FPS. "
    "High-Quality mode requests HD capture and disables pre-resize."
)

# Keep OpenCV threads sane on small CPUs
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ── Fast / pretty drawing helpers ─────────────────────────────────────────────
def fast_draw(img_bgr, boxes):
    """Minimal, fast rectangle draw without labels."""
    if boxes is None or len(boxes) == 0:
        return img_bgr
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (50, 220, 50), 2, lineType=cv2.LINE_AA)
    return img_bgr

def pretty_filters(img_bgr):
    """Light denoise + unsharp + mild contrast for a cleaner preview (optional)."""
    img = cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 21)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)
    img = cv2.convertScaleAbs(img, alpha=1.08, beta=4)
    return img

# ── Video processor ───────────────────────────────────────────────────────────
class YOLOProcessor(VideoProcessorBase):
    def __init__(self, init_conf, init_imgsz, init_resize, init_draw, show_fps, use_filters, hq):
        self.model = load_yolo_model()

        # Try to use GPU + half precision when possible
        self.device = "cpu"
        self.use_half = False
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda")
                torch.backends.cudnn.benchmark = True
                self.device = 0  # CUDA device 0
                self.use_half = True
        except Exception:
            pass

        # Runtime state (updated live from sidebar)
        self.conf = init_conf
        self.imgsz = int(init_imgsz)
        self.resize_factor = float(init_resize)
        self.draw_mode = init_draw
        self.show_fps = bool(show_fps)
        self.use_filters = bool(use_filters)
        self.hq = bool(hq)

        # FPS meter
        self._t0, self._cnt, self._fps = time.time(), 0, 0.0

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        # Visual filters (do not affect detection; preview clarity only)
        if self.use_filters:
            img_bgr = pretty_filters(img_bgr)

        # Optional pre-resize BEFORE inference (skipped in HQ)
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

        # Draw
        if self.draw_mode.startswith("Ultra"):
            out = r.plot(line_width=3, labels=True, conf=True)  # prettier, a bit slower
        else:
            out = fast_draw(img_bgr.copy(), r.boxes)

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

# ── Choose camera constraints based on HQ toggle ──────────────────────────────
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

# ── WebRTC streamer ───────────────────────────────────────────────────────────
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
        use_filters=apply_filters,
        hq=hq,
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

# Live-update processor from sidebar
if ctx and ctx.video_processor:
    vp = ctx.video_processor
    vp.conf = conf
    vp.imgsz = int(imgsz)
    vp.resize_factor = float(resize_factor)
    vp.draw_mode = draw_mode
    vp.show_fps = bool(show_fps)
    vp.use_filters = bool(apply_filters)
