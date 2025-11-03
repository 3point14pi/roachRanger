# app.py (hardened)
# Streamlit YOLO live tab + safe Upload→Split→Rebuild pipeline
# Key fixes:
# - Live tab can be paused to free resources during uploads
# - Strict caps on file size, duration, and frame count
# - Default path avoids writing frames/segments; builds final MP4 directly
# - Optional frames/segments use JPEG (smaller) and are zipped ONLY on demand
# - Outputs served from memory (BytesIO / st.session_state), not open file handles
# - Aggressive cleanup + gc to avoid OOM and dangling tempdirs

import os
import io
import gc
import time
import math
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import av
import numpy as np
import streamlit as st

# Optional: comment out if not using live tab
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# If you need YOLO for live tab
try:
    from model_utils import load_yolo_model
except Exception:
    load_yolo_model = None

# ───────────────────────────────────────────────────────────────────────────────
# Page
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cockroach Detection • Live + Upload (Hardened)", layout="wide")
st.title("🐞 Cockroach Detection • 📡 Live (WebRTC) + 🎞️ Upload → Split → Rebuild")

# Global perf hints
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass

# Session defaults
if "pause_live" not in st.session_state:
    st.session_state.pause_live = False
if "final_mp4_bytes" not in st.session_state:
    st.session_state.final_mp4_bytes = None
if "frames_zip_bytes" not in st.session_state:
    st.session_state.frames_zip_bytes = None
if "segments_zip_bytes" not in st.session_state:
    st.session_state.segments_zip_bytes = None

# Resource caps (tune for your host)
MAX_UPLOAD_MB = 300  # raw file size cap
MAX_FRAMES = 3000    # safety cap for frame extraction
MAX_SECONDS = 180    # duration cap used if we can estimate

# Tabs
live_tab, upload_tab = st.tabs(["📡 Live (WebRTC + YOLO)", "📼 Upload → Split → Rebuild"])

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _probe_video(path: str) -> Tuple[int, int, float, int]:
    """Return (width, height, fps, nb_frames_est). nb_frames_est may be 0."""
    with av.open(path) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        if not vstreams:
            raise RuntimeError("No video stream found.")
        vs = vstreams[0]
        width = int(vs.codec_context.width)
        height = int(vs.codec_context.height)
        fps = float(vs.average_rate) if vs.average_rate else (vs.base_rate and float(vs.base_rate)) or 30.0
        duration = float(vs.duration * vs.time_base) if (vs.duration and vs.time_base) else None
        nb_est = int(round(duration * fps)) if duration else int(vs.frames or 0)
        return width, height, fps, nb_est

def _safe_fourcc(preference: str = "avc1"):
    if preference == "mp4v":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if preference == "H264-fallback":
        return cv2.VideoWriter_fourcc(*"H264")
    return cv2.VideoWriter_fourcc(*"avc1")  # default


# Direct rebuild path: decode input and write out to final (no frames/segments on disk)

def _direct_reencode(in_path: Path, out_path: Path, fps: float, size: Tuple[int, int], codec_pref: str = "avc1"):
    fourcc = _safe_fourcc(codec_pref)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter for final output.")

    cap = cv2.VideoCapture(str(in_path))
    try:
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            if (frm.shape[1], frm.shape[0]) != size:
                frm = cv2.resize(frm, size, interpolation=cv2.INTER_AREA)
            vw.write(frm)
    finally:
        cap.release()
        vw.release()


def _extract_frames_jpeg(in_path: Path, out_dir: Path, every_n: int, pbar=None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with av.open(str(in_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if (i % every_n) != 0:
                continue
            img = frame.to_ndarray(format="bgr24")
            fname = out_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved += 1
            if pbar and (saved % 10 == 0):
                denom = max(1, stream.frames or (saved * every_n))
                pbar.progress(min(1.0, (i + 1) / denom))
            if saved >= MAX_FRAMES:
                break
    return saved


def _split_frames_into_segments(frames_dir: Path, fps: float, chunk_sec: int, codec_pref: str = "avc1") -> List[Path]:
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return []
    size_img = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    size = (size_img.shape[1], size_img.shape[0])
    frames_per_chunk = max(1, int(round(fps * chunk_sec)))
    segments = []
    fourcc = _safe_fourcc(codec_pref)

    chunk_idx = 0
    vw = None
    count_in_chunk = 0
    for i, fp in enumerate(frames):
        if vw is None:
            seg_path = frames_dir.parent / f"segment_{chunk_idx:03d}.mp4"
            vw = cv2.VideoWriter(str(seg_path), fourcc, fps, size)
            if not vw.isOpened():
                raise RuntimeError("Failed to open VideoWriter for segment.")
            segments.append(seg_path)
            count_in_chunk = 0

        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        vw.write(img)
        count_in_chunk += 1

        if count_in_chunk >= frames_per_chunk:
            vw.release()
            vw = None
            chunk_idx += 1
            count_in_chunk = 0

    if vw is not None:
        vw.release()

    return segments


def _concat_segments_to_final(segments: List[Path], fps: float, out_path: Path, codec_pref: str = "avc1"):
    if not segments:
        raise RuntimeError("No segments to concatenate.")
    cap0 = cv2.VideoCapture(str(segments[0]))
    ok, frame0 = cap0.read()
    cap0.release()
    if not ok or frame0 is None:
        raise RuntimeError("Failed to read first segment to infer size.")
    size = (frame0.shape[1], frame0.shape[0])

    fourcc = _safe_fourcc(codec_pref)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter for final output.")

    try:
        for seg in segments:
            cap = cv2.VideoCapture(str(seg))
            while True:
                ok, frm = cap.read()
                if not ok:
                    break
                if (frm.shape[1], frm.shape[0]) != size:
                    frm = cv2.resize(frm, size, interpolation=cv2.INTER_AREA)
                vw.write(frm)
            cap.release()
    finally:
        vw.release()


def _to_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ───────────────────────────────────────────────────────────────────────────────
# 📡 Live Tab (toggle-able)
# ───────────────────────────────────────────────────────────────────────────────
with live_tab:
    st.sidebar.header("Global Controls")
    st.sidebar.checkbox("Pause Live tab while processing Upload tab", value=st.session_state.pause_live, key="pause_live")

    if not st.session_state.pause_live:
        st.subheader("Live (WebRTC + optional YOLO)")
        enable_yolo = st.checkbox("Enable YOLO inference (requires model_utils)", value=False)
        preview_width = st.slider("Preview width (%)", 30, 100, 65, 5)
        req_fps = st.selectbox("Requested camera FPS", [15, 24, 30, 60], index=2)
        hq = st.toggle("High-Quality mode (GPU recommended)", value=False)
        draw_labels = st.toggle("Show labels", value=False)
        show_fps = st.toggle("Show FPS overlay", True)

        if enable_yolo and not load_yolo_model:
            st.warning("YOLO not available (model_utils import failed). Live stream will run without inference.")
            enable_yolo = False

        class YOLOProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = load_yolo_model() if (enable_yolo and load_yolo_model) else None
                self.conf = 0.5
                self.imgsz = 640
                self.resize_factor = 1.0
                self.draw_labels = draw_labels
                self.show_fps = show_fps
                self._lock = threading.Lock()
                self._infer_busy = False
                self._last_boxes_xyxy: List[Tuple[int, int, int, int]] = []
                self._last_labels: List[str] = []
                self._t0, self._cnt, self._fps = time.time(), 0, 0.0

            def _predict_once(self, img_bgr):
                if not self.model:
                    return [], []
                t0 = time.time()
                results = self.model.predict(img_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)
                r = results[0]
                boxes = r.boxes
                names = getattr(getattr(self.model, "names", None), "values", None) or getattr(self.model, "names", None)
                xyxy_list, labels = [], []
                if boxes is not None:
                    for i in range(len(boxes)):
                        b = boxes[i]
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cls = int(b.cls[0]) if b.cls is not None else -1
                        conf = float(b.conf[0]) if b.conf is not None else 0.0
                        lab = f"{names[cls]} {conf:.2f}" if (cls >= 0 and names and self.draw_labels) else None
                        xyxy_list.append((x1, y1, x2, y2))
                        labels.append(lab)
                return xyxy_list, labels

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                boxes_xyxy, labels = self._predict_once(img)
                out = img.copy()
                for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
                    cv2.rectangle(out, (x1, y1), (x2, y2), (50, 220, 50), 2, lineType=cv2.LINE_AA)
                    if lab:
                        cv2.putText(out, lab, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if self.show_fps:
                    self._cnt += 1
                    if self._cnt >= 15:
                        t1 = time.time()
                        self._fps = self._cnt / (t1 - self._t0)
                        self._t0, self._cnt = t1, 0
                    cv2.putText(out, f"{self._fps:.1f} FPS", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return av.VideoFrame.from_ndarray(out, format="bgr24")

        facing = "environment"
        capture_constraints = {
            "video": {
                "width":  {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": int(req_fps), "max": 60},
                "facingMode": {"ideal": facing},
            },
            "audio": False,
        }

        webrtc_streamer(
            key="yolo-live",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints=capture_constraints,
            video_processor_factory=YOLOProcessor,
            async_processing=True,
            video_html_attrs={
                "style": {"width": f"{preview_width}%", "margin": "auto"},
                "controls": False,
                "autoPlay": True,
                "muted": True,
                "playsInline": True,
            },
        )
    else:
        st.info("Live tab is paused to free resources for uploads.")

# ───────────────────────────────────────────────────────────────────────────────
# 📼 Upload → Split → Rebuild (Hardened)
# ───────────────────────────────────────────────────────────────────────────────
with upload_tab:
    st.subheader("Upload a video, (optionally) split into frames/segments, and rebuild a final MP4")

    # Safety toggles
    hard_pause_live = st.checkbox("Automatically pause Live tab during processing", value=True)

    c1, c2 = st.columns(2)
    with c1:
        every_n = st.number_input("Save every Nth frame (1 = every frame)", min_value=1, max_value=100, value=1, step=1)
        chunk_sec = st.number_input("Segment length (seconds) for splitting", min_value=1, max_value=600, value=10, step=1)
        want_frames = st.toggle("Also export frames (JPEG)", value=False)
        want_segments = st.toggle("Also export segments (MP4)", value=False)
    with c2:
        codec_choice = st.selectbox("Output codec (container .mp4)", ["avc1", "mp4v", "H264-fallback"], index=0)
        final_basename = st.text_input("Final output base filename (no extension)", value="rebuilt_video")
        make_zip_frames = st.toggle("Make ZIP of frames (on demand)", value=False)
        make_zip_segments = st.toggle("Make ZIP of segments (on demand)", value=False)

    uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "m4v", "avi", "mkv"], accept_multiple_files=False)

    # On-demand ZIP buttons appear after a successful run
    if st.session_state.final_mp4_bytes:
        st.download_button("🎬 ⬇️ Download rebuilt video (MP4)", st.session_state.final_mp4_bytes, file_name=f"{final_basename}.mp4", mime="video/mp4")
    if st.session_state.frames_zip_bytes:
        st.download_button("⬇️ Download frames (ZIP)", st.session_state.frames_zip_bytes, file_name="frames.zip", mime="application/zip")
    if st.session_state.segments_zip_bytes:
        st.download_button("⬇️ Download segments (ZIP)", st.session_state.segments_zip_bytes, file_name="segments.zip", mime="application/zip")

    # Run pipeline
    if uploaded is not None:
        st.success(f"Uploaded: {uploaded.name}")
        run = st.button("Run processing")
        if run:
            # Pause live to free resources
            if hard_pause_live:
                st.session_state.pause_live = True
                st.rerun()

            # Clear previous outputs from memory
            st.session_state.final_mp4_bytes = None
            st.session_state.frames_zip_bytes = None
            st.session_state.segments_zip_bytes = None

            with st.spinner("Processing video…"):
                with tempfile.TemporaryDirectory() as td:
                    tdir = Path(td)
                    in_path = tdir / uploaded.name

                    # Size guard
                    if hasattr(uploaded, "size") and uploaded.size and uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                        st.error(f"File too large (> {MAX_UPLOAD_MB} MB). Please upload a smaller clip.")
                        st.stop()

                    # Write input
                    with open(in_path, "wb") as f:
                        f.write(uploaded.read())

                    # Probe
                    width, height, fps, nb_est = _probe_video(str(in_path))
                    est_secs = (nb_est / fps) if (nb_est and fps) else None
                    st.write(f"Detected: **{width}×{height} @ {fps:.3f} fps**  (frames≈{nb_est or 'unknown'})")

                    if est_secs and est_secs > MAX_SECONDS:
                        st.error(f"Clip too long (> {MAX_SECONDS}s). Please trim and retry.")
                        st.stop()

                    # Default: direct re-encode → final
                    final_path = tdir / f"{final_basename}.mp4"
                    _direct_reencode(in_path, final_path, fps=fps, size=(width, height), codec_pref=codec_choice)

                    frames_dir = None
                    seg_paths: List[Path] = []

                    # Optional frames/segments
                    if want_frames or want_segments:
                        frames_dir = tdir / "frames"
                        pbar = st.progress(0.0, text="Extracting frames (JPEG)…")
                        saved = _extract_frames_jpeg(in_path, frames_dir, every_n=int(every_n), pbar=pbar)
                        pbar.progress(1.0)
                        st.write(f"Saved **{saved}** JPEG frame(s). (Capped at {MAX_FRAMES})")

                        if want_segments:
                            st.write("Creating segments…")
                            seg_paths = _split_frames_into_segments(frames_dir, fps=fps, chunk_sec=int(chunk_sec), codec_pref=codec_choice)
                            st.write(f"Created **{len(seg_paths)}** segment file(s).")

                    # Prepare download bytes BEFORE leaving tempdir
                    st.session_state.final_mp4_bytes = _to_bytes(final_path)

                    if make_zip_frames and frames_dir and frames_dir.exists():
                        frames_zip = tdir / "frames.zip"
                        shutil.make_archive(str(frames_zip.with_suffix('')), 'zip', frames_dir)
                        st.session_state.frames_zip_bytes = _to_bytes(frames_zip)

                    if make_zip_segments and seg_paths:
                        seg_root = tdir / "segments"
                        seg_root.mkdir(exist_ok=True)
                        for p in seg_paths:
                            shutil.copy2(p, seg_root / p.name)
                        seg_zip = tdir / "segments.zip"
                        shutil.make_archive(str(seg_zip.with_suffix('')), 'zip', seg_root)
                        st.session_state.segments_zip_bytes = _to_bytes(seg_zip)

                # Temp directory is gone now; only in-memory bytes remain
                gc.collect()

            st.success("Done. Downloads below.")
            st.experimental_rerun()  # Show download buttons fed from session_state
