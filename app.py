# app.py
import os
import io
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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Your util that returns a preloaded ultralytics YOLO model (e.g., YOLO("yolov8n.pt"))
# If you don't need YOLO for the upload flow, this can stay as-is.
from model_utils import load_yolo_model

# ───────────────────────────────────────────────────────────────────────────────
# Page & Sidebar
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Cockroach Detection + Video Split/Rebuild", layout="wide")
st.title("🐞 Live Cockroach Detection (WebRTC + YOLO)  •  🎞️ Upload → Split → Rebuild")

tabs = st.tabs(["📡 Live (WebRTC + YOLO)", "📼 Upload → Split → Rebuild"])

# ───────────────────────────────────────────────────────────────────────────────
# Common perf hints for OpenCV
# ───────────────────────────────────────────────────────────────────────────────
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass

# ───────────────────────────────────────────────────────────────────────────────
# Helpers (drawing/tracking) for the live tab — unchanged from your base
# ───────────────────────────────────────────────────────────────────────────────
def _make_tracker():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
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
# Live (WebRTC + YOLO) tab
# ───────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.sidebar.header("Controls (Live)")
    camera_mode = st.sidebar.selectbox("Camera", ["Back (environment)", "Front (user)"], index=0)
    preview_width = st.sidebar.slider("Preview width (%)", 30, 100, 65, 5)
    req_fps = st.sidebar.selectbox("Requested camera FPS", [15, 24, 30, 60], index=2)

    hq = st.sidebar.toggle("High-Quality mode (GPU recommended)", value=False)
    draw_labels = st.sidebar.toggle("Show labels", value=False)

    if hq:
        conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
        imgsz = st.sidebar.selectbox("Inference size (px)", [480, 640, 800], index=1)
        resize_factor = 1.0
    else:
        conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.65, 0.05)
        imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640], index=1)
        resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 0.7, 0.05)

    show_fps = st.sidebar.toggle("Show FPS overlay", True)

    st.sidebar.caption(
        "Tips: Use a small model (e.g., yolov8n), imgsz 320–480, pre-resize ~0.5–0.75, and higher conf (0.65+)."
    )

    class YOLOProcessor(VideoProcessorBase):
        def __init__(self, init_conf, init_imgsz, init_resize, show_fps, draw_labels, hq, req_fps):
            self.model = load_yolo_model()
            self.conf = float(init_conf)
            self.imgsz = int(init_imgsz)
            self.resize_factor = float(init_resize)
            self.show_fps = bool(show_fps)
            self.draw_labels = bool(draw_labels)
            self.hq = bool(hq)
            self.req_fps = int(req_fps)

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
                        self.model.fuse()
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

            self._lock = threading.Lock()
            self._infer_busy = False
            self._last_det_time = 0.05
            self._target_det_fps = 20 if self.use_half else (12 if self.hq else 10)
            self._frame_skip = 1
            self._frame_id = 0

            self._last_boxes_xyxy: List[Tuple[int, int, int, int]] = []
            self._last_labels: List[str] = []

            self._trackers = []
            self._track_labels = []

            self._t0, self._cnt, self._fps = time.time(), 0, 0.0

        def _adapt_frame_skip(self):
            if self.req_fps <= 0:
                self._frame_skip = 1
                return
            est_det_fps = 1.0 / max(self._last_det_time, 1e-6)
            ratio = max(0.5, min(2.0, self._target_det_fps / max(1.0, est_det_fps)))
            desired_det_every = max(1, int(round(self.req_fps / max(1.0, self._target_det_fps * ratio))))
            self._frame_skip = max(1, desired_det_every)

        def _rescale_boxes_inplace(self, boxes, fx, fy):
            if boxes is None or len(boxes) == 0 or (fx == 1.0 and fy == 1.0):
                return boxes
            xyxy = boxes.xyxy
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
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                boxes_xyxy.append((x1, y1, x2, y2))
                alive_trackers.append(trk)
                alive_labels.append(lab)
            self._trackers = alive_trackers
            self._track_labels = alive_labels
            return boxes_xyxy, alive_labels

        def _predict_once(self, img_bgr):
            if 0.4 <= self.resize_factor < 1.0:
                img_in = cv2.resize(img_bgr, (0, 0),
                                    fx=self.resize_factor, fy=self.resize_factor,
                                    interpolation=cv2.INTER_AREA)
                fx = fy = self.resize_factor
            else:
                img_in = img_bgr
                fx = fy = 1.0

            kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
            if hasattr(self, "device") and self.device != "cpu":
                kwargs["device"] = self.device
                kwargs["half"] = self.use_half

            t0 = time.time()
            results = self.model.predict(img_in, **kwargs)
            self._last_det_time = max(1e-6, time.time() - t0)

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
                    self._start_trackers(frame_copy, boxes_xyxy, labels)
                    self._adapt_frame_skip()
                finally:
                    self._infer_busy = False

            threading.Thread(target=_task, args=(img_bgr.copy(),), daemon=True).start()

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

            out = _draw_fast(img.copy(), boxes_xyxy, labels if self.draw_labels else None)

            if self.show_fps:
                self._cnt += 1
                if self._cnt >= 15:
                    t1 = time.time()
                    self._fps = self._cnt / (t1 - self._t0)
                    self._t0, self._cnt = t1, 0
                cv2.putText(out, f"{self._fps:.1f} FPS  det~{1.0/max(self._last_det_time,1e-6):.1f} FPS  skip:{self._frame_skip}",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(out, format="bgr24")

    facing = "environment" if camera_mode.startswith("Back") else "user"
    capture_constraints = {
        "video": {
            "width":  {"ideal": 1280, "min": 960} if hq else {"ideal": 640},
            "height": {"ideal": 720,  "min": 540} if hq else {"ideal": 480},
            "frameRate": {"ideal": int(req_fps), "max": 60},
            "facingMode": {"ideal": facing},
        },
        "audio": False,
    }

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

    if ctx and ctx.video_processor:
        vp = ctx.video_processor
        vp.conf = float(conf)
        vp.imgsz = int(imgsz)
        vp.resize_factor = float(resize_factor)
        vp.draw_labels = bool(draw_labels)
        vp.hq = bool(hq)
        vp.req_fps = int(req_fps)

# ───────────────────────────────────────────────────────────────────────────────
# Upload → Split → Rebuild tab
# ───────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Upload a video, split into frames/segments, and rebuild")

    c1, c2 = st.columns(2)
    with c1:
        every_n = st.number_input("Save every Nth frame (1 = every frame)", min_value=1, max_value=100, value=1, step=1)
        chunk_sec = st.number_input("Segment length (seconds) for splitting", min_value=1, max_value=600, value=10, step=1)
        keep_frames = st.toggle("ZIP & offer frames for download", value=True)
        keep_segments = st.toggle("ZIP & offer segments for download", value=True)
    with c2:
        codec_choice = st.selectbox("Output codec (container .mp4)", ["mp4v", "avc1", "H264-fallback"], index=0)
        final_basename = st.text_input("Final output base filename (no extension)", value="rebuilt_video")

    uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "m4v", "avi", "mkv"], accept_multiple_files=False)

    # ── Video utility functions ────────────────────────────────────────────────
    def _probe_video(path: str):
        """Return (width, height, fps, nb_frames_est)"""
        with av.open(path) as container:
            vstreams = [s for s in container.streams if s.type == "video"]
            if not vstreams:
                raise RuntimeError("No video stream found.")
            vs = vstreams[0]
            width = int(vs.codec_context.width)
            height = int(vs.codec_context.height)
            fps = float(vs.average_rate) if vs.average_rate else (vs.base_rate and float(vs.base_rate)) or 30.0
            # nb_frames may be 0/None; estimate via duration if available
            duration = float(vs.duration * vs.time_base) if (vs.duration and vs.time_base) else None
            nb_est = int(round(duration * fps)) if duration else 0
            return width, height, fps, nb_est

    def _extract_frames_to_dir(in_path: str, out_dir: Path, every_n: int, pbar=None) -> int:
        """
        Decode with PyAV, save PNG frames numbered sequentially. Returns frame count saved.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        with av.open(in_path) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if (i % every_n) != 0:
                    continue
                img = frame.to_ndarray(format="bgr24")
                fname = out_dir / f"frame_{saved:06d}.png"
                cv2.imwrite(str(fname), img)
                saved += 1
                if pbar and (saved % 10 == 0):
                    pbar.progress(min(1.0, (i + 1) / max(1, stream.frames or (saved * every_n))))

        return saved

    def _write_video_from_frames(frames_dir: Path, fps: float, size: Tuple[int, int], out_path: Path):
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            raise RuntimeError("No frames to assemble.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if codec_choice == "avc1":
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
        elif codec_choice == "H264-fallback":
            # Sometimes OpenCV builds alias H264 differently; 'avc1' usually works best cross-platform.
            fourcc = cv2.VideoWriter_fourcc(*"H264")
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if not vw.isOpened():
            raise RuntimeError("Failed to open VideoWriter. Try switching codec.")
        for fp in frames:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            vw.write(img)
        vw.release()

    def _split_frames_into_segments(frames_dir: Path, fps: float, chunk_sec: int) -> List[Path]:
        """
        Builds segment_X.mp4 files from frames in `frames_dir` by time chunks.
        Returns list of segment paths.
        """
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return []
        size_img = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        size = (size_img.shape[1], size_img.shape[0])
        frames_per_chunk = int(round(fps * chunk_sec))
        segments = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # use same container codec choice later if desired

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

        if vw is not None:
            vw.release()

        return segments

    def _concat_segments_to_final(segments: List[Path], fps: float, out_path: Path):
        """
        Concatenate by decoding each segment and writing into one continuous writer.
        (Portable approach without shell ffmpeg.)
        """
        if not segments:
            raise RuntimeError("No segments to concatenate.")
        # Probe size from first frame of first segment
        cap0 = cv2.VideoCapture(str(segments[0]))
        ok, frame0 = cap0.read()
        cap0.release()
        if not ok or frame0 is None:
            raise RuntimeError("Failed to read first segment to infer size.")
        size = (frame0.shape[1], frame0.shape[0])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if codec_choice == "avc1":
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
        elif codec_choice == "H264-fallback":
            fourcc = cv2.VideoWriter_fourcc(*"H264")

        vw = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if not vw.isOpened():
            raise RuntimeError("Failed to open VideoWriter for final output.")

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
        vw.release()

    # ── UI actions ─────────────────────────────────────────────────────────────
    if uploaded:
        st.success(f"Uploaded: {uploaded.name}")
        run = st.button("Run: Split into frames → Build segments → Rebuild final")
        if run:
            with st.spinner("Processing video…"):
                with tempfile.TemporaryDirectory() as td:
                    tdir = Path(td)
                    in_path = tdir / uploaded.name
                    with open(in_path, "wb") as f:
                        f.write(uploaded.read())

                    # Probe
                    width, height, fps, nb_est = _probe_video(str(in_path))
                    st.write(f"Detected: **{width}×{height} @ {fps:.3f} fps**  (frames≈{nb_est or 'unknown'})")

                    # Extract frames
                    frames_dir = tdir / "frames"
                    pbar = st.progress(0.0, text="Extracting frames…")
                    saved = _extract_frames_to_dir(str(in_path), frames_dir, every_n=int(every_n), pbar=pbar)
                    pbar.progress(1.0)
                    st.write(f"Saved **{saved}** frame(s) to `{frames_dir.name}`")

                    # Build segments
                    seg_paths: List[Path] = []
                    if chunk_sec > 0:
                        st.write("Creating segments…")
                        seg_paths = _split_frames_into_segments(frames_dir, fps=fps, chunk_sec=int(chunk_sec))
                        st.write(f"Created **{len(seg_paths)}** segment file(s).")

                    # Concat segments → final (or build final directly from frames if only one segment)
                    final_path = tdir / f"{final_basename}.mp4"
                    if seg_paths:
                        _concat_segments_to_final(seg_paths, fps=fps, out_path=final_path)
                    else:
                        # If user chose a very large chunk or disabled splitting, just rebuild from frames
                        _write_video_from_frames(frames_dir, fps=fps, size=(width, height), out_path=final_path)

                    # Prepare downloads
                    st.success("Done.")

                    # ZIP frames (optional)
                    if keep_frames:
                        frames_zip = tdir / "frames.zip"
                        shutil.make_archive(str(frames_zip.with_suffix('')), 'zip', frames_dir)
                        with open(frames_zip, "rb") as zf:
                            st.download_button("⬇️ Download frames (ZIP)", zf, file_name="frames.zip", mime="application/zip")

                    # ZIP segments (optional)
                    if keep_segments and seg_paths:
                        seg_root = tdir / "segments"
                        seg_root.mkdir(exist_ok=True)
                        for p in seg_paths:
                            shutil.copy2(p, seg_root / p.name)
                        seg_zip = tdir / "segments.zip"
                        shutil.make_archive(str(seg_zip.with_suffix('')), 'zip', seg_root)
                        with open(seg_zip, "rb") as zf:
                            st.download_button("⬇️ Download segments (ZIP)", zf, file_name="segments.zip", mime="application/zip")

                    # Final video
                    with open(final_path, "rb") as vf:
                        st.download_button("🎬 ⬇️ Download rebuilt video (MP4)", vf, file_name=final_path.name, mime="video/mp4")

                    st.caption("Tip: If the final file won’t play on iOS, switch codec to **avc1** and rerun.")
