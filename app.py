# app.py
import os, math, tempfile, time
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Your util that returns a preloaded Ultralytics YOLO detect model
from model_utils import load_yolo_model

st.set_page_config(page_title="Chunked Video Detect & Rebuild", layout="wide")
st.title("🐞 Upload → Split → Detect → Rebuild")

# ── Controls ───────────────────────────────────────────────────────────────────
with st.sidebar:
    conf = st.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
    imgsz = st.selectbox("Inference size (px)", [320, 480, 640, 800], index=2)
    draw_labels = st.toggle("Show labels", value=True)
    resize_factor = st.slider("Pre-resize (fx/fy)", 0.4, 1.0, 1.0, 0.05)
    every_n = st.number_input("Detect every Nth frame", 1, 20, 1, 1)
    chunk_secs = st.number_input("Chunk length (seconds)", 2, 120, 10, 1)
    st.caption("Outputs a new video (video-only).")

# ── Helpers ────────────────────────────────────────────────────────────────────
def _draw_fast(img, boxes_xyxy: List[Tuple[int,int,int,int]], labels=None):
    for i, (x1,y1,x2,y2) in enumerate(boxes_xyxy or []):
        cv2.rectangle(img, (x1,y1), (x2,y2), (50,220,50), 2, lineType=cv2.LINE_AA)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(img, labels[i], (x1, max(20, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

def _boxes_to_int_xyxy_and_labels(boxes, names, want_labels=True):
    if boxes is None or len(boxes) == 0: return [], []
    xyxy = getattr(boxes, "xyxy", None); cls = getattr(boxes, "cls", None); conf = getattr(boxes, "conf", None)
    if xyxy is None: return [], []
    try:
        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy,"detach") else np.array(xyxy)
    except Exception:
        return [], []
    norm_names = names
    if isinstance(norm_names, dict):
        try:
            mx = max(norm_names.keys())
            if all(k in norm_names for k in range(mx+1)):
                norm_names = [norm_names[k] for k in range(mx+1)]
        except Exception:
            pass
    out_b, out_l = [], []
    for i in range(xyxy_np.shape[0]):
        x1,y1,x2,y2 = map(int, xyxy_np[i].tolist()); out_b.append((x1,y1,x2,y2))
        label = None
        if want_labels and cls is not None and conf is not None:
            try:
                c = int(cls[i].item() if hasattr(cls[i],"item") else cls[i])
                p = float(conf[i].item() if hasattr(conf[i],"item") else conf[i])
                if isinstance(norm_names,(list,tuple)) and 0 <= c < len(norm_names):
                    label = f"{norm_names[c]} {p:.2f}"
                elif isinstance(norm_names,dict) and c in norm_names:
                    label = f"{norm_names[c]} {p:.2f}"
                else:
                    label = f"id{c} {p:.2f}"
            except Exception:
                label = None
        out_l.append(label)
    return out_b, out_l

def _predict_once(model, img_bgr, conf, imgsz, resize_factor, device, use_half, want_labels):
    if 0.4 <= resize_factor < 1.0:
        img_in = cv2.resize(img_bgr, (0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        fx = fy = resize_factor
    else:
        img_in = img_bgr; fx = fy = 1.0
    kwargs = dict(conf=conf, imgsz=imgsz, verbose=False)
    if device != "cpu":
        kwargs["device"] = device
        if use_half: kwargs["half"] = True
    r0 = model.predict(img_in, **kwargs)[0]
    if 0.4 <= resize_factor < 1.0:
        try:
            r0.boxes.xyxy[:,[0,2]] /= fx; r0.boxes.xyxy[:,[1,3]] /= fy
        except Exception:
            pass
    names = getattr(model, "names", None)
    return _boxes_to_int_xyxy_and_labels(r0.boxes, names, want_labels)

def make_frame_processor(model, conf, imgsz, resize_factor, every_n, draw_labels, device, use_half):
    state = {"i": 0, "last": ([], [])}
    def process(frame_rgb):
        # MoviePy gives RGB; convert to BGR
        bgr = frame_rgb[:, :, ::-1].copy()
        i = state["i"]; do_det = (i % every_n == 0)
        if do_det:
            boxes, labels = _predict_once(model, bgr, conf, imgsz, resize_factor, device, use_half, draw_labels)
            state["last"] = (boxes, labels)
        else:
            boxes, labels = state["last"]
        cv2.putText(bgr, f"det_cnt:{len(boxes)}", (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        out = _draw_fast(bgr, boxes, labels if draw_labels else None)
        state["i"] += 1
        return out[:, :, ::-1]  # back to RGB
    return process

# ── Model init (GPU if present) ────────────────────────────────────────────────
device, use_half = "cpu", False
try:
    import torch
    if torch.cuda.is_available():
        device = 0
except Exception:
    pass

model = load_yolo_model()
try:
    if device != "cpu":
        model.to("cuda")
        if hasattr(model, "fuse"):
            try: model.fuse()
            except Exception: pass
        try:
            model.model.half(); use_half = True
        except Exception:
            use_half = False
except Exception:
    device, use_half = "cpu", False

# ── UI ─────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a video (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"])

if uploaded:
    try:
        # Save input
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp_in:
            tmp_in.write(uploaded.read()); input_path = tmp_in.name

        # Open once to get fps/duration
        base_clip = VideoFileClip(input_path)  # audio ignored downstream
        fps = base_clip.fps or 30
        duration = base_clip.duration
        n_chunks = max(1, math.ceil(duration / chunk_secs))
        st.write(f"Duration: {duration:.2f}s · FPS: {fps:.2f} · Chunks: {n_chunks} (~{chunk_secs}s each)")

        prog = st.progress(0); status = st.empty()
        chunk_paths = []

        for i in range(n_chunks):
            t0 = i * chunk_secs
            t1 = min((i + 1) * chunk_secs, duration + 1e-3)
            sub = base_clip.subclip(t0, t1)

            # per-chunk processor (keeps lightweight state)
            process = make_frame_processor(model, conf, imgsz, resize_factor, every_n, draw_labels, device, use_half)
            sub_proc = sub.fl_image(process)

            out_chunk = os.path.join(tempfile.gettempdir(), f"chunk_{i}_{int(time.time())}.mp4")
            sub_proc.write_videofile(out_chunk, codec="libx264", audio=False, fps=fps, verbose=False, logger=None)
            chunk_paths.append(out_chunk)

            prog.progress((i + 1) / n_chunks)
            status.text(f"Processed chunk {i+1}/{n_chunks}")

            # Free resources early
            sub.close(); sub_proc.close()

        base_clip.close()

        # Concat
        status.text("Stitching chunks…")
        clips = [VideoFileClip(p) for p in chunk_paths]
        final = concatenate_videoclips(clips, method="compose")
        final_path = os.path.join(tempfile.gettempdir(), f"final_annotated_{int(time.time())}.mp4")
        final.write_videofile(final_path, codec="libx264", audio=False, fps=fps, verbose=False, logger=None)

        for c in clips: c.close()
        for p in chunk_paths:
            try: os.remove(p)
            except Exception: pass
        try: os.remove(input_path)
        except Exception: pass

        st.success("Done! Preview below 👇")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("Download final annotated video", data=f, file_name="annotated.mp4", mime="video/mp4")

    except Exception as e:
        st.error("Something went wrong while processing.")
        st.exception(e)
else:
    st.info("Upload a video to start.")

