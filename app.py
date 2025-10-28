# app.py
import os
import time
import math
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

# Your util that returns a preloaded ultralytics YOLO model (e.g., YOLO("yolov8n.pt"))
from model_utils import load_yolo_model

st.set_page_config(page_title="Chunked Video Detection & Rebuild", layout="wide")
st.title("🐞 Chunked Cockroach Detection → Rebuilt Video")

st.sidebar.header("Controls")
conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640, 800], index=2)
draw_labels = st.sidebar.toggle("Show labels", value=True)
resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 1.0, 0.05)
every_n = st.sidebar.number_input("Detect every Nth frame (1 = every frame)", 1, 20, 1, 1)
chunk_secs = st.sidebar.number_input("Chunk length (seconds)", 2, 60, 10, 1)

st.sidebar.caption("We split the video into chunks, annotate each chunk, then stitch it back together (video-only).")

# ----------------- Helpers -----------------
def _draw_fast(img, boxes_xyxy: List[Tuple[int, int, int, int]], labels=None):
    if not boxes_xyxy:
        return img
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        cv2.rectangle(img, (x1, y1), (x2, y2), (50, 220, 50), 2, lineType=cv2.LINE_AA)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(img, labels[i], (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def _boxes_to_int_xyxy_and_labels(boxes, names, draw_labels_flag=True):
    if boxes is None or len(boxes) == 0:
        return [], []
    xyxy = getattr(boxes, "xyxy", None)
    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    if xyxy is None:
        return [], []
    try:
        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.array(xyxy)
    except Exception:
        return [], []
    norm_names = names
    if isinstance(norm_names, dict):
        try:
            max_k = max(norm_names.keys())
            if all(k in norm_names for k in range(max_k + 1)):
                norm_names = [norm_names[k] for k in range(max_k + 1)]
        except Exception:
            pass
    out_boxes, out_labels = [], []
    for i in range(xyxy_np.shape[0]):
        x1, y1, x2, y2 = map(int, xyxy_np[i].tolist())
        out_boxes.append((x1, y1, x2, y2))
        label = None
        if draw_labels_flag and cls is not None and conf is not None:
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

def _predict_once(model, img_bgr, conf, imgsz, resize_factor, device, use_half, draw_labels):
    if 0.4 <= resize_factor < 1.0:
        img_in = cv2.resize(img_bgr, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        fx = fy = resize_factor
    else:
        img_in = img_bgr
        fx = fy = 1.0
    kwargs = dict(conf=conf, imgsz=imgsz, verbose=False)
    if device != "cpu":
        kwargs["device"] = device
        if use_half:
            kwargs["half"] = True
    results = model.predict(img_in, **kwargs)
    r = results[0]
    # undo scaling if any
    if 0.4 <= resize_factor < 1.0:
        try:
            if hasattr(r, "boxes") and hasattr(r.boxes, "xyxy"):
                r.boxes.xyxy[:, [0, 2]] = r.boxes.xyxy[:, [0, 2]] / fx
                r.boxes.xyxy[:, [1, 3]] = r.boxes.xyxy[:, [1, 3]] / fy
        except Exception:
            pass
    names = getattr(model, "names", None)
    boxes_xyxy, labels = _boxes_to_int_xyxy_and_labels(r.boxes, names, draw_labels)
    return boxes_xyxy, labels

def _process_chunk(cap, start_f, end_f, out_path, fps, size, model, conf, imgsz, resize_factor, device, use_half, every_n, draw_labels):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    last_boxes, last_labels = [], []
    idx = start_f
    while idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        do_det = ((idx - start_f) % every_n == 0)
        if do_det:
            boxes, labels = _predict_once(
                model=model, img_bgr=frame, conf=conf, imgsz=imgsz,
                resize_factor=resize_factor, device=device, use_half=use_half, draw_labels=draw_labels
            )
            last_boxes, last_labels = boxes, labels
        else:
            boxes, labels = last_boxes, last_labels

        cv2.putText(frame, f"det_cnt:{len(boxes)}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        annotated = _draw_fast(frame, boxes, labels if draw_labels else None)
        writer.write(annotated)
        idx += 1
    writer.release()

def _concat_chunks(chunk_paths, final_path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(final_path, fourcc, fps, size)
    for p in chunk_paths:
        cap = cv2.VideoCapture(p)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # ensure size
            if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                frame = cv2.resize(frame, size)
            writer.write(frame)
        cap.release()
    writer.release()

# ----------------- Model init (GPU if present) -----------------
device = "cpu"
use_half = False
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
            model.model.half()
            use_half = True
        except Exception:
            use_half = False
except Exception:
    device = "cpu"
    use_half = False

# ----------------- UI: upload & run -----------------
uploaded = st.file_uploader("Upload a video (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    # Save input
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if np.isnan(fps) or fps <= 0: fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        size = (w, h)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames_per_chunk = max(1, int(round(fps * chunk_secs)))
        n_chunks = max(1, math.ceil(total / frames_per_chunk)) if total > 0 else 1

        st.write(f"Input: {w}×{h} @ {fps:.2f} FPS · {total} frames · {n_chunks} chunks (~{chunk_secs}s each)")
        prog = st.progress(0)
        status = st.empty()

        # Make and process chunks
        chunk_paths = []
        t0 = time.time()
        for ci in range(n_chunks):
            start_f = ci * frames_per_chunk
            end_f = min(total, (ci + 1) * frames_per_chunk) if total > 0 else start_f + frames_per_chunk
            out_chunk = os.path.join(tempfile.gettempdir(), f"chunk_{ci}_{int(time.time())}.mp4")
            _process_chunk(
                cap=cap, start_f=start_f, end_f=end_f, out_path=out_chunk, fps=fps, size=size,
                model=model, conf=conf, imgsz=imgsz, resize_factor=resize_factor,
                device=device, use_half=use_half, every_n=every_n, draw_labels=draw_labels
            )
            chunk_paths.append(out_chunk)
            if total > 0:
                prog.progress(min(1.0, (ci + 1) / n_chunks))
                status.text(f"Processed chunk {ci+1}/{n_chunks}")

        cap.release()

        # Stitch chunks -> final
        final_path = os.path.join(tempfile.gettempdir(), f"final_annotated_{int(time.time())}.mp4")
        status.text("Stitching chunks into final video…")
        _concat_chunks(chunk_paths, final_path, fps=fps, size=size)

        # Cleanup chunk files
        for p in chunk_paths:
            try: os.remove(p)
            except Exception: pass
        try: os.remove(input_path)
        except Exception: pass

        elapsed = time.time() - t0
        st.success(f"Done in {elapsed:.1f}s. Preview below 👇 (video-only).")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("Download final annotated video", data=f, file_name="annotated.mp4", mime="video/mp4")
else:
    st.info("Upload a video to start.")
