# app.py
import os
import time
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

# Your util that returns a preloaded ultralytics YOLO model (e.g., YOLO("yolov8n.pt"))
from model_utils import load_yolo_model

# ───────────────────────────────────────────────────────────────────────────────
# Page
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Video Cockroach Detection", layout="wide")
st.title("🐞 Cockroach Detection on Uploaded Video (YOLO)")

st.sidebar.header("Controls")

# Detection settings
conf = st.sidebar.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
imgsz = st.sidebar.selectbox("Inference size (px)", [320, 480, 640, 800], index=2)
draw_labels = st.sidebar.toggle("Show labels", value=True)
resize_factor = st.sidebar.slider("Pre-resize input (fx/fy)", 0.4, 1.0, 1.0, 0.05)

# Performance
every_n = st.sidebar.number_input("Detect every Nth frame (1 = every frame)", min_value=1, max_value=20, value=1, step=1)

st.sidebar.caption(
    "Tip: Increase N to speed up processing. For example, N=2 halves detection calls."
)

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
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
    """
    Robustly convert Ultralytics Boxes -> (list[int xyxy], list[str|None] labels).
    Works across minor API differences (dict/list names, tensor/array internals).
    """
    if boxes is None or len(boxes) == 0:
        return [], []

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

    # Normalize names (support dict or list)
    norm_names = names
    if isinstance(norm_names, dict):
        try:
            max_k = max(norm_names.keys())
            if all(k in norm_names for k in range(max_k + 1)):
                norm_names = [norm_names[k] for k in range(max_k + 1)]
        except Exception:
            pass

    out_boxes, out_labels = [], []
    n = xyxy_np.shape[0]
    for i in range(n):
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


def _predict_once(model, img_bgr, conf, imgsz, resize_factor, device, use_half):
    # Optional pre-resize for inference only
    if 0.4 <= resize_factor < 1.0:
        img_in = cv2.resize(img_bgr, (0, 0),
                            fx=resize_factor, fy=resize_factor,
                            interpolation=cv2.INTER_AREA)
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
    boxes = r.boxes

    # Undo scaling on results if we pre-resized
    if 0.4 <= resize_factor < 1.0:
        try:
            if hasattr(r, "boxes") and hasattr(r.boxes, "xyxy"):
                r.boxes.xyxy[:, [0, 2]] = r.boxes.xyxy[:, [0, 2]] / fx
                r.boxes.xyxy[:, [1, 3]] = r.boxes.xyxy[:, [1, 3]] / fy
        except Exception:
            pass

    names = getattr(model, "names", None)
    boxes_xyxy, labels = _boxes_to_int_xyxy_and_labels(boxes, names, draw_labels)
    return boxes_xyxy, labels


# ───────────────────────────────────────────────────────────────────────────────
# Model init (GPU if available)
# ───────────────────────────────────────────────────────────────────────────────
device = "cpu"
use_half = False
torch = None
try:
    import torch as _torch
    torch = _torch
    if torch.cuda.is_available():
        device = 0
except Exception:
    pass

model = load_yolo_model()
try:
    if device != "cpu":
        model.to("cuda")
        if hasattr(model, "fuse"):
            try:
                model.fuse()
            except Exception:
                pass
        try:
            # half precision speeds up on many GPUs
            model.model.half()
            use_half = True
        except Exception:
            use_half = False
except Exception:
    device = "cpu"
    use_half = False

# ───────────────────────────────────────────────────────────────────────────────
# Uploader
# ───────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a video (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"])

if uploaded is not None:
    # Save to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    # Prepare output path
    out_path = os.path.join(tempfile.gettempdir(), f"annotated_{int(time.time())}.mp4")

    # Open input
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
    else:
        in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if np.isnan(in_fps) or in_fps <= 0:
            in_fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        # Writer (mp4v widely supported)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, in_fps, (w, h))

        # UI
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        prog = st.progress(0 if total_frames > 0 else 0)
        status = st.empty()

        last_boxes, last_labels = [], []
        frame_idx = 0
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            do_detect = (frame_idx % every_n == 0)

            if do_detect:
                boxes, labels = _predict_once(
                    model=model,
                    img_bgr=frame,
                    conf=conf,
                    imgsz=imgsz,
                    resize_factor=resize_factor,
                    device=device,
                    use_half=use_half,
                )
                last_boxes, last_labels = boxes, labels
            else:
                boxes, labels = last_boxes, last_labels

            # Overlay debug count
            cv2.putText(frame, f"det_cnt:{len(boxes)}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw
            annotated = _draw_fast(frame, boxes, labels if draw_labels else None)
            writer.write(annotated)

            # Progress
            frame_idx += 1
            if total_frames > 0:
                prog.progress(min(1.0, frame_idx / total_frames))
                if frame_idx % max(1, total_frames // 50) == 0:
                    elapsed = time.time() - t0
                    status.text(f"Processing frame {frame_idx}/{total_frames} • elapsed {elapsed:.1f}s")

        cap.release()
        writer.release()

        st.success("Done! Preview below 👇")
        st.video(out_path)

        with open(out_path, "rb") as f:
            st.download_button("Download annotated video", data=f, file_name="annotated.mp4", mime="video/mp4")

        # Cleanup input (keep output so user can download)
        try:
            os.remove(input_path)
        except Exception:
            pass
else:
    st.info("Upload a video file to start.")
