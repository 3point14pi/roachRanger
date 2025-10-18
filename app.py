import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
from model_utils import load_yolo_model
import tempfile
import shutil
import os
import io
import mimetypes

# ---------- constants ----------
IMAGE_ADDRESS = "https://i.ytimg.com/vi/bEwCA_nrY5Q/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLB4y6YJGxw6oVxFm-Uucbrjfqhu2Q"
ALLOWED_IMG = ["jpg", "jpeg", "png"]
ALLOWED_VID = ["mp4", "mov", "avi", "mkv", "webm"]

# ---------- load model once ----------
yolo_model = load_yolo_model()

def save_upload_to_disk(uploaded_file) -> Path:
    """
    Save an uploaded file to a NamedTemporaryFile and return the path.
    Preserves the file extension so Ultralytics can infer type.
    """
    suffix = Path(uploaded_file.name).suffix
    if not suffix:
        # fall back to mime if no suffix
        guessed = mimetypes.guess_extension(uploaded_file.type or "")
        suffix = guessed or ""

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)

def run_yolo_and_get_outputs(input_path: Path, conf: float = 0.5, is_video: bool = False):
    """
    Run yolo_model on a path (image or video). Returns:
    - save_dir: Path to Ultralytics save directory (runs/detect/predictX)
    - outputs: list[Path] of files written there (images or a video)
    """
    # For video, consider skipping frames for speed
    kwargs = dict(save=True, conf=conf)
    if is_video:
        # vid_stride=2 means process every 2nd frame. Increase for more speed, less accuracy.
        kwargs["vid_stride"] = 2

    results = yolo_model.predict(str(input_path), **kwargs)
    save_dir = Path(results[0].save_dir)
    outputs = sorted([p for p in save_dir.glob("*") if p.is_file()])
    return save_dir, outputs

def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower().lstrip(".") in ALLOWED_IMG

def is_video_file(name: str) -> bool:
    return Path(name).suffix.lower().lstrip(".") in ALLOWED_VID

# ---------- UI ----------
st.title("Cockroach Detection")
st.image(IMAGE_ADDRESS, caption="Cockroach Detection")

uploaded = st.file_uploader(
    "Upload an image or a video...",
    type=ALLOWED_IMG + ALLOWED_VID
)

if uploaded is not None:
    filename = uploaded.name
    ext = Path(filename).suffix.lower().lstrip(".")

    # Branch by type
    if is_image_file(filename):
        # Show and save image
        image = Image.open(uploaded).convert("RGB")
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Persist the image to disk for YOLO
        tmp_img_path = Path(tempfile.gettempdir()) / f"uploaded_image.{ext}"
        image.save(tmp_img_path)

        with st.spinner("Running detection on image..."):
            save_dir, outputs = run_yolo_and_get_outputs(tmp_img_path, conf=0.5, is_video=False)

        # Find the annotated output image
        # Ultralytics typically writes an output with the same basename
        pred_img = None
        for p in outputs:
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                pred_img = p
                break

        st.subheader("Detections")
        if pred_img and pred_img.exists():
            st.image(str(pred_img), use_column_width=True)
        else:
            st.error("Could not locate predicted image output.", icon="🚨")

    elif is_video_file(filename):
        # Save raw bytes to disk (don’t try PIL)
        tmp_vid_path = save_upload_to_disk(uploaded)

        st.subheader("Original Video")
        st.video(str(tmp_vid_path))

        with st.spinner("Running detection on video (this may take a bit)..."):
            save_dir, outputs = run_yolo_and_get_outputs(tmp_vid_path, conf=0.5, is_video=True)

        # Look for an output video in save_dir
        pred_video = None
        for p in outputs:
            if p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                pred_video = p
                break

        st.subheader("Detections (Annotated Video)")
        if pred_video and pred_video.exists():
            st.video(str(pred_video))
            st.caption(f"Saved outputs in: {save_dir}")
        else:
            st.error("Could not locate predicted video output.", icon="🚨")

        # Optional light cleanup of temp upload
        try:
            tmp_vid_path.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        st.error("Unsupported file type. Please upload an image (jpg/png) or a video (mp4/mov/avi/mkv/webm).")
