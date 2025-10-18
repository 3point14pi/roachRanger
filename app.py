# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import numpy as np
from model_utils import load_yolo_model

st.set_page_config(page_title="Live Cockroach Detection", layout="wide")
st.title("🐞 Live Cockroach Detection")

conf = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
imgsz = st.sidebar.selectbox("Image size", [320, 480, 640, 800], index=2)

class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_yolo_model()
        self.conf = conf
        self.imgsz = imgsz

    def recv(self, frame):
        # frame -> ndarray (BGR)
        img_bgr = frame.to_ndarray(format="bgr24")

        # Run YOLO
        results = self.model.predict(
            img_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        # Draw boxes/labels
        plotted_bgr = results[0].plot()  # returns BGR ndarray
        return av.VideoFrame.from_ndarray(plotted_bgr, format="bgr24")

ctx = webrtc_streamer(
    key="yolo-live",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=YOLOProcessor,
)

# Let sidebar controls update the running processor
if ctx and ctx.video_processor:
    ctx.video_processor.conf = conf
    ctx.video_processor.imgsz = imgsz
