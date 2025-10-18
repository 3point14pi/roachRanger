# app_webrtc.py
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("models/yolov8n.pt")

model = load_model()

st.set_page_config(page_title="Cockroach Live (WebRTC)", layout="wide")
st.title("Cockroach Detection — Live (WebRTC)")

conf = st.slider("Confidence", 0.1, 0.9, 0.5, 0.05)

# Public STUN servers so WebRTC can connect
rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]
})

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        results = model.predict(img_bgr, conf=conf, verbose=False)
        out_bgr = results[0].plot()
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

webrtc_streamer(
    key="cockroach-live",
    mode="recvonly",  # we only receive from the user's webcam and send back annotated frames
    video_processor_factory=YOLOProcessor,
    rtc_configuration=rtc_config,
    async_processing=True,
)
