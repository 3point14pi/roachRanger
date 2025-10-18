import streamlit as st, time, cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from model_utils import load_yolo_model

yolo_model = yolo_model if 'yolo_model' in globals() else load_yolo_model()

st.subheader("Live Camera via WebRTC (1 frame every N seconds)")
interval = st.number_input("Interval (sec)", 1, 30, 5)
conf = st.slider("Confidence", 0.1, 0.9, 0.5, 0.05)

class FrameProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_ts = 0.0
        self.annotated_bgr = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        now = time.time()
        # Only run YOLO every `interval` seconds
        if now - self.last_ts >= st.session_state.get("interval", 5):
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = yolo_model.predict(rgb, conf=st.session_state.get("conf", 0.5), save=False, verbose=False)
            ann_rgb = results[0].plot()
            self.annotated_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
            self.last_ts = now

        # Show last annotated frame; fall back to raw frame
        out = self.annotated_bgr if self.annotated_bgr is not None else img_bgr
        return av.VideoFrame.from_ndarray(out, format="bgr24")

# store UI values so processor can read them
st.session_state["interval"] = int(interval)
st.session_state["conf"] = float(conf)

webrtc_streamer(
    key="live-webrtc",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=FrameProcessor,
)
