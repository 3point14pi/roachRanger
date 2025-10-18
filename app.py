import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from model_utils import load_yolo_model  # your loader

@st.cache_resource
def get_model():
    model = load_yolo_model()
    # (optional) model.fuse(); model.to("cuda"); model.half()
    return model

class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = get_model()

    def recv(self, frame):
        # frame: av.VideoFrame -> numpy (BGR)
        img = frame.to_ndarray(format="bgr24")

        # run inference (Ultralytics models accept numpy arrays)
        results = self.model.predict(img, conf=0.5, verbose=False)

        # draw boxes on img
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].int().tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{r.names[cls]} {conf:.2f}"
                # simple rectangle/label (OpenCV)
                import cv2
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, label, (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # return to browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Live Cockroach Detection")
webrtc_streamer(
    key="live",
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
