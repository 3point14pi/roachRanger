import streamlit as st
from PIL import Image
import numpy as np
from model_utils import load_yolo_model
import os
import shutil

IMAGE_ADDRESS = "https://i.ytimg.com/vi/bEwCA_nrY5Q/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLB4y6YJGxw6oVxFm-Uucbrjfqhu2Q"
IMAGE_NAME = "uploaded.png"
PRED_IMAGE_PATH = "runs/detect/predict/uploaded.png"
DIRECTORY = "runs/detect/predict"
PRED_MOVE_NAME = "pred_image.png"
yolo_model = load_yolo_model()

def make_predictions(IMAGE_PATH):
  results = yolo_model.predict(IMAGE_PATH, save=True, conf=0.75)
  try:
        if os.path.exists(PRED_IMAGE_PATH):
            shutil.move(PRED_IMAGE_PATH, PRED_MOVE_NAME)
            os.rmdir(DIRECTORY)

            return True
        else:
            return False
    except Exception as error:
        print(str(error))

        return False
  
st.title("Cockroach Detection")
st.image(IMAGE_ADDRESS, caption = "Cockroach Detection"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open the uploaded file using PIL
    image = Image.open(uploaded_file)

    # Display the image
    image.save(IMAGE_NAME)

    # get predictions
    with st.spinner("Getting Predictions......"):
        mask_response = make_predictions(IMAGE_NAME)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image)

        with col2:
            st.subheader("Detections")
            if mask_response:
                st.image(PRED_MOVE_NAME)
            else:
                st.error("Error Getting Predictions", icon="🚨")
