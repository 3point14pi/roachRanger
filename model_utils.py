from ultralytics import YOLO
import streamlit as st
model_name = "cockroach_detection.pt"

@st.cache_resource
def load_yolo_model(image_path = model_name):
  return YOLO(image_path)
