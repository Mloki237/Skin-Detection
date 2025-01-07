# -*- coding: utf-8 -*-
pip install ultralytics==8.2.103 -q

import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load your custom YOLO model
model = YOLO('yolov8n.pt')  # Path to your trained YOLO model

st.title('Custom YOLOv8 Object Detection App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption='Uploaded Image')

    # Run YOLO inference on the image
    results = model.predict(source=image, save=False)

    # Plot and display the results with bounding boxes
    result_image = results[0].plot()  # Get the processed image with bounding boxes
    st.image(result_image, caption='Detected Image', use_column_width=True)
