import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import torch
import tempfile
import supervision as sv
from model_loader import load_model
import cv2


# Step 1: Load the model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Title and description
st.title("Document Layout Analysis with YOLOv10")
st.write("Upload a document (image or PDF), and the app will apply object detection using YOLOv10.")

# File upload
uploaded_file = st.file_uploader("Upload an image (PNG, JPG) or PDF", type=["png", "jpg", "jpeg", "pdf"])

def process_pdf(uploaded_file):
    """Convert the first page of a PDF to a NumPy image."""
    pages = convert_from_bytes(uploaded_file.read())
    if not pages:
        st.error("Could not extract pages from the PDF.")
        st.stop()
    return np.array(pages[0])  # Convert first page to NumPy array

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Load and preprocess image
    if "image" in file_type:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif "pdf" in file_type:
        image = process_pdf(uploaded_file)
    else:
        st.error("Unsupported file format!")
        st.stop()

    # Display uploaded image
    st.image(image, caption="Uploaded Document", use_column_width=True)

    # Run the model on the image
    st.write("Processing the document...")
    results = list(model.predict(image))  # Ensure it's a list

    if not results:
        st.error("No detections found!")
        st.stop()

    # Convert results to Supervision detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Annotate image with bounding boxes and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Display annotated image
    st.image(annotated_image, caption="Annotated Document", use_column_width=True)

    # Provide download link for the annotated image
    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        annotated_image_pil.save(temp_file.name)
        st.download_button("Download Annotated Image", data=open(temp_file.name, "rb"), file_name="annotated_document.png")









