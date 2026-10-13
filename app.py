import os
import shutil
import subprocess
import tempfile

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Dossier du projet (là où se trouve app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def detect_image(image, model, original_filename="detected_image"):
    results = model(image)
    annotated_bgr = results[0].plot()  # image annotée au format BGR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{original_filename}_detected.jpg")
    cv2.imwrite(output_path, annotated_bgr)

    # Conversion en RGB pour l'affichage dans Streamlit
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results[0], output_path


def detect_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    raw_path = os.path.join(OUTPUT_DIR, f"{base_name}_raw.mp4")
    final_path = os.path.join(OUTPUT_DIR, f"{base_name}_detected.mp4")

    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        out.write(results[0].plot())

    cap.release()
    out.release()

    # Si ffmpeg est installé, on convertit en H.264 pour une lecture dans le navigateur
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-vcodec", "libx264", final_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(raw_path)
            return final_path
        except Exception:
            pass

    # Sinon on garde le fichier mp4v (téléchargeable, mais parfois illisible dans le navigateur)
    os.replace(raw_path, final_path)
    return final_path


# ---------------- Interface Streamlit ----------------
st.title("Personal Protective Equipment (PPE) Detection")
st.sidebar.header("Choose Input Type")
input_type = st.sidebar.radio("Select input type", ["Image", "Video"])

if not os.path.exists(MODEL_PATH):
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

model = load_model()

if input_type == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        st.subheader("Detection Results:")
        filename = os.path.splitext(uploaded_image.name)[0]
        annotated_image, results, image_output_path = detect_image(
            image, model, original_filename=filename
        )
        st.image(annotated_image, caption="Detected Image", use_container_width=True)

        st.write("Detected Objects:")
        st.dataframe(results.boxes.data.cpu().numpy())

        with open(image_output_path, "rb") as f:
            st.download_button(
                "Download Detected Image", f, file_name=os.path.basename(image_output_path)
            )

elif input_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        st.video(uploaded_video)

        with st.spinner("Running object detection on video..."):
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
                temp_input.write(uploaded_video.read())
                video_path = temp_input.name

            output_video_path = detect_video(video_path, model)

        st.success("Done! Here's the processed video:")
        st.video(output_video_path)

        with open(output_video_path, "rb") as f:
            st.download_button(
                "Download Processed Video", f, file_name=os.path.basename(output_video_path)
            )