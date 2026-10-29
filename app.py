from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "facenet_model.pt"

st.set_page_config(
    page_title="FaceNet Recognition",
    page_icon="🧑‍💻",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 55%);}
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #152238, #315b85);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: .45rem 0 0 0; color: #e6eef8;}
    .small-note {color: #5d6b7b; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>FaceNet Face Recognition</h1>
      <p>Upload a photo to detect faces and compare them with the enrolled member embeddings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading saved FaceNet model...")
def load_recognition_model(model_path_text: str):
    model_path = Path(model_path_text)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. "
            "Place facenet_model.pt beside app.py before deploying."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    required = {"state", "classes", "centroids", "threshold"}
    missing = required - set(checkpoint.keys())
    if missing:
        raise KeyError(
            f"Model checkpoint is missing keys: {sorted(missing)}. "
            "Use the facenet_model.pt saved by the project notebook."
        )

    backbone = InceptionResnetV1(pretrained=None).to(device)
    backbone.load_state_dict(checkpoint["state"], strict=False)
    backbone.eval()

    # keep_all=True allows the upload page to process multiple detected faces.
    mtcnn = MTCNN(device=device, keep_all=True)
    classes = checkpoint["classes"]
    centroids = torch.as_tensor(
        checkpoint["centroids"], dtype=torch.float32, device=device
    )
    saved_threshold = float(checkpoint["threshold"])
    return device, backbone, mtcnn, classes, centroids, saved_threshold


def recognize_image(image: Image.Image, device, backbone, mtcnn, classes, centroids, threshold):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    boxes, probabilities = mtcnn.detect(rgb)
    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    results = []

    if boxes is None or probabilities is None:
        return annotated, results

    for box, probability in zip(boxes, probabilities):
        if probability is None or float(probability) < 0.90:
            continue

        face = extract_face(rgb, box, image_size=160, margin=20).float()
        face = fixed_image_standardization(face).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = F.normalize(backbone(face), dim=1)
            distances = torch.cdist(embedding, centroids)
            min_distance, index = distances.min(dim=1)

        distance = float(min_distance.item())
        if distance <= threshold:
            label = str(classes[int(index.item())])
            matched = True
        else:
            label = "Unidentified"
            matched = False

        x1, y1, x2, y2 = [int(v) for v in box]
        color = "#159957" if matched else "#d93025"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label_text = f"{label} | distance: {distance:.3f}"
        # Draw a solid label background so it stays readable.
        text_box = draw.textbbox((x1, max(0, y1 - 24)), label_text)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, max(0, y1 - 24)), label_text, fill="white")

        results.append({
            "Face": len(results) + 1,
            "Prediction": label,
            "Distance": round(distance, 4),
            "Detection score": round(float(probability), 4),
            "Result": "Matched" if matched else "Unidentified",
        })

    return annotated, results


left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("1. Choose an image source")
    image_source = st.radio(
        "Select how you want to provide the image",
        ["Upload image", "Use camera"],
        horizontal=True,
        help="Camera access is requested by your browser. Allow permission when prompted.",
    )

    uploaded_image = None
    if image_source == "Upload image":
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            help="For best results, upload a clear, well-lit photo with visible faces.",
        )
        if uploaded_file:
            try:
                uploaded_image = Image.open(uploaded_file)
                st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
            except Exception:
                st.error("This image could not be opened. Please upload a valid JPG or PNG file.")
    else:
        camera_file = st.camera_input(
            "Take a photo using your camera",
            help="Allow camera permission in your browser. On desktop, use a connected webcam.",
        )
        if camera_file:
            try:
                uploaded_image = Image.open(camera_file)
                st.image(uploaded_image, caption="Captured camera image", use_container_width=True)
            except Exception:
                st.error("The captured image could not be opened. Please try again.")

with right:
    st.subheader("2. Recognition settings")
    st.markdown(
        "The app compares each detected face embedding with the saved member centroids."
    )
    threshold_input = st.number_input(
        "Distance threshold",
        min_value=0.10,
        max_value=2.00,
        value=0.60,
        step=0.05,
        help="The notebook saved 0.60 as the default threshold. Lower values are stricter; higher values may create false matches.",
    )
    st.caption("A lower distance means the face embedding is closer to a saved member centroid.")

if uploaded_image is not None:
    st.divider()
    st.subheader("3. Recognition result")
    try:
        device, backbone, mtcnn, classes, centroids, saved_threshold = load_recognition_model(str(MODEL_PATH))
        # Use the threshold saved with the checkpoint as the default when the slider remains at 0.60.
        effective_threshold = float(threshold_input)
        st.caption(f"Model classes: {', '.join(map(str, classes))} · Device: {device} · Saved threshold: {saved_threshold:.2f}")

        with st.spinner("Detecting faces and comparing embeddings..."):
            annotated_image, results = recognize_image(
                uploaded_image, device, backbone, mtcnn, classes, centroids, effective_threshold
            )

        result_col, details_col = st.columns([1.05, 0.95], gap="large")
        with result_col:
            st.image(annotated_image, caption="Recognition output", use_container_width=True)
        with details_col:
            if results:
                matched_count = sum(row["Result"] == "Matched" for row in results)
                st.metric("Faces processed", len(results))
                st.metric("Recognized members", matched_count)
                st.dataframe(results, use_container_width=True, hide_index=True)
            else:
                st.warning("No clear face was detected. Try a sharper, front-facing photo with better lighting.")

        st.info(
            "This is a similarity-based demo, not a secure identity-verification system. "
            "Recognition can be wrong, especially with low-quality images, pose changes, or similar-looking faces."
        )
    except Exception as exc:
        st.error(f"Could not run recognition: {exc}")
        st.markdown(
            "**Check:** `facenet_model.pt` must be in the same project folder as `app.py`, "
            "and it must be the checkpoint saved by your notebook."
        )
else:
    st.info("Upload an image or capture a photo with your camera to begin. The app uses the saved model; it does not train the model.")
    st.markdown(
        '<p class="small-note">Privacy tip: upload only photos you have permission to use. '
        'This demo processes the image for recognition and does not include code to save uploaded images.</p>',
        unsafe_allow_html=True,
    )
