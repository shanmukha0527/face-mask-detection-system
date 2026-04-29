# ─────────────────────────────────────────────
#  Face Mask Detector — Streamlit Web App
#  Run: streamlit run app.py
# ─────────────────────────────────────────────

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector", page_icon="😷 😁", layout="centered"
)

# ── Constants ─────────────────────────────────
MODEL_PATH = "../MOdel/mask_detector.h5"
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

#  Must match exactly what was used during training
CLASSES = ["with_mask", "with_out mask"]

COLOR_MAP = {
    "with_mask": (0, 200, 0),  # green
    "with_out mask": (200, 0, 0),  # red
}
MIN_CONF = 0.3  # lowered to detect more faces


# ── Load models (cached) ──────────────────────
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
    mask_net = load_model(MODEL_PATH)
    return face_net, mask_net


# ── Inference ─────────────────────────────────
def predict_image(image_pil, face_net, mask_net):
    frame = np.array(image_pil.convert("RGB"))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    (h, w) = frame_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []
    annotated = frame.copy()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < MIN_CONF:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        (x1, y1) = (max(0, x1), max(0, y1))
        (x2, y2) = (min(w - 1, x2), min(h - 1, y2))

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_input = cv2.resize(face, (224, 224))
        face_input = img_to_array(face_input)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        pred = mask_net.predict(face_input, verbose=0)[0]
        class_idx = np.argmax(pred)
        label = CLASSES[class_idx]
        prob = float(pred[class_idx])
        color = COLOR_MAP[label]

        results.append(
            {
                "face_id": len(results) + 1,
                "label": label,
                "confidence": prob,
                "raw_pred": pred.tolist(),
            }
        )

        # Draw on frame
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color[::-1], 2)
        display_label = "With Mask" if label == "with_mask" else "No Mask ❌"
        text = f"{display_label}: {prob * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = y1 - 10 if y1 - 10 > 10 else y2 + th + 10
        cv2.rectangle(
            annotated, (x1, ly - th - 4), (x1 + tw + 4, ly + 4), color[::-1], cv2.FILLED
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 2, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return Image.fromarray(annotated), results


# ── Fallback: classify whole image (no face detected) ──
def classify_whole_image(image_pil, mask_net):
    img = image_pil.convert("RGB").resize((224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    pred = mask_net.predict(img, verbose=0)[0]
    class_idx = np.argmax(pred)
    return CLASSES[class_idx], float(pred[class_idx]), pred.tolist()


# ── UI ────────────────────────────────────────
st.title(" Face Mask Detector")
st.markdown("Upload a photo to detect whether people are wearing face masks.")

st.sidebar.header("About")
st.sidebar.info("Built with **MobileNetV2** + **OpenCV**.\n\n With Mask\n Without Mask")
st.sidebar.header("Model Info")
st.sidebar.markdown("- Architecture: MobileNetV2")
st.sidebar.markdown("- Input size: 224 × 224")
st.sidebar.markdown("- Training epochs: 20")

# Load models
try:
    face_net, mask_net = load_models()
    st.sidebar.success(" Models loaded successfully")
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Detecting …"):
        annotated_img, results = predict_image(image, face_net, mask_net)

    if not results:
        st.warning(" No face detected by OpenCV. Running whole-image classification …")
        label, prob, raw = classify_whole_image(image, mask_net)
        display = " With Mask" if label == "with_mask" else " No Mask"
        st.metric("Prediction", display, f"{prob * 100:.1f}% confidence")

        # Debug info
        with st.expander("🔍 Debug — raw model output"):
            st.write(f"with_mask score:    {raw[0]:.4f}")
            st.write(f"with_out mask score: {raw[1]:.4f}")
    else:
        st.image(annotated_img, caption="Detection result", use_column_width=True)
        st.subheader(f"Detected {len(results)} face(s)")

        for r in results:
            display = "✅ With Mask" if r["label"] == "with_mask" else "❌ No Mask"
            st.metric(
                f"Face {r['face_id']}",
                display,
                f"{r['confidence'] * 100:.1f}% confidence",
            )

            # Debug info
            with st.expander(f"Debug — Face {r['face_id']} raw scores"):
                st.write(f"with_mask score:     {r['raw_pred'][0]:.4f}")
                st.write(f"with_out mask score: {r['raw_pred'][1]:.4f}")

        with_mask_count = sum(1 for r in results if r["label"] == "with_mask")
        without_mask_count = len(results) - with_mask_count

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("✅ With Mask", with_mask_count)
        c2.metric("❌ Without Mask", without_mask_count)

        if without_mask_count == 0:
            st.success("All detected faces are wearing masks!")
        else:
            st.error(f"{without_mask_count} person(s) detected WITHOUT a mask!")

        buf = io.BytesIO()
        annotated_img.save(buf, format="PNG")
        st.download_button(
            "Download result", buf.getvalue(), "mask_detection_result.png", "image/png"
        )
else:
    st.info("👆 Upload an image to get started.")
