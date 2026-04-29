# ─────────────────────────────────────────────
#  Face Mask Detector — Real-Time Detection
#  Uses: OpenCV + trained MobileNetV2 model
#  Run : python detect_realtime.py
# ─────────────────────────────────────────────

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import time

# ── Config ────────────────────────────────────
MODEL_PATH = "../MOdel/mask_detector.h5"
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
MIN_CONF = 0.3

# Must match training class names exactly
CLASSES = ["with_mask", "with_out mask"]

# Box colours (BGR)
COLOR_MAP = {
    "with_mask": (0, 200, 0),  # green
    "with_out mask": (0, 0, 200),  # red
}


# ── Helper: detect & classify faces ──────────
def detect_and_predict(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs, preds = [], [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < MIN_CONF:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    if faces:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32, verbose=0)

    return locs, preds


# ── Draw overlay ──────────────────────────────
def draw_prediction(frame, loc, pred):
    (startX, startY, endX, endY) = loc
    class_idx = np.argmax(pred)
    label = CLASSES[class_idx]
    prob = pred[class_idx]
    color = COLOR_MAP.get(label, (200, 200, 200))

    display = "With Mask" if label == "with_mask" else "NO MASK!"
    text = f"{display}: {prob * 100:.1f}%"

    # Bounding box
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = startY - 10 if startY - 10 > 10 else startY + th + 10
    cv2.rectangle(
        frame,
        (startX, label_y - th - 4),
        (startX + tw + 4, label_y + 4),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (startX + 2, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


# ── Main ──────────────────────────────────────
def main():
    print("[INFO] Loading models …")
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
    mask_net = load_model(MODEL_PATH)

    print("[INFO] Starting webcam … (press Q to quit)")
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame — exiting")
            break

        frame = cv2.resize(frame, (800, 600))

        locs, preds = detect_and_predict(frame, face_net, mask_net)

        for loc, pred in zip(locs, preds):
            draw_prediction(frame, loc, pred)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_time
        fps = frame_count / max(elapsed, 0.001)
        if elapsed >= 1.0:
            frame_count = 0
            fps_time = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Alert if no mask detected
        if any(np.argmax(p) == 1 for p in preds):
            cv2.putText(
                frame,
                "⚠ NO MASK DETECTED!",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Face Mask Detector — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")


if __name__ == "__main__":
    main()
