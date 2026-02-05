#procter-ai/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
import time
import numpy as np
import mediapipe as mp

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MediaPipe ----------------
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="blaze_face_short_range.tflite"
    ),
    running_mode=RunningMode.IMAGE
)

face_detector = FaceDetector.create_from_options(options)

# ---------------- STATE ----------------
last_face_time = time.time()
cheating_score = 0
cheating_locked = False

# ---------------- API ----------------
@app.post("/analyze")
def analyze(data: dict):
    global last_face_time, cheating_score, cheating_locked

    # Decode image
    img_data = data["image"]
    img_bytes = base64.b64decode(img_data.split(",")[1])
    frame = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect faces
    result = face_detector.detect(mp_image)

    faces = 0
    for det in result.detections:
        if det.categories[0].score > 0.7:
            faces += 1

    now = time.time()

    # ---------------- LOGIC ----------------
    if cheating_locked:
        return {
            "faces": faces,
            "cheating_score": cheating_score,
            "status": "cheating"
        }

    if faces == 1:
        last_face_time = now
        cheating_score = max(cheating_score - 2, 0)

    elif faces == 0:
        missing_time = now - last_face_time
        if missing_time > 6:
            cheating_score += 1

    elif faces > 1:
        cheating_score += 4

    # ---------------- STATUS ----------------
    if cheating_score >= 8:
        cheating_locked = True
        status = "cheating"
    elif cheating_score >= 4:
        status = "warning"
    else:
        status = "ok"

    return {
        "faces": faces,
        "cheating_score": cheating_score,
        "status": status
    }
