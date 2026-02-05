#procter-ai/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64, cv2, numpy as np
import time
from eye_time_tracker import EyeDirectionTracker

from cheating_engine import CheatingEngine
from head_movement import detect_faces, detect_head_movement
from eye_tracking import detect_eye_state
from time_logic import HeadMovementTracker

app = FastAPI()

# 🔥 GLOBAL SINGLE INSTANCES
engine = CheatingEngine()
tracker = HeadMovementTracker(window_seconds=5)
eye_tracker = EyeDirectionTracker(threshold_seconds=10)

# 👤 FACE-MISSING TIMER
last_face_seen_time = time.time()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
def analyze(data: dict):
    global last_face_seen_time

    # 1️⃣ Decode image
    img_data = data["image"]
    img_bytes = base64.b64decode(img_data.split(",")[1])
    frame = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    # 2️⃣ FACE DETECTION
    faces = detect_faces(frame)
    face_count = len(faces)

    # 🟢 DEBUG: draw boxes (testing only)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    now = time.time()

    # ⏳ FACE GRACE LOGIC (VERY IMPORTANT)
    if face_count > 0:
        last_face_seen_time = now
    else:
        # agar face 3 sec se kam gayab hai → ignore
        if now - last_face_seen_time < 3:
            face_count = 1   # pretend face is present

    # 3️⃣ EYE TRACKING
    eye_state = detect_eye_state(frame, faces)
    eye_cheating, eye_time = eye_tracker.update(eye_state)

    # 4️⃣ HEAD MOVEMENT
    movement = detect_head_movement(frame, faces)
    tracker.update(movement)
    head_suspicious = tracker.suspicious_count()

    # 5️⃣ CHEATING ENGINE
    score = engine.update(
        face_count=face_count,
        head_suspicious=head_suspicious,
        eye_state=eye_state
    )
    engine.decay()

    # 6️⃣ RESPONSE
    return {
        "faces": face_count,
        "eye_state": eye_state,
        "eye_cheating": eye_cheating,   # 🔥 NEW
        "eye_time": eye_time, 
        "head_suspicious": head_suspicious,
        "score": score,
        "status": engine.get_status()
    }
