#proctor-ai/main.py
import cv2
from head_movement import detect_faces, detect_head_movement
from time_logic import HeadMovementTracker
from cheating_engine import CheatingEngine
from eye_tracking import detect_eye_state

# -----------------------------
# INITIALIZE
# -----------------------------
cap = cv2.VideoCapture(0)

tracker = HeadMovementTracker(window_seconds=5)
engine = CheatingEngine()

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Face detection
    faces = detect_faces(frame)
    face_count = len(faces)

    # 2️⃣ Head movement
    movement = detect_head_movement(frame, faces)
    eye_state = detect_eye_state(frame, faces)

    # 3️⃣ Time-based tracking
    tracker.update(movement)
    suspicious_moves = tracker.suspicious_count()

    # 4️⃣ Cheating score update
    score = engine.update( face_count=face_count,
    head_suspicious=suspicious_moves,
    eye_state=eye_state)
    engine.decay()
    status = "CHEATING" if engine.is_cheating() else "OK"

    # -----------------------------
    # VISUALS
    # -----------------------------
    # Draw face boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Faces: {face_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Head: {movement}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Suspicious (5s): {suspicious_moves}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.putText(frame, f"Score: {score}", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Status: {status}", (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Eyes: {eye_state}", (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("AI Proctoring System", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
