#procter-ai/head_movement.py
import cv2

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------
# FACE DETECTION
# -------------------------------
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )
    return faces


# -------------------------------
# HEAD MOVEMENT DETECTION
# -------------------------------
def detect_head_movement(frame, faces):
    if len(faces) == 0:
        return "NO_FACE"

    h, w = frame.shape[:2]
    frame_center_x = w // 2

    # Use FIRST face only (main candidate)
    (x, y, fw, fh) = faces[0]
    face_center_x = x + fw // 2

    OFFSET = 40  # sensitivity (tune later)

    if face_center_x < frame_center_x - OFFSET:
        return "LEFT"
    elif face_center_x > frame_center_x + OFFSET:
        return "RIGHT"
    else:
        return "CENTER"
