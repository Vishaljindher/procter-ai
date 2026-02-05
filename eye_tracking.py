#procter-ai/eye_tracking.py
import cv2

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_eye_state(frame, faces):
    if len(faces) == 0:
        return "NO_EYES"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Take first face
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(face_roi, 1.3, 5)

    if len(eyes) == 0:
        return "EYES_CLOSED"

    # Take first eye
    (ex, ey, ew, eh) = eyes[0]
    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

    _, thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)

    h, w = thresh.shape
    left = thresh[:, :w//2]
    right = thresh[:, w//2:]

    left_white = cv2.countNonZero(left)
    right_white = cv2.countNonZero(right)

    if left_white > right_white * 1.3:
        return "EYES_RIGHT"
    elif right_white > left_white * 1.3:
        return "EYES_LEFT"
    else:
        return "EYES_CENTER"
