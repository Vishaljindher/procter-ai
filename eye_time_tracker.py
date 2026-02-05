# eye_time_tracker.py
import time

class EyeDirectionTracker:
    def __init__(self, threshold_seconds=10):
        self.threshold = threshold_seconds
        self.last_center_time = time.time()
        self.cheating = False

    def update(self, eye_state):
        now = time.time()

        if eye_state == "EYES_CENTER":
            self.last_center_time = now
            self.cheating = False
        else:
            if now - self.last_center_time >= self.threshold:
                self.cheating = True

        return self.cheating, round(now - self.last_center_time, 1)
