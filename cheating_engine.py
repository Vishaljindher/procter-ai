#procter-ai/cheating_engine.py
import time


class CheatingEngine:
    def __init__(self):
        self.score = 0
        self.last_update = 0

        # ⏱ timing controls
        self.cooldown = 1.5          # seconds between scoring
        self.last_event_time = {}    # per-event cooldown

        # ⚖️ thresholds
        self.WARNING_SCORE = 4
        self.CHEATING_SCORE = 8

    # ----------------------------------
    # INTERNAL: safe add score
    # ----------------------------------
    def _add_score(self, event, points, event_cooldown=2):
        now = time.time()

        last_time = self.last_event_time.get(event, 0)
        if now - last_time < event_cooldown:
            return  # ignore spam

        self.score += points
        self.last_event_time[event] = now

    # ----------------------------------
    # MAIN UPDATE
    # ----------------------------------
    def update(self, face_count, head_suspicious, eye_state):
        now = time.time()

        # ⛔ global rate limit
        if now - self.last_update < self.cooldown:
            return self.score

        # 👁️ Eye logic (LOW weight)
        if eye_state in ["EYES_LEFT", "EYES_RIGHT"]:
            self._add_score("EYE_AWAY", 2, event_cooldown=2)

        elif eye_state == "EYES_CLOSED":
            self._add_score("EYES_CLOSED", 3, event_cooldown=3)

        # 👤 Face missing
        if face_count == 0:
            self._add_score("NO_FACE", 3, event_cooldown=3)

        # 👥 Multiple faces (HIGH severity)
        if face_count >= 2:
            self._add_score("MULTIPLE_FACES", 10, event_cooldown=5)

        # ↔️ Head movement
        if head_suspicious >= 3:
            self._add_score("HEAD_MOVEMENT", 2, event_cooldown=2)

        self.last_update = now
        return self.score

    # ----------------------------------
    # SCORE DECAY
    # ----------------------------------
    def decay(self):
        if self.score > 0:
            self.score = max(0, self.score - 0.5)

    # ----------------------------------
    # STATUS HELPERS
    # ----------------------------------
    def is_cheating(self):
        return self.score >= self.CHEATING_SCORE

    def is_warning(self):
        return self.WARNING_SCORE <= self.score < self.CHEATING_SCORE

    def get_status(self):
        if self.is_cheating():
            return "cheating"
        elif self.is_warning():
            return "warning"
        return "ok"
