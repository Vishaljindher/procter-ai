#procter-ai/time_logic.py
import time
from collections import deque

class HeadMovementTracker:
    def __init__(self, window_seconds=5):
        self.window = window_seconds
        self.events = deque()

    def update(self, movement):
        now = time.time()
        self.events.append((now, movement))

        while self.events and now - self.events[0][0] > self.window:
            self.events.popleft()

    def suspicious_count(self):
        return sum(1 for _, m in self.events if m in ["LEFT", "RIGHT"])
