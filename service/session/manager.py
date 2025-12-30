import time
from threading import Lock
from service.session.schemas import SessionState

class SessionManager:
    def __init__(self, ttl=1800):
        self.sessions = {}
        self.lock = Lock()
        self.ttl = ttl

    def get(self, session_id):
        with self.lock:
            s = self.sessions.get(session_id)
            if s:
                s.last_active = time.time()
            return s

    def create(self, session_id, gpu_id):
        with self.lock:
            s = SessionState(
                session_id=session_id,
                gpu_id=gpu_id,
                last_active=time.time()
            )
            self.sessions[session_id] = s
            return s

    def cleanup(self):
        now = time.time()
        with self.lock:
            expired = [
                k for k, v in self.sessions.items()
                if now - v.last_active > self.ttl
            ]
            for k in expired:
                del self.sessions[k]
