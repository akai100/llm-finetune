from dataclasses import dataclass
import time

@dataclass
class SessionState:
    session_id: str
    gpu_id: int
    last_active: float
    past_key_values: object = None
