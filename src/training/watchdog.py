import time
import threading
import logging
import os
import signal

logger = logging.getLogger(__name__)

class TrainingWatchdog:
    def __init__(self, timeout=600):
        self.timeout = timeout
        self.last_heartbeat = time.time()
        self._stop = False

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def start(self):
        t = threading.Thread(target=self._monitor, daemon=True)
        t.start()

    def _monitor(self):
        while not self._stop:
            if time.time() - self.last_heartbeat > self.timeout:
                logger.error("[Watchdog] Training stalled, killing process.")
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(10)

    def stop(self):
        self._stop = True
