import threading
import logging
import os
import signal

logger = logging.getLogger(__name__)

class InferenceWatchdog:
    def __init__(self, timeout_sec=60):
        self.timeout = timeout_sec
        self.timer = None

    def start(self, gpu_id, session_id=None):
        def _timeout():
            logger.error(
                f"[InferenceWatchdog] "
                f"GPU {gpu_id} inference timeout, "
                f"session={session_id}"
            )
            # 直接杀掉当前进程（最干净）
            os.kill(os.getpid(), signal.SIGTERM)

        self.timer = threading.Timer(self.timeout, _timeout)
        self.timer.daemon = True
        self.timer.start()

    def stop(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
