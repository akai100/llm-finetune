import hashlib
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class DataAuditor:
    def __init__(self):
        self.hashes = set()
        self.lengths = []
        self.duplicate_count = 0
        self.bad_format_count = 0

    def _hash_text(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def audit_sample(self, sample):
        text = sample.get("text", "")

        # 1️. 重复检测
        h = self._hash_text(text)
        if h in self.hashes:
            self.duplicate_count += 1
            return False
        self.hashes.add(h)

        # 2️. 格式检测
        if "### Response:" not in text:
            self.bad_format_count += 1
            return False

        # 3️. 长度统计
        self.lengths.append(len(text.split()))

        return True

    def report(self):
        if self.lengths:
            avg_len = sum(self.lengths) / len(self.lengths)
        else:
            avg_len = 0

        logger.warning(
            f"[DataAudit] duplicates={self.duplicate_count}, "
            f"bad_format={self.bad_format_count}, "
            f"avg_len={avg_len:.2f}"
        )
