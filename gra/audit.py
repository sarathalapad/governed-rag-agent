"""Tamper-evident audit log.

Each line stores the hash of the previous line, so editing, deleting or
reordering any past event breaks the chain. If GRA_AUDIT_KEY is set, hashes
are HMACs, so someone with write access cannot simply recompute the chain.
"""
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: Path, key: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = key if key is not None else os.environ.get("GRA_AUDIT_KEY")
        self._key = key.encode() if key else None

    def _digest(self, body: str) -> str:
        if self._key:
            return hmac.new(self._key, body.encode(), hashlib.sha256).hexdigest()
        return hashlib.sha256(body.encode()).hexdigest()

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def append(self, event: str, **data) -> dict:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event,
                  "data": data, "prev": self._last_hash()}
        body = json.dumps(record, sort_keys=True, ensure_ascii=False)
        record["hash"] = self._digest(body)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def verify(self) -> tuple[bool, int | None]:
        """Return (True, None) if the chain is intact, else (False, bad_line_number)."""
        if not self.path.exists():
            return True, None
        prev = GENESIS
        with self.path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    stored = record.pop("hash")
                except (ValueError, KeyError):
                    return False, n
                body = json.dumps(record, sort_keys=True, ensure_ascii=False)
                if record.get("prev") != prev or not hmac.compare_digest(stored, self._digest(body)):
                    return False, n
                prev = stored
        return True, None
