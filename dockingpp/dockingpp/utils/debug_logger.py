"""Logger de debug estruturado (JSONL) com overhead mínimo."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any


class DebugLogger:
    """Logger simples para eventos JSONL de debug."""

    _LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30}

    def __init__(self, enabled: bool, path: str, level: str = "INFO") -> None:
        self.enabled = bool(enabled)
        self.path = path
        self.level = self._normalize_level(level)
        self._handle = None
        self._events_since_flush = 0
        self.flush_every = 50
        self.run_id: str | None = None

        if not self.enabled:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning("Falha ao abrir debug log em %s", path)
            self.enabled = False

    def _normalize_level(self, level: str) -> int:
        if not isinstance(level, str):
            return self._LEVELS["INFO"]
        return self._LEVELS.get(level.upper(), self._LEVELS["INFO"])

    def _should_log(self, level: str) -> bool:
        return self._normalize_level(level) >= self.level

    def log(self, event: dict[str, Any], level: str = "INFO") -> None:
        if not self.enabled or self._handle is None:
            return
        if not self._should_log(level):
            return
        try:
            payload = dict(event)
            payload.setdefault("ts_utc", datetime.utcnow().isoformat() + "Z")
            if "run_id" not in payload and self.run_id:
                payload["run_id"] = self.run_id
            payload.setdefault("pid", os.getpid())
            payload.setdefault("thread", threading.get_ident())
            self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._events_since_flush += 1
            if self._events_since_flush >= self.flush_every:
                self._handle.flush()
                self._events_since_flush = 0
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning("Falha ao escrever evento de debug.")

    def close(self) -> None:
        if not self.enabled or self._handle is None:
            return
        try:
            self._handle.flush()
            self._handle.close()
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning("Falha ao fechar debug log.")
        finally:
            self._handle = None
            self._events_since_flush = 0
