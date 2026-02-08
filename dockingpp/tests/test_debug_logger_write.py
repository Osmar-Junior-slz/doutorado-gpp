"""Testes para DebugLogger com escrita."""

from __future__ import annotations

import json

from dockingpp.utils.debug_logger import DebugLogger


def test_debug_logger_write(tmp_path):
    """PT-BR: eventos devem ser gravados como JSONL."""

    path = tmp_path / "debug.jsonl"
    logger = DebugLogger(enabled=True, path=str(path))
    logger.log({"type": "evento_a"})
    logger.log({"type": "evento_b", "run_id": "run-123"})
    logger.close()

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["type"] == "evento_a"
    assert payloads[1]["type"] == "evento_b"
    assert "ts_utc" in payloads[0]
    assert "pid" in payloads[0]
    assert "thread" in payloads[0]
