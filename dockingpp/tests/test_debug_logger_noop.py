"""Testes para DebugLogger em modo no-op."""

from __future__ import annotations

from dockingpp.utils.debug_logger import DebugLogger


def test_debug_logger_noop(tmp_path):
    """PT-BR: quando desabilitado, não deve criar arquivo."""

    path = tmp_path / "debug.jsonl"
    logger = DebugLogger(enabled=False, path=str(path))
    logger.log({"type": "noop"})
    logger.close()
    assert not path.exists()
