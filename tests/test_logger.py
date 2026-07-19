import logging
from pathlib import Path

from voiceinput.logger import setup_logger


def test_disabled_logger_uses_null_handler(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = setup_logger(log_dir, enabled=False)
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert not log_dir.exists()  # disabled なら log_dir は作らない


def test_enabled_logger_writes_to_rotating_file(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = setup_logger(log_dir, enabled=True)
    logger.info("hello voiceinput")
    log_file = log_dir / "voiceinput.log"
    assert log_file.exists()
    assert "hello voiceinput" in log_file.read_text()


def test_setup_logger_clears_previous_handlers(tmp_path: Path):
    setup_logger(tmp_path / "a", enabled=True)
    logger = setup_logger(tmp_path / "b", enabled=True)
    # 2 回呼んでも既存 handler を引きずらない
    assert len(logger.handlers) == 1
