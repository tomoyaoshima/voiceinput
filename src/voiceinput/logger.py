import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(log_dir: Path, enabled: bool, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("voiceinput")
    logger.setLevel(level)
    logger.handlers.clear()
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "voiceinput.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger
