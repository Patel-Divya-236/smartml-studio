"""Logging configuration for SmartML Studio.

Call ``setup_logging()`` once at application startup (in ``app.py``)
to configure console and rotating-file handlers for the entire app.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "smartml.log")
_LOG_FORMAT = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root 'smartml' logger with console and file handlers.

    Args:
        level: Logging level (default ``logging.INFO``).
    """
    os.makedirs(_LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger("smartml")
    root_logger.setLevel(level)

    # Avoid duplicate handlers on hot-reload
    if root_logger.handlers:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialised — console + file (%s).", _LOG_FILE)
