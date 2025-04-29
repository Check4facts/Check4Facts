import logging
import sys
import os

ENV = os.getenv("ENV", "dev")
LOG_LEVEL = logging.DEBUG if ENV == "dev" else logging.INFO

# ANSI color codes
RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[90m",      # Bright Black / Gray
    logging.INFO: "\033[32m",       # Green
    logging.WARNING: "\033[33m",    # Yellow
    logging.ERROR: "\033[31m",      # Red
    logging.CRITICAL: "\033[1;31m", # Bold Red
}

class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = f"{COLORS.get(record.levelno, '')}{record.levelname}{RESET}:{' ' * (10 - (len(record.levelname) + 1))}{record.getMessage()}"
        return log_fmt

def get_logger(name: str = "check4facts") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Avoid adding multiple handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(LOG_LEVEL)
        handler.setFormatter(CustomFormatter())

        logger.addHandler(handler)
        logger.propagate = False  # Prevent logs from bubbling to the root logger

    return logger
