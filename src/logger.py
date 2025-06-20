import logging
import os
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Set up logging"""
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{LOGS_DIR}/chatbot_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
