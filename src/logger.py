import logging
import os
from datetime import datetime


def setup_logging() -> logging.Logger:
    """Set up logging"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Configure logger
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(f"logs/chatbot_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
