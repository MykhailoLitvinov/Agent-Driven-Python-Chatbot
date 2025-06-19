import logging
import os
import tempfile
from unittest.mock import patch

import pytest

from src.logger import setup_logging


@pytest.fixture
def temp_logs_dir():
    """Create a temporary logs directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("src.logger.LOGS_DIR", temp_dir):
            yield temp_dir


def test_setup_logging_creates_logs_directory(temp_logs_dir):
    """Test that setup_logging creates the logs directory"""
    # Remove temp directory to test creation
    os.rmdir(temp_logs_dir)
    assert not os.path.exists(temp_logs_dir)

    setup_logging()

    assert os.path.exists(temp_logs_dir)


def test_setup_logging_directory_already_exists(temp_logs_dir):
    """Test that setup_logging works when logs directory already exists"""
    assert os.path.exists(temp_logs_dir)

    # Should not raise an exception
    logger = setup_logging()

    assert isinstance(logger, logging.Logger)
    assert os.path.exists(temp_logs_dir)


def test_setup_logging_returns_logger(temp_logs_dir):
    """Test that setup_logging returns a Logger instance"""
    logger = setup_logging()

    assert isinstance(logger, logging.Logger)
    assert logger.name == "chatbot"


def test_setup_logging_logger_level(temp_logs_dir):
    """Test that logger is set to INFO level"""
    logger = setup_logging()

    assert logger.level == logging.INFO


def test_setup_logging_formatter(temp_logs_dir):
    """Test that the correct formatter is applied to the handler"""
    logger = setup_logging()

    handler = logger.handlers[0]
    formatter = handler.formatter

    assert isinstance(formatter, logging.Formatter)
    assert formatter._fmt == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def test_setup_logging_multiple_calls_same_logger(temp_logs_dir):
    """Test that multiple calls to setup_logging return the same logger instance"""
    logger1 = setup_logging()
    logger2 = setup_logging()

    # Should return the same logger instance (by name)
    assert logger1.name == logger2.name
    assert logger1 is logger2


def test_setup_logging_multiple_calls_handlers(temp_logs_dir):
    """Test that multiple calls don't create duplicate handlers"""
    logger1 = setup_logging()
    initial_handler_count = len(logger1.handlers)

    logger2 = setup_logging()

    # Should not add duplicate handlers
    assert len(logger2.handlers) >= initial_handler_count


@patch("src.logger.os.makedirs")
def test_setup_logging_makedirs_called(mock_makedirs, temp_logs_dir):
    """Test that os.makedirs is called with correct parameters"""
    setup_logging()

    mock_makedirs.assert_called_once_with(temp_logs_dir, exist_ok=True)


def test_setup_logging_can_write_to_log_file(temp_logs_dir):
    """Test that the logger can actually write to the log file"""
    logger = setup_logging()

    test_message = "Test log message"
    logger.info(test_message)

    # Find the log file
    log_files = [f for f in os.listdir(temp_logs_dir) if f.startswith("chatbot_") and f.endswith(".log")]
    assert len(log_files) == 1

    log_file_path = os.path.join(temp_logs_dir, log_files[0])
    assert os.path.exists(log_file_path)

    # Check that the message was written
    with open(log_file_path, "r") as f:
        content = f.read()
        assert test_message in content
        assert "chatbot" in content
        assert "INFO" in content


def test_setup_logging_log_levels(temp_logs_dir):
    """Test that logger respects log levels"""
    logger = setup_logging()

    log_files = [f for f in os.listdir(temp_logs_dir) if f.startswith("chatbot_") and f.endswith(".log")]
    if log_files:
        # Clear existing log file
        log_file_path = os.path.join(temp_logs_dir, log_files[0])
        with open(log_file_path, "w") as f:
            f.write("")

    # Log messages at different levels
    logger.debug("Debug message")  # Should not appear (below INFO)
    logger.info("Info message")  # Should appear
    logger.warning("Warning message")  # Should appear
    logger.error("Error message")  # Should appear

    # Check log file content
    log_files = [f for f in os.listdir(temp_logs_dir) if f.startswith("chatbot_") and f.endswith(".log")]
    assert len(log_files) == 1

    log_file_path = os.path.join(temp_logs_dir, log_files[0])
    with open(log_file_path, "r") as f:
        content = f.read()

        assert "Debug message" not in content  # DEBUG level filtered out
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content


@patch("src.logger.os.makedirs")
def test_setup_logging_makedirs_exception_handling(mock_makedirs, temp_logs_dir):
    """Test that setup_logging handles os.makedirs exceptions gracefully"""
    mock_makedirs.side_effect = OSError("Permission denied")

    # Should not raise an exception due to exist_ok=True behavior
    with pytest.raises(OSError):
        setup_logging()
