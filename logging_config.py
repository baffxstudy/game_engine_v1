"""
Logging configuration for the Football Match Analysis Engine.

Sets up structured logging with file rotation and console output.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import (
    LOG_DIR,
    LOG_FILE,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_LEVEL
)


def safe_log(message: str) -> str:
    """Convert Unicode emojis to ASCII text for Windows compatibility."""
    replacements = {
        '✅': '[OK]',
        '🚀': '[START]',
        '🔧': '[PROCESS]',
        '📊': '[DATA]',
        '🔍': '[ANALYZE]',
        '⚠️': '[WARN]',
        '❌': '[ERROR]',
        '🎯': '[TARGET]',
        '📦': '[FEAT]',
        '📍': '[LOC]',
        '📝': '[LOG]',
        '📁': '[FILE]',
        '🔌': '[CONNECT]',
        '⚙️': '[CONFIG]',
        '💥': '[CRASH]',
        '🔒': '[SECURE]',
        '📈': '[STATS]',
        '🔍': '[SEARCH]',
        '💰': '[MONEY]',
        '⚽': '[FOOTBALL]',
        '🎰': '[CASINO]',
    }
    for emoji, text in replacements.items():
        message = message.replace(emoji, text)
    return message


def setup_logging() -> logging.Logger:
    """
    Configure logging for the application.
    
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    )
    
    # Setup Rotating File Handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Setup Console Handler - ASCII only for Windows compatibility
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger("engine_api")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    # Log initialization
    logger.info(safe_log("=" * 80))
    logger.info(safe_log("[START] Logging system initialized"))
    logger.info(safe_log(f"[CONFIG] Log Level: {LOG_LEVEL}"))
    logger.info(safe_log(f"[CONFIG] Log File: {LOG_FILE}"))
    logger.info(safe_log("=" * 80))
    
    return logger
