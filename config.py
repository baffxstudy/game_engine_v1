"""
Configuration management for the Football Match Analysis Engine.

Centralizes all configuration settings, environment variables, and constants.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Logging Configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "engine.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 10
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Engine Configuration
ENGINE_VERSION = "2.2.0"
ENABLE_MONTE_CARLO = os.getenv("ENABLE_MONTE_CARLO", "true").lower() == "true"
NUM_SIMULATIONS = int(os.getenv("NUM_SIMULATIONS", "10000"))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
API_TITLE = "Intelligent Football Slip Builder API"
API_DESCRIPTION = (
    "Dual-strategy slip generation (Balanced + MaxWin) with Monte Carlo optimization. "
    "Performs all calculations and intelligence for the football match analysis platform."
)

# Callback Configuration (Laravel Integration)
LARAVEL_BASE_URL = os.getenv("LARAVEL_BASE_URL", "http://localhost:8000")
CALLBACK_TIMEOUT = float(os.getenv("CALLBACK_TIMEOUT", "300.0"))

# Performance Configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300.0"))  # 5 minutes

# Feature Flags
FEATURE_SPO_ENABLED = os.getenv("FEATURE_SPO_ENABLED", "true").lower() == "true"
FEATURE_INSIGHT_ENGINE = os.getenv("FEATURE_INSIGHT_ENGINE", "true").lower() == "true"

# Validation Configuration
MIN_MATCHES_BALANCED = 3
MIN_MATCHES_MAXWIN = 3
MIN_MATCHES_COMPOUND = 7
MAX_MATCHES = int(os.getenv("MAX_MATCHES", "50"))
MIN_STAKE = float(os.getenv("MIN_STAKE", "0.01"))
MAX_STAKE = float(os.getenv("MAX_STAKE", "100000.0"))

# Default Strategy
DEFAULT_STRATEGY = "balanced"

# Supported Strategies
SUPPORTED_STRATEGIES = ["balanced", "maxwin", "compound"]

# Response Configuration
DEFAULT_SLIP_COUNT = 50
OPTIMIZED_SLIP_COUNT = 20  # After SPO

# Betting Configuration (for automated betting module)
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "30000"))  # milliseconds
VIEWPORT_WIDTH = int(os.getenv("VIEWPORT_WIDTH", "1920"))
VIEWPORT_HEIGHT = int(os.getenv("VIEWPORT_HEIGHT", "1080"))
BETTING_TEST_MODE = os.getenv("BETTING_TEST_MODE", "true").lower() == "true"
TEST_SITE_URL = os.getenv("TEST_SITE_URL", "https://www.example.com")
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Error Messages
ERROR_MESSAGES = {
    "ENGINE_UNAVAILABLE": "Slip builder engine not available. Please check server logs.",
    "INVALID_PAYLOAD": "Invalid payload structure",
    "INVALID_STRATEGY": "Invalid strategy specified",
    "INSUFFICIENT_MATCHES": "Insufficient matches for selected strategy",
    "MARKET_INTEGRITY": "Market data integrity error",
    "GENERATION_FAILED": "Slip generation failed",
    "CALLBACK_FAILED": "Failed to post results to callback URL",
    "SPO_UNAVAILABLE": "Portfolio optimization not available",
}


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "engine": {
            "version": ENGINE_VERSION,
            "monte_carlo_enabled": ENABLE_MONTE_CARLO,
            "num_simulations": NUM_SIMULATIONS,
        },
        "api": {
            "host": API_HOST,
            "port": API_PORT,
            "title": API_TITLE,
        },
        "logging": {
            "log_dir": str(LOG_DIR),
            "log_file": str(LOG_FILE),
            "log_level": LOG_LEVEL,
        },
        "features": {
            "spo_enabled": FEATURE_SPO_ENABLED,
            "insight_engine": FEATURE_INSIGHT_ENGINE,
        },
        "validation": {
            "min_matches_balanced": MIN_MATCHES_BALANCED,
            "min_matches_maxwin": MIN_MATCHES_MAXWIN,
            "min_matches_compound": MIN_MATCHES_COMPOUND,
            "max_matches": MAX_MATCHES,
            "min_stake": MIN_STAKE,
            "max_stake": MAX_STAKE,
        },
        "strategies": {
            "default": DEFAULT_STRATEGY,
            "supported": SUPPORTED_STRATEGIES,
        },
        "betting": {
            "browser_headless": BROWSER_HEADLESS,
            "browser_timeout": BROWSER_TIMEOUT,
            "viewport_width": VIEWPORT_WIDTH,
            "viewport_height": VIEWPORT_HEIGHT,
            "test_mode": BETTING_TEST_MODE,
            "test_site_url": TEST_SITE_URL,
            "screenshots_dir": str(SCREENSHOTS_DIR),
        },
    }


def validate_config() -> bool:
    """Validate configuration values."""
    errors = []
    
    if NUM_SIMULATIONS < 1000:
        errors.append("NUM_SIMULATIONS must be at least 1000")
    
    if NUM_SIMULATIONS > 100000:
        errors.append("NUM_SIMULATIONS should not exceed 100000 for performance")
    
    if API_PORT < 1 or API_PORT > 65535:
        errors.append("API_PORT must be between 1 and 65535")
    
    if MIN_STAKE < 0:
        errors.append("MIN_STAKE cannot be negative")
    
    if MAX_STAKE <= MIN_STAKE:
        errors.append("MAX_STAKE must be greater than MIN_STAKE")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True