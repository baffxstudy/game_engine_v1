"""
Custom exception hierarchy for the Football Match Analysis Engine.

Provides structured error handling with clear error types and messages.
"""

from typing import Optional, Dict, Any


class EngineError(Exception):
    """Base exception for all engine-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class SlipBuilderError(EngineError):
    """Base exception for slip builder errors."""
    pass


class PayloadValidationError(SlipBuilderError):
    """Raised when payload validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="PAYLOAD_VALIDATION_ERROR", **kwargs)
        if field:
            self.details["field"] = field


class MarketIntegrityError(SlipBuilderError):
    """Raised when market data integrity issues are detected."""
    
    def __init__(self, message: str, market_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MARKET_INTEGRITY_ERROR", **kwargs)
        if market_id:
            self.details["market_id"] = market_id


class StrategyError(SlipBuilderError):
    """Raised when strategy-related errors occur."""
    
    def __init__(self, message: str, strategy: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="STRATEGY_ERROR", **kwargs)
        if strategy:
            self.details["strategy"] = strategy


class MatchCountError(PayloadValidationError):
    """Raised when match count is insufficient for selected strategy."""
    
    def __init__(
        self,
        strategy: str,
        required: int,
        actual: int,
        **kwargs
    ):
        message = (
            f"Strategy '{strategy}' requires at least {required} matches "
            f"(got {actual})"
        )
        super().__init__(message, **kwargs)
        self.details.update({
            "strategy": strategy,
            "required": required,
            "actual": actual
        })


class ConfigurationError(EngineError):
    """Raised when configuration errors occur."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        if config_key:
            self.details["config_key"] = config_key


class CallbackError(EngineError):
    """Raised when callback operations fail."""
    
    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CALLBACK_ERROR", **kwargs)
        if url:
            self.details["url"] = url


class SPOError(EngineError):
    """Raised when portfolio optimization fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SPO_ERROR", **kwargs)
