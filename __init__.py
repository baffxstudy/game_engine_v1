# game_engine/__init__.py

"""
Football Game Engine - Main Package Exports
"""

# Export the main app
from .app import app

# Export commonly used schemas
from .schemas import (
    MasterSlipRequest,
    EngineResponse,
    AnalysisRequest,
    AnalysisResponse,
    GeneratedSlip,
    MatchData,
    MasterSlipData
)

# Export engine components
try:
    from .engine import (
        process_slip_builder_payload,
        SlipBuilderError,
        PayloadValidationError,
        MarketIntegrityError,
        __version__ as ENGINE_VERSION
    )
except ImportError:
    # Fallback for when engine is not available
    process_slip_builder_payload = None
    SlipBuilderError = Exception
    PayloadValidationError = Exception
    MarketIntegrityError = Exception
    ENGINE_VERSION = "unknown"

# Package metadata
__version__ = "1.0.0"
__author__ = "Football Engine Team"
__description__ = "Deterministic slip generation engine with flexible payload handling"

# Public API exports
__all__ = [
    # Main app
    'app',
    
    # Schemas
    'MasterSlipRequest',
    'EngineResponse',
    'AnalysisRequest',
    'AnalysisResponse',
    'GeneratedSlip',
    'MatchData',
    'MasterSlipData',
    
    # Engine functions
    'process_slip_builder_payload',
    
    # Exceptions
    'SlipBuilderError',
    'PayloadValidationError',
    'MarketIntegrityError',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__',
    'ENGINE_VERSION'
]