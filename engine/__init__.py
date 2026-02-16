"""   init.py file inside engine folder, same folder as slip_builder
Intelligent Football Slip Builder Engine v2.2
Public Engine Interface

Stable, import-safe API surface exposed to FastAPI.
Internal modules MUST NOT be imported directly by app.py.

Updates in v2.2:
- Triple-strategy support (Balanced + MaxWin + Compound)
- Compound accumulator builder (5-7 legs per slip)
- Strategy factory for dynamic builder selection
- Match count validation for compound strategy
- Expected Value (EV) optimization
- Monte Carlo coverage optimization (preserved)
- Portfolio diversification theory (preserved)
- Advanced statistical risk modeling (preserved)
- Hedging strategies (preserved)

Migration Guide:
- v2.1 → v2.2 is 100% backward compatible
- Existing code continues to work without changes
- New compound strategy is opt-in via payload
- Compound requires minimum 7 matches (validated automatically)
"""

from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger("engine")

# ---------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------

__version__ = "2.2.0"

# ---------------------------------------------------------------------
# Exceptions (PUBLIC CONTRACT)
# ---------------------------------------------------------------------

class SlipBuilderError(Exception):
    """
    Unified engine-level exception.
    All internal slip builder failures are normalized to this.
    """
    pass


# --- Backward compatibility aliases ---
# These are REQUIRED because app.py and older code import them directly.

class PayloadValidationError(SlipBuilderError):
    """Backward compatibility alias for payload validation errors"""
    pass


class MarketIntegrityError(SlipBuilderError):
    """Backward compatibility alias for market data errors"""
    pass


# ---------------------------------------------------------------------
# Strategy Factory Import (UPDATED for v2.2 - Triple Strategy)
# ---------------------------------------------------------------------

_STRATEGY_FACTORY_AVAILABLE = False
_SUPPORTED_STRATEGIES = []

try:
    from .slip_builder_factory import (
        create_slip_builder,
        get_available_strategies,
        get_strategy_info,
        validate_strategy,
        validate_match_count_for_strategy  # NEW in v2.2
    )
    _STRATEGY_FACTORY_AVAILABLE = True
    _SUPPORTED_STRATEGIES = get_available_strategies()
    
    logger.info("[ENGINE INIT] Strategy factory loaded successfully")
    logger.info(f"[ENGINE INIT] Supported strategies: {', '.join(_SUPPORTED_STRATEGIES)}")
    
except ImportError as e:
    logger.warning(f"[ENGINE INIT] Strategy factory not available: {e}")
    logger.warning("[ENGINE INIT] Falling back to legacy single-strategy mode")
    
    # Fallback: Use legacy slip builder directly
    def create_slip_builder(strategy: str = "balanced", **kwargs):
        """Fallback: use legacy slip builder"""
        from .slip_builder import SlipBuilder
        logger.warning(f"[ENGINE INIT] Using legacy SlipBuilder (strategy '{strategy}' ignored)")
        return SlipBuilder(**kwargs)
    
    def get_available_strategies() -> List[str]:
        return ["balanced"]
    
    def get_strategy_info() -> Dict[str, Dict[str, Any]]:
        return {
            "balanced": {
                "name": "Balanced Portfolio",
                "description": "Legacy single-strategy mode",
                "status": "active"
            }
        }
    
    def validate_strategy(strategy: str) -> bool:
        return strategy == "balanced"
    
    def validate_match_count_for_strategy(strategy: str, match_count: int) -> Dict[str, Any]:
        """Fallback match count validator"""
        return {
            "valid": match_count >= 3,
            "required": 3,
            "actual": match_count,
            "message": f"Balanced strategy requires at least 3 matches (got {match_count})"
        }


# ---------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------

_ENGINE_INITIALIZED = False
_ENGINE_CONFIG = {
    "enable_monte_carlo": True,
    "num_simulations": 10000
}


def initialize_engine(
    enable_monte_carlo: bool = True,
    num_simulations: int = 10000
) -> None:
    """
    Engine bootstrap hook.
    
    The intelligent slip builder can be configured at initialization.
    
    Args:
        enable_monte_carlo: Whether to run Monte Carlo simulations (default: True)
        num_simulations: Number of Monte Carlo simulations (default: 10000)
    """
    global _ENGINE_INITIALIZED, _ENGINE_CONFIG

    if _ENGINE_INITIALIZED:
        logger.debug("[ENGINE INIT] Engine already initialized; skipping")
        return

    try:
        # Store configuration for later use
        _ENGINE_CONFIG = {
            "enable_monte_carlo": enable_monte_carlo,
            "num_simulations": num_simulations
        }
        
        _ENGINE_INITIALIZED = True
        
        logger.info(
            f"[ENGINE INIT] Intelligent Slip Builder v{__version__} initialized | "
            f"Monte Carlo: {'ENABLED' if enable_monte_carlo else 'DISABLED'} | "
            f"Simulations: {num_simulations}"
        )
        
        # Log strategy mode
        if _STRATEGY_FACTORY_AVAILABLE:
            available = get_available_strategies()
            logger.info(f"[ENGINE INIT] Strategy Mode: TRIPLE-STRATEGY (v2.2)")
            logger.info(f"[ENGINE INIT] Available Strategies: {', '.join(available)}")
            
            # Log strategy details
            strategy_info = get_strategy_info()
            for strategy_name, info in strategy_info.items():
                min_matches = info.get('min_matches', 3)
                logger.info(
                    f"[ENGINE INIT]   - {strategy_name}: {info.get('name', 'Unknown')} "
                    f"(min {min_matches} matches)"
                )
        else:
            logger.info(f"[ENGINE INIT] Strategy Mode: LEGACY (balanced only)")
        
    except Exception as e:
        logger.error(f"[ENGINE INIT] Unexpected error during initialization: {e}")
        raise SlipBuilderError(f"Engine initialization failed: {e}") from e


# ---------------------------------------------------------------------
# Primary public API (UPDATED for triple-strategy in v2.2)
# ---------------------------------------------------------------------

def process_slip_builder_payload(
    payload: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Primary public API for intelligent slip generation.
    
    UPDATED in v2.2: Supports triple-strategy selection (balanced/maxwin/compound)
    NEW in v2.2: Automatic match count validation for compound strategy
    
    Process:
    1. Validates Laravel payload structure
    2. Extracts strategy from payload (defaults to "balanced")
    3. Validates match count for selected strategy
    4. Creates appropriate slip builder via factory
    5. Parses matches and markets
    6. Runs Monte Carlo simulations (if enabled)
    7. Generates 50 diverse slips
    8. Calculates coverage and diversity metrics
    9. Returns slips with comprehensive metadata
    
    Strategy Selection:
        Include "strategy" field in master_slip:
        {
            "master_slip": {
                "master_slip_id": 12345,
                "strategy": "compound",  // "balanced", "maxwin", or "compound"
                "matches": [...]
            }
        }
        
        If "strategy" is omitted, defaults to "balanced" (backward compatible)
    
    Match Count Requirements:
        - balanced: minimum 3 matches
        - maxwin: minimum 3 matches
        - compound: minimum 7 matches (automatically validated)
    
    Args:
        payload: Laravel payload containing master_slip with matches
        **kwargs: Additional configuration (ignored for now)
    
    Returns:
        Dictionary with:
        - generated_slips: List of 50 slip dictionaries
        - metadata: Generation metadata and statistics (includes strategy_used)
    
    Raises:
        SlipBuilderError: If generation fails
        PayloadValidationError: If payload is malformed or match count insufficient
        MarketIntegrityError: If market data is invalid
    """
    # Ensure engine is initialized
    if not _ENGINE_INITIALIZED:
        logger.warning("[SLIP BUILDER] Engine not initialized, initializing now...")
        initialize_engine()
    
    try:
        # Extract master_slip data
        master_slip_data = payload.get("master_slip", {})
        master_slip_id = master_slip_data.get("master_slip_id", "unknown")
        matches = master_slip_data.get("matches", [])
        match_count = len(matches)
        
        # Extract strategy from payload (NEW in v2.1, preserved in v2.2)
        requested_strategy = master_slip_data.get("strategy", "balanced")
        
        # Validate and normalize strategy
        if _STRATEGY_FACTORY_AVAILABLE:
            if not validate_strategy(requested_strategy):
                logger.warning(
                    f"[SLIP BUILDER] Invalid strategy '{requested_strategy}' requested, "
                    f"defaulting to 'balanced'"
                )
                requested_strategy = "balanced"
            
            # NEW in v2.2: Validate match count for strategy
            match_validation = validate_match_count_for_strategy(requested_strategy, match_count)
            
            if not match_validation["valid"]:
                error_msg = match_validation["message"]
                logger.error(f"[SLIP BUILDER] {error_msg}")
                
                # Provide helpful suggestion
                required = match_validation["required"]
                actual = match_validation["actual"]
                
                raise PayloadValidationError(
                    f"{error_msg}. "
                    f"Please add {required - actual} more match(es) or "
                    f"use a different strategy (balanced/maxwin require only 3 matches)."
                )
            
            logger.info(
                f"[SLIP BUILDER] Match count validation passed: "
                f"{match_count} matches (required: {match_validation['required']})"
            )
            
        else:
            # Factory not available - force balanced
            if requested_strategy != "balanced":
                logger.warning(
                    f"[SLIP BUILDER] Strategy '{requested_strategy}' requested "
                    f"but factory not available - using 'balanced'"
                )
            requested_strategy = "balanced"
        
        logger.info(
            f"[SLIP BUILDER] Starting generation | "
            f"Master Slip: {master_slip_id} | "
            f"Strategy: {requested_strategy} | "
            f"Matches: {match_count}"
        )
        
        # Create strategy-specific builder
        builder = create_slip_builder(
            strategy=requested_strategy,
            enable_monte_carlo=_ENGINE_CONFIG["enable_monte_carlo"],
            num_simulations=_ENGINE_CONFIG["num_simulations"]
        )
        
        logger.info(f"[SLIP BUILDER] Using builder: {builder.__class__.__name__}")
        
        # Generate slips using selected strategy
        result = builder.generate(payload)
        
        # Add strategy metadata
        metadata = result.get("metadata", {})
        metadata["strategy_used"] = requested_strategy
        metadata["match_count"] = match_count
        result["metadata"] = metadata
        
        # Log generation summary
        total_slips = metadata.get("total_slips", 0)
        input_matches = metadata.get("input_matches", 0)
        
        logger.info(
            f"[SLIP BUILDER] Generation complete | "
            f"Master Slip: {master_slip_id} | "
            f"Strategy: {requested_strategy} | "
            f"Slips: {total_slips} | "
            f"Matches: {input_matches}"
        )
        
        # Log portfolio metrics if available
        portfolio_metrics = metadata.get("portfolio_metrics", {})
        if portfolio_metrics:
            # Common metrics (all strategies)
            coverage = portfolio_metrics.get("coverage_percentage", 0)
            avg_confidence = portfolio_metrics.get("average_confidence", 0)
            avg_odds = portfolio_metrics.get("average_odds", 0)
            
            logger.info(
                f"[SLIP BUILDER] Portfolio Metrics | "
                f"Coverage: {coverage:.1f}% | "
                f"Avg Confidence: {avg_confidence:.3f} | "
                f"Avg Odds: {avg_odds:.2f}x"
            )
            
            # Strategy-specific metrics
            
            # Monte Carlo metrics (balanced/maxwin)
            if "average_wins_per_simulation" in portfolio_metrics:
                avg_wins = portfolio_metrics.get("average_wins_per_simulation", 0)
                logger.info(f"[SLIP BUILDER] Monte Carlo | Avg Wins/Sim: {avg_wins:.2f}")
            
            # EV metrics (maxwin/compound)
            if "portfolio_ev" in portfolio_metrics:
                portfolio_ev = portfolio_metrics.get("portfolio_ev", 0)
                avg_slip_ev = portfolio_metrics.get("average_slip_ev", 0)
                positive_ev_count = portfolio_metrics.get("positive_ev_count", 0)
                
                logger.info(
                    f"[SLIP BUILDER] EV Metrics | "
                    f"Portfolio EV: ${portfolio_ev:.2f} | "
                    f"Avg Slip EV: ${avg_slip_ev:.2f} | "
                    f"Positive EV Slips: {positive_ev_count}/{total_slips}"
                )
            
            # Accumulator metrics (compound only)
            if "average_legs" in portfolio_metrics:
                avg_legs = portfolio_metrics.get("average_legs", 0)
                min_odds = portfolio_metrics.get("min_odds", 0)
                max_odds = portfolio_metrics.get("max_odds", 0)
                
                logger.info(
                    f"[SLIP BUILDER] Accumulator Metrics | "
                    f"Avg Legs: {avg_legs:.1f} | "
                    f"Odds Range: {min_odds:.1f}x - {max_odds:.1f}x"
                )
        
        return result

    except SlipBuilderError:
        # Already normalized — propagate cleanly
        raise
        
    except Exception as e:
        # Catch *anything* unexpected and normalize it
        logger.exception(f"[SLIP BUILDER] Unexpected error during generation")
        
        # Determine error type
        error_msg = str(e).lower()
        
        if "payload" in error_msg or "matches" in error_msg or "valid" in error_msg:
            raise PayloadValidationError(
                f"Invalid payload structure: {str(e)}"
            ) from e
        elif "market" in error_msg or "odds" in error_msg or "selection" in error_msg:
            raise MarketIntegrityError(
                f"Market data integrity error: {str(e)}"
            ) from e
        else:
            raise SlipBuilderError(
                f"Slip generation failed: {str(e)}"
            ) from e


# ---------------------------------------------------------------------
# Portfolio Optimization (SPO) — untouched, preserved for compatibility
# ---------------------------------------------------------------------

def run_portfolio_optimization(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Portfolio optimization entrypoint.
    Preserved for compatibility with SPO pipeline.
    
    This takes the 50 generated slips and optimizes them down to 20
    with optimal stake distribution.
    
    Args:
        payload: Dictionary containing:
            - bankroll: Total available bankroll
            - generated_slips: List of 50 slips from slip builder
            - constraints: Optimization constraints
    
    Returns:
        Dictionary with optimized portfolio results
    
    Raises:
        SlipBuilderError: If optimization fails
    """
    try:
        logger.info("[SPO] Starting portfolio optimization")
        
        from .portfolio_optimizer import run_portfolio_optimization as _run
        
        result = _run(payload)
        
        # Log SPO summary
        final_slips = result.get("final_slips", [])
        logger.info(f"[SPO] Optimization complete | Final slips: {len(final_slips)}")
        
        return result

    except ImportError as e:
        logger.error(f"[SPO] Portfolio optimizer module not available: {e}")
        raise SlipBuilderError(
            "Portfolio optimization module not available"
        ) from e
        
    except Exception as e:
        logger.exception(f"[SPO] Portfolio optimization failed")
        raise SlipBuilderError(
            f"Portfolio optimization failed: {str(e)}"
        ) from e


# ---------------------------------------------------------------------
# Utility functions for engine status (UPDATED for triple-strategy)
# ---------------------------------------------------------------------

def get_engine_status() -> Dict[str, Any]:
    """
    Get current engine status and configuration.
    
    Returns:
        Dictionary with engine status information including:
        - Engine version and initialization status
        - Monte Carlo configuration
        - Available strategies (UPDATED in v2.2 for triple-strategy)
        - Feature list
    """
    status = {
        "version": __version__,
        "initialized": _ENGINE_INITIALIZED,
        "engine_type": "Intelligent Slip Builder v2.2",
        "strategy_mode": "triple-strategy" if _STRATEGY_FACTORY_AVAILABLE else "legacy",
        "features": [
            "Monte Carlo Coverage Optimization",
            "Portfolio Diversification Theory",
            "Statistical Risk Modeling",
            "Hedging Strategies",
            "Deterministic Generation"
        ]
    }
    
    # Add triple-strategy features if available
    if _STRATEGY_FACTORY_AVAILABLE:
        status["features"].extend([
            "Triple-Strategy Support (Balanced + MaxWin + Compound)",
            "Expected Value (EV) Optimization",
            "Compound Accumulator Builder (5-7 legs)",
            "Strategy Factory Routing",
            "Automatic Match Count Validation"
        ])
        status["available_strategies"] = get_available_strategies()
        
        # Add strategy details
        try:
            strategy_details = get_strategy_info()
            status["strategy_details"] = {
                name: {
                    "name": info.get("name", "Unknown"),
                    "legs_per_slip": info.get("legs_per_slip", "N/A"),
                    "win_rate": info.get("win_rate", "N/A"),
                    "risk_profile": info.get("risk_profile", "N/A"),
                    "min_matches": info.get("min_matches", 3)
                }
                for name, info in strategy_details.items()
            }
        except Exception as e:
            logger.warning(f"[ENGINE STATUS] Could not load strategy details: {e}")
    
    # Add Monte Carlo configuration
    if _ENGINE_INITIALIZED:
        status["monte_carlo_enabled"] = _ENGINE_CONFIG["enable_monte_carlo"]
        status["num_simulations"] = _ENGINE_CONFIG["num_simulations"]
    
    return status


# ---------------------------------------------------------------------
# Strategy information utilities (UPDATED for v2.2)
# ---------------------------------------------------------------------

def get_strategies() -> List[str]:
    """
    Get list of available strategies.
    
    Returns:
        List of strategy names
        
    Examples:
        >>> get_strategies()
        ['balanced', 'maxwin', 'compound']  # v2.2
    """
    return get_available_strategies()


def get_strategy_details() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about each strategy.
    
    Returns:
        Dictionary mapping strategy name to details
        
    Examples:
        >>> details = get_strategy_details()
        >>> print(details['compound']['legs_per_slip'])
        '5-7 (5=low, 6=medium, 7=high)'
        >>> print(details['compound']['min_matches'])
        7
    """
    return get_strategy_info()


def is_strategy_available(strategy: str) -> bool:
    """
    Check if a strategy is available.
    
    Args:
        strategy: Strategy name to check
        
    Returns:
        True if strategy is available, False otherwise
        
    Examples:
        >>> is_strategy_available('balanced')
        True
        >>> is_strategy_available('compound')
        True  # v2.2
        >>> is_strategy_available('invalid')
        False
    """
    return validate_strategy(strategy)


def check_match_count_requirement(strategy: str, match_count: int) -> Dict[str, Any]:
    """
    Check if match count meets requirements for strategy.
    
    NEW in v2.2: Helper function for frontend validation
    
    Args:
        strategy: Strategy name ('balanced', 'maxwin', or 'compound')
        match_count: Number of matches available
        
    Returns:
        Dictionary with validation result:
        {
            "valid": bool,
            "required": int,
            "actual": int,
            "message": str
        }
        
    Examples:
        >>> check_match_count_requirement('compound', 5)
        {
            'valid': False,
            'required': 7,
            'actual': 5,
            'message': 'Compound strategy requires at least 7 matches (got 5)'
        }
        
        >>> check_match_count_requirement('compound', 10)
        {
            'valid': True,
            'required': 7,
            'actual': 10,
            'message': '✅ 10 matches is sufficient for compound strategy'
        }
    """
    return validate_match_count_for_strategy(strategy, match_count)


# ---------------------------------------------------------------------
# Explicit export list (UPDATED for v2.2)
# ---------------------------------------------------------------------

__all__ = [
    # Core API
    "process_slip_builder_payload",
    "initialize_engine",
    "run_portfolio_optimization",
    "get_engine_status",
    
    # Exceptions
    "SlipBuilderError",
    "PayloadValidationError",
    "MarketIntegrityError",
    
    # Strategy utilities (UPDATED in v2.2)
    "get_strategies",
    "get_strategy_details",
    "is_strategy_available",
    "check_match_count_requirement",  # NEW in v2.2
    
    # Strategy factory functions (UPDATED in v2.2)
    "create_slip_builder",
    "get_available_strategies",
    "get_strategy_info",
    "validate_strategy",
    "validate_match_count_for_strategy",  # NEW in v2.2
    
    # Version
    "__version__",
]