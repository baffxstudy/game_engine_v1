"""
SLIP BUILDER FACTORY - Strategy Routing & Management
=====================================================

Manages all slip generation strategies with a unified interface.
Supports easy addition of new strategies without modifying core logic.

Available Strategies:
1. "balanced"          - Coverage-optimized portfolio (original)
2. "maxwin"            - EV-optimized profit hunting
3. "compound"          - High-leg accumulators (5-7 legs)
4. "composition_slips" - NEW: Intelligent multi-leg composition with market clustering

Version: 2.2.0-quadrupleStrategy
"""

import logging
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ============================================================================
# STRATEGY REGISTRY - CENTRALIZED & EXTENSIBLE
# ============================================================================

STRATEGIES = {
    "balanced": {
        "module": "engine.slip_builder",
        "class": "SlipBuilder",
        "name": "Balanced Portfolio",
        "description": "Consistent winners across simulations (4-6 per scenario)",
        "goal": "Frequent wins with moderate returns",
        "risk_profile": "Medium",
        "variance": "Low-Medium",
        "ideal_for": "Users who value consistency and regular wins",
        "legs_per_slip": "2-3 (up to 4 for high risk)",
        "win_rate": "40-50%",
        "avg_odds": "6-15x",
        "min_matches": 3,
        "features": [
            "Monte Carlo simulations",
            "Risk-based allocation",
            "Hedging enforcement",
            "Diversity optimization"
        ],
        "strategy_type": "coverage"
    },
    "maxwin": {
        "module": "engine.slip_builder_maxwin",
        "class": "MaxWinSlipBuilder",
        "name": "Maximum Win Potential",
        "description": "Maximize expected value (fewer but bigger wins)",
        "goal": "Maximum long-term profit",
        "risk_profile": "Aggressive",
        "variance": "High",
        "ideal_for": "Users who prioritize profit over win frequency",
        "legs_per_slip": "2-4 (flexible based on EV)",
        "win_rate": "25-35%",
        "avg_odds": "15-50x",
        "min_matches": 3,
        "features": [
            "Expected Value (EV) scoring",
            "Profit optimization",
            "High variance selection",
            "Monte Carlo simulations"
        ],
        "strategy_type": "ev_optimization"
    },
    "compound": {
        "module": "engine.slip_builder_compound",
        "class": "CompoundSlipBuilder",
        "name": "Compound Accumulator",
        "description": "High-leg accumulators with EV optimization (5-7 legs)",
        "goal": "Massive upside through compound stacking",
        "risk_profile": "Very Aggressive",
        "variance": "Very High",
        "ideal_for": "Users seeking lottery-style payouts with mathematical backing",
        "legs_per_slip": "5-7 (5=low, 6=medium, 7=high)",
        "win_rate": "5-15%",
        "avg_odds": "30-150x",
        "min_matches": 7,
        "features": [
            "High-leg stacking",
            "EV filtering",
            "Compound interest calculation",
            "Aggressive selection"
        ],
        "strategy_type": "accumulator"
    },
    # ✅ NEW: Composition Slips Strategy
    "composition_slips": {
        "module": "engine.slip_builder_composition",
        "class": "CompositionSlipsBuilder",
        "name": "Intelligent Composition",
        "description": "Market-clustered multi-leg compositions with intelligent selection",
        "goal": "Balanced profitability through intelligent market clustering",
        "risk_profile": "Medium-Aggressive",
        "variance": "Medium",
        "ideal_for": "Users who want strategic market combinations with AI-driven selection",
        "legs_per_slip": "3-5 (based on market clusters)",
        "win_rate": "30-40%",
        "avg_odds": "8-30x",
        "min_matches": 4,
        "features": [
            "Market similarity clustering",
            "Correlated outcome selection",
            "Composition-based diversification",
            "Intelligent multi-leg pairing",
            "Monte Carlo simulations"
        ],
        "strategy_type": "composition"
    }
}

DEFAULT_STRATEGY = "balanced"

# ============================================================================
# ABSTRACT BASE CLASS FOR ALL BUILDERS
# ============================================================================

class BaseSlipBuilder(ABC):
    """
    Abstract base class that all slip builders must inherit from.
    
    Ensures consistent interface across all strategies.
    Every builder must implement the generate() method.
    """
    
    def __init__(
        self,
        enable_monte_carlo: bool = True,
        num_simulations: int = 10000,
        **kwargs
    ):
        """
        Initialize builder with common parameters.
        
        Args:
            enable_monte_carlo: Whether to run Monte Carlo simulations
            num_simulations: Number of simulations to run
            **kwargs: Strategy-specific parameters
        """
        self.enable_monte_carlo = enable_monte_carlo
        self.num_simulations = int(num_simulations)
        self.strategy_kwargs = kwargs
    
    @abstractmethod
    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate slips from payload.
        
        ALL BUILDERS MUST IMPLEMENT THIS.
        
        Args:
            payload: Master slip payload from Laravel
        
        Returns:
            {
                "generated_slips": [
                    {
                        "slip_id": str,
                        "risk_level": str,
                        "legs": List[Dict],
                        "total_odds": float,
                        "confidence_score": float,
                        "coverage_score": float,
                        "diversity_score": float,
                        ...strategy_specific_fields
                    },
                    ...
                ],
                "metadata": {
                    "master_slip_id": int,
                    "total_slips": int,
                    "strategy_used": str,
                    "portfolio_metrics": Dict,
                    ...
                }
            }
        """
        pass


# ============================================================================
# FACTORY FUNCTION - UNIFIED ROUTING
# ============================================================================

def create_slip_builder(
    strategy: Optional[str] = None,
    enable_monte_carlo: bool = True,
    num_simulations: int = 10000,
    **kwargs
) -> BaseSlipBuilder:
    """
    Create slip builder instance for specified strategy.
    
    This is the main entry point for all slip generation. It routes to the
    appropriate builder based on the strategy parameter, with intelligent
    fallback and validation.
    
    Args:
        strategy: Strategy name from ["balanced", "maxwin", "compound", "composition_slips"]
                 If None or invalid, defaults to "balanced"
        enable_monte_carlo: Whether to run Monte Carlo simulations (default: True)
        num_simulations: Number of Monte Carlo simulations (default: 10000)
        **kwargs: Strategy-specific parameters passed to builder constructor
    
    Returns:
        Builder instance (all have .generate(payload) method)
    
    Examples:
        # Balanced strategy (original)
        builder = create_slip_builder("balanced")
        result = builder.generate(payload)
        
        # MaxWin strategy
        builder = create_slip_builder("maxwin", num_simulations=5000)
        result = builder.generate(payload)
        
        # Compound strategy
        builder = create_slip_builder("compound", num_simulations=10000)
        result = builder.generate(payload)
        
        # NEW: Composition strategy with custom parameters
        builder = create_slip_builder(
            "composition_slips",
            cluster_algorithm="kmeans",
            similarity_threshold=0.7
        )
        result = builder.generate(payload)
        
        # Default (backward compatible)
        builder = create_slip_builder()  # defaults to "balanced"
        result = builder.generate(payload)
    
    Raises:
        ImportError: If strategy module cannot be imported
        AttributeError: If strategy class doesn't exist
        ValueError: If strategy initialization fails
    """
    # Normalize strategy input
    if strategy is None:
        strategy = DEFAULT_STRATEGY
        logger.debug(f"[FACTORY] No strategy specified, using default: '{DEFAULT_STRATEGY}'")
    
    strategy = str(strategy).lower().strip()
    
    # Validate strategy
    if not validate_strategy(strategy):
        logger.warning(
            f"[FACTORY] Invalid strategy '{strategy}', "
            f"defaulting to '{DEFAULT_STRATEGY}'. "
            f"Available strategies: {', '.join(get_available_strategies())}"
        )
        strategy = DEFAULT_STRATEGY
    
    # Get strategy metadata
    strategy_info = STRATEGIES[strategy]
    
    logger.info(
        f"[FACTORY] Creating builder: {strategy_info['name']} "
        f"({strategy_info['goal']})"
    )
    
    # Import and instantiate the appropriate builder
    try:
        builder = _instantiate_builder(
            strategy,
            strategy_info,
            enable_monte_carlo=enable_monte_carlo,
            num_simulations=num_simulations,
            **kwargs
        )
        
        logger.info(f"[FACTORY] ✓ Builder created successfully: {strategy}")
        logger.debug(
            f"[FACTORY] Builder config: MC={enable_monte_carlo}, "
            f"Sims={num_simulations}, ExtraKwargs={list(kwargs.keys())}"
        )
        return builder
    
    except ImportError as e:
        logger.error(
            f"[FACTORY] ✗ Failed to import strategy module: {strategy_info['module']}. "
            f"Error: {str(e)}"
        )
        raise
    
    except AttributeError as e:
        logger.error(
            f"[FACTORY] ✗ Strategy class not found: {strategy_info['class']}. "
            f"Error: {str(e)}"
        )
        raise
    
    except Exception as e:
        logger.error(
            f"[FACTORY] ✗ Builder instantiation failed for strategy '{strategy}': {str(e)}"
        )
        raise


def _instantiate_builder(
    strategy: str,
    strategy_info: Dict[str, Any],
    enable_monte_carlo: bool = True,
    num_simulations: int = 10000,
    **kwargs
) -> BaseSlipBuilder:
    """
    Internal helper to instantiate the correct builder class.
    
    Keeps the factory function clean and maintainable.
    """
    # Route to appropriate builder class
    if strategy == "balanced":
        from .slip_builder import SlipBuilder
        return SlipBuilder(
            enable_monte_carlo=enable_monte_carlo,
            num_simulations=num_simulations,
            **kwargs
        )
    
    elif strategy == "maxwin":
        from .slip_builder_maxwin import MaxWinSlipBuilder
        return MaxWinSlipBuilder(
            enable_monte_carlo=enable_monte_carlo,
            num_simulations=num_simulations,
            **kwargs
        )
    
    elif strategy == "compound":
        from .slip_builder_compound import CompoundSlipBuilder
        return CompoundSlipBuilder(
            enable_monte_carlo=enable_monte_carlo,
            num_simulations=num_simulations,
            **kwargs
        )
    
    # ✅ NEW: Composition slips strategy
    elif strategy == "composition_slips":
        from .slip_builder_composition import CompositionSlipsBuilder
        return CompositionSlipsBuilder(
            enable_monte_carlo=enable_monte_carlo,
            num_simulations=num_simulations,
            **kwargs
        )
    
    else:
        # Should never reach here due to validate_strategy check
        raise ValueError(f"Strategy '{strategy}' not implemented")


# ============================================================================
# VALIDATION & DISCOVERY
# ============================================================================

def validate_strategy(strategy: str) -> bool:
    """
    Check if a strategy name is valid.
    
    Args:
        strategy: Strategy name to validate
    
    Returns:
        True if strategy exists in registry, False otherwise
    
    Example:
        >>> validate_strategy("balanced")
        True
        >>> validate_strategy("composition_slips")
        True
        >>> validate_strategy("invalid")
        False
    """
    if not isinstance(strategy, str):
        return False
    
    return strategy.lower().strip() in STRATEGIES


def get_available_strategies() -> List[str]:
    """
    Get list of available strategy names.
    
    Returns:
        List of strategy identifiers in order of addition
    
    Example:
        >>> strategies = get_available_strategies()
        >>> print(strategies)
        ['balanced', 'maxwin', 'compound', 'composition_slips']
    """
    return list(STRATEGIES.keys())


def get_strategy_info(strategy: Optional[str] = None) -> Dict[str, Any]:
    """
    Get metadata about a specific strategy or all strategies.
    
    Args:
        strategy: Strategy name. If None, returns info for all strategies.
    
    Returns:
        Dictionary with strategy metadata (excludes internal fields: module, class)
        
        If strategy is None:
            {
                "balanced": {...},
                "maxwin": {...},
                "compound": {...},
                "composition_slips": {...}
            }
        
        If strategy is specified:
            {"name": "...", "description": "...", "features": [...], ...}
    
    Examples:
        # Get all strategies
        >>> all_info = get_strategy_info()
        >>> print(all_info["composition_slips"]["name"])
        'Intelligent Composition'
        
        # Get specific strategy
        >>> comp_info = get_strategy_info("composition_slips")
        >>> print(comp_info["features"])
        ['Market similarity clustering', 'Correlated outcome selection', ...]
    
    Raises:
        ValueError: If specified strategy doesn't exist
    """
    if strategy is None:
        # Return all strategies with internal fields removed
        return {
            name: {
                k: v for k, v in info.items() 
                if k not in ["module", "class"]
            }
            for name, info in STRATEGIES.items()
        }
    
    # Return specific strategy
    strategy = str(strategy).lower().strip()
    
    if not validate_strategy(strategy):
        raise ValueError(
            f"Invalid strategy: '{strategy}'. "
            f"Available: {', '.join(get_available_strategies())}"
        )
    
    info = STRATEGIES[strategy].copy()
    # Remove internal fields
    info.pop("module", None)
    info.pop("class", None)
    
    return info


def get_default_strategy() -> str:
    """
    Get the default strategy name.
    
    Returns:
        Default strategy identifier
    
    Example:
        >>> default = get_default_strategy()
        >>> print(default)
        'balanced'
    """
    return DEFAULT_STRATEGY


# ============================================================================
# STRATEGY COMPARISON & DISCOVERY UTILITIES
# ============================================================================

def compare_strategies() -> Dict[str, Any]:
    """
    Get a comprehensive comparison matrix of all available strategies.
    
    Useful for documentation, UI selection, decision support, and analysis.
    
    Returns:
        Dictionary with comparison data and matrices
    
    Example:
        >>> comparison = compare_strategies()
        >>> for strategy, info in comparison['strategies'].items():
        ...     print(f"{strategy}: {info['risk_profile']}")
        balanced: Medium
        maxwin: Aggressive
        compound: Very Aggressive
        composition_slips: Medium-Aggressive
    """
    strategies_info = get_strategy_info()
    
    return {
        "total_strategies": len(STRATEGIES),
        "default_strategy": DEFAULT_STRATEGY,
        "strategies": strategies_info,
        "comparison_matrix": {
            "risk_profile": {
                name: info["risk_profile"] 
                for name, info in STRATEGIES.items()
            },
            "variance": {
                name: info["variance"] 
                for name, info in STRATEGIES.items()
            },
            "goal": {
                name: info["goal"] 
                for name, info in STRATEGIES.items()
            },
            "legs_per_slip": {
                name: info["legs_per_slip"] 
                for name, info in STRATEGIES.items()
            },
            "win_rate": {
                name: info["win_rate"] 
                for name, info in STRATEGIES.items()
            },
            "avg_odds": {
                name: info["avg_odds"] 
                for name, info in STRATEGIES.items()
            },
            "min_matches": {
                name: info["min_matches"] 
                for name, info in STRATEGIES.items()
            },
            "strategy_type": {
                name: info["strategy_type"] 
                for name, info in STRATEGIES.items()
            }
        }
    }


def suggest_strategy(
    user_profile: str = "balanced",
    risk_tolerance: str = "medium",
    match_count: int = 5,
    optimization_goal: str = "frequency"
) -> str:
    """
    Suggest optimal strategy based on user profile and constraints.
    
    Now considers match count, risk tolerance, and optimization goals.
    
    Args:
        user_profile: User type ("conservative", "balanced", "aggressive", "extreme")
        risk_tolerance: Risk tolerance ("low", "medium", "high", "very_high")
        match_count: Number of available matches
        optimization_goal: "frequency" (more wins) or "profit" (bigger wins)
    
    Returns:
        Recommended strategy name
    
    Example:
        >>> suggest_strategy("extreme", "very_high", match_count=10)
        'compound'
        >>> suggest_strategy("aggressive", "high", match_count=6, optimization_goal="profit")
        'maxwin'
        >>> suggest_strategy("balanced", "medium", match_count=5)
        'composition_slips'  # NEW: Recommended for medium profiles
        >>> suggest_strategy("conservative", "low", match_count=8)
        'balanced'
    """
    user_profile = user_profile.lower()
    risk_tolerance = risk_tolerance.lower()
    optimization_goal = optimization_goal.lower()
    
    # Decision tree with new composition_slips option
    
    # Extreme profile OR very high risk tolerance → Compound (if enough matches)
    if (user_profile == "extreme" or risk_tolerance == "very_high") and match_count >= 7:
        return "compound"
    
    # Aggressive profile + profit goal → MaxWin (if not enough matches for compound)
    if user_profile == "aggressive" and optimization_goal == "profit":
        if match_count >= 7:
            return "compound"
        return "maxwin"
    
    # Aggressive profile + frequency goal → Composition (balanced-aggressive)
    if user_profile == "aggressive" and optimization_goal == "frequency":
        if match_count >= 4:
            return "composition_slips"
        return "balanced"
    
    # Balanced profile → NEW: Default to composition_slips for balanced users
    if user_profile == "balanced" and match_count >= 4:
        return "composition_slips"
    
    # Conservative profile OR low risk tolerance → Balanced
    if user_profile == "conservative" or risk_tolerance == "low":
        return "balanced"
    
    # Default fallback
    return "balanced"


def get_strategy_by_type(strategy_type: str) -> List[str]:
    """
    Get all strategies of a specific type.
    
    Strategy types: "coverage", "ev_optimization", "accumulator", "composition"
    
    Args:
        strategy_type: Type filter
    
    Returns:
        List of strategy names matching the type
    
    Example:
        >>> aggressive_strategies = get_strategy_by_type("accumulator")
        >>> print(aggressive_strategies)
        ['compound']
        
        >>> composition_strategies = get_strategy_by_type("composition")
        >>> print(composition_strategies)
        ['composition_slips']
    """
    return [
        name for name, info in STRATEGIES.items()
        if info.get("strategy_type") == strategy_type
    ]


# ============================================================================
# MATCH COUNT VALIDATION
# ============================================================================

def validate_match_count_for_strategy(
    strategy: str,
    match_count: int
) -> Dict[str, Any]:
    """
    Validate if match count is sufficient for strategy.
    
    Critical for strategies with high minimum match requirements
    (e.g., compound requires ≥7, composition_slips requires ≥4).
    
    Args:
        strategy: Strategy name
        match_count: Number of available matches
    
    Returns:
        {
            "valid": bool,
            "required": int,
            "actual": int,
            "message": str,
            "suggestion": str (if invalid)
        }
    
    Example:
        >>> validate_match_count_for_strategy("composition_slips", 3)
        {
            "valid": False,
            "required": 4,
            "actual": 3,
            "message": "❌ Composition_slips strategy requires at least 4 matches (got 3)",
            "suggestion": "Use 'balanced' strategy instead"
        }
        
        >>> validate_match_count_for_strategy("composition_slips", 5)
        {
            "valid": True,
            "required": 4,
            "actual": 5,
            "message": "✅ 5 matches is sufficient for composition_slips strategy"
        }
    """
    if not validate_strategy(strategy):
        return {
            "valid": False,
            "required": 0,
            "actual": match_count,
            "message": f"Invalid strategy: {strategy}",
            "suggestion": f"Use one of: {', '.join(get_available_strategies())}"
        }
    
    strategy_info = STRATEGIES[strategy]
    required = strategy_info.get("min_matches", 3)
    
    valid = match_count >= required
    
    result = {
        "valid": valid,
        "required": required,
        "actual": match_count,
    }
    
    if valid:
        result["message"] = (
            f"✅ {match_count} matches is sufficient for {strategy} strategy"
        )
    else:
        result["message"] = (
            f"❌ {strategy.capitalize()} strategy requires at least {required} matches "
            f"(got {match_count})"
        )
        
        # Suggest alternative strategy
        alternative = _suggest_alternative_strategy(match_count)
        result["suggestion"] = f"Use '{alternative}' strategy instead"
    
    return result


def _suggest_alternative_strategy(match_count: int) -> str:
    """
    Suggest a suitable strategy for a given match count.
    
    Internal helper for validate_match_count_for_strategy.
    """
    if match_count < 3:
        return "balanced"  # Minimum viable
    elif match_count < 4:
        return "balanced"
    elif match_count < 7:
        return "composition_slips"  # Good fit for 4-6 matches
    else:
        return "compound"  # Enough matches for high-leg strategy


# ============================================================================
# LOGGING & DIAGNOSTICS
# ============================================================================

def log_strategy_registry() -> None:
    """
    Log all available strategies for debugging and startup validation.
    
    Useful during application startup to verify strategy availability.
    Called automatically when module is imported.
    """
    logger.info("=" * 100)
    logger.info("[STRATEGY REGISTRY] Available slip building strategies:")
    logger.info("=" * 100)
    
    for i, (name, info) in enumerate(STRATEGIES.items(), 1):
        logger.info(
            f"  {i}. {name:18s} | {info['name']:30s} | "
            f"Risk: {info['risk_profile']:15s} | "
            f"Legs: {info['legs_per_slip']:20s} | "
            f"Min Matches: {info['min_matches']}"
        )
        logger.info(
            f"     Goal: {info['goal']}"
        )
        logger.info(
            f"     Features: {', '.join(info['features'][:2])}... "
            f"(+{len(info['features']) - 2} more)" if len(info['features']) > 2 
            else f"     Features: {', '.join(info['features'])}"
        )
    
    logger.info("=" * 100)
    logger.info(f"[STRATEGY REGISTRY] Default strategy: {DEFAULT_STRATEGY}")
    logger.info(f"[STRATEGY REGISTRY] Total strategies: {len(STRATEGIES)}")
    logger.info("=" * 100)


def get_strategy_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive diagnostics about all strategies.
    
    Useful for debugging, testing, and API documentation.
    
    Returns:
        Detailed information about all strategies and their capabilities
    """
    return {
        "total_strategies": len(STRATEGIES),
        "default": DEFAULT_STRATEGY,
        "registry": get_strategy_info(),
        "comparison": compare_strategies(),
        "types": {
            strategy_type: get_strategy_by_type(strategy_type)
            for strategy_type in set(
                info.get("strategy_type") for info in STRATEGIES.values()
            )
        }
    }


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Log available strategies when module is imported
log_strategy_registry()