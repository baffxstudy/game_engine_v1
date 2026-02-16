"""
Slip Portfolio Optimizer (SPO) - Phase 2 Engine
Portfolio Optimization Module for transforming 50 slips into 20-slip portfolio
"""

from .spo_engine import (
    # Core Classes
    SlipPortfolioOptimizer,
    Leg,
    Slip,
    CoverageGraph,
    
    # Enums
    RiskCategory,
    CoverageRole,
    
    # Main Functions
    run_portfolio_optimization,
    

)

__version__ = "1.0.0"
__author__ = "Freedom Train AI - Quantitative Systems Team"
__description__ = "Portfolio optimization engine transforming 50 betting slips into optimized 20-slip portfolio"

# Export the main components for easy importing
__all__ = [
    "SlipPortfolioOptimizer",
    "run_portfolio_optimization",
    "Leg",
    "Slip", 
    "CoverageGraph",
    "RiskCategory",
    "CoverageRole",

]

# Module-level configuration
DEFAULT_CONFIG = {
    "redundancy_threshold": 0.7,
    "min_coverage_score": 0.9,
    "target_distribution": {
        "low": 0.35,
        "medium": 0.35,
        "high": 0.30
    },
    "max_portfolio_size": 20,
    "bankroll_default": 1000
}

def get_version():
    """Get the current version of the SPO engine"""
    return __version__

def get_default_config():
    """Get the default configuration for the SPO engine"""
    return DEFAULT_CONFIG.copy()

def create_optimizer(config=None):
    """
    Factory function to create a new SlipPortfolioOptimizer instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        SlipPortfolioOptimizer instance
    """
    return SlipPortfolioOptimizer(config)

# Optional: Performance monitoring decorator
def profile_optimization(func):
    """Decorator to profile optimization performance"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        if isinstance(result, dict) and "metrics" in result:
            result["metrics"]["optimization_time_ms"] = round((end_time - start_time) * 1000, 2)
        
        print(f"🔄 SPO Optimization completed in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper

# Optional: Validation function for input payload
def validate_spo_input(input_data: dict) -> tuple:
    """
    Validate input data for SPO optimization
    
    Args:
        input_data: Input payload dictionary
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = ["generated_slips"]
    
    for field in required_fields:
        if field not in input_data:
            return False, f"Missing required field: {field}"
    
    if not isinstance(input_data["generated_slips"], list):
        return False, "generated_slips must be a list"
    
    if len(input_data["generated_slips"]) < 20:
        return False, f"Need at least 20 slips for optimization, got {len(input_data['generated_slips'])}"
    
    # Validate slip structure
    for i, slip in enumerate(input_data["generated_slips"]):
        if "slip_id" not in slip:
            return False, f"Slip at index {i} missing slip_id"
        if "legs" not in slip:
            return False, f"Slip {slip.get('slip_id', f'index_{i}')} missing legs"
        if "total_odds" not in slip:
            return False, f"Slip {slip.get('slip_id')} missing total_odds"
    
    return True, "Input validation passed"

# Optional: Utility to convert legacy format to SPO format
def convert_to_spo_format(phase1_output: dict) -> dict:
    """
    Convert Phase 1 output to SPO input format
    
    Args:
        phase1_output: Output from Phase 1 slip builder
        
    Returns:
        dict: Formatted input for SPO
    """
    return {
        "bankroll": 1000,  # Default bankroll
        "generated_slips": phase1_output.get("generated_slips", []),
        "constraints": {
            "final_slips": 20,
            "allow_mixed_risk": True
        },
        "metadata": {
            "phase1_master_slip_id": phase1_output.get("master_slip_id"),
            "phase1_timestamp": phase1_output.get("timestamp"),
            "original_slip_count": len(phase1_output.get("generated_slips", []))
        }
    }

# Optional: Export metrics calculation for external use
def calculate_portfolio_metrics(portfolio: list, bankroll: float = 1000) -> dict:
    """
    Calculate metrics for any portfolio (not just SPO output)
    
    Args:
        portfolio: List of slip dictionaries
        bankroll: Total bankroll for stake calculation
        
    Returns:
        dict: Portfolio metrics
    """
    if not portfolio:
        return {}
    
    total_stake = sum(slip.get("stake", 0) for slip in portfolio)
    avg_odds = sum(slip.get("total_odds", 1.0) for slip in portfolio) / len(portfolio)
    avg_confidence = sum(slip.get("confidence_score", 0.5) for slip in portfolio) / len(portfolio)
    
    # Risk distribution (if available)
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    for slip in portfolio:
        risk = slip.get("risk_category", "medium")
        if risk in risk_counts:
            risk_counts[risk] += 1
        else:
            risk_counts["medium"] += 1
    
    return {
        "total_slips": len(portfolio),
        "total_stake": round(total_stake, 2),
        "bankroll_percentage": round(total_stake / bankroll * 100, 1) if bankroll > 0 else 0,
        "avg_odds": round(avg_odds, 2),
        "avg_confidence": round(avg_confidence, 3),
        "risk_distribution": risk_counts,
        "stake_range": {
            "min": min([s.get("stake", 0) for s in portfolio]),
            "max": max([s.get("stake", 0) for s in portfolio]),
            "avg": total_stake / len(portfolio)
        }
    }

# Initialize module
print(f"✅ Slip Portfolio Optimizer (SPO) v{__version__} initialized")