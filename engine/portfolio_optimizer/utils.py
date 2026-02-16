"""
Utility functions for SPO engine
"""

import json
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

def format_slip_for_output(slip_dict: Dict) -> Dict:
    """
    Format a slip for final output (ensures proper types)
    """
    formatted = slip_dict.copy()
    
    # Ensure numeric fields are floats
    numeric_fields = ["total_odds", "confidence_score", "stake", 
                     "true_risk_score", "diversity_score"]
    
    for field in numeric_fields:
        if field in formatted:
            formatted[field] = float(formatted[field])
    
    return formatted

def calculate_coverage_score(slips: List[Dict]) -> float:
    """
    Calculate simple coverage score based on unique matches
    """
    if not slips:
        return 0.0
    
    all_matches = set()
    for slip in slips:
        for leg in slip.get("legs", []):
            all_matches.add(leg.get("match_id", ""))
    
    # Remove empty strings
    all_matches.discard("")
    
    # Simple coverage: more unique matches = better
    # Normalize to 0-1 scale (assume max 20 unique matches for 50 slips)
    coverage = len(all_matches) / min(20, len(slips) * 2)
    return min(1.0, coverage)

def log_optimization_result(result: Dict, log_file: str = "spo_logs.json"):
    """
    Log SPO optimization results to a file
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "master_slip_id": result.get("master_slip_id"),
        "original_slips": result.get("original_slip_count", 0),
        "optimized_slips": len(result.get("final_slips", [])),
        "coverage_score": result.get("metrics", {}).get("coverage_score", 0),
        "risk_distribution": result.get("metrics", {}).get("risk_distribution", {}),
        "engine_version": result.get("engine_version", "1.0.0")
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log SPO result: {e}")

def validate_odds(odds: float) -> float:
    """Validate and clamp odds to reasonable range"""
    if odds <= 0:
        return 1.01
    if odds > 1000:
        return 1000.0
    return round(odds, 2)

def get_market_complexity(market_name: str) -> float:
    """
    Get complexity score for a market (0-1)
    Higher = more complex/volatile
    """
    complexity_map = {
        "Match Result": 0.3,
        "Over/Under": 0.5,
        "Both Teams to Score": 0.6,
        "Correct Score": 0.9,
        "Half-Time/Full-Time": 0.8,
        "Asian Handicap": 0.7,
        "Double Chance": 0.4,
        "Draw No Bet": 0.4,
        "To Qualify": 0.6,
        "Anytime Goalscorer": 0.7
    }
    return complexity_map.get(market_name, 0.5)

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default fallback"""
    if denominator == 0:
        return default
    return numerator / denominator

def generate_slip_id(prefix: str = "SPO", index: int = 0) -> str:
    """Generate a unique slip ID"""
    return f"{prefix}_{index:03d}_{datetime.now().strftime('%H%M%S')}"