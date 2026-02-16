# game_engine/utils/math_utils.py

import math
from typing import List, Dict

class MathUtils:
    """
    Foundational mathematical utilities for betting calculations,
    odds conversion, and statistical normalization.
    """

    @staticmethod
    def decimal_to_implied(odds: float) -> float:
        """
        Converts decimal odds to implied probability.
        Why: Bookmaker odds represent the 'market' view of probability.
        """
        if odds <= 1.0:
            # Odds of 1.0 or less are mathematically impossible in betting
            return 0.0
        return round(1.0 / odds, 4)

    @staticmethod
    def implied_to_decimal(probability: float) -> float:
        """
        Converts a probability (0.0 to 1.0) back into decimal odds.
        Why: Used to show Laravel what 'Fair Odds' should look like 
        compared to the bookmaker's price.
        """
        if probability <= 0:
            return 999.0  # Represents a near-impossible outcome
        return round(1.0 / probability, 2)

    @staticmethod
    def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
        """
        Ensures a set of probabilities (e.g., Home/Draw/Away) sums exactly to 1.0.
        Why: Bookmaker odds include 'vig' or 'overround' (summing to > 1.0). 
        Our simulations must use 'clean' 100% distributions.
        """
        total = sum(probs.values())
        if total == 0:
            return {k: 0.0 for k in probs}
        
        # Divide each probability by the total to remove the overround
        return {k: round(v / total, 4) for k, v in probs.items()}

    @staticmethod
    def calculate_standard_deviation(data: List[float]) -> float:
        """
        Calculates the standard deviation for a list of values.
        Why: High standard deviation in Monte Carlo results indicates 
        high volatility/risk for that specific match.
        """
        if not data:
            return 0.0
        
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        return round(math.sqrt(variance), 4)

    @staticmethod
    def calculate_kelly_criterion(bankroll: float, odds: float, prob: float) -> float:
        """
        Calculates the suggested stake based on the Kelly Criterion.
        Formula: ((Odds - 1) * Prob - (1 - Prob)) / (Odds - 1)
        Why: This is a professional-grade way to determine optimal 
        risk for a 'Value' bet.
        """
        if odds <= 1.0:
            return 0.0
        
        b = odds - 1
        q = 1 - prob
        kelly_fraction = (b * prob - q) / b
        
        # We return a 'Fractional Kelly' (0.25) to be conservative
        # This prevents the engine from suggesting dangerously high stakes
        conservative_kelly = kelly_fraction * 0.25
        return max(0.0, round(conservative_kelly * bankroll, 2))