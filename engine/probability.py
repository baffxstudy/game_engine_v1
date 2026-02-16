# game_engine/engine/probability.py

import math
from typing import Any

class ProbabilityEngine:
    @staticmethod
    def get_blended_probabilities(match: Any) -> float:
        """
        Uses xG (Expected Goals) and weights provided by Laravel 
        to calculate the probability of the 'selected_market' winning.
        """
        inputs = match.model_inputs
        market = match.selected_market
        
        # Simple Poisson-based logic for Home Win vs Away Win based on xG
        # This uses the weights Laravel calculated
        raw_prob = (inputs.home_xg / (inputs.home_xg + inputs.away_xg))
        
        # Blend with the Bookmaker's implied probability (Market Sentiment)
        # Weight: 70% Model Inputs, 30% Market Odds
        blended_prob = (raw_prob * 0.7) + (market.implied_probability * 0.3)
        
        # Adjust for volatility (Higher volatility = push towards 50/50)
        volatility_factor = inputs.volatility_score / 10.0
        if blended_prob > 0.5:
            blended_prob -= (volatility_factor * 0.1)
        else:
            blended_prob += (volatility_factor * 0.1)

        return max(0.01, min(0.99, blended_prob))