# game_engine/engine/scoring.py (Updated)
"""
Updated scoring engine that evaluates slip quality based on selections
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Evaluates slip quality based on selection consistency and hedging"""
    
    def __init__(self):
        self.selection_consistency_weights = {
            'core': 1.2,      # Higher weight for core selections
            'hedge': 1.0,     # Standard weight for hedges
            'balanced': 0.9,  # Slightly lower for balanced
            'high_risk': 0.7  # Lower weight for high risk
        }
    
    def calculate_confidence_score(self, slip: Dict) -> float:
        """
        Calculate confidence score based on:
        1. Selection consistency
        2. Odds appropriateness
        3. Market diversity
        4. Portfolio fit
        """
        try:
            legs = slip.get('legs', [])
            variation_type = slip.get('variation_type', 'balanced')
            
            if not legs:
                return 0.5
            
            # Base score from variation type
            base_score = self.selection_consistency_weights.get(variation_type, 1.0)
            
            # Calculate odds consistency
            odds_scores = []
            for leg in legs:
                odds = leg.get('odds', 1.0)
                if odds < 1.1:
                    odds_scores.append(0.3)  # Too low odds
                elif odds < 2.0:
                    odds_scores.append(0.8)  # Good odds range
                elif odds < 5.0:
                    odds_scores.append(0.6)  # Moderate odds
                else:
                    odds_scores.append(0.4)  # Very high odds
            
            avg_odds_score = np.mean(odds_scores) if odds_scores else 0.5
            
            # Calculate market diversity score
            markets = [leg.get('market', '') for leg in legs]
            unique_markets = len(set(markets))
            market_score = min(1.0, unique_markets / len(markets) * 1.5)
            
            # Calculate selection clarity score
            selections = [leg.get('selection', '') for leg in legs]
            clear_selections = all(s and s != 'N/A' for s in selections)
            clarity_score = 1.0 if clear_selections else 0.3
            
            # Combine scores
            final_score = (
                base_score * 0.3 +
                avg_odds_score * 0.3 +
                market_score * 0.2 +
                clarity_score * 0.2
            )
            
            return max(0.1, min(0.95, final_score))
            
        except Exception as e:
            logger.warning(f"Confidence scoring failed: {e}")
            return 0.5
    
    def rank_slips(self, slips: List[Dict]) -> List[Dict]:
        """Rank slips by quality score"""
        if not slips:
            return slips
        
        # Calculate scores for all slips
        scored_slips = []
        for slip in slips:
            score = self.calculate_confidence_score(slip)
            slip['confidence_score'] = score
            scored_slips.append(slip)
        
        # Sort by score (descending)
        scored_slips.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return scored_slips
    
    def assign_risk_category(self, confidence_score: float) -> str:
        """Assign risk category based on confidence score"""
        if confidence_score >= 0.8:
            return "Low"
        elif confidence_score >= 0.6:
            return "Medium"
        elif confidence_score >= 0.4:
            return "High"
        else:
            return "Very High"