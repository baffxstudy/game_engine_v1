# game_engine/engine/confidence_scorer.py

import numpy as np
from typing import List, Dict, Any
import math
import logging
from statistics import mean, stdev

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """
    Calculate confidence scores for slips and predictions.
    Combines multiple factors to assess prediction reliability.
    """
    
    def __init__(self, 
                 simulation_weight: float = 0.4,
                 probability_weight: float = 0.3,
                 market_weight: float = 0.2,
                 volatility_weight: float = 0.1):
        """
        Initialize confidence scorer with configurable weights.
        
        Args:
            simulation_weight: Weight for Monte Carlo simulation results
            probability_weight: Weight for probability consistency
            market_weight: Weight for market consensus
            volatility_weight: Weight for match volatility
        """
        self.simulation_weight = simulation_weight
        self.probability_weight = probability_weight
        self.market_weight = market_weight
        self.volatility_weight = volatility_weight
        
        # Validate weights sum to 1
        total = sum([simulation_weight, probability_weight, market_weight, volatility_weight])
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")
        
        logger.info(f"Initialized ConfidenceScorer with weights: "
                   f"simulation={simulation_weight}, probability={probability_weight}, "
                   f"market={market_weight}, volatility={volatility_weight}")
    
    def _calculate_simulation_confidence(self, match_sims: List[Dict]) -> float:
        """
        Calculate confidence based on simulation consistency.
        
        Args:
            match_sims: List of match simulation results
            
        Returns:
            Confidence score (0-1)
        """
        if not match_sims:
            return 0.5
        
        try:
            # Extract simulation success rates
            sim_successes = [sim.get("sim_success", 0.5) for sim in match_sims]
            
            # Average success rate
            avg_success = mean(sim_successes)
            
            # Consistency (lower standard deviation = higher confidence)
            if len(sim_successes) > 1:
                consistency = 1.0 - min(1.0, stdev(sim_successes))
            else:
                consistency = 0.8  # Assume decent consistency for single match
            
            # Weight average success with consistency
            simulation_confidence = avg_success * consistency
            
            # Adjust for number of matches (more matches = potentially lower confidence)
            match_count_factor = 1.0 / (1.0 + 0.1 * len(match_sims))  # Diminishing returns
            
            return simulation_confidence * match_count_factor
            
        except Exception as e:
            logger.warning(f"Simulation confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_probability_confidence(self, match_sims: List[Dict]) -> float:
        """
        Calculate confidence based on probability distributions.
        
        Args:
            match_sims: List of match simulation results
            
        Returns:
            Confidence score (0-1)
        """
        if not match_sims:
            return 0.5
        
        try:
            confidence_scores = []
            
            for sim in match_sims:
                # Get true probabilities for the match
                true_prob = sim.get("true_prob", {})
                if not true_prob:
                    # Try to get from context
                    if "context" in sim:
                        true_prob = sim["context"].get("inferred_probs", 
                                                      {'home': 0.33, 'draw': 0.34, 'away': 0.33})
                    else:
                        true_prob = {'home': 0.33, 'draw': 0.34, 'away': 0.33}
                
                # Calculate probability concentration
                probs = list(true_prob.values())
                
                if len(probs) < 2:
                    confidence_scores.append(0.5)
                    continue
                
                # Higher confidence when one outcome dominates
                max_prob = max(probs)
                
                # Calculate entropy (lower entropy = higher confidence)
                entropy = 0.0
                for p in probs:
                    if p > 0:
                        entropy -= p * math.log2(p)
                
                # Normalize entropy (max entropy for 3 outcomes is log2(3) ≈ 1.585)
                normalized_entropy = entropy / 1.585
                
                # Confidence is combination of max probability and low entropy
                prob_confidence = max_prob * (1.0 - normalized_entropy)
                confidence_scores.append(prob_confidence)
            
            # Return average confidence across all matches
            return mean(confidence_scores) if confidence_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Probability confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_market_confidence(self, match_sims: List[Dict]) -> float:
        """
        Calculate confidence based on market consensus.
        
        Args:
            match_sims: List of match simulation results
            
        Returns:
            Confidence score (0-1)
        """
        if not match_sims:
            return 0.5
        
        try:
            market_confidences = []
            
            for sim in match_sims:
                # Get market data from context
                context = sim.get("context")
                if not context:
                    market_confidences.append(0.5)
                    continue
                
                # Check if using fallback markets (lower confidence)
                primary_market = getattr(context, "primary_market", None)
                if primary_market and getattr(primary_market, "is_fallback", False):
                    market_confidences.append(0.3)  # Lower confidence for fallbacks
                    continue
                
                # Check market odds convergence
                safe_markets = getattr(context, "safe_markets", [])
                if len(safe_markets) < 2:
                    market_confidences.append(0.5)
                    continue
                
                # Calculate odds spread (tighter spread = higher confidence)
                odds = [market.odds for market in safe_markets[:3]]  # Top 3 markets
                if len(odds) > 1:
                    odds_spread = (max(odds) - min(odds)) / mean(odds)
                    # Convert spread to confidence (lower spread = higher confidence)
                    market_confidence = max(0.1, 1.0 - odds_spread)
                else:
                    market_confidence = 0.5
                
                market_confidences.append(market_confidence)
            
            return mean(market_confidences) if market_confidences else 0.5
            
        except Exception as e:
            logger.warning(f"Market confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_volatility_confidence(self, match_sims: List[Dict]) -> float:
        """
        Calculate confidence based on match volatility.
        
        Args:
            match_sims: List of match simulation results
            
        Returns:
            Confidence score (0-1)
        """
        if not match_sims:
            return 0.5
        
        try:
            volatility_scores = []
            
            for sim in match_sims:
                # Try to get volatility from various sources
                volatility = 5.0  # Default
                
                # Try from model inputs
                context = sim.get("context")
                if context and hasattr(context, "match"):
                    match_obj = context.match
                    if hasattr(match_obj, 'model_inputs'):
                        inputs = match_obj.model_inputs
                        vol = getattr(inputs, 'volatility_score', 5.0)
                        volatility = float(vol)
                
                # Convert volatility to confidence (higher volatility = lower confidence)
                # Volatility score typically 0-10, with 5 being average
                vol_confidence = 1.0 - (volatility / 20.0)  # Map 0-10 to 1.0-0.5
                volatility_scores.append(max(0.1, min(1.0, vol_confidence)))
            
            return mean(volatility_scores) if volatility_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Volatility confidence calculation failed: {e}")
            return 0.5
    
    def calculate_confidence_score(self, match_sims: List[Dict]) -> float:
        """
        Calculate overall confidence score for a slip.
        
        Args:
            match_sims: List of match simulation dictionaries
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not match_sims:
                logger.warning("No match simulations provided for confidence scoring")
                return 0.5
            
            # Calculate individual confidence components
            sim_confidence = self._calculate_simulation_confidence(match_sims)
            prob_confidence = self._calculate_probability_confidence(match_sims)
            market_confidence = self._calculate_market_confidence(match_sims)
            vol_confidence = self._calculate_volatility_confidence(match_sims)
            
            # Weighted combination
            total_confidence = (
                sim_confidence * self.simulation_weight +
                prob_confidence * self.probability_weight +
                market_confidence * self.market_weight +
                vol_confidence * self.volatility_weight
            )
            
            # Apply slip size adjustment
            # Larger slips (more matches) generally have lower confidence
            slip_size_factor = 1.0 / (1.0 + 0.15 * len(match_sims))
            
            final_confidence = total_confidence * slip_size_factor
            
            # Ensure bounds
            final_confidence = max(0.01, min(0.99, final_confidence))
            
            # Round to 3 decimal places
            final_confidence = round(final_confidence, 3)
            
            logger.debug(
                f"Confidence score calculated: {final_confidence} "
                f"(sim={sim_confidence:.3f}, prob={prob_confidence:.3f}, "
                f"market={market_confidence:.3f}, vol={vol_confidence:.3f})"
            )
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Confidence scoring failed: {e}")
            return 0.5
    
    def calculate_match_confidence(self, match_data: Dict) -> Dict:
        """
        Calculate detailed confidence metrics for a single match.
        
        Args:
            match_data: Match data dictionary
            
        Returns:
            Dictionary with confidence breakdown
        """
        try:
            # Create a mock match_sims structure for single match
            match_sim = {
                "true_prob": match_data.get("probabilities", 
                                          {'home': 0.33, 'draw': 0.34, 'away': 0.33}),
                "sim_success": match_data.get("simulation_confidence", 0.5),
                "context": type('obj', (object,), {
                    "primary_market": type('obj', (object,), {
                        "is_fallback": match_data.get("is_fallback_market", False),
                        "odds": match_data.get("odds", 1.5)
                    })(),
                    "safe_markets": [],
                    "match": type('obj', (object,), {
                        "model_inputs": type('obj', (object,), {
                            "volatility_score": match_data.get("volatility", 5.0)
                        })()
                    })()
                })()
            }
            
            # Calculate all components
            components = {
                "simulation_confidence": self._calculate_simulation_confidence([match_sim]),
                "probability_confidence": self._calculate_probability_confidence([match_sim]),
                "market_confidence": self._calculate_market_confidence([match_sim]),
                "volatility_confidence": self._calculate_volatility_confidence([match_sim])
            }
            
            # Calculate overall
            overall = (
                components["simulation_confidence"] * self.simulation_weight +
                components["probability_confidence"] * self.probability_weight +
                components["market_confidence"] * self.market_weight +
                components["volatility_confidence"] * self.volatility_weight
            )
            
            return {
                "overall_confidence": round(overall, 3),
                "confidence_components": {k: round(v, 3) for k, v in components.items()},
                "weights": {
                    "simulation": self.simulation_weight,
                    "probability": self.probability_weight,
                    "market": self.market_weight,
                    "volatility": self.volatility_weight
                }
            }
            
        except Exception as e:
            logger.error(f"Match confidence calculation failed: {e}")
            return {
                "overall_confidence": 0.5,
                "confidence_components": {},
                "error": str(e)
            }