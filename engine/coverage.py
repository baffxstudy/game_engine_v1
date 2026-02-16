# game_engine/engine/coverage.py

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math

logger = logging.getLogger(__name__)

@dataclass
class SlipMetadata:
    """Enhanced slip metadata for intelligent stake distribution"""
    slip_id: str
    confidence_score: float
    total_odds: float
    legs: List[Dict[str, Any]]
    variation_type: str
    expected_value: float = 0.0
    risk_category: str = "medium"
    edge_score: float = 0.5  # How much better than market odds


class CoverageOptimizer:
    """
    Enhanced stake distribution optimizer with adaptive strategies.
    
    Features:
    1. Multiple distribution strategies (aggressive, balanced, conservative)
    2. Edge-aware allocation (considers expected value)
    3. Risk-adjusted weighting
    4. Portfolio diversification
    5. Minimum viable stake enforcement
    """
    
    # Distribution strategies
    STRATEGIES = {
        "aggressive": {"base_pool_pct": 0.2, "performance_pool_pct": 0.8, "power": 3},
        "balanced": {"base_pool_pct": 0.3, "performance_pool_pct": 0.7, "power": 2},
        "conservative": {"base_pool_pct": 0.4, "performance_pool_pct": 0.6, "power": 1.5},
        "equal": {"base_pool_pct": 1.0, "performance_pool_pct": 0.0, "power": 1}
    }
    
    def __init__(self, strategy: str = "balanced", min_stake: float = 0.01):
        """
        Initialize the coverage optimizer.
        
        Args:
            strategy: Distribution strategy ("aggressive", "balanced", "conservative", "equal")
            min_stake: Minimum stake per slip (prevents tiny, worthless bets)
        """
        self.strategy = strategy if strategy in self.STRATEGIES else "balanced"
        self.min_stake = max(0.01, min_stake)
        self.strategy_params = self.STRATEGIES[self.strategy]
        
        logger.info(f"Initialized CoverageOptimizer with '{self.strategy}' strategy")
    
    def _validate_inputs(self, total_stake: float, slips: List[SlipMetadata]) -> Tuple[bool, str]:
        """
        Validate inputs before processing.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if total_stake <= 0:
            return False, f"Total stake must be positive, got {total_stake}"
        
        if not slips:
            return False, "No slips provided for stake distribution"
        
        # Check if all slips have required metadata
        for i, slip in enumerate(slips):
            if slip.confidence_score < 0 or slip.confidence_score > 1:
                return False, f"Slip {i} has invalid confidence score: {slip.confidence_score}"
            
            if slip.total_odds < 1.01:
                return False, f"Slip {i} has invalid odds: {slip.total_odds}"
        
        # Check if total stake is sufficient for minimum stakes
        min_required = self.min_stake * len(slips)
        if total_stake < min_required:
            return False, f"Total stake {total_stake} is less than minimum required {min_required}"
        
        return True, ""
    
    def _calculate_edge_scores(self, slips: List[SlipMetadata]) -> List[float]:
        """
        Calculate edge scores for each slip based on odds and confidence.
        Higher edge = better value bet.
        """
        edge_scores = []
        
        for slip in slips:
            # Calculate implied probability from odds
            implied_prob = 1 / slip.total_odds
            
            # Edge = confidence - implied probability
            # Positive edge means we're more confident than the market suggests
            edge = slip.confidence_score - implied_prob
            
            # Adjust for number of legs (more legs = higher variance)
            leg_count = len(slip.legs)
            variance_factor = 1 / math.sqrt(leg_count)  # Diminishing risk with more legs
            
            # Adjust for variation type
            type_factor = {
                "primary": 1.0,
                "mixed": 0.9,
                "exploratory": 0.7
            }.get(slip.variation_type, 0.8)
            
            # Final edge score
            final_edge = max(0.01, edge * variance_factor * type_factor)
            slip.edge_score = final_edge
            edge_scores.append(final_edge)
        
        return edge_scores
    
    def _calculate_risk_weights(self, slips: List[SlipMetadata]) -> List[float]:
        """
        Calculate risk-adjusted weights for each slip.
        """
        risk_weights = []
        
        for slip in slips:
            # Base weight from confidence
            confidence_weight = slip.confidence_score ** 2
            
            # Adjust for odds (higher odds = higher risk)
            odds_factor = 1 / (1 + math.log(slip.total_odds))
            
            # Adjust for number of legs
            leg_factor = 1.0 if len(slip.legs) <= 3 else 0.8
            
            # Risk category adjustment
            risk_factor = {
                "low": 1.2,
                "medium": 1.0,
                "high": 0.7,
                "very_high": 0.4
            }.get(slip.risk_category, 1.0)
            
            # Combined weight
            weight = confidence_weight * odds_factor * leg_factor * risk_factor
            risk_weights.append(max(0.01, weight))
        
        return risk_weights
    
    def _calculate_diversification_penalties(self, slips: List[SlipMetadata]) -> List[float]:
        """
        Apply penalties to similar slips to encourage diversification.
        """
        if len(slips) <= 1:
            return [1.0] * len(slips)
        
        penalties = [1.0] * len(slips)
        
        # Group slips by similarity (simplified - same legs count for now)
        slip_groups = {}
        for i, slip in enumerate(slips):
            leg_count = len(slip.legs)
            if leg_count not in slip_groups:
                slip_groups[leg_count] = []
            slip_groups[leg_count].append(i)
        
        # Apply penalties within groups
        for group_indices in slip_groups.values():
            if len(group_indices) > 1:
                penalty = 1.0 / len(group_indices)
                for idx in group_indices:
                    penalties[idx] = penalty
        
        return penalties
    
    def _distribute_with_strategy(self, total_stake: float, slips: List[SlipMetadata]) -> List[float]:
        """
        Core distribution logic using configured strategy.
        """
        params = self.strategy_params
        
        # Calculate weights
        risk_weights = self._calculate_risk_weights(slips)
        edge_scores = self._calculate_edge_scores(slips)
        diversification_penalties = self._calculate_diversification_penalties(slips)
        
        # Combine weights
        combined_weights = []
        for i in range(len(slips)):
            # Power weighting based on strategy
            weight = (
                (risk_weights[i] ** params["power"]) *
                (edge_scores[i] ** 2) *
                diversification_penalties[i]
            )
            combined_weights.append(weight)
        
        # Base pool (flat distribution)
        base_pool = total_stake * params["base_pool_pct"]
        stake_floor = base_pool / len(slips) if len(slips) > 0 else 0
        
        # Performance pool (weighted distribution)
        performance_pool = total_stake * params["performance_pool_pct"]
        total_weight = sum(combined_weights)
        
        if total_weight <= 0:
            # Fallback to equal distribution
            logger.warning("Total weight is zero or negative, falling back to equal distribution")
            equal_stake = total_stake / len(slips)
            return [round(equal_stake, 2) for _ in slips]
        
        # Calculate final stakes
        stakes = []
        for i, weight in enumerate(combined_weights):
            # Performance bonus based on weight
            weight_ratio = weight / total_weight
            bonus = performance_pool * weight_ratio
            
            # Combined stake
            stake = stake_floor + bonus
            
            # Apply minimum stake
            stake = max(self.min_stake, stake)
            
            stakes.append(stake)
        
        return stakes
    
    def _normalize_stakes(self, stakes: List[float], total_stake: float) -> List[float]:
        """
        Normalize stakes to ensure they sum to total_stake exactly.
        """
        if not stakes:
            return []
        
        # Convert to Decimal for precise arithmetic
        stakes_decimal = [Decimal(str(s)) for s in stakes]
        total_decimal = Decimal(str(total_stake))
        
        current_sum = sum(stakes_decimal)
        
        if current_sum == total_decimal:
            # Already perfect
            return [float(s.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) for s in stakes_decimal]
        
        # Calculate adjustment factor
        adjustment_factor = total_decimal / current_sum
        
        # Apply adjustment
        adjusted_stakes = [s * adjustment_factor for s in stakes_decimal]
        
        # Round to 2 decimal places
        rounded_stakes = [float(s.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) for s in adjusted_stakes]
        
        # Final check and fix floating point errors
        rounded_sum = sum(rounded_stakes)
        diff = total_stake - rounded_sum
        
        if abs(diff) > 0.001:  # Significant difference
            logger.warning(f"Stake normalization error: sum={rounded_sum}, target={total_stake}, diff={diff}")
            # Distribute difference to largest stake
            if rounded_stakes:
                max_index = max(range(len(rounded_stakes)), key=lambda i: rounded_stakes[i])
                rounded_stakes[max_index] = round(rounded_stakes[max_index] + diff, 2)
        
        return rounded_stakes
    
    def distribute_stake(self, total_stake: float, slips: List[SlipMetadata]) -> List[float]:
        """
        Enhanced stake distribution with multiple optimization strategies.
        
        Args:
            total_stake: Total amount to distribute
            slips: List of SlipMetadata objects
            
        Returns:
            List of stakes for each slip
        """
        try:
            # Validate inputs
            is_valid, error_msg = self._validate_inputs(total_stake, slips)
            if not is_valid:
                logger.error(f"Input validation failed: {error_msg}")
                raise ValueError(error_msg)
            
            logger.info(f"Distributing {total_stake} across {len(slips)} slips using '{self.strategy}' strategy")
            
            # Calculate stakes using selected strategy
            raw_stakes = self._distribute_with_strategy(total_stake, slips)
            
            # Normalize to ensure exact total
            normalized_stakes = self._normalize_stakes(raw_stakes, total_stake)
            
            # Final validation
            final_sum = sum(normalized_stakes)
            if not math.isclose(final_sum, total_stake, rel_tol=0.01):
                logger.error(f"Stake distribution failed: sum={final_sum}, target={total_stake}")
                # Fallback to proportional by confidence
                return self._fallback_distribution(total_stake, slips)
            
            # Log distribution summary
            self._log_distribution_summary(slips, normalized_stakes)
            
            return normalized_stakes
            
        except Exception as e:
            logger.error(f"Stake distribution failed: {e}")
            # Fallback to simple proportional distribution
            return self._fallback_distribution(total_stake, slips)
    
    def _fallback_distribution(self, total_stake: float, slips: List[SlipMetadata]) -> List[float]:
        """Fallback distribution method when primary method fails"""
        logger.warning("Using fallback stake distribution")
        
        if not slips:
            return []
        
        # Simple proportional distribution by confidence
        confidences = [s.confidence_score for s in slips]
        total_confidence = sum(confidences)
        
        if total_confidence <= 0:
            # Equal distribution
            equal_stake = total_stake / len(slips)
            return [round(equal_stake, 2) for _ in slips]
        
        # Proportional distribution
        stakes = []
        for confidence in confidences:
            stake = (confidence / total_confidence) * total_stake
            stake = max(self.min_stake, stake)
            stakes.append(round(stake, 2))
        
        # Normalize
        return self._normalize_stakes(stakes, total_stake)
    
    def _log_distribution_summary(self, slips: List[SlipMetadata], stakes: List[float]):
        """Log detailed distribution summary"""
        if not slips or not stakes:
            return
        
        summary = []
        for i, (slip, stake) in enumerate(zip(slips, stakes)):
            summary.append({
                "slip_id": slip.slip_id[:8],
                "confidence": slip.confidence_score,
                "edge": slip.edge_score,
                "odds": slip.total_odds,
                "stake": stake,
                "stake_pct": round((stake / sum(stakes)) * 100, 1),
                "legs": len(slip.legs)
            })
        
        # Sort by stake (descending)
        summary.sort(key=lambda x: x["stake"], reverse=True)
        
        logger.info("Stake Distribution Summary:")
        for item in summary[:5]:  # Log top 5
            logger.info(
                f"  {item['slip_id']}: {item['stake']:.2f} ({item['stake_pct']}%) - "
                f"Conf: {item['confidence']:.2f}, Edge: {item['edge']:.3f}, "
                f"Odds: {item['odds']:.2f}, Legs: {item['legs']}"
            )
        
        # Log statistics
        stakes_list = [item["stake"] for item in summary]
        logger.info(
            f"  Stats: Min={min(stakes_list):.2f}, Max={max(stakes_list):.2f}, "
            f"Avg={np.mean(stakes_list):.2f}, Std={np.std(stakes_list):.2f}"
        )
    
    def optimize_portfolio(self, slips: List[Dict[str, Any]], 
                          total_stake: float,
                          strategy: str = None) -> List[Dict[str, Any]]:
        """
        Complete portfolio optimization with stake distribution.
        
        Args:
            slips: List of slip dictionaries from slip builder
            total_stake: Total stake amount
            strategy: Optional override for distribution strategy
            
        Returns:
            Enhanced slips with optimized stakes
        """
        if strategy and strategy in self.STRATEGIES:
            self.strategy = strategy
            self.strategy_params = self.STRATEGIES[strategy]
        
        # Convert slips to SlipMetadata objects
        slip_metadatas = []
        for slip in slips:
            try:
                metadata = SlipMetadata(
                    slip_id=slip.get("slip_id", f"unknown_{hash(str(slip))}"),
                    confidence_score=float(slip.get("confidence_score", 0.5)),
                    total_odds=float(slip.get("total_odds", 1.5)),
                    legs=slip.get("legs", []),
                    variation_type=slip.get("variation_type", "mixed"),
                    expected_value=float(slip.get("expected_value", 0.0)),
                    risk_category=slip.get("risk_category", "medium")
                )
                slip_metadatas.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to convert slip to metadata: {e}")
                continue
        
        if not slip_metadatas:
            logger.error("No valid slips for portfolio optimization")
            return slips  # Return original slips unchanged
        
        # Distribute stakes
        stakes = self.distribute_stake(total_stake, slip_metadatas)
        
        # Enhance slips with stakes and metadata
        enhanced_slips = []
        for i, (slip, stake) in enumerate(zip(slips, stakes)):
            enhanced_slip = slip.copy()
            enhanced_slip["optimized_stake"] = stake
            enhanced_slip["stake_percentage"] = round((stake / total_stake) * 100, 2)
            enhanced_slip["expected_return"] = round(stake * slip.get("total_odds", 1.0), 2)
            enhanced_slip["edge_score"] = slip_metadatas[i].edge_score
            enhanced_slip["optimization_strategy"] = self.strategy
            
            enhanced_slips.append(enhanced_slip)
        
        # Sort by expected return (descending)
        enhanced_slips.sort(key=lambda x: x.get("expected_return", 0), reverse=True)
        
        logger.info(f"Portfolio optimization complete: {len(enhanced_slips)} slips enhanced")
        return enhanced_slips


# Backward compatibility function
def distribute_stake(total_stake: float, num_slips: int, confidence_scores: List[float]) -> List[float]:
    """
    Backward compatible function for existing code.
    """
    optimizer = CoverageOptimizer(strategy="balanced")
    
    # Convert confidence scores to mock slips
    slips = []
    for i, confidence in enumerate(confidence_scores):
        slips.append(SlipMetadata(
            slip_id=f"legacy_{i}",
            confidence_score=confidence,
            total_odds=2.0,  # Default odds
            legs=[{"match_id": f"legacy_match_{i}"}],
            variation_type="primary"
        ))
    
    stakes = optimizer.distribute_stake(total_stake, slips)
    return stakes[:num_slips]  # Ensure correct length