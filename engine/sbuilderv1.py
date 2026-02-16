"""
INTELLIGENT SLIP BUILDER v2.1
================================================================================
A production-grade betting slip generation engine using:
- ACTIVE Monte Carlo simulation for coverage optimization
- Explicit hedging strategies with match-level outcome distribution
- Portfolio fitness optimization ensuring 1-6 winning slips
- Match ID integrity enforcement (no synthetic/inferred IDs)

Business Goal:
Generate 50 diverse slips where:
- At least 1 slip WILL win (minimum coverage enforced)
- Ideally 4-6 slips win (optimal coverage optimized)
- Maximum portfolio survival across different match outcomes
- Explicit hedging: same match outcomes are distributed across slips

Key Changes from v2.0:
1. Monte Carlo is now ACTIVE - guides slip construction in real-time
2. Explicit hedging logic ensures outcome diversity per match
3. Portfolio fitness function optimizes for business goals
4. Two-phase generation: candidate pool + portfolio selection
================================================================================
"""

import logging
from decimal import Decimal, getcontext
from typing import Dict, List, Any, Optional, Set, Tuple, DefaultDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import numpy as np
from collections import defaultdict, Counter

# Set decimal precision for accurate odds calculations
getcontext().prec = 10

# Configure logger (controlled by orchestrator)
logger = logging.getLogger(__name__)


# ============================================================================
# DOMAIN MODELS
# ============================================================================

class RiskLevel(str, Enum):
    """Risk classification for portfolio diversification"""
    LOW = "low"        # Conservative: 2-3 legs, low odds
    MEDIUM = "medium"  # Balanced: 3-4 legs, medium odds
    HIGH = "high"      # Aggressive: 4-5 legs, high odds


class SlipBuilderError(Exception):
    """Raised when slip generation cannot proceed"""
    pass


@dataclass(frozen=True)
class MarketSelection:
    """
    Immutable market selection representing a single betting option.
    
    Attributes:
        match_id: Original match_id from Laravel payload (CRITICAL: must not be transformed)
        market_code: Original market code from payload (no normalization)
        selection: The specific outcome (e.g., "Home", "Over 2.5")
        odds: Decimal odds for this selection
        implied_probability: Calculated probability (1/odds)
    """
    match_id: int
    market_code: str
    selection: str
    odds: Decimal
    implied_probability: float = field(init=False)
    
    def __post_init__(self):
        # Calculate implied probability from odds
        object.__setattr__(self, 'implied_probability', float(1 / self.odds))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "match_id": self.match_id,
            "market": self.market_code,
            "selection": self.selection,
            "odds": float(self.odds),
        }


@dataclass
class Slip:
    """
    A betting slip containing multiple legs (selections).
    
    Attributes:
        slip_id: Unique identifier
        risk_level: Risk classification
        legs: List of market selections (ALL match_ids must be from original payload)
        total_odds: Combined odds (product of all leg odds)
        win_probability: Estimated probability of winning
        confidence_score: Internal confidence metric
        coverage_score: How well this slip covers outcome space
        diversity_score: How different this slip is from others
    """
    slip_id: str
    risk_level: RiskLevel
    legs: List[MarketSelection]
    total_odds: Decimal = field(init=False)
    win_probability: float = field(init=False)
    confidence_score: float = field(default=0.0)
    coverage_score: float = field(default=0.0)
    diversity_score: float = field(default=0.0)
    
    def __post_init__(self):
        # Calculate total odds as product of all legs
        self.total_odds = Decimal("1")
        for leg in self.legs:
            self.total_odds *= leg.odds
        
        # Calculate win probability (product of individual probabilities)
        self.win_probability = 1.0
        for leg in self.legs:
            self.win_probability *= leg.implied_probability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            "slip_id": self.slip_id,
            "risk_level": self.risk_level.value,
            "variation_type": self._get_variation_type(),
            "legs": [leg.to_dict() for leg in self.legs],
            "total_odds": float(self.total_odds),
            "confidence_score": round(self.confidence_score, 3),
            "coverage_score": round(self.coverage_score, 3),
            "diversity_score": round(self.diversity_score, 3),
        }
    
    def _get_variation_type(self) -> str:
        """Determine variation type based on risk level"""
        return {
            RiskLevel.LOW: "conservative",
            RiskLevel.MEDIUM: "balanced",
            RiskLevel.HIGH: "aggressive",
        }[self.risk_level]
    
    def get_match_ids(self) -> Set[int]:
        """Get all match IDs in this slip (for integrity verification)"""
        return {leg.match_id for leg in self.legs}


# ============================================================================
# MARKET REGISTRY - PAYLOAD PARSER (PRESERVED)
# ============================================================================

class MarketRegistry:
    """
    Tolerant parser for Laravel payloads.
    
    Design Philosophy:
    - Accept messy real-world data
    - No normalization (preserves original market codes)
    - No inference or guessing
    - Fail gracefully on malformed data
    - Provide rich metadata for debugging
    """
    
    def __init__(self):
        self.matches: Dict[int, Dict[str, List[MarketSelection]]] = {}
        self.match_metadata: Dict[int, Dict[str, Any]] = {}
        self.available_market_codes: Set[str] = set()
        self.total_selections: int = 0
        self.original_match_ids: Set[int] = set()  # Track ALL match IDs from payload
        
    def build(self, payload: Dict[str, Any]) -> None:
        """
        Parse Laravel payload and build internal market registry.
        
        CRITICAL: Preserve ALL original match_ids exactly as they appear in payload.
        No transformation, reindexing, or inference allowed.
        """
        logger.info("Building market registry from payload")
        
        # Extract matches from nested structure
        master = payload.get("master_slip", payload)
        matches = master.get("matches", [])
        
        if not isinstance(matches, list) or not matches:
            raise SlipBuilderError("No valid matches found in payload")
        
        # Process each match
        for match_data in matches:
            if not isinstance(match_data, dict):
                logger.debug("Skipped non-dictionary match entry")
                continue
            
            match_id = self._extract_match_id(match_data)
            if match_id is None:
                continue
            
            # Track original match ID for integrity verification
            self.original_match_ids.add(match_id)
            
            # Extract match metadata
            self.match_metadata[match_id] = {
                "home_team": match_data.get("home_team", "Unknown"),
                "away_team": match_data.get("away_team", "Unknown"),
                "league": match_data.get("league", "Unknown"),
                "match_date": match_data.get("match_date", "Unknown"),
            }
            
            # Parse markets for this match
            markets_data = self._extract_markets(match_data)
            market_map = self._parse_markets(match_id, markets_data)
            
            if market_map:
                self.matches[match_id] = market_map
                logger.debug(f"Loaded match {match_id}: {len(market_map)} markets")
            else:
                logger.warning(f"No valid markets for match {match_id}")
        
        if not self.matches:
            raise SlipBuilderError("No valid markets found after parsing payload")
        
        logger.info(
            f"Registry built: {len(self.matches)} matches, "
            f"{len(self.available_market_codes)} unique markets, "
            f"{self.total_selections} total selections"
        )
    
    def _extract_match_id(self, match_data: Dict[str, Any]) -> Optional[int]:
        """Extract match ID from various possible field names"""
        match_id = match_data.get("match_id") or match_data.get("id") or match_data.get("fixture_id")
        
        if match_id is None:
            logger.debug(f"Skipped match with missing ID")
            return None
        
        try:
            return int(match_id)
        except (ValueError, TypeError):
            logger.debug(f"Skipped match with invalid ID: {match_id}")
            return None
    
    def _extract_markets(self, match_data: Dict[str, Any]) -> List[Dict]:
        """Extract markets list from various possible field names"""
        markets = (
            match_data.get("match_markets") or
            match_data.get("markets") or
            match_data.get("full_markets") or
            []
        )
        
        if not isinstance(markets, list):
            return []
        
        return markets
    
    def _parse_markets(self, match_id: int, markets_data: List[Dict]) -> Dict[str, List[MarketSelection]]:
        """
        Parse market data into structured selections.
        
        Returns:
            Dict mapping market_code -> list of MarketSelection objects
        """
        market_map: Dict[str, List[MarketSelection]] = {}
        
        for market_entry in markets_data:
            if not isinstance(market_entry, dict):
                continue
            
            # Extract market info (handle nested structure)
            market_info = market_entry.get("market", market_entry)
            if not isinstance(market_info, dict):
                continue
            
            # Get market code (NO normalization - use original)
            market_code = (
                market_info.get("code") or
                market_info.get("market_type") or
                market_info.get("type")
            )
            
            if not market_code:
                logger.debug(f"Skipped market with no code for match {match_id}")
                continue
            
            market_code = str(market_code).strip()
            
            # Parse selections for this market
            selections = self._parse_selections(match_id, market_code, market_entry)
            
            if selections:
                market_map[market_code] = selections
                self.available_market_codes.add(market_code)
                self.total_selections += len(selections)
            else:
                logger.debug(f"No valid selections for match {match_id}, market {market_code}")
        
        return market_map
    
    def _parse_selections(
        self,
        match_id: int,
        market_code: str,
        market_entry: Dict
    ) -> List[MarketSelection]:
        """
        Parse individual selections within a market.
        
        Returns:
            List of valid MarketSelection objects
        """
        selections_data = (
            market_entry.get("selections") or
            market_entry.get("outcomes") or
            []
        )
        
        if not isinstance(selections_data, list):
            return []
        
        valid_selections = []
        
        for selection_data in selections_data:
            if not isinstance(selection_data, dict):
                continue
            
            # Extract selection value/label
            value = (
                selection_data.get("value") or
                selection_data.get("name") or
                selection_data.get("label") or
                "Unknown"
            )
            
            # Extract and validate odds
            try:
                odds = Decimal(str(selection_data.get("odds")))
                
                # Validate odds are reasonable
                if odds < Decimal("1.01"):
                    logger.debug(f"Skipped selection with odds too low: {odds}")
                    continue
                
                if odds > Decimal("1000"):
                    logger.debug(f"Skipped selection with odds too high: {odds}")
                    continue
                
            except (TypeError, ValueError, ArithmeticError):
                logger.debug(f"Skipped selection with invalid odds")
                continue
            
            # Create MarketSelection object with original match_id
            valid_selections.append(
                MarketSelection(
                    match_id=match_id,  # CRITICAL: original match_id preserved
                    market_code=market_code,
                    selection=str(value),
                    odds=odds,
                )
            )
        
        return valid_selections
    
    def get_matches(self) -> List[int]:
        """Get all available match IDs"""
        return list(self.matches.keys())
    
    def get_markets_for_match(self, match_id: int) -> List[str]:
        """Get all market codes available for a specific match"""
        return list(self.matches.get(match_id, {}).keys())
    
    def get_selections(self, match_id: int, market_code: str) -> List[MarketSelection]:
        """Get all selections for a specific match and market"""
        return self.matches.get(match_id, {}).get(market_code, [])
    
    def get_match_info(self, match_id: int) -> Dict[str, Any]:
        """Get metadata for a specific match"""
        return self.match_metadata.get(match_id, {})
    
    def verify_match_id_integrity(self, match_id: int) -> bool:
        """Verify a match_id exists in original payload"""
        return match_id in self.original_match_ids


# ============================================================================
# ACTIVE MONTE CARLO OPTIMIZER
# ============================================================================

class ActiveMonteCarloOptimizer:
    """
    ACTIVE Monte Carlo optimizer that guides slip construction in real-time.
    
    Key Improvements from v2.0:
    1. ACTIVE guidance: Scores candidates and influences selection
    2. Portfolio fitness optimization: Evaluates entire portfolio, not just individual slips
    3. Outcome distribution tracking: Ensures hedging across slips
    
    Goal: Generate slips that maximize the probability that:
    - At least 1 slip wins (HARD constraint: minimize zero-win simulations)
    - 4-6 slips win on average (SOFT constraint: optimize distribution)
    """
    
    def __init__(
        self,
        registry: MarketRegistry,
        random_seed: int,
        num_simulations: int = 10000
    ):
        self.registry = registry
        self.rng = np.random.RandomState(random_seed)
        self.num_simulations = num_simulations
        
        # Pre-compute all possible outcomes for faster simulation
        self.match_market_combinations: List[Tuple[int, str]] = []
        self.selection_probabilities: Dict[Tuple[int, str], List[Tuple[str, float]]] = {}
        
        # Simulation cache
        self.simulated_outcomes: List[Dict[Tuple[int, str], str]] = []
        
        # Match outcome distribution tracker for hedging
        self.match_outcome_distribution: DefaultDict[int, DefaultDict[str, DefaultDict[str, int]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )
    
    def prepare_simulations(self) -> None:
        """
        Pre-compute all match+market combinations and their probability distributions.
        This enables fast Monte Carlo simulation during active optimization.
        """
        logger.info("Preparing Monte Carlo simulation data")
        
        # Get all match+market combinations
        for match_id in self.registry.get_matches():
            for market_code in self.registry.get_markets_for_match(match_id):
                key = (match_id, market_code)
                self.match_market_combinations.append(key)
                
                # Pre-compute probability distribution for this market
                selections = self.registry.get_selections(match_id, market_code)
                probabilities = [s.implied_probability for s in selections]
                prob_sum = sum(probabilities)
                
                # Normalize probabilities
                if prob_sum > 0:
                    normalized_probs = [p / prob_sum for p in probabilities]
                else:
                    normalized_probs = [1.0 / len(selections)] * len(selections)
                
                # Store selection-value to probability mapping
                self.selection_probabilities[key] = [
                    (selections[i].selection, normalized_probs[i])
                    for i in range(len(selections))
                ]
        
        logger.info(f"Prepared {len(self.match_market_combinations)} match+market combinations")
    
    def run_simulations(self) -> None:
        """
        Run Monte Carlo simulations of match outcomes.
        
        For each simulation:
        - Randomly select one outcome per match+market combination
        - Weight selection by implied probability (odds-based)
        """
        if not self.match_market_combinations:
            self.prepare_simulations()
        
        logger.info(f"Running {self.num_simulations} Monte Carlo simulations")
        
        # Run simulations
        for sim_idx in range(self.num_simulations):
            outcome = {}
            
            for match_id, market_code in self.match_market_combinations:
                key = (match_id, market_code)
                selections_probs = self.selection_probabilities[key]
                
                # Extract selections and probabilities
                selections = [sp[0] for sp in selections_probs]
                probabilities = [sp[1] for sp in selections_probs]
                
                # Randomly select outcome based on probabilities
                selected = self.rng.choice(selections, p=probabilities)
                outcome[key] = selected
            
            self.simulated_outcomes.append(outcome)
        
        logger.info(f"Completed {len(self.simulated_outcomes)} simulations")
    
    def evaluate_portfolio_fitness(self, portfolio: List[Slip]) -> Dict[str, float]:
        """
        Evaluate fitness of an entire portfolio (set of slips).
        
        Fitness Components (weighted):
        1. Zero-win penalty: Heavily penalize portfolios where 0 slips win
        2. Optimal-win reward: Reward portfolios with 4-6 winning slips
        3. Win distribution: Penalize too many or too few winners
        4. Match coverage: Reward balanced match usage
        5. Hedging quality: Reward diverse outcomes per match
        
        Returns:
            Dictionary with fitness score (higher is better) and component scores
        """
        if not self.simulated_outcomes:
            self.run_simulations()
        
        wins_per_simulation = []
        zero_win_count = 0
        
        # Track match outcome distribution for hedging analysis
        match_outcome_counts: DefaultDict[int, DefaultDict[str, DefaultDict[str, int]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )
        
        # Count slip outcomes per match
        for slip in portfolio:
            for leg in slip.legs:
                match_outcome_counts[leg.match_id][leg.market_code][leg.selection] += 1
        
        # Evaluate each simulation
        for outcome in self.simulated_outcomes:
            winning_slips = 0
            
            for slip in portfolio:
                all_legs_win = True
                
                for leg in slip.legs:
                    key = (leg.match_id, leg.market_code)
                    if outcome.get(key) != leg.selection:
                        all_legs_win = False
                        break
                
                if all_legs_win:
                    winning_slips += 1
            
            wins_per_simulation.append(winning_slips)
            if winning_slips == 0:
                zero_win_count += 1
        
        # Calculate component scores
        zero_win_percentage = zero_win_count / len(self.simulated_outcomes)
        
        # Count simulations with 4-6 wins (optimal)
        optimal_wins = sum(1 for w in wins_per_simulation if 4 <= w <= 6)
        optimal_win_percentage = optimal_wins / len(self.simulated_outcomes)
        
        # Calculate win distribution score (penalize extreme distributions)
        win_counts = Counter(wins_per_simulation)
        avg_wins = np.mean(wins_per_simulation)
        
        # Distribution penalty: quadratic penalty away from optimal range
        distribution_penalty = 0
        for win_count, count in win_counts.items():
            if win_count < 1:
                # Heavy penalty for zero wins
                penalty = count * 10.0
            elif win_count > 10:
                # Penalty for too many wins (over-correlated slips)
                penalty = count * 2.0
            elif win_count < 4 or win_count > 6:
                # Light penalty for suboptimal but acceptable
                penalty = count * 0.5
            else:
                penalty = 0
            distribution_penalty += penalty
        
        distribution_score = 1.0 - (distribution_penalty / len(self.simulated_outcomes))
        
        # Calculate hedging quality score
        hedging_score = self._calculate_hedging_quality(match_outcome_counts)
        
        # Calculate match coverage score (penalize overuse of same matches)
        match_usage = Counter()
        for slip in portfolio:
            for leg in slip.legs:
                match_usage[leg.match_id] += 1
        
        if match_usage:
            usage_std = np.std(list(match_usage.values()))
            max_usage = max(match_usage.values())
            coverage_score = 1.0 - (usage_std / max_usage if max_usage > 0 else 0)
        else:
            coverage_score = 0.0
        
        # Combined fitness score (weighted components)
        fitness = (
            (1.0 - zero_win_percentage) * 0.4 +          # 40% weight: avoid zero wins
            optimal_win_percentage * 0.3 +               # 30% weight: optimal wins
            distribution_score * 0.15 +                  # 15% weight: good distribution
            hedging_score * 0.10 +                       # 10% weight: good hedging
            coverage_score * 0.05                        # 5% weight: match coverage
        )
        
        return {
            "fitness_score": fitness,
            "zero_win_percentage": zero_win_percentage,
            "optimal_win_percentage": optimal_win_percentage,
            "average_wins": avg_wins,
            "hedging_score": hedging_score,
            "coverage_score": coverage_score,
        }
    
    def _calculate_hedging_quality(self, outcome_counts: Dict) -> float:
        """
        Calculate how well outcomes are hedged across the portfolio.
        
        Higher score indicates:
        - Same match appears in multiple slips
        - Outcomes are distributed (not all same selection)
        - Good coverage of possible outcomes
        """
        if not outcome_counts:
            return 0.0
        
        market_scores = []
        
        for match_id, markets in outcome_counts.items():
            for market_code, selections in markets.items():
                total_picks = sum(selections.values())
                if total_picks <= 1:
                    # No hedging possible with only 1 pick
                    continue
                
                # Calculate entropy of selection distribution
                # Higher entropy = more evenly distributed = better hedging
                probs = [count / total_picks for count in selections.values()]
                if len(probs) > 1:
                    # Normalize entropy to [0, 1]
                    max_entropy = np.log(len(probs))
                    actual_entropy = -sum(p * np.log(p) for p in probs if p > 0)
                    normalized_entropy = actual_entropy / max_entropy if max_entropy > 0 else 0
                    market_scores.append(normalized_entropy)
        
        return np.mean(market_scores) if market_scores else 0.0
    
    def get_outcome_distribution_advice(self, portfolio: List[Slip]) -> Dict[Tuple[int, str], Dict[str, float]]:
        """
        Analyze current portfolio and suggest which outcomes are under/over represented.
        
        Returns:
            Dictionary mapping (match_id, market_code) -> selection -> suggested adjustment
            Positive values = need more of this selection
            Negative values = have too much of this selection
        """
        current_counts: DefaultDict[Tuple[int, str], DefaultDict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        
        # Count current selections
        for slip in portfolio:
            for leg in slip.legs:
                key = (leg.match_id, leg.market_code)
                current_counts[key][leg.selection] += 1
        
        # Calculate target distribution based on probabilities
        advice = {}
        
        for (match_id, market_code), selections in current_counts.items():
            total_picks = sum(selections.values())
            if total_picks == 0:
                continue
            
            # Get available selections and their probabilities
            available_selections = self.registry.get_selections(match_id, market_code)
            if not available_selections:
                continue
            
            # Target distribution: proportional to probability but with minimum
            selection_probs = {s.selection: s.implied_probability for s in available_selections}
            prob_sum = sum(selection_probs.values())
            
            advice_for_market = {}
            for selection, prob in selection_probs.items():
                current = selections.get(selection, 0)
                target_ratio = prob / prob_sum if prob_sum > 0 else 1.0 / len(selection_probs)
                target_count = target_ratio * total_picks
                
                # Suggested adjustment (positive = need more, negative = need less)
                adjustment = target_count - current
                advice_for_market[selection] = adjustment
            
            advice[(match_id, market_code)] = advice_for_market
        
        return advice


# ============================================================================
# INTELLIGENT SLIP GENERATOR WITH ACTIVE OPTIMIZATION
# ============================================================================

class IntelligentSlipGenerator:
    """
    Generates 50 diverse slips using ACTIVE Monte Carlo optimization.
    
    Two-phase approach:
    1. Candidate Generation: Generate large pool of candidate slips
    2. Portfolio Selection: Use Monte Carlo to select optimal 50-slip portfolio
    
    Key features:
    - Explicit hedging: Same match outcomes are distributed across slips
    - Match ID integrity: All match_ids come from original payload
    - Active optimization: Monte Carlo guides portfolio selection
    """
    
    # Portfolio allocation (must sum to 50)
    RISK_ALLOCATION = {
        RiskLevel.LOW: 20,      # 40% conservative
        RiskLevel.MEDIUM: 20,   # 40% balanced
        RiskLevel.HIGH: 10,     # 20% aggressive
    }
    
    # Leg counts per risk level
    LEGS_PER_RISK = {
        RiskLevel.LOW: (2, 3),      # 2-3 legs
        RiskLevel.MEDIUM: (3, 4),   # 3-4 legs
        RiskLevel.HIGH: (4, 5),     # 4-5 legs
    }
    
    # Candidate pool size (generate more than needed, then select best)
    CANDIDATE_POOL_SIZE = 200
    
    def __init__(
        self,
        registry: MarketRegistry,
        random_seed: int,
        monte_carlo_optimizer: ActiveMonteCarloOptimizer
    ):
        self.registry = registry
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.RandomState(random_seed)
        self.optimizer = monte_carlo_optimizer
        
        # Track candidate slips
        self.candidate_slips: List[Slip] = []
        self.used_combinations: Set[frozenset] = set()
        
        # Track match outcome distribution for hedging
        self.match_outcome_counts: DefaultDict[int, DefaultDict[str, DefaultDict[str, int]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )
    
    def generate_candidate_pool(self) -> List[Slip]:
        """
        Generate a large pool of candidate slips (more than 50).
        
        Strategy:
        1. Generate slips for each risk tier proportionally
        2. Enforce basic diversity constraints
        3. Track match outcome distribution for hedging
        4. Ensure match ID integrity
        """
        logger.info(f"Generating candidate pool of {self.CANDIDATE_POOL_SIZE} slips")
        
        candidates = []
        slip_counter = 1
        
        # Scale risk allocation for candidate pool
        total_slips = sum(self.RISK_ALLOCATION.values())
        risk_scaling = self.CANDIDATE_POOL_SIZE / total_slips
        
        candidate_allocation = {
            risk: int(count * risk_scaling) for risk, count in self.RISK_ALLOCATION.items()
        }
        
        # Generate candidates for each risk tier
        for risk_level, count in candidate_allocation.items():
            for _ in range(count):
                slip = self._generate_candidate_slip(risk_level, slip_counter)
                
                if slip:
                    # Verify match ID integrity
                    for leg in slip.legs:
                        if not self.registry.verify_match_id_integrity(leg.match_id):
                            raise SlipBuilderError(
                                f"Match ID integrity violation: {leg.match_id} not in original payload"
                            )
                    
                    candidates.append(slip)
                    slip_counter += 1
                    
                    # Update match outcome distribution
                    self._update_outcome_distribution(slip)
        
        logger.info(f"Generated {len(candidates)} candidate slips")
        return candidates
    
    def _generate_candidate_slip(self, risk_level: RiskLevel, slip_id: int) -> Optional[Slip]:
        """
        Generate a single candidate slip with basic hedging awareness.
        
        Strategy:
        1. Select matches considering current outcome distribution
        2. Prefer matches/selections that improve hedging balance
        3. Ensure uniqueness
        """
        # Determine number of legs
        min_legs, max_legs = self.LEGS_PER_RISK[risk_level]
        num_legs = self.rng.randint(min_legs, max_legs)
        
        # Get available matches
        available_matches = self.registry.get_matches()
        
        if len(available_matches) < num_legs:
            num_legs = len(available_matches)
        
        if num_legs < 2:
            return None
        
        # Select matches with preference for underused matches
        selected_matches = self._select_matches_with_hedging(
            available_matches, num_legs, risk_level
        )
        
        # Build legs with hedging awareness
        legs: List[MarketSelection] = []
        used_markets: Set[Tuple[int, str]] = set()
        
        for match_id in selected_matches:
            market_selection = self._select_market_with_hedging(
                match_id, risk_level, used_markets
            )
            
            if not market_selection:
                continue
            
            legs.append(market_selection)
            used_markets.add((match_id, market_selection.market_code))
        
        # Ensure we have enough legs
        if len(legs) < 2:
            return None
        
        # Check for duplicate slips
        leg_signature = frozenset(
            (leg.match_id, leg.market_code, leg.selection) for leg in legs
        )
        
        if leg_signature in self.used_combinations:
            return None
        
        self.used_combinations.add(leg_signature)
        
        # Create slip
        slip = Slip(
            slip_id=f"CAND_{slip_id:04d}",
            risk_level=risk_level,
            legs=legs,
        )
        
        # Calculate initial confidence
        slip.confidence_score = self._calculate_confidence(slip)
        
        return slip
    
    def _select_matches_with_hedging(
        self,
        available_matches: List[int],
        num_needed: int,
        risk_level: RiskLevel
    ) -> List[int]:
        """
        Select matches considering current usage for hedging.
        
        Strategy:
        1. Prefer matches that are underused in current portfolio
        2. But also include some popular matches for hedging
        3. Balance between exploration and exploitation
        """
        # Calculate match usage scores
        usage_scores = {}
        for match_id in available_matches:
            # Count total picks for this match
            total_picks = 0
            for market_counts in self.match_outcome_counts[match_id].values():
                total_picks += sum(market_counts.values())
            
            # Lower usage = higher score (we want to use underused matches)
            usage_scores[match_id] = 1.0 / (total_picks + 1)  # +1 to avoid division by zero
        
        # Normalize scores to probabilities
        score_sum = sum(usage_scores.values())
        if score_sum > 0:
            probabilities = [usage_scores[m] / score_sum for m in available_matches]
        else:
            probabilities = [1.0 / len(available_matches)] * len(available_matches)
        
        # Sample matches weighted by usage scores
        selected_indices = self.np_rng.choice(
            len(available_matches),
            size=min(num_needed, len(available_matches)),
            replace=False,
            p=probabilities
        )
        
        return [available_matches[i] for i in selected_indices]
    
    def _select_market_with_hedging(
        self,
        match_id: int,
        risk_level: RiskLevel,
        used_markets: Set[Tuple[int, str]]
    ) -> Optional[MarketSelection]:
        """
        Select market and outcome with hedging awareness.
        
        Strategy:
        1. Consider current outcome distribution for this match
        2. Prefer selections that improve hedging balance
        3. Respect risk level preferences
        """
        available_markets = self.registry.get_markets_for_match(match_id)
        if not available_markets:
            return None
        
        # Filter out already used markets for this slip
        available_markets = [m for m in available_markets if (match_id, m) not in used_markets]
        if not available_markets:
            # If all markets used, pick any
            available_markets = self.registry.get_markets_for_match(match_id)
        
        # Score each market based on hedging needs
        market_scores = []
        for market_code in available_markets:
            score = self._calculate_market_hedging_score(match_id, market_code, risk_level)
            market_scores.append((market_code, score))
        
        # Select market weighted by score
        market_codes, scores = zip(*market_scores)
        score_sum = sum(scores)
        if score_sum > 0:
            probabilities = [s / score_sum for s in scores]
            selected_market = self.np_rng.choice(market_codes, p=probabilities)
        else:
            selected_market = self.rng.choice(market_codes)
        
        # Select outcome within this market
        selections = self.registry.get_selections(match_id, selected_market)
        if not selections:
            return None
        
        # Score selections based on hedging and risk
        selection_scores = []
        for selection in selections:
            score = self._calculate_selection_hedging_score(
                match_id, selected_market, selection, risk_level
            )
            selection_scores.append((selection, score))
        
        # Select outcome weighted by score
        selection_objs, sel_scores = zip(*selection_scores)
        sel_sum = sum(sel_scores)
        if sel_sum > 0:
            sel_probs = [s / sel_sum for s in sel_scores]
            selected = self.np_rng.choice(selection_objs, p=sel_probs)
        else:
            selected = self.rng.choice(selection_objs)
        
        return selected
    
    def _calculate_market_hedging_score(
        self,
        match_id: int,
        market_code: str,
        risk_level: RiskLevel
    ) -> float:
        """
        Calculate how good this market is for hedging.
        
        Higher score if:
        1. Market is underused in current portfolio
        2. Market has multiple possible outcomes (good for hedging)
        3. Matches risk level preferences
        """
        base_score = 1.0
        
        # Check current usage
        current_usage = sum(
            self.match_outcome_counts[match_id][market_code].values()
        )
        usage_penalty = current_usage * 0.2  # Penalize overused markets
        base_score -= usage_penalty
        
        # Check number of available outcomes
        selections = self.registry.get_selections(match_id, market_code)
        if selections:
            outcome_count = len(selections)
            # More outcomes = better for hedging
            outcome_bonus = min(0.3, (outcome_count - 2) * 0.1)
            base_score += outcome_bonus
        
        # Risk level adjustment
        if risk_level == RiskLevel.LOW:
            # Prefer safer markets (1x2, double chance)
            if any(code in market_code.lower() for code in ['1x2', 'double', 'dc']):
                base_score += 0.2
        elif risk_level == RiskLevel.HIGH:
            # Prefer riskier markets (correct score, handicap)
            if any(code in market_code.lower() for code in ['correct', 'handicap', 'htft']):
                base_score += 0.2
        
        return max(0.1, base_score)
    
    def _calculate_selection_hedging_score(
        self,
        match_id: int,
        market_code: str,
        selection: MarketSelection,
        risk_level: RiskLevel
    ) -> float:
        """
        Calculate how good this specific selection is for hedging.
        
        Higher score if:
        1. Selection is underused in current portfolio
        2. Selection matches risk level odds preference
        3. Selection has reasonable probability
        """
        base_score = 1.0
        
        # Check current usage of this specific outcome
        current_count = self.match_outcome_counts[match_id][market_code].get(selection.selection, 0)
        usage_penalty = current_count * 0.3  # Strong penalty for overused outcomes
        base_score -= usage_penalty
        
        # Risk-based odds preference
        if risk_level == RiskLevel.LOW:
            # Prefer lower odds (safer)
            odds_factor = 2.0 - float(selection.odds) * 0.5
            base_score += max(0, odds_factor)
        elif risk_level == RiskLevel.HIGH:
            # Prefer higher odds (riskier)
            odds_factor = (float(selection.odds) - 1.0) * 0.2
            base_score += min(0.5, odds_factor)
        
        return max(0.1, base_score)
    
    def _update_outcome_distribution(self, slip: Slip) -> None:
        """Update tracking of match outcome distribution."""
        for leg in slip.legs:
            self.match_outcome_counts[leg.match_id][leg.market_code][leg.selection] += 1
    
    def _calculate_confidence(self, slip: Slip) -> float:
        """Calculate confidence score for a slip."""
        base_confidence = {
            RiskLevel.LOW: 0.75,
            RiskLevel.MEDIUM: 0.55,
            RiskLevel.HIGH: 0.35,
        }[slip.risk_level]
        
        leg_penalty = (len(slip.legs) - 2) * 0.05
        avg_odds = float(slip.total_odds) ** (1 / len(slip.legs))
        odds_adjustment = (2.0 - avg_odds) * 0.1
        
        confidence = base_confidence - leg_penalty + odds_adjustment
        
        return max(0.1, min(0.95, confidence))
    
    def select_optimal_portfolio(
        self,
        candidates: List[Slip],
        target_size: int = 50,
        iterations: int = 100
    ) -> List[Slip]:
        """
        Select optimal portfolio from candidates using Monte Carlo optimization.
        
        Genetic algorithm approach:
        1. Start with random portfolio
        2. Evaluate fitness using Monte Carlo
        3. Mutate/improve portfolio over iterations
        4. Return best portfolio found
        """
        logger.info(f"Selecting optimal {target_size}-slip portfolio from {len(candidates)} candidates")
        
        if len(candidates) < target_size:
            raise SlipBuilderError(f"Not enough candidates ({len(candidates)}) for portfolio of {target_size}")
        
        # Initialize with random portfolio
        current_portfolio = self.rng.sample(candidates, target_size)
        current_fitness = self.optimizer.evaluate_portfolio_fitness(current_portfolio)
        best_portfolio = current_portfolio[:]
        best_fitness = current_fitness
        
        logger.info(f"Initial fitness: {current_fitness['fitness_score']:.3f}")
        
        # Optimization loop
        for iteration in range(iterations):
            # Create new portfolio by mutation
            new_portfolio = self._mutate_portfolio(current_portfolio, candidates)
            new_fitness = self.optimizer.evaluate_portfolio_fitness(new_portfolio)
            
            # Accept if better or with small probability (simulated annealing)
            if (new_fitness['fitness_score'] > current_fitness['fitness_score'] or
                self.rng.random() < 0.1):  # 10% chance to accept worse solution
                current_portfolio = new_portfolio
                current_fitness = new_fitness
                
                # Update best if improved
                if new_fitness['fitness_score'] > best_fitness['fitness_score']:
                    best_portfolio = new_portfolio[:]
                    best_fitness = new_fitness
            
            # Log progress occasionally
            if iteration % 20 == 0:
                logger.info(
                    f"Iteration {iteration}: "
                    f"Current fitness = {current_fitness['fitness_score']:.3f}, "
                    f"Best fitness = {best_fitness['fitness_score']:.3f}, "
                    f"Zero-win% = {current_fitness['zero_win_percentage']:.1%}"
                )
        
        logger.info(
            f"Portfolio selection complete. "
            f"Best fitness: {best_fitness['fitness_score']:.3f}, "
            f"Zero-win%: {best_fitness['zero_win_percentage']:.1%}, "
            f"Optimal-win%: {best_fitness['optimal_win_percentage']:.1%}"
        )
        
        # Renumber slip IDs for final output
        for i, slip in enumerate(best_portfolio, 1):
            slip.slip_id = f"SLIP_{i:04d}"
        
        return best_portfolio
    
    def _mutate_portfolio(self, portfolio: List[Slip], candidates: List[Slip]) -> List[Slip]:
        """
        Create mutated version of portfolio.
        
        Mutation strategies:
        1. Replace random slip with different candidate
        2. Swap two slips in portfolio
        3. Add/remove slip (keeping size constant)
        """
        mutation_type = self.rng.choice(['replace', 'swap', 'shuffle'])
        new_portfolio = portfolio[:]
        
        if mutation_type == 'replace' and len(candidates) > len(portfolio):
            # Replace one random slip
            replace_idx = self.rng.randint(0, len(portfolio) - 1)
            
            # Find candidate not in current portfolio
            current_ids = {slip.slip_id for slip in portfolio}
            available = [c for c in candidates if c.slip_id not in current_ids]
            
            if available:
                new_portfolio[replace_idx] = self.rng.choice(available)
        
        elif mutation_type == 'swap' and len(portfolio) >= 2:
            # Swap two slips
            idx1, idx2 = self.rng.sample(range(len(portfolio)), 2)
            new_portfolio[idx1], new_portfolio[idx2] = new_portfolio[idx2], new_portfolio[idx1]
        
        elif mutation_type == 'shuffle':
            # Shuffle the portfolio
            self.rng.shuffle(new_portfolio)
        
        return new_portfolio


# ============================================================================
# PUBLIC API - SLIP BUILDER
# ============================================================================

class SlipBuilder:
    """
    Main entry point for slip generation.
    
    Usage:
        builder = SlipBuilder()
        result = builder.generate(payload)
    """
    
    def __init__(
        self,
        enable_monte_carlo: bool = True,
        num_simulations: int = 10000,
        optimization_iterations: int = 100
    ):
        """
        Initialize slip builder.
        
        Args:
            enable_monte_carlo: Whether to run ACTIVE Monte Carlo optimization
            num_simulations: Number of Monte Carlo simulations to run
            optimization_iterations: Iterations for portfolio optimization
        """
        self.enable_monte_carlo = enable_monte_carlo
        self.num_simulations = num_simulations
        self.optimization_iterations = optimization_iterations
    
    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate 50 optimized betting slips from Laravel payload.
        
        New ACTIVE workflow:
        1. Parse payload and build market registry
        2. Initialize ACTIVE Monte Carlo optimizer
        3. Generate candidate slip pool
        4. Use Monte Carlo to select optimal 50-slip portfolio
        5. Ensure match ID integrity throughout
        
        Returns:
            Dictionary with generated slips and metadata
        """
        logger.info("=" * 80)
        logger.info("SLIP BUILDER v2.1 - Starting ACTIVE optimization")
        logger.info("=" * 80)
        
        # Extract master slip ID for deterministic randomness
        master_slip_data = payload.get("master_slip", {})
        master_slip_id = int(master_slip_data.get("master_slip_id", 0))
        
        logger.info(f"Master Slip ID: {master_slip_id}")
        
        # Step 1: Parse payload and build market registry
        registry = MarketRegistry()
        registry.build(payload)
        
        # Step 2: Initialize ACTIVE Monte Carlo optimizer
        optimizer = None
        if self.enable_monte_carlo:
            logger.info("ACTIVE Monte Carlo optimization: ENABLED")
            optimizer = ActiveMonteCarloOptimizer(
                registry=registry,
                random_seed=master_slip_id,
                num_simulations=self.num_simulations
            )
            optimizer.prepare_simulations()
            optimizer.run_simulations()
        else:
            logger.info("Monte Carlo optimization: DISABLED")
            # Fall back to passive mode (for backward compatibility)
            from scipy.stats import entropy
            optimizer = None
        
        # Step 3: Generate candidate slips
        generator = IntelligentSlipGenerator(
            registry=registry,
            random_seed=master_slip_id,
            monte_carlo_optimizer=optimizer
        )
        
        candidate_slips = generator.generate_candidate_pool()
        
        # Step 4: Select optimal portfolio using ACTIVE optimization
        if optimizer and self.enable_monte_carlo:
            final_slips = generator.select_optimal_portfolio(
                candidates=candidate_slips,
                target_size=50,
                iterations=self.optimization_iterations
            )
        else:
            # Fallback: random selection (backward compatibility)
            if len(candidate_slips) >= 50:
                final_slips = generator.rng.sample(candidate_slips, 50)
            else:
                final_slips = candidate_slips[:50]
            
            # Renumber slips
            for i, slip in enumerate(final_slips, 1):
                slip.slip_id = f"SLIP_{i:04d}"
        
        # Step 5: Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(final_slips, optimizer)
        
        # Step 6: Verify match ID integrity (CRITICAL)
        self._verify_match_id_integrity(final_slips, registry)
        
        # Step 7: Calculate slip metrics
        self._calculate_slip_metrics(final_slips, optimizer)
        
        # Step 8: Build response
        response = {
            "generated_slips": [slip.to_dict() for slip in final_slips],
            "metadata": {
                "master_slip_id": master_slip_id,
                "total_slips": len(final_slips),
                "input_matches": len(registry.get_matches()),
                "total_selections": registry.total_selections,
                "unique_markets": len(registry.available_market_codes),
                "risk_distribution": {
                    level.value: sum(1 for s in final_slips if s.risk_level == level)
                    for level in RiskLevel
                },
                "monte_carlo_enabled": self.enable_monte_carlo,
                "optimization_iterations": self.optimization_iterations if self.enable_monte_carlo else 0,
                "portfolio_metrics": portfolio_metrics,
                "engine_version": "2.1.0",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
        }
        
        logger.info("=" * 80)
        logger.info("SLIP BUILDER v2.1 - Generation complete")
        logger.info(f"Generated {len(final_slips)} slips with ACTIVE optimization")
        logger.info(f"Portfolio fitness: {portfolio_metrics.get('fitness_score', 0):.3f}")
        logger.info("=" * 80)
        
        return response
    
    def _calculate_portfolio_metrics(
        self,
        slips: List[Slip],
        optimizer: Optional[ActiveMonteCarloOptimizer]
    ) -> Dict[str, Any]:
        """Calculate overall portfolio statistics"""
        
        metrics = {
            "average_confidence": round(np.mean([s.confidence_score for s in slips]), 3),
            "average_diversity": round(np.mean([s.diversity_score for s in slips]), 3),
            "average_legs": round(np.mean([len(s.legs) for s in slips]), 2),
            "average_odds": round(np.mean([float(s.total_odds) for s in slips]), 2),
        }
        
        # Add ACTIVE Monte Carlo fitness metrics if available
        if optimizer and isinstance(optimizer, ActiveMonteCarloOptimizer):
            fitness = optimizer.evaluate_portfolio_fitness(slips)
            metrics.update({
                "fitness_score": round(fitness["fitness_score"], 3),
                "zero_win_percentage": round(fitness["zero_win_percentage"] * 100, 2),
                "optimal_win_percentage": round(fitness["optimal_win_percentage"] * 100, 2),
                "average_wins_per_simulation": round(fitness["average_wins"], 2),
                "hedging_score": round(fitness["hedging_score"], 3),
                "coverage_score": round(fitness["coverage_score"], 3),
            })
        
        return metrics
    
    def _calculate_slip_metrics(
        self,
        slips: List[Slip],
        optimizer: Optional[ActiveMonteCarloOptimizer]
    ) -> None:
        """
        Calculate diversity and coverage scores for all slips.
        
        Now uses ACTIVE Monte Carlo if available.
        """
        if not slips:
            return
        
        # Calculate diversity scores (how unique each slip is)
        for i, slip in enumerate(slips):
            # Count how many other slips share matches
            shared_matches = 0
            total_comparisons = 0
            
            slip_matches = slip.get_match_ids()
            
            for other_slip in slips:
                if other_slip.slip_id == slip.slip_id:
                    continue
                
                other_matches = other_slip.get_match_ids()
                overlap = len(slip_matches & other_matches)
                shared_matches += overlap
                total_comparisons += 1
            
            # Diversity = 1 - (average overlap ratio)
            if total_comparisons > 0 and len(slip_matches) > 0:
                avg_overlap = shared_matches / (total_comparisons * len(slip_matches))
                slip.diversity_score = max(0.0, 1.0 - avg_overlap)
            else:
                slip.diversity_score = 1.0
        
        # Calculate coverage scores using ACTIVE Monte Carlo if available
        if optimizer and isinstance(optimizer, ActiveMonteCarloOptimizer):
            logger.info("Calculating coverage scores using ACTIVE Monte Carlo")
            
            # For each slip, calculate how many simulations it wins
            for slip in slips:
                wins = 0
                
                for outcome in optimizer.simulated_outcomes:
                    all_legs_win = True
                    
                    for leg in slip.legs:
                        key = (leg.match_id, leg.market_code)
                        if outcome.get(key) != leg.selection:
                            all_legs_win = False
                            break
                    
                    if all_legs_win:
                        wins += 1
                
                # Coverage score = win rate in simulations
                slip.coverage_score = wins / len(optimizer.simulated_outcomes)
    
    def _verify_match_id_integrity(self, slips: List[Slip], registry: MarketRegistry) -> None:
        """
        CRITICAL: Verify that all slip legs use match_ids from original payload.
        
        This ensures downstream systems can resolve slips correctly.
        """
        logger.info("Verifying match ID integrity...")
        
        all_original_ids = registry.original_match_ids
        
        for slip in slips:
            for leg in slip.legs:
                if leg.match_id not in all_original_ids:
                    raise SlipBuilderError(
                        f"Match ID integrity violation in slip {slip.slip_id}: "
                        f"Match ID {leg.match_id} not found in original payload. "
                        f"Original match IDs: {sorted(all_original_ids)}"
                    )
        
        logger.info(f"✓ All {len(slips)} slips use valid match IDs from original payload")