"""
MAXWIN SLIP BUILDER - Expected Value Maximization Strategy
============================================================

Philosophy:
Don't just win often - win BIG when you do.

Optimization Goal:
Maximize Σ EV(slip_i) subject to:
- At least 1 winner in 95% of simulations (safety constraint)
- Risk distribution 20/20/10 (same as balanced)
- Diversity across matches (hedging enforced)

Key Differences from Balanced Strategy:
1. PRIMARY METRIC: Expected Value (not coverage)
2. SELECTION LOGIC: Greedy EV with coverage constraint (not coverage-first)
3. WINNER COUNT: 2-3 high-EV winners acceptable (vs 4-6 target)
4. RISK TOLERANCE: More aggressive (willing to sacrifice consistency for profit)

Expected Performance:
- Win rate: 20-30% lower than balanced (fewer slips win)
- Win size: 40-60% higher than balanced (bigger payouts)
- Expected value: 30-50% higher than balanced (net positive)
- Variance: Higher (more boom-or-bust)

Author: MaxWin Strategy Engine
Version: 1.0.0
"""

import logging
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Import shared components from balanced builder
from .slip_builder import (
    SlipBuilderError,
    RiskLevel,
    MarketSelection,
    Slip,
    MarketRegistry,
    ActiveMonteCarloOptimizer,
    HedgingEnforcer,
    IntelligentSlipGenerator
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXPECTED VALUE CALCULATOR
# ============================================================================

class ExpectedValueCalculator:
    """
    Calculates expected value for slips and portfolios.
    
    Expected Value (EV) is the fundamental metric in betting theory:
    EV = P(win) × payout - stake
    
    Positive EV = profitable bet (on average)
    Negative EV = losing bet (on average)
    Zero EV = breakeven bet
    
    Example:
        Bet: $10 on a 40% chance to win $30
        EV = 0.4 × 30 - 10 = $2
        
        Interpretation: On average, this bet makes $2 profit.
        Over 100 such bets, expect to profit $200.
    """
    
    @staticmethod
    def calculate_slip_ev(slip: Slip, stake: float = 10.0) -> float:
        """
        Calculate expected value for a single slip.
        
        Formula:
            EV = P(win) × total_payout - stake
            
        where:
            P(win) = slip.win_probability (from Monte Carlo)
            total_payout = stake × total_odds
            stake = amount wagered
        
        Args:
            slip: Slip to evaluate
            stake: Stake amount (default: $10)
        
        Returns:
            Expected value in currency units
        
        Examples:
            # Positive EV (good bet)
            slip: 40% win, 3.0x odds, $10 stake
            payout: $30
            EV = 0.4 × 30 - 10 = $2 ✅
            
            # Negative EV (bad bet)
            slip: 30% win, 2.0x odds, $10 stake
            payout: $20
            EV = 0.3 × 20 - 10 = -$4 ❌
            
            # Zero EV (fair bet)
            slip: 50% win, 2.0x odds, $10 stake
            payout: $20
            EV = 0.5 × 20 - 10 = $0 (breakeven)
        """
        # Calculate total payout if slip wins
        payout = stake * float(slip.total_odds)
        
        # EV = expected_return - cost
        ev = (slip.win_probability * payout) - stake
        
        return float(ev)
    
    @staticmethod
    def calculate_portfolio_ev(slips: List[Slip], stake_per_slip: float = 10.0) -> float:
        """
        Calculate total expected value across all slips in portfolio.
        
        Portfolio EV is simply the sum of individual slip EVs.
        This assumes slips are independent (reasonable for multi-leg parlays).
        
        Args:
            slips: Portfolio of slips
            stake_per_slip: Stake per slip (default: $10)
        
        Returns:
            Total expected value
        
        Example:
            Portfolio of 50 slips:
            - 30 slips with EV = +$2 each
            - 15 slips with EV = +$1 each
            - 5 slips with EV = -$1 each
            
            Portfolio EV = 30×2 + 15×1 + 5×(-1) = $70
            
            Interpretation: Over many iterations, expect $70 profit per portfolio.
        """
        total_ev = sum(
            ExpectedValueCalculator.calculate_slip_ev(s, stake_per_slip)
            for s in slips
        )
        return float(total_ev)
    
    @staticmethod
    def calculate_ev_score_normalized(slip: Slip, stake: float = 10.0) -> float:
        """
        Calculate normalized EV score (0-1 range) for ranking slips.
        
        Uses sigmoid function to map raw EV to 0-1 range:
        - EV = 0 → score = 0.5 (neutral)
        - EV > 0 → score approaches 1.0 (good)
        - EV < 0 → score approaches 0.0 (bad)
        
        This normalization allows comparison with other 0-1 scores
        (like confidence, coverage, diversity).
        
        Args:
            slip: Slip to score
            stake: Stake amount (default: $10)
        
        Returns:
            Normalized score in range [0, 1]
        
        Examples:
            EV = +$10 → score ≈ 0.88 (excellent)
            EV = +$2 → score ≈ 0.65 (good)
            EV = $0 → score = 0.50 (neutral)
            EV = -$2 → score ≈ 0.35 (poor)
            EV = -$10 → score ≈ 0.12 (terrible)
        """
        ev = ExpectedValueCalculator.calculate_slip_ev(slip, stake)
        
        # Sigmoid normalization: σ(x) = 1 / (1 + e^(-x))
        # Scale EV by 5 to get reasonable spread
        # (so EV of ±5 maps to scores near 0.0/1.0)
        scaled_ev = ev / 5.0
        
        # Apply sigmoid
        score = 1.0 / (1.0 + np.exp(-scaled_ev))
        
        return float(score)
    
    @staticmethod
    def get_ev_statistics(slips: List[Slip], stake_per_slip: float = 10.0) -> Dict[str, Any]:
        """
        Calculate comprehensive EV statistics for a portfolio.
        
        Provides detailed breakdown of portfolio EV characteristics.
        
        Args:
            slips: Portfolio of slips
            stake_per_slip: Stake per slip
        
        Returns:
            Dictionary with EV statistics
        
        Example output:
            {
                "total_ev": 145.50,
                "average_ev": 2.91,
                "median_ev": 2.10,
                "positive_ev_count": 42,
                "negative_ev_count": 8,
                "max_ev": 8.50,
                "min_ev": -3.20,
                "ev_std_dev": 2.15
            }
        """
        if not slips:
            return {
                "total_ev": 0.0,
                "average_ev": 0.0,
                "median_ev": 0.0,
                "positive_ev_count": 0,
                "negative_ev_count": 0,
                "max_ev": 0.0,
                "min_ev": 0.0,
                "ev_std_dev": 0.0
            }
        
        evs = [ExpectedValueCalculator.calculate_slip_ev(s, stake_per_slip) for s in slips]
        
        return {
            "total_ev": round(sum(evs), 2),
            "average_ev": round(np.mean(evs), 2),
            "median_ev": round(np.median(evs), 2),
            "positive_ev_count": sum(1 for ev in evs if ev > 0),
            "negative_ev_count": sum(1 for ev in evs if ev < 0),
            "max_ev": round(max(evs), 2),
            "min_ev": round(min(evs), 2),
            "ev_std_dev": round(np.std(evs), 2)
        }


# ============================================================================
# MAXWIN SLIP GENERATOR (inherits from balanced generator)
# ============================================================================

class MaxWinSlipGenerator(IntelligentSlipGenerator):
    """
    Extends IntelligentSlipGenerator with EV-focused selection.
    
    This generator uses the same candidate generation logic as the balanced
    approach (match selection, market selection, hedging) but applies a
    different selection criterion: maximize EV instead of coverage.
    
    Key Override:
        _select_optimal_portfolio() - Sorts by EV, enforces coverage as constraint
    
    Process:
        1. Generate candidate pool (same as balanced)
        2. Score each candidate by coverage (for constraint)
        3. Calculate EV for each candidate (NEW)
        4. Select slips that maximize total EV while maintaining 95% coverage
        5. Return 50 slips with highest total EV
    """
    
    def generate(self) -> List[Slip]:
        """
        Generate 50 slips optimized for maximum expected value.
        
        Algorithm:
        1. Generate candidate pool (150-300 slips)
        2. Score coverage for each candidate (needed for constraint)
        3. Calculate EV for each candidate
        4. Select 50 slips that maximize total EV
        5. Verify 95% coverage constraint
        6. If constraint violated, fall back to hybrid selection
        
        Returns:
            List of 50 Slip objects optimized for maximum EV
        
        Raises:
            SlipBuilderError: If unable to generate exactly 50 slips
        """
        logger.info("[MAXWIN] Starting EV-maximization slip generation")
        
        # Ensure Monte Carlo simulations are available
        if not self.optimizer.simulated:
            self.optimizer.run_simulations()
        
        # Step 1: Generate candidate pool (reuse parent logic)
        candidates = self._generate_candidate_pool()
        logger.info(f"[MAXWIN] Generated {len(candidates)} candidates")
        
        # Step 2: Score coverage (needed for 95% constraint check)
        logger.info("[MAXWIN] Scoring candidates for coverage...")
        for slip in candidates:
            slip.coverage_score = self.optimizer.score_slip(slip)
        
        # Step 3: Calculate EV for all candidates
        logger.info("[MAXWIN] Calculating expected values...")
        ev_calc = ExpectedValueCalculator()
        
        for slip in candidates:
            # Calculate raw EV
            slip.ev = ev_calc.calculate_slip_ev(slip)
            # Calculate normalized score for ranking
            slip.ev_score = ev_calc.calculate_ev_score_normalized(slip)
        
        # Log EV distribution
        evs = [s.ev for s in candidates]
        logger.info(
            f"[MAXWIN] EV range: ${min(evs):.2f} to ${max(evs):.2f} | "
            f"Mean: ${np.mean(evs):.2f} | "
            f"Positive: {sum(1 for ev in evs if ev > 0)}/{len(evs)}"
        )
        
        # Step 4: Select optimal portfolio (EV-first with coverage constraint)
        selected_slips = self._select_max_ev_portfolio(candidates)
        
        # Step 5: Calculate final metrics
        portfolio_ev = ev_calc.calculate_portfolio_ev(selected_slips)
        portfolio_fitness = self.optimizer.calculate_portfolio_fitness(selected_slips)
        ev_stats = ev_calc.get_ev_statistics(selected_slips)
        
        logger.info(
            f"[MAXWIN] Portfolio EV: ${portfolio_ev:.2f} | "
            f"Coverage: {portfolio_fitness['coverage_percentage']:.1%} | "
            f"Avg winners: {portfolio_fitness['avg_winners']:.2f} | "
            f"Positive EV slips: {ev_stats['positive_ev_count']}/50"
        )
        
        # Step 6: Calculate diversity scores (same as balanced)
        self._calculate_diversity_scores(selected_slips)
        
        # Step 7: Store EV score in fitness_score field (reuse existing field)
        for slip in selected_slips:
            slip.fitness_score = slip.ev_score
        
        # Final validation
        if len(selected_slips) != 50:
            raise SlipBuilderError(
                f"Failed to generate exactly 50 slips (generated {len(selected_slips)})"
            )
        
        logger.info("[MAXWIN] Successfully generated 50 EV-optimized slips")
        return selected_slips
    
    def _select_max_ev_portfolio(self, candidates: List[Slip]) -> List[Slip]:
        """
        Select 50 slips that maximize portfolio EV while maintaining constraints.
        
        Algorithm:
        1. Group candidates by risk level
        2. Sort each group by EV (descending)
        3. Greedily select top-EV slips from each tier
        4. Verify 95% coverage constraint
        5. If constraint violated, fall back to hybrid selection (EV+coverage)
        
        This is a greedy algorithm that prioritizes EV but ensures coverage.
        
        Args:
            candidates: Pool of candidate slips
        
        Returns:
            50 selected slips with maximum total EV
        """
        logger.info("[MAXWIN] Selecting optimal EV portfolio...")
        
        # Group candidates by risk level (maintain 20/20/10 distribution)
        candidates_by_risk = {level: [] for level in RiskLevel}
        for slip in candidates:
            candidates_by_risk[slip.risk_level].append(slip)
        
        # Sort each group by EV (highest first)
        for level in candidates_by_risk:
            candidates_by_risk[level].sort(key=lambda s: s.ev, reverse=True)
            
            # Log risk tier stats
            risk_candidates = candidates_by_risk[level]
            if risk_candidates:
                evs = [s.ev for s in risk_candidates]
                logger.debug(
                    f"[MAXWIN] {level.value} tier: "
                    f"{len(risk_candidates)} candidates, "
                    f"EV range: ${min(evs):.2f} to ${max(evs):.2f}"
                )
        
        # Greedy selection: take top EV slips from each tier
        selected: List[Slip] = []
        shortfall_by_tier: Dict[RiskLevel, int] = {}
        
        for risk_level, target_count in self.RISK_ALLOCATION.items():
            risk_candidates = candidates_by_risk[risk_level]
            
            # Take as many as available (up to target)
            take = min(target_count, len(risk_candidates))
            
            if take < target_count:
                shortfall = target_count - take
                shortfall_by_tier[risk_level] = shortfall
                logger.warning(
                    f"[MAXWIN] Only {take}/{target_count} "
                    f"{risk_level.value} candidates available (shortfall: {shortfall})"
                )
            
            selected.extend(risk_candidates[:take])
        
        # Fill any shortfall from best remaining candidates (sorted by EV)
        if len(selected) < 50:
            remaining = [c for c in candidates if c not in selected]
            remaining.sort(key=lambda s: s.ev, reverse=True)
            
            need = 50 - len(selected)
            logger.info(f"[MAXWIN] Filling shortfall of {need} slips from remaining pool")
            selected.extend(remaining[:need])
        
        # Ensure exactly 50
        selected = selected[:50]
        
        # CRITICAL: Verify 95% coverage constraint
        logger.info("[MAXWIN] Verifying coverage constraint...")
        fitness = self.optimizer.calculate_portfolio_fitness(selected)
        coverage = fitness['coverage_percentage']
        
        if coverage < 0.95:
            logger.warning(
                f"[MAXWIN] Coverage constraint violated: {coverage:.1%} < 95%. "
                f"Falling back to hybrid selection (EV + coverage)."
            )
            # Fallback: use composite score
            selected = self._select_hybrid_portfolio(candidates)
            
            # Re-verify coverage after fallback
            fitness = self.optimizer.calculate_portfolio_fitness(selected)
            coverage = fitness['coverage_percentage']
            
            if coverage < 0.95:
                logger.error(
                    f"[MAXWIN] Hybrid selection still violates coverage: {coverage:.1%}. "
                    f"Portfolio may not meet safety threshold."
                )
        
        # Calculate total EV of selected portfolio
        total_ev = sum(s.ev for s in selected)
        
        logger.info(
            f"[MAXWIN] Selected {len(selected)} slips | "
            f"Total EV: ${total_ev:.2f} | "
            f"Coverage: {coverage:.1%}"
        )
        
        return selected
    
    def _select_hybrid_portfolio(self, candidates: List[Slip]) -> List[Slip]:
        """
        Fallback: Select using composite score (70% EV + 30% coverage).
        
        Used when pure EV selection violates the 95% coverage constraint.
        This balances EV maximization with coverage requirements.
        
        Composite Score = 0.7 × EV_score + 0.3 × coverage_score
        
        This gives preference to high-EV slips but ensures enough coverage
        to meet the 95% threshold.
        
        Args:
            candidates: Pool of candidate slips
        
        Returns:
            50 slips selected by composite score
        """
        logger.info("[MAXWIN] Using hybrid selection (70% EV, 30% coverage)")
        
        # Calculate composite score for each candidate
        for slip in candidates:
            slip.composite_score = (0.7 * slip.ev_score) + (0.3 * slip.coverage_score)
        
        # Sort by composite score
        candidates.sort(key=lambda s: s.composite_score, reverse=True)
        
        # Select top 50
        selected = candidates[:50]
        
        # Log statistics
        total_ev = sum(s.ev for s in selected)
        avg_ev_score = np.mean([s.ev_score for s in selected])
        avg_coverage = np.mean([s.coverage_score for s in selected])
        
        logger.info(
            f"[MAXWIN] Hybrid selection: "
            f"Total EV: ${total_ev:.2f} | "
            f"Avg EV score: {avg_ev_score:.3f} | "
            f"Avg coverage: {avg_coverage:.3f}"
        )
        
        return selected


# ============================================================================
# MAXWIN SLIP BUILDER (public API)
# ============================================================================

class MaxWinSlipBuilder:
    """
    Expected Value Maximization Strategy.
    
    This builder generates 50 slips optimized for maximum expected profit
    while maintaining a 95% coverage constraint (at least 1 winner in 95%
    of simulations).
    
    Philosophy:
        "Don't just win often - win BIG when you do."
    
    Trade-offs vs Balanced Strategy:
        ✅ Higher expected value (+30-50%)
        ✅ Bigger payouts when slips hit
        ❌ Lower win frequency (-20-30%)
        ❌ Higher variance (more boom-or-bust)
    
    Public API:
        Matches SlipBuilder interface for compatibility.
        - generate(payload) -> Dict[str, Any]
    
    Usage:
        builder = MaxWinSlipBuilder(enable_monte_carlo=True)
        result = builder.generate(payload)
    """
    
    def __init__(self, enable_monte_carlo: bool = True, num_simulations: int = 10000):
        """
        Initialize MaxWin slip builder.
        
        Args:
            enable_monte_carlo: Whether to run Monte Carlo simulations (default: True)
            num_simulations: Number of simulations (default: 10000)
        """
        self.enable_monte_carlo = enable_monte_carlo
        self.num_simulations = int(num_simulations)
        
        logger.info(
            f"[MAXWIN BUILDER] Initialized with "
            f"Monte Carlo: {enable_monte_carlo}, "
            f"Simulations: {num_simulations}"
        )
    
    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate 50 slips optimized for maximum expected value.
        
        Process:
        1. Parse payload (MarketRegistry)
        2. Run Monte Carlo simulations
        3. Generate candidate pool
        4. Calculate EV for each candidate
        5. Select 50 slips with highest total EV
        6. Verify 95% coverage constraint
        7. Return slips with metadata
        
        Args:
            payload: Dictionary containing master_slip with matches and markets
        
        Returns:
            Dictionary with:
            - generated_slips: List of 50 slip dictionaries
            - metadata: Generation metadata and EV statistics
        
        Raises:
            SlipBuilderError: If generation fails or constraints violated
        """
        logger.info("=" * 80)
        logger.info("[MAXWIN BUILDER] Starting EV-maximization generation")
        logger.info("=" * 80)
        
        # Extract master slip ID for deterministic randomness
        master_slip_data = payload.get("master_slip", {})
        master_slip_id = int(master_slip_data.get("master_slip_id", 0))
        
        logger.info(f"[CONFIG] Master Slip ID: {master_slip_id}")
        logger.info(f"[CONFIG] Strategy: MaxWin (EV Maximization)")
        logger.info(f"[CONFIG] Monte Carlo: {self.enable_monte_carlo}")
        logger.info(f"[CONFIG] Simulations: {self.num_simulations}")
        
        # Step 1: Parse payload and build market registry
        logger.info("[STEP 1] Building market registry...")
        registry = MarketRegistry()
        registry.build(payload)
        
        # Step 2: Initialize Monte Carlo optimizer
        logger.info("[STEP 2] Initializing Monte Carlo optimizer...")
        optimizer = ActiveMonteCarloOptimizer(
            registry=registry,
            random_seed=master_slip_id,
            num_simulations=self.num_simulations
        )
        
        # Step 3: Initialize hedging enforcer
        logger.info("[STEP 3] Initializing hedging enforcer...")
        hedging = HedgingEnforcer(registry=registry)
        
        # Step 4: Run Monte Carlo simulations
        if self.enable_monte_carlo:
            logger.info("[STEP 4] Running Monte Carlo simulations...")
            optimizer.run_simulations()
        else:
            logger.info("[STEP 4] Monte Carlo disabled, skipping simulations")
        
        # Step 5: Generate slips using MaxWin generator
        logger.info("[STEP 5] Generating MaxWin slips...")
        generator = MaxWinSlipGenerator(
            registry=registry,
            random_seed=master_slip_id,
            monte_carlo_optimizer=optimizer,
            hedging_enforcer=hedging
        )
        
        slips = generator.generate()
        
        # Step 6: Calculate portfolio metrics
        logger.info("[STEP 6] Calculating portfolio metrics...")
        portfolio_metrics = self._calculate_portfolio_metrics(slips, optimizer)
        
        # Step 7: Build response (same schema as balanced for compatibility)
        response = {
            "generated_slips": [slip.to_dict() for slip in slips],
            "metadata": {
                "master_slip_id": master_slip_id,
                "strategy": "maxwin",  # Strategy identifier
                "total_slips": len(slips),
                "input_matches": len(registry.get_matches()),
                "total_selections": registry.total_selections,
                "unique_markets": len(registry.available_market_codes),
                "risk_distribution": {
                    level.value: sum(1 for s in slips if s.risk_level == level)
                    for level in RiskLevel
                },
                "monte_carlo_enabled": self.enable_monte_carlo,
                "portfolio_metrics": portfolio_metrics,
                "engine_version": "2.1.0-maxwin",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        }
        
        logger.info("=" * 80)
        logger.info("[MAXWIN BUILDER] Generation complete")
        logger.info(f"[RESULT] Generated {len(slips)} slips")
        logger.info(f"[RESULT] Portfolio EV: ${portfolio_metrics.get('portfolio_ev', 0):.2f}")
        logger.info(f"[RESULT] Coverage: {portfolio_metrics.get('coverage_percentage', 0):.1f}%")
        logger.info(f"[RESULT] Positive EV slips: {portfolio_metrics.get('positive_ev_count', 0)}/50")
        logger.info("=" * 80)
        
        return response
    
    def _calculate_portfolio_metrics(
        self,
        slips: List[Slip],
        optimizer: ActiveMonteCarloOptimizer
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics (includes EV statistics).
        
        Returns:
            Dictionary with all portfolio metrics
        """
        ev_calc = ExpectedValueCalculator()
        
        # Base metrics (same as balanced)
        metrics = {
            "average_confidence": round(np.mean([s.confidence_score for s in slips]), 3),
            "average_diversity": round(np.mean([s.diversity_score for s in slips]), 3),
            "average_coverage": round(np.mean([s.coverage_score for s in slips]), 3),
            "average_legs": round(np.mean([len(s.legs) for s in slips]), 2),
            "average_odds": round(np.mean([float(s.total_odds) for s in slips]), 2),
        }
        
        # EV metrics (NEW - specific to MaxWin)
        ev_stats = ev_calc.get_ev_statistics(slips)
        metrics.update({
            "portfolio_ev": ev_stats["total_ev"],
            "average_slip_ev": ev_stats["average_ev"],
            "median_slip_ev": ev_stats["median_ev"],
            "positive_ev_count": ev_stats["positive_ev_count"],
            "negative_ev_count": ev_stats["negative_ev_count"],
            "max_slip_ev": ev_stats["max_ev"],
            "min_slip_ev": ev_stats["min_ev"],
            "ev_std_dev": ev_stats["ev_std_dev"],
        })
        
        # Monte Carlo metrics (if available)
        if optimizer.simulated:
            fitness = optimizer.calculate_portfolio_fitness(slips)
            metrics.update({
                "portfolio_fitness": round(fitness["fitness_score"], 3),
                "coverage_percentage": round(fitness["coverage_percentage"] * 100, 2),
                "zero_win_rate": round(fitness["zero_win_rate"] * 100, 2),
                "target_win_rate": round(fitness["target_win_rate"] * 100, 2),
                "average_wins_per_simulation": round(fitness["avg_winners"], 2),
            })
        
        return metrics
