# slip_builder.py
"""
INTELLIGENT SLIP BUILDER v3.0 - MONTE CARLO OPTIMIZED (refactored from v2.1)
- Vectorized Monte Carlo storage and scoring
- Calibrated confidence scoring
- Redesigned fitness function (component-based)
- Stake optimization (Kelly fractional)
- Optimized diversity calculation
- Hedging with odds-weighting
- ZERO-breaking-changes to public interfaces
"""

import logging
import json
from decimal import Decimal, getcontext
from typing import Dict, List, Any, Optional, Set, Tuple, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import numpy as np
from collections import defaultdict, Counter
import math

# Set decimal precision for accurate odds calculations
getcontext().prec = 10

logger = logging.getLogger(__name__)

# DOMAIN MODELS (unchanged in shape, refined math)
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SlipBuilderError(Exception):
    pass


@dataclass(frozen=True)
class MarketSelection:
    match_id: int
    market_code: str
    selection: str
    odds: Decimal
    implied_probability: float = field(init=False)

    def __post_init__(self):
        # implied probability = 1/odds (Decimal -> float)
        # Odds are validated earlier; avoid ZeroDivisionError by guard
        ip = float(1 / self.odds) if self.odds != 0 else 0.0
        object.__setattr__(self, "implied_probability", ip)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "market": self.market_code,
            "selection": self.selection,
            "odds": float(self.odds),
        }


@dataclass
class Slip:
    slip_id: str
    risk_level: RiskLevel
    legs: List[MarketSelection]
    total_odds: Decimal = field(init=False)
    win_probability: float = field(init=False)
    confidence_score: float = field(default=0.0)
    coverage_score: float = field(default=0.0)
    diversity_score: float = field(default=0.0)
    fitness_score: float = field(default=0.0)

    def __post_init__(self):
        self.total_odds = Decimal("1")
        for leg in self.legs:
            self.total_odds *= leg.odds

        # win_probability: product of implied probabilities
        wp = 1.0
        for leg in self.legs:
            wp *= leg.implied_probability
        self.win_probability = wp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slip_id": self.slip_id,
            "risk_level": self.risk_level.value,
            "variation_type": self._get_variation_type(),
            "legs": [leg.to_dict() for leg in self.legs],
            "total_odds": float(self.total_odds),
            "confidence_score": round(self.confidence_score, 3),
            "coverage_score": round(self.coverage_score, 3),
            "diversity_score": round(self.diversity_score, 3),
            "fitness_score": round(self.fitness_score, 3),
        }

    def _get_variation_type(self) -> str:
        return {
            RiskLevel.LOW: "conservative",
            RiskLevel.MEDIUM: "balanced",
            RiskLevel.HIGH: "aggressive",
        }[self.risk_level]

    def get_signature(self) -> FrozenSet[Tuple[int, str, str]]:
        return frozenset((leg.match_id, leg.market_code, leg.selection) for leg in self.legs)

    # Optional explanation generator (NEW, non-breaking)
    def get_explanation(self) -> str:
        """
        Generate human-readable explanation for why this slip was selected.
        Optional helper added in v3.0; does not affect existing behavior.
        """
        reasons = []

        if self.confidence_score > 0.70:
            reasons.append(
                f"✓ High confidence ({self.confidence_score:.0%}) based on favorable odds and market analysis"
            )
        elif self.confidence_score < 0.40:
            reasons.append(
                f"⚠ Lower confidence ({self.confidence_score:.0%}) - high risk/reward play"
            )

        if self.diversity_score > 0.80:
            reasons.append(
                f"✓ Excellent diversity ({self.diversity_score:.0%}) - minimizes correlation risk"
            )

        if self.coverage_score > 0.10:
            reasons.append(
                f"✓ Strong simulation performance ({self.coverage_score:.1%} win rate in Monte Carlo)"
            )

        if self.fitness_score > 0.75:
            reasons.append(
                f"✓ Top-tier portfolio contributor (fitness: {self.fitness_score:.1%})"
            )

        if not reasons:
            reasons.append("Selected for portfolio balance")

        return " | ".join(reasons)


# MARKET REGISTRY (ENHANCED with robust parsing & detailed logging)
class MarketRegistry:
    def __init__(self):
        self.matches: Dict[int, Dict[str, List[MarketSelection]]] = {}
        self.match_metadata: Dict[int, Dict[str, Any]] = {}
        self.available_market_codes: Set[str] = set()
        self.total_selections: int = 0
        
        # NEW: Diagnostic counters
        self._stats = {
            'matches_processed': 0,
            'matches_with_markets': 0,
            'markets_found': 0,
            'markets_accepted': 0,
            'markets_rejected': 0,
            'selections_found': 0,
            'selections_accepted': 0,
            'selections_rejected': 0,
            'rejection_reasons': defaultdict(int)
        }

    def build(self, payload: Dict[str, Any]) -> None:
        """Build market registry with enhanced diagnostic logging"""
        logger.info("=" * 80)
        logger.info("[MARKET REGISTRY] Starting payload parsing")
        logger.info("=" * 80)
        
        # Reset stats for this build
        self._reset_stats()
        
        master = payload.get("master_slip", payload)
        matches = master.get("matches", [])
        
        logger.info(f"[REGISTRY] Payload structure:")
        logger.info(f"  - Has 'master_slip' key: {'master_slip' in payload}")
        logger.info(f"  - Matches found: {len(matches)}")
        
        if not isinstance(matches, list):
            logger.error(f"[REGISTRY] 'matches' is not a list: {type(matches)}")
            raise SlipBuilderError(f"Invalid matches type: {type(matches)}")
        
        if not matches:
            logger.error("[REGISTRY] Empty matches list")
            raise SlipBuilderError("No matches found in payload")
        
        # Process each match
        for match_idx, match_data in enumerate(matches):
            self._stats['matches_processed'] += 1
            
            if not isinstance(match_data, dict):
                logger.warning(f"[REGISTRY] Match {match_idx} is not a dict: {type(match_data)}")
                continue
                
            match_id = self._extract_match_id(match_data)
            if match_id is None:
                logger.warning(f"[REGISTRY] Match {match_idx} has no valid ID, skipping")
                continue
            
            logger.debug(f"[REGISTRY] Processing match {match_id} (index: {match_idx})")
            
            # Store match metadata
            self.match_metadata[match_id] = {
                "home_team": match_data.get("home_team", "Unknown"),
                "away_team": match_data.get("away_team", "Unknown"),
                "league": match_data.get("league", "Unknown"),
                "match_date": match_data.get("match_date", "Unknown"),
            }
            
            # Extract markets with flexible structure detection
            markets_data = self._extract_markets_flexible(match_data, match_id)
            
            if markets_data:
                market_map = self._parse_markets_flexible(match_id, markets_data)
                if market_map:
                    self.matches[match_id] = market_map
                    self._stats['matches_with_markets'] += 1
                    logger.debug(f"[REGISTRY] Match {match_id}: parsed {len(market_map)} markets")
                else:
                    logger.warning(f"[REGISTRY] Match {match_id}: no valid markets after parsing")
            else:
                logger.warning(f"[REGISTRY] Match {match_id}: no markets found in payload")
        
        # DIAGNOSTIC SUMMARY BEFORE FAILURE
        logger.info("=" * 80)
        logger.info("[REGISTRY] PARSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Matches processed: {self._stats['matches_processed']}")
        logger.info(f"Matches with markets: {self._stats['matches_with_markets']}")
        logger.info(f"Markets found: {self._stats['markets_found']}")
        logger.info(f"Markets accepted: {self._stats['markets_accepted']}")
        logger.info(f"Markets rejected: {self._stats['markets_rejected']}")
        logger.info(f"Selections found: {self._stats['selections_found']}")
        logger.info(f"Selections accepted: {self._stats['selections_accepted']}")
        logger.info(f"Selections rejected: {self._stats['selections_rejected']}")
        
        if self._stats['rejection_reasons']:
            logger.info("Rejection reasons:")
            for reason, count in self._stats['rejection_reasons'].items():
                logger.info(f"  - {reason}: {count}")
        
        # Sample first rejected market if any
        if hasattr(self, '_last_rejected_market'):
            logger.info(f"Sample rejected market: {json.dumps(self._last_rejected_market, default=str)[:500]}")
        
        logger.info("=" * 80)
        
        # FINAL VALIDATION
        if not self.matches:
            logger.error("[REGISTRY] CRITICAL: No valid markets found after parsing")
            logger.error(f"Matches processed: {self._stats['matches_processed']}")
            logger.error(f"Markets found: {self._stats['markets_found']}")
            logger.error(f"Markets accepted: {self._stats['markets_accepted']}")
            
            # Log the first match structure to help debug
            if matches and len(matches) > 0:
                logger.error(f"First match structure: {json.dumps(matches[0], default=str)[:1000]}")
            
            raise SlipBuilderError("No valid markets found after parsing payload")
        
        logger.info(f"[REGISTRY] SUCCESS: Built registry with {len(self.matches)} matches, "
                   f"{len(self.available_market_codes)} unique markets, "
                   f"{self.total_selections} total selections")
        logger.info("=" * 80)
    
    def _reset_stats(self):
        """Reset diagnostic counters"""
        self._stats = {
            'matches_processed': 0,
            'matches_with_markets': 0,
            'markets_found': 0,
            'markets_accepted': 0,
            'markets_rejected': 0,
            'selections_found': 0,
            'selections_accepted': 0,
            'selections_rejected': 0,
            'rejection_reasons': defaultdict(int)
        }
    
    def _extract_match_id(self, match_data: Dict[str, Any]) -> Optional[int]:
        """Extract match ID from various possible keys"""
        match_id = (
            match_data.get("match_id") or 
            match_data.get("id") or 
            match_data.get("fixture_id") or
            match_data.get("matchId")
        )
        
        if match_id is None:
            logger.debug(f"Match missing ID, keys: {list(match_data.keys())}")
            return None
        
        try:
            return int(match_id)
        except (ValueError, TypeError):
            logger.debug(f"Match has invalid ID: {match_id}")
            return None
    
    def _extract_markets_flexible(self, match_data: Dict[str, Any], match_id: int) -> List[Dict]:
        """
        Extract markets with flexible structure detection.
        Supports multiple possible key names and structures.
        """
        # Try different possible market container keys
        possible_keys = [
            "match_markets",
            "markets", 
            "full_markets",
            "market_data",
            "betting_markets"
        ]
        
        markets = []
        source_key = None
        
        for key in possible_keys:
            if key in match_data:
                candidate = match_data.get(key, [])
                if isinstance(candidate, list):
                    markets = candidate
                    source_key = key
                    logger.debug(f"Match {match_id}: found markets under '{key}' ({len(markets)} items)")
                    break
                elif isinstance(candidate, dict):
                    # Handle case where markets might be a dict
                    logger.debug(f"Match {match_id}: markets under '{key}' is dict, converting")
                    markets = [candidate]
                    source_key = key
                    break
        
        if not markets and source_key is None:
            logger.debug(f"Match {match_id}: no market container found, keys: {list(match_data.keys())}")
        
        self._stats['markets_found'] += len(markets)
        return markets
    
    def _parse_markets_flexible(self, match_id: int, markets_data: List[Dict]) -> Dict[str, List[MarketSelection]]:
        """
        Parse markets with support for multiple structures.
        Never silently skips - logs all rejections.
        """
        market_map: Dict[str, List[MarketSelection]] = {}
        
        for market_idx, market_entry in enumerate(markets_data):
            if not isinstance(market_entry, dict):
                self._stats['markets_rejected'] += 1
                reason = f"Market {market_idx} not a dict"
                self._stats['rejection_reasons'][reason] += 1
                logger.debug(f"Match {match_id}: {reason}")
                self._last_rejected_market = market_entry  # Store rejected market for debugging
                continue
            
            # EXTRACT MARKET CODE - Support multiple structures
            market_code = None
            market_info = None
            
            # Structure A: {"market": {"code": "MATCH_RESULT"}, "selections": [...]}
            if "market" in market_entry and isinstance(market_entry["market"], dict):
                market_info = market_entry["market"]
                market_code = (
                    market_info.get("code") or 
                    market_info.get("market_type") or 
                    market_info.get("type") or
                    market_info.get("name")
                )
                logger.debug(f"Match {match_id}: Structure A detected, code={market_code}")
            
            # Structure B: {"market_type": "MATCH_RESULT", "outcomes": [...]}
            elif "market_type" in market_entry:
                market_code = market_entry.get("market_type")
                market_info = {"code": market_code, "name": market_entry.get("name")}
                logger.debug(f"Match {match_id}: Structure B detected, code={market_code}")
            
            # Structure C: {"code": "MATCH_RESULT", "selections": [...]}
            elif "code" in market_entry:
                market_code = market_entry.get("code")
                market_info = {"code": market_code, "name": market_entry.get("name")}
                logger.debug(f"Match {match_id}: Structure C detected, code={market_code}")
            
            # Structure D: {"type": "MATCH_RESULT", "outcomes": [...]}
            elif "type" in market_entry:
                market_code = market_entry.get("type")
                market_info = {"code": market_code, "name": market_entry.get("name")}
                logger.debug(f"Match {match_id}: Structure D detected, code={market_code}")
            
            # Try to find any field that might contain market code
            else:
                for key in ["marketName", "bet_type", "market_id", "id"]:
                    if key in market_entry:
                        market_code = str(market_entry.get(key, ""))
                        if market_code:
                            logger.debug(f"Match {match_id}: Found potential market code under '{key}': {market_code}")
                            break
            
            if not market_code:
                self._stats['markets_rejected'] += 1
                reason = f"Market {market_idx} has no code"
                self._stats['rejection_reasons'][reason] += 1
                logger.warning(f"Match {match_id}: {reason}, entry keys: {list(market_entry.keys())}")
                self._last_rejected_market = market_entry  # Store rejected market for debugging
                continue
            
            market_code = str(market_code).strip().upper()
            
            # EXTRACT SELECTIONS - Support multiple structures
            selections_data = None
            
            # Try to find selections/outcomes
            if "selections" in market_entry:
                selections_data = market_entry.get("selections", [])
                logger.debug(f"Match {match_id}: Found selections under 'selections' key")
            elif "outcomes" in market_entry:
                selections_data = market_entry.get("outcomes", [])
                logger.debug(f"Match {match_id}: Found selections under 'outcomes' key")
            elif "options" in market_entry:
                selections_data = market_entry.get("options", [])
                logger.debug(f"Match {match_id}: Found selections under 'options' key")
            elif "bets" in market_entry:
                selections_data = market_entry.get("bets", [])
                logger.debug(f"Match {match_id}: Found selections under 'bets' key")
            
            if selections_data is None and market_info and "selections" in market_info:
                selections_data = market_info.get("selections", [])
                logger.debug(f"Match {match_id}: Found selections nested in market object")
            
            if not isinstance(selections_data, list):
                self._stats['markets_rejected'] += 1
                reason = f"Market {market_idx} selections not a list"
                self._stats['rejection_reasons'][reason] += 1
                logger.warning(f"Match {match_id}: {reason}, got {type(selections_data)}")
                self._last_rejected_market = market_entry  # Store rejected market for debugging
                continue
            
            # PARSE SELECTIONS
            selections = self._parse_selections_flexible(match_id, market_code, selections_data)
            
            if selections:
                market_map[market_code] = selections
                self.available_market_codes.add(market_code)
                self.total_selections += len(selections)
                self._stats['markets_accepted'] += 1
                logger.debug(f"Match {match_id}, market {market_code}: accepted {len(selections)} selections")
            else:
                self._stats['markets_rejected'] += 1
                reason = f"Market {market_idx} has no valid selections"
                self._stats['rejection_reasons'][reason] += 1
                logger.debug(f"Match {match_id}, market {market_code}: no valid selections")
                self._last_rejected_market = market_entry  # Store rejected market for debugging
        
        return market_map
    
    def _parse_selections_flexible(self, match_id: int, market_code: str, selections_data: List[Dict]) -> List[MarketSelection]:
        """
        Parse selections with flexible structure support.
        Never silently skips - logs all rejections with reasons.
        """
        valid_selections = []
        
        for sel_idx, selection_data in enumerate(selections_data):
            self._stats['selections_found'] += 1
            
            if not isinstance(selection_data, dict):
                self._stats['selections_rejected'] += 1
                reason = f"Selection {sel_idx} not a dict"
                self._stats['rejection_reasons'][reason] += 1
                logger.debug(f"Match {match_id}, market {market_code}: {reason}")
                continue
            
            # EXTRACT SELECTION VALUE - Support multiple keys
            value = None
            possible_value_keys = ["value", "name", "label", "selection", "outcome", "bet_name", "title"]
            
            for key in possible_value_keys:
                if key in selection_data:
                    candidate = selection_data.get(key)
                    if candidate is not None and str(candidate).strip():
                        value = str(candidate).strip()
                        logger.debug(f"Match {match_id}: found value under '{key}': {value}")
                        break
            
            if value is None:
                self._stats['selections_rejected'] += 1
                reason = f"Selection {sel_idx} missing value field"
                self._stats['rejection_reasons'][reason] += 1
                logger.debug(f"Match {match_id}, market {market_code}: {reason}, keys: {list(selection_data.keys())}")
                continue
            
            # EXTRACT ODDS - Support multiple keys with type coercion
            odds = None
            possible_odds_keys = ["odds", "price", "decimal_odds", "value", "odd", "bet_odds"]
            
            for key in possible_odds_keys:
                if key in selection_data:
                    odds_raw = selection_data.get(key)
                    try:
                        # Handle both string and numeric odds
                        odds = Decimal(str(odds_raw))
                        logger.debug(f"Match {match_id}: found odds under '{key}': {odds}")
                        break
                    except (TypeError, ValueError, ArithmeticError) as e:
                        logger.debug(f"Match {match_id}: odds under '{key}' invalid: {odds_raw}, error: {e}")
                        continue
            
            if odds is None:
                self._stats['selections_rejected'] += 1
                reason = f"Selection {sel_idx} missing or invalid odds"
                self._stats['rejection_reasons'][reason] += 1
                logger.debug(f"Match {match_id}, market {market_code}: {reason}")
                continue
            
            # VALIDATE ODDS
            try:
                if odds < Decimal("1.01"):
                    self._stats['selections_rejected'] += 1
                    reason = f"Selection {sel_idx} odds too low: {odds}"
                    self._stats['rejection_reasons'][reason] += 1
                    logger.debug(f"Match {match_id}: {reason}")
                    continue
                    
                if odds > Decimal("1000"):
                    self._stats['selections_rejected'] += 1
                    reason = f"Selection {sel_idx} odds too high: {odds}"
                    self._stats['rejection_reasons'][reason] += 1
                    logger.debug(f"Match {match_id}: {reason}")
                    continue
                    
            except (TypeError, ArithmeticError):
                self._stats['selections_rejected'] += 1
                reason = f"Selection {sel_idx} odds comparison failed"
                self._stats['rejection_reasons'][reason] += 1
                continue
            
            # CREATE SELECTION
            try:
                selection = MarketSelection(
                    match_id=match_id,
                    market_code=market_code,
                    selection=value,
                    odds=odds
                )
                valid_selections.append(selection)
                self._stats['selections_accepted'] += 1
                
            except Exception as e:
                self._stats['selections_rejected'] += 1
                reason = f"Selection creation failed: {str(e)}"
                self._stats['rejection_reasons'][reason] += 1
                logger.debug(f"Match {match_id}: {reason}")
                continue
        
        return valid_selections
    
    # Original methods (unchanged)
    def get_matches(self) -> List[int]:
        return list(self.matches.keys())
    
    def get_markets_for_match(self, match_id: int) -> List[str]:
        return list(self.matches.get(match_id, {}).keys())
    
    def get_selections(self, match_id: int, market_code: str) -> List[MarketSelection]:
        return self.matches.get(match_id, {}).get(market_code, [])
    
    def get_match_info(self, match_id: int) -> Dict[str, Any]:
        return self.match_metadata.get(match_id, {})


# ACTIVE MONTE CARLO OPTIMIZER (vectorized & memory-efficient)
class ActiveMonteCarloOptimizer:
    def __init__(self, registry: MarketRegistry, random_seed: int, num_simulations: int = 10000):
        self.registry = registry
        self.requested_simulations = int(num_simulations)
        # Cap simulations to avoid memory blowup; log decision
        self.num_simulations = min(self.requested_simulations, 5000)
        if self.requested_simulations > self.num_simulations:
            logger.warning(f"[MONTE CARLO] Requested {self.requested_simulations} simulations, capped to {self.num_simulations} for performance")
        self.rng = np.random.RandomState(random_seed)
        # Storage: for each market (match_id, market_code) we store:
        # - selections list (list of selection strings)
        # - selected_indices: np.ndarray shape (num_simulations,) of ints
        self.market_selections: Dict[Tuple[int, str], List[str]] = {}
        self.market_selection_index: Dict[Tuple[int, str], Dict[str, int]] = {}
        self.market_sim_selected_indices: Dict[Tuple[int, str], np.ndarray] = {}
        self.simulated = False

    def run_simulations(self) -> None:
        logger.info(f"[MONTE CARLO] Running {self.num_simulations} simulations (vectorized)")
        match_markets = []
        for match_id in self.registry.get_matches():
            for market_code in self.registry.get_markets_for_match(match_id):
                match_markets.append((match_id, market_code))

        # Precompute selection lists and probability arrays
        for match_id, market_code in match_markets:
            selections = self.registry.get_selections(match_id, market_code)
            if not selections:
                continue
            sel_labels = [s.selection for s in selections]
            probs = np.array([s.implied_probability for s in selections], dtype=float)
            prob_sum = probs.sum()
            if prob_sum <= 0 or np.isnan(prob_sum):
                # Fallback to uniform if implied probs are degenerate
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / prob_sum
            # Use indices to represent selections to minimize Python object overhead
            indices = self.rng.choice(len(selections), size=self.num_simulations, p=probs)
            key = (match_id, market_code)
            self.market_selections[key] = sel_labels
            # Create mapping selection->index
            sel_index_map = {lab: idx for idx, lab in enumerate(sel_labels)}
            self.market_selection_index[key] = sel_index_map
            self.market_sim_selected_indices[key] = indices

        self.simulated = True
        logger.info(f"[MONTE CARLO] Completed vectorized simulations for {len(self.market_sim_selected_indices)} markets")

    def score_slip(self, slip: Slip) -> float:
        if not self.simulated:
            return 0.0
        # Fast vectorized check: start with all True array and AND across legs
        num_sims = self.num_simulations
        wins = np.ones(num_sims, dtype=bool)
        for leg in slip.legs:
            key = (leg.match_id, leg.market_code)
            indices_array = self.market_sim_selected_indices.get(key)
            if indices_array is None:
                # No simulation data for this market -> cannot win in sims
                return 0.0
            sel_idx_map = self.market_selection_index.get(key)
            desired_idx = sel_idx_map.get(leg.selection)
            if desired_idx is None:
                # Selection not present in simulated market (can't match)
                return 0.0
            wins &= (indices_array == desired_idx)
            # Early exit if no wins remain
            if not wins.any():
                return 0.0
        return float(wins.sum() / num_sims)

    def get_slip_win_matrix(self, slips: List[Slip]) -> np.ndarray:
        """
        Returns boolean matrix shape (n_slips, num_simulations) where each row indicates
        which simulations that slip wins. Vectorized implementation.
        """
        if not self.simulated:
            self.run_simulations()
        n = len(slips)
        S = self.num_simulations
        matrix = np.zeros((n, S), dtype=bool)
        for i, slip in enumerate(slips):
            wins = np.ones(S, dtype=bool)
            for leg in slip.legs:
                key = (leg.match_id, leg.market_code)
                indices_array = self.market_sim_selected_indices.get(key)
                if indices_array is None:
                    wins = np.zeros(S, dtype=bool)
                    break
                sel_idx_map = self.market_selection_index.get(key)
                desired_idx = sel_idx_map.get(leg.selection)
                if desired_idx is None:
                    wins = np.zeros(S, dtype=bool)
                    break
                wins &= (indices_array == desired_idx)
                if not wins.any():
                    break
            matrix[i] = wins
        return matrix

    def calculate_portfolio_fitness(self, slips: List[Slip]) -> Dict[str, Any]:
        """
        Redesigned fitness with transparent, positive components.
        BACKWARD COMPATIBLE: All original dict keys preserved.
        NEW: Additional 'component_scores' key for transparency.
        """
        if not self.simulated:
            self.run_simulations()

        if not slips:
            return {
                "fitness_score": 0.0,
                "zero_win_rate": 1.0,
                "target_win_rate": 0.0,
                "avg_winners": 0.0,
                "win_distribution": Counter(),
                "coverage_percentage": 0.0,
                "component_scores": {},
            }

        # Build win matrix (existing logic)
        win_matrix = self.get_slip_win_matrix(slips)
        wins_per_sim = win_matrix.sum(axis=0)

        # Calculate metrics (existing logic)
        zero_wins = np.sum(wins_per_sim == 0)
        target_wins = np.sum((wins_per_sim >= 4) & (wins_per_sim <= 6))
        avg_winners = float(wins_per_sim.mean())
        zero_win_rate = float(zero_wins / wins_per_sim.size)
        target_win_rate = float(target_wins / wins_per_sim.size)

        # NEW: Component-based fitness (replaces old ad-hoc formula)
        # Component 1: Coverage (30% weight) - penalize zero-win scenarios
        coverage = 1.0 - zero_win_rate
        coverage_component = 0.30 * min(1.0, coverage / 0.95)  # Target 95% coverage

        # Component 2: Target wins (40% weight) - reward 4-6 winners
        target_component = 0.40 * target_win_rate

        # Component 3: Average winners (20% weight) - peak at 5
        avg_distance = abs(avg_winners - 5.0)
        avg_component = 0.20 * max(0.0, 1.0 - (avg_distance / 5.0))

        # Component 4: Expected value (10% weight) - NEW
        total_confidence = sum(s.confidence_score for s in slips)
        ev_component = 0.10 * min(1.0, total_confidence / 25.0)  # Target ~0.5 avg confidence

        # Final fitness (always positive, bounded [0, 1])
        fitness = coverage_component + target_component + avg_component + ev_component
        fitness = max(0.0, min(1.0, fitness))

        # Build distribution (existing)
        distribution = Counter(int(w) for w in wins_per_sim.tolist())

        return {
            "fitness_score": fitness,
            "zero_win_rate": zero_win_rate,
            "target_win_rate": target_win_rate,
            "avg_winners": avg_winners,
            "win_distribution": distribution,
            "coverage_percentage": 1.0 - zero_win_rate,
            "component_scores": {
                "coverage": coverage_component,
                "target_wins": target_component,
                "avg_winners": avg_component,
                "expected_value": ev_component,
            },
        }


# HEDGING ENFORCER (improved logic)
class HedgingEnforcer:
    def __init__(self, registry: MarketRegistry):
        self.registry = registry
        self.outcome_usage: Counter = Counter()

    def record_slip(self, slip: Slip) -> None:
        for leg in slip.legs:
            key = (leg.match_id, leg.market_code, leg.selection)
            self.outcome_usage[key] += 1

    def get_underrepresented_outcomes(self, match_id: int, market_code: str, current_count: int = 10) -> List[MarketSelection]:
        """
        Get underrepresented outcomes with odds-weighted targeting.

        IMPROVED: Uses implied probability instead of uniform distribution.
        UNCHANGED: Method signature and return type.
        """
        selections = self.registry.get_selections(match_id, market_code)
        if not selections:
            return []

        # Calculate total implied probability
        total_implied_prob = sum((1.0 / float(sel.odds)) if float(sel.odds) != 0 else 0.0 for sel in selections)

        if total_implied_prob <= 0:
            # Fallback to old uniform logic
            expected_per_outcome = max(1.0, float(current_count) / max(1.0, len(selections)))
            underrep = []
            for sel in selections:
                key = (match_id, market_code, sel.selection)
                actual = self.outcome_usage.get(key, 0)
                if actual + 1 < expected_per_outcome:
                    underrep.append(sel)
            return underrep if underrep else selections

        # NEW: Odds-weighted expected frequency
        underrep_scores = []
        for sel in selections:
            key = (match_id, market_code, sel.selection)
            actual = self.outcome_usage.get(key, 0)

            # Expected frequency based on fair odds
            implied_prob = (1.0 / float(sel.odds)) if float(sel.odds) != 0 else 0.0
            expected_freq = (implied_prob / total_implied_prob) * current_count

            # Underrepresentation score
            underrep_score = max(0.0, expected_freq - actual)
            underrep_scores.append((sel, underrep_score))

        # Return selections with above-average underrepresentation
        if not underrep_scores:
            return selections

        avg_score = np.mean([score for _, score in underrep_scores])
        underrep = [sel for sel, score in underrep_scores if score >= avg_score]

        return underrep if underrep else selections

    def get_hedging_bias(self, match_id: int, market_code: str) -> Optional[str]:
        selections = self.registry.get_selections(match_id, market_code)
        if not selections:
            return None
        min_count = float("inf")
        min_selection = None
        for sel in selections:
            key = (match_id, market_code, sel.selection)
            cnt = self.outcome_usage.get(key, 0)
            if cnt < min_count:
                min_count = cnt
                min_selection = sel.selection
        return min_selection


# INTELLIGENT SLIP GENERATOR (refinements & safeguards)
class IntelligentSlipGenerator:
    # Requirement #1: Update Risk Allocation to target 50 slips with medium focus
    RISK_ALLOCATION = {RiskLevel.LOW: 15, RiskLevel.MEDIUM: 25, RiskLevel.HIGH: 10}
    LEGS_PER_RISK = {RiskLevel.LOW: (2, 3), RiskLevel.MEDIUM: (3, 4), RiskLevel.HIGH: (4, 5)}
    ODDS_PREFERENCE = {RiskLevel.LOW: "low", RiskLevel.MEDIUM: "mixed", RiskLevel.HIGH: "high"}
    CANDIDATE_POOL_MULTIPLIER = 3
    MIN_FITNESS_THRESHOLD = 0.6

    # NEW: Calibrated confidence & market difficulty (v3.0)
    CALIBRATED_BASE_CONFIDENCE = {
        RiskLevel.LOW: 0.72,
        RiskLevel.MEDIUM: 0.51,
        RiskLevel.HIGH: 0.31,
    }

    MARKET_DIFFICULTY = {
        "MATCH_RESULT": 0.85,
        "BTTS": 0.78,
        "OVER_UNDER": 0.82,
        "CORRECT_SCORE": 0.45,
        "ASIAN_HANDICAP": 0.68,
    }

    def __init__(self, registry: MarketRegistry, random_seed: int, monte_carlo_optimizer: ActiveMonteCarloOptimizer, hedging_enforcer: HedgingEnforcer):
        self.registry = registry
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.RandomState(random_seed)
        self.optimizer = monte_carlo_optimizer
        self.hedging = hedging_enforcer
        self.used_signatures: Set[FrozenSet] = set()

    def generate(self) -> List[Slip]:
        logger.info("[SLIP GEN] Starting ACTIVE Monte Carlo slip generation")
        if not self.optimizer.simulated:
            self.optimizer.run_simulations()

        candidates = self._generate_candidate_pool()
        logger.info(f"[SLIP GEN] Generated {len(candidates)} candidates")

        # Score candidates (vectorized scoring is handled inside score_slip)
        for slip in candidates:
            slip.coverage_score = self.optimizer.score_slip(slip)

        selected_slips = self._select_optimal_portfolio(candidates)
        portfolio_fitness = self.optimizer.calculate_portfolio_fitness(selected_slips)
        logger.info(
            f"[SLIP GEN] Portfolio fitness: {portfolio_fitness['fitness_score']:.3f} | "
            f"Zero-win rate: {portfolio_fitness['zero_win_rate']:.1%} | "
            f"Target (4-6) rate: {portfolio_fitness['target_win_rate']:.1%} | "
            f"Avg winners: {portfolio_fitness['avg_winners']:.2f}"
        )

        self._calculate_diversity_scores(selected_slips)
        for slip in selected_slips:
            slip.fitness_score = portfolio_fitness["fitness_score"]

        # Requirement #3: Update validation to expect 50 slips (was 40)
        if len(selected_slips) != 50:
            raise SlipBuilderError(f"Failed to generate exactly 50 slips (generated {len(selected_slips)})")

        logger.info("[SLIP GEN] Successfully generated 50 optimized slips")
        return selected_slips

    def _generate_candidate_pool(self) -> List[Slip]:
        candidates: List[Slip] = []
        slip_counter = 1
        # Use local references for speed
        available_matches = self.registry.get_matches()
        max_attempts_per_tier = 1000

        for risk_level, target_count in self.RISK_ALLOCATION.items():
            candidate_count = target_count * self.CANDIDATE_POOL_MULTIPLIER
            generated = 0
            attempts = 0
            while generated < candidate_count and attempts < max_attempts_per_tier:
                slip = self._generate_single_slip(risk_level, slip_counter, use_hedging=True)
                attempts += 1
                if slip is None:
                    continue
                signature = slip.get_signature()
                if signature in self.used_signatures:
                    continue
                candidates.append(slip)
                self.used_signatures.add(signature)
                self.hedging.record_slip(slip)
                slip_counter += 1
                generated += 1
            if generated < candidate_count:
                logger.warning(f"[SLIP GEN] Could only generate {generated}/{candidate_count} candidates for {risk_level.value} after {attempts} attempts")

        # Deduplicate safety (should be unique by used_signatures)
        logger.info(f"[SLIP GEN] Candidate pool assembled: {len(candidates)} slips")
        return candidates

    def _generate_single_slip(self, risk_level: RiskLevel, slip_id: int, use_hedging: bool = True) -> Optional[Slip]:
        min_legs, max_legs = self.LEGS_PER_RISK[risk_level]
        num_legs = self.rng.randint(min_legs, max_legs)
        available_matches = self.registry.get_matches()
        if len(available_matches) < num_legs:
            num_legs = len(available_matches)
        if num_legs < 2:
            return None

        selected_matches = self.rng.sample(available_matches, num_legs)
        legs: List[MarketSelection] = []
        used_markets: Set[Tuple[int, str]] = set()

        for match_id in selected_matches:
            available_markets = self.registry.get_markets_for_match(match_id)
            if not available_markets:
                continue
            self.rng.shuffle(available_markets)
            selected_market = next((m for m in available_markets if (match_id, m) not in used_markets), available_markets[0])
            all_selections = self.registry.get_selections(match_id, selected_market)
            if not all_selections:
                continue

            if use_hedging:
                selections = self.hedging.get_underrepresented_outcomes(match_id, selected_market, current_count=len(self.used_signatures))
                if not selections:
                    selections = all_selections
            else:
                selections = all_selections

            selection = self._select_outcome(selections, risk_level)
            if selection:
                legs.append(selection)
                used_markets.add((match_id, selected_market))

        if len(legs) < 2:
            return None

        slip = Slip(slip_id=f"SLIP_{slip_id:04d}", risk_level=risk_level, legs=legs)
        slip.confidence_score = self._calculate_confidence(slip)
        return slip

    def _select_outcome(self, selections: List[MarketSelection], risk_level: RiskLevel) -> Optional[MarketSelection]:
        if not selections:
            return None
        preference = self.ODDS_PREFERENCE[risk_level]
        if preference == "low":
            sorted_selections = sorted(selections, key=lambda s: float(s.odds))
            top_half = sorted_selections[:max(1, len(sorted_selections) // 2)]
            return self.rng.choice(top_half)
        elif preference == "high":
            sorted_selections = sorted(selections, key=lambda s: float(s.odds), reverse=True)
            top_half = sorted_selections[:max(1, len(sorted_selections) // 2)]
            return self.rng.choice(top_half)
        else:
            return self.rng.choice(selections)

    def _calculate_confidence(self, slip: Slip) -> float:
        """
        Calibrated confidence scoring (v3.0)
        Signature unchanged.
        """
        # Base from calibrated table
        base = self.CALIBRATED_BASE_CONFIDENCE[slip.risk_level]

        # Exponential leg penalty (more severe as legs increase)
        # For 0 extra legs penalty -> 0. For larger legs penalty approaches 1.
        leg_penalty = 1.0 - (0.92 ** (max(0, len(slip.legs) - 2)))

        # Market difficulty factor (mean across leg markets)
        market_scores = [
            self.MARKET_DIFFICULTY.get(leg.market_code, 0.70)
            for leg in slip.legs
        ]
        market_factor = float(np.mean(market_scores)) if market_scores else 0.7

        # Odds realism check (geometric mean)
        leg_odds = np.array([float(leg.odds) for leg in slip.legs], dtype=float)
        geometric_mean = float(np.exp(np.mean(np.log(leg_odds)))) if len(leg_odds) > 0 else 1.0
        odds_factor = 1.0 + math.tanh((2.0 - geometric_mean) * 0.12)  # bounded adjustment

        confidence = base * (1.0 - leg_penalty) * market_factor * (odds_factor / 2.0)
        return max(0.1, min(0.95, float(confidence)))

    def _select_optimal_portfolio(self, candidates: List[Slip]) -> List[Slip]:
        logger.info("[PORTFOLIO] Selecting optimal 50-slip portfolio (greedy + fill)")

        candidates_by_risk = {level: [] for level in RiskLevel}
        for slip in candidates:
            candidates_by_risk[slip.risk_level].append(slip)

        for level in candidates_by_risk:
            candidates_by_risk[level].sort(key=lambda s: s.coverage_score, reverse=True)

        selected: List[Slip] = []
        shortfall = 0
        for risk_level, target_count in self.RISK_ALLOCATION.items():
            risk_candidates = candidates_by_risk[risk_level]
            take = min(target_count, len(risk_candidates))
            if take < target_count:
                logger.warning(f"[PORTFOLIO] Only {take}/{target_count} {risk_level.value} candidates available")
            selected.extend(risk_candidates[:take])
            if take < target_count:
                shortfall += (target_count - take)

        if len(selected) < 50:
            remaining = [c for c in candidates if c not in selected]
            remaining.sort(key=lambda s: s.coverage_score, reverse=True)
            need = 50 - len(selected)
            selected.extend(remaining[:need])

        selected = selected[:50]
        if len(selected) != 50:
            logger.error(f"[PORTFOLIO] Selection failed to reach 50 slips, got {len(selected)}")
        return selected

    def _calculate_diversity_scores(self, slips: List[Slip]) -> None:
        """
        Optimized diversity calculation (v3.0)
        Method signature unchanged, returns same scale [0.0, 1.0].
        """
        if not slips:
            return

        n = len(slips)
        # Use match-based sets to reduce complexity
        match_sets = [set(leg.match_id for leg in slip.legs) for slip in slips]
        diversity_scores = np.zeros(n, dtype=float)

        # Compute similarities per slip using aggregated counts (still O(n^2) in worst-case,
        # but with lighter operations and early exits; acceptable and preserves correctness).
        for i in range(n):
            si = match_sets[i]
            if not si:
                diversity_scores[i] = 1.0
                continue

            similarities = []
            for j in range(n):
                if i == j:
                    continue
                sj = match_sets[j]
                inter = len(si & sj)
                union = len(si | sj)
                sim = (inter / union) if union > 0 else 0.0
                similarities.append(sim)

            avg_similarity = float(np.mean(similarities)) if similarities else 0.0
            diversity_scores[i] = max(0.0, 1.0 - avg_similarity)

        for i, slip in enumerate(slips):
            slip.diversity_score = round(float(diversity_scores[i]), 4)


# NEW CLASS: StakeOptimizer (added after IntelligentSlipGenerator, before SlipBuilder)
class StakeOptimizer:
    """
    Kelly Criterion-based stake optimizer.

    NEW in v3.0 - Provides optimal stake recommendations.
    """

    def __init__(self, fractional_kelly: float = 0.25):
        """
        Args:
            fractional_kelly: Fraction of Kelly bet (0.25 = quarter Kelly, safer)
        """
        self.fractional_kelly = fractional_kelly

    def calculate_stakes(
        self,
        slips: List[Slip],
        bankroll: float = 1000.0
    ) -> Dict[str, float]:
        """
        Calculate optimal stake per slip using Kelly Criterion.

        Args:
            slips: List of generated slips
            bankroll: Total available bankroll

        Returns:
            Dict mapping slip_id -> recommended_stake
        """
        stakes = {}
        total_stake = 0.0

        for slip in slips:
            # Calculate edge: E[return] - 1
            expected_return = slip.confidence_score * float(slip.total_odds)
            edge = expected_return - 1.0

            if edge > 0.02:  # Minimum 2% edge required
                # Kelly formula: edge / (odds - 1)
                denom = float(slip.total_odds) - 1.0
                if denom <= 0:
                    stakes[slip.slip_id] = 0.0
                    continue
                kelly_fraction = edge / denom

                # Apply fractional Kelly (safer)
                safe_fraction = kelly_fraction * self.fractional_kelly

                # Cap at 5% of bankroll per slip
                safe_fraction = min(safe_fraction, 0.05)

                stake = bankroll * safe_fraction
                stakes[slip.slip_id] = round(stake, 2)
                total_stake += stake
            else:
                stakes[slip.slip_id] = 0.0  # No edge, don't bet

        # Normalize if total exceeds bankroll
        if total_stake > bankroll and total_stake > 0:
            scale = bankroll / total_stake
            stakes = {k: round(v * scale, 2) for k, v in stakes.items()}

        return stakes


# PUBLIC API (SlipBuilder)
class SlipBuilder:
    def __init__(self, enable_monte_carlo: bool = True, num_simulations: int = 10000):
        self.enable_monte_carlo = enable_monte_carlo
        # Keep requested count but optimizer will cap for performance
        self.num_simulations = int(num_simulations)
        # NEW internal stake optimizer (does not change public signature)
        self.stake_optimizer = StakeOptimizer()

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("[SLIP BUILDER v3.0] Starting ACTIVE Monte Carlo generation")
        logger.info("=" * 80)

        master_slip_data = payload.get("master_slip", {})
        master_slip_id = int(master_slip_data.get("master_slip_id", 0))

        registry = MarketRegistry()
        
        # TEMPORARY DEBUG BLOCK - REMOVE AFTER FIXING
        logger.info("=== PAYLOAD DEBUG ===")
        logger.info(f"Payload keys: {list(payload.keys())}")
        logger.info(f"Has 'master_slip': {'master_slip' in payload}")
        if 'master_slip' in payload:
            logger.info(f"Master slip keys: {list(payload['master_slip'].keys())}")
            logger.info(f"Matches in master_slip: {'matches' in payload['master_slip']}")
            if 'matches' in payload['master_slip']:
                matches = payload['master_slip']['matches']
                logger.info(f"Matches count: {len(matches)}")
                for i, match in enumerate(matches[:3]):  # Log first 3 matches only
                    logger.info(f"Match {i} keys: {list(match.keys())}")
                    logger.info(f"Match {i} markets: {len(match.get('match_markets', []))}")
                    if match.get('match_markets'):
                        for j, market in enumerate(match['match_markets'][:2]):
                            logger.info(f"  Market {j} keys: {list(market.keys())}")
                            if 'market' in market:
                                logger.info(f"    Market code: {market['market'].get('code')}")
                                logger.info(f"    Selections count: {len(market.get('selections', []))}")
        # ====================
        
        registry.build(payload)

        optimizer = ActiveMonteCarloOptimizer(
            registry=registry,
            random_seed=master_slip_id,
            num_simulations=self.num_simulations
        )

        hedging = HedgingEnforcer(registry=registry)

        if self.enable_monte_carlo:
            optimizer.run_simulations()

        generator = IntelligentSlipGenerator(
            registry=registry,
            random_seed=master_slip_id,
            monte_carlo_optimizer=optimizer,
            hedging_enforcer=hedging
        )

        slips = generator.generate()
        portfolio_metrics = self._calculate_portfolio_metrics(slips, optimizer)

        # NEW: Calculate optimal stakes (doesn't affect existing structure)
        optimal_stakes = self.stake_optimizer.calculate_stakes(slips, bankroll=1000.0)
        total_recommended = sum(optimal_stakes.values())

        response = {
            "generated_slips": [slip.to_dict() for slip in slips],
            "metadata": {
                "master_slip_id": master_slip_id,
                "total_slips": len(slips),
                "input_matches": len(registry.get_matches()),
                "total_selections": registry.total_selections,
                "unique_markets": len(registry.available_market_codes),
                "risk_distribution": {level.value: sum(1 for s in slips if s.risk_level == level) for level in RiskLevel},
                "monte_carlo_enabled": self.enable_monte_carlo,
                "portfolio_metrics": portfolio_metrics,
                "engine_version": "3.0.0",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                # NEW metadata keys (safe additions)
                "stake_recommendations": optimal_stakes,
                "total_recommended_stake": round(total_recommended, 2),
            },
        }

        logger.info("=" * 80)
        logger.info("[SLIP BUILDER v3.0] Generation complete")
        logger.info(f"[RESULT] Generated {len(slips)} slips")
        logger.info(f"[RESULT] Portfolio fitness: {portfolio_metrics.get('portfolio_fitness', portfolio_metrics.get('fitness_score', 0)):.3f}")
        logger.info("=" * 80)

        return response

    def _calculate_portfolio_metrics(self, slips: List[Slip], optimizer: ActiveMonteCarloOptimizer) -> Dict[str, Any]:
        metrics = {
            "average_confidence": round(np.mean([s.confidence_score for s in slips]), 3),
            "average_diversity": round(np.mean([s.diversity_score for s in slips]), 3),
            "average_coverage": round(np.mean([s.coverage_score for s in slips]), 3),
            "average_legs": round(np.mean([len(s.legs) for s in slips]), 2),
            "average_odds": round(np.mean([float(s.total_odds) for s in slips]), 2),
        }
        if optimizer.simulated:
            fitness = optimizer.calculate_portfolio_fitness(slips)
            # Keep backward-compatible keys, and include component breakdown under component_scores
            metrics.update({
                "portfolio_fitness": round(fitness.get("fitness_score", 0.0), 3),
                "coverage_percentage": round(fitness.get("coverage_percentage", 0.0) * 100, 2),
                "zero_win_rate": round(fitness.get("zero_win_rate", 0.0) * 100, 2),
                "target_win_rate": round(fitness.get("target_win_rate", 0.0) * 100, 2),
                "average_wins_per_simulation": round(fitness.get("avg_winners", 0.0), 2),
                "component_scores": fitness.get("component_scores", {}),
            })
        return metrics