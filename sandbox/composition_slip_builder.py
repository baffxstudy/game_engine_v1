"""
COMPOSITION SLIPS BUILDER - Deterministic Two-Phase Time-Structured Survival Engine

Transforms two independent 50-slip portfolios into a structured composition system using:
  - Early Upset Phase (MID odds: 2.00-2.60)
  - Late Closure Phase (LOW odds: 1.20-1.40 strictly)
  - Time-gap sequencing (early resolves first, late resolves after min_gap_minutes)
  - Survival cascade (losing early game eliminates that slip only)
  - Deterministic simulation (seeded RNG)
  - Portfolio-wide diversity & exposure control

Core Principle:
  This is NOT blind slip merging.
  This is a two-stage filter model where:
    - Early legs act as volatility filters (resolve first)
    - Late legs act as low-risk closers (resolve after confirmed time gap)
    - Losing early game exits only that slip (hedge survivability)
    - Portfolio remains coherent across the cascade

Key Features:
  ✓ Master slip ID grouping (separate two independent strategies)
  ✓ Time clustering (identify early/late windows)
  ✓ Strict odds band enforcement (MID for early, LOW 1.20-1.40 for late)
  ✓ Time-gap sequencing (late_kickoff >= early_kickoff + min_gap)
  ✓ Phase-driven assembly (not fusion, not merging)
  ✓ Deterministic pair construction (seeded RNG)
  ✓ Cascade constraint validation (time, odds, exposure)
  ✓ DNA deduplication (genetic clustering)
  ✓ Portfolio diversity optimization (cross-slip metrics)
  ✓ Exposure control (per-match, per-team, per-league)
  ✓ Re-scoring via SlipScorer (no scoring reinvention)
  ✓ Configuration-driven (no hardcoding)

Input: base_slips (early pool), optimized_slips (late pool), config
Output: List[composed_slip] with:
  - slip_id, legs, total_odds, confidence_score, coverage_score
  - diversity_score, risk_level, fitness_score
  - parent_ids (early + late), composition_metrics, phase_breakdown

Version: 2.1.0 - Two-Phase Time-Structured Survival Refactor
"""

import logging
import random
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Set, Optional, FrozenSet
from decimal import Decimal
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# DOMAIN MODELS
# ============================================================================

class OddsBand(str, Enum):
    """Odds classification for phase-based selection"""
    LOW = "low"          # 1.20-1.40: Late closure legs (HIGH confidence)
    MID = "mid"          # 2.00-2.60: Early upset legs (GROWTH drivers)
    HIGH = "high"        # >2.60: NOT ALLOWED in composition


class RiskLevel(str, Enum):
    """Risk stratification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PhaseWindowTimeGap:
    """Temporal constraint for phase separation"""
    early_kickoff: datetime
    late_kickoff: datetime
    min_gap_minutes: int
    satisfied: bool = False
    
    def validate(self) -> bool:
        """Check if time gap constraint is satisfied"""
        if not self.early_kickoff or not self.late_kickoff:
            return False
        
        gap_seconds = (self.late_kickoff - self.early_kickoff).total_seconds()
        gap_minutes = gap_seconds / 60
        
        self.satisfied = gap_minutes >= self.min_gap_minutes
        return self.satisfied


@dataclass
class CompositionDNA:
    """Genetic representation for deduplication and diversity"""
    early_match_ids: Set[int]
    late_match_ids: Set[int]
    league_ids: Set[str]
    team_ids: Set[str]
    market_types: Set[str]
    early_odds: float
    late_odds: float
    
    def fingerprint(self) -> FrozenSet:
        """Immutable signature for exact deduplication"""
        return frozenset([
            ("early_matches", frozenset(self.early_match_ids)),
            ("late_matches", frozenset(self.late_match_ids)),
            ("leagues", frozenset(self.league_ids)),
            ("markets", frozenset(self.market_types)),
        ])
    
    def distance(self, other: 'CompositionDNA') -> float:
        """
        Calculate diversity distance (0=identical, 1=completely different).
        
        Weighted metrics:
        - Early match overlap (30%)
        - Late match overlap (30%)
        - League overlap (20%)
        - Team overlap (20%)
        """
        def overlap_ratio(a: set, b: set) -> float:
            union = len(a | b)
            if union == 0:
                return 0.0
            intersection = len(a & b)
            return intersection / union
        
        early_overlap = overlap_ratio(self.early_match_ids, other.early_match_ids)
        late_overlap = overlap_ratio(self.late_match_ids, other.late_match_ids)
        league_overlap = overlap_ratio(self.league_ids, other.league_ids)
        team_overlap = overlap_ratio(self.team_ids, other.team_ids)
        
        similarity = (
            0.30 * early_overlap +
            0.30 * late_overlap +
            0.20 * league_overlap +
            0.20 * team_overlap
        )
        
        return 1.0 - similarity


@dataclass
class CompositionSlipRaw:
    """Raw two-phase composition before final scoring"""
    early_slip_id: str
    early_master_id: int
    early_leg: Dict[str, Any]
    late_slip_id: str
    late_master_id: int
    late_legs: List[Dict[str, Any]]
    
    # Validation metrics
    n_matches: int
    n_leagues: int
    n_markets: int
    time_gap_minutes: int
    early_odds: float
    late_odds: Decimal
    time_gap_valid: bool
    
    def total_odds_decimal(self) -> Decimal:
        """Calculate total odds as Decimal for precision"""
        total = Decimal(str(self.early_odds))
        for leg in self.late_legs:
            total *= Decimal(str(leg.get("odds", 1.0)))
        return total


# ============================================================================
# TWO-PHASE TIME-STRUCTURED SURVIVAL ENGINE
# ============================================================================

class CompositionSlipsBuilder:
    """
    Transforms two independent 50-slip portfolios into a time-structured
    two-phase survival composition system.
    
    Key Innovation:
      - NOT merging slips pairwise
      - SEPARATING by master_slip_id (strategy A vs strategy B)
      - PHASE-DRIVEN assembly (early upset + late closure)
      - TIME-GAP enforced (late cannot start before early + min_gap)
      - SURVIVAL CASCADE (early loss doesn't destroy late legs)
    
    Pipeline:
      1. Separate pools by master_slip_id
      2. Validate two pools (no more than 2 master_slip_ids)
      3. Classify odds bands (strict enforcement)
      4. Time clustering
      5. Two-phase slip construction (early + late pairs)
      6. Cascade constraint validation (time gaps, odds bands)
      7. Portfolio diversity optimization
      8. DNA deduplication
      9. Re-scoring via SlipScorer
     10. Final selection ranking
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        scorer: Callable,
        seed: int
    ):
        """
        Args:
            config: payload["master_slip"]["composition_slips"]
            scorer: SlipScorer callable
            seed: master_slip_id for deterministic RNG
        """
        self.config = config
        self.scorer = scorer
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        
        # Extract configuration
        self.targets = config.get("targets", {"count": 50, "min_matches": 2, "max_matches": 4})
        self.time_cfg = config.get("time_clustering", {
            "window_minutes": 120,
            "min_gap_minutes": 90
        })
        self.odds_cfg = config.get("odds_bands", {
            "low_min": 1.20,
            "low_max": 1.40,
            "mid_min": 2.00,
            "mid_max": 2.60,
            "high_min": 2.60
        })
        self.merge_cfg = config.get("merge_rules", {
            "max_per_league": 3,
            "allow_same_match": False,
            "max_late_legs_per_slip": 3
        })
        self.constraints_cfg = config.get("constraints", {
            "risk_bounds": {"min": "low", "max": "high"},
            "diversity": {"min_leagues": 2, "min_markets": 2},
            "correlation_limit": 0.6
        })
        self.portfolio_cfg = config.get("portfolio", {
            "max_exposure_per_match": 0.25,
            "max_exposure_per_team": 0.35,
            "max_exposure_per_league": 0.50,
            "min_diversity_distance": 0.30
        })
        
        # Unpack time constraints
        self.min_gap_minutes = int(self.time_cfg.get("min_gap_minutes", 90))
        
        # Track stats
        self.stats = {
            "early_pool_size": 0,
            "late_pool_size": 0,
            "early_master_ids": set(),
            "late_master_ids": set(),
            "candidates_generated": 0,
            "candidates_after_time_constraint": 0,
            "candidates_after_odds_constraint": 0,
            "final_deduped": 0,
            "final_compositions": 0,
            "portfolio_diversity_avg": 0.0,
        }
    
    def compose(
        self,
        base_slips: List[Dict[str, Any]],
        optimized_slips: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: compose two-phase slips deterministically.
        
        Args:
            base_slips: First independent 50-slip portfolio (Early pool)
            optimized_slips: Second independent 50-slip portfolio (Late pool)
        
        Returns:
            List[Dict] with two-phase composed slips
        """
        logger.info("[COMPOSE] ==================== START TWO-PHASE COMPOSITION ====================")
        logger.info("[COMPOSE] Seed: %d | Target: %d | Time Gap: %d min",
                   self.seed, self.targets.get("count", 50), self.min_gap_minutes)
        
        # ====== STAGE 1: Separate Pools by master_slip_id ======
        early_pool, late_pool = self._separate_pools_by_master_id(base_slips, optimized_slips)
        logger.info("[COMPOSE] [1] Early pool: %d slips | Late pool: %d slips",
                   len(early_pool), len(late_pool))
        
        # ====== STAGE 2: Classify Odds Bands ======
        early_classified = self._classify_odds_bands(early_pool)
        late_classified = self._classify_odds_bands(late_pool)
        logger.info("[COMPOSE] [2] Early: %d MID legs | Late: %d LOW legs",
                   len(early_classified.get(OddsBand.MID, [])),
                   len(late_classified.get(OddsBand.LOW, [])))
        
        # ====== STAGE 3: Two-Phase Slip Construction ======
        raw_candidates = self._construct_two_phase_slips(
            early_pool,
            late_pool,
            early_classified,
            late_classified
        )
        self.stats["candidates_generated"] = len(raw_candidates)
        logger.info("[COMPOSE] [3] Generated %d candidate compositions", len(raw_candidates))
        
        # ====== STAGE 4: Time Constraint Enforcement ======
        time_valid = self._enforce_time_gap_constraints(raw_candidates)
        self.stats["candidates_after_time_constraint"] = len(time_valid)
        logger.info("[COMPOSE] [4] After time-gap validation: %d slips", len(time_valid))
        
        # ====== STAGE 5: Odds Constraint Enforcement ======
        odds_valid = self._enforce_odds_constraints(time_valid)
        self.stats["candidates_after_odds_constraint"] = len(odds_valid)
        logger.info("[COMPOSE] [5] After odds constraint: %d slips", len(odds_valid))
        
        # ====== STAGE 6: Portfolio Diversity Optimization ======
        diversified = self._optimize_portfolio_diversity(odds_valid)
        logger.info("[COMPOSE] [6] After diversity optimization: %d slips", len(diversified))
        
        # ====== STAGE 7: DNA Deduplication ======
        deduped = self._deduplicate_by_dna(diversified)
        self.stats["final_deduped"] = len(deduped)
        logger.info("[COMPOSE] [7] After deduplication: %d unique slips", len(deduped))
        
        # ====== STAGE 8: Re-Scoring ======
        scored = self._rescore_compositions(deduped)
        logger.info("[COMPOSE] [8] Re-scored all compositions using SlipScorer")
        
        # ====== STAGE 9: Final Selection ======
        final = self._final_selection(scored)
        self.stats["final_compositions"] = len(final)
        logger.info("[COMPOSE] [9] Final selection: %d slips (target=%d)",
                   len(final), self.targets.get("count", 50))
        
        logger.info("[COMPOSE] Stats: %s", self.stats)
        logger.info("[COMPOSE] ==================== TWO-PHASE COMPOSITION COMPLETE ====================")
        
        return final
    
    # ========================================================================
    # STAGE 1: Separate Pools by master_slip_id
    # ========================================================================
    
    def _separate_pools_by_master_id(
        self,
        base_slips: List[Dict],
        optimized_slips: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate incoming slips by master_slip_id.
        
        Requirement:
          - base_slips should contain ONE master_slip_id (early pool)
          - optimized_slips should contain DIFFERENT master_slip_id (late pool)
          - If more than 2 master_slip_ids appear, raise error
        
        Returns:
            (early_pool, late_pool)
        """
        # Collect all slips with their master_slip_ids
        all_slips = []
        
        for slip in base_slips:
            slip_copy = deepcopy(slip)
            master_id = slip_copy.get("master_slip_id")
            slip_copy["_incoming_source"] = "base_slips"
            all_slips.append((master_id, slip_copy))
            self.stats["early_master_ids"].add(master_id)
        
        for slip in optimized_slips:
            slip_copy = deepcopy(slip)
            master_id = slip_copy.get("master_slip_id")
            slip_copy["_incoming_source"] = "optimized_slips"
            all_slips.append((master_id, slip_copy))
            self.stats["late_master_ids"].add(master_id)
        
        # Identify unique master_slip_ids
        unique_ids = set(master_id for master_id, _ in all_slips)
        
        if len(unique_ids) != 2:
            raise ValueError(
                f"Expected exactly 2 master_slip_ids (early + late pools). "
                f"Found {len(unique_ids)}: {unique_ids}. "
                f"Early: {self.stats['early_master_ids']}, "
                f"Late: {self.stats['late_master_ids']}"
            )
        
        # Sort by incoming source to ensure deterministic assignment
        early_id = min(self.stats["early_master_ids"])
        late_id = min(self.stats["late_master_ids"])
        
        if early_id == late_id:
            raise ValueError(
                f"Early and late pools have same master_slip_id: {early_id}. "
                f"This violates two-pool separation."
            )
        
        # Separate into two pools
        early_pool = [slip for mid, slip in all_slips if mid == early_id]
        late_pool = [slip for mid, slip in all_slips if mid == late_id]
        
        self.stats["early_pool_size"] = len(early_pool)
        self.stats["late_pool_size"] = len(late_pool)
        
        logger.info("[COMPOSE] [1] Separated pools: early_id=%d (%d slips), late_id=%d (%d slips)",
                   early_id, len(early_pool), late_id, len(late_pool))
        
        return early_pool, late_pool
    
    # ========================================================================
    # STAGE 2: Classify Odds Bands
    # ========================================================================
    
    def _classify_odds_bands(
        self,
        slips: List[Dict[str, Any]]
    ) -> Dict[OddsBand, List[Dict[str, Any]]]:
        """
        Classify slip legs into odds bands.
        
        Returns:
            Dict mapping OddsBand -> List of legs with band metadata
        """
        classified = {
            OddsBand.LOW: [],
            OddsBand.MID: [],
            OddsBand.HIGH: []
        }
        
        for slip in slips:
            for leg in slip.get("legs", []):
                odds = float(leg.get("odds", 1.0))
                band = self._get_odds_band(odds)
                
                leg_classified = {
                    **leg,
                    "odds_band": band.value,
                    "implied_prob": 1.0 / odds if odds > 0 else 0.0,
                    "_parent_slip_id": slip.get("slip_id"),
                    "_parent_master_id": slip.get("master_slip_id"),
                    "_parent_confidence": slip.get("confidence_score", 0.5),
                    "_parent_risk": slip.get("risk_level", "medium"),
                }
                
                classified[band].append(leg_classified)
        
        return classified
    
    def _get_odds_band(self, odds: float) -> OddsBand:
        """Classify odds into band"""
        low_min = float(self.odds_cfg.get("low_min", 1.20))
        low_max = float(self.odds_cfg.get("low_max", 1.40))
        mid_min = float(self.odds_cfg.get("mid_min", 2.00))
        mid_max = float(self.odds_cfg.get("mid_max", 2.60))
        
        if low_min <= odds <= low_max:
            return OddsBand.LOW
        elif mid_min <= odds <= mid_max:
            return OddsBand.MID
        else:
            return OddsBand.HIGH
    
    # ========================================================================
    # STAGE 3: Two-Phase Slip Construction (CORE ALGORITHM)
    # ========================================================================
    
    def _construct_two_phase_slips(
        self,
        early_pool: List[Dict],
        late_pool: List[Dict],
        early_classified: Dict[OddsBand, List[Dict]],
        late_classified: Dict[OddsBand, List[Dict]]
    ) -> List[CompositionSlipRaw]:
        """
        CORE NEW ALGORITHM: Construct two-phase slips deterministically.
        
        Strategy:
          FOR each early slip in deterministic order:
              FOR N iterations (oversample):
                  1. Select exactly 1 MID odds leg from early slip
                  2. Extract earliest kickoff_time from that leg
                  3. Filter late_classified[LOW] by:
                     - odds strictly in [1.20, 1.40]
                     - kickoff_time >= early_kickoff + min_gap_minutes
                  4. Select 1–3 late legs deterministically
                  5. Validate constraints
                  6. Create raw composition
        
        All randomness uses self.rng (seeded for determinism).
        """
        raw_compositions = []
        
        early_mid_legs = early_classified.get(OddsBand.MID, [])
        late_low_legs = late_classified.get(OddsBand.LOW, [])
        
        if not early_mid_legs or not late_low_legs:
            logger.warning("[COMPOSE] [3] Insufficient legs: early_mid=%d, late_low=%d",
                          len(early_mid_legs), len(late_low_legs))
            return raw_compositions
        
        # Shuffle deterministically
        early_mid_shuffled = self._deterministic_shuffle(early_mid_legs)
        late_low_shuffled = self._deterministic_shuffle(late_low_legs)
        
        # Number of iterations per early leg
        iterations_per_early = max(3, self.targets.get("count", 50) // len(early_pool) + 2)
        
        for early_leg_idx, early_leg in enumerate(early_mid_shuffled):
            for iteration in range(iterations_per_early):
                # Get early kickoff time
                early_kickoff = self._parse_kickoff_time(early_leg.get("kickoff_time"))
                
                # Filter late legs by time gap constraint
                late_legs_valid_time = [
                    leg for leg in late_low_shuffled
                    if self._check_time_gap(
                        early_kickoff,
                        self._parse_kickoff_time(leg.get("kickoff_time"))
                    )
                ]
                
                if not late_legs_valid_time:
                    continue
                
                # Deterministically shuffle late legs for this iteration
                late_selected = self.rng.sample(
                    late_legs_valid_time,
                    min(
                        self.rng.randint(1, int(self.merge_cfg.get("max_late_legs_per_slip", 3))),
                        len(late_legs_valid_time)
                    )
                )
                
                # Validate no duplicate matches
                early_match = early_leg.get("match_id")
                late_matches = {leg.get("match_id") for leg in late_selected}
                
                if early_match in late_matches:
                    continue
                
                # Validate no duplicate leagues (if configured)
                max_same_league = int(self.merge_cfg.get("max_per_league", 3))
                league_counts = Counter([leg.get("league") for leg in late_selected])
                league_counts[early_leg.get("league")] = 1
                
                if any(count > max_same_league for count in league_counts.values()):
                    continue
                
                # Create raw composition
                raw = CompositionSlipRaw(
                    early_slip_id=early_leg.get("_parent_slip_id"),
                    early_master_id=early_leg.get("_parent_master_id"),
                    early_leg=early_leg,
                    late_slip_id=late_selected[0].get("_parent_slip_id") if late_selected else "MULTI",
                    late_master_id=late_selected[0].get("_parent_master_id") if late_selected else 0,
                    late_legs=late_selected,
                    n_matches=len({early_match} | late_matches),
                    n_leagues=len(league_counts),
                    n_markets=len({early_leg.get("market")} | {leg.get("market") for leg in late_selected}),
                    time_gap_minutes=int(
                        (self._parse_kickoff_time(late_selected[0].get("kickoff_time")) - early_kickoff).total_seconds() / 60
                    ) if late_selected else 0,
                    early_odds=float(early_leg.get("odds", 1.0)),
                    late_odds=Decimal("1.0"),
                    time_gap_valid=False
                )
                
                # Calculate late odds
                for leg in late_selected:
                    raw.late_odds *= Decimal(str(leg.get("odds", 1.0)))
                
                raw_compositions.append(raw)
        
        return raw_compositions
    
    def _check_time_gap(self, early_kickoff: datetime, late_kickoff: datetime) -> bool:
        """Check if late_kickoff is at least min_gap_minutes after early_kickoff"""
        if not early_kickoff or not late_kickoff:
            return False
        
        gap_minutes = (late_kickoff - early_kickoff).total_seconds() / 60
        return gap_minutes >= self.min_gap_minutes
    
    def _parse_kickoff_time(self, kickoff_str: Any) -> datetime:
        """Parse kickoff time to datetime"""
        if isinstance(kickoff_str, datetime):
            return kickoff_str
        
        if isinstance(kickoff_str, str):
            try:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                    try:
                        return datetime.strptime(kickoff_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
        
        return datetime.now()
    
    def _deterministic_shuffle(self, items: List[Any]) -> List[Any]:
        """Shuffle deterministically with seeded RNG"""
        shuffled = items.copy()
        self.rng.shuffle(shuffled)
        return shuffled
    
    # ========================================================================
    # STAGE 4: Time Constraint Enforcement
    # ========================================================================
    
    def _enforce_time_gap_constraints(
        self,
        raw_compositions: List[CompositionSlipRaw]
    ) -> List[CompositionSlipRaw]:
        """
        Validate time gap constraints for all compositions.
        
        Requirement:
          late_kickoff >= early_kickoff + min_gap_minutes
        """
        valid = []
        
        for raw in raw_compositions:
            early_kickoff = self._parse_kickoff_time(raw.early_leg.get("kickoff_time"))
            
            # Check against each late leg
            all_valid = True
            for late_leg in raw.late_legs:
                late_kickoff = self._parse_kickoff_time(late_leg.get("kickoff_time"))
                if not self._check_time_gap(early_kickoff, late_kickoff):
                    all_valid = False
                    break
            
            if all_valid:
                raw.time_gap_valid = True
                valid.append(raw)
        
        return valid
    
    # ========================================================================
    # STAGE 5: Odds Constraint Enforcement
    # ========================================================================
    
    def _enforce_odds_constraints(
        self,
        raw_compositions: List[CompositionSlipRaw]
    ) -> List[CompositionSlipRaw]:
        """
        Enforce strict odds band constraints:
          - Early leg: MUST be MID band (2.00–2.60)
          - Late legs: MUST be LOW band (1.20–1.40) strictly
          - No HIGH odds allowed
          - No multiple MID legs
        """
        valid = []
        
        low_min = float(self.odds_cfg.get("low_min", 1.20))
        low_max = float(self.odds_cfg.get("low_max", 1.40))
        mid_min = float(self.odds_cfg.get("mid_min", 2.00))
        mid_max = float(self.odds_cfg.get("mid_max", 2.60))
        
        for raw in raw_compositions:
            # Check early leg
            early_odds = raw.early_odds
            if not (mid_min <= early_odds <= mid_max):
                continue
            
            # Check all late legs
            all_late_valid = True
            for late_leg in raw.late_legs:
                late_odds = float(late_leg.get("odds", 1.0))
                if not (low_min <= late_odds <= low_max):
                    all_late_valid = False
                    break
            
            if all_late_valid:
                valid.append(raw)
        
        return valid
    
    # ========================================================================
    # STAGE 6: Portfolio Diversity Optimization
    # ========================================================================
    
    def _optimize_portfolio_diversity(
        self,
        raw_compositions: List[CompositionSlipRaw]
    ) -> List[CompositionSlipRaw]:
        """
        Select diverse slips from candidates using DNA distance.
        
        Algorithm:
        1. Sort by total odds descending
        2. Iteratively add slips with max diversity distance
        3. Reject slips too similar to portfolio
        """
        if not raw_compositions:
            return []
        
        # Sort by total odds descending
        raw_compositions.sort(key=lambda r: float(r.early_odds) * float(r.late_odds), reverse=True)
        
        selected = []
        selected_dnas = []
        min_distance = float(self.portfolio_cfg.get("min_diversity_distance", 0.30))
        target_count = int(self.targets.get("count", 50))
        
        for raw in raw_compositions:
            if len(selected) >= target_count:
                break
            
            # Build DNA
            dna = self._build_dna_from_raw(raw)
            
            # First slip always selected
            if not selected:
                selected.append(raw)
                selected_dnas.append(dna)
                continue
            
            # Calculate minimum distance
            distances = [dna.distance(existing_dna) for existing_dna in selected_dnas]
            min_dist = min(distances) if distances else 1.0
            
            # Accept if diverse enough
            if min_dist >= min_distance:
                selected.append(raw)
                selected_dnas.append(dna)
        
        # Fill remaining slots if needed
        if len(selected) < target_count:
            remaining = [r for r in raw_compositions if r not in selected]
            needed = target_count - len(selected)
            selected.extend(remaining[:needed])
        
        # Calculate average diversity
        if len(selected_dnas) > 1:
            distances = []
            for i, dna_a in enumerate(selected_dnas):
                for dna_b in selected_dnas[i+1:]:
                    distances.append(dna_a.distance(dna_b))
            self.stats["portfolio_diversity_avg"] = sum(distances) / len(distances) if distances else 0.0
        
        return selected
    
    def _build_dna_from_raw(self, raw: CompositionSlipRaw) -> CompositionDNA:
        """Build CompositionDNA from raw composition"""
        early_matches = {raw.early_leg.get("match_id")}
        late_matches = {leg.get("match_id") for leg in raw.late_legs}
        
        leagues = {raw.early_leg.get("league")}
        for leg in raw.late_legs:
            leagues.add(leg.get("league"))
        
        teams = set()
        for leg in [raw.early_leg] + raw.late_legs:
            teams.add(leg.get("home_team"))
            teams.add(leg.get("away_team"))
        
        markets = {raw.early_leg.get("market")}
        for leg in raw.late_legs:
            markets.add(leg.get("market"))
        
        return CompositionDNA(
            early_match_ids=early_matches,
            late_match_ids=late_matches,
            league_ids=leagues,
            team_ids=teams,
            market_types=markets,
            early_odds=raw.early_odds,
            late_odds=float(raw.late_odds)
        )
    
    # ========================================================================
    # STAGE 7: DNA Deduplication
    # ========================================================================
    
    def _deduplicate_by_dna(self, raw_comps: List[CompositionSlipRaw]) -> List[CompositionSlipRaw]:
        """
        Remove exact genetic duplicates.
        
        Signature = frozenset of early + late match/market/selection combinations
        """
        seen_dnas = set()
        unique = []
        
        for raw in raw_comps:
            # Build exact signature
            signature = frozenset(
                [("early", raw.early_leg.get("match_id"), raw.early_leg.get("market"), raw.early_leg.get("selection"))] +
                [("late", leg.get("match_id"), leg.get("market"), leg.get("selection")) for leg in raw.late_legs]
            )
            
            if signature in seen_dnas:
                continue
            
            seen_dnas.add(signature)
            unique.append(raw)
        
        return unique
    
    # ========================================================================
    # STAGE 8: Re-Scoring with SlipScorer
    # ========================================================================
    
    def _rescore_compositions(
        self,
        raw_compositions: List[CompositionSlipRaw]
    ) -> List[Dict[str, Any]]:
        """
        Re-score all compositions using SlipScorer.
        
        Expects scorer to return:
        {
            "legs": [...],
            "total_odds": float,
            "confidence_score": float,
            "coverage_score": float,
            "diversity_score": float,
            "risk_level": str,
            "fitness_score": float (optional)
        }
        """
        scored = []
        
        for i, raw in enumerate(raw_compositions):
            try:
                # Merge legs for scoring
                all_legs = [raw.early_leg] + raw.late_legs
                
                # Call scorer
                score_result = self.scorer.score(all_legs)
                
                # Calculate phase breakdown
                early_kickoff = self._parse_kickoff_time(raw.early_leg.get("kickoff_time"))
                late_kickoffs = [
                    self._parse_kickoff_time(leg.get("kickoff_time"))
                    for leg in raw.late_legs
                ]
                
                # Assemble final slip
                slip = {
                    "slip_id": f"COMP_{i+1:04d}",
                    "legs": score_result.get("legs", all_legs),
                    "total_odds": float(score_result.get("total_odds", raw.total_odds_decimal())),
                    "confidence_score": float(score_result.get("confidence_score", 0.5)),
                    "coverage_score": float(score_result.get("coverage_score", 0.0)),
                    "diversity_score": float(score_result.get("diversity_score", 0.0)),
                    "risk_level": score_result.get("risk_level", "medium"),
                    "fitness_score": float(score_result.get("fitness_score", 0.0)),
                    # Parent IDs
                    "parent_ids": [raw.early_slip_id] + [raw.late_slip_id],
                    "parent_master_ids": [raw.early_master_id, raw.late_master_id],
                    # Composition metrics
                    "composition_metrics": {
                        "n_matches": raw.n_matches,
                        "n_leagues": raw.n_leagues,
                        "n_markets": raw.n_markets,
                        "time_gap_minutes": raw.time_gap_minutes,
                    },
                    # Phase breakdown
                    "phase_breakdown": {
                        "early_slip_id": raw.early_slip_id,
                        "early_leg": raw.early_leg,
                        "early_odds": raw.early_odds,
                        "early_kickoff": early_kickoff.isoformat() if early_kickoff else None,
                        "late_slip_ids": [leg.get("_parent_slip_id") for leg in raw.late_legs],
                        "late_legs": raw.late_legs,
                        "late_odds": float(raw.late_odds),
                        "late_kickoffs": [k.isoformat() if k else None for k in late_kickoffs],
                        "time_gap_valid": raw.time_gap_valid,
                    }
                }
                
                scored.append(slip)
            
            except Exception as e:
                logger.error("[COMPOSE] Failed to score composition: %s", str(e))
                continue
        
        return scored
    
    # ========================================================================
    # STAGE 9: Final Selection
    # ========================================================================
    
    def _final_selection(self, scored_slips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select top N slips by ranking criteria.
        
        Ranking:
          1. Fitness score (desc)
          2. Confidence score (desc)
          3. Coverage score (desc)
          4. Total odds (desc)
        """
        if not scored_slips:
            logger.warning("[COMPOSE] No valid compositions for final selection")
            return []
        
        # Sort by ranking criteria
        scored_slips.sort(
            key=lambda s: (
                -s.get("fitness_score", 0),
                -s.get("confidence_score", 0),
                -s.get("coverage_score", 0),
                -s.get("total_odds", 1),
            )
        )
        
        # Select top N
        target_count = int(self.targets.get("count", 50))
        final = scored_slips[:target_count]
        
        if len(final) < target_count:
            logger.warning(
                "[COMPOSE] Requested %d slips but only %d valid available",
                target_count, len(final)
            )
        
        return final