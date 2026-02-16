"""
COMPOSITION SLIPS BUILDER - Deterministic Two-Phase Time-Structured Survival Engine
Version: 3.0.0 - Phase-Driven Survival Cascade Engine

Core Strategy:
  Strategy A (Early Upset Pool) → MID odds (2.00-2.60) for early matches
  Strategy B (Late Stabilizer Pool) → LOW odds (1.20-1.40) for late matches
  
  Each composed slip MUST contain:
    - Exactly 1 leg from Early Pool (MID band, early kickoff)
    - 1-3 legs from Late Pool (LOW band, after min_gap_minutes)
  
  This creates survival cascade:
    Early leg resolves first → If win, slip continues with low-risk legs
    If lose, slip dies early minimizing exposure

Input: Two independent 50-slip portfolios (base_slips, optimized_slips)
Output: List[composed_slip] with strict phase structure
"""

import logging
import random
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Set, Optional, Callable, FrozenSet
from decimal import Decimal
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# DOMAIN MODELS
# ============================================================================

class OddsBand(str, Enum):
    """Odds classification for strategic selection"""
    LOW = "low"          # 1.20-1.40: Late stabilizer legs only
    MID = "mid"          # 2.00-2.60: Early upset legs only
    HIGH = "high"        # NOT ALLOWED in phase structure


@dataclass
class PhaseCandidate:
    """Candidate leg for phase-based composition"""
    match_id: int
    market: str
    selection: str
    odds: float
    kickoff_time: datetime
    league: str
    home_team: str
    away_team: str
    master_slip_id: int
    parent_slip_id: str
    confidence_score: float
    risk_level: str
    odds_band: OddsBand
    leg_data: Dict[str, Any]  # Original leg data


@dataclass
class TimeGroup:
    """Group of slips by master_slip_id and time characteristics"""
    master_slip_id: int
    slips: List[Dict[str, Any]]
    avg_kickoff: datetime
    avg_odds: float
    odds_distribution: Dict[OddsBand, int]
    is_early_candidate: bool


@dataclass
class PhaseComposition:
    """Raw phase composition before scoring"""
    early_leg: PhaseCandidate
    late_legs: List[PhaseCandidate]
    early_slip_id: str
    late_slip_ids: List[str]
    total_odds: float
    early_kickoff: datetime
    latest_late_kickoff: datetime
    n_leagues: int
    n_markets: int
    n_teams: int
    composition_metrics: Dict[str, Any]


@dataclass
class CompositionDNA:
    """Normalized genetic representation for deduplication"""
    selections: Dict[Tuple[int, str], Dict[str, Any]]
    
    def fingerprint(self) -> FrozenSet:
        """Return immutable (match, market, selection) signature for deduplication"""
        return frozenset(
            (k[0], k[1], v.get("selection", ""))
            for k, v in self.selections.items()
        )
    
    def distance(self, other: 'CompositionDNA') -> float:
        """
        Calculate diversity distance (0=identical, 1=completely different).
        """
        if not self.selections or not other.selections:
            return 0.0
        
        # Extract dimensions
        self_matches = set(k[0] for k in self.selections.keys())
        other_matches = set(k[0] for k in other.selections.keys())
        
        self_markets = set(k[1] for k in self.selections.keys())
        other_markets = set(k[1] for k in other.selections.keys())
        
        self_leagues = set(v.get("league") for v in self.selections.values())
        other_leagues = set(v.get("league") for v in other.selections.values())
        
        self_teams = set()
        for v in self.selections.values():
            self_teams.add(v.get("home_team"))
            self_teams.add(v.get("away_team"))
        
        other_teams = set()
        for v in other.selections.values():
            other_teams.add(v.get("home_team"))
            other_teams.add(v.get("away_team"))
        
        # Calculate overlaps (0-1, where 1 = complete overlap)
        def overlap_ratio(a: set, b: set) -> float:
            union = len(a | b)
            if union == 0:
                return 0.0
            intersection = len(a & b)
            return intersection / union
        
        match_overlap = overlap_ratio(self_matches, other_matches)
        market_overlap = overlap_ratio(self_markets, other_markets)
        league_overlap = overlap_ratio(self_leagues, other_leagues)
        team_overlap = overlap_ratio(self_teams, other_teams)
        
        # Weighted similarity
        similarity = (
            0.35 * match_overlap +
            0.15 * market_overlap +
            0.20 * league_overlap +
            0.30 * team_overlap
        )
        
        return 1.0 - similarity


# ============================================================================
# COMPOSITION SLIPS BUILDER - PHASE-DRIVEN SURVIVAL ENGINE
# ============================================================================

class CompositionSlipsBuilder:
    """
    Deterministic Two-Phase Time-Structured Survival Engine.
    
    Strategic phases:
      1. Group slips by master_slip_id (must be exactly 2 groups)
      2. Identify Early Pool (MID odds, early kickoff)
      3. Identify Late Pool (LOW odds 1.20-1.40, later kickoff)
      4. Construct phase compositions with strict time gap
      5. Enforce odds band constraints
      6. Portfolio diversity optimization
      7. Deduplication and final selection
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        scorer: Callable,
        seed: int
    ):
        """
        Args:
            config: payload["master_slip"]["composition_slips"] (full subtree)
            scorer: Callable that accepts List[Dict] legs, returns scored Dict
            seed: master_slip_id (for deterministic RNG)
        """
        self.config = config
        self.scorer = scorer
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        
        # Extract configuration sections
        self.targets = config.get("targets", {"count": 50, "min_matches": 2, "max_matches": 4})
        self.time_cfg = config.get("time_clustering", {
            "window_minutes": 120,
            "min_gap_minutes": 90
        })
        self.odds_cfg = config.get("odds_bands", {
            "low_min": 1.20,
            "low_max": 1.40,
            "mid_min": 2.00,
            "mid_max": 2.60
        })
        self.constraints_cfg = config.get("constraints", {
            "risk_bounds": {"min": "low", "max": "high"},
            "diversity": {"min_leagues": 2, "min_markets": 2},
            "correlation_limit": 0.5
        })
        self.portfolio_cfg = config.get("portfolio", {
            "max_exposure_per_match": 0.25,
            "max_exposure_per_team": 0.35,
            "max_exposure_per_league": 0.50,
            "min_diversity_distance": 0.30
        })
        
        # Track stats for logging
        self.stats = {
            "total_slips": 0,
            "early_group_size": 0,
            "late_group_size": 0,
            "early_legs_candidates": 0,
            "late_legs_candidates": 0,
            "phase_compositions_created": 0,
            "compositions_after_constraint": 0,
            "deduped_compositions": 0,
            "final_compositions": 0,
            "portfolio_diversity_avg": 0.0,
        }
    
    def compose(
        self,
        base_slips: List[Dict[str, Any]],
        optimized_slips: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: compose slips using phase-driven survival engine.
        
        Args:
            base_slips: First independent 50-slip portfolio (master_slip_id = 9001)
            optimized_slips: Second independent 50-slip portfolio (master_slip_id = 9002)
        
        Returns:
            List[Dict] with strict phase structure
        """
        logger.info("[COMPOSE] ========== PHASE-DRIVEN SURVIVAL ENGINE START ==========")
        logger.info("[COMPOSE] Seed: %d | Target: %d", self.seed, self.targets.get("count", 50))
        logger.info("[COMPOSE] Strict odds bands: LOW=%.2f-%.2f, MID=%.2f-%.2f",
                   self.odds_cfg.get("low_min", 1.20), self.odds_cfg.get("low_max", 1.40),
                   self.odds_cfg.get("mid_min", 2.00), self.odds_cfg.get("mid_max", 2.60))
        
        # ====== STAGE 1: Group by master_slip_id ======
        all_slips = deepcopy(base_slips) + deepcopy(optimized_slips)
        self.stats["total_slips"] = len(all_slips)
        
        time_groups = self._group_slips_by_master_slip_id(all_slips)
        
        if len(time_groups) != 2:
            raise ValueError(
                f"Expected exactly 2 distinct master_slip_id groups, found {len(time_groups)}. "
                f"Groups: {[g.master_slip_id for g in time_groups]}"
            )
        
        logger.info("[COMPOSE] [1] Found %d master_slip_id groups", len(time_groups))
        for group in time_groups:
            logger.info("[COMPOSE]     Master slip %d: %d slips, avg kickoff: %s",
                       group.master_slip_id, len(group.slips), group.avg_kickoff)
        
        # ====== STAGE 2: Identify Early and Late Pools ======
        early_group, late_group = self._identify_phase_pools(time_groups)
        self.stats["early_group_size"] = len(early_group.slips)
        self.stats["late_group_size"] = len(late_group.slips)
        
        logger.info("[COMPOSE] [2] Phase identification:")
        logger.info("[COMPOSE]     Early Pool: master_slip_id=%d, %d slips",
                   early_group.master_slip_id, len(early_group.slips))
        logger.info("[COMPOSE]     Late Pool: master_slip_id=%d, %d slips",
                   late_group.master_slip_id, len(late_group.slips))
        
        # ====== STAGE 3: Extract Phase Candidates ======
        early_candidates = self._extract_early_candidates(early_group)
        late_candidates = self._extract_late_candidates(late_group)
        
        self.stats["early_legs_candidates"] = len(early_candidates)
        self.stats["late_legs_candidates"] = len(late_candidates)
        
        logger.info("[COMPOSE] [3] Phase candidates extracted:")
        logger.info("[COMPOSE]     Early (MID band): %d legs", len(early_candidates))
        logger.info("[COMPOSE]     Late (LOW band 1.20-1.40): %d legs", len(late_candidates))
        
        # ====== STAGE 4: Construct Phase Compositions ======
        phase_compositions = self._construct_two_phase_slips(early_candidates, late_candidates)
        self.stats["phase_compositions_created"] = len(phase_compositions)
        
        logger.info("[COMPOSE] [4] Constructed %d phase compositions", len(phase_compositions))
        
        # ====== STAGE 5: Constraint Enforcement ======
        constrained = self._enforce_all_constraints(phase_compositions)
        self.stats["compositions_after_constraint"] = len(constrained)
        logger.info("[COMPOSE] [5] After constraints: %d valid compositions", len(constrained))
        
        # ====== STAGE 6: Portfolio Diversity Optimization ======
        diversified = self._optimize_portfolio_diversity(constrained)
        logger.info("[COMPOSE] [6] After diversity optimization: %d compositions", len(diversified))
        
        # ====== STAGE 7: Deduplication ======
        deduped = self._deduplicate_by_dna(diversified)
        self.stats["deduped_compositions"] = len(deduped)
        logger.info("[COMPOSE] [7] After dedup: %d unique compositions", len(deduped))
        
        # ====== STAGE 8: Re-Scoring ======
        scored = self._rescore_compositions(deduped)
        logger.info("[COMPOSE] [8] Re-scored all compositions using SlipScorer")
        
        # ====== STAGE 9: Final Selection ======
        final = self._final_selection(scored)
        self.stats["final_compositions"] = len(final)
        logger.info("[COMPOSE] [9] Final selection: %d slips (target=%d)", 
                   len(final), self.targets.get("count", 50))
        
        logger.info("[COMPOSE] Stats: %s", self.stats)
        logger.info("[COMPOSE] ========== PHASE-DRIVEN SURVIVAL ENGINE COMPLETE ==========")
        
        return final
    
    # ========================================================================
    # STAGE 1: Group by master_slip_id
    # ========================================================================
    
    def _group_slips_by_master_slip_id(self, slips: List[Dict]) -> List[TimeGroup]:
        """Group slips by their master_slip_id"""
        groups = defaultdict(list)
        
        for slip in slips:
            master_slip_id = slip.get("master_slip_id")
            if master_slip_id is None:
                # Try to extract from slip_id if format is MS{id}_{num}
                slip_id = slip.get("slip_id", "")
                if slip_id.startswith("MS"):
                    try:
                        master_slip_id = int(slip_id.split("_")[0][2:])
                    except (ValueError, IndexError):
                        master_slip_id = 0
                else:
                    master_slip_id = 0
            
            groups[master_slip_id].append(slip)
        
        time_groups = []
        for master_slip_id, slip_list in groups.items():
            # Calculate average kickoff time
            all_kickoffs = []
            for slip in slip_list:
                for leg in slip.get("legs", []):
                    kickoff = self._parse_kickoff_time(leg.get("kickoff_time"))
                    all_kickoffs.append(kickoff)
            
            avg_kickoff = self._calculate_average_datetime(all_kickoffs) if all_kickoffs else datetime.now()
            
            # Calculate odds distribution
            odds_dist = self._calculate_odds_distribution(slip_list)
            
            time_groups.append(TimeGroup(
                master_slip_id=master_slip_id,
                slips=slip_list,
                avg_kickoff=avg_kickoff,
                avg_odds=self._calculate_average_odds(slip_list),
                odds_distribution=odds_dist,
                is_early_candidate=False  # Will be determined later
            ))
        
        # Sort by average kickoff time
        time_groups.sort(key=lambda g: g.avg_kickoff)
        
        return time_groups
    
    def _parse_kickoff_time(self, kickoff_str: Any) -> datetime:
        """Parse kickoff time string to datetime"""
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
    
    def _calculate_average_datetime(self, datetimes: List[datetime]) -> datetime:
        """Calculate average datetime from list"""
        if not datetimes:
            return datetime.now()
        
        # Convert to timestamps, average, convert back
        timestamps = [dt.timestamp() for dt in datetimes]
        avg_timestamp = sum(timestamps) / len(timestamps)
        return datetime.fromtimestamp(avg_timestamp)
    
    def _calculate_average_odds(self, slips: List[Dict]) -> float:
        """Calculate average odds across all slips"""
        total_odds = 0.0
        count = 0
        
        for slip in slips:
            odds = slip.get("total_odds")
            if isinstance(odds, (int, float, Decimal)):
                total_odds += float(odds)
                count += 1
        
        return total_odds / count if count > 0 else 1.0
    
    def _calculate_odds_distribution(self, slips: List[Dict]) -> Dict[OddsBand, int]:
        """Calculate distribution of odds bands across slips"""
        distribution = {band: 0 for band in OddsBand}
        
        for slip in slips:
            for leg in slip.get("legs", []):
                odds = float(leg.get("odds", 1.0))
                band = self._get_odds_band(odds)
                distribution[band] += 1
        
        return distribution
    
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
    # STAGE 2: Identify Early and Late Pools
    # ========================================================================
    
    def _identify_phase_pools(self, time_groups: List[TimeGroup]) -> Tuple[TimeGroup, TimeGroup]:
        """
        Identify which group is Early Pool and which is Late Pool.
        
        Strategy:
        - Group with earlier average kickoff → Early Pool candidate
        - Group with more MID odds → Early Pool
        - Group with more LOW odds → Late Pool
        """
        if len(time_groups) != 2:
            raise ValueError(f"Expected 2 groups, got {len(time_groups)}")
        
        # Sort by average kickoff
        sorted_groups = sorted(time_groups, key=lambda g: g.avg_kickoff)
        
        # Calculate MID odds percentage for each group
        for group in sorted_groups:
            total_legs = sum(group.odds_distribution.values())
            if total_legs > 0:
                mid_pct = group.odds_distribution.get(OddsBand.MID, 0) / total_legs
            else:
                mid_pct = 0.0
            
            # Determine if this group is better as early pool
            group.is_early_candidate = mid_pct > 0.3  # At least 30% MID odds
        
        # If both groups have MID odds, use kickoff time as tiebreaker
        early_candidates = [g for g in sorted_groups if g.is_early_candidate]
        late_candidates = [g for g in sorted_groups if not g.is_early_candidate]
        
        if len(early_candidates) == 2:
            # Both have MID odds, use earlier kickoff as early pool
            early_group = sorted_groups[0]
            late_group = sorted_groups[1]
        elif len(early_candidates) == 1:
            # Exactly one has MID odds
            early_group = early_candidates[0]
            late_group = [g for g in sorted_groups if g != early_group][0]
        else:
            # Neither has MID odds, use earlier kickoff as early pool
            early_group = sorted_groups[0]
            late_group = sorted_groups[1]
            logger.warning("[COMPOSE] No group has sufficient MID odds for Early Pool")
        
        # Verify Late Pool has sufficient LOW odds
        late_low_pct = late_group.odds_distribution.get(OddsBand.LOW, 0) / max(sum(late_group.odds_distribution.values()), 1)
        if late_low_pct < 0.5:
            logger.warning("[COMPOSE] Late Pool has low percentage of LOW odds (%.1f%%)", late_low_pct * 100)
        
        return early_group, late_group
    
    # ========================================================================
    # STAGE 3: Extract Phase Candidates
    # ========================================================================
    
    def _extract_early_candidates(self, early_group: TimeGroup) -> List[PhaseCandidate]:
        """Extract MID odds legs from Early Pool"""
        candidates = []
        mid_min = float(self.odds_cfg.get("mid_min", 2.00))
        mid_max = float(self.odds_cfg.get("mid_max", 2.60))
        
        for slip in early_group.slips:
            slip_id = slip.get("slip_id", "")
            confidence = float(slip.get("confidence_score", 0.5))
            risk_level = slip.get("risk_level", "medium")
            
            for leg in slip.get("legs", []):
                odds = float(leg.get("odds", 1.0))
                
                # STRICT: Must be in MID band
                if not (mid_min <= odds <= mid_max):
                    continue
                
                # Parse kickoff time
                kickoff = self._parse_kickoff_time(leg.get("kickoff_time"))
                
                candidate = PhaseCandidate(
                    match_id=leg.get("match_id"),
                    market=leg.get("market", ""),
                    selection=leg.get("selection", ""),
                    odds=odds,
                    kickoff_time=kickoff,
                    league=leg.get("league", ""),
                    home_team=leg.get("home_team", ""),
                    away_team=leg.get("away_team", ""),
                    master_slip_id=early_group.master_slip_id,
                    parent_slip_id=slip_id,
                    confidence_score=confidence,
                    risk_level=risk_level,
                    odds_band=OddsBand.MID,
                    leg_data=deepcopy(leg)
                )
                
                candidates.append(candidate)
        
        # Sort by kickoff time (earliest first)
        candidates.sort(key=lambda c: c.kickoff_time)
        
        return candidates
    
    def _extract_late_candidates(self, late_group: TimeGroup) -> List[PhaseCandidate]:
        """Extract LOW odds legs (1.20-1.40) from Late Pool"""
        candidates = []
        low_min = float(self.odds_cfg.get("low_min", 1.20))
        low_max = float(self.odds_cfg.get("low_max", 1.40))
        
        # STRICT: Override config to enforce 1.20-1.40 range
        low_min = max(low_min, 1.20)
        low_max = min(low_max, 1.40)
        
        for slip in late_group.slips:
            slip_id = slip.get("slip_id", "")
            confidence = float(slip.get("confidence_score", 0.5))
            risk_level = slip.get("risk_level", "medium")
            
            for leg in slip.get("legs", []):
                odds = float(leg.get("odds", 1.0))
                
                # STRICT: Must be in LOW band 1.20-1.40
                if not (low_min <= odds <= low_max):
                    continue
                
                # Parse kickoff time
                kickoff = self._parse_kickoff_time(leg.get("kickoff_time"))
                
                candidate = PhaseCandidate(
                    match_id=leg.get("match_id"),
                    market=leg.get("market", ""),
                    selection=leg.get("selection", ""),
                    odds=odds,
                    kickoff_time=kickoff,
                    league=leg.get("league", ""),
                    home_team=leg.get("home_team", ""),
                    away_team=leg.get("away_team", ""),
                    master_slip_id=late_group.master_slip_id,
                    parent_slip_id=slip_id,
                    confidence_score=confidence,
                    risk_level=risk_level,
                    odds_band=OddsBand.LOW,
                    leg_data=deepcopy(leg)
                )
                
                candidates.append(candidate)
        
        # Sort by kickoff time
        candidates.sort(key=lambda c: c.kickoff_time)
        
        return candidates
    
    # ========================================================================
    # STAGE 4: Construct Two-Phase Slips (REPLACES STAGES 6-7)
    # ========================================================================
    
    def _construct_two_phase_slips(
        self,
        early_candidates: List[PhaseCandidate],
        late_candidates: List[PhaseCandidate]
    ) -> List[PhaseComposition]:
        """
        Construct phase-driven compositions with strict time gap.
        
        Algorithm:
        For each early candidate:
            For N deterministic iterations:
                1. Filter late candidates by time gap (min_gap_minutes)
                2. Select 1-3 late legs
                3. Validate constraints
                4. Create composition
        """
        compositions = []
        min_gap_minutes = int(self.time_cfg.get("min_gap_minutes", 90))
        target_count = self.targets.get("count", 50)
        iterations_per_early = 5  # Deterministic iterations per early leg
        
        # Shuffle deterministically
        early_shuffled = list(early_candidates)
        self.rng.shuffle(early_shuffled)
        
        late_shuffled = list(late_candidates)
        self.rng.shuffle(late_shuffled)
        
        for i, early_candidate in enumerate(early_shuffled[:target_count * 2]):  # Limit search
            early_kickoff = early_candidate.kickoff_time
            
            # Filter late candidates by time gap
            valid_late = [
                late for late in late_shuffled
                if (late.kickoff_time - early_kickoff).total_seconds() / 60 >= min_gap_minutes
            ]
            
            if not valid_late:
                continue  # No valid late legs for this early candidate
            
            # Create multiple compositions per early candidate
            for iter_num in range(iterations_per_early):
                # Determine number of late legs (1-3)
                n_late = self.rng.randint(1, 3)
                
                # Select late legs deterministically
                if len(valid_late) >= n_late:
                    # Use deterministic selection based on iteration number
                    selection_start = (i * iterations_per_early + iter_num) % len(valid_late)
                    selected_late = []
                    
                    for j in range(n_late):
                        idx = (selection_start + j) % len(valid_late)
                        selected_late.append(valid_late[idx])
                    
                    # Check for duplicates
                    match_ids = {early_candidate.match_id}
                    market_pairs = {(early_candidate.match_id, early_candidate.market)}
                    
                    for late in selected_late:
                        match_ids.add(late.match_id)
                        market_pairs.add((late.match_id, late.market))
                    
                    # Skip if duplicate matches or duplicate (match, market) pairs
                    if len(market_pairs) != (1 + len(selected_late)):
                        continue
                    
                    # Calculate composition metrics
                    late_slip_ids = [late.parent_slip_id for late in selected_late]
                    
                    # Calculate total odds
                    total_odds = early_candidate.odds
                    for late in selected_late:
                        total_odds *= late.odds
                    
                    # Calculate diversity metrics
                    all_leagues = {early_candidate.league}
                    all_markets = {early_candidate.market}
                    all_teams = {early_candidate.home_team, early_candidate.away_team}
                    
                    for late in selected_late:
                        all_leagues.add(late.league)
                        all_markets.add(late.market)
                        all_teams.add(late.home_team)
                        all_teams.add(late.away_team)
                    
                    latest_late_kickoff = max(late.kickoff_time for late in selected_late)
                    
                    composition = PhaseComposition(
                        early_leg=early_candidate,
                        late_legs=selected_late,
                        early_slip_id=early_candidate.parent_slip_id,
                        late_slip_ids=late_slip_ids,
                        total_odds=total_odds,
                        early_kickoff=early_kickoff,
                        latest_late_kickoff=latest_late_kickoff,
                        n_leagues=len(all_leagues),
                        n_markets=len(all_markets),
                        n_teams=len(all_teams),
                        composition_metrics={
                            "early_master_slip_id": early_candidate.master_slip_id,
                            "late_master_slip_id": selected_late[0].master_slip_id if selected_late else 0,
                            "time_gap_minutes": (latest_late_kickoff - early_kickoff).total_seconds() / 60,
                            "has_valid_time_gap": True,
                            "phase_structure": "1_MID_+" + f"{n_late}_LOW",
                            "odds_compliance": "STRICT"
                        }
                    )
                    
                    compositions.append(composition)
        
        logger.info("[COMPOSE] [4] Generated %d phase compositions", len(compositions))
        return compositions
    
    # ========================================================================
    # STAGE 5: Constraint Enforcement
    # ========================================================================
    
    def _enforce_all_constraints(
        self,
        compositions: List[PhaseComposition]
    ) -> List[PhaseComposition]:
        """
        Filter compositions that violate hard constraints.
        
        Constraints:
          1. Match count: 2 <= total_matches <= 4 (1 early + 1-3 late)
          2. League diversity: n_leagues >= min_leagues
          3. Market diversity: n_markets >= min_markets
          4. Correlation limit: no market type >correlation_limit% of slip
          5. Time gap: already enforced in construction
        """
        valid = []
        min_matches = self.targets.get("min_matches", 2)
        max_matches = self.targets.get("max_matches", 4)
        
        for comp in compositions:
            if self._satisfies_all_constraints(comp, min_matches, max_matches):
                valid.append(comp)
        
        return valid
    
    def _satisfies_all_constraints(
        self,
        comp: PhaseComposition,
        min_matches: int,
        max_matches: int
    ) -> bool:
        """Check all hard constraints for a single composition"""
        total_matches = 1 + len(comp.late_legs)
        
        # 1. Match count bounds
        if not (min_matches <= total_matches <= max_matches):
            return False
        
        # 2. League diversity
        min_leagues = self.constraints_cfg.get("diversity", {}).get("min_leagues", 2)
        if comp.n_leagues < min_leagues:
            return False
        
        # 3. Market diversity
        min_markets = self.constraints_cfg.get("diversity", {}).get("min_markets", 2)
        if comp.n_markets < min_markets:
            return False
        
        # 4. Correlation limit
        correlation_limit = float(self.constraints_cfg.get("correlation_limit", 0.5))
        market_counts = Counter()
        market_counts[comp.early_leg.market] += 1
        for late in comp.late_legs:
            market_counts[late.market] += 1
        
        total_legs = 1 + len(comp.late_legs)
        max_market_freq = max(market_counts.values()) / total_legs
        if max_market_freq > correlation_limit:
            return False
        
        return True
    
    # ========================================================================
    # STAGE 6: Portfolio Diversity Optimization
    # ========================================================================
    
    def _optimize_portfolio_diversity(
        self,
        compositions: List[PhaseComposition]
    ) -> List[PhaseComposition]:
        """
        Select diverse slips from candidates using DNA distance.
        
        Algorithm:
        1. Start with highest-odds slips
        2. Iteratively add slips with maximum diversity distance
        3. Reject slips too similar to portfolio
        """
        if not compositions:
            return []
        
        # Sort by total odds descending
        compositions.sort(key=lambda c: c.total_odds, reverse=True)
        
        selected = []
        selected_dnas = []
        min_distance = float(self.portfolio_cfg.get("min_diversity_distance", 0.30))
        target_count = self.targets.get("count", 50)
        
        for comp in compositions:
            if len(selected) >= target_count:
                break
            
            # Build DNA for this composition
            dna = self._build_dna_from_phase_composition(comp)
            
            # First composition always selected
            if not selected:
                selected.append(comp)
                selected_dnas.append(dna)
                continue
            
            # Calculate minimum distance to existing portfolio
            distances = [dna.distance(existing_dna) for existing_dna in selected_dnas]
            min_dist = min(distances) if distances else 1.0
            
            # Accept if sufficiently diverse
            if min_dist >= min_distance:
                selected.append(comp)
                selected_dnas.append(dna)
        
        # If we don't have enough diverse slips, fill from remaining
        if len(selected) < target_count:
            remaining = [c for c in compositions if c not in selected]
            needed = target_count - len(selected)
            selected.extend(remaining[:needed])
        
        # Calculate average diversity distance
        if len(selected_dnas) > 1:
            distances = []
            for i, dna_a in enumerate(selected_dnas):
                for dna_b in selected_dnas[i+1:]:
                    distances.append(dna_a.distance(dna_b))
            self.stats["portfolio_diversity_avg"] = sum(distances) / len(distances) if distances else 0.0
        
        return selected
    
    def _build_dna_from_phase_composition(self, comp: PhaseComposition) -> CompositionDNA:
        """Build CompositionDNA from PhaseComposition for diversity comparison"""
        selections = {}
        
        # Add early leg
        early_key = (comp.early_leg.match_id, comp.early_leg.market)
        selections[early_key] = {
            "match_id": comp.early_leg.match_id,
            "market": comp.early_leg.market,
            "selection": comp.early_leg.selection,
            "league": comp.early_leg.league,
            "home_team": comp.early_leg.home_team,
            "away_team": comp.early_leg.away_team,
            "odds": comp.early_leg.odds,
            "kickoff_time": comp.early_leg.kickoff_time,
            "phase": "early"
        }
        
        # Add late legs
        for i, late in enumerate(comp.late_legs):
            late_key = (late.match_id, late.market)
            selections[late_key] = {
                "match_id": late.match_id,
                "market": late.market,
                "selection": late.selection,
                "league": late.league,
                "home_team": late.home_team,
                "away_team": late.away_team,
                "odds": late.odds,
                "kickoff_time": late.kickoff_time,
                "phase": "late"
            }
        
        return CompositionDNA(selections=selections)
    
    # ========================================================================
    # STAGE 7: Deduplication
    # ========================================================================
    
    def _deduplicate_by_dna(self, compositions: List[PhaseComposition]) -> List[PhaseComposition]:
        """Remove compositions with identical genetic signatures"""
        seen_dnas = set()
        unique = []
        
        for comp in compositions:
            # Build signature
            signature = frozenset(
                [(comp.early_leg.match_id, comp.early_leg.market, comp.early_leg.selection)] +
                [(late.match_id, late.market, late.selection) for late in comp.late_legs]
            )
            
            if signature in seen_dnas:
                continue
            
            seen_dnas.add(signature)
            unique.append(comp)
        
        return unique
    
    # ========================================================================
    # STAGE 8: Re-Scoring with SlipScorer
    # ========================================================================
    
    def _rescore_compositions(
        self,
        compositions: List[PhaseComposition]
    ) -> List[Dict[str, Any]]:
        """
        Re-score all compositions using existing SlipScorer.
        
        CRITICAL: Do not invent scoring. Call scorer.
        """
        scored = []
        
        for i, comp in enumerate(compositions):
            try:
                # Build legs list from phase composition
                legs = []
                
                # Add early leg
                early_leg_dict = deepcopy(comp.early_leg.leg_data)
                legs.append(early_leg_dict)
                
                # Add late legs
                for late in comp.late_legs:
                    late_leg_dict = deepcopy(late.leg_data)
                    legs.append(late_leg_dict)
                
                # Call scorer on merged legs
                score_result = self.scorer.score(legs)
                
                # Assemble final slip with metadata
                slip = {
                    "slip_id": f"COMP_{i+1:04d}",
                    "legs": score_result.get("legs", legs),
                    "total_odds": float(score_result.get("total_odds", comp.total_odds)),
                    "confidence_score": float(score_result.get("confidence_score", 0.5)),
                    "coverage_score": float(score_result.get("coverage_score", 0.0)),
                    "diversity_score": float(score_result.get("diversity_score", 0.0)),
                    "risk_level": score_result.get("risk_level", "medium"),
                    "fitness_score": float(score_result.get("fitness_score", 0.0)),
                    # Composition metadata
                    "parent_ids": [comp.early_slip_id] + comp.late_slip_ids,
                    "parent_confidences": [comp.early_leg.confidence_score] + 
                                         [late.confidence_score for late in comp.late_legs],
                    "parent_odds": [comp.early_leg.odds] + [late.odds for late in comp.late_legs],
                    "parent_risks": [comp.early_leg.risk_level] + [late.risk_level for late in comp.late_legs],
                    "composition_metrics": {
                        "n_matches": 1 + len(comp.late_legs),
                        "n_leagues": comp.n_leagues,
                        "n_markets": comp.n_markets,
                        "n_unique_selections": 1 + len(comp.late_legs),
                        "overlap_ratio": 0.0,  # No overlap by construction
                        "merge_conflicts_resolved": 0,
                        "hedge_drops": 0,
                        "phase_structure": comp.composition_metrics.get("phase_structure", ""),
                        "time_gap_minutes": comp.composition_metrics.get("time_gap_minutes", 0),
                        "early_master_slip_id": comp.composition_metrics.get("early_master_slip_id", 0),
                        "late_master_slip_id": comp.composition_metrics.get("late_master_slip_id", 0),
                        "odds_compliance": comp.composition_metrics.get("odds_compliance", "")
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
        Select top N slips based on ranking criteria.
        
        Ranking:
          1. Fitness score (if available)
          2. Confidence score (desc)
          3. Coverage score (desc)
          4. Diversity score (desc)
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
                -s.get("diversity_score", 0),
            )
        )
        
        # Enforce target count
        target_count = int(self.targets.get("count", 50))
        final = scored_slips[:target_count]
        
        if len(final) < target_count:
            logger.warning(
                "[COMPOSE] Requested %d slips but only %d valid compositions available",
                target_count, len(final)
            )
        
        return final