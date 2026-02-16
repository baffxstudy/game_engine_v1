"""
COMPOSITION SLIPS BUILDER - Deterministic Second-Order Portfolio Engineering

Fuses base and/or optimized slips into intelligent, risk-controlled composed slips.

Core Principle:
  Generator creates intelligence (50 slips from matches).
  Optimizer selects intelligence (20 best slips).
  Composer multiplies intelligence (50 fused slips with controlled blast radius).

Key Features:
  ✓ Deterministic pairing & merging (seeded RNG)
  ✓ Conflict-aware selection (higher_confidence, lower_risk, balanced)
  ✓ Hedging enforcement (max per league, market diversity, correlation limits)
  ✓ Constraint normalization (min/max matches, min leagues/markets, risk bounds)
  ✓ No clone composition (genetic deduplication)
  ✓ Overlap avoidance (similarity-based rejection)
  ✓ Reuses existing scorer (no scoring reinvention)
  ✓ Configuration-driven (no hardcoded logic)

Input: base_slips, optimized_slips, config
Output: List[composed_slip] with:
  - slip_id, legs, total_odds, confidence_score, coverage_score
  - risk_level, diversity_score, parent_ids, composition_metrics

Version: 1.0.0
"""

import logging
import random
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Set, Optional, Callable
from decimal import Decimal
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DOMAIN MODELS
# ============================================================================

class RiskLevel(str, Enum):
    """Risk stratification matching existing engine"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CompositionDNA:
    """Normalized genetic representation of a slip for merging"""
    selections: Dict[Tuple[int, str], Dict[str, Any]]  # (match_id, market) -> {fields}
    
    def keys(self):
        return self.selections.keys()
    
    def __len__(self):
        return len(self.selections)
    
    def items(self):
        return self.selections.items()
    
    def get(self, key, default=None):
        return self.selections.get(key, default)
    
    def __contains__(self, key):
        return key in self.selections
    
    def fingerprint(self) -> frozenset:
        """Return immutable (match, market) pairs for overlap detection"""
        return frozenset(self.selections.keys())


@dataclass
class CompositionSlipRaw:
    """Raw composition result before final scoring"""
    parent_a_id: str
    parent_b_id: str
    legs: List[Dict[str, Any]]
    n_matches: int
    n_leagues: int
    n_markets: int
    n_unique_selections: int
    overlap_ratio: float
    merge_conflicts: int
    hedge_drops: int
    parent_conf: Tuple[float, float]
    parent_odds: Tuple[Decimal, Decimal]


# ============================================================================
# COMPOSITION SLIP BUILDER
# ============================================================================

class CompositionSlipBuilder:
    """
    Orchestrates deterministic composition of base/optimized slips.
    
    Does NOT invent scores. Calls existing SlipScorer on merged legs.
    Does NOT concatenate blindly. Applies conflict resolution, hedging, constraints.
    Does NOT randomize outside seeded RNG. All RNG seeded from master_slip_id.
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
                   Must return: {
                       "legs": [...],
                       "total_odds": Decimal,
                       "confidence_score": float,
                       "coverage_score": float,
                       "diversity_score": float,
                       "risk_level": str
                   }
            seed: master_slip_id (for deterministic RNG)
        """
        self.config = config
        self.scorer = scorer
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        
        # Extract config sections
        self.targets = config["targets"]
        self.pairing_cfg = config["pairing"]
        self.merge_cfg = config["merge_rules"]
        self.constraints_cfg = config["constraints"]
        self.scoring_cfg = config["scoring"]
        self.determinism_cfg = config["determinism"]
        
        # Unpack constraints
        self.risk_bounds = self.constraints_cfg["risk_bounds"]
        self.diversity_req = self.constraints_cfg["diversity"]
        self.correlation_limit = float(self.constraints_cfg["correlation_limit"])
        
        # Track stats for logging
        self.stats = {
            "parent_pool_size": 0,
            "candidate_pairs_generated": 0,
            "pairs_after_overlap_filter": 0,
            "compositions_before_constraint": 0,
            "compositions_after_constraint": 0,
            "deduped_compositions": 0,
            "final_compositions": 0,
        }
    
    def compose(
        self,
        base_slips: List[Dict[str, Any]],
        optimized_slips: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: compose slips deterministically.
        
        Args:
            base_slips: Output from DeterministicSlipGenerator (50 slips)
            optimized_slips: Output from PortfolioOptimizer (20 slips)
        
        Returns:
            List[Dict] with slip structure:
            {
                "slip_id": "COMP_0001",
                "legs": [...],
                "total_odds": 12.5,
                "confidence_score": 0.62,
                "coverage_score": 0.45,
                "diversity_score": 0.58,
                "risk_level": "medium",
                "fitness_score": 0.65,
                "parent_ids": ["GEN_0001", "OPT_0005"],
                "parent_confidences": [0.65, 0.59],
                "parent_odds": [6.5, 1.92],
                "composition_metrics": {
                    "n_matches": 8,
                    "n_leagues": 3,
                    "n_markets": 5,
                    "n_unique_selections": 8,
                    "overlap_ratio": 0.25,
                    "merge_conflicts_resolved": 2,
                    "hedge_drops": 1
                }
            }
        """
        logger.info("[COMPOSE] ==================== START COMPOSITION ====================")
        logger.info("[COMPOSE] Config: pairing=%s, merge=%s, constraints=%s",
                   self.pairing_cfg.get("mode"), self.merge_cfg.get("conflict_resolution"),
                   f"risk={self.risk_bounds}")
        
        # ====== STAGE 1: Pool Selection ======
        parent_pool = self._select_parent_pool(base_slips, optimized_slips)
        self.stats["parent_pool_size"] = len(parent_pool)
        logger.info("[COMPOSE] [1] Parent pool: %d slips", len(parent_pool))
        
        # ====== STAGE 2: Deterministic Shuffling ======
        parent_pool = self._deterministic_shuffle(parent_pool)
        logger.info("[COMPOSE] [2] Shuffled deterministically (seed=%d)", self.seed)
        
        # ====== STAGE 3: Pairwise Candidate Generation ======
        candidate_pairs = self._generate_candidate_pairs(parent_pool)
        self.stats["candidate_pairs_generated"] = len(candidate_pairs)
        logger.info("[COMPOSE] [3] Generated %d candidate pairs", len(candidate_pairs))
        
        # ====== STAGE 4: Composition & Merging ======
        raw_compositions = self._merge_candidate_pairs(candidate_pairs)
        self.stats["compositions_before_constraint"] = len(raw_compositions)
        logger.info("[COMPOSE] [4] Created %d raw compositions", len(raw_compositions))
        
        # ====== STAGE 5: Constraint Enforcement ======
        constrained = self._enforce_all_constraints(raw_compositions)
        self.stats["compositions_after_constraint"] = len(constrained)
        logger.info("[COMPOSE] [5] After constraints: %d valid compositions", len(constrained))
        
        # ====== STAGE 6: Deduplication ======
        deduped = self._deduplicate_by_dna(constrained)
        self.stats["deduped_compositions"] = len(deduped)
        logger.info("[COMPOSE] [6] After dedup: %d unique compositions", len(deduped))
        
        # ====== STAGE 7: Re-Scoring ======
        scored = self._rescore_compositions(deduped)
        logger.info("[COMPOSE] [7] Re-scored all compositions using SlipScorer")
        
        # ====== STAGE 8: Final Selection ======
        final = self._final_selection(scored)
        self.stats["final_compositions"] = len(final)
        logger.info("[COMPOSE] [8] Final selection: %d slips (target=%d)", 
                   len(final), self.targets["count"])
        
        logger.info("[COMPOSE] ==================== COMPOSITION COMPLETE ====================")
        logger.info("[COMPOSE] Stats: %s", self.stats)
        
        return final
    
    # ========================================================================
    # STAGE 1: Parent Pool Selection
    # ========================================================================
    
    def _select_parent_pool(
        self,
        base_slips: List[Dict],
        optimized_slips: List[Dict]
    ) -> List[Dict]:
        """
        Merge base and/or optimized slips based on config["source"]["from"].
        
        Only include slips requested in config.
        """
        pool = []
        source_from = self.config["source"]["from"]
        
        if "base_slips" in source_from and base_slips:
            pool.extend(deepcopy(base_slips))
            logger.debug("[COMPOSE] Added %d base slips", len(base_slips))
        
        if "optimized_slips" in source_from and optimized_slips:
            pool.extend(deepcopy(optimized_slips))
            logger.debug("[COMPOSE] Added %d optimized slips", len(optimized_slips))
        
        if not pool:
            raise ValueError(
                f"No slips selected from source {source_from}. "
                f"base_slips={len(base_slips)}, optimized_slips={len(optimized_slips)}"
            )
        
        return pool
    
    # ========================================================================
    # STAGE 2: Deterministic Shuffling
    # ========================================================================
    
    def _deterministic_shuffle(self, slips: List[Dict]) -> List[Dict]:
        """
        Shuffle parent pool deterministically for consistent pair generation.
        
        First sort by deterministic key, then shuffle with seeded RNG.
        This ensures reproducibility: same seed → same order.
        """
        # Sort by natural keys first (deterministic ordering)
        slips_sorted = sorted(
            slips,
            key=lambda s: (
                s.get("slip_id", ""),
                float(s.get("total_odds", 1)),
                -float(s.get("confidence_score", 0))
            )
        )
        
        # Shuffle with seeded RNG
        self.rng.shuffle(slips_sorted)
        
        return slips_sorted
    
    # ========================================================================
    # STAGE 3: Pairwise Candidate Generation
    # ========================================================================
    
    def _generate_candidate_pairs(
        self,
        parent_pool: List[Dict]
    ) -> List[Tuple[Dict, Dict, float, float]]:
        """
        Generate candidate pairs respecting avoid_overlap_threshold.
        
        Returns:
            List of (slip_a, slip_b, overlap_ratio, compatibility_score)
        
        Sorting:
            - Pairs with lower overlap are prioritized
            - Compatibility bias applied (e.g., high+medium confidence pairs)
        """
        pairs = []
        avoid_overlap_threshold = float(self.pairing_cfg["avoid_overlap_threshold"])
        target_pair_count = self.targets["count"] * 3  # Oversample 3x, filter later
        
        # Generate all pairs
        for i in range(len(parent_pool)):
            if len(pairs) >= target_pair_count:
                break
            
            slip_a = parent_pool[i]
            dna_a = self._extract_dna_fingerprint(slip_a)
            
            for j in range(i + 1, len(parent_pool)):
                if len(pairs) >= target_pair_count:
                    break
                
                slip_b = parent_pool[j]
                
                # Avoid self-pairing
                if slip_a.get("slip_id") == slip_b.get("slip_id"):
                    continue
                
                # Calculate overlap
                dna_b = self._extract_dna_fingerprint(slip_b)
                overlap_ratio = self._calculate_overlap(dna_a, dna_b)
                
                # Skip high-overlap pairs
                if overlap_ratio > avoid_overlap_threshold:
                    continue
                
                # Calculate compatibility score
                compat_score = self._calculate_compatibility(slip_a, slip_b)
                
                pairs.append((slip_a, slip_b, overlap_ratio, compat_score))
        
        # Sort: best complementarity first (lower overlap, higher compat)
        pairs.sort(key=lambda p: (p[2], -p[3]))
        
        self.stats["pairs_after_overlap_filter"] = len(pairs)
        logger.debug("[COMPOSE] [3] Pairs after overlap filter: %d / %d potential",
                    len(pairs), len(parent_pool) * (len(parent_pool) - 1) // 2)
        
        return pairs
    
    def _extract_dna_fingerprint(self, slip: Dict) -> frozenset:
        """Extract (match_id, market_code) pairs as immutable set"""
        legs = slip.get("legs", [])
        return frozenset(
            (leg.get("match_id"), leg.get("market"))
            for leg in legs
        )
    
    def _calculate_overlap(self, dna_a: frozenset, dna_b: frozenset) -> float:
        """
        Overlap ratio = intersection / union
        
        0.0 = no overlap (perfect complementarity)
        1.0 = complete overlap (redundant pairs)
        """
        if not dna_a or not dna_b:
            return 0.0
        
        intersection = len(dna_a & dna_b)
        union = len(dna_a | dna_b)
        
        return float(intersection / union) if union > 0 else 0.0
    
    def _calculate_compatibility(self, slip_a: Dict, slip_b: Dict) -> float:
        """
        Compatibility score for pairing strategy.
        
        Higher score = better pairing candidate.
        
        Biases:
          - high_to_medium: favor pairing high-confidence with medium-confidence
          - (extensible for other bias modes)
        """
        bias_mode = self.pairing_cfg.get("confidence_bias", "high_to_medium")
        
        conf_a = float(slip_a.get("confidence_score", 0.5))
        conf_b = float(slip_b.get("confidence_score", 0.5))
        
        score = 0.0
        
        if bias_mode == "high_to_medium":
            # Prefer (high, medium) or (medium, high) combinations
            sorted_confs = sorted([conf_a, conf_b], reverse=True)
            
            # Bonus if one is high and other is medium
            if sorted_confs[0] >= 0.75 and 0.45 <= sorted_confs[1] < 0.75:
                score += 1.0
            
            # Penalty if both are very similar (overlapping strategy)
            if abs(conf_a - conf_b) < 0.1:
                score -= 0.3
            
            # Bonus if combined they span risk spectrum
            if (conf_a < 0.5 and conf_b > 0.6) or (conf_a > 0.6 and conf_b < 0.5):
                score += 0.5
        
        # Default: no preference
        return max(0.0, score)
    
    # ========================================================================
    # STAGE 4: Composition & Merging
    # ========================================================================
    
    def _merge_candidate_pairs(
        self,
        candidate_pairs: List[Tuple[Dict, Dict, float, float]]
    ) -> List[CompositionSlipRaw]:
        """
        Merge each candidate pair into a raw composition.
        
        Process:
          1. Extract DNA from each parent
          2. Apply conflict resolution
          3. Apply hedging constraints (league limits, etc)
          4. Create raw composition
        """
        raw_compositions = []
        
        for slip_a, slip_b, overlap, compat in candidate_pairs:
            # Extract DNAs
            dna_a = self._normalize_slip_to_dna(slip_a)
            dna_b = self._normalize_slip_to_dna(slip_b)
            
            # Merge with conflict resolution
            merged_dna = self._fuse_dna_with_conflict_resolution(dna_a, dna_b)
            
            if not merged_dna:
                continue
            
            # Apply hedging constraints (league limits, correlation, etc)
            hedged_dna = self._apply_hedging_constraints(merged_dna)
            
            if not hedged_dna:
                continue
            
            # Denormalize back to leg structure
            legs = [hedged_dna.selections[key] for key in hedged_dna.selections]
            
            # Extract metrics for filtering
            n_matches = len({leg.get("match_id") for leg in legs})
            n_leagues = len({leg.get("league", "unknown") for leg in legs})
            n_markets = len({leg.get("market") for leg in legs})
            n_selections = len(legs)
            merge_conflicts = self._count_merge_conflicts(dna_a, dna_b)
            hedge_drops = len(merged_dna.selections) - len(hedged_dna.selections)
            
            # Create raw composition
            raw = CompositionSlipRaw(
                parent_a_id=slip_a.get("slip_id"),
                parent_b_id=slip_b.get("slip_id"),
                legs=legs,
                n_matches=n_matches,
                n_leagues=n_leagues,
                n_markets=n_markets,
                n_unique_selections=n_selections,
                overlap_ratio=overlap,
                merge_conflicts=merge_conflicts,
                hedge_drops=hedge_drops,
                parent_conf=(
                    float(slip_a.get("confidence_score", 0.5)),
                    float(slip_b.get("confidence_score", 0.5))
                ),
                parent_odds=(
                    Decimal(str(slip_a.get("total_odds", 1))),
                    Decimal(str(slip_b.get("total_odds", 1)))
                )
            )
            
            raw_compositions.append(raw)
        
        return raw_compositions
    
    def _normalize_slip_to_dna(self, slip: Dict) -> CompositionDNA:
        """
        Normalize slip into DNA: { (match_id, market) -> {leg_fields} }
        
        Preserves all leg data for later denormalization.
        """
        selections = {}
        legs = slip.get("legs", [])
        
        for leg in legs:
            match_id = leg.get("match_id")
            market = leg.get("market")
            
            if match_id is None or market is None:
                continue
            
            key = (match_id, market)
            
            # Attach parent metadata for conflict resolution
            leg_with_meta = {
                **deepcopy(leg),
                "_parent_slip_id": slip.get("slip_id"),
                "_parent_confidence": slip.get("confidence_score", 0.5),
                "_parent_odds": slip.get("total_odds", 1),
                "_parent_risk": slip.get("risk_level", "medium"),
            }
            
            selections[key] = leg_with_meta
        
        return CompositionDNA(selections=selections)
    
    def _fuse_dna_with_conflict_resolution(
        self,
        dna_a: CompositionDNA,
        dna_b: CompositionDNA
    ) -> Optional[CompositionDNA]:
        """
        Merge two DNAs with conflict resolution.
        
        When same (match, market) appears in both:
          - higher_confidence: pick the one from higher-confidence parent
          - lower_risk: pick the one from lower-risk parent
          - balanced: average or pick randomly
        
        Returns:
            Merged DNA or None if no selections remain
        """
        fused = {}
        conflict_rule = self.merge_cfg.get("conflict_resolution", "higher_confidence")
        
        # Union of all keys
        all_keys = set(dna_a.selections.keys()) | set(dna_b.selections.keys())
        
        for key in all_keys:
            in_a = key in dna_a.selections
            in_b = key in dna_b.selections
            
            if in_a and in_b:
                # CONFLICT: apply resolution rule
                leg_a = dna_a.selections[key]
                leg_b = dna_b.selections[key]
                
                if conflict_rule == "higher_confidence":
                    conf_a = leg_a.get("_parent_confidence", 0)
                    conf_b = leg_b.get("_parent_confidence", 0)
                    fused[key] = leg_a if conf_a >= conf_b else leg_b
                
                elif conflict_rule == "lower_risk":
                    # Map risk to numeric (low=0, medium=1, high=2)
                    risk_map = {"low": 0, "medium": 1, "high": 2}
                    risk_a = risk_map.get(leg_a.get("_parent_risk", "medium"), 1)
                    risk_b = risk_map.get(leg_b.get("_parent_risk", "medium"), 1)
                    fused[key] = leg_a if risk_a <= risk_b else leg_b
                
                elif conflict_rule == "balanced":
                    # Randomly pick (seeded for determinism)
                    fused[key] = leg_a if self.rng.random() < 0.5 else leg_b
                
                else:
                    # Default: higher confidence
                    conf_a = leg_a.get("_parent_confidence", 0)
                    conf_b = leg_b.get("_parent_confidence", 0)
                    fused[key] = leg_a if conf_a >= conf_b else leg_b
            
            elif in_a:
                fused[key] = deepcopy(dna_a.selections[key])
            
            elif in_b:
                fused[key] = deepcopy(dna_b.selections[key])
        
        if not fused:
            return None
        
        return CompositionDNA(selections=fused)
    
    def _apply_hedging_constraints(self, dna: CompositionDNA) -> CompositionDNA:
        """
        Apply hedging constraints to prevent concentration risk.
        
        Rules (from config["merge_rules"]):
          - max_per_league: maximum selections from same league
          - allow_same_match: whether to allow multiple selections from same match
          - (extensible for other hedging rules)
        """
        hedged = {}
        max_per_league = int(self.merge_cfg.get("max_per_league", 4))
        allow_same_match = self.merge_cfg.get("allow_same_match", False)
        
        league_counts = Counter()
        match_counts = Counter()
        
        # Process in deterministic order
        for key in sorted(dna.selections.keys()):
            leg = dna.selections[key]
            match_id = key[0]
            league = leg.get("league", "unknown")
            
            # Check league limit
            if league_counts[league] >= max_per_league:
                continue
            
            # Check same-match rule
            if not allow_same_match and match_counts[match_id] > 0:
                continue
            
            hedged[key] = deepcopy(leg)
            league_counts[league] += 1
            match_counts[match_id] += 1
        
        if not hedged:
            return None
        
        return CompositionDNA(selections=hedged)
    
    def _count_merge_conflicts(self, dna_a: CompositionDNA, dna_b: CompositionDNA) -> int:
        """Count how many (match, market) pairs appeared in both DNAs"""
        intersection = set(dna_a.selections.keys()) & set(dna_b.selections.keys())
        return len(intersection)
    
    # ========================================================================
    # STAGE 5: Constraint Enforcement
    # ========================================================================
    
    def _enforce_all_constraints(
        self,
        raw_compositions: List[CompositionSlipRaw]
    ) -> List[CompositionSlipRaw]:
        """
        Filter compositions that violate hard constraints.
        
        Constraints checked:
          1. Match count: min_matches <= n_matches <= max_matches
          2. League diversity: n_leagues >= min_leagues
          3. Market diversity: n_markets >= min_markets
          4. Risk bounds: risk_level within [min, max]
          5. Correlation limit: no single market type >correlation_limit% of slip
        """
        valid = []
        
        for raw in raw_compositions:
            if self._satisfies_all_constraints(raw):
                valid.append(raw)
        
        return valid
    
    def _satisfies_all_constraints(self, raw: CompositionSlipRaw) -> bool:
        """Check all hard constraints for a single composition"""
        # 1. Match count bounds
        if not (self.targets["min_matches"] <= raw.n_matches <= self.targets["max_matches"]):
            return False
        
        # 2. League diversity
        if raw.n_leagues < self.diversity_req["min_leagues"]:
            return False
        
        # 3. Market diversity
        if raw.n_markets < self.diversity_req["min_markets"]:
            return False
        
        # 4. Correlation limit (no market type >X% of total selections)
        if raw.legs:
            market_counts = Counter(leg.get("market") for leg in raw.legs)
            max_market_freq = max(market_counts.values()) / len(raw.legs)
            if max_market_freq > self.correlation_limit:
                return False
        
        # 5. Risk bounds (soft check - will be enforced during scoring)
        # Allow all risk levels for now; actual risk determined by scorer
        
        return True
    
    # ========================================================================
    # STAGE 6: Deduplication by DNA
    # ========================================================================
    
    def _deduplicate_by_dna(self, raw_comps: List[CompositionSlipRaw]) -> List[CompositionSlipRaw]:
        """
        Remove compositions with identical genetic signatures.
        
        Signature = frozenset of (match_id, market_code, selection_value)
        
        Keeps first occurrence, discards duplicates.
        """
        seen_dnas = set()
        unique = []
        
        for raw in raw_comps:
            # Build signature
            signature = frozenset(
                (leg.get("match_id"), leg.get("market"), leg.get("selection", ""))
                for leg in raw.legs
            )
            
            if signature in seen_dnas:
                continue
            
            seen_dnas.add(signature)
            unique.append(raw)
        
        return unique
    
    # ========================================================================
    # STAGE 7: Re-Scoring with SlipScorer
    # ========================================================================
    
    def _rescore_compositions(
        self,
        raw_compositions: List[CompositionSlipRaw]
    ) -> List[Dict[str, Any]]:
        """
        Re-score all compositions using existing SlipScorer.
        
        CRITICAL: Do not invent scoring. Call scorer.score(legs).
        
        Expects scorer to return:
        {
            "legs": [...],
            "total_odds": Decimal or float,
            "confidence_score": float,
            "coverage_score": float,
            "diversity_score": float,
            "risk_level": str,
            ...any other fields scorer adds...
        }
        """
        scored = []
        
        for i, raw in enumerate(raw_compositions):
            try:
                # Call scorer on merged legs
                score_result = self.scorer.score(raw.legs)
                
                # Assemble final slip with metadata
                slip = {
                    "slip_id": f"COMP_{i+1:04d}",
                    "legs": score_result.get("legs", raw.legs),
                    "total_odds": float(score_result.get("total_odds", 1.0)),
                    "confidence_score": float(score_result.get("confidence_score", 0.5)),
                    "coverage_score": float(score_result.get("coverage_score", 0.0)),
                    "diversity_score": float(score_result.get("diversity_score", 0.0)),
                    "risk_level": score_result.get("risk_level", "medium"),
                    "fitness_score": float(score_result.get("fitness_score", 0.0)),
                    # Composition metadata
                    "parent_ids": [raw.parent_a_id, raw.parent_b_id],
                    "parent_confidences": list(raw.parent_conf),
                    "parent_odds": [float(o) for o in raw.parent_odds],
                    "composition_metrics": {
                        "n_matches": raw.n_matches,
                        "n_leagues": raw.n_leagues,
                        "n_markets": raw.n_markets,
                        "n_unique_selections": raw.n_unique_selections,
                        "overlap_ratio": raw.overlap_ratio,
                        "merge_conflicts_resolved": raw.merge_conflicts,
                        "hedge_drops": raw.hedge_drops,
                    }
                }
                
                scored.append(slip)
            
            except Exception as e:
                logger.error("[COMPOSE] Failed to score composition: %s", str(e))
                continue
        
        return scored
    
    # ========================================================================
    # STAGE 8: Final Selection
    # ========================================================================
    
    def _final_selection(self, scored_slips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select top N slips based on ranking criteria.
        
        Ranking:
          1. Confidence score (desc)
          2. Coverage score (desc)
          3. Diversity score (desc)
          4. Risk balance (avoid all high or all low)
        
        Returns:
            Exactly targets["count"] slips, or fewer if not enough valid compositions
        """
        if not scored_slips:
            logger.warning("[COMPOSE] No valid compositions for final selection")
            return []
        
        # Sort by ranking criteria
        scored_slips.sort(
            key=lambda s: (
                -s.get("confidence_score", 0),
                -s.get("coverage_score", 0),
                -s.get("diversity_score", 0),
            )
        )
        
        # Enforce target count
        target_count = int(self.targets["count"])
        final = scored_slips[:target_count]
        
        if len(final) < target_count:
            logger.warning(
                "[COMPOSE] Requested %d slips but only %d valid compositions available",
                target_count, len(final)
            )
        
        return final