"""
COMPOSITION SLIPS SERVICE - Orchestration Layer

Manages end-to-end composition workflow:
  1. Validate payload
  2. Extract base/optimized slips
  3. Initialize builder with scorer
  4. Run composition
  5. Log results
"""

import logging
from typing import Dict, Any, List, Tuple

from builders.composition_slips_builder import CompositionSlipBuilder
from engine.slip_scorer import SlipScorer  # Existing scorer (do not modify)

logger = logging.getLogger(__name__)


class CompositionSlipService:
    """
    Service layer for composition slip generation.
    
    Responsibilities:
      - Validate configuration
      - Initialize builder and scorer
      - Execute composition pipeline
      - Handle errors gracefully
      - Log all operations
    """
    
    def __init__(self):
        """Initialize service (stateless)"""
        pass
    
    def run(
        self,
        payload: Dict[str, Any],
        base_slips: List[Dict[str, Any]],
        optimized_slips: List[Dict[str, Any]]
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Run composition pipeline.
        
        Args:
            payload: Full request payload from Laravel
            base_slips: 50 generated slips from DeterministicSlipGenerator
            optimized_slips: 20 optimized slips from PortfolioOptimizer
        
        Returns:
            (success: bool, slips: List[Dict], message: str)
        """
        master_slip = payload.get("master_slip", {})
        master_slip_id = master_slip.get("master_slip_id", 0)
        
        logger.info("[SERVICE] ========== COMPOSITION SERVICE START ==========")
        logger.info("[SERVICE] Master Slip ID: %d", master_slip_id)
        logger.info("[SERVICE] Base slips: %d, Optimized slips: %d",
                   len(base_slips), len(optimized_slips))
        
        try:
            # ====== VALIDATE PAYLOAD ======
            config = self._validate_and_extract_config(payload)
            logger.info("[SERVICE] Configuration validated")
            
            # ====== INITIALIZE SCORER ======
            scorer = self._initialize_scorer(payload)
            logger.info("[SERVICE] SlipScorer initialized")
            
            # ====== INITIALIZE BUILDER ======
            builder = CompositionSlipBuilder(
                config=config,
                scorer=scorer,
                seed=master_slip_id
            )
            logger.info("[SERVICE] CompositionSlipBuilder initialized")
            
            # ====== RUN COMPOSITION ======
            composed_slips = builder.compose(base_slips, optimized_slips)
            logger.info("[SERVICE] Composition complete: %d slips", len(composed_slips))
            
            # ====== VALIDATION ======
            if not composed_slips:
                msg = "Composition produced zero slips"
                logger.warning("[SERVICE] %s", msg)
                return False, [], msg
            
            logger.info("[SERVICE] ========== COMPOSITION SERVICE COMPLETE ==========")
            return True, composed_slips, "Success"
        
        except ValueError as e:
            msg = f"Configuration error: {str(e)}"
            logger.error("[SERVICE] %s", msg)
            return False, [], msg
        
        except Exception as e:
            msg = f"Composition failed: {str(e)}"
            logger.error("[SERVICE] %s", exc_info=True)
            return False, [], msg
    
    def _validate_and_extract_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate composition config from payload.
        
        Raises:
            ValueError: If config is invalid or missing required fields
        """
        master_slip = payload.get("master_slip", {})
        composition_cfg = master_slip.get("composition_slips")
        
        if not composition_cfg:
            raise ValueError("composition_slips config missing from payload")
        
        if not composition_cfg.get("enabled", False):
            raise ValueError("composition_slips.enabled is not true")
        
        # Validate required sections
        required_sections = ["source", "targets", "pairing", "merge_rules", "constraints", "scoring", "determinism"]
        for section in required_sections:
            if section not in composition_cfg:
                raise ValueError(f"composition_slips.{section} is required")
        
        # Validate source
        source_from = composition_cfg.get("source", {}).get("from", [])
        if not isinstance(source_from, list) or not source_from:
            raise ValueError("composition_slips.source.from must be non-empty list")
        
        valid_sources = {"base_slips", "optimized_slips"}
        if not set(source_from) <= valid_sources:
            raise ValueError(f"composition_slips.source.from must be subset of {valid_sources}")
        
        # Validate targets
        targets = composition_cfg.get("targets", {})
        if not isinstance(targets.get("count"), int) or targets["count"] <= 0:
            raise ValueError("composition_slips.targets.count must be positive integer")
        
        min_matches = targets.get("min_matches", 3)
        max_matches = targets.get("max_matches", 20)
        if min_matches > max_matches:
            raise ValueError("min_matches cannot exceed max_matches")
        
        # Validate constraints
        constraints = composition_cfg.get("constraints", {})
        risk_bounds = constraints.get("risk_bounds", {})
        if not risk_bounds.get("min") or not risk_bounds.get("max"):
            raise ValueError("risk_bounds must specify min and max")
        
        diversity = constraints.get("diversity", {})
        if not isinstance(diversity.get("min_leagues"), int) or diversity["min_leagues"] < 1:
            raise ValueError("diversity.min_leagues must be positive integer")
        
        if not isinstance(diversity.get("min_markets"), int) or diversity["min_markets"] < 1:
            raise ValueError("diversity.min_markets must be positive integer")
        
        logger.debug("[SERVICE] Configuration validated successfully")
        return composition_cfg
    
    def _initialize_scorer(self, payload: Dict[str, Any]):
        """
        Initialize SlipScorer with configuration from payload.
        
        CRITICAL: Use your existing SlipScorer. Do not invent scoring logic.
        """
        master_slip = payload.get("master_slip", {})
        
        # Initialize with existing SlipScorer (adjust constructor as needed)
        # This is a placeholder - adapt to your actual SlipScorer API
        scorer = SlipScorer(
            enable_monte_carlo=payload.get("enable_monte_carlo", True),
            num_simulations=payload.get("num_simulations", 10000)
        )
        
        return scorer