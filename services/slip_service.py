"""
Slip Generation Service

Handles the core business logic for slip generation, including:
- Strategy selection and validation
- Builder creation and execution
- Result processing and enrichment
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import DEFAULT_STRATEGY, DEFAULT_SLIP_COUNT

# Import exceptions - try engine first, fallback to our exceptions
try:
    from ..engine import (
        SlipBuilderError,
        PayloadValidationError,
    )
except ImportError:
    from ..exceptions import (
        SlipBuilderError,
        PayloadValidationError,
    )

from ..exceptions import (
    StrategyError,
    MatchCountError,
)

logger = logging.getLogger("engine_api.services")


class SlipService:
    """Service for generating betting slips."""

    def __init__(self):
        self._strategy_factory_available = False
        self._initialize_strategy_factory()

    def _initialize_strategy_factory(self):
        """Initialize strategy factory if available."""
        try:
            from ..engine.slip_builder_factory import (
                create_slip_builder,
                get_available_strategies,
                get_strategy_info,
                validate_strategy,
                validate_match_count_for_strategy
            )
            self._create_slip_builder = create_slip_builder
            self._get_available_strategies = get_available_strategies
            self._get_strategy_info = get_strategy_info
            self._validate_strategy = validate_strategy
            self._validate_match_count = validate_match_count_for_strategy
            self._strategy_factory_available = True
            logger.info("[SERVICE] Strategy factory initialized")
        except ImportError as e:
            logger.warning(f"[SERVICE] Strategy factory not available: {e}")
            self._strategy_factory_available = False

    # ------------------------------------------------------------------
    # Internal: single normalisation gate for every strategy string.
    # Every method that receives a strategy value calls this once before
    # passing it downstream.  Nothing else in this file sanitises strings.
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_strategy(raw: Any, request_id: str = "unknown") -> str:
        """
        Strip whitespace / control chars (\r\n\t etc.) and lower-case.
        This is the only place strategy sanitisation happens in the service.
        """
        normalized = str(raw).strip().lower()

        # Single boundary-level log — makes invisible-character issues
        # visible in traces without spamming on every healthy request.
        if normalized != str(raw):
            logger.debug(
                f"[{request_id}] [SERVICE] Strategy normalised: "
                f"raw={repr(raw)} → normalized={repr(normalized)}"
            )

        return normalized

    # ------------------------------------------------------------------
    # extract_strategy
    # ------------------------------------------------------------------
    def extract_strategy(self, payload: Dict[str, Any]) -> str:
        """
        Extract and validate strategy from payload.

        Args:
            payload: Request payload containing master_slip

        Returns:
            Validated, normalised strategy name

        Raises:
            StrategyError: If strategy is invalid
        """
        master_slip = payload.get("master_slip", {})

        # Normalise at the service boundary — strips \r\n, whitespace, lowercases.
        # Everything downstream sees the clean value.
        requested_strategy = self._normalize_strategy(
            master_slip.get("strategy", DEFAULT_STRATEGY)
        )

        if not self._strategy_factory_available:
            if requested_strategy != DEFAULT_STRATEGY:
                logger.warning(
                    f"[SERVICE] Strategy '{requested_strategy}' requested "
                    f"but factory not available - using '{DEFAULT_STRATEGY}'"
                )
            return DEFAULT_STRATEGY

        # Validate the now-clean strategy key against the factory registry
        if not self._validate_strategy(requested_strategy):
            available = self.get_available_strategies()
            logger.warning(
                f"[SERVICE] Unknown strategy '{requested_strategy}'. "
                f"Available: {available}. Defaulting to '{DEFAULT_STRATEGY}'"
            )
            return DEFAULT_STRATEGY

        return requested_strategy

    # ------------------------------------------------------------------
    # validate_match_count
    # ------------------------------------------------------------------
    def validate_match_count(
        self,
        strategy: str,
        match_count: int,
        request_id: str = "unknown"
    ) -> None:
        """
        Validate match count for selected strategy.

        Args:
            strategy: Strategy name (will be normalised defensively)
            match_count: Number of matches
            request_id: Request identifier for logging

        Raises:
            MatchCountError: If match count is insufficient
        """
        # Defensive normalise — even though generate_slips already normalised,
        # this method is part of the public interface and may be called
        # independently.  Idempotent on a clean string, costs nothing.
        strategy = self._normalize_strategy(strategy, request_id)

        if not self._strategy_factory_available:
            # Legacy mode - minimum 3 matches
            if match_count < 3:
                raise MatchCountError(
                    strategy="balanced",
                    required=3,
                    actual=match_count
                )
            return

        # Factory lookup — strategy is guaranteed clean at this point
        validation = self._validate_match_count(strategy, match_count)

        if not validation["valid"]:
            raise MatchCountError(
                strategy=strategy,
                required=validation["required"],
                actual=validation["actual"]
            )

        logger.info(
            f"[{request_id}] [SERVICE] Match count validation passed: "
            f"{match_count} matches (required: {validation['required']})"
        )

    # ------------------------------------------------------------------
    # generate_slips
    # ------------------------------------------------------------------
    def generate_slips(
        self,
        payload: Dict[str, Any],
        strategy: Optional[str] = None,
        enable_monte_carlo: bool = True,
        num_simulations: int = 10000,
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Generate betting slips using specified strategy.

        Args:
            payload: Laravel payload with master_slip data
            strategy: Strategy to use (extracted from payload if None)
            enable_monte_carlo: Whether to enable Monte Carlo simulations
            num_simulations: Number of Monte Carlo simulations
            request_id: Request identifier for logging

        Returns:
            Dictionary with generated slips and metadata

        Raises:
            SlipBuilderError: If generation fails
            PayloadValidationError: If payload is invalid
        """
        # Resolve strategy: extract from payload if not passed in, then
        # normalise.  extract_strategy already normalises internally, but if
        # a raw value was passed directly we normalise it here too.
        if strategy is None:
            strategy = self.extract_strategy(payload)
        else:
            strategy = self._normalize_strategy(strategy, request_id)

        # Validate match count (strategy is clean, no KeyError risk)
        master_slip = payload.get("master_slip", {})
        matches = master_slip.get("matches", [])
        match_count = len(matches)

        self.validate_match_count(strategy, match_count, request_id)

        # Create builder
        logger.info(
            f"[{request_id}] [SERVICE] Creating slip builder: {strategy}"
        )

        builder = self._create_slip_builder(
            strategy=strategy,
            enable_monte_carlo=enable_monte_carlo,
            num_simulations=num_simulations
        )

        # Generate slips
        logger.info(
            f"[{request_id}] [SERVICE] Generating slips with {strategy} strategy..."
        )

        start_time = time.time()
        try:
            result = builder.generate(payload)
        except SlipBuilderError as e:
            error_msg = str(e)
            # If compound strategy fails due to EV filtering, suggest fallback
            if strategy == "compound" and "No valid selections" in error_msg:
                logger.warning(
                    f"[{request_id}] [SERVICE] Compound strategy failed: {error_msg}. "
                    f"This may indicate unfavorable odds or insufficient favorable selections."
                )
                # Re-raise with enhanced message
                raise StrategyError(
                    f"Compound strategy failed: {error_msg}. "
                    f"Try using 'balanced' or 'maxwin' strategy instead.",
                    strategy=strategy
                )
            raise
        generation_time = time.time() - start_time

        # Enrich result with metadata
        metadata = result.get("metadata", {})
        metadata.update({
            "strategy_used": strategy,
            "match_count": match_count,
            "generation_time_seconds": round(generation_time, 4),
            "generated_at": datetime.utcnow().isoformat() + "Z"
        })
        result["metadata"] = metadata

        # Log summary
        generated_slips = result.get("generated_slips", [])
        logger.info(
            f"[{request_id}] [SERVICE] Generation complete | "
            f"Strategy: {strategy} | "
            f"Slips: {len(generated_slips)} | "
            f"Time: {generation_time:.4f}s"
        )

        return result

    # ------------------------------------------------------------------
    # enrich_slips_with_stake  (unchanged)
    # ------------------------------------------------------------------
    def enrich_slips_with_stake(
        self,
        slips: list,
        master_stake: float,
        request_id: str = "unknown"
    ) -> list:
        """
        Add stake and possible_return to all slips.

        Args:
            slips: List of slip dictionaries
            master_stake: Default stake amount
            request_id: Request identifier for logging

        Returns:
            List of enriched slips
        """
        enriched_slips = []
        slips_with_stake = 0
        slips_without_stake = 0

        for slip in slips:
            # Add stake
            if "stake" not in slip or slip.get("stake") == 0:
                slip["stake"] = master_stake
                slips_without_stake += 1
            else:
                slips_with_stake += 1

            # Calculate possible_return
            total_odds = slip.get("total_odds", 1.0)
            slip_stake = slip.get("stake", master_stake)
            slip["possible_return"] = round(slip_stake * total_odds, 2)

            enriched_slips.append(slip)

        logger.info(
            f"[{request_id}] [SERVICE] Enriched {len(slips)} slips | "
            f"With stake: {slips_with_stake} | "
            f"Using master stake: {slips_without_stake}"
        )

        return enriched_slips

    # ------------------------------------------------------------------
    # get_strategy_info  (unchanged)
    # ------------------------------------------------------------------
    def get_strategy_info(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available strategies.

        Args:
            strategy: Strategy name (None for all strategies)

        Returns:
            Strategy information dictionary
        """
        if not self._strategy_factory_available:
            return {
                "balanced": {
                    "name": "Balanced Portfolio",
                    "description": "Legacy single-strategy mode",
                    "status": "active"
                }
            }

        # Normalise if a specific strategy was requested
        if strategy is not None:
            strategy = self._normalize_strategy(strategy)

        return self._get_strategy_info(strategy)

    # ------------------------------------------------------------------
    # get_available_strategies  (unchanged)
    # ------------------------------------------------------------------
    def get_available_strategies(self) -> list:
        """Get list of available strategies."""
        if not self._strategy_factory_available:
            return [DEFAULT_STRATEGY]

        return self._get_available_strategies()