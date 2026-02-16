"""
Validation Service

Handles input validation and payload sanitization:
- Payload structure validation
- Data type validation
- Business rule validation
"""

import logging
from typing import Dict, Any, Optional

from ..config import (
    MIN_STAKE,
    MAX_STAKE,
    MAX_MATCHES,
    SUPPORTED_STRATEGIES
)
from ..exceptions import PayloadValidationError

logger = logging.getLogger("engine_api.services")


class ValidationService:
    """Service for validating inputs and payloads."""
    
    def validate_master_slip_payload(
        self,
        payload: Dict[str, Any],
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Validate master slip payload structure.
        
        Args:
            payload: Request payload
            request_id: Request identifier for logging
            
        Returns:
            Normalized payload dictionary
            
        Raises:
            PayloadValidationError: If validation fails
        """
        if not payload:
            raise PayloadValidationError(
                "Empty request payload",
                field="payload"
            )
        
        # Handle Pydantic models
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif hasattr(payload, "dict") and not isinstance(payload, dict):
            payload = payload.dict()
        
        # Validate master_slip structure
        if "master_slip" not in payload:
            raise PayloadValidationError(
                "Missing 'master_slip' in payload",
                field="master_slip"
            )
        
        master_slip = payload["master_slip"]
        
        # Validate master_slip_id
        if "master_slip_id" not in master_slip:
            raise PayloadValidationError(
                "Missing 'master_slip_id' in master_slip",
                field="master_slip.master_slip_id"
            )
        
        master_slip_id = master_slip.get("master_slip_id")
        if not isinstance(master_slip_id, (int, str)):
            raise PayloadValidationError(
                f"Invalid 'master_slip_id' type: {type(master_slip_id)}",
                field="master_slip.master_slip_id"
            )
        
        # Validate matches
        if "matches" not in master_slip:
            raise PayloadValidationError(
                "Missing 'matches' in master_slip",
                field="master_slip.matches"
            )
        
        matches = master_slip.get("matches", [])
        if not isinstance(matches, list):
            raise PayloadValidationError(
                f"Invalid 'matches' type: {type(matches)}",
                field="master_slip.matches"
            )
        
        if len(matches) == 0:
            raise PayloadValidationError(
                "Empty matches list",
                field="master_slip.matches"
            )
        
        if len(matches) > MAX_MATCHES:
            raise PayloadValidationError(
                f"Too many matches: {len(matches)} (max: {MAX_MATCHES})",
                field="master_slip.matches"
            )
        
        # Validate stake if present
        stake = master_slip.get("stake", 0)
        if stake is not None:
            try:
                stake = float(stake)
                if stake < MIN_STAKE:
                    raise PayloadValidationError(
                        f"Stake too low: {stake} (min: {MIN_STAKE})",
                        field="master_slip.stake"
                    )
                if stake > MAX_STAKE:
                    raise PayloadValidationError(
                        f"Stake too high: {stake} (max: {MAX_STAKE})",
                        field="master_slip.stake"
                    )
            except (ValueError, TypeError):
                raise PayloadValidationError(
                    f"Invalid stake value: {stake}",
                    field="master_slip.stake"
                )
        
        # Validate strategy if present
        strategy = master_slip.get("strategy")
        if strategy is not None:
            if not isinstance(strategy, str):
                raise PayloadValidationError(
                    f"Invalid strategy type: {type(strategy)}",
                    field="master_slip.strategy"
                )
            if strategy.lower() not in SUPPORTED_STRATEGIES:
                logger.warning(
                    f"[{request_id}] [VALIDATION] Unsupported strategy: {strategy}"
                )
        
        logger.debug(
            f"[{request_id}] [VALIDATION] Payload validation passed | "
            f"Master Slip ID: {master_slip_id} | "
            f"Matches: {len(matches)}"
        )
        
        return payload
    
    def validate_match_structure(
        self,
        match: Dict[str, Any],
        match_index: int
    ) -> None:
        """
        Validate individual match structure.
        
        Args:
            match: Match dictionary
            match_index: Index of match in list
            
        Raises:
            PayloadValidationError: If validation fails
        """
        if not isinstance(match, dict):
            raise PayloadValidationError(
                f"Match at index {match_index} is not a dictionary",
                field=f"matches[{match_index}]"
            )
        
        # Check for match identifier
        if "match_id" not in match and "id" not in match:
            raise PayloadValidationError(
                f"Match at index {match_index} missing identifier",
                field=f"matches[{match_index}].match_id"
            )
        
        # Check for teams
        if "home_team" not in match and "homeTeam" not in match:
            raise PayloadValidationError(
                f"Match at index {match_index} missing home team",
                field=f"matches[{match_index}].home_team"
            )
        
        if "away_team" not in match and "awayTeam" not in match:
            raise PayloadValidationError(
                f"Match at index {match_index} missing away team",
                field=f"matches[{match_index}].away_team"
            )
        
        # Check for markets
        markets = match.get("match_markets") or match.get("markets", [])
        if not isinstance(markets, list):
            raise PayloadValidationError(
                f"Match at index {match_index} has invalid markets",
                field=f"matches[{match_index}].markets"
            )
        
        if len(markets) == 0:
            logger.warning(
                f"[VALIDATION] Match at index {match_index} has no markets"
            )
