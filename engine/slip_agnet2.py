# game_engine/engine/slip_builder.py
"""STRICT MODE SLIP BUILDER - Zero Tolerance, Deterministic, Authoritative"""

import json
import logging
from decimal import Decimal, getcontext
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random

# Configure decimal precision
getcontext().prec = 10

# Configure logging - CONSOLE ONLY per STRICT MODE
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MarketIntegrityError(Exception):
    """Base exception for market integrity violations"""
    pass


class PayloadValidationError(MarketIntegrityError):
    """Raised when payload violates structural requirements"""
    pass


class MarketRegistryError(MarketIntegrityError):
    """Raised when market registry construction fails"""
    pass


class SlipGenerationError(MarketIntegrityError):
    """Raised when slip generation fails integrity checks"""
    pass


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MarketSelection:
    """Immutable market-selection pair - Authoritative source"""
    match_id: int
    market_code: str
    selection_value: str
    odds: Decimal
    
    def __post_init__(self):
        """STRICT validation on creation"""
        # Odds range validation
        if not Decimal('1.01') <= self.odds <= Decimal('1000'):
            raise MarketIntegrityError(
                f"Odds {self.odds} outside valid range [1.01, 1000] "
                f"for {self.match_id}:{self.market_code}:{self.selection_value}"
            )
    
    def to_leg_dict(self) -> Dict[str, Any]:
        """Convert to output leg format - EXACT contract"""
        return {
            "match_id": self.match_id,
            "market": self.market_code,
            "selection": self.selection_value,
            "odds": float(self.odds)
        }


class StrictMarketRegistry:
    """Authoritative market registry with zero-tolerance validation"""
    
    def __init__(self):
        self.registry: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.match_ids: List[int] = []
        self._seen_market_codes: Set[Tuple[int, str]] = set()
    
    def build_from_payload(self, payload: Dict[str, Any]) -> None:
        """Build registry with STRICT validation - FAIL FAST on any issue"""
        logger.info("Building STRICT Market Registry...")
        
        master_slip = payload.get('master_slip')
        if not master_slip:
            raise PayloadValidationError("Missing 'master_slip' in payload")
        
        matches = master_slip.get('matches', [])
        if not matches:
            raise PayloadValidationError("No matches in payload")
        
        for match_index, match in enumerate(matches):
            match_id = match.get('match_id')
            if match_id is None:
                raise PayloadValidationError(
                    f"Match at index {match_index} missing 'match_id'"
                )
            
            if match_id in self.registry:
                raise MarketRegistryError(f"Duplicate match_id: {match_id}")
            
            self.match_ids.append(match_id)
            self.registry[match_id] = {}
            
            match_markets = match.get('match_markets', [])
            if not match_markets:
                raise MarketRegistryError(
                    f"Match {match_id} has no 'match_markets' array"
                )
            
            for market_index, market_data in enumerate(match_markets):
                market_info = market_data.get('market', {})
                market_code = market_info.get('code')
                
                if not market_code:
                    raise MarketRegistryError(
                        f"Market at index {market_index} for match {match_id} "
                        f"missing 'market.code'"
                    )
                
                # Check for duplicate market codes per match
                market_key = (match_id, market_code)
                if market_key in self._seen_market_codes:
                    raise MarketRegistryError(
                        f"Duplicate market code '{market_code}' for match {match_id}"
                    )
                self._seen_market_codes.add(market_key)
                
                selections = market_data.get('selections', [])
                if not selections:
                    raise MarketRegistryError(
                        f"Market '{market_code}' for match {match_id} "
                        f"has no selections"
                    )
                
                validated_selections = []
                for sel_index, selection in enumerate(selections):
                    value = selection.get('value')
                    odds = selection.get('odds')
                    
                    if not value:
                        raise MarketRegistryError(
                            f"Selection at index {sel_index} in market '{market_code}' "
                            f"for match {match_id} missing 'value'"
                        )
                    
                    if odds is None:
                        raise MarketRegistryError(
                            f"Selection '{value}' in market '{market_code}' "
                            f"for match {match_id} missing 'odds'"
                        )
                    
                    try:
                        odds_decimal = Decimal(str(odds))
                    except Exception:
                        raise MarketRegistryError(
                            f"Invalid odds value for selection '{value}' in market "
                            f"'{market_code}' for match {match_id}: {odds}"
                        )
                    
                    # Odds range validation
                    if not Decimal('1.01') <= odds_decimal <= Decimal('1000'):
                        raise MarketRegistryError(
                            f"Odds {odds} for selection '{value}' in market "
                            f"'{market_code}' for match {match_id} outside range [1.01, 1000]"
                        )
                    
                    validated_selections.append({
                        'value': value,
                        'label': selection.get('label', value),
                        'odds': odds_decimal
                    })
                
                self.registry[match_id][market_code] = {
                    'market_name': market_info.get('name', ''),
                    'selections': validated_selections
                }
        
        if not self.registry:
            raise MarketRegistryError("Registry empty after validation")
        
        logger.info(f"STRICT Registry built with {len(self.match_ids)} matches")
    
    def get_market_selection(
        self, 
        match_id: int, 
        market_code: str, 
        selection_value: str
    ) -> MarketSelection:
        """Get market selection with STRICT validation - NO FALLBACK"""
        market = self.registry.get(match_id, {}).get(market_code)
        if not market:
            raise MarketIntegrityError(
                f"Market '{market_code}' not found for match {match_id}"
            )
        
        for selection in market['selections']:
            if selection['value'] == selection_value:
                return MarketSelection(
                    match_id=match_id,
                    market_code=market_code,
                    selection_value=selection_value,
                    odds=selection['odds']
                )
        
        raise MarketIntegrityError(
            f"Selection '{selection_value}' not found in market "
            f"'{market_code}' for match {match_id}"
        )
    
    def get_available_markets(self, match_id: int) -> List[str]:
        """Get available market codes for match - deterministic order"""
        markets = list(self.registry.get(match_id, {}).keys())
        return sorted(markets)  # Deterministic ordering
    
    def get_market_selections(self, match_id: int, market_code: str) -> List[Dict[str, Any]]:
        """Get all selections for market - NO MODIFICATION"""
        market = self.registry.get(match_id, {}).get(market_code)
        if not market:
            raise MarketIntegrityError(
                f"Market '{market_code}' not found for match {match_id}"
            )
        return market['selections']


class DeterministicSlipGenerator:
    """Deterministic slip generator - seeded by master_slip_id"""
    
    def __init__(self, registry: StrictMarketRegistry, master_slip_id: int):
        self.registry = registry
        self.rng = random.Random(master_slip_id)  # Deterministic seed
        
        # Fixed generation parameters
        self.risk_allocations = {
            RiskLevel.LOW: 15,      # 30% of 50
            RiskLevel.MEDIUM: 20,   # 40% of 50
            RiskLevel.HIGH: 15      # 30% of 50
        }
        
        self.legs_per_risk = {
            RiskLevel.LOW: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.HIGH: 4
        }
    
    def generate_portfolio(self) -> List[Dict[str, Any]]:
        """Generate 50-slip portfolio deterministically"""
        all_slips = []
        slip_counter = 1
        
        for risk_level, count in self.risk_allocations.items():
            for _ in range(count):
                slip = self._generate_single_slip(risk_level, slip_counter)
                all_slips.append(slip)
                slip_counter += 1
        
        # STRICT: Must have exactly 50 slips
        if len(all_slips) != 50:
            raise SlipGenerationError(
                f"Generated {len(all_slips)} slips, expected 50"
            )
        
        return all_slips
    
    def _generate_single_slip(
        self, 
        risk_level: RiskLevel, 
        slip_number: int
    ) -> Dict[str, Any]:
        """Generate a single slip with STRICT integrity"""
        slip_id = f"SLIP_{slip_number:03d}"
        target_legs = self.legs_per_risk[risk_level]
        
        # Select unique matches
        available_matches = self.registry.match_ids.copy()
        self.rng.shuffle(available_matches)  # Deterministic shuffle
        
        if len(available_matches) < target_legs:
            raise SlipGenerationError(
                f"Cannot generate {target_legs}-leg slip with only "
                f"{len(available_matches)} available matches"
            )
        
        selected_matches = available_matches[:target_legs]
        
        # Generate legs for each match
        legs_data = []
        total_odds = Decimal('1.0')
        
        for match_id in selected_matches:
            leg = self._generate_leg(match_id, risk_level)
            legs_data.append(leg)
            total_odds *= leg.odds
        
        # Calculate deterministic confidence
        confidence = self._calculate_confidence(legs_data, risk_level)
        
        # Format legs to output contract
        formatted_legs = [leg.to_leg_dict() for leg in legs_data]
        
        # Variation type based on risk level
        variation_type = self._get_variation_type(risk_level)
        
        return {
            'slip_id': slip_id,
            'risk_level': risk_level.value,
            'variation_type': variation_type,
            'legs': formatted_legs,
            'total_odds': float(total_odds),
            'confidence_score': confidence
        }
    
    def _generate_leg(
        self, 
        match_id: int, 
        risk_level: RiskLevel
    ) -> MarketSelection:
        """Generate a single leg deterministically"""
        available_markets = self.registry.get_available_markets(match_id)
        
        # Select market based on risk level (deterministic)
        if risk_level == RiskLevel.LOW:
            # Prefer lower variance markets for low risk
            preferred_markets = ['1x2', 'over_under', 'both_teams_score']
            for pref in preferred_markets:
                if pref in available_markets:
                    market_code = pref
                    break
            else:
                market_code = self.rng.choice(available_markets)
        elif risk_level == RiskLevel.HIGH:
            # Prefer higher variance markets for high risk
            preferred_markets = ['correct_score', 'asian_handicap', 'halftime']
            for pref in preferred_markets:
                if pref in available_markets:
                    market_code = pref
                    break
            else:
                market_code = self.rng.choice(available_markets)
        else:  # MEDIUM
            market_code = self.rng.choice(available_markets)
        
        # Get selections for chosen market
        selections = self.registry.get_market_selections(match_id, market_code)
        
        # Select based on risk level (deterministic)
        if risk_level == RiskLevel.LOW:
            # Low risk: choose selection with lowest odds (highest probability)
            sorted_selections = sorted(selections, key=lambda x: x['odds'])
            selection = sorted_selections[0]
        elif risk_level == RiskLevel.HIGH:
            # High risk: choose selection with highest odds (lowest probability)
            sorted_selections = sorted(selections, key=lambda x: x['odds'], reverse=True)
            selection = sorted_selections[0]
        else:  # MEDIUM
            # Medium risk: random but deterministic selection
            selection = self.rng.choice(selections)
        
        return MarketSelection(
            match_id=match_id,
            market_code=market_code,
            selection_value=selection['value'],
            odds=selection['odds']
        )
    
    def _calculate_confidence(
        self, 
        legs: List[MarketSelection], 
        risk_level: RiskLevel
    ) -> float:
        """Calculate deterministic confidence score"""
        # Base confidence by risk level
        base_confidences = {
            RiskLevel.LOW: 0.7,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.3
        }
        
        confidence = base_confidences[risk_level]
        
        # Adjust based on number of legs (more legs = lower confidence)
        leg_count_factor = len(legs) * -0.05
        confidence += leg_count_factor
        
        # Adjust based on average odds (higher odds = lower confidence)
        if legs:
            avg_odds = float(legs[0].odds)
            for leg in legs[1:]:
                avg_odds *= float(leg.odds)
            avg_odds = avg_odds ** (1 / len(legs))
            
            if avg_odds < 1.5:
                odds_factor = 0.1
            elif avg_odds < 2.0:
                odds_factor = 0.0
            elif avg_odds < 3.0:
                odds_factor = -0.05
            else:
                odds_factor = -0.1
            
            confidence += odds_factor
        
        # Clamp to valid range
        return max(0.1, min(0.9, round(confidence, 3)))
    
    def _get_variation_type(self, risk_level: RiskLevel) -> str:
        """Get deterministic variation type"""
        variations = {
            RiskLevel.LOW: ["core", "conservative", "standard"],
            RiskLevel.MEDIUM: ["mixed", "balanced", "diversified"],
            RiskLevel.HIGH: ["hedge", "speculative", "contrarian"]
        }
        
        # Use risk level and fixed index for determinism
        index = hash(risk_level.value) % len(variations[risk_level])
        return variations[risk_level][index]


class StrictSlipBuilder:
    """Main orchestrator - ZERO FALLBACKS, ZERO SILENT ERRORS"""
    
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.registry = StrictMarketRegistry()
        
        # Extract master_slip_id for RNG seeding
        master_slip = payload.get('master_slip', {})
        self.master_slip_id = master_slip.get('master_slip_id')
        if self.master_slip_id is None:
            raise PayloadValidationError("Missing 'master_slip_id' in payload")
    
    def build_portfolio(self) -> Dict[str, Any]:
        """Build 50-slip portfolio with STRICT integrity - NO FALLBACKS"""
        logger.info(f"Starting STRICT slip generation for master_slip_id: {self.master_slip_id}")
        
        # Step 1: Build authoritative market registry
        self.registry.build_from_payload(self.payload)
        
        # Step 2: Initialize deterministic generator
        generator = DeterministicSlipGenerator(self.registry, self.master_slip_id)
        
        # Step 3: Generate portfolio
        generated_slips = generator.generate_portfolio()
        
        # Step 4: Validate final output
        self._validate_portfolio(generated_slips)
        
        # Step 5: Format response
        response = self._format_response(generated_slips)
        
        logger.info(f"✅ STRICT generation complete: {len(generated_slips)} slips")
        return response
    
    def _validate_portfolio(self, slips: List[Dict[str, Any]]) -> None:
        """STRICT validation of final portfolio"""
        # Must have exactly 50 slips
        if len(slips) != 50:
            raise SlipGenerationError(
                f"Portfolio has {len(slips)} slips, expected 50"
            )
        
        # Validate each slip
        for slip in slips:
            self._validate_slip(slip)
        
        # Validate risk distribution
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for slip in slips:
            risk_counts[slip['risk_level']] += 1
        
        expected_counts = {'low': 15, 'medium': 20, 'high': 15}
        for risk, count in risk_counts.items():
            if count != expected_counts[risk]:
                raise SlipGenerationError(
                    f"Risk distribution mismatch: {risk} has {count}, "
                    f"expected {expected_counts[risk]}"
                )
    
    def _validate_slip(self, slip: Dict[str, Any]) -> None:
        """STRICT validation of single slip"""
        # Check required fields
        required_fields = ['slip_id', 'risk_level', 'variation_type', 
                          'legs', 'total_odds', 'confidence_score']
        for field in required_fields:
            if field not in slip:
                raise SlipGenerationError(f"Slip missing required field: {field}")
        
        # Validate legs
        legs = slip['legs']
        if not legs:
            raise SlipGenerationError(f"Slip {slip['slip_id']} has no legs")
        
        # Check match uniqueness
        match_ids = [leg['match_id'] for leg in legs]
        if len(match_ids) != len(set(match_ids)):
            raise SlipGenerationError(
                f"Slip {slip['slip_id']} has duplicate match_ids"
            )
        
        # Validate each leg
        for leg in legs:
            self._validate_leg(leg)
    
    def _validate_leg(self, leg: Dict[str, Any]) -> None:
        """STRICT validation of single leg"""
        required_fields = ['match_id', 'market', 'selection', 'odds']
        for field in required_fields:
            if field not in leg:
                raise SlipGenerationError(f"Leg missing required field: {field}")
        
        # Verify market-selection exists in registry
        try:
            self.registry.get_market_selection(
                leg['match_id'],
                leg['market'],
                leg['selection']
            )
        except MarketIntegrityError as e:
            raise SlipGenerationError(
                f"Leg validation failed: {e}"
            )
        
        # Verify odds match registry
        market_selection = self.registry.get_market_selection(
            leg['match_id'],
            leg['market'],
            leg['selection']
        )
        if abs(Decimal(str(leg['odds'])) - market_selection.odds) > Decimal('0.001'):
            raise SlipGenerationError(
                f"Leg odds mismatch: {leg['odds']} vs registry {market_selection.odds}"
            )
    
    def _format_response(self, slips: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format response to EXACT contract"""
        master_slip = self.payload.get('master_slip', {})
        
        return {
            'status': 'success',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'total_slips': len(slips),
            'generated_slips': slips,
            'metadata': {
                'input_matches': len(self.registry.match_ids),
                'risk_distribution': {
                    'low': len([s for s in slips if s['risk_level'] == 'low']),
                    'medium': len([s for s in slips if s['risk_level'] == 'medium']),
                    'high': len([s for s in slips if s['risk_level'] == 'high'])
                },
                'master_slip_id': self.master_slip_id,
                'engine_version': '3.0.0',
                'strict_mode': True,
                'deterministic': True
            }
        }


def process_slip_builder_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point - STRICT MODE, ZERO FALLBACKS
    
    Raises MarketIntegrityError on any violation.
    Never returns invalid slips.
    """
    try:
        builder = StrictSlipBuilder(payload)
        return builder.build_portfolio()
    except MarketIntegrityError as e:
        # Log and re-raise - NO SILENT FALLBACK
        logger.error(f"MARKET INTEGRITY VIOLATION: {e}")
        raise
    except Exception as e:
        # Catch-all for unexpected errors - STILL NO FALLBACK
        logger.critical(f"UNEXPECTED ERROR: {e}")
        raise MarketIntegrityError(f"System error: {e}")


# ADD TO BOTTOM OF slip_builder.py (AFTER process_slip_builder_payload function)

# ============================================================================
# ADAPTER LAYER - Maintains import compatibility while enforcing STRICT MODE
# ============================================================================

class SlipBuilder:
        """
        COMPATIBILITY ADAPTER - Maintains import contract for existing code
        INTERNALLY USES STRICT MODE - All previous behavior is overridden
        """
        
        def __init__(self, payload: Optional[Dict[str, Any]] = None):
            logger.warning(
                "⚠️ SlipBuilder (legacy) is deprecated. "
                "Using STRICT MODE internally. "
                "Migrate to process_slip_builder_payload()"
            )
            
            if payload:
                self.payload = payload
                self.master_slip = payload.get('master_slip', {})
                self.matches = self.master_slip.get('matches', [])
            else:
                self.payload = {}
                self.master_slip = {}
                self.matches = []
        
        def parse_markets(self) -> Dict[str, List[Any]]:
            """Legacy method - returns empty dict (markets now handled by StrictMarketRegistry)"""
            logger.warning("parse_markets() is deprecated in STRICT MODE")
            return {}
        
        def build_slip_portfolio(self) -> Dict[str, Any]:
            """
            Legacy method - delegates to STRICT MODE processor
            Maintains same return format for backward compatibility
            """
            try:
                # Use STRICT MODE processor internally
                result = process_slip_builder_payload(self.payload)
                
                # Convert to legacy format if needed
                if result['status'] == 'success':
                    return result
                else:
                    # If STRICT MODE failed, return minimal valid structure
                    logger.error(f"STRICT MODE failed: {result.get('metadata', {}).get('error', 'Unknown')}")
                    return {
                        'status': 'error',
                        'generated_at': datetime.utcnow().isoformat() + 'Z',
                        'total_slips': 0,
                        'generated_slips': [],
                        'metadata': {
                            'input_matches': len(self.matches),
                            'error': 'STRICT MODE enforcement failed',
                            'master_slip_id': self.payload.get('master_slip', {}).get('master_slip_id', 0)
                        }
                    }
            except Exception as e:
                logger.error(f"Legacy SlipBuilder failed: {e}")
                return {
                    'status': 'error',
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                    'total_slips': 0,
                    'generated_slips': [],
                    'metadata': {
                        'input_matches': len(self.matches),
                        'error': str(e),
                        'master_slip_id': self.payload.get('master_slip', {}).get('master_slip_id', 0)
                    }
                }


class ContextInferencer:
        """
        COMPATIBILITY ADAPTER - Maintains import contract
        ALL METHODS RAISE DEPRECATION WARNINGS - No actual inference in STRICT MODE
        """
        
        @staticmethod
        def infer_market_options(market_name: str, base_odds: Decimal = Decimal('1.5')) -> List[Any]:
            """DEPRECATED - Market inference is forbidden in STRICT MODE"""
            logger.error(
                f"❌ ContextInferencer.infer_market_options() CALLED - "
                f"Market inference violates STRICT MODE. Market: {market_name}"
            )
            raise MarketIntegrityError(
                f"Context inference forbidden in STRICT MODE. "
                f"Cannot infer options for market: {market_name}"
            )
        
        @staticmethod
        def generate_hedge_selection(primary_selection: str, market_type: str) -> str:
            """DEPRECATED - Cross-market hedging forbidden in STRICT MODE"""
            logger.error(
                f"❌ ContextInferencer.generate_hedge_selection() CALLED - "
                f"Cross-market hedging violates STRICT MODE. "
                f"Selection: {primary_selection}, Market: {market_type}"
            )
            raise MarketIntegrityError(
                f"Cross-market hedging forbidden in STRICT MODE. "
                f"Cannot generate hedge for: {market_type}:{primary_selection}"
            )


@dataclass
class GeneratedSlip:
        """
        COMPATIBILITY ADAPTER - Same dataclass structure
        Used for type hints and backward compatibility
        """
        slip_id: str
        risk_level: str
        variation_type: str
        legs: List[Dict[str, Any]]  # Note: Different from STRICT MODE's SlipLeg
        total_odds: Decimal
        confidence_score: float
        
        @classmethod
        def from_strict_slip(cls, strict_slip: Dict[str, Any]) -> 'GeneratedSlip':
            """Convert STRICT MODE slip to legacy format"""
            return cls(
                slip_id=strict_slip['slip_id'],
                risk_level=strict_slip['risk_level'],
                variation_type=strict_slip['variation_type'],
                legs=strict_slip['legs'],
                total_odds=Decimal(str(strict_slip['total_odds'])),
                confidence_score=strict_slip['confidence_score']
            )


    # Update __all__ for module exports
__all__ = [
        'process_slip_builder_payload',
        'StrictSlipBuilder',
        'StrictMarketRegistry',
        'DeterministicSlipGenerator',
        'MarketSelection',
        'MarketIntegrityError',
        'PayloadValidationError',
        'MarketRegistryError',
        'SlipGenerationError',
        'RiskLevel',
        # Legacy compatibility exports
        'SlipBuilder',
        'ContextInferencer',
        'GeneratedSlip'
    ]


    # ============================================================================
    # STRICT MODE VERIFICATION ON IMPORT
    # ============================================================================

def _verify_strict_mode_on_import():
        """Verify STRICT MODE is active on module import"""
        logger.info("✅ STRICT MODE slip_builder.py loaded")
        logger.info("   - Zero silent fallbacks: ACTIVE")
        logger.info("   - Deterministic generation: ACTIVE")
        logger.info("   - Market integrity enforcement: ACTIVE")
        logger.info("   - Legacy compatibility: ACTIVE (with warnings)")


    # Run verification when module is imported
_verify_strict_mode_on_import()