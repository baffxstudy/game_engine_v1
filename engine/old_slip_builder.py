# game_engine/engine/slip_builder.py

import math
import random
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import logging
import time
from collections import defaultdict
import numpy as np

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configure logging for production-level visibility
logger = logging.getLogger(__name__)

class IDFlex:
    """Utility class for handling flexible ID types and ensuring string consistency"""
    
    @staticmethod
    def to_string(value: Any, prefix: str = "") -> str:
        """Convert any value to string ID - Hardened against None and complex types"""
        try:
            if value is None:
                return f"{prefix}unknown_{int(time.time())}"
            
            # Handle Pydantic models or dict-like objects
            if hasattr(value, 'model_dump'):
                value = str(value)
            
            if isinstance(value, (int, float)):
                return f"{prefix}{int(value)}"
            
            if isinstance(value, str):
                val = value.strip()
                return f"{prefix}{val}" if val else f"{prefix}empty_{id(value)}"
            
            # Recursive check for nested ID attributes
            for attr in ['id', 'match_id', 'uuid']:
                if hasattr(value, attr):
                    return IDFlex.to_string(getattr(value, attr), prefix)
            
            return f"{prefix}{str(value)}"
        except Exception as e:
            logger.warning(f"IDFlex failure: {e}")
            return f"{prefix}fallback_{random.randint(1000, 9999)}"

@dataclass
class MarketData:
    """Standardized market data structure with default factories for safety"""
    market_name: str = "Unknown Market"
    market_type: str = "1X2"
    outcomes: Dict[str, float] = field(default_factory=lambda: {"Home": 1.85, "Away": 1.85})

@dataclass
class MatchData:
    """Standardized match data structure - Hardened with sensible defaults"""
    match_id: str
    selected_market: MarketData
    alternative_markets: List[MarketData] = field(default_factory=list)
    original_selection: str = "Home"
    original_odds: float = 1.85
    venue: str = "neutral"
    home_team: str = "Home"
    away_team: str = "Away"
    # Added inference fields for smarter fallbacks
    form_rating: float = 50.0 
    rank_inference: int = 10

class _ContextInferencer:
    """
    INTERNAL MEDIC: Analyzes available crumbs to synthesize missing data.
    Ensures the engine has enough 'intelligence' to proceed when Laravel sends partials.
    """
    @staticmethod
    def infer_odds(market_type: str) -> float:
        """Baseline odds if bookie data is missing"""
        defaults = {"1X2": 1.95, "BTTS": 1.90, "O/U": 1.90, "DC": 1.35}
        return defaults.get(market_type, 2.0)

    @staticmethod
    def normalize_prob(home_rank: Any, away_rank: Any) -> Dict[str, float]:
        """Normalize ranks to a probability distribution if stats are missing"""
        try:
            h = float(home_rank or 10)
            a = float(away_rank or 10)
            total = h + a
            return {"home": a/total, "away": h/total, "draw": 0.25}
        except:
            return {"home": 0.33, "away": 0.33, "draw": 0.34}

class MarketExtractor:
    """Extracts and standardizes market data from payload - DEFENSIVE GATEKEEPER"""
    
    @staticmethod
    def extract_match_data(match_payload: Any) -> MatchData:
        try:
            # 1. Safe Dictionary Conversion
            if hasattr(match_payload, 'model_dump'):
                m_dict = match_payload.model_dump()
            elif hasattr(match_payload, '__dict__'):
                m_dict = match_payload.__dict__
            elif isinstance(match_payload, dict):
                m_dict = match_payload
            else:
                m_dict = {}

            match_id = IDFlex.to_string(m_dict.get('match_id') or m_dict.get('id'))
            
            # 2. Selected Market Extraction (with recursive safety)
            sel_raw = m_dict.get('selected_market', {})
            # Handle if selected_market is a Pydantic object
            if hasattr(sel_raw, 'model_dump'): sel_raw = sel_raw.model_dump()
            
            orig_selection = str(sel_raw.get('selection') or "Home")
            try:
                orig_odds = float(sel_raw.get('odds') or 1.85)
            except (TypeError, ValueError):
                orig_odds = _ContextInferencer.infer_odds("1X2")

            selected_market = MarketData(
                market_name=str(sel_raw.get('market') or "Match Result"),
                market_type=str(sel_raw.get('market_type') or "1X2"),
                outcomes={orig_selection: orig_odds}
            )
            
            # 3. Alternative Markets Extraction
            alt_markets = []
            raw_list = m_dict.get('markets', m_dict.get('full_markets', []))
            
            if isinstance(raw_list, list):
                for m_raw in raw_list:
                    try:
                        if hasattr(m_raw, 'model_dump'): m_raw = m_raw.model_dump()
                        if not isinstance(m_raw, dict): continue
                        
                        m_name = m_raw.get('market_name') or m_raw.get('name') or "Unknown"
                        m_type = m_raw.get('market_type') or "1X2"
                        
                        outcomes = {}
                        options = m_raw.get('options', m_raw.get('selections', []))
                        
                        if isinstance(options, list):
                            for opt in options:
                                if hasattr(opt, 'model_dump'): opt = opt.model_dump()
                                # Cycle through common naming conventions in betting APIs
                                key = next((str(opt[f]) for f in ['selection', 'name', 'type', 'score'] if opt.get(f)), None)
                                val = opt.get('odds') or opt.get('price')
                                if key and val:
                                    outcomes[key] = float(val)
                        
                        if outcomes:
                            alt_markets.append(MarketData(m_name, m_type, outcomes))
                    except Exception as inner_e:
                        logger.debug(f"Skipping malformed market: {inner_e}")
                        continue

            return MatchData(
                match_id=match_id,
                selected_market=selected_market,
                alternative_markets=alt_markets,
                original_selection=orig_selection,
                original_odds=orig_odds,
                venue=str(m_dict.get('venue', 'neutral')),
                home_team=str(m_dict.get('home_team', 'Home')),
                away_team=str(m_dict.get('away_team', 'Away'))
            )
            
        except Exception as e:
            logger.error(f"Critical failure in MarketExtractor: {e}")
            return MatchData(
                match_id=f"err_{int(time.time())}",
                selected_market=MarketData()
            )

class HedgingEngine:
    """Generates intelligent hedging strategies - Preserves logic but adds safety"""
    
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed or int(time.time()))
    
    def get_hedged_selection(self, match_data: MatchData, 
                           original_selection: str,
                           hedge_type: str = 'opposite') -> Tuple[str, str, float]:
        """Safety-wrapped selection logic to ensure (Market, Selection, Odds) always returns"""
        try:
            # SAFETY CHECK: Ensure we have markets to work with
            all_pools = [match_data.selected_market] + match_data.alternative_markets
            valid_pools = [m for m in all_pools if m.outcomes]
            
            if not valid_pools:
                # Ultimate fallback
                return ("Match Result", original_selection, 1.85)
                
            if hedge_type == 'core':
                return (match_data.selected_market.market_name, original_selection, match_data.original_odds)
            
            if hedge_type == 'opposite':
                for m in all_pools:
                    others = {k: v for k, v in m.outcomes.items() if k != original_selection}
                    if others:
                        sel, odd = self.random.choice(list(others.items()))
                        return (m.market_name, sel, odd)

            elif hedge_type in ['adjacent', 'correlated']:
                keywords = ['double', 'draw', 'over', 'under', 'btts', 'both']
                # Check BOTH selected market AND alternative markets
                for m in all_pools:
                    # Also check market_type for correlation
                    if any(k in m.market_name.lower() for k in keywords) or \
                       any(k in m.market_type.lower() for k in ['draw', 'over', 'under', 'btts']):
                        if m.outcomes:
                            sel, odd = self.random.choice(list(m.outcomes.items()))
                            return (m.market_name, sel, odd)

            # Global Balanced Fallback
            valid_markets = [m for m in all_pools if m.outcomes]
            if valid_markets:
                tgt = self.random.choice(valid_markets)
                sel, odd = self.random.choice(list(tgt.outcomes.items()))
                return (tgt.market_name, sel, odd)

        except Exception as e:
            logger.warning(f"Hedge engine fallback triggered: {e}")
            
        return (match_data.selected_market.market_name, original_selection, match_data.original_odds)

    def generate_strategic_portfolio(self, matches: List[MatchData], num_slips: int = 50) -> List[List[Tuple[str, str, str]]]:
        """Generate slips strategically: high-risk/high-reward to safe hedges"""
        portfolio = []
        if not matches or num_slips <= 0:
            return portfolio
        
        # Define strategy types with target distribution
        strategies = [
            # (strategy_name, weight, description)
            ("high_risk", 0.25, "High odds, low probability - big returns"),
            ("balanced_high", 0.25, "Moderate risk with good return potential"),
            ("hedge_coverage", 0.30, "Spread selections to cover multiple outcomes"),
            ("safe_return", 0.20, "Lower returns but higher probability")
        ]
        
        # Generate configs for each strategy
        for strategy, weight, _ in strategies:
            count = max(1, int(num_slips * weight))
            
            for _ in range(count):
                config = []
                for i, match in enumerate(matches):
                    # Apply strategy logic per match
                    if strategy == "high_risk":
                        # Always pick opposite for maximum risk/reward
                        config.append(("opposite", "any", "any"))
                    
                    elif strategy == "balanced_high":
                        # Mix of core and opposite (60/40)
                        if self.random.random() < 0.6:
                            config.append(("core", "any", "any"))
                        else:
                            config.append(("opposite", "any", "any"))
                    
                    elif strategy == "hedge_coverage":
                        # Spread across different markets for maximum coverage
                        # Alternate between core, opposite, and correlated
                        choice = i % 3
                        if choice == 0:
                            config.append(("core", "any", "any"))
                        elif choice == 1:
                            config.append(("opposite", "any", "any"))
                        else:
                            # Use 'correlated' only if we have alternative markets, otherwise use 'opposite'
                            if match.alternative_markets:
                                config.append(("correlated", "any", "any"))
                            else:
                                config.append(("opposite", "any", "any"))
                    
                    elif strategy == "safe_return":
                        # Mostly core selections (80%) with some hedge
                        if self.random.random() < 0.8:
                            config.append(("core", "any", "any"))
                        else:
                            config.append(("opposite", "any", "any"))
                
                portfolio.append(config)
        
        # Trim to exact count and add randomness
        final_portfolio = portfolio[:num_slips]
        
        # If we need more, fill with random strategies
        while len(final_portfolio) < num_slips:
            config = []
            for match in matches:
                # Random strategy for remaining slips
                rand = self.random.random()
                if rand < 0.33:
                    config.append(("core", "any", "any"))
                elif rand < 0.66:
                    config.append(("opposite", "any", "any"))
                else:
                    # Use 'correlated' only if we have alternative markets, otherwise use 'opposite'
                    if match.alternative_markets:
                        config.append(("correlated", "any", "any"))
                    else:
                        config.append(("opposite", "any", "any"))
            final_portfolio.append(config)
        
        return final_portfolio

class SlipVariationGenerator:
    """Generates slip variations - Hardened against math errors """
    
    def __init__(self, hedging_engine: HedgingEngine):
        self.hedging_engine = hedging_engine
        self.variation_counter = 0
    
    def generate_slip(self, matches: List[MatchData], 
                     slip_config: List[Tuple[str, str, str]]) -> Optional[Dict[str, Any]]:
        try:
            legs = []
            total_odds = Decimal('1.0')
            
            for match, config in zip(matches, slip_config):
                h_type, _, _ = config
                m_name, sel, odd = self.hedging_engine.get_hedged_selection(match, match.original_selection, h_type)
                
                # Sanitize odds - never allow < 1.0 or NaN
                clean_odd = max(1.01, float(odd) if not np.isnan(odd) else 1.85)
                
                legs.append({
                    "match_id": match.match_id,
                    "market": m_name,
                    "selection": sel,
                    "odds": clean_odd
                })
                total_odds *= Decimal(str(clean_odd))
            
            self.variation_counter += 1
            v_type = slip_config[0][0] if slip_config else "balanced"
            
            # Use Decimal for financial precision
            final_odds = float(total_odds.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            
            return {
                "slip_id": f"SLIP_{self.variation_counter:06d}",
                "variation_type": v_type,
                "legs": legs,
                "total_odds": final_odds,
                "confidence_score": self._calculate_confidence(legs, v_type),
                "stake": 0.0,
                "possible_return": 0.0,
                "risk_level": "Medium"
            }
        except Exception as e:
            logger.error(f"Failed to generate slip variation: {e}")
            return None
    
    def _calculate_confidence(self, legs: List[Dict], v_type: str) -> float:
        """Calculate confidence based on actual slip characteristics"""
        try:
            # Base confidence by variation type
            base_conf = {
                "core": 0.7,  # Original selections
                "opposite": 0.3,  # Opposite selections (risky)
                "correlated": 0.5,  # Related markets
                "hedge": 0.6,  # Hedge strategies
            }.get(v_type, 0.5)
            
            # Analyze actual slip content
            avg_odds = np.mean([l.get('odds', 2.0) for l in legs]) if legs else 2.0
            
            # Odds-based adjustment: higher odds = lower confidence
            if avg_odds < 1.8:
                odds_factor = 0.15  # Low odds = higher confidence
            elif avg_odds < 3.0:
                odds_factor = 0.0  # Moderate odds = neutral
            elif avg_odds < 5.0:
                odds_factor = -0.15  # Higher odds = lower confidence
            else:
                odds_factor = -0.25  # Very high odds = much lower confidence
            
            # Market diversity factor
            markets = [l.get('market', '') for l in legs]
            unique_markets = len(set(markets))
            diversity_factor = (unique_markets / len(legs)) * 0.1 if legs else 0
            
            # Final confidence (clamped between 0.1 and 0.95)
            confidence = base_conf + odds_factor + diversity_factor
            return float(np.clip(confidence, 0.1, 0.95))
            
        except Exception:
            return 0.5  # Safe fallback

class PortfolioOptimizer:
    """Safely distributes total_stake across the generated slips"""
    
    def distribute_stakes(self, slips: List[Dict], total_stake: float) -> List[Dict]:
        if not slips or total_stake <= 0: return slips
        
        try:
            total_conf = sum(s.get('confidence_score', 0.5) for s in slips)
            if total_conf <= 0: total_conf = len(slips)
            
            for slip in slips:
                share = slip.get('confidence_score', 0.5) / total_conf
                slip['stake'] = round(total_stake * share, 2)
                slip['possible_return'] = round(slip['stake'] * slip['total_odds'], 2)
        except Exception as e:
            logger.warning(f"Stake distribution failed: {e}")
            # Equal weight fallback
            eq = round(total_stake / len(slips), 2)
            for s in slips:
                s['stake'] = eq
                s['possible_return'] = round(eq * s['total_odds'], 2)
                
        return slips

class SlipBuilder:
    """The main orchestrator: Hardened for production-grade stability"""
    
    def __init__(self, **kwargs):
        self.market_extractor = MarketExtractor()
        self.hedging_engine = HedgingEngine()
        self.slip_generator = SlipVariationGenerator(self.hedging_engine)
        self.portfolio_optimizer = PortfolioOptimizer()
    
    def generate(self, payload: Any) -> Dict[str, Any]:
        start_time = time.time()
        print(f"\n{'='*60}")
        print("🚀  SLIP BUILDER ENGINE STARTED")
        print(f"{'='*60}")
        
        # 1. DEFENSIVE UNWRAPPING
        try:
            master_slip_id = 0  # Initialize with default
            
            if hasattr(payload, 'master_slip'):
                m_slip = payload.master_slip
                # Extract from master_slip object
                ms_id_raw = getattr(m_slip, 'master_slip_id', 0)
            elif isinstance(payload, dict):
                m_slip = payload.get('master_slip', payload)
                ms_id_raw = m_slip.get('master_slip_id', 0)
            else:
                m_slip = payload
                ms_id_raw = 0

            # FORCE INTEGER CONVERSION - IMPROVED
            print(f"📋 Raw master_slip_id: {ms_id_raw}")
            
            try:
                # Handle MSL-20251229-010 format
                if isinstance(ms_id_raw, str):
                    # Extract numbers from string like "MSL-20251229-010"
                    numbers = ''.join(filter(str.isdigit, ms_id_raw))
                    if numbers:
                        # Take the last part after the last dash: "010" -> 10
                        parts = ms_id_raw.split('-')
                        if len(parts) >= 3:
                            # Get the last part and convert to int
                            last_part = parts[-1]
                            if last_part.isdigit():
                                master_slip_id = int(last_part)
                            else:
                                master_slip_id = int(numbers)
                        else:
                            master_slip_id = int(numbers)
                    else:
                        master_slip_id = 0
                else:
                    master_slip_id = int(ms_id_raw)
            except (ValueError, TypeError) as e:
                print(f"⚠️  Could not parse master_slip_id: {e}")
                master_slip_id = 0
            
            print(f"📋  Master Slip ID: {master_slip_id}")
            
            # Resolve Stake
            try:
                raw_stake = getattr(m_slip, 'stake', 100.0)
                if isinstance(raw_stake, dict): raw_stake = 100.0 # Error check
                total_stake = float(raw_stake)
                print(f"💰  Total Stake: ${total_stake:.2f}")
            except:
                total_stake = 100.0
                print(f"💰  Total Stake (default): ${total_stake:.2f}")

            # Resolve Matches
            matches_raw = getattr(m_slip, 'matches', [])
            if not matches_raw and isinstance(m_slip, dict):
                matches_raw = m_slip.get('matches', [])

            if not matches_raw:
                raise ValueError("Payload contains no match data")

            print(f"⚽  Extracting data from {len(matches_raw)} matches...")
            
            # 2. Extract and Standardize (Fail-Safe)
            matches = [self.market_extractor.extract_match_data(m) for m in matches_raw]
            print(f"✅  Match data extracted: {len(matches)} matches ready")
            
            # 3. Plan Portfolio
            print("🎯  Planning STRATEGIC portfolio...")
            configs = self.hedging_engine.generate_strategic_portfolio(matches, 50)
            print(f"📊  Portfolio planned: {len(configs)} slip configurations")
            
            # 4. Generate Variations
            print("🔄  Generating slip variations...")
            slips = []
            failed_slips = 0
            
            for i, cfg in enumerate(configs):
                s = self.slip_generator.generate_slip(matches, cfg)
                if s: 
                    slips.append(s)
                    # Show progress every 10 slips
                    if (i + 1) % 10 == 0:
                        print(f"   ↳ Generated {i + 1}/{len(configs)} slips...")
                else:
                    failed_slips += 1
                    # DEBUG: Show why slips are failing
                    if failed_slips <= 3:  # Limit debug output
                        print(f"   ⚠️ Slip {i} failed with config: {cfg}")

            if failed_slips > 0:
                print(f"   ⚠️ Total failed slips: {failed_slips}/{len(configs)}")

            if not slips: 
                # ADD MORE DEBUG INFO
                print(f"💥 DEBUG: matches count: {len(matches)}")
                if matches:
                    print(f"💥 DEBUG: first match alt markets: {len(matches[0].alternative_markets)}")
                    print(f"💥 DEBUG: first match selected market: {matches[0].selected_market}")
                print(f"💥 DEBUG: configs count: {len(configs)}")
                print(f"💥 DEBUG: first config: {configs[0] if configs else 'none'}")
                raise ValueError("Generator failed to produce valid slips")
            
            print(f"✅  Successfully generated {len(slips)} slip variations")
            
            # 5. Distribute Stake
            print("📈  Distributing stakes across slips...")
            optimized_slips = self.portfolio_optimizer.distribute_stakes(slips, total_stake)
            
            # Calculate some stats
            total_potential = sum(slip.get('possible_return', 0) for slip in optimized_slips)
            avg_confidence = np.mean([slip.get('confidence_score', 0) for slip in optimized_slips])
            
            print(f"\n📊  GENERATION SUMMARY:")
            print(f"   • Slips Generated: {len(optimized_slips)}")
            print(f"   • Total Potential Return: ${total_potential:.2f}")
            print(f"   • Average Confidence: {avg_confidence:.2%}")
            print(f"   • Processing Time: {(time.time() - start_time):.2f}s")
            print(f"{'='*60}")
            
            return {
                "success": True,
                "master_slip_id": master_slip_id,  # Returned as INT
                "generated_slips": optimized_slips,
                "metadata": {
                    "total_slips": len(optimized_slips),
                    "engine_version": "3.1.0-stable",
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "parsed_master_slip_id": master_slip_id,
                }
            }

        except Exception as e:
            print(f"\n💥  ENGINE CRASH!")
            print(f"   Error: {str(e)}")
            print(f"   Time elapsed: {(time.time() - start_time):.2f}s")
            print(f"{'='*60}")
            logger.critical(f"Engine Crash Suppressed: {e}", exc_info=True)
            return self._emergency_fallback(100.0, str(e), master_slip_id if 'master_slip_id' in locals() else 0)

    def _emergency_fallback(self, stake: float, error_msg: str, master_slip_id: int = 0) -> Dict[str, Any]:
        """Final line of defense: Returns a valid, non-crashing response even if logic explodes"""
        return {
            "success": False,
            "master_slip_id": master_slip_id,
            "error": error_msg,
            "generated_slips": [{
                "slip_id": "ERROR_FALLBACK_001",
                "variation_type": "core",
                "legs": [{"match_id": "fallback", "market": "1X2", "selection": "Home", "odds": 1.01}],
                "total_odds": 1.01,
                "confidence_score": 0.1,
                "stake": stake,
                "possible_return": stake,
                "risk_level": "High"
            }],
            "metadata": {
                "error": error_msg, 
                "status": "degraded",
                "master_slip_id": master_slip_id
            }
        }