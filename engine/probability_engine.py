# game_engine/engine/probability_engine.py

import math
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class ProbabilityEngine:
    """
    Enhanced probability engine that provides comprehensive probability calculations
    for all match outcomes, not just the selected market.
    Compatible with the new SlipBuilder requirements.
    """
    
    def __init__(self, model_weight: float = 0.7, market_weight: float = 0.3):
        """
        Initialize with configurable weights.
        
        Args:
            model_weight: Weight for model-based probabilities (0-1)
            market_weight: Weight for market-based probabilities (0-1)
        """
        self.model_weight = model_weight
        self.market_weight = market_weight
        self._validate_weights()
        
    def _validate_weights(self):
        """Ensure weights sum to 1"""
        total = self.model_weight + self.market_weight
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def _extract_match_data(self, match: Any) -> Dict:
        """
        Safely extract data from match object with comprehensive fallbacks.
        """
        try:
            # Extract model inputs with fallbacks
            inputs = getattr(match, 'model_inputs', None)
            if not inputs:
                logger.warning("No model_inputs found in match, using defaults")
                inputs = type('obj', (object,), {
                    'home_xg': 1.5,
                    'away_xg': 1.2,
                    'volatility_score': 5.0,
                    'form_strength': 0.5,
                    'momentum': 0.0
                })()
            
            # Extract head-to-head data
            h2h = getattr(match, 'head_to_head', {}) or {}
            if not h2h and hasattr(match, 'head_to_head_stats'):
                h2h = getattr(match, 'head_to_head_stats', {})
            
            # Extract form data
            form = getattr(match, 'team_form', {}) or {}
            if not form and hasattr(match, 'form_analysis'):
                form = getattr(match, 'form_analysis', {})
            
            # Extract market odds
            selected_market = getattr(match, 'selected_market', None)
            full_markets = getattr(match, 'full_markets', [])
            
            return {
                'inputs': inputs,
                'h2h': h2h,
                'form': form,
                'selected_market': selected_market,
                'full_markets': full_markets,
                'venue': getattr(match, 'venue', 'Neutral').lower(),
                'home_team': getattr(match, 'home_team', 'Home'),
                'away_team': getattr(match, 'away_team', 'Away')
            }
            
        except Exception as e:
            logger.error(f"Failed to extract match data: {e}")
            # Return safe defaults
            return {
                'inputs': type('obj', (object,), {
                    'home_xg': 1.5, 'away_xg': 1.2,
                    'volatility_score': 5.0, 'form_strength': 0.5
                })(),
                'h2h': {},
                'form': {},
                'selected_market': None,
                'full_markets': [],
                'venue': 'neutral',
                'home_team': 'Home',
                'away_team': 'Away'
            }
    
    def _calculate_base_probabilities(self, match_data: Dict) -> Dict[str, float]:
        """
        Calculate base probabilities using xG (Expected Goals) model.
        """
        inputs = match_data['inputs']
        venue = match_data['venue']
        
        # Extract xG values with safety
        try:
            home_xg = float(getattr(inputs, 'home_xg', 1.5))
            away_xg = float(getattr(inputs, 'away_xg', 1.2))
        except (ValueError, TypeError):
            home_xg, away_xg = 1.5, 1.2
        
        # Total expected goals
        total_xg = home_xg + away_xg
        if total_xg == 0:
            total_xg = 1.0  # Avoid division by zero
        
        # Home win probability from xG difference
        xg_diff = home_xg - away_xg
        home_win_prob = 0.5 + (xg_diff / (2 * max(abs(xg_diff), 1.0)))
        
        # Draw probability (Poisson model simplification)
        # Probability of draw decreases as total goals increase
        draw_prob = 0.25 * math.exp(-total_xg / 4)
        
        # Away win probability is remainder
        away_win_prob = 1.0 - home_win_prob - draw_prob
        
        # Apply venue adjustment
        if 'home' in venue:
            home_win_prob *= 1.1
            away_win_prob *= 0.9
        elif 'away' in venue:
            home_win_prob *= 0.9
            away_win_prob *= 1.1
        
        # Normalize to ensure sum = 1
        probs = {
            'home': max(0.05, min(0.95, home_win_prob)),
            'draw': max(0.05, min(0.95, draw_prob)),
            'away': max(0.05, min(0.95, away_win_prob))
        }
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def _apply_head_to_head_adjustment(self, base_probs: Dict[str, float], 
                                     h2h_data: Dict) -> Dict[str, float]:
        """
        Adjust probabilities based on head-to-head history.
        """
        if not h2h_data:
            return base_probs.copy()
        
        try:
            # Extract H2H stats
            home_wins = float(h2h_data.get('home_wins', 0))
            draws = float(h2h_data.get('draws', 0))
            away_wins = float(h2h_data.get('away_wins', 0))
            total_matches = home_wins + draws + away_wins
            
            if total_matches < 1:
                return base_probs.copy()
            
            # H2H probabilities (with Laplace smoothing)
            h2h_probs = {
                'home': (home_wins + 1) / (total_matches + 3),
                'draw': (draws + 1) / (total_matches + 3),
                'away': (away_wins + 1) / (total_matches + 3)
            }
            
            # Blend with base probabilities (H2H gets 40% weight if enough data)
            h2h_weight = min(0.4, total_matches / 10)
            blended = {}
            for key in base_probs:
                blended[key] = (base_probs[key] * (1 - h2h_weight) + 
                              h2h_probs.get(key, base_probs[key]) * h2h_weight)
            
            # Normalize
            total = sum(blended.values())
            return {k: v/total for k, v in blended.items()}
            
        except Exception as e:
            logger.warning(f"H2H adjustment failed: {e}")
            return base_probs.copy()
    
    def _apply_form_adjustment(self, probs: Dict[str, float], 
                             form_data: Dict) -> Dict[str, float]:
        """
        Adjust probabilities based on team form.
        """
        if not form_data:
            return probs.copy()
        
        try:
            # Extract form metrics
            home_form = float(form_data.get('home_form', 0.5))
            away_form = float(form_data.get('away_form', 0.5))
            
            # Form strength difference (range: -1 to 1)
            form_diff = (home_form - away_form)
            
            # Adjust probabilities based on form difference
            adjustment_factor = 0.2 * form_diff  # Max 20% adjustment
            
            adjusted_probs = probs.copy()
            adjusted_probs['home'] *= (1 + adjustment_factor)
            adjusted_probs['away'] *= (1 - adjustment_factor)
            
            # Keep draw probability relatively stable
            draw_change = (adjusted_probs['home'] + adjusted_probs['away'] - 
                         probs['home'] - probs['away']) / 2
            adjusted_probs['draw'] -= draw_change
            
            # Ensure valid probabilities
            adjusted_probs = {k: max(0.05, min(0.95, v)) 
                            for k, v in adjusted_probs.items()}
            
            # Normalize
            total = sum(adjusted_probs.values())
            return {k: v/total for k, v in adjusted_probs.items()}
            
        except Exception as e:
            logger.warning(f"Form adjustment failed: {e}")
            return probs.copy()
    
    def _apply_market_adjustment(self, probs: Dict[str, float], 
                               match_data: Dict) -> Dict[str, float]:
        """
        Blend with market implied probabilities.
        """
        try:
            # Try to get market odds
            markets = match_data['full_markets']
            if not markets:
                return probs.copy()
            
            # Find Match Result market
            match_result_market = None
            for market in markets:
                if hasattr(market, 'market_name') and 'Match Result' in market.market_name:
                    match_result_market = market
                    break
            
            if not match_result_market:
                return probs.copy()
            
            # Extract odds from market options
            options = getattr(match_result_market, 'options', [])
            if not options:
                return probs.copy()
            
            # Convert odds to implied probabilities
            market_probs = {}
            for opt in options:
                selection = getattr(opt, 'selection', '').lower()
                odds = float(getattr(opt, 'odds', 1.0))
                
                if odds <= 1.0:
                    continue
                
                implied_prob = 1 / odds
                
                # Map selection to outcome
                if 'home' in selection or match_data['home_team'].lower() in selection:
                    market_probs['home'] = implied_prob
                elif 'away' in selection or match_data['away_team'].lower() in selection:
                    market_probs['away'] = implied_prob
                elif 'draw' in selection:
                    market_probs['draw'] = implied_prob
            
            # If we got all three market probabilities, blend them
            if len(market_probs) == 3:
                # Normalize market probs (they won't sum to 1 due to bookmaker margin)
                total_market = sum(market_probs.values())
                normalized_market = {k: v/total_market for k, v in market_probs.items()}
                
                # Blend with model probabilities
                blended = {}
                for key in probs:
                    blended[key] = (probs[key] * self.model_weight + 
                                  normalized_market.get(key, probs[key]) * self.market_weight)
                
                # Final normalization
                total = sum(blended.values())
                return {k: v/total for k, v in blended.items()}
            
            return probs.copy()
            
        except Exception as e:
            logger.warning(f"Market adjustment failed: {e}")
            return probs.copy()
    
    def _apply_volatility_adjustment(self, probs: Dict[str, float], 
                                   volatility_score: float) -> Dict[str, float]:
        """
        Adjust for match volatility (higher volatility = more uncertainty).
        """
        try:
            volatility = float(volatility_score) if volatility_score else 5.0
            
            # Volatility pulls probabilities toward 0.33 each
            volatility_factor = volatility / 20.0  # Range: 0.25 to 0.75
            
            # Blend with uniform distribution
            uniform = {'home': 0.33, 'draw': 0.34, 'away': 0.33}
            
            adjusted = {}
            for key in probs:
                adjusted[key] = (probs[key] * (1 - volatility_factor) + 
                               uniform[key] * volatility_factor)
            
            return adjusted
            
        except Exception:
            return probs.copy()
    
    def get_blended_probabilities(self, match: Any) -> Dict[str, float]:
        """
        Calculate comprehensive blended probabilities for all outcomes.
        
        Returns:
            Dictionary with keys: 'home', 'draw', 'away' and their probabilities
        """
        try:
            # Extract all match data
            match_data = self._extract_match_data(match)
            
            # Step 1: Base probabilities from xG
            probs = self._calculate_base_probabilities(match_data)
            
            # Step 2: Apply head-to-head adjustment
            probs = self._apply_head_to_head_adjustment(probs, match_data['h2h'])
            
            # Step 3: Apply form adjustment
            probs = self._apply_form_adjustment(probs, match_data['form'])
            
            # Step 4: Apply market adjustment
            probs = self._apply_market_adjustment(probs, match_data)
            
            # Step 5: Apply volatility adjustment
            volatility = getattr(match_data['inputs'], 'volatility_score', 5.0)
            probs = self._apply_volatility_adjustment(probs, volatility)
            
            # Final validation and rounding
            probs = {k: round(max(0.01, min(0.99, v)), 4) 
                    for k, v in probs.items()}
            
            # Ensure sum is exactly 1.0
            total = sum(probs.values())
            if not math.isclose(total, 1.0, rel_tol=1e-9):
                # Distribute difference proportionally
                diff = 1.0 - total
                probs = {k: v + (diff * v / total) for k, v in probs.items()}
                probs = {k: round(v, 4) for k, v in probs.items()}
            
            logger.debug(f"Calculated probabilities: {probs}")
            return probs
            
        except Exception as e:
            logger.error(f"Probability calculation failed: {e}")
            # Return reasonable defaults
            return {'home': 0.33, 'draw': 0.34, 'away': 0.33}
    
    def get_market_probability(self, match: Any, market_type: str = None) -> float:
        """
        Get probability for a specific market (backward compatibility).
        
        Args:
            match: Match data object
            market_type: Specific market (optional)
            
        Returns:
            Probability (0-1) for the selected market
        """
        try:
            # Get all probabilities
            all_probs = self.get_blended_probabilities(match)
            
            # If no specific market requested, use selected_market
            if not market_type:
                selected = getattr(match, 'selected_market', None)
                if selected:
                    market_type = getattr(selected, 'selection', '').lower()
                else:
                    # Return highest probability outcome
                    return max(all_probs.values())
            
            # Map market type to probability key
            market_type_lower = market_type.lower()
            
            if 'home' in market_type_lower or '1' in market_type_lower:
                return all_probs['home']
            elif 'away' in market_type_lower or '2' in market_type_lower:
                return all_probs['away']
            elif 'draw' in market_type_lower or 'x' in market_type_lower:
                return all_probs['draw']
            else:
                # Return the highest probability
                return max(all_probs.values())
                
        except Exception as e:
            logger.error(f"Market probability failed: {e}")
            return 0.5