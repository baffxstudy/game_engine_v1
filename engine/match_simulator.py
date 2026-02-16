# game_engine/engine/match_simulator.py

import random
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)

class MatchSimulator:
    """
    Monte Carlo match simulator for football matches.
    Simulates match outcomes based on probabilities and generates statistics.
    """
    
    def __init__(self, num_simulations: int = 10000, random_seed: int = None):
        """
        Initialize the match simulator.
        
        Args:
            num_simulations: Number of Monte Carlo simulations to run
            random_seed: Seed for reproducible results
        """
        self.num_simulations = num_simulations
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            logger.info(f"Initialized simulator with seed {random_seed}")
        else:
            logger.info(f"Initialized simulator with {num_simulations} simulations")
    
    def _simulate_outcome(self, probabilities: Dict[str, float]) -> str:
        """
        Simulate a single match outcome based on probabilities.
        
        Args:
            probabilities: Dict with 'home', 'draw', 'away' probabilities
            
        Returns:
            'home', 'draw', or 'away'
        """
        # Validate probabilities
        prob_sum = sum(probabilities.values())
        if not math.isclose(prob_sum, 1.0, rel_tol=0.01):
            logger.warning(f"Probabilities sum to {prob_sum}, normalizing")
            probabilities = {k: v/prob_sum for k, v in probabilities.items()}
        
        # Generate random number
        rand = random.random()
        
        # Determine outcome based on cumulative probabilities
        cumulative = 0.0
        for outcome, prob in probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return outcome
        
        # Fallback - return most probable outcome
        return max(probabilities.items(), key=lambda x: x[1])[0]
    
    def _simulate_score(self, home_xg: float, away_xg: float) -> Tuple[int, int]:
        """
        Simulate match score using Poisson distribution.
        
        Args:
            home_xg: Home team expected goals
            away_xg: Away team expected goals
            
        Returns:
            Tuple of (home_score, away_score)
        """
        try:
            # Sample from Poisson distributions
            home_score = np.random.poisson(home_xg)
            away_score = np.random.poisson(away_xg)
            
            # Ensure non-negative scores
            home_score = max(0, int(home_score))
            away_score = max(0, int(away_score))
            
            return home_score, away_score
            
        except Exception:
            # Fallback method
            home_score = int(max(0, home_xg + random.gauss(0, 0.5)))
            away_score = int(max(0, away_xg + random.gauss(0, 0.5)))
            return home_score, away_score
    
    def _extract_match_parameters(self, match: Any) -> Dict:
        """
        Extract simulation parameters from match object.
        """
        try:
            # Get probabilities (assuming they're already calculated)
            if hasattr(match, 'true_prob'):
                probs = match.true_prob
            elif hasattr(match, 'probabilities'):
                probs = match.probabilities
            else:
                # Try to get from context
                if hasattr(match, 'context'):
                    probs = getattr(match.context, 'inferred_probs', 
                                   {'home': 0.33, 'draw': 0.34, 'away': 0.33})
                else:
                    probs = {'home': 0.33, 'draw': 0.34, 'away': 0.33}
            
            # Get xG values
            if hasattr(match, 'model_inputs'):
                inputs = match.model_inputs
                home_xg = float(getattr(inputs, 'home_xg', 1.5))
                away_xg = float(getattr(inputs, 'away_xg', 1.2))
            else:
                home_xg, away_xg = 1.5, 1.2
            
            # Get volatility
            volatility = 5.0
            if hasattr(match, 'model_inputs'):
                inputs = match.model_inputs
                volatility = float(getattr(inputs, 'volatility_score', 5.0))
            
            return {
                'probabilities': probs,
                'home_xg': home_xg,
                'away_xg': away_xg,
                'volatility': volatility,
                'match_id': getattr(match, 'match_id', 'unknown')
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract match parameters: {e}")
            return {
                'probabilities': {'home': 0.33, 'draw': 0.34, 'away': 0.33},
                'home_xg': 1.5,
                'away_xg': 1.2,
                'volatility': 5.0,
                'match_id': 'unknown'
            }
    
    def simulate_match(self, probabilities: Dict[str, float]) -> float:
        """
        Simulate a single match and return success rate.
        Compatible with SlipBuilder interface.
        
        Args:
            probabilities: Dict with 'home', 'draw', 'away' probabilities
            
        Returns:
            Success rate (0-1) - higher means more confidence in predictions
        """
        try:
            # Run multiple simulations
            outcomes = []
            scores = []
            
            # Extract xG from probabilities or use defaults
            home_prob = probabilities.get('home', 0.33)
            away_prob = probabilities.get('away', 0.33)
            
            # Estimate xG from probabilities (simplified)
            home_xg = max(0.5, home_prob * 3.0)
            away_xg = max(0.5, away_prob * 3.0)
            
            for _ in range(self.num_simulations):
                # Simulate outcome
                outcome = self._simulate_outcome(probabilities)
                outcomes.append(outcome)
                
                # Simulate score
                home_score, away_score = self._simulate_score(home_xg, away_xg)
                scores.append((home_score, away_score))
            
            # Calculate outcome distribution
            outcome_counts = Counter(outcomes)
            total_simulations = len(outcomes)
            
            # Calculate probabilities from simulations
            sim_probs = {}
            for outcome in ['home', 'draw', 'away']:
                sim_probs[outcome] = outcome_counts.get(outcome, 0) / total_simulations
            
            # Calculate score statistics
            home_scores = [s[0] for s in scores]
            away_scores = [s[1] for s in scores]
            
            score_stats = {
                'avg_home_score': np.mean(home_scores),
                'avg_away_score': np.mean(away_scores),
                'max_home_score': np.max(home_scores),
                'max_away_score': np.max(away_scores),
                'btts_probability': sum(1 for h, a in scores if h > 0 and a > 0) / total_simulations,
                'over_2_5_probability': sum(1 for h, a in scores if h + a > 2.5) / total_simulations
            }
            
            # Calculate confidence score (how consistent simulations are)
            # Higher confidence when one outcome dominates
            max_prob = max(sim_probs.values())
            confidence = max_prob * (1 - np.std(list(sim_probs.values())))
            
            # Store simulation results
            self.last_simulation = {
                'outcomes': outcomes,
                'scores': scores,
                'probabilities': sim_probs,
                'score_stats': score_stats,
                'confidence': confidence
            }
            
            logger.debug(f"Simulation complete: {outcome_counts}, confidence: {confidence:.3f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Match simulation failed: {e}")
            return 0.5  # Default confidence
    
    def simulate_match_detailed(self, match: Any) -> Dict:
        """
        Run detailed simulation with comprehensive statistics.
        
        Args:
            match: Match data object
            
        Returns:
            Dictionary with detailed simulation results
        """
        try:
            # Extract parameters
            params = self._extract_match_parameters(match)
            
            # Get probabilities
            probabilities = params['probabilities']
            
            # Run simulations
            outcomes = []
            scores = []
            
            for _ in range(self.num_simulations):
                outcome = self._simulate_outcome(probabilities)
                outcomes.append(outcome)
                
                home_score, away_score = self._simulate_score(
                    params['home_xg'], params['away_xg']
                )
                scores.append((home_score, away_score))
            
            # Calculate statistics
            outcome_counts = Counter(outcomes)
            total = len(outcomes)
            
            # Outcome probabilities
            outcome_probs = {
                outcome: outcome_counts.get(outcome, 0) / total
                for outcome in ['home', 'draw', 'away']
            }
            
            # Score statistics
            home_scores = [s[0] for s in scores]
            away_scores = [s[1] for s in scores]
            
            # Calculate various market probabilities
            btts_count = sum(1 for h, a in scores if h > 0 and a > 0)
            over_1_5_count = sum(1 for h, a in scores if h + a > 1.5)
            over_2_5_count = sum(1 for h, a in scores if h + a > 2.5)
            over_3_5_count = sum(1 for h, a in scores if h + a > 3.5)
            
            return {
                'match_id': params['match_id'],
                'outcome_probabilities': outcome_probs,
                'score_statistics': {
                    'average_home_score': float(np.mean(home_scores)),
                    'average_away_score': float(np.mean(away_scores)),
                    'median_home_score': float(np.median(home_scores)),
                    'median_away_score': float(np.median(away_scores)),
                    'home_score_std': float(np.std(home_scores)),
                    'away_score_std': float(np.std(away_scores))
                },
                'market_probabilities': {
                    'btts': btts_count / total,
                    'over_1_5': over_1_5_count / total,
                    'over_2_5': over_2_5_count / total,
                    'over_3_5': over_3_5_count / total,
                    'home_clean_sheet': sum(1 for a in away_scores if a == 0) / total,
                    'away_clean_sheet': sum(1 for h in home_scores if h == 0) / total
                },
                'simulation_summary': {
                    'total_simulations': total,
                    'most_likely_outcome': max(outcome_probs.items(), key=lambda x: x[1])[0],
                    'confidence_score': max(outcome_probs.values()) * (1 - np.std(list(outcome_probs.values()))),
                    'volatility': params['volatility']
                },
                'raw_scores_sample': scores[:10]  # Sample of first 10 scores
            }
            
        except Exception as e:
            logger.error(f"Detailed simulation failed: {e}")
            return {
                'match_id': getattr(match, 'match_id', 'unknown'),
                'error': str(e),
                'outcome_probabilities': {'home': 0.33, 'draw': 0.34, 'away': 0.33},
                'confidence_score': 0.5
            }
    
    def batch_simulate(self, matches: List[Any]) -> List[Dict]:
        """
        Simulate multiple matches efficiently.
        
        Args:
            matches: List of match data objects
            
        Returns:
            List of simulation results for each match
        """
        results = []
        for match in matches:
            try:
                result = self.simulate_match_detailed(match)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to simulate match: {e}")
                results.append({
                    'match_id': getattr(match, 'match_id', 'unknown'),
                    'error': str(e),
                    'outcome_probabilities': {'home': 0.33, 'draw': 0.34, 'away': 0.33}
                })
        
        return results