# game_engine/engine/insight_engine.py

import numpy as np
from typing import Dict, Any, List, Optional
import math
import logging
from datetime import datetime

# Import the new engines
from .probability_engine import ProbabilityEngine
from .match_simulator import MatchSimulator
from .confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)

class MatchInsightEngine:
    def __init__(self, probability_engine: Optional[ProbabilityEngine] = None,
                 match_simulator: Optional[MatchSimulator] = None,
                 confidence_scorer: Optional[ConfidenceScorer] = None):
        """
        Enhanced insight engine that integrates with the new probability and simulation engines.
        
        Args:
            probability_engine: ProbabilityEngine instance (will create if not provided)
            match_simulator: MatchSimulator instance (will create if not provided)
            confidence_scorer: ConfidenceScorer instance (will create if not provided)
        """
        # Initialize core engines
        self.prob_engine = probability_engine or ProbabilityEngine()
        self.simulator = match_simulator or MatchSimulator(num_simulations=5000)
        self.scorer = confidence_scorer or ConfidenceScorer()
        
        # Market templates with enhanced metadata
        self.market_templates = {
            "match_result": {
                "name": "Match Result",
                "category": "outcome",
                "complexity": "low"
            },
            "btts_yes": {
                "name": "Both Teams to Score - Yes",
                "category": "goals",
                "complexity": "medium"
            },
            "btts_no": {
                "name": "Both Teams to Score - No", 
                "category": "goals",
                "complexity": "medium"
            },
            "over_1.5": {
                "name": "Over 1.5 Goals",
                "category": "goals",
                "complexity": "low"
            },
            "over_2.5": {
                "name": "Over 2.5 Goals",
                "category": "goals", 
                "complexity": "medium"
            },
            "under_2.5": {
                "name": "Under 2.5 Goals",
                "category": "goals",
                "complexity": "medium"
            },
            "double_chance_1x": {
                "name": "Double Chance - Home or Draw",
                "category": "outcome",
                "complexity": "low"
            },
            "double_chance_x2": {
                "name": "Double Chance - Draw or Away",
                "category": "outcome",
                "complexity": "low"
            },
            "half_time_draw": {
                "name": "Half Time Draw",
                "category": "timing",
                "complexity": "high"
            },
            "away_clean_sheet": {
                "name": "Away Clean Sheet",
                "category": "defense",
                "complexity": "medium"
            },
            "home_clean_sheet": {
                "name": "Home Clean Sheet", 
                "category": "defense",
                "complexity": "medium"
            },
            "over_3.5": {
                "name": "Over 3.5 Goals",
                "category": "goals",
                "complexity": "high"
            },
            "draw_no_bet_1": {
                "name": "Draw No Bet - Home",
                "category": "outcome",
                "complexity": "low"
            },
            "draw_no_bet_2": {
                "name": "Draw No Bet - Away",
                "category": "outcome", 
                "complexity": "low"
            },
            "handicap_home_-1": {
                "name": "Handicap Home -1",
                "category": "handicap",
                "complexity": "high"
            },
            "handicap_away_-1": {
                "name": "Handicap Away -1",
                "category": "handicap",
                "complexity": "high"
            }
        }
        
        logger.info("MatchInsightEngine initialized with integrated engines")

    def _normalize_match_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate incoming match data with comprehensive fallbacks.
        """
        normalized = {
            "match_id": match_data.get("match_id", f"unknown_{hash(str(match_data))}"),
            "home_team": match_data.get("home_team", "Home Team"),
            "away_team": match_data.get("away_team", "Away Team"),
            "venue": match_data.get("venue", "Neutral").lower(),
            "timestamp": match_data.get("timestamp", datetime.now().isoformat())
        }
        
        # Normalize team forms with fallbacks
        team_forms = match_data.get("team_forms", [])
        if isinstance(team_forms, list) and len(team_forms) >= 2:
            # Find home and away forms
            home_forms = [f for f in team_forms if isinstance(f, dict) and 
                         f.get("venue", "").lower() == "home"]
            away_forms = [f for f in team_forms if isinstance(f, dict) and 
                         f.get("venue", "").lower() == "away"]
            
            normalized["home_form"] = home_forms[0] if home_forms else self._create_default_form()
            normalized["away_form"] = away_forms[0] if away_forms else self._create_default_form()
        else:
            normalized["home_form"] = self._create_default_form()
            normalized["away_form"] = self._create_default_form()
        
        # Normalize head-to-head data
        h2h_data = match_data.get("head_to_head", {})
        normalized["head_to_head"] = {
            "home_wins": int(h2h_data.get("home_wins", 0)),
            "draws": int(h2h_data.get("draws", 0)),
            "away_wins": int(h2h_data.get("away_wins", 0)),
            "total_matches": int(h2h_data.get("total_matches", 0))
        }
        
        # Normalize model inputs
        model_inputs = match_data.get("model_inputs", {})
        normalized["model_inputs"] = {
            "home_xg": float(model_inputs.get("home_xg", 1.5)),
            "away_xg": float(model_inputs.get("away_xg", 1.2)),
            "volatility_score": float(model_inputs.get("volatility_score", 5.0)),
            "form_strength": float(model_inputs.get("form_strength", 0.5)),
            "momentum": float(model_inputs.get("momentum", 0.0))
        }
        
        # Normalize markets
        normalized["markets"] = match_data.get("markets", [])
        
        logger.debug(f"Normalized match data for {normalized['match_id']}")
        return normalized
    
    def _create_default_form(self) -> Dict[str, float]:
        """Create default form data when not provided"""
        return {
            "venue": "neutral",
            "avg_goals_scored": 1.5,
            "avg_goals_conceded": 1.2,
            "form_rating": 0.5,
            "win_rate": 0.33,
            "draw_rate": 0.33,
            "loss_rate": 0.34,
            "clean_sheet_rate": 0.2,
            "scoring_frequency": 0.6
        }
    
    def _create_match_object(self, normalized_data: Dict) -> Any:
        """
        Create a match object that can be consumed by the probability engine.
        """
        class MatchObject:
            def __init__(self, data):
                self.match_id = data["match_id"]
                self.home_team = data["home_team"]
                self.away_team = data["away_team"]
                self.venue = data["venue"]
                
                # Create model_inputs object
                inputs = data["model_inputs"]
                self.model_inputs = type('obj', (object,), {
                    'home_xg': inputs["home_xg"],
                    'away_xg': inputs["away_xg"],
                    'volatility_score': inputs["volatility_score"],
                    'form_strength': inputs["form_strength"],
                    'momentum': inputs["momentum"]
                })()
                
                # Create head_to_head object
                h2h = data["head_to_head"]
                self.head_to_head = type('obj', (object,), {
                    'home_wins': h2h["home_wins"],
                    'draws': h2h["draws"],
                    'away_wins': h2h["away_wins"],
                    'total_matches': h2h["total_matches"]
                })()
                
                # Create team_form object
                home_form = data["home_form"]
                away_form = data["away_form"]
                self.team_form = type('obj', (object,), {
                    'home_form': home_form.get("form_rating", 0.5),
                    'away_form': away_form.get("form_rating", 0.5),
                    'home_attack': home_form.get("avg_goals_scored", 1.5),
                    'away_attack': away_form.get("avg_goals_scored", 1.2),
                    'home_defense': home_form.get("avg_goals_conceded", 1.2),
                    'away_defense': away_form.get("avg_goals_conceded", 1.5)
                })()
                
                # Create markets
                self.full_markets = []
                for market in data.get("markets", []):
                    market_obj = type('obj', (object,), {
                        'market_name': market.get("name", "Unknown"),
                        'slug': market.get("slug", "unknown"),
                        'options': [
                            type('obj', (object,), {
                                'selection': opt.get("selection", "N/A"),
                                'odds': float(opt.get("odds", 1.5))
                            })() for opt in market.get("options", [])
                        ]
                    })()
                    self.full_markets.append(market_obj)
                
                # Set selected market if available
                if data.get("markets"):
                    first_market = data["markets"][0]
                    if first_market.get("options"):
                        self.selected_market = type('obj', (object,), {
                            'market_type': first_market.get("slug", "match_result"),
                            'selection': first_market["options"][0].get("selection", "Home"),
                            'odds': float(first_market["options"][0].get("odds", 1.5))
                        })()
        
        return MatchObject(normalized_data)
    
    def _calculate_market_probabilities(self, match_obj: Any, 
                                      normalized_data: Dict) -> Dict[str, float]:
        """
        Calculate probabilities for different markets using integrated engines.
        """
        market_probs = {}
        
        # Get base probabilities from probability engine
        base_probs = self.prob_engine.get_blended_probabilities(match_obj)
        
        # Run detailed simulation for advanced metrics
        sim_result = self.simulator.simulate_match_detailed(match_obj)
        
        # Match Result probabilities (from probability engine)
        market_probs["match_result_home"] = base_probs.get("home", 0.33)
        market_probs["match_result_draw"] = base_probs.get("draw", 0.34)
        market_probs["match_result_away"] = base_probs.get("away", 0.33)
        
        # Goal-based markets (from simulation)
        if "market_probabilities" in sim_result:
            sim_probs = sim_result["market_probabilities"]
            market_probs["btts_yes"] = sim_probs.get("btts", 0.5)
            market_probs["btts_no"] = 1 - market_probs["btts_yes"]
            market_probs["over_1.5"] = sim_probs.get("over_1_5", 0.6)
            market_probs["over_2.5"] = sim_probs.get("over_2_5", 0.4)
            market_probs["under_2.5"] = 1 - market_probs["over_2.5"]
            market_probs["over_3.5"] = sim_probs.get("over_3_5", 0.2)
            market_probs["home_clean_sheet"] = sim_probs.get("home_clean_sheet", 0.3)
            market_probs["away_clean_sheet"] = sim_probs.get("away_clean_sheet", 0.3)
        
        # Derived markets
        # Double Chance: Home or Draw
        market_probs["double_chance_1x"] = market_probs["match_result_home"] + market_probs["match_result_draw"]
        
        # Double Chance: Draw or Away
        market_probs["double_chance_x2"] = market_probs["match_result_draw"] + market_probs["match_result_away"]
        
        # Draw No Bet (refund if draw)
        market_probs["draw_no_bet_1"] = market_probs["match_result_home"] / (
            market_probs["match_result_home"] + market_probs["match_result_away"]
        ) if (market_probs["match_result_home"] + market_probs["match_result_away"]) > 0 else 0.5
        
        market_probs["draw_no_bet_2"] = 1 - market_probs["draw_no_bet_1"]
        
        # Half Time Draw (simplified - half of full time draw probability)
        market_probs["half_time_draw"] = market_probs["match_result_draw"] * 1.5  # More likely at HT
        
        # Handicap markets (simplified)
        goal_diff = normalized_data["model_inputs"]["home_xg"] - normalized_data["model_inputs"]["away_xg"]
        market_probs["handicap_home_-1"] = max(0.1, min(0.9, 0.5 + goal_diff * 0.2))
        market_probs["handicap_away_-1"] = 1 - market_probs["handicap_home_-1"]
        
        # Ensure all probabilities are valid
        for key, value in market_probs.items():
            market_probs[key] = max(0.05, min(0.95, value))
        
        return market_probs
    
    def _calculate_value_score(self, probability: float, market_odds: float) -> float:
        """
        Calculate the value score (expected value) of a bet.
        
        Args:
            probability: True probability (0-1)
            market_odds: Decimal odds offered by bookmaker
            
        Returns:
            Value score (positive = value bet, negative = bad bet)
        """
        if market_odds <= 1.0 or probability <= 0:
            return -1.0
        
        expected_value = (probability * (market_odds - 1)) - (1 - probability)
        return round(expected_value, 3)
    
    def _select_best_market(self, market_probs: Dict[str, float], 
                          normalized_data: Dict) -> Dict[str, Any]:
        """
        Select the best market based on value, confidence, and edge.
        """
        best_market = None
        best_score = -float('inf')
        
        # Get available markets from data
        available_markets = {}
        for market in normalized_data.get("markets", []):
            slug = market.get("slug", "")
            if slug in self.market_templates:
                for option in market.get("options", []):
                    market_key = f"{slug}_{option.get('selection', '').lower().replace(' ', '_')}"
                    available_markets[market_key] = {
                        "slug": slug,
                        "selection": option.get("selection", "Unknown"),
                        "odds": float(option.get("odds", 1.5)),
                        "market_name": self.market_templates[slug]["name"],
                        "category": self.market_templates[slug]["category"]
                    }
        
        # Evaluate each market
        for market_slug, prob in market_probs.items():
            # Find corresponding market data
            market_info = None
            for key, data in available_markets.items():
                if market_slug.startswith(data["slug"]):
                    market_info = data
                    break
            
            if not market_info:
                # Synthetic market - use fair odds
                fair_odds = 1 / max(0.05, prob)
                market_info = {
                    "slug": market_slug.split("_")[0],
                    "selection": self._get_selection_from_slug(market_slug, normalized_data),
                    "odds": fair_odds,
                    "market_name": self.market_templates.get(market_slug.split("_")[0], {}).get("name", "Unknown"),
                    "category": "synthetic"
                }
            
            # Calculate metrics
            value_score = self._calculate_value_score(prob, market_info["odds"])
            
            # Confidence from probability concentration
            confidence = min(0.95, prob * (1 + (1 - prob)))  # Higher for extreme probabilities
            
            # Complexity penalty
            complexity = self.market_templates.get(market_info["slug"], {}).get("complexity", "medium")
            complexity_factor = {"low": 1.0, "medium": 0.9, "high": 0.8}.get(complexity, 0.9)
            
            # Overall score
            overall_score = (value_score * 0.5) + (confidence * 0.3) + (complexity_factor * 0.2)
            
            if overall_score > best_score:
                best_score = overall_score
                best_market = {
                    "slug": market_info["slug"],
                    "market_name": market_info["market_name"],
                    "selection": market_info["selection"],
                    "probability": round(prob, 3),
                    "confidence": round(confidence * 100, 1),
                    "fair_odds": round(1 / max(0.05, prob), 2),
                    "market_odds": round(market_info["odds"], 2),
                    "value_score": value_score,
                    "category": market_info["category"],
                    "complexity": complexity,
                    "is_synthetic": "synthetic" in market_info.get("category", ""),
                    "overall_score": round(overall_score, 3)
                }
        
        # Ensure we always return a market
        if not best_market:
            # Fallback to highest probability match result
            best_market = {
                "slug": "match_result",
                "market_name": "Match Result",
                "selection": f"{normalized_data['home_team']} to Win",
                "probability": market_probs.get("match_result_home", 0.33),
                "confidence": 50.0,
                "fair_odds": 1 / max(0.05, market_probs.get("match_result_home", 0.33)),
                "market_odds": 1.5,
                "value_score": 0.0,
                "category": "outcome",
                "complexity": "low",
                "is_synthetic": True,
                "overall_score": 0.5
            }
        
        return best_market
    
    def _get_selection_from_slug(self, slug: str, data: Dict) -> str:
        """Convert slug to human-readable selection"""
        if "home" in slug or "1" in slug:
            return f"{data['home_team']} to Win"
        elif "away" in slug or "2" in slug:
            return f"{data['away_team']} to Win"
        elif "draw" in slug or "x" in slug:
            return "Draw"
        elif "btts_yes" in slug:
            return "Both Teams to Score: Yes"
        elif "btts_no" in slug:
            return "Both Teams to Score: No"
        elif "over_" in slug:
            goals = slug.split("_")[1]
            return f"Over {goals} Goals"
        elif "under_" in slug:
            goals = slug.split("_")[1]
            return f"Under {goals} Goals"
        elif "clean_sheet" in slug:
            team = "Home" if "home" in slug else "Away"
            return f"{team} Clean Sheet"
        else:
            return slug.replace("_", " ").title()
    
    def analyze_single_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced single match analysis with integrated engines.
        
        Args:
            match_data: Dictionary containing match information
            
        Returns:
            Comprehensive analysis dictionary
        """
        try:
            logger.info(f"Starting analysis for match: {match_data.get('match_id', 'unknown')}")
            
            # 1. Normalize and validate input data
            normalized_data = self._normalize_match_data(match_data)
            
            # 2. Create match object for engine consumption
            match_obj = self._create_match_object(normalized_data)
            
            # 3. Calculate comprehensive market probabilities
            market_probs = self._calculate_market_probabilities(match_obj, normalized_data)
            
            # 4. Run detailed simulation for additional insights
            sim_result = self.simulator.simulate_match_detailed(match_obj)
            
            # 5. Select best market with value assessment
            best_market = self._select_best_market(market_probs, normalized_data)
            
            # 6. Calculate match-level analytics
            analytics = self._generate_analytics_snapshot(normalized_data, market_probs, sim_result)
            
            # 7. Generate confidence metrics using confidence scorer
            confidence_data = {
                "probabilities": market_probs,
                "simulation_confidence": sim_result.get("simulation_summary", {}).get("confidence_score", 0.5),
                "volatility": normalized_data["model_inputs"]["volatility_score"],
                "is_fallback_market": best_market["is_synthetic"],
                "odds": best_market["market_odds"]
            }
            confidence_breakdown = self.scorer.calculate_match_confidence(confidence_data)
            
            # 8. Compile final analysis
            analysis = {
                "match_info": {
                    "match_id": normalized_data["match_id"],
                    "fixture": f"{normalized_data['home_team']} vs {normalized_data['away_team']}",
                    "venue": normalized_data["venue"],
                    "timestamp": normalized_data["timestamp"]
                },
                "recommendation": {
                    "market": best_market["market_name"],
                    "selection": best_market["selection"],
                    "slug": best_market["slug"],
                    "probability": best_market["probability"],
                    "fair_odds": best_market["fair_odds"],
                    "market_odds": best_market["market_odds"],
                    "value_rating": self._get_value_rating(best_market["value_score"]),
                    "value_score": best_market["value_score"],
                    "is_synthetic": best_market["is_synthetic"],
                    "complexity": best_market["complexity"],
                    "category": best_market["category"]
                },
                "confidence_metrics": {
                    "overall_confidence": confidence_breakdown["overall_confidence"],
                    "confidence_breakdown": confidence_breakdown["confidence_components"],
                    "recommendation_score": best_market["overall_score"]
                },
                "market_analysis": {
                    "top_3_markets": self._get_top_markets(market_probs, normalized_data, 3),
                    "market_coverage": len([m for m in market_probs.values() if m > 0.6]),
                    "best_value_market": best_market["slug"],
                    "probability_range": {
                        "min": round(min(market_probs.values()), 3),
                        "max": round(max(market_probs.values()), 3),
                        "avg": round(sum(market_probs.values()) / len(market_probs), 3)
                    }
                },
                "analytics_snapshot": analytics,
                "risk_assessment": {
                    "risk_rating": self._assign_risk_category(confidence_breakdown["overall_confidence"]),
                    "volatility_level": self._get_volatility_level(normalized_data["model_inputs"]["volatility_score"]),
                    "certainty_score": round(best_market["confidence"] / 100, 2),
                    "factors_considered": ["probability", "market_value", "simulation", "volatility"]
                },
                "engine_metadata": {
                    "engine_version": "2.0.0",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "engines_used": ["ProbabilityEngine", "MatchSimulator", "ConfidenceScorer"]
                }
            }
            
            logger.info(f"Analysis completed for {normalized_data['match_id']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Match analysis failed: {e}", exc_info=True)
            return self._create_error_analysis(match_data, str(e))
    
    def _generate_analytics_snapshot(self, data: Dict, market_probs: Dict, 
                                   sim_result: Dict) -> Dict[str, Any]:
        """Generate comprehensive analytics snapshot"""
        h_form = data["home_form"]
        a_form = data["away_form"]
        inputs = data["model_inputs"]
        h2h = data["head_to_head"]
        
        # Calculate expected goals
        expected_home_goals = inputs["home_xg"]
        expected_away_goals = inputs["away_xg"]
        total_expected_goals = expected_home_goals + expected_away_goals
        
        # Determine match characteristics
        if total_expected_goals > 3.0:
            match_type = "High-Scoring"
        elif total_expected_goals > 2.0:
            match_type = "Moderate-Scoring"
        else:
            match_type = "Low-Scoring"
        
        # H2H dominance
        total_h2h = max(1, h2h["home_wins"] + h2h["draws"] + h2h["away_wins"])
        home_dominance = h2h["home_wins"] / total_h2h
        away_dominance = h2h["away_wins"] / total_h2h
        
        if home_dominance > 0.6:
            h2h_bias = "Home Dominant"
        elif away_dominance > 0.6:
            h2h_bias = "Away Dominant"
        elif abs(home_dominance - away_dominance) < 0.2:
            h2h_bias = "Balanced"
        else:
            h2h_bias = "Slightly Favored"
        
        return {
            "match_characteristics": {
                "type": match_type,
                "goal_expectancy": round(total_expected_goals, 2),
                "expected_score": f"{round(expected_home_goals, 1)}-{round(expected_away_goals, 1)}",
                "volatility_level": self._get_volatility_level(inputs["volatility_score"]),
                "momentum": "Positive" if inputs["momentum"] > 0 else "Negative" if inputs["momentum"] < 0 else "Neutral"
            },
            "team_analysis": {
                "home_team": {
                    "attack_strength": round(h_form.get("avg_goals_scored", 1.5), 2),
                    "defense_weakness": round(h_form.get("avg_goals_conceded", 1.2), 2),
                    "form_rating": round(h_form.get("form_rating", 0.5), 2),
                    "clean_sheet_rate": round(h_form.get("clean_sheet_rate", 0.2), 2)
                },
                "away_team": {
                    "attack_strength": round(a_form.get("avg_goals_scored", 1.5), 2),
                    "defense_weakness": round(a_form.get("avg_goals_conceded", 1.2), 2),
                    "form_rating": round(a_form.get("form_rating", 0.5), 2),
                    "clean_sheet_rate": round(a_form.get("clean_sheet_rate", 0.2), 2)
                }
            },
            "historical_context": {
                "h2h_record": f"{h2h['home_wins']}-{h2h['draws']}-{h2h['away_wins']}",
                "h2h_bias": h2h_bias,
                "total_meetings": h2h["total_matches"],
                "recent_trend": "Strong Home" if home_dominance > 0.5 else "Strong Away" if away_dominance > 0.5 else "Mixed"
            },
            "simulation_insights": {
                "most_likely_outcome": sim_result.get("simulation_summary", {}).get("most_likely_outcome", "draw"),
                "simulation_confidence": round(sim_result.get("simulation_summary", {}).get("confidence_score", 0.5), 2),
                "btts_probability": round(sim_result.get("market_probabilities", {}).get("btts", 0.5), 2),
                "avg_total_goals": round(
                    sim_result.get("score_statistics", {}).get("average_home_score", 1.5) +
                    sim_result.get("score_statistics", {}).get("average_away_score", 1.2), 2
                )
            }
        }
    
    def _get_top_markets(self, market_probs: Dict, data: Dict, n: int = 3) -> List[Dict]:
        """Get top N markets by probability"""
        # Convert to list of tuples and sort
        market_list = []
        for slug, prob in market_probs.items():
            market_list.append({
                "slug": slug.split("_")[0],
                "selection": self._get_selection_from_slug(slug, data),
                "probability": round(prob, 3),
                "fair_odds": round(1 / max(0.05, prob), 2)
            })
        
        # Sort by probability (descending) and take top N
        market_list.sort(key=lambda x: x["probability"], reverse=True)
        return market_list[:n]
    
    def _get_value_rating(self, value_score: float) -> str:
        """Convert value score to human-readable rating"""
        if value_score > 0.2:
            return "Excellent Value"
        elif value_score > 0.1:
            return "Good Value"
        elif value_score > 0:
            return "Slight Value"
        elif value_score > -0.1:
            return "Fair"
        else:
            return "Poor Value"
    
    def _assign_risk_category(self, confidence: float) -> str:
        """Assign risk category based on confidence"""
        if confidence >= 0.7:
            return "Low Risk"
        elif confidence >= 0.5:
            return "Moderate Risk"
        elif confidence >= 0.3:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _get_volatility_level(self, volatility_score: float) -> str:
        """Convert volatility score to level"""
        if volatility_score >= 7:
            return "High Volatility"
        elif volatility_score >= 4:
            return "Moderate Volatility"
        else:
            return "Low Volatility"
    
    def _create_error_analysis(self, match_data: Dict, error_msg: str) -> Dict[str, Any]:
        """Create fallback analysis when main analysis fails"""
        return {
            "match_info": {
                "match_id": match_data.get("match_id", "unknown"),
                "fixture": f"{match_data.get('home_team', 'Home')} vs {match_data.get('away_team', 'Away')}",
                "status": "analysis_failed"
            },
            "recommendation": {
                "market": "Not Available",
                "selection": "Analysis Failed",
                "probability": 0.5,
                "fair_odds": 2.0,
                "value_rating": "Unknown",
                "is_synthetic": True,
                "error": error_msg[:100]
            },
            "confidence_metrics": {
                "overall_confidence": 0.1,
                "note": "Analysis failed, using fallback"
            },
            "analytics_snapshot": {
                "match_characteristics": {
                    "type": "Unknown",
                    "goal_expectancy": 0.0,
                    "volatility_level": "Unknown"
                }
            },
            "risk_assessment": {
                "risk_rating": "Very High Risk",
                "note": "Analysis unreliable due to error"
            },
            "engine_metadata": {
                "engine_version": "2.0.0",
                "analysis_timestamp": datetime.now().isoformat(),
                "error": True,
                "error_message": error_msg
            }
        }