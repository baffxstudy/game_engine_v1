"""
Analyzes historical slip performance to improve future generation
"""

from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics
import json
from datetime import datetime, timedelta


class HistoryAnalyzer:
    """Analyzes historical slip performance"""
    
    def __init__(self, history_data: List[Dict]):
        self.history = history_data
        self.insights = {}
    
    def calculate_market_success_rates(self) -> Dict[str, Dict]:
        """Calculate success rates by market type and odds range"""
        market_stats = defaultdict(lambda: {
            'total': 0,
            'won': 0,
            'lost': 0,
            'odds_ranges': defaultdict(lambda: {'total': 0, 'won': 0})
        })
        
        for slip in self.history:
            if slip['result'] not in ['won', 'lost']:
                continue
                
            for leg in slip['legs']:
                market = leg['market']
                odds = leg['odds']
                odds_range = self._get_odds_range(odds)
                
                market_stats[market]['total'] += 1
                market_stats[market]['odds_ranges'][odds_range]['total'] += 1
                
                if slip['result'] == 'won':
                    market_stats[market]['won'] += 1
                    market_stats[market]['odds_ranges'][odds_range]['won'] += 1
                else:
                    market_stats[market]['lost'] += 1
        
        # Calculate success rates
        insights = {}
        for market, stats in market_stats.items():
            if stats['total'] > 0:
                insights[market] = {
                    'success_rate': stats['won'] / stats['total'],
                    'total_bets': stats['total'],
                    'odds_ranges': {
                        range_name: {
                            'success_rate': range_stats['won'] / range_stats['total'] if range_stats['total'] > 0 else 0,
                            'total': range_stats['total']
                        }
                        for range_name, range_stats in stats['odds_ranges'].items()
                        if range_stats['total'] >= 5  # Minimum sample size
                    }
                }
        
        return insights
    
    def analyze_confidence_predictiveness(self) -> Dict[str, Any]:
        """Test if confidence_score actually predicts success"""
        confidence_buckets = defaultdict(lambda: {'total': 0, 'won': 0})
        
        for slip in self.history:
            if slip['result'] not in ['won', 'lost']:
                continue
                
            bucket = self._get_confidence_bucket(slip['confidence_score'])
            confidence_buckets[bucket]['total'] += 1
            if slip['result'] == 'won':
                confidence_buckets[bucket]['won'] += 1
        
        # Calculate correlation
        buckets = []
        for bucket_name, stats in sorted(confidence_buckets.items()):
            if stats['total'] >= 5:  # Minimum sample
                success_rate = stats['won'] / stats['total']
                buckets.append({
                    'confidence_range': bucket_name,
                    'success_rate': success_rate,
                    'total_slips': stats['total']
                })
        
        return {
            'buckets': buckets,
            'predictive_power': self._calculate_correlation(buckets)
        }
    
    def get_improvement_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for slip generation"""
        market_insights = self.calculate_market_success_rates()
        confidence_insights = self.analyze_confidence_predictiveness()
        
        recommendations = {
            'high_success_markets': [],
            'avoid_markets': [],
            'optimal_odds_ranges': [],
            'confidence_adjustments': {}
        }
        
        # Market recommendations
        for market, insights in market_insights.items():
            if insights['success_rate'] > 0.6 and insights['total_bets'] >= 10:
                recommendations['high_success_markets'].append({
                    'market': market,
                    'success_rate': insights['success_rate'],
                    'sample_size': insights['total_bets']
                })
            elif insights['success_rate'] < 0.3 and insights['total_bets'] >= 10:
                recommendations['avoid_markets'].append({
                    'market': market,
                    'success_rate': insights['success_rate'],
                    'sample_size': insights['total_bets']
                })
        
        # Optimal odds ranges
        for market, insights in market_insights.items():
            for odds_range, range_insights in insights.get('odds_ranges', {}).items():
                if range_insights.get('success_rate', 0) > 0.55:
                    recommendations['optimal_odds_ranges'].append({
                        'market': market,
                        'odds_range': odds_range,
                        'success_rate': range_insights['success_rate'],
                        'sample_size': range_insights['total']
                    })
        
        # Confidence adjustments
        for bucket in confidence_insights.get('buckets', []):
            recommendations['confidence_adjustments'][bucket['confidence_range']] = {
                'actual_success_rate': bucket['success_rate'],
                'suggested_adjustment': max(0.1, min(0.9, bucket['success_rate'] * 1.2))
            }
        
        return recommendations
    
    def _get_odds_range(self, odds: float) -> str:
        """Categorize odds into ranges"""
        if odds < 1.5:
            return 'low (<1.5)'
        elif odds < 2.0:
            return 'medium_low (1.5-2.0)'
        elif odds < 3.0:
            return 'medium (2.0-3.0)'
        elif odds < 5.0:
            return 'high (3.0-5.0)'
        else:
            return 'very_high (>5.0)'
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Bucket confidence scores"""
        if confidence < 0.3:
            return 'low (<0.3)'
        elif confidence < 0.5:
            return 'medium_low (0.3-0.5)'
        elif confidence < 0.7:
            return 'medium (0.5-0.7)'
        elif confidence < 0.85:
            return 'high (0.7-0.85)'
        else:
            return 'very_high (>0.85)'
    
    def _calculate_correlation(self, buckets: List[Dict]) -> float:
        """Calculate correlation between confidence and success"""
        if len(buckets) < 2:
            return 0.0
        
        # Simple correlation calculation
        conf_midpoints = []
        success_rates = []
        
        for bucket in buckets:
            # Get midpoint of confidence range
            range_str = bucket['confidence_range']
            if '<' in range_str:
                conf_midpoints.append(0.15)  # <0.3
            elif '0.3-0.5' in range_str:
                conf_midpoints.append(0.4)
            elif '0.5-0.7' in range_str:
                conf_midpoints.append(0.6)
            elif '0.7-0.85' in range_str:
                conf_midpoints.append(0.775)
            else:
                conf_midpoints.append(0.925)  # >0.85
            
            success_rates.append(bucket['success_rate'])
        
        
        # ADD THIS TO SLIP BUILDER
        # Simple linear correlation
    #     return statistics.correlation(conf_midpoints, success_rates) if len(conf_midpoints) > 1 else 0.0
    # class SlipBuilder:
    # def __init__(self, payload: Optional[Dict[str, Any]] = None, 
    #              historical_insights: Optional[Dict] = None):
    #     # ... existing code ...
    #     self.historical_insights = historical_insights or {}
    
    # def generate_low_risk_slips(self, market_registry: Dict, count: int = 15) -> List[GeneratedSlip]:
    #     # Use historical insights if available
    #     if self.historical_insights.get('high_success_markets'):
    #         # Prefer markets with high historical success rates
    #         pass
    
#     new api endpoint 
#     @app.post("/api/v1/analyze-history")
# async def analyze_history(request: Request):
#     """Analyze historical slip performance and return insights"""
#     try:
#         historical_data = await request.json()
#         from .engine.history_analyzer import HistoryAnalyzer
        
#         analyzer = HistoryAnalyzer(historical_data)
#         insights = {
#             'market_success_rates': analyzer.calculate_market_success_rates(),
#             'confidence_analysis': analyzer.analyze_confidence_predictiveness(),
#             'recommendations': analyzer.get_improvement_recommendations(),
#             'analysis_timestamp': datetime.utcnow().isoformat()
#         }
        
#         return {"status": "success", "insights": insights}
#     except Exception as e:
#         logger.error(f"History analysis failed: {str(e)}")
#         return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})