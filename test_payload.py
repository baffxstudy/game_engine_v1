# test_payload.py
import requests

URL = "http://127.0.0.1:5000/generate-slips"

payload = {
    "master_slip": {
        "master_slip_id": "MSL-20240515-001",
        "stake": 50.0,
        "currency": "EUR",
        "risk_profile": "medium",
        "matches": [
            {
                "match_id": "EPL-20240518-001",
                "home_team": "Liverpool",
                "away_team": "Wolves",
                "home_form": {}, # Added
                "away_form": {}, # Added
                "head_to_head": {}, # Added
                "model_inputs": {
                    "expected_goals": 3.1, # Added
                    "home_xg": 2.2,
                    "away_xg": 0.9,
                    "volatility_score": 2.3,
                    "home_form_weight": 0.35,
                    "away_form_weight": 0.25,
                    "h2h_weight": 0.15,
                    "venue_advantage": 0.85
                },
                "selected_market": {
                    "market_type": "1X2",
                    "selection": "Home",
                    "odds": 1.35,
                    "implied_probability": 0.74,
                    "confidence_rating": 8.2
                },
                "full_markets": [
                    {
                        "market_name": "BTTS",
                        "options": [{"selection": "Yes", "odds": 1.9, "implied_probability": 0.526}]
                    }
                ]
            },
            {
                "match_id": "SERIEA-20240519-001",
                "home_team": "AC Milan",
                "away_team": "Juventus",
                "home_form": {}, # Added
                "away_form": {}, # Added
                "head_to_head": {}, # Added
                "model_inputs": {
                    "expected_goals": 2.4, # Added
                    "home_xg": 1.3,
                    "away_xg": 1.1,
                    "volatility_score": 3.8,
                    "home_form_weight": 0.3,
                    "away_form_weight": 0.3,
                    "h2h_weight": 0.2,
                    "venue_advantage": 0.7
                },
                "selected_market": {
                    "market_type": "1X2",
                    "selection": "Draw",
                    "odds": 3.4,
                    "implied_probability": 0.294,
                    "confidence_rating": 6.5
                },
                "full_markets": [
                    {
                        "market_name": "O/U",
                        "options": [{"selection": "Over 2.5", "odds": 2.3, "implied_probability": 0.435}]
                    }
                ]
            }
        ]
    }
}

response = requests.post(URL, json=payload)
if response.status_code == 200:
    print("✅ Success! Engine processed the data.")
    print(response.json())
else:
    print(f"❌ Error {response.status_code}: {response.text}")