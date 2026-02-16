# test_slip_generation.py
import requests
import json

# Test payload with multiple matches
test_payload = {
    "master_slip": {
        "master_slip_id": "test_123",
        "stake": 100.0,
        "currency": "EUR",
        "matches": [
            {
                "match_id": 1,  # Integer ID
                "home_team": "Team A",
                "away_team": "Team B",
                "selected_market": {
                    "market_type": "Match Result",
                    "selection": "Home",
                    "odds": 1.8
                },
                "model_inputs": {
                    "home_xg": 1.5,
                    "away_xg": 1.2,
                    "volatility_score": 5.0
                }
            },
            {
                "match_id": "match_2",  # String ID
                "home_team": "Team C",
                "away_team": "Team D",
                "selected_market": {
                    "market_type": "Match Result", 
                    "selection": "Away",
                    "odds": 2.1
                },
                "model_inputs": {
                    "home_xg": 1.0,
                    "away_xg": 1.5,
                    "volatility_score": 6.0
                }
            }
        ]
    }
}

response = requests.post(
    "http://localhost:5000/generate-slips",
    json=test_payload,
    headers={"Content-Type": "application/json"}
)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")