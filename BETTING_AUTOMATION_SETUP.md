# Betting Automation Framework - Setup Guide

## Overview

This betting automation framework provides browser-based bet placement simulation for **educational and testing purposes only**.

## Features

- ✅ Human-like browser automation using Playwright
- ✅ Anti-detection measures (webdriver override, realistic headers)
- ✅ Human behavior simulation (bezier curves, typing delays, typos)
- ✅ Screenshot capture for verification
- ✅ Test mode (simulated) and production mode (placeholder)
- ✅ REST API endpoints for bet placement

## Installation

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install playwright>=1.40.0 fake-useragent>=1.4.0

# Install Playwright browsers
playwright install chromium
```

### 2. Verify Installation

```bash
python -c "from playwright.async_api import async_playwright; print('Playwright OK')"
python -c "from fake_useragent import UserAgent; print('UserAgent OK')"
```

## Configuration

### Environment Variables

Add to your `.env` file or set as environment variables:

```bash
# Browser Settings
BROWSER_HEADLESS=false          # Set to true for headless mode
BROWSER_TIMEOUT=30000          # Browser timeout in ms
VIEWPORT_WIDTH=1920            # Browser viewport width
VIEWPORT_HEIGHT=1080           # Browser viewport height

# Timing Settings
MIN_DELAY_MS=500               # Minimum delay between actions (ms)
MAX_DELAY_MS=3000              # Maximum delay between actions (ms)
TYPING_DELAY_MIN=50            # Minimum typing delay per character (ms)
TYPING_DELAY_MAX=150           # Maximum typing delay per character (ms)

# Test Settings
BETTING_TEST_MODE=true         # Enable test mode (simulated)
TEST_SITE_URL=https://www.sportybet.com/gh/m/sport
```

## API Endpoints

### Health Check

```bash
GET /api/v1/betting/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "betting-automation",
  "test_mode": true,
  "test_site_url": "https://www.sportybet.com/gh/m/sport",
  "screenshots_dir": "screenshots"
}
```

### Place Bet

```bash
POST /api/v1/place-bet
Content-Type: application/json

{
  "slip": {
    "slip_id": "TEST_001",
    "legs": [
      {
        "match_id": "MATCH_1",
        "selection": "Home Win",
        "odds": 2.0,
        "market_type": "MATCH_RESULT"
      }
    ],
    "stake": 10.0,
    "total_odds": 2.0
  },
  "test_mode": true
}
```

**Response:**
```json
{
  "success": true,
  "slip_id": "TEST_001",
  "timestamp": "20260119_123456",
  "stake": 10.0,
  "total_odds": 2.0,
  "potential_return": 20.0,
  "screenshot": "screenshots/bet_TEST_001_20260119_123456.png",
  "test_mode": true,
  "simulated": true,
  "bet_reference": "SIM_TEST_001_1234567890"
}
```

## Directory Structure

```
game_engine/
├── models.py                    # Pydantic models
├── config.py                    # Configuration (updated)
├── routers/
│   └── betting.py              # Betting API endpoints
├── services/
│   ├── browser_manager.py      # Browser context management
│   └── bet_placer.py           # Bet placement logic
└── utils/
    └── human_behavior.py       # Human behavior simulation

screenshots/                     # Screenshot storage
logs/                           # Application logs
```

## Usage Examples

### Python Client

```python
import httpx

async def place_test_bet():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:5000/api/v1/place-bet",
            json={
                "slip": {
                    "slip_id": "TEST_001",
                    "legs": [
                        {
                            "match_id": "MATCH_1",
                            "selection": "Home Win",
                            "odds": 2.0,
                            "market_type": "MATCH_RESULT"
                        }
                    ],
                    "stake": 10.0,
                    "total_odds": 2.0
                },
                "test_mode": True
            }
        )
        return response.json()
```

### cURL

```bash
curl -X POST "http://localhost:5000/api/v1/place-bet" \
  -H "Content-Type: application/json" \
  -d '{
    "slip": {
      "slip_id": "TEST_001",
      "legs": [{
        "match_id": "MATCH_1",
        "selection": "Home Win",
        "odds": 2.0,
        "market_type": "MATCH_RESULT"
      }],
      "stake": 10.0,
      "total_odds": 2.0
    },
    "test_mode": true
  }'
```

## Customization

### Site-Specific Selectors

The framework uses placeholder selectors. To customize for a specific site, modify:

1. **`bet_placer.py`** → `_add_selection()` method
   - Update selectors for match search
   - Update selectors for selection clicking
   - Add site-specific navigation logic

2. **`bet_placer.py`** → `_set_stake()` method
   - Update selector for stake input field
   - Add validation logic

3. **`bet_placer.py`** → `_simulate_bet_placement()` method
   - Update selector for place bet button
   - Add confirmation handling

### Example Customization

```python
async def _add_selection(self, page: Page, leg: BetLeg) -> None:
    # Site-specific: Search for match
    search_input = await page.wait_for_selector('input.search-input')
    await self.behavior.human_type(page, 'input.search-input', leg.match_id)
    
    # Site-specific: Click match result
    match_link = await page.wait_for_selector(f'a[data-match="{leg.match_id}"]')
    await self.behavior.human_click(page, f'a[data-match="{leg.match_id}"]')
    
    # Site-specific: Click selection
    selection = await page.wait_for_selector(
        f'button[data-selection="{leg.selection}"][data-odds="{leg.odds}"]'
    )
    await self.behavior.human_click(
        page,
        f'button[data-selection="{leg.selection}"][data-odds="{leg.odds}"]'
    )
```

## Screenshots

Screenshots are automatically saved to `screenshots/` directory:

- **Success:** `bet_{slip_id}_{timestamp}.png`
- **Error:** `error_{slip_id}_{timestamp}.png`

Screenshots are full-page captures for verification.

## Error Handling

The framework handles errors gracefully:

- **Element not found:** Logs warning, simulates action in test mode
- **Timeout:** Returns error with screenshot
- **Network error:** Returns error with details
- **Unexpected error:** Returns error with traceback

## Security & Legal

⚠️ **IMPORTANT:**

1. **Educational Purpose Only:** This framework is for testing and education
2. **Terms of Service:** Always comply with website terms of service
3. **Local Laws:** Ensure compliance with local gambling/betting laws
4. **No Production Use:** Production mode is not implemented
5. **Test Mode Default:** Always defaults to test mode

## Troubleshooting

### Playwright Not Found

```bash
pip install playwright
playwright install chromium
```

### UserAgent Not Found

```bash
pip install fake-useragent
```

### Browser Launch Fails

- Check if Chrome/Chromium is installed
- Try running with `BROWSER_HEADLESS=false` to see browser
- Check system permissions

### Screenshots Not Saving

- Ensure `screenshots/` directory exists
- Check write permissions
- Verify path in config

## Testing

### Unit Tests

```python
from game_engine.services.bet_placer import BetPlacer
from game_engine.models import BetSlip, BetLeg

async def test_bet_placement():
    placer = BetPlacer(test_mode=True)
    slip = BetSlip(
        slip_id="TEST_001",
        legs=[BetLeg(match_id="M1", selection="Home", odds=2.0, market_type="RESULT")],
        stake=10.0,
        total_odds=2.0
    )
    result = await placer.place_bet(slip)
    assert result["success"] == True
```

### Integration Tests

```python
from fastapi.testclient import TestClient
from game_engine.app import app

client = TestClient(app)

def test_place_bet():
    response = client.post("/api/v1/place-bet", json={
        "slip": {...},
        "test_mode": True
    })
    assert response.status_code == 200
    assert response.json()["success"] == True
```

## Next Steps

1. **Customize Selectors:** Update site-specific selectors in `bet_placer.py`
2. **Add Validation:** Add bet slip validation before placement
3. **Error Recovery:** Implement retry logic for transient errors
4. **Monitoring:** Add metrics and monitoring
5. **Documentation:** Document site-specific implementation

## Support

For issues or questions:
1. Check logs in `logs/engine.log`
2. Review error screenshots in `screenshots/`
3. Test with `test_mode=True` first
4. Verify browser installation

---

**Remember:** This framework is for **educational and testing purposes only**. Always comply with local laws and website terms of service.
