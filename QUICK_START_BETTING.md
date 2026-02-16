# Betting Automation Framework - Quick Start

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install playwright>=1.40.0 fake-useragent>=1.4.0
playwright install chromium
```

### Step 2: Create Directories

```bash
# Run setup script
python game_engine/setup_betting_dirs.py

# Or manually:
mkdir game_engine/screenshots game_engine/logs
```

### Step 3: Start Server

```bash
python -m game_engine.app
# or
python -m game_engine.app_refactored
```

### Step 4: Test Endpoint

```bash
# Health check
curl http://localhost:5000/api/v1/betting/health

# Place test bet
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

## 📋 API Endpoints

### Health Check
```
GET /api/v1/betting/health
```

### Place Bet
```
POST /api/v1/place-bet
Content-Type: application/json

{
  "slip": {
    "slip_id": "TEST_001",
    "legs": [...],
    "stake": 10.0,
    "total_odds": 2.0
  },
  "test_mode": true
}
```

## 🔧 Configuration

Set environment variables:

```bash
BROWSER_HEADLESS=false      # Show browser window
BETTING_TEST_MODE=true      # Test mode (simulated)
TEST_SITE_URL=https://...    # Target site URL
```

## 📸 Screenshots

Screenshots are saved to `screenshots/`:
- Success: `bet_{slip_id}_{timestamp}.png`
- Error: `error_{slip_id}_{timestamp}.png`

## ⚠️ Important

- **Educational/Testing Only**
- Defaults to test mode (simulated)
- Production mode not implemented
- Always comply with local laws

## 🐛 Troubleshooting

**Playwright not found:**
```bash
pip install playwright
playwright install chromium
```

**UserAgent error:**
```bash
pip install fake-useragent
```

**Browser won't launch:**
- Set `BROWSER_HEADLESS=false` to see browser
- Check Chrome/Chromium installation
- Verify system permissions

## 📚 Documentation

- Full setup: `BETTING_AUTOMATION_SETUP.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md`
- API docs: `http://localhost:5000/docs`

---

**Ready to test!** 🎯
