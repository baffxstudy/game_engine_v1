# Python Backend Refactoring - Architecture Documentation

## Overview

The Python backend has been refactored into a clean, modular architecture that separates concerns and improves maintainability. This document describes the new structure and how to use it.

## Architecture Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Service Layer**: Business logic is separated from API endpoints
3. **Error Handling**: Structured exception hierarchy with proper error responses
4. **Configuration Management**: Centralized configuration with validation
5. **Modular Design**: Easy to test, extend, and maintain

## Project Structure

```
game_engine/
├── app.py                    # Original app (preserved for backward compatibility)
├── app_refactored.py         # New refactored main application
├── config.py                 # Configuration management
├── exceptions.py             # Custom exception hierarchy
├── logging_config.py         # Logging setup
├── middleware.py             # Request/response middleware
│
├── routers/                  # API endpoint routers
│   ├── __init__.py
│   ├── slips.py              # Slip generation endpoints
│   ├── analysis.py           # Match analysis endpoints
│   └── health.py             # Health check endpoints
│
├── services/                 # Business logic services
│   ├── __init__.py
│   ├── slip_service.py       # Slip generation logic
│   ├── callback_service.py   # Laravel callback handling
│   └── validation_service.py # Input validation
│
├── engine/                   # Core engine (unchanged)
│   ├── __init__.py
│   ├── slip_builder.py
│   ├── slip_builder_factory.py
│   └── ...
│
└── schemas.py                # Pydantic models (if exists)
```

## Key Components

### 1. Configuration (`config.py`)

Centralized configuration management with environment variable support:

```python
from game_engine.config import (
    ENGINE_VERSION,
    ENABLE_MONTE_CARLO,
    NUM_SIMULATIONS,
    API_HOST,
    API_PORT,
    get_config
)
```

**Environment Variables:**
- `ENABLE_MONTE_CARLO`: Enable/disable Monte Carlo simulations (default: "true")
- `NUM_SIMULATIONS`: Number of simulations (default: "10000")
- `API_HOST`: API host (default: "0.0.0.0")
- `API_PORT`: API port (default: "5000")
- `LARAVEL_BASE_URL`: Laravel backend URL (default: "http://localhost:8000")
- `LOG_LEVEL`: Logging level (default: "INFO")

### 2. Exception Handling (`exceptions.py`)

Structured exception hierarchy:

```python
from game_engine.exceptions import (
    EngineError,              # Base exception
    SlipBuilderError,        # Slip generation errors
    PayloadValidationError,   # Payload validation errors
    MarketIntegrityError,     # Market data errors
    StrategyError,            # Strategy-related errors
    MatchCountError,          # Match count validation errors
    CallbackError,            # Callback errors
    SPOError                  # Portfolio optimization errors
)
```

### 3. Services Layer

#### SlipService (`services/slip_service.py`)

Handles slip generation business logic:

```python
from game_engine.services import SlipService

service = SlipService()

# Extract strategy from payload
strategy = service.extract_strategy(payload)

# Validate match count
service.validate_match_count(strategy, match_count)

# Generate slips
result = service.generate_slips(payload, strategy=strategy)

# Enrich slips with stake
enriched_slips = service.enrich_slips_with_stake(slips, master_stake)
```

#### CallbackService (`services/callback_service.py`)

Handles Laravel callback communication:

```python
from game_engine.services import CallbackService

service = CallbackService()

# Post results to Laravel
service.post_callback(url, payload)

# Run portfolio optimization
spo_result = service.run_portfolio_optimization(generated_slips)

# Get callback URLs
urls = service.get_callback_urls(master_slip_id)
```

#### ValidationService (`services/validation_service.py`)

Input validation and sanitization:

```python
from game_engine.services import ValidationService

service = ValidationService()

# Validate master slip payload
validated_payload = service.validate_master_slip_payload(payload)

# Validate match structure
service.validate_match_structure(match, index)
```

### 4. Routers

#### Slips Router (`routers/slips.py`)

- `POST /api/v1/generate-slips` - Generate betting slips
- `GET /api/v1/strategies` - List available strategies

#### Analysis Router (`routers/analysis.py`)

- `POST /api/v1/analyze-match` - Analyze single match

#### Health Router (`routers/health.py`)

- `GET /health` - Health check
- `GET /engine-info` - Engine information

### 5. Middleware (`middleware.py`)

Request logging and error handling middleware:

- Generates unique request IDs
- Logs all requests/responses
- Adds custom headers
- Handles errors gracefully

## Migration Guide

### Option 1: Use Refactored App (Recommended)

Replace your startup command:

```bash
# Old
python -m game_engine.app

# New
python -m game_engine.app_refactored
```

Or update your import:

```python
# Old
from game_engine.app import app

# New
from game_engine.app_refactored import app
```

### Option 2: Gradual Migration

The refactored code is designed to work alongside the original `app.py`. You can:

1. Test the refactored version in parallel
2. Gradually migrate endpoints
3. Keep backward compatibility

## API Endpoints

### Generate Slips

```http
POST /api/v1/generate-slips
Content-Type: application/json

{
  "master_slip": {
    "master_slip_id": 12345,
    "strategy": "balanced",  // Optional: "balanced", "maxwin", or "compound"
    "stake": 100,
    "currency": "EUR",
    "matches": [...]
  }
}
```

**Response:**
```json
{
  "master_slip_id": 12345,
  "generated_slips": [...],
  "metadata": {...},
  "status": "success",
  "total_slips": 50
}
```

### List Strategies

```http
GET /api/v1/strategies
```

**Response:**
```json
{
  "status": "success",
  "available_strategies": ["balanced", "maxwin", "compound"],
  "strategies": {
    "balanced": {
      "name": "Balanced Portfolio",
      "description": "...",
      "risk_profile": "Medium",
      "min_matches": 3
    },
    ...
  }
}
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "operational",
  "engine_version": "2.2.0",
  "engine_available": true,
  "available_strategies": [...]
}
```

## Error Handling

All errors follow a consistent structure:

```json
{
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "details": {
    "field": "field_name",
    "additional": "info"
  },
  "request_id": "req_1234567890_abcd"
}
```

**HTTP Status Codes:**
- `400` - Bad Request (validation errors)
- `422` - Unprocessable Entity (market/slip errors)
- `500` - Internal Server Error
- `503` - Service Unavailable (engine not available)

## Testing

### Unit Tests

```python
from game_engine.services import SlipService, ValidationService

def test_slip_service():
    service = SlipService()
    # Test strategy extraction
    strategy = service.extract_strategy(payload)
    assert strategy == "balanced"

def test_validation_service():
    service = ValidationService()
    # Test payload validation
    validated = service.validate_master_slip_payload(payload)
    assert validated is not None
```

### Integration Tests

```python
from fastapi.testclient import TestClient
from game_engine.app_refactored import app

client = TestClient(app)

def test_generate_slips():
    response = client.post("/api/v1/generate-slips", json=payload)
    assert response.status_code == 200
    assert len(response.json()["generated_slips"]) == 50
```

## Logging

Structured logging with request IDs:

```
[2024-01-01 12:00:00] [INFO] [engine_api] [req_1234567890_abcd] ========== SLIP GENERATION STARTED ==========
[2024-01-01 12:00:01] [INFO] [engine_api] [req_1234567890_abcd] [STEP 1] Validating payload...
[2024-01-01 12:00:02] [INFO] [engine_api] [req_1234567890_abcd] [STEP 2] Extracting strategy...
```

**Log Files:**
- `logs/engine.log` - Main log file (rotates at 10MB, keeps 10 backups)

## Performance Considerations

1. **Monte Carlo Simulations**: Configurable via `NUM_SIMULATIONS` (default: 10,000)
2. **Background Tasks**: Callbacks run asynchronously to avoid blocking responses
3. **Request Timeout**: Default 5 minutes for slip generation
4. **Logging**: File rotation prevents disk space issues

## Best Practices

1. **Always use services** for business logic, not routers
2. **Validate inputs** using ValidationService before processing
3. **Handle errors** using custom exceptions for proper error responses
4. **Log important events** with request IDs for traceability
5. **Use configuration** from `config.py` instead of hardcoding values

## Troubleshooting

### Engine Not Available

If you see `ENGINE_AVAILABLE = False`:
1. Check that `engine` module imports successfully
2. Verify all dependencies are installed
3. Check logs for import errors

### Strategy Factory Not Available

If strategies are limited to "balanced":
1. Check that `slip_builder_factory` module exists
2. Verify strategy modules are importable
3. Check logs for import warnings

### Callback Failures

If callbacks to Laravel fail:
1. Verify `LARAVEL_BASE_URL` is correct
2. Check network connectivity
3. Verify Laravel endpoints are accessible
4. Check callback timeout settings

## Future Improvements

- [ ] Add request rate limiting
- [ ] Add caching layer for frequently accessed data
- [ ] Add metrics collection (Prometheus)
- [ ] Add distributed tracing support
- [ ] Add comprehensive test suite
- [ ] Add API versioning support

## Support

For issues or questions:
1. Check logs in `logs/engine.log`
2. Review error messages and request IDs
3. Check health endpoint: `GET /health`
4. Review engine info: `GET /engine-info`
