# game_engine/app.py

import uvicorn
import time
import os
import logging
import json
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from typing import Dict, Any, Optional, List
import traceback
from datetime import datetime

# --- Configure logging first (before other imports) ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
log_file = os.path.join(LOG_DIR, "engine.log")

# Setup Rotating File Handler (Max 10MB per file, keeps last 10 files)
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Setup Console Handler - ASCII only for Windows compatibility
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

# Apply configuration
logger = logging.getLogger("engine_api")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- ASCII-safe logging helper for Windows ---
def safe_log(message: str) -> str:
    """Convert Unicode emojis to ASCII text for Windows compatibility"""
    replacements = {
        '✅': '[OK]',
        '🚀': '[START]',
        '🔧': '[PROCESS]',
        '📊': '[DATA]',
        '🔍': '[ANALYZE]',
        '⚠️': '[WARN]',
        '❌': '[ERROR]',
        '🎯': '[TARGET]',
        '📦': '[FEAT]',
        '📍': '[LOC]',
        '📝': '[LOG]',
        '📁': '[FILE]',
        '🔌': '[CONNECT]',
        '⚙️': '[CONFIG]',
        '💥': '[CRASH]',
        '🔒': '[SECURE]',
        '📈': '[STATS]',
        '🔍': '[SEARCH]',
        '💰': '[MONEY]',
        '⚽': '[FOOTBALL]',
        '🎰': '[CASINO]',
    }
    for emoji, text in replacements.items():
        message = message.replace(emoji, text)
    return message

logger.info(safe_log("=" * 80))
logger.info(safe_log("[START] Football Game Engine v2.1 Initializing..."))
logger.info(safe_log("[START] Dual-Strategy Support: Balanced + MaxWin"))
logger.info(safe_log("=" * 80))

# --- Import schemas ---
try:
    from .schemas import (
        MasterSlipRequest, 
        EngineResponse, 
        AnalysisRequest, 
        AnalysisResponse
    )
    SCHEMAS_AVAILABLE = True
    logger.info(safe_log("[OK] Schemas loaded successfully"))
except ImportError as e:
    logger.error(safe_log(f"[ERROR] Failed to import schemas: {e}"))
    SCHEMAS_AVAILABLE = False
    
    # Create minimal schemas if import fails
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    
    class MasterSlipRequest(BaseModel):
        master_slip: Dict[str, Any]
    
    class EngineResponse(BaseModel):
        master_slip_id: int
        generated_slips: List[Dict[str, Any]]
        metadata: Optional[Dict[str, Any]] = None
        status: Optional[str] = "success"
        generated_at: Optional[str] = None
        total_slips: Optional[int] = None
    
    class AnalysisRequest(BaseModel):
        data: Dict[str, Any]
        request_id: Optional[str] = None
    
    class AnalysisResponse(BaseModel):
        status: str = "success"
        analysis: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        request_id: Optional[str] = None

# --- Import engine with v2.1 dual-strategy slip builder ---
ENGINE_AVAILABLE = False
ENGINE_VERSION = "unknown"
RUN_SPO_AVAILABLE = False
STRATEGY_FACTORY_AVAILABLE = False

logger.info(safe_log("[PROCESS] Loading engine modules..."))

try:
    from .engine import (
        process_slip_builder_payload, 
        SlipBuilderError, 
        PayloadValidationError, 
        MarketIntegrityError,
        initialize_engine,
        get_engine_status,
        __version__ as ENGINE_VERSION,
        run_portfolio_optimization,
    )
    ENGINE_AVAILABLE = True
    RUN_SPO_AVAILABLE = callable(run_portfolio_optimization)
    
    logger.info(safe_log(f"[OK] Engine module loaded successfully"))
    logger.info(safe_log(f"[OK] Engine version: {ENGINE_VERSION}"))
    logger.info(safe_log(f"[OK] Portfolio optimization: {'AVAILABLE' if RUN_SPO_AVAILABLE else 'NOT AVAILABLE'}"))
    
except ImportError as e:
    logger.error(safe_log(f"[ERROR] Engine module import failed: {e}"))
    logger.error(safe_log(f"[ERROR] Traceback: {traceback.format_exc()}"))
    
    # Fallback definitions
    class SlipBuilderError(Exception): pass
    class PayloadValidationError(Exception): pass
    class MarketIntegrityError(Exception): pass
    
    def process_slip_builder_payload(payload, target_count=50, **kwargs):
        raise ImportError("Engine module not available")
    
    def initialize_engine():
        logger.warning(safe_log("[WARN] Engine initialization skipped - engine not available"))
    
    def get_engine_status():
        return {"error": "Engine not available"}
    
    def run_portfolio_optimization(*args, **kwargs):
        raise ImportError("SPO not available")

# --- Import strategy factory for dual-strategy support ---
try:
    from .engine.slip_builder_factory import (
        create_slip_builder,
        get_available_strategies,
        get_strategy_info,
        validate_strategy
    )
    STRATEGY_FACTORY_AVAILABLE = True
    logger.info(safe_log("[OK] Strategy factory loaded successfully"))
    
    # Log available strategies
    available_strategies = get_available_strategies()
    logger.info(safe_log(f"[OK] Available strategies: {', '.join(available_strategies)}"))
    
except ImportError as e:
    STRATEGY_FACTORY_AVAILABLE = False
    logger.warning(safe_log(f"[WARN] Strategy factory not available: {e}"))
    logger.warning(safe_log("[WARN] Falling back to legacy single-strategy mode"))
    
    # Fallback functions
    def create_slip_builder(strategy: str = "balanced", **kwargs):
        """Fallback: use legacy slip builder"""
        from .engine.slip_builder import SlipBuilder
        logger.warning(safe_log(f"[WARN] Using legacy SlipBuilder (strategy '{strategy}' ignored)"))
        return SlipBuilder(**kwargs)
    
    def get_available_strategies():
        return ["balanced"]
    
    def get_strategy_info():
        return {
            "balanced": {
                "name": "Balanced Portfolio",
                "description": "Legacy single-strategy mode",
                "status": "active"
            }
        }
    
    def validate_strategy(strategy: str):
        return strategy == "balanced"

# --- Import insight engine with fallback ---
try:
    from .engine.insight_engine import MatchInsightEngine
    INSIGHT_ENGINE_AVAILABLE = True
    logger.info(safe_log("[OK] Insight engine loaded successfully"))
except ImportError as e:
    MatchInsightEngine = None
    INSIGHT_ENGINE_AVAILABLE = False
    logger.warning(safe_log(f"[WARN] Insight engine not available: {e}"))

# Initialize the engine if available
if ENGINE_AVAILABLE:
    try:
        logger.info(safe_log("[PROCESS] Initializing intelligent slip builder..."))
        
        # Initialize with Monte Carlo enabled (can be configured via env vars)
        enable_mc = os.getenv("ENABLE_MONTE_CARLO", "true").lower() == "true"
        num_sims = int(os.getenv("NUM_SIMULATIONS", "10000"))
        
        initialize_engine(enable_monte_carlo=enable_mc, num_simulations=num_sims)
        
        # Get and log engine status
        engine_status = get_engine_status()
        logger.info(safe_log("[OK] Engine initialized successfully"))
        logger.info(safe_log(f"[CONFIG] Engine Type: {engine_status.get('engine_type', 'Unknown')}"))
        logger.info(safe_log(f"[CONFIG] Monte Carlo: {'ENABLED' if enable_mc else 'DISABLED'}"))
        logger.info(safe_log(f"[CONFIG] Simulations: {num_sims}"))
        logger.info(safe_log(f"[CONFIG] Strategy Factory: {'ENABLED' if STRATEGY_FACTORY_AVAILABLE else 'DISABLED'}"))
        
        # Log features
        features = engine_status.get('features', [])
        if features:
            logger.info(safe_log("[FEAT] Engine Features:"))
            for feature in features:
                logger.info(safe_log(f"[FEAT]   - {feature}"))
        
        # Log strategy info if factory is available
        if STRATEGY_FACTORY_AVAILABLE:
            logger.info(safe_log("[STRATEGY] Available Strategies:"))
            strategy_info = get_strategy_info()
            for strategy_name, info in strategy_info.items():
                logger.info(safe_log(f"[STRATEGY]   - {strategy_name}: {info.get('name', 'Unknown')}"))
                logger.info(safe_log(f"[STRATEGY]     Goal: {info.get('goal', 'N/A')}"))
                
    except Exception as e:
        logger.warning(safe_log(f"[WARN] Engine initialization failed: {e}"))
        logger.warning(safe_log(f"[WARN] Traceback: {traceback.format_exc()}"))
else:
    logger.error(safe_log("[ERROR] Engine not available - API will return 503 errors"))

# Create FastAPI app
app = FastAPI(
    title="Intelligent Football Slip Builder API",
    version=ENGINE_VERSION,
    description=f"Dual-strategy slip generation (Balanced + MaxWin) with Monte Carlo optimization v{ENGINE_VERSION}",
    docs_url="/docs",
    redoc_url="/redoc"
)

logger.info(safe_log("[OK] FastAPI application created"))

# Initialize insight engine if available
if INSIGHT_ENGINE_AVAILABLE:
    try:
        insight_engine = MatchInsightEngine()
        logger.info(safe_log("[OK] Insight engine initialized successfully"))
    except Exception as e:
        insight_engine = None
        logger.warning(safe_log(f"[WARN] Insight engine initialization failed: {e}"))
        INSIGHT_ENGINE_AVAILABLE = False
else:
    insight_engine = None

# ------------------------------
# Health check endpoint
# ------------------------------
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint for monitoring"""
    engine_status = "healthy" if ENGINE_AVAILABLE else "unavailable"
    insight_status = "available" if INSIGHT_ENGINE_AVAILABLE else "unavailable"
    schemas_status = "available" if SCHEMAS_AVAILABLE else "unavailable"
    strategy_status = "dual-strategy" if STRATEGY_FACTORY_AVAILABLE else "legacy-single-strategy"
    
    # Check log directory
    log_dir_status = "exists" if os.path.exists(LOG_DIR) else "missing"
    
    # Get detailed engine status if available
    detailed_engine_status = {}
    if ENGINE_AVAILABLE:
        try:
            detailed_engine_status = get_engine_status()
        except Exception as e:
            logger.error(f"Failed to get engine status: {e}")
    
    # Get available strategies
    available_strategies = []
    if STRATEGY_FACTORY_AVAILABLE:
        try:
            available_strategies = get_available_strategies()
        except Exception as e:
            logger.error(f"Failed to get available strategies: {e}")
    
    health_response = {
        "status": "operational" if ENGINE_AVAILABLE else "degraded",
        "service": "intelligent-football-slip-builder",
        "engine_version": ENGINE_VERSION,
        "engine_available": ENGINE_AVAILABLE,
        "engine_status": engine_status,
        "engine_details": detailed_engine_status,
        "strategy_mode": strategy_status,
        "available_strategies": available_strategies,
        "insight_engine": insight_status,
        "schemas_available": schemas_status,
        "spo_available": RUN_SPO_AVAILABLE,
        "log_dir": log_dir_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoints": {
            "/generate-slips": "POST - Generate 50 intelligent betting slips (supports strategy selection)",
            "/strategies": "GET - List available strategies and their details",
            "/api/v1/analyze-match": "POST - Analyze single match",
            "/health": "GET - Health check",
            "/engine-info": "GET - Detailed engine information",
            "/docs": "GET - Interactive API documentation"
        }
    }
    
    logger.info(f"[HEALTH] Health check requested - Status: {health_response['status']}")
    
    return health_response

@app.get("/engine-info")
async def engine_info():
    """Detailed engine information endpoint"""
    
    base_info = {
        "name": "Intelligent Football Slip Builder Engine",
        "version": ENGINE_VERSION,
        "description": "Dual-strategy slip generation with Monte Carlo optimization and portfolio diversification",
        "engine_available": ENGINE_AVAILABLE,
        "strategy_factory_available": STRATEGY_FACTORY_AVAILABLE,
        "insight_engine_available": INSIGHT_ENGINE_AVAILABLE,
        "spo_available": RUN_SPO_AVAILABLE,
    }
    
    if ENGINE_AVAILABLE:
        try:
            engine_status = get_engine_status()
            base_info.update({
                "engine_type": engine_status.get("engine_type", "Unknown"),
                "initialized": engine_status.get("initialized", False),
                "monte_carlo_enabled": engine_status.get("monte_carlo_enabled", False),
                "num_simulations": engine_status.get("num_simulations", 0),
                "features": engine_status.get("features", []),
                "capabilities": {
                    "slips_per_request": 50,
                    "deterministic": True,
                    "monte_carlo_optimization": engine_status.get("monte_carlo_enabled", False),
                    "portfolio_diversification": True,
                    "hedging_strategies": True,
                    "coverage_optimization": True,
                    "fault_tolerant": True,
                    "dual_strategy_support": STRATEGY_FACTORY_AVAILABLE,
                    "risk_levels": ["LOW", "MEDIUM", "HIGH"],
                    "metrics": [
                        "confidence_score",
                        "coverage_score",
                        "diversity_score",
                        "win_probability",
                        "expected_value"  # New for MaxWin strategy
                    ]
                }
            })
            
            # Add strategy information if factory is available
            if STRATEGY_FACTORY_AVAILABLE:
                base_info["strategies"] = get_strategy_info()
                base_info["available_strategies"] = get_available_strategies()
                
        except Exception as e:
            logger.error(f"Failed to get detailed engine info: {e}")
            base_info["error"] = str(e)
    else:
        base_info["features"] = ["Engine module not available"]
        base_info["capabilities"] = {}
    
    logger.info(f"[ENGINE INFO] Engine info requested")
    
    return base_info

@app.get("/strategies")
async def list_strategies():
    """
    List all available slip generation strategies and their details.
    
    Returns information about each strategy including:
    - Name and description
    - Optimization goals
    - Risk profile
    - Expected performance characteristics
    """
    if not STRATEGY_FACTORY_AVAILABLE:
        return {
            "status": "legacy_mode",
            "message": "Strategy factory not available - using legacy single-strategy mode",
            "available_strategies": ["balanced"],
            "strategies": {
                "balanced": {
                    "name": "Balanced Portfolio",
                    "description": "Legacy single-strategy mode",
                    "status": "active"
                }
            }
        }
    
    try:
        strategies = get_strategy_info()
        available = get_available_strategies()
        
        logger.info(f"[STRATEGIES] Strategy info requested - {len(available)} strategies available")
        
        return {
            "status": "success",
            "available_strategies": available,
            "default_strategy": "balanced",
            "strategies": strategies,
            "usage": {
                "description": "Include 'strategy' field in master_slip payload",
                "example": {
                    "master_slip": {
                        "master_slip_id": 12345,
                        "strategy": "maxwin",
                        "matches": "..."
                    }
                },
                "backward_compatibility": "If 'strategy' field is omitted, defaults to 'balanced'"
            }
        }
    except Exception as e:
        logger.error(f"[STRATEGIES] Failed to get strategy info: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# ------------------------------
# Payload normalization helper
# ------------------------------
def normalize_payload_for_slip_builder(raw_payload: Any, endpoint: str = "unknown") -> Dict[str, Any]:
    """
    Standardizes inputs for the intelligent slip_builder v2.1.
    
    The new slip builder handles most normalization internally,
    we just need to convert Pydantic models to dicts here.
    """
    if raw_payload is None:
        logger.error(f"[{endpoint}] Empty request payload received")
        raise ValueError(f"[{endpoint}] Empty request payload")

    # Handle Pydantic models (FastAPI default)
    if hasattr(raw_payload, "model_dump"):
        logger.debug(f"[{endpoint}] Converting Pydantic model (model_dump)")
        return raw_payload.model_dump()
    
    if hasattr(raw_payload, "dict") and not isinstance(raw_payload, dict):
        logger.debug(f"[{endpoint}] Converting Pydantic model (dict)")
        return raw_payload.dict()

    # Pass dicts or strings directly
    if isinstance(raw_payload, (dict, str)):
        logger.debug(f"[{endpoint}] Payload already in correct format: {type(raw_payload)}")
        return raw_payload

    logger.error(f"[{endpoint}] Unsupported payload type: {type(raw_payload)}")
    raise ValueError(f"[{endpoint}] Unsupported payload type: {type(raw_payload)}")

# ------------------------------
# Enhanced payload audit logger
# ------------------------------
def log_payload_summary(payload: Dict[str, Any], request_id: str = "unknown") -> None:
    """
    Log detailed summary of received payload for observability.
    """
    try:
        master_slip = payload.get("master_slip", {})
        master_slip_id = master_slip.get("master_slip_id", "unknown")
        stake = master_slip.get("stake", 0)
        currency = master_slip.get("currency", "unknown")
        strategy = master_slip.get("strategy", "balanced")  # NEW: Extract strategy
        matches = master_slip.get("matches", [])
        
        logger.info(safe_log(f"[{request_id}] ========== PAYLOAD SUMMARY =========="))
        logger.info(safe_log(f"[{request_id}] Master Slip ID: {master_slip_id}"))
        logger.info(safe_log(f"[{request_id}] Strategy: {strategy}"))  # NEW: Log strategy
        logger.info(safe_log(f"[{request_id}] Stake: {stake} {currency}"))
        logger.info(safe_log(f"[{request_id}] Number of Matches: {len(matches)}"))
        
        # Log match details
        for i, match in enumerate(matches[:5], 1):  # Log first 5 matches
            match_id = match.get("match_id") or match.get("id", "unknown")
            home = match.get("home_team", "Unknown")
            away = match.get("away_team", "Unknown")
            markets = match.get("match_markets") or match.get("markets", [])
            
            logger.info(safe_log(
                f"[{request_id}] Match {i}: ID={match_id} | "
                f"{home} vs {away} | Markets={len(markets)}"
            ))
        
        if len(matches) > 5:
            logger.info(safe_log(f"[{request_id}] ... and {len(matches) - 5} more matches"))
        
        logger.info(safe_log(f"[{request_id}] ===================================="))
        
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to log payload summary: {e}")

# ------------------------------
# Strategy extraction and validation helper
# ------------------------------
def extract_and_validate_strategy(payload: Dict[str, Any], request_id: str = "unknown") -> str:
    """
    Extract and validate strategy from payload.
    
    Returns:
        Validated strategy name (defaults to "balanced" if missing/invalid)
    """
    master_slip = payload.get("master_slip", {})
    requested_strategy = master_slip.get("strategy", "balanced")
    
    # Validate strategy if factory is available
    if STRATEGY_FACTORY_AVAILABLE:
        if not validate_strategy(requested_strategy):
            logger.warning(safe_log(
                f"[{request_id}] [STRATEGY] Invalid strategy '{requested_strategy}' requested, "
                f"defaulting to 'balanced'"
            ))
            return "balanced"
        
        logger.info(safe_log(f"[{request_id}] [STRATEGY] Using strategy: {requested_strategy}"))
        return requested_strategy
    else:
        # Factory not available - ignore strategy and use legacy mode
        if requested_strategy != "balanced":
            logger.warning(safe_log(
                f"[{request_id}] [STRATEGY] Strategy '{requested_strategy}' requested "
                f"but factory not available - using legacy 'balanced' mode"
            ))
        return "balanced"

# ------------------------------
# Middleware and exception handlers
# ------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Enhanced request logging middleware with detailed metrics"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}_{os.urandom(4).hex()}"
    path = request.url.path
    method = request.method
    client_ip = request.client.host if request.client else "unknown"
    
    request.state.request_id = request_id
    
    logger.info(safe_log("=" * 80))
    logger.info(safe_log(
        f"[{request_id}] {method} {path} | "
        f"Client: {client_ip} | "
        f"Started at: {datetime.utcnow().isoformat()}"
    ))
    logger.info(safe_log("=" * 80))
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        logger.info(safe_log(
            f"[{request_id}] COMPLETED | "
            f"Status: {response.status_code} | "
            f"Duration: {duration:.4f}s | "
            f"Path: {path}"
        ))
        
        # Add custom headers
        if isinstance(response, Response):
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Engine-Version"] = ENGINE_VERSION
            response.headers["X-Processing-Time"] = f"{duration:.4f}"
            response.headers["X-Strategy-Mode"] = "dual" if STRATEGY_FACTORY_AVAILABLE else "legacy"
        
        logger.info(safe_log("=" * 80))
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(safe_log("=" * 80))
        logger.error(safe_log(
            f"[{request_id}] CRASHED | "
            f"Error: {str(e)} | "
            f"Duration: {duration:.4f}s | "
            f"Path: {path}"
        ))
        logger.error(safe_log(f"[{request_id}] Traceback: {traceback.format_exc()}"))
        logger.error(safe_log("=" * 80))
        
        return Response(
            content=json.dumps({
                "error": "Internal server error",
                "request_id": request_id,
                "message": str(e)
            }),
            status_code=500,
            media_type="application/json"
        )

# Map new Engine Exceptions to HTTP responses
if ENGINE_AVAILABLE:
    @app.exception_handler(PayloadValidationError)
    async def payload_validation_handler(request: Request, exc: PayloadValidationError):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(safe_log(f"[{request_id}] PayloadValidationError: {str(exc)}"))
        
        return Response(
            content=json.dumps({
                "error": "Payload validation failed",
                "detail": str(exc),
                "status_code": 400,
                "request_id": request_id
            }),
            status_code=400,
            media_type="application/json"
        )

    @app.exception_handler(MarketIntegrityError)
    async def market_integrity_handler(request: Request, exc: MarketIntegrityError):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(safe_log(f"[{request_id}] MarketIntegrityError: {str(exc)}"))
        
        return Response(
            content=json.dumps({
                "error": "Market data integrity error",
                "detail": str(exc),
                "status_code": 422,
                "request_id": request_id
            }),
            status_code=422,
            media_type="application/json"
        )

# ------------------------------
# Background helper to post to callback and run SPO
# ------------------------------
import httpx

def _post_sync(url: str, payload: Dict[str, Any], timeout: float = 30.0, request_id: str = "unknown") -> None:
    """
    Synchronous HTTP POST for background tasks.
    Posts results back to Laravel callback URL.
    """
    try:
        logger.info(safe_log(f"[{request_id}] [CALLBACK] Posting to {url}..."))
        
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            
            if resp.status_code in [200, 201]:
                logger.info(safe_log(
                    f"[{request_id}] [CALLBACK] SUCCESS | "
                    f"URL: {url} | "
                    f"Status: {resp.status_code}"
                ))
            else:
                logger.warning(safe_log(
                    f"[{request_id}] [CALLBACK] UNEXPECTED STATUS | "
                    f"URL: {url} | "
                    f"Status: {resp.status_code} | "
                    f"Response: {resp.text[:200]}"
                ))
                
    except httpx.TimeoutException:
        logger.error(safe_log(f"[{request_id}] [CALLBACK] TIMEOUT | URL: {url}"))
    except Exception as e:
        logger.error(safe_log(f"[{request_id}] [CALLBACK] FAILED | URL: {url} | Error: {str(e)}"))

def _run_spo_and_callback(
    phase1_result: Dict[str, Any],
    phase1_callback_url: Optional[str],
    spo_callback_url: Optional[str],
    master_slip_id: int,
    strategy_used: str,  # NEW: Track which strategy was used
    request_id: str = "unknown"
):
    """
    Background task to:
    1. Post Phase 1 (50 slips) results to Laravel
    2. Run Portfolio Optimization (SPO)
    3. Post Phase 2 (20 optimized slips) results to Laravel
    """
    try:
        logger.info(safe_log(f"[{request_id}] [BACKGROUND] Starting background pipeline"))
        logger.info(safe_log(f"[{request_id}] [BACKGROUND] Master Slip ID: {master_slip_id}"))
        logger.info(safe_log(f"[{request_id}] [BACKGROUND] Strategy Used: {strategy_used}"))
        
        # PHASE 1: Post 50 generated slips to Laravel
        if phase1_callback_url:
            logger.info(safe_log(f"[{request_id}] [PHASE 1] Preparing Laravel-compatible payload..."))
            
            # Extract data from engine result
            engine_metadata = phase1_result.get("metadata", {})
            generated_slips = phase1_result.get("generated_slips", [])
            
            # Build Laravel-compatible Phase 1 payload
            laravel_phase1_payload = {
                "success": True,
                "generated_slips": generated_slips,
                "metadata": {
                    "master_slip_id": master_slip_id,
                    "strategy": strategy_used,  # NEW: Include strategy in metadata
                    "input_matches": engine_metadata.get("input_matches", 0),
                    "dropped_matches_count": 0,  # New slip builder doesn't drop matches
                    "dropped_match_ids": [],
                    "registered_markets": engine_metadata.get("unique_markets", 0),
                    "unmapped_markets": 0,  # New slip builder preserves original market codes
                    "malformed_markets": 0,
                    "duplicate_markets": 0,
                    "duplicate_matches": 0,
                    "deterministic": True,
                    "strict_mode": False,
                    "engine_version": engine_metadata.get("engine_version", "2.1.0"),
                    "phase": "phase1",
                    "total_slips": len(generated_slips),
                    "spo_queued": RUN_SPO_AVAILABLE  # True if SPO is available
                }
            }
            
            logger.info(safe_log(f"[{request_id}] [PHASE 1] Laravel payload prepared:"))
            logger.info(safe_log(f"[{request_id}] [PHASE 1]   Success: {laravel_phase1_payload['success']}"))
            logger.info(safe_log(f"[{request_id}] [PHASE 1]   Strategy: {strategy_used}"))
            logger.info(safe_log(f"[{request_id}] [PHASE 1]   Total Slips: {laravel_phase1_payload['metadata']['total_slips']}"))
            logger.info(safe_log(f"[{request_id}] [PHASE 1]   Input Matches: {laravel_phase1_payload['metadata']['input_matches']}"))
            logger.info(safe_log(f"[{request_id}] [PHASE 1]   SPO Queued: {laravel_phase1_payload['metadata']['spo_queued']}"))
            
            logger.info(safe_log(f"[{request_id}] [PHASE 1] Posting 50 slips to Laravel..."))
            _post_sync(phase1_callback_url, laravel_phase1_payload, request_id=request_id)
        else:
            logger.warning(safe_log(f"[{request_id}] [PHASE 1] No callback URL provided, skipping"))

        # PHASE 2: Run Portfolio Optimization
        if RUN_SPO_AVAILABLE:
            logger.info(safe_log(f"[{request_id}] [PHASE 2] Starting portfolio optimization..."))
            
            generated_slips = phase1_result.get("generated_slips", [])
            logger.info(safe_log(
                f"[{request_id}] [SPO] Input: {len(generated_slips)} slips | "
                f"Target: 20 optimized slips"
            ))
            
            spo_input = {
                "bankroll": 1000,  # Can be made configurable
                "generated_slips": generated_slips,
                "constraints": {"final_slips": 20}
            }
            
            try:
                spo_result = run_portfolio_optimization(spo_input)
                
                final_slips = spo_result.get("final_slips", [])
                logger.info(safe_log(
                    f"[{request_id}] [SPO] Optimization complete | "
                    f"Final slips: {len(final_slips)}"
                ))
                
                # Post SPO results to Laravel (Phase 2)
                spo_payload = {
                    "success": True,
                    "master_slip_id": master_slip_id,
                    "strategy": strategy_used,  # NEW: Include strategy
                    "spo_result": spo_result,
                    "request_id": request_id
                }
                
                if spo_callback_url:
                    logger.info(safe_log(f"[{request_id}] [PHASE 2] Posting SPO results to Laravel..."))
                    _post_sync(spo_callback_url, spo_payload, request_id=request_id)
                else:
                    logger.warning(safe_log(f"[{request_id}] [PHASE 2] No SPO callback URL, skipping"))
                    
            except Exception as e:
                logger.error(safe_log(
                    f"[{request_id}] [SPO] Optimization failed | "
                    f"Error: {str(e)}"
                ))
                logger.error(safe_log(f"[{request_id}] [SPO] Traceback: {traceback.format_exc()}"))
        else:
            logger.warning(safe_log(f"[{request_id}] [SPO] Portfolio optimizer not available, skipping"))
        
        logger.info(safe_log(f"[{request_id}] [BACKGROUND] Pipeline complete"))
        
    except Exception as e:
        logger.error(safe_log(
            f"[{request_id}] [BACKGROUND] Pipeline error | "
            f"Error: {str(e)}"
        ))
        logger.error(safe_log(f"[{request_id}] [BACKGROUND] Traceback: {traceback.format_exc()}"))

# ------------------------------
# Main slip generation endpoint (UPDATED FOR DUAL-STRATEGY)
# ------------------------------
@app.post("/generate-slips", response_model=EngineResponse)
async def generate_slips(payload: MasterSlipRequest, background_tasks: BackgroundTasks, request: Request):
    """
    Generate 50 intelligent betting slips with strategy selection.
    
    NEW in v2.1: Dual-Strategy Support
    - Include "strategy" field in master_slip payload
    - Options: "balanced" (default) or "maxwin"
    - Backward compatible: omitting strategy defaults to "balanced"
    
    Process:
    1. Receive Laravel payload with match data
    2. Extract and validate strategy selection
    3. Route to appropriate slip builder
    4. Parse and validate matches/markets
    5. Run Monte Carlo simulations (10,000 iterations)
    6. Generate 50 diverse slips across risk tiers
    7. Calculate coverage, confidence, and diversity metrics
    8. Add stake and possible_return to all slips
    9. Return slips immediately (non-blocking)
    10. Background: Post to Laravel callback
    11. Background: Run portfolio optimization
    12. Background: Post optimized results to Laravel
    """
    request_id = getattr(request.state, "request_id", f"gen_{int(time.time() * 1000)}")
    
    logger.info(safe_log(f"[{request_id}] ========== SLIP GENERATION STARTED =========="))
    
    if not ENGINE_AVAILABLE:
        logger.error(safe_log(f"[{request_id}] Engine not available - returning 503"))
        raise HTTPException(
            status_code=503,
            detail="Slip builder engine not available. Please check server logs."
        )
    
    try:
        # Step 1: Normalize Pydantic model to dict
        logger.info(safe_log(f"[{request_id}] [STEP 1] Normalizing payload..."))
        normalized_dict = normalize_payload_for_slip_builder(payload, endpoint="generate-slips")
        
        # Log payload summary for observability
        log_payload_summary(normalized_dict, request_id)
        
        # === DEFENSIVE VALIDATION ===
        logger.info(safe_log(f"[{request_id}] [VALIDATION] Starting defensive payload validation..."))
        
        master_slip = normalized_dict.get("master_slip")
        if not master_slip:
            logger.error(safe_log(f"[{request_id}] [VALIDATION] Missing master_slip - payload structure is wrong"))
            logger.error(safe_log(f"[{request_id}] [VALIDATION] Payload keys: {list(normalized_dict.keys())}"))
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid payload structure",
                    "detail": "Missing 'master_slip' key. Check Laravel payload builder.",
                    "received_keys": list(normalized_dict.keys())
                }
            )
        
        matches = master_slip.get("matches", [])
        if not matches:
            logger.error(safe_log(f"[{request_id}] [VALIDATION] Empty matches array"))
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No matches in payload",
                    "detail": "The 'matches' array is empty or missing."
                }
            )
        
        logger.info(safe_log(f"[{request_id}] [VALIDATION] Found {len(matches)} matches"))
        
        # Check first match has markets
        first_match = matches[0]
        if "match_markets" not in first_match and "markets" not in first_match:
            logger.error(safe_log(f"[{request_id}] [VALIDATION] First match missing market fields"))
            logger.error(safe_log(f"[{request_id}] [VALIDATION] Match keys: {list(first_match.keys())}"))
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Match missing markets",
                    "detail": f"Match {first_match.get('match_id')} has no markets",
                    "received_keys": list(first_match.keys())
                }
            )
        
        # Count markets for validation
        markets_key = "match_markets" if "match_markets" in first_match else "markets"
        markets_count = len(first_match.get(markets_key, []))
        logger.info(safe_log(f"[{request_id}] [VALIDATION] First match has {markets_count} markets"))
        
        if markets_count == 0:
            logger.warning(safe_log(f"[{request_id}] [VALIDATION] First match has zero markets - may cause issues"))
        
        logger.info(safe_log(f"[{request_id}] [VALIDATION] Defensive validation passed ✓"))
        # === END DEFENSIVE VALIDATION ===
        
        # Step 1a: Extract and validate strategy (NEW)
        logger.info(safe_log(f"[{request_id}] [STEP 1a] Extracting strategy..."))
        strategy = extract_and_validate_strategy(normalized_dict, request_id)
        
        # Log strategy decision
        if STRATEGY_FACTORY_AVAILABLE:
            strategy_info = get_strategy_info().get(strategy, {})
            logger.info(safe_log(f"[{request_id}] [STRATEGY] Selected: {strategy}"))
            logger.info(safe_log(f"[{request_id}] [STRATEGY] Name: {strategy_info.get('name', 'Unknown')}"))
            logger.info(safe_log(f"[{request_id}] [STRATEGY] Goal: {strategy_info.get('goal', 'N/A')}"))
        else:
            logger.info(safe_log(f"[{request_id}] [STRATEGY] Legacy mode: {strategy}"))
        
        # Extract master slip data for stake calculation
        master_slip_data = normalized_dict.get("master_slip", {})
        master_stake = master_slip_data.get("stake", 0)
        
        logger.info(safe_log(f"[{request_id}] [MASTER SLIP] Stake: {master_stake}"))
        
        # Step 2: Create strategy-specific builder (NEW)
        logger.info(safe_log(f"[{request_id}] [STEP 2] Creating slip builder..."))
        
        enable_mc = os.getenv("ENABLE_MONTE_CARLO", "true").lower() == "true"
        num_sims = int(os.getenv("NUM_SIMULATIONS", "10000"))
        
        builder = create_slip_builder(
            strategy=strategy,
            enable_monte_carlo=enable_mc,
            num_simulations=num_sims
        )
        
        logger.info(safe_log(f"[{request_id}] [BUILDER] Created {builder.__class__.__name__}"))
        logger.info(safe_log(f"[{request_id}] [BUILDER] Monte Carlo: {enable_mc}"))
        logger.info(safe_log(f"[{request_id}] [BUILDER] Simulations: {num_sims}"))
        
        # Step 3: Generate slips using selected strategy
        logger.info(safe_log(f"[{request_id}] [STEP 3] Generating slips with {strategy} strategy..."))
        logger.info(safe_log(f"[{request_id}] [PROCESS] Running Monte Carlo simulations..."))
        
        start_time = time.time()
        engine_result = builder.generate(normalized_dict)
        generation_time = time.time() - start_time
        
        logger.info(safe_log(
            f"[{request_id}] [STEP 3] Slip generation complete | "
            f"Duration: {generation_time:.4f}s"
        ))
        
        # Step 4: Extract results and add stake/possible_return
        metadata = engine_result.get("metadata", {})
        ms_id = metadata.get("master_slip_id", 0)
        generated_slips = engine_result.get("generated_slips", [])
        
        logger.info(safe_log(f"[{request_id}] [STEP 4] Extracting results..."))
        logger.info(safe_log(
            f"[{request_id}] [RESULTS] Master Slip ID: {ms_id} | "
            f"Slips Generated: {len(generated_slips)}"
        ))
        
        # Step 4a: Add stake and possible_return to all slips
        logger.info(safe_log(f"[{request_id}] [STEP 4a] Adding stake and possible_return to slips..."))
        
        slips_with_stake = 0
        slips_without_stake = 0
        
        for slip in generated_slips:
            # Add stake (use existing or master stake)
            if "stake" not in slip or slip.get("stake") == 0:
                slip["stake"] = master_stake
                slips_without_stake += 1
            else:
                slips_with_stake += 1
            
            # Calculate possible_return
            total_odds = slip.get("total_odds", 1)
            slip_stake = slip.get("stake", master_stake)
            
            if "possible_return" not in slip:
                slip["possible_return"] = round(slip_stake * total_odds, 2)
            else:
                # Recalculate to ensure consistency
                slip["possible_return"] = round(slip_stake * total_odds, 2)
        
        logger.info(safe_log(
            f"[{request_id}] [STAKE CALC] Slips with stake: {slips_with_stake} | "
            f"Slips using master stake: {slips_without_stake}"
        ))
        logger.info(safe_log(
            f"[{request_id}] [STAKE CALC] All {len(generated_slips)} slips now have stake and possible_return"
        ))
        
        # Add master slip data and strategy to metadata
        metadata["master_slip_data"] = master_slip_data
        metadata["strategy_used"] = strategy  # NEW: Track which strategy was used
        
        # Log portfolio metrics
        portfolio_metrics = metadata.get("portfolio_metrics", {})
        if portfolio_metrics:
            logger.info(safe_log(f"[{request_id}] [METRICS] Portfolio Metrics:"))
            
            # Log common metrics
            logger.info(safe_log(
                f"[{request_id}] [METRICS]   Coverage: {portfolio_metrics.get('coverage_percentage', 0)}%"
            ))
            logger.info(safe_log(
                f"[{request_id}] [METRICS]   Avg Wins/Sim: {portfolio_metrics.get('average_wins_per_simulation', 0):.2f}"
            ))
            logger.info(safe_log(
                f"[{request_id}] [METRICS]   Avg Confidence: {portfolio_metrics.get('average_confidence', 0):.3f}"
            ))
            logger.info(safe_log(
                f"[{request_id}] [METRICS]   Avg Diversity: {portfolio_metrics.get('average_diversity', 0):.3f}"
            ))
            
            # Log strategy-specific metrics (NEW for MaxWin)
            if "portfolio_ev" in portfolio_metrics:
                logger.info(safe_log(
                    f"[{request_id}] [METRICS]   Portfolio EV: ${portfolio_metrics.get('portfolio_ev', 0):.2f}"
                ))
                logger.info(safe_log(
                    f"[{request_id}] [METRICS]   Avg Slip EV: ${portfolio_metrics.get('average_slip_ev', 0):.2f}"
                ))
                logger.info(safe_log(
                    f"[{request_id}] [METRICS]   Positive EV Slips: {portfolio_metrics.get('positive_ev_count', 0)}/{len(generated_slips)}"
                ))
        
        # Log risk distribution
        risk_dist = metadata.get("risk_distribution", {})
        if risk_dist:
            logger.info(safe_log(f"[{request_id}] [METRICS] Risk Distribution:"))
            for level, count in risk_dist.items():
                logger.info(safe_log(f"[{request_id}] [METRICS]   {level.upper()}: {count} slips"))
        
        # Step 5: Setup callback URLs
        logger.info(safe_log(f"[{request_id}] [STEP 5] Setting up callbacks..."))
        
        # Phase 1: 50 generated slips callback
        p1_url = f"http://localhost:8000/api/python-callback/{ms_id}"
        
        # Phase 2: SPO optimized slips callback  
        spo_url = f"http://localhost:8000/api/spo-callback/{ms_id}"
        
        logger.info(safe_log(f"[{request_id}] [CALLBACK] Phase 1 URL (50 slips): {p1_url}"))
        logger.info(safe_log(f"[{request_id}] [CALLBACK] Phase 2 URL (SPO): {spo_url}"))
        
        # Step 6: Schedule background pipeline
        logger.info(safe_log(f"[{request_id}] [STEP 6] Scheduling background pipeline..."))
        
        # Update engine_result with modified slips
        engine_result["generated_slips"] = generated_slips
        engine_result["metadata"] = metadata
        
        background_tasks.add_task(
            _run_spo_and_callback,
            engine_result,
            p1_url,
            spo_url,
            ms_id,
            strategy,  # NEW: Pass strategy to background task
            request_id
        )
        
        logger.info(safe_log(f"[{request_id}] [BACKGROUND] Pipeline scheduled successfully"))
        
        # Step 7: Build response
        response = {
            "master_slip_id": ms_id,
            "generated_slips": generated_slips,
            "metadata": {
                **metadata,
                "request_id": request_id,
                "generation_time_seconds": round(generation_time, 4)
            },
            "status": "success",
            "generated_at": metadata.get("generated_at"),
            "total_slips": len(generated_slips)
        }
        
        logger.info(safe_log(f"[{request_id}] ========== SLIP GENERATION COMPLETE =========="))
        
        return response

    except PayloadValidationError as e:
        logger.error(safe_log(f"[{request_id}] [ERROR] Payload validation failed: {str(e)}"))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payload structure: {str(e)}"
        )
        
    except MarketIntegrityError as e:
        logger.error(safe_log(f"[{request_id}] [ERROR] Market integrity error: {str(e)}"))
        raise HTTPException(
            status_code=422,
            detail=f"Market data error: {str(e)}"
        )
        
    except SlipBuilderError as e:
        logger.error(safe_log(f"[{request_id}] [ERROR] Slip builder error: {str(e)}"))
        raise HTTPException(
            status_code=422,
            detail=f"Slip generation failed: {str(e)}"
        )
        
    except Exception as e:
        logger.error(safe_log(f"[{request_id}] [CRASH] Unexpected error: {str(e)}"))
        logger.error(safe_log(f"[{request_id}] [CRASH] Traceback: {traceback.format_exc()}"))
        raise HTTPException(
            status_code=500,
            detail=f"Internal engine failure: {str(e)}"
        )

# ------------------------------
# Match analysis endpoint
# ------------------------------
@app.post("/api/v1/analyze-match", response_model=AnalysisResponse)
async def analyze_match(request_data: AnalysisRequest, request: Request):
    """Analyze single match using insight engine"""
    request_id = getattr(request.state, "request_id", request_data.request_id or f"analyze_{int(time.time() * 1000)}")
    
    logger.info(safe_log(f"[{request_id}] [ANALYZE] Match analysis requested"))
    
    if not INSIGHT_ENGINE_AVAILABLE or not insight_engine:
        logger.warning(safe_log(f"[{request_id}] [ANALYZE] Insight engine not available"))
        return AnalysisResponse(
            status="error",
            error="Insight engine not available",
            request_id=request_id
        )
    
    try:
        logger.info(safe_log(f"[{request_id}] [ANALYZE] Running analysis..."))
        result = insight_engine.analyze_single_match(request_data.data)
        
        logger.info(safe_log(f"[{request_id}] [ANALYZE] Analysis complete"))
        
        return AnalysisResponse(
            status="success",
            analysis=result,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(safe_log(f"[{request_id}] [ANALYZE] Analysis failed: {str(e)}"))
        logger.error(safe_log(f"[{request_id}] [ANALYZE] Traceback: {traceback.format_exc()}"))
        
        return AnalysisResponse(
            status="error",
            error=str(e),
            request_id=request_id
        )

# ------------------------------
# Root endpoint
# ------------------------------
# Import betting router (educational/testing only)
try:
    from .routers.betting import router as betting_router
    BETTING_ROUTER_AVAILABLE = True
    logger.info(safe_log("[OK] Betting router loaded successfully"))
    app.include_router(betting_router)
except ImportError as e:
    BETTING_ROUTER_AVAILABLE = False
    logger.warning(safe_log(f"[WARN] Betting router not available: {e}"))

@app.get("/")
async def root():
    """Service information endpoint"""
    return {
        "service": "Intelligent Football Slip Builder API",
        "version": ENGINE_VERSION,
        "status": "operational" if ENGINE_AVAILABLE else "degraded",
        "description": "Dual-strategy slip generation (Balanced + MaxWin) with Monte Carlo optimization",
        "strategy_mode": "dual-strategy" if STRATEGY_FACTORY_AVAILABLE else "legacy-single-strategy",
        "available_strategies": get_available_strategies() if STRATEGY_FACTORY_AVAILABLE else ["balanced"],
        "betting_automation": "available" if BETTING_ROUTER_AVAILABLE else "unavailable",
        "documentation": "/docs",
        "health_check": "/health",
        "engine_info": "/engine-info",
        "strategies_info": "/strategies"
    }

# ------------------------------
# Application startup
# ------------------------------
@app.on_event("startup")
async def startup_event():
    """Log application startup"""
    logger.info(safe_log("=" * 80))
    logger.info(safe_log("[START] Application startup complete"))
    logger.info(safe_log(f"[START] Engine Version: {ENGINE_VERSION}"))
    logger.info(safe_log(f"[START] Engine Available: {ENGINE_AVAILABLE}"))
    logger.info(safe_log(f"[START] Strategy Mode: {'Dual-Strategy' if STRATEGY_FACTORY_AVAILABLE else 'Legacy'}"))
    if STRATEGY_FACTORY_AVAILABLE:
        strategies = get_available_strategies()
        logger.info(safe_log(f"[START] Available Strategies: {', '.join(strategies)}"))
    logger.info(safe_log(f"[START] SPO Available: {RUN_SPO_AVAILABLE}"))
    logger.info(safe_log(f"[START] Insight Engine Available: {INSIGHT_ENGINE_AVAILABLE}"))
    logger.info(safe_log("[START] Ready to accept requests"))
    logger.info(safe_log("=" * 80))

@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown"""
    logger.info(safe_log("=" * 80))
    logger.info(safe_log("[SHUTDOWN] Application shutting down"))
    logger.info(safe_log("=" * 80))

# ------------------------------
# Run application
# ------------------------------
if __name__ == "__main__":
    logger.info(safe_log("[START] Starting uvicorn server..."))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )