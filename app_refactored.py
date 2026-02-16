"""
Intelligent Football Slip Builder API - Refactored Main Application

This is the brain of the football match analysis platform:
- Performs all calculations and intelligence
- Does not manage UI or database persistence directly
- Receives clean, structured input from Laravel
- Returns pure analytical results

Architecture:
- React.js → UI only (displays results)
- Laravel → Orchestration layer (database, API endpoints, calls Python)
- Python Backend → This service (all calculations and intelligence)
"""

import uvicorn
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Configure logging FIRST (before other imports)
from .logging_config import setup_logging, safe_log
from .config import (
    ENGINE_VERSION,
    API_TITLE,
    API_DESCRIPTION,
    API_HOST,
    API_PORT,
    ENABLE_MONTE_CARLO,
    NUM_SIMULATIONS,
    validate_config
)

# Setup logging
logger = setup_logging()

logger.info(safe_log("=" * 80))
logger.info(safe_log(f"[START] Football Game Engine v{ENGINE_VERSION} Initializing..."))
logger.info(safe_log("[START] Multi-Strategy Support: Balanced + MaxWin + Compound"))
logger.info(safe_log("=" * 80))

# Validate configuration
try:
    validate_config()
    logger.info(safe_log("[OK] Configuration validated"))
except ValueError as e:
    logger.error(safe_log(f"[ERROR] Configuration validation failed: {e}"))
    raise

# Import schemas with fallback
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

# Import engine with error handling
ENGINE_AVAILABLE = False
RUN_SPO_AVAILABLE = False

logger.info(safe_log("[PROCESS] Loading engine modules..."))

try:
    from .engine import (
        initialize_engine,
        get_engine_status,
        run_portfolio_optimization,
        SlipBuilderError,
        PayloadValidationError,
        MarketIntegrityError,
    )
    
    ENGINE_AVAILABLE = True
    RUN_SPO_AVAILABLE = callable(run_portfolio_optimization)
    
    logger.info(safe_log("[OK] Engine module loaded successfully"))
    logger.info(safe_log(f"[OK] Engine version: {ENGINE_VERSION}"))
    logger.info(safe_log(f"[OK] Portfolio optimization: {'AVAILABLE' if RUN_SPO_AVAILABLE else 'NOT AVAILABLE'}"))
    
except ImportError as e:
    logger.error(safe_log(f"[ERROR] Engine module import failed: {e}"))
    logger.error(safe_log(f"[ERROR] Traceback: {traceback.format_exc()}"))
    
    # Fallback exception classes (use our custom exceptions as fallback)
    from .exceptions import (
        SlipBuilderError,
        PayloadValidationError,
        MarketIntegrityError,
    )

# Initialize the engine if available
if ENGINE_AVAILABLE:
    try:
        logger.info(safe_log("[PROCESS] Initializing intelligent slip builder..."))
        
        initialize_engine(
            enable_monte_carlo=ENABLE_MONTE_CARLO,
            num_simulations=NUM_SIMULATIONS
        )
        
        # Get and log engine status
        engine_status = get_engine_status()
        logger.info(safe_log("[OK] Engine initialized successfully"))
        logger.info(safe_log(f"[CONFIG] Engine Type: {engine_status.get('engine_type', 'Unknown')}"))
        logger.info(safe_log(f"[CONFIG] Monte Carlo: {'ENABLED' if ENABLE_MONTE_CARLO else 'DISABLED'}"))
        logger.info(safe_log(f"[CONFIG] Simulations: {NUM_SIMULATIONS}"))
        
        # Log features
        features = engine_status.get('features', [])
        if features:
            logger.info(safe_log("[FEAT] Engine Features:"))
            for feature in features:
                logger.info(safe_log(f"[FEAT]   - {feature}"))
        
        # Log strategy info
        available_strategies = engine_status.get('available_strategies', [])
        if available_strategies:
            logger.info(safe_log(f"[STRATEGY] Available Strategies: {', '.join(available_strategies)}"))
            
    except Exception as e:
        logger.warning(safe_log(f"[WARN] Engine initialization failed: {e}"))
        logger.warning(safe_log(f"[WARN] Traceback: {traceback.format_exc()}"))
else:
    logger.error(safe_log("[ERROR] Engine not available - API will return 503 errors"))

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=ENGINE_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

logger.info(safe_log("[OK] FastAPI application created"))

# Import routers
from .routers import slips_router, analysis_router, health_router

# Import betting router (educational/testing only)
try:
    from .routers.betting import router as betting_router
    BETTING_ROUTER_AVAILABLE = True
    logger.info(safe_log("[OK] Betting router loaded successfully"))
except ImportError as e:
    BETTING_ROUTER_AVAILABLE = False
    logger.warning(safe_log(f"[WARN] Betting router not available: {e}"))

# Register routers
app.include_router(slips_router)
app.include_router(analysis_router)
app.include_router(health_router)

# Register betting router if available
if BETTING_ROUTER_AVAILABLE:
    app.include_router(betting_router)
    logger.info(safe_log("[OK] Betting router registered"))

# Import and add middleware
from .middleware import RequestLoggingMiddleware
app.add_middleware(RequestLoggingMiddleware)

# Exception handlers
@app.exception_handler(PayloadValidationError)
async def payload_validation_handler(request, exc: PayloadValidationError):
    """Handle payload validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(safe_log(f"[{request_id}] PayloadValidationError: {str(exc)}"))
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Payload validation failed",
            "detail": str(exc),
            "status_code": 400,
            "request_id": request_id
        }
    )


@app.exception_handler(MarketIntegrityError)
async def market_integrity_handler(request, exc: MarketIntegrityError):
    """Handle market integrity errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(safe_log(f"[{request_id}] MarketIntegrityError: {str(exc)}"))
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Market data integrity error",
            "detail": str(exc),
            "status_code": 422,
            "request_id": request_id
        }
    )


@app.exception_handler(SlipBuilderError)
async def slip_builder_handler(request, exc: SlipBuilderError):
    """Handle slip builder errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(safe_log(f"[{request_id}] SlipBuilderError: {str(exc)}"))
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Slip generation failed",
            "detail": str(exc),
            "status_code": 422,
            "request_id": request_id
        }
    )


# Root endpoint
@app.get("/")
async def root():
    """Service information endpoint."""
    from .services import SlipService
    
    try:
        slip_service = SlipService()
        available_strategies = slip_service.get_available_strategies()
    except Exception:
        available_strategies = ["balanced"]
    
    return {
        "service": API_TITLE,
        "version": ENGINE_VERSION,
        "status": "operational" if ENGINE_AVAILABLE else "degraded",
        "description": API_DESCRIPTION,
        "strategy_mode": "multi-strategy" if len(available_strategies) > 1 else "legacy",
        "available_strategies": available_strategies,
        "betting_automation": "available" if BETTING_ROUTER_AVAILABLE else "unavailable",
        "documentation": "/docs",
        "health_check": "/health",
        "engine_info": "/engine-info",
        "strategies_info": "/api/v1/strategies"
    }


# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info(safe_log("=" * 80))
    logger.info(safe_log("[START] Application startup complete"))
    logger.info(safe_log(f"[START] Engine Version: {ENGINE_VERSION}"))
    logger.info(safe_log(f"[START] Engine Available: {ENGINE_AVAILABLE}"))
    
    if ENGINE_AVAILABLE:
        try:
            from .services import SlipService
            slip_service = SlipService()
            strategies = slip_service.get_available_strategies()
            logger.info(safe_log(f"[START] Available Strategies: {', '.join(strategies)}"))
        except Exception:
            pass
    
    logger.info(safe_log(f"[START] SPO Available: {RUN_SPO_AVAILABLE}"))
    logger.info(safe_log("[START] Ready to accept requests"))
    logger.info(safe_log("=" * 80))


@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info(safe_log("=" * 80))
    logger.info(safe_log("[SHUTDOWN] Application shutting down"))
    logger.info(safe_log("=" * 80))


# Run application
if __name__ == "__main__":
    logger.info(safe_log("[START] Starting uvicorn server..."))
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
