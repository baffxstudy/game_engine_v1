"""
Health Check and System Information Endpoints

Provides health checks, system status, and engine information.
"""

import logging
import os
from datetime import datetime
from fastapi import APIRouter

from ..config import (
    ENGINE_VERSION,
    LOG_DIR,
    ENABLE_MONTE_CARLO,
    NUM_SIMULATIONS,
    FEATURE_SPO_ENABLED,
    FEATURE_INSIGHT_ENGINE
)
from ..engine import get_engine_status

logger = logging.getLogger("engine_api.routers")
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint for monitoring.
    
    Returns:
        - Service status
        - Engine availability
        - Feature availability
        - System information
    """
    try:
        # Check engine availability
        engine_status = get_engine_status()
        engine_available = engine_status.get("initialized", False)
        
        # Check log directory
        log_dir_status = "exists" if os.path.exists(LOG_DIR) else "missing"
        
        # Get available strategies
        available_strategies = []
        try:
            from ..services import SlipService
            slip_service = SlipService()
            available_strategies = slip_service.get_available_strategies()
        except Exception as e:
            logger.warning(f"Failed to get strategies: {e}")
        
        health_response = {
            "status": "operational" if engine_available else "degraded",
            "service": "intelligent-football-slip-builder",
            "engine_version": ENGINE_VERSION,
            "engine_available": engine_available,
            "engine_status": "healthy" if engine_available else "unavailable",
            "engine_details": engine_status,
            "strategy_mode": "multi-strategy" if len(available_strategies) > 1 else "legacy",
            "available_strategies": available_strategies,
            "insight_engine": "available" if FEATURE_INSIGHT_ENGINE else "unavailable",
            "spo_available": FEATURE_SPO_ENABLED,
            "log_dir": log_dir_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "endpoints": {
                "/api/v1/generate-slips": "POST - Generate 50 intelligent betting slips",
                "/api/v1/strategies": "GET - List available strategies",
                "/api/v1/analyze-match": "POST - Analyze single match",
                "/health": "GET - Health check",
                "/engine-info": "GET - Detailed engine information",
                "/docs": "GET - Interactive API documentation"
            }
        }
        
        logger.info(f"[HEALTH] Health check requested - Status: {health_response['status']}")
        
        return health_response
        
    except Exception as e:
        logger.error(f"[HEALTH] Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


@router.get("/engine-info")
async def engine_info():
    """
    Detailed engine information endpoint.
    
    Returns comprehensive information about:
    - Engine capabilities
    - Available strategies
    - Configuration
    - Feature flags
    """
    try:
        engine_status = get_engine_status()
        
        base_info = {
            "name": "Intelligent Football Slip Builder Engine",
            "version": ENGINE_VERSION,
            "description": (
                "Multi-strategy slip generation with Monte Carlo optimization "
                "and portfolio diversification"
            ),
            "engine_available": engine_status.get("initialized", False),
            "strategy_factory_available": len(engine_status.get("available_strategies", [])) > 1,
            "insight_engine_available": FEATURE_INSIGHT_ENGINE,
            "spo_available": FEATURE_SPO_ENABLED,
        }
        
        if engine_status.get("initialized"):
            base_info.update({
                "engine_type": engine_status.get("engine_type", "Unknown"),
                "initialized": True,
                "monte_carlo_enabled": ENABLE_MONTE_CARLO,
                "num_simulations": NUM_SIMULATIONS,
                "features": engine_status.get("features", []),
                "capabilities": {
                    "slips_per_request": 50,
                    "deterministic": True,
                    "monte_carlo_optimization": ENABLE_MONTE_CARLO,
                    "portfolio_diversification": True,
                    "hedging_strategies": True,
                    "coverage_optimization": True,
                    "fault_tolerant": True,
                    "multi_strategy_support": len(engine_status.get("available_strategies", [])) > 1,
                    "risk_levels": ["LOW", "MEDIUM", "HIGH"],
                    "metrics": [
                        "confidence_score",
                        "coverage_score",
                        "diversity_score",
                        "win_probability",
                        "expected_value"
                    ]
                }
            })
            
            # Add strategy information
            if "available_strategies" in engine_status:
                base_info["strategies"] = engine_status.get("strategy_details", {})
                base_info["available_strategies"] = engine_status.get("available_strategies", [])
        else:
            base_info["features"] = ["Engine module not available"]
            base_info["capabilities"] = {}
        
        logger.info("[ENGINE INFO] Engine info requested")
        
        return base_info
        
    except Exception as e:
        logger.error(f"[ENGINE INFO] Failed to get engine info: {e}", exc_info=True)
        return {
            "error": str(e),
            "version": ENGINE_VERSION
        }
