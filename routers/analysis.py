"""
Match Analysis Endpoints

Handles match analysis requests:
- POST /api/v1/analyze-match - Analyze single match
"""

import logging
import time
from fastapi import APIRouter, Request
from typing import Dict, Any

try:
    from ..schemas import AnalysisRequest, AnalysisResponse
except ImportError:
    from pydantic import BaseModel
    from typing import Dict, Any, Optional
    
    class AnalysisRequest(BaseModel):
        data: Dict[str, Any]
        request_id: Optional[str] = None
    
    class AnalysisResponse(BaseModel):
        status: str = "success"
        analysis: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        request_id: Optional[str] = None

from ..config import FEATURE_INSIGHT_ENGINE

logger = logging.getLogger("engine_api.routers")
router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze-match", response_model=AnalysisResponse)
async def analyze_match(
    request_data: AnalysisRequest,
    request: Request
):
    """
    Analyze single match using insight engine.
    
    Performs comprehensive analysis including:
    - Team form analysis
    - Head-to-head analysis
    - Market probability calculations
    - Confidence scoring
    """
    request_id = getattr(
        request.state,
        "request_id",
        request_data.request_id or f"analyze_{int(time.time() * 1000)}"
    )
    
    logger.info(f"[{request_id}] [ANALYZE] Match analysis requested")
    
    if not FEATURE_INSIGHT_ENGINE:
        logger.warning(f"[{request_id}] [ANALYZE] Insight engine not available")
        return AnalysisResponse(
            status="error",
            error="Insight engine not available",
            request_id=request_id
        )
    
    try:
        # Import insight engine
        from ..engine.insight_engine import MatchInsightEngine
        
        logger.info(f"[{request_id}] [ANALYZE] Running analysis...")
        
        insight_engine = MatchInsightEngine()
        result = insight_engine.analyze_single_match(request_data.data)
        
        logger.info(f"[{request_id}] [ANALYZE] Analysis complete")
        
        return AnalysisResponse(
            status="success",
            analysis=result,
            request_id=request_id
        )
        
    except ImportError:
        logger.error(f"[{request_id}] [ANALYZE] Insight engine module not available")
        return AnalysisResponse(
            status="error",
            error="Insight engine module not available",
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(
            f"[{request_id}] [ANALYZE] Analysis failed: {str(e)}",
            exc_info=True
        )
        
        return AnalysisResponse(
            status="error",
            error=str(e),
            request_id=request_id
        )
