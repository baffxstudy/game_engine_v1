"""
COMPOSITION SLIPS API ROUTES

Endpoint:
  POST /slips/compose - Compose base + optimized slips into composed slips
"""

import logging
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from typing import Dict, Any

from services.composition_slips_service import CompositionSlipService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["composition"])


@router.post("/slips/compose")
async def compose_slips(
    payload: Dict[str, Any],
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Compose slips endpoint.
    
    Input:
    {
        "master_slip": {
            "master_slip_id": 12345,
            "base_slips": [...50 slips...],
            "optimized_slips": [...20 slips...],
            "composition_slips": {
                "enabled": true,
                "source": {"from": ["base_slips", "optimized_slips"]},
                "targets": {"count": 50, "min_matches": 6, "max_matches": 14},
                ...
            }
        }
    }
    
    Output:
    {
        "success": true,
        "base_slips": [...],
        "optimized_slips": [...],
        "composition_slips": [...50 composed slips...],
        "metadata": {
            "master_slip_id": 12345,
            "total_composed": 50,
            ...
        }
    }
    """
    request_id = getattr(request.state, "request_id", f"compose_{id(request)}")
    
    logger.info(f"[{request_id}] POST /slips/compose - Composition requested")
    
    try:
        # Extract slips from payload
        master_slip = payload.get("master_slip", {})
        base_slips = master_slip.get("base_slips", [])
        optimized_slips = master_slip.get("optimized_slips", [])
        
        logger.info(
            f"[{request_id}] Received: base={len(base_slips)}, optimized={len(optimized_slips)}"
        )
        
        # Run service
        service = CompositionSlipService()
        success, composed_slips, message = service.run(payload, base_slips, optimized_slips)
        
        if not success:
            logger.error(f"[{request_id}] Composition failed: {message}")
            raise HTTPException(
                status_code=422,
                detail={"error": message, "request_id": request_id}
            )
        
        # Build response
        response = {
            "success": True,
            "request_id": request_id,
            "master_slip_id": master_slip.get("master_slip_id"),
            "base_slips": base_slips,
            "optimized_slips": optimized_slips,
            "composition_slips": composed_slips,
            "metadata": {
                "total_composed": len(composed_slips),
                "composition_enabled": True,
                "message": message
            }
        }
        
        logger.info(
            f"[{request_id}] Composition successful: {len(composed_slips)} slips generated"
        )
        
        return response
    
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": str(e), "request_id": request_id}
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "request_id": request_id}
        )