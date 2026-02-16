"""
Slip Generation Endpoints

Handles all slip generation related endpoints:
- POST /generate-slips - Generate betting slips
- GET /strategies - List available strategies
"""

import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from datetime import datetime

try:
    from ..schemas import MasterSlipRequest, EngineResponse
except ImportError:
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

from ..services import SlipService, CallbackService, ValidationService

# Import exceptions - try engine first, fallback to our exceptions
try:
    from ..engine import (
        SlipBuilderError,
        PayloadValidationError,
        MarketIntegrityError,
    )
except ImportError:
    from ..exceptions import (
        SlipBuilderError,
        PayloadValidationError,
        MarketIntegrityError,
    )

from ..exceptions import (
    StrategyError,
    MatchCountError,
)

from ..config import DEFAULT_SLIP_COUNT

logger = logging.getLogger("engine_api.routers")
router = APIRouter(prefix="/api/v1", tags=["slips"])

# Initialize services
slip_service = SlipService()
callback_service = CallbackService()
validation_service = ValidationService()


def _run_background_pipeline(
    engine_result: Dict[str, Any],
    phase1_url: str,
    spo_url: str,
    master_slip_id: int,
    strategy_used: str,
    request_id: str
):
    """
    Background task to:
    1. Post Phase 1 (50 slips) results to Laravel
    2. Run Portfolio Optimization (SPO)
    3. Post Phase 2 (20 optimized slips) results to Laravel
    """
    try:
        logger.info(f"[{request_id}] [BACKGROUND] Starting pipeline")

        # Phase 1: Post 50 generated slips
        if phase1_url:
            logger.info(f"[{request_id}] [PHASE 1] Posting results...")
            phase1_payload = callback_service.build_phase1_payload(
                engine_result,
                master_slip_id,
                strategy_used
            )
            callback_service.post_callback(phase1_url, phase1_payload, request_id)

        # Phase 2: Run SPO if available
        if callback_service.is_spo_available():
            logger.info(f"[{request_id}] [PHASE 2] Running portfolio optimization...")

            generated_slips = engine_result.get("generated_slips", [])
            spo_result = callback_service.run_portfolio_optimization(
                generated_slips,
                bankroll=1000.0,
                target_slips=20,
                request_id=request_id
            )

            # Post SPO results
            if spo_url:
                logger.info(f"[{request_id}] [PHASE 2] Posting SPO results...")
                spo_payload = callback_service.build_spo_payload(
                    spo_result,
                    master_slip_id,
                    strategy_used,
                    request_id
                )
                callback_service.post_callback(spo_url, spo_payload, request_id)
        else:
            logger.warning(f"[{request_id}] [SPO] Not available, skipping")

        logger.info(f"[{request_id}] [BACKGROUND] Pipeline complete")

    except Exception as e:
        logger.error(
            f"[{request_id}] [BACKGROUND] Pipeline error: {str(e)}",
            exc_info=True
        )


@router.post("/generate-slips", response_model=EngineResponse)
async def generate_slips(
    payload: MasterSlipRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Generate 50 intelligent betting slips with strategy selection.

    Backwards-compatible: will accept responses from SlipBuilder v3.0 which
    include additional metadata keys such as engine_version and stake_recommendations.
    """
    request_id = getattr(request.state, "request_id", f"gen_{int(time.time() * 1000)}")

    logger.info(f"[{request_id}] ========== SLIP GENERATION STARTED ==========")

    try:
        # Step 1: Validate payload
        logger.info(f"[{request_id}] [STEP 1] Validating payload...")
        normalized_payload = validation_service.validate_master_slip_payload(
            payload,
            request_id
        )

        master_slip_data = normalized_payload.get("master_slip", {})
        master_slip_id = master_slip_data.get("master_slip_id", 0)
        master_stake = master_slip_data.get("stake", 0)

        # Step 2: Extract strategy
        logger.info(f"[{request_id}] [STEP 2] Extracting strategy...")
        strategy = slip_service.extract_strategy(normalized_payload)

        # Step 3: Generate slips
        logger.info(f"[{request_id}] [STEP 3] Generating slips...")
        engine_result = slip_service.generate_slips(
            normalized_payload,
            strategy=strategy,
            request_id=request_id
        )

        # Ensure metadata exists and set sensible defaults for v3.0
        metadata = engine_result.get("metadata") or {}
        # Prefer any engine-provided version, otherwise assume 3.0.0 (newer builder)
        metadata.setdefault("engine_version", metadata.get("engine_version", "3.0.0"))
        # Preserve stake recommendations if produced by SlipBuilder v3.0
        stake_recs = metadata.get("stake_recommendations", {})
        total_recommended = metadata.get("total_recommended_stake", sum(stake_recs.values()) if isinstance(stake_recs, dict) else 0.0)
        metadata["total_recommended_stake"] = round(total_recommended, 2)
        engine_result["metadata"] = metadata

        logger.debug(f"[{request_id}] Engine metadata: {metadata}")

        # Step 4: Enrich slips with stake (master stake blending)
        logger.info(f"[{request_id}] [STEP 4] Enriching slips with stake...")
        generated_slips = engine_result.get("generated_slips", [])
        enriched_slips = slip_service.enrich_slips_with_stake(
            generated_slips,
            master_stake,
            request_id
        )

        # If engine already provided stake_recommendations, reconcile/attach to enriched slips
        if metadata.get("stake_recommendations"):
            # Attach recommended stake per slip where applicable (non-destructive)
            recs = metadata.get("stake_recommendations", {})
            for s in enriched_slips:
                sid = s.get("slip_id")
                if sid and sid in recs:
                    s.setdefault("recommended_stake", recs[sid])

        # Step 5: Update result with enriched slips
        engine_result["generated_slips"] = enriched_slips
        metadata["master_slip_data"] = master_slip_data
        engine_result["metadata"] = metadata

        # Step 6: Setup background tasks
        logger.info(f"[{request_id}] [STEP 5] Setting up background tasks...")
        callback_urls = callback_service.get_callback_urls(master_slip_id)

        background_tasks.add_task(
            _run_background_pipeline,
            engine_result,
            callback_urls.get("phase1"),
            callback_urls.get("spo"),
            master_slip_id,
            strategy,
            request_id
        )

        # Step 7: Build response (backward-compatible structure)
        response = {
            "master_slip_id": master_slip_id,
            "generated_slips": enriched_slips,
            "metadata": {
                **metadata,
                "request_id": request_id
            },
            "status": "success",
            "generated_at": metadata.get("generated_at"),
            "total_slips": len(enriched_slips)
        }

        logger.info(f"[{request_id}] ========== SLIP GENERATION COMPLETE ==========")

        return response

    except PayloadValidationError as e:
        logger.error(f"[{request_id}] [ERROR] Payload validation failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payload structure: {str(e)}"
        )

    except MatchCountError as e:
        logger.error(f"[{request_id}] [ERROR] Match count error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except MarketIntegrityError as e:
        logger.error(f"[{request_id}] [ERROR] Market integrity error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Market data error: {str(e)}"
        )

    except StrategyError as e:
        logger.error(f"[{request_id}] [ERROR] Strategy error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Strategy error: {str(e)}"
        )

    except SlipBuilderError as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] [ERROR] Slip builder error: {error_msg}")

        # Provide helpful error message for Compound strategy EV filtering
        if "No valid selections for Compound strategy after EV filtering" in error_msg:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Compound strategy cannot generate slips with current match data",
                    "reason": "No selections passed Expected Value (EV) filtering criteria",
                    "suggestion": (
                        "The Compound strategy requires selections with favorable Expected Value. "
                        "Try using 'balanced' or 'maxwin' strategy instead, or ensure matches have "
                        "sufficient favorable odds and probabilities."
                    ),
                    "request_id": request_id
                }
            )

        raise HTTPException(
            status_code=422,
            detail=f"Slip generation failed: {error_msg}"
        )

    except Exception as e:
        logger.error(
            f"[{request_id}] [CRASH] Unexpected error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal engine failure: {str(e)}"
        )


@router.get("/strategies")
async def list_strategies():
    """
    List all available slip generation strategies and their details.

    Returns information about each strategy including:
    - Name and description
    - Optimization goals
    - Risk profile
    - Expected performance characteristics
    - Minimum match requirements
    """
    try:
        strategies = slip_service.get_strategy_info()
        available = slip_service.get_available_strategies()

        logger.info(f"[STRATEGIES] Strategy info requested - {len(available)} strategies")

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