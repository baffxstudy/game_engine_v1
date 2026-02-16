"""
EDUCATIONAL BETTING AUTOMATION FRAMEWORK
For testing and educational purposes only.
Always comply with local laws and website terms of service.

Betting automation API endpoints.
"""

import logging
from fastapi import APIRouter, HTTPException

from ..models import BettingRequest, BettingResult
from ..services.bet_placer import BetPlacer

logger = logging.getLogger("engine_api.routers")
router = APIRouter(prefix="/api/v1", tags=["betting"])


@router.post("/place-bet", response_model=BettingResult)
async def place_bet(request: BettingRequest):
    """
    Place a bet using browser automation.
    
    **Educational/Testing Purpose Only**
    
    This endpoint simulates bet placement using browser automation.
    In test mode (default), it simulates the process without actual submission.
    
    **Process:**
    1. Opens browser with anti-detection measures
    2. Navigates to betting site
    3. Adds selections to bet slip
    4. Sets stake amount
    5. Places bet (simulated in test mode)
    6. Takes screenshot for verification
    7. Returns result with bet reference
    
    **Test Mode:**
    - Default: `test_mode=True`
    - Simulates bet placement
    - Returns simulated bet reference
    - Takes screenshots for verification
    
    **Production Mode:**
    - Set `test_mode=False`
    - WARNING: Not implemented - for educational purposes only
    - Would require site-specific implementation
    
    **Example Request:**
    ```json
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
    
    **Example Response:**
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
    """
    try:
        logger.info(f"[BETTING] Bet placement requested: {request.slip.slip_id}")
        
        placer = BetPlacer(test_mode=request.test_mode)
        result = await placer.place_bet(request.slip)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=422,
                detail=result.get("error", "Bet placement failed")
            )
        
        logger.info(f"[BETTING] Bet placement successful: {request.slip.slip_id}")
        return BettingResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[BETTING] Bet placement error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Bet placement failed: {str(e)}"
        )


@router.get("/betting/health")
async def betting_health_check():
    """
    Health check endpoint for betting automation service.
    
    Returns:
        Service status and configuration
    """
    from ..config import BETTING_TEST_MODE, TEST_SITE_URL, SCREENSHOTS_DIR
    
    return {
        "status": "healthy",
        "service": "betting-automation",
        "test_mode": BETTING_TEST_MODE,
        "test_site_url": TEST_SITE_URL,
        "screenshots_dir": str(SCREENSHOTS_DIR),
        "message": "Betting automation service is operational"
    }
