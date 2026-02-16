"""
Example FastAPI Integration for Slip Builder v2.0
==================================================

This demonstrates how to integrate the slip builder into your FastAPI application.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from slip_builder import SlipBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI
app = FastAPI(
    title="Football Slip Builder API",
    version="2.0.0",
    description="Intelligent betting slip generation using Monte Carlo optimization"
)

# Initialize slip builder (reuse instance for performance)
slip_builder = SlipBuilder(
    enable_monte_carlo=True,
    num_simulations=10000
)


# Request/Response Models
class GenerateSlipsRequest(BaseModel):
    """Request model for slip generation"""
    master_slip: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "master_slip": {
                    "master_slip_id": 123,
                    "stake": 100,
                    "currency": "USD",
                    "matches": [
                        {
                            "match_id": 69,
                            "home_team": "Atlas",
                            "away_team": "Pachuca",
                            "league": "Mexico - Liga MX",
                            "match_date": "2026-01-07",
                            "match_markets": [
                                {
                                    "market": {
                                        "code": "1x2",
                                        "name": "Full Time Result"
                                    },
                                    "selections": [
                                        {"value": "Home", "odds": 6.8},
                                        {"value": "Draw", "odds": 4.9},
                                        {"value": "Away", "odds": 1.32}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        }


class GenerateSlipsResponse(BaseModel):
    """Response model for slip generation"""
    success: bool
    generated_slips: list
    metadata: Dict[str, Any]
    error: str | None = None


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Slip Builder API",
        "version": "2.0.0"
    }


@app.post("/api/v2/generate-slips", response_model=GenerateSlipsResponse)
async def generate_slips(request: GenerateSlipsRequest):
    """
    Generate 50 optimized betting slips from match data.
    
    This endpoint:
    1. Receives match data from Laravel
    2. Runs Monte Carlo simulations
    3. Generates 50 diverse slips
    4. Returns slip data with metrics
    
    Args:
        request: GenerateSlipsRequest containing master_slip data
    
    Returns:
        GenerateSlipsResponse with 50 slips and metadata
    """
    try:
        # Convert request to dict
        payload = request.model_dump()
        
        # Generate slips
        result = slip_builder.generate(payload)
        
        # Return success response
        return GenerateSlipsResponse(
            success=True,
            generated_slips=result["generated_slips"],
            metadata=result["metadata"],
            error=None
        )
        
    except Exception as e:
        # Log error
        logging.error(f"Slip generation failed: {str(e)}", exc_info=True)
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "message": "Failed to generate slips. Check payload format."
            }
        )


@app.post("/api/v2/generate-slips-fast")
async def generate_slips_fast(request: GenerateSlipsRequest):
    """
    Fast slip generation (without Monte Carlo).
    
    Use this endpoint when:
    - Speed is critical (< 1 second response time)
    - You don't need coverage metrics
    - Testing or development
    
    Note: Coverage scores will be 0 without Monte Carlo.
    """
    try:
        # Create fast builder (no Monte Carlo)
        fast_builder = SlipBuilder(enable_monte_carlo=False)
        
        # Generate slips
        payload = request.model_dump()
        result = fast_builder.generate(payload)
        
        return GenerateSlipsResponse(
            success=True,
            generated_slips=result["generated_slips"],
            metadata=result["metadata"],
            error=None
        )
        
    except Exception as e:
        logging.error(f"Fast slip generation failed: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "message": "Failed to generate slips."
            }
        )


# For running directly (development)
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "example_fastapi_integration:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
    # Access at: http://localhost:8000/docs