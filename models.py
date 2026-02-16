"""
EDUCATIONAL BETTING AUTOMATION FRAMEWORK
For testing and educational purposes only.
Always comply with local laws and website terms of service.

Pydantic models for betting automation framework.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class BetLeg(BaseModel):
    """Represents a single bet leg/selection."""
    match_id: str = Field(..., description="Match identifier")
    selection: str = Field(..., description="Betting selection (e.g., 'Home Win', 'Over 2.5')")
    odds: float = Field(..., gt=0, description="Odds for this selection")
    market_type: str = Field(..., description="Market type (e.g., 'MATCH_RESULT', 'OVER_UNDER')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "match_id": "MATCH_123",
                "selection": "Home Win",
                "odds": 2.5,
                "market_type": "MATCH_RESULT"
            }
        }


class BetSlip(BaseModel):
    """Represents a complete betting slip."""
    slip_id: str = Field(..., description="Unique slip identifier")
    legs: List[BetLeg] = Field(..., min_items=1, description="List of bet legs")
    stake: float = Field(..., gt=0, description="Stake amount")
    total_odds: float = Field(..., gt=0, description="Combined odds for all legs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "slip_id": "SLIP_001",
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
            }
        }


class BettingRequest(BaseModel):
    """Request model for placing a bet."""
    slip: BetSlip = Field(..., description="Betting slip to place")
    test_mode: bool = Field(True, description="Whether to run in test mode (simulated)")
    
    class Config:
        json_schema_extra = {
            "example": {
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
                "test_mode": True
            }
        }


class BettingResult(BaseModel):
    """Response model for bet placement result."""
    success: bool = Field(..., description="Whether bet placement was successful")
    slip_id: str = Field(..., description="Slip identifier")
    timestamp: Optional[str] = Field(None, description="Timestamp of bet placement")
    stake: Optional[float] = Field(None, description="Stake amount")
    total_odds: Optional[float] = Field(None, description="Total odds")
    potential_return: Optional[float] = Field(None, description="Potential return amount")
    screenshot: Optional[str] = Field(None, description="Path to screenshot")
    test_mode: Optional[bool] = Field(None, description="Whether test mode was used")
    error: Optional[str] = Field(None, description="Error message if failed")
    message: Optional[str] = Field(None, description="Additional message")
    bet_reference: Optional[str] = Field(None, description="Bet reference number")
    simulated: Optional[bool] = Field(None, description="Whether bet was simulated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "slip_id": "TEST_001",
                "timestamp": "20260119_123456",
                "stake": 10.0,
                "total_odds": 2.0,
                "potential_return": 20.0,
                "screenshot": "screenshots/bet_TEST_001_20260119_123456.png",
                "test_mode": True,
                "simulated": True,
                "bet_reference": "SIM_TEST_001_1234567890"
            }
        }
