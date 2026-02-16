# game_engine/schemas.py

from pydantic import BaseModel, Field, ConfigDict, validator
from typing import List, Dict, Any, Optional, Union
from decimal import Decimal
import re
from datetime import datetime

class FlexibleIDMixin:
    """Mixin to handle flexible ID types (string or integer)"""
    
    @validator('match_id', 'master_slip_id', 'original_master_slip_id', 'team_id', 
               'slip_id', 'match_id', pre=True, check_fields=False)
    def convert_to_string(cls, v):
        """Convert any ID value to string"""
        if v is None:
            return ""
        return str(v)

class SlipLeg(BaseModel, FlexibleIDMixin):
    match_id: str = Field(..., description="Match ID as string")
    market: str
    selection: str
    odds: float
    is_fallback: bool = False
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

class GeneratedSlip(BaseModel, FlexibleIDMixin):
    slip_id: str = Field(..., description="Slip ID as string")
    legs: List[SlipLeg]
    total_odds: float
    confidence_score: float
    stake: float = Field(default=0.0)
    possible_return: float = Field(default=0.0)
    risk_level: str = Field(default="Unknown Risk")
    error: Optional[str] = None
    variation_type: Optional[str] = None
    edge_score: Optional[float] = Field(default=0.0)
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

class MatchData(BaseModel, FlexibleIDMixin):
    """Flexible match data model accepting both string and integer IDs"""
    match_id: str
    home_team: str
    away_team: str
    venue: str = "Neutral"
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    selected_market: Optional[Dict[str, Any]] = None
    full_markets: List[Dict[str, Any]] = Field(default_factory=list)
    team_form: Optional[Dict[str, Any]] = None
    head_to_head: Optional[Dict[str, Any]] = None
    model_inputs: Optional[Dict[str, Any]] = None
    probabilities: Optional[Dict[str, float]] = None
    
    @validator('match_id', 'home_team_id', 'away_team_id', pre=True)
    def normalize_ids(cls, v):
        if v is None:
            return ""
        return str(v)
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

class MasterSlipData(BaseModel, FlexibleIDMixin):
    """Flexible master slip data model"""
    master_slip_id: str
    original_master_slip_id: Optional[str] = None
    stake: float = Field(ge=0.0)
    currency: str = "EUR"
    matches: List[MatchData]
    
    @validator('master_slip_id', 'original_master_slip_id', pre=True)
    def normalize_slip_ids(cls, v):
        if v is None:
            return ""
        return str(v)
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

class MasterSlipRequest(BaseModel):
    master_slip: Dict[str, Any]
    
    # Fix for Pydantic protected namespace warning
    model_config = ConfigDict(
        protected_namespaces=()  # Allow 'model_' prefix in field names
    )

# NEW: Analysis-related schemas
class MatchAnalysis(BaseModel):
    """Analysis result for a single match"""
    match_id: str
    predictions: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    recommended_markets: List[str] = Field(default_factory=list)
    risk_assessment: str = "unknown"
    insights: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

class AnalysisRequest(BaseModel):
    """Request model for match analysis endpoint"""
    data: Dict[str, Any] = Field(..., description="Match data to analyze")
    request_id: Optional[str] = None
    analysis_type: str = "full"
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

class AnalysisResponse(BaseModel, FlexibleIDMixin):
    """Response model for match analysis endpoint"""
    status: str = "success"
    analysis: Optional[MatchAnalysis] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    processing_time: Optional[float] = None
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

class EngineResponse(BaseModel, FlexibleIDMixin):
    master_slip_id: str
    generated_slips: List[GeneratedSlip]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    status: Optional[str] = "success"
    generated_at: Optional[str] = None
    total_slips: Optional[int] = None
    error: Optional[str] = None
    
    @validator('master_slip_id', pre=True)
    def normalize_master_slip_id(cls, v):
        if v is None:
            return ""
        return str(v)
    
    @validator('generated_at', pre=True)
    def format_timestamp(cls, v):
        if v is None:
            return datetime.utcnow().isoformat() + "Z"
        return v
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

# NEW: Debug and health check schemas
class HealthCheckResponse(BaseModel):
    status: str
    service: str
    engine_version: str
    engine_status: str
    insight_engine: str
    log_dir: str
    timestamp: float
    endpoints: Dict[str, str]
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

class EngineInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    features: List[str]
    capabilities: Dict[str, Any]
    
    model_config = ConfigDict(
        protected_namespaces=()
    )

class DebugPayloadResponse(BaseModel):
    request_id: str
    analysis: Dict[str, Any]
    raw_payload_type: str
    raw_payload_preview: Union[str, Dict[str, Any]]
    
    model_config = ConfigDict(
        protected_namespaces=()
    )