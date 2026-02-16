from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from .engine.insight_engine import MatchInsightEngine

app = FastAPI(title="Freedom Train AI - Insight API")
engine = MatchInsightEngine()

class LaravelPayload(BaseModel):
    success: bool
    data: Dict[str, Any]

@app.post("/api/v1/predict-match")
async def predict_match(payload: LaravelPayload):
    """
    Refined endpoint that consumes Laravel match data and 
    returns a high-confidence prediction.
    """
    try:
        match_data = payload.data
        
        # Data Normalization: Ensure IDs exist
        if "id" not in match_data:
            raise ValueError("Match ID missing from payload")

        # Run the Analysis Pipeline
        prediction = engine.analyze_single_match(match_data)
        
        return {
            "status": "success",
            "match_id": match_data["id"],
            "data": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)