# game_engine/app.py

import uvicorn
import time
import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Request
from .schemas import MasterSlipRequest, EngineResponse
from .engine import SlipBuilder
from .engine.insight_engine import MatchInsightEngine
from pydantic import BaseModel
from typing import Dict, Any, Optional

# --- ROBUST LOGGING SETUP ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log_file = os.path.join(LOG_DIR, "engine.log")

# Setup Rotating File Handler (Max 5MB per file, keeps last 5 files)
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Setup Console Handler (to see logs in your terminal)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Apply configuration
logger = logging.getLogger("engine_logger")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = FastAPI(title="Football Game Engine")
builder = SlipBuilder()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"{method} {path} | Status: {response.status_code} | Time: {duration:.4f}s")
        return response
    except Exception as e:
        logger.error(f"Middleware caught crash: {str(e)}", exc_info=True)
        raise

@app.post("/generate-slips", response_model=EngineResponse)
async def generate_slips(payload: MasterSlipRequest):
    ms_id = payload.master_slip.master_slip_id
    logger.info(f"--- Starting Generation for Master Slip: {ms_id} ---")
    
    try:
        # Pass the payload to the orchestrator
        results = builder.generate(payload)
        
        logger.info(f"Successfully generated 100 slips for {ms_id}")
        return {
            "master_slip_id": ms_id,
            "generated_slips": results
        }
    except Exception as e:
        # This captures the EXACT line number in slip_builder.py where it fails
        logger.error(f"Generation Failed for {ms_id} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")

@app.post("/api/v1/analyze-match")
async def analyze_match(request: Request):
    # Receive the Laravel JSON
    raw_payload = await request.json()
    
    # Senior Analyst Decision: Strip the 'success' and 'data' wrappers 
    # to feed only the core match attributes to the engine.
    match_data = raw_payload.get("data", {})
    
    if not match_data:
        return {"error": "Invalid Data Structure"}

    # Execute Analysis
    result = insight_engine.analyze_single_match(match_data)
    
    return {
        "status": "success",
        "analysis": result
    }

if __name__ == "__main__":
    uvicorn.run("game_engine.app:app", host="0.0.0.0", port=5000, reload=True)