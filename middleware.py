"""
Middleware for request logging and error handling.

Provides:
- Request/response logging
- Request ID generation
- Error handling
- Performance metrics
"""

import logging
import time
import os
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

from .config import ENGINE_VERSION

logger = logging.getLogger("engine_api.middleware")


def safe_log(message: str) -> str:
    """Convert Unicode emojis to ASCII text for Windows compatibility."""
    replacements = {
        '✅': '[OK]',
        '🚀': '[START]',
        '🔧': '[PROCESS]',
        '📊': '[DATA]',
        '🔍': '[ANALYZE]',
        '⚠️': '[WARN]',
        '❌': '[ERROR]',
        '🎯': '[TARGET]',
        '📦': '[FEAT]',
        '📍': '[LOC]',
        '📝': '[LOG]',
        '📁': '[FILE]',
        '🔌': '[CONNECT]',
        '⚙️': '[CONFIG]',
        '💥': '[CRASH]',
        '🔒': '[SECURE]',
        '📈': '[STATS]',
        '🔍': '[SEARCH]',
        '💰': '[MONEY]',
        '⚽': '[FOOTBALL]',
        '🎰': '[CASINO]',
    }
    for emoji, text in replacements.items():
        message = message.replace(emoji, text)
    return message


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}_{os.urandom(4).hex()}"
        path = request.url.path
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        
        # Store request ID in request state
        request.state.request_id = request_id
        
        # Log request start
        logger.info(safe_log("=" * 80))
        logger.info(safe_log(
            f"[{request_id}] {method} {path} | "
            f"Client: {client_ip} | "
            f"Started at: {datetime.utcnow().isoformat()}"
        ))
        logger.info(safe_log("=" * 80))
        
        try:
            # Process request
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful completion
            logger.info(safe_log(
                f"[{request_id}] COMPLETED | "
                f"Status: {response.status_code} | "
                f"Duration: {duration:.4f}s | "
                f"Path: {path}"
            ))
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Engine-Version"] = ENGINE_VERSION
            response.headers["X-Processing-Time"] = f"{duration:.4f}"
            
            logger.info(safe_log("=" * 80))
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(safe_log("=" * 80))
            logger.error(safe_log(
                f"[{request_id}] CRASHED | "
                f"Error: {str(e)} | "
                f"Duration: {duration:.4f}s | "
                f"Path: {path}"
            ))
            logger.error(safe_log(f"[{request_id}] Traceback:"))
            logger.exception(e)
            logger.error(safe_log("=" * 80))
            
            # Return error response
            import json
            return Response(
                content=json.dumps({
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": str(e)
                }),
                status_code=500,
                media_type="application/json",
                headers={
                    "X-Request-ID": request_id,
                    "X-Engine-Version": ENGINE_VERSION
                }
            )
