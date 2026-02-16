"""
Callback Service

Handles communication with Laravel backend:
- Posting results to callback URLs
- Running portfolio optimization
- Managing background tasks
"""

import logging
import httpx
from typing import Dict, Any, Optional

from ..config import CALLBACK_TIMEOUT, LARAVEL_BASE_URL
from ..exceptions import CallbackError, SPOError

logger = logging.getLogger("engine_api.services")


class CallbackService:
    """Service for handling callbacks to Laravel backend."""
    
    def __init__(self, timeout: float = CALLBACK_TIMEOUT):
        self.timeout = timeout
        self._spo_available = False
        self._initialize_spo()
    
    def _initialize_spo(self):
        """Check if portfolio optimization is available."""
        try:
            from ..engine import run_portfolio_optimization
            self._run_spo = run_portfolio_optimization
            self._spo_available = True
            logger.info("[SERVICE] Portfolio optimization available")
        except ImportError:
            self._spo_available = False
            logger.warning("[SERVICE] Portfolio optimization not available")
    
    def post_callback(
        self,
        url: str,
        payload: Dict[str, Any],
        request_id: str = "unknown"
    ) -> bool:
        """
        Post results to Laravel callback URL.
        
        Args:
            url: Callback URL
            payload: Data to post
            request_id: Request identifier for logging
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            CallbackError: If callback fails critically
        """
        try:
            logger.info(
                f"[{request_id}] [CALLBACK] Posting to {url}..."
            )
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                
                if response.status_code in [200, 201]:
                    logger.info(
                        f"[{request_id}] [CALLBACK] SUCCESS | "
                        f"URL: {url} | "
                        f"Status: {response.status_code}"
                    )
                    return True
                else:
                    logger.warning(
                        f"[{request_id}] [CALLBACK] UNEXPECTED STATUS | "
                        f"URL: {url} | "
                        f"Status: {response.status_code} | "
                        f"Response: {response.text[:200]}"
                    )
                    return False
                    
        except httpx.TimeoutException:
            logger.error(
                f"[{request_id}] [CALLBACK] TIMEOUT | URL: {url}"
            )
            raise CallbackError(
                f"Callback timeout after {self.timeout}s",
                url=url
            )
        except Exception as e:
            logger.error(
                f"[{request_id}] [CALLBACK] FAILED | "
                f"URL: {url} | "
                f"Error: {str(e)}"
            )
            raise CallbackError(
                f"Callback failed: {str(e)}",
                url=url
            )
    
    def build_phase1_payload(
        self,
        engine_result: Dict[str, Any],
        master_slip_id: int,
        strategy_used: str
    ) -> Dict[str, Any]:
        """
        Build Phase 1 payload (50 generated slips) for Laravel.
        
        Args:
            engine_result: Result from slip generation
            master_slip_id: Master slip identifier
            strategy_used: Strategy that was used
            
        Returns:
            Laravel-compatible payload
        """
        metadata = engine_result.get("metadata", {})
        generated_slips = engine_result.get("generated_slips", [])
        
        return {
            "success": True,
            "generated_slips": generated_slips,
            "metadata": {
                "master_slip_id": master_slip_id,
                "strategy": strategy_used,
                "input_matches": metadata.get("input_matches", 0),
                "dropped_matches_count": 0,
                "dropped_match_ids": [],
                "registered_markets": metadata.get("unique_markets", 0),
                "unmapped_markets": 0,
                "malformed_markets": 0,
                "duplicate_markets": 0,
                "duplicate_matches": 0,
                "deterministic": True,
                "strict_mode": False,
                "engine_version": metadata.get("engine_version", "2.2.0"),
                "phase": "phase1",
                "total_slips": len(generated_slips),
                "spo_queued": self._spo_available
            }
        }
    
    def run_portfolio_optimization(
        self,
        generated_slips: list,
        bankroll: float = 1000.0,
        target_slips: int = 20,
        request_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Run portfolio optimization on generated slips.
        
        Args:
            generated_slips: List of generated slips
            bankroll: Available bankroll
            target_slips: Target number of optimized slips
            request_id: Request identifier for logging
            
        Returns:
            Optimization result
            
        Raises:
            SPOError: If optimization fails
        """
        if not self._spo_available:
            raise SPOError("Portfolio optimization not available")
        
        try:
            logger.info(
                f"[{request_id}] [SPO] Starting optimization | "
                f"Input: {len(generated_slips)} slips | "
                f"Target: {target_slips} slips"
            )
            
            spo_input = {
                "bankroll": bankroll,
                "generated_slips": generated_slips,
                "constraints": {"final_slips": target_slips}
            }
            
            result = self._run_spo(spo_input)
            
            final_slips = result.get("final_slips", [])
            logger.info(
                f"[{request_id}] [SPO] Optimization complete | "
                f"Final slips: {len(final_slips)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"[{request_id}] [SPO] Optimization failed | "
                f"Error: {str(e)}"
            )
            raise SPOError(f"Portfolio optimization failed: {str(e)}")
    
    def build_spo_payload(
        self,
        spo_result: Dict[str, Any],
        master_slip_id: int,
        strategy_used: str,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Build SPO payload for Laravel callback.
        
        Args:
            spo_result: Result from portfolio optimization
            master_slip_id: Master slip identifier
            strategy_used: Strategy that was used
            request_id: Request identifier
            
        Returns:
            Laravel-compatible SPO payload
        """
        return {
            "success": True,
            "master_slip_id": master_slip_id,
            "strategy": strategy_used,
            "spo_result": spo_result,
            "request_id": request_id
        }
    
    def is_spo_available(self) -> bool:
        """Check if portfolio optimization is available."""
        return self._spo_available
    
    def get_callback_urls(self, master_slip_id: int) -> Dict[str, str]:
        """
        Get callback URLs for a master slip.
        
        Args:
            master_slip_id: Master slip identifier
            
        Returns:
            Dictionary with phase1 and spo callback URLs
        """
        base_url = LARAVEL_BASE_URL.rstrip("/")
        
        return {
            "phase1": f"{base_url}/api/python-callback/{master_slip_id}",
            "spo": f"{base_url}/api/spo-callback/{master_slip_id}"
        }
