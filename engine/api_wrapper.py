"""
FastAPI wrapper for SlipBuilder
"""
import json
import logging
from typing import Dict, Any, Optional
from .slip_builder import SlipBuilder, process_slip_builder_payload

logger = logging.getLogger(__name__)

class SlipBuilderAPI:
    """FastAPI-compatible wrapper for SlipBuilder"""
    
    def __init__(self):
        """Initialize without payload (for FastAPI lifespan)"""
        self.slip_builder = None
        logger.info("✅ SlipBuilderAPI initialized (empty)")
    
    def set_payload(self, payload: Dict[str, Any]) -> None:
        """Set payload for slip generation"""
        if self.slip_builder is None:
            self.slip_builder = SlipBuilder(payload)
        else:
            self.slip_builder.set_payload(payload)
    
    def generate_slips(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate slips from payload
        
        Args:
            payload: Optional payload. If None, uses existing payload
            
        Returns:
            Dict with generated slips
        """
        try:
            if payload is not None:
                self.set_payload(payload)
            
            if self.slip_builder is None or not self.slip_builder.matches:
                raise ValueError("No payload set. Provide a payload or set one first.")
            
            # Use the existing process_slip_builder_payload function
            return process_slip_builder_payload(self.slip_builder.payload)
            
        except Exception as e:
            logger.error(f"Error in generate_slips: {str(e)}")
            # Return emergency response
            return {
                'status': 'error',
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'total_slips': 50,
                'generated_slips': [],
                'metadata': {
                    'error': str(e),
                    'note': 'Emergency fallback triggered'
                }
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'service': 'SlipBuilder',
            'version': '1.0.0',
            'has_payload': self.slip_builder is not None and bool(self.slip_builder.matches),
            'matches_loaded': len(self.slip_builder.matches) if self.slip_builder else 0
        }