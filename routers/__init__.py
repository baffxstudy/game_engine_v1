"""
API routers for the Football Match Analysis Engine.

Separates endpoints into logical groups for better organization.
"""

from .slips import router as slips_router
from .analysis import router as analysis_router
from .health import router as health_router

__all__ = [
    "slips_router",
    "analysis_router",
    "health_router",
]
