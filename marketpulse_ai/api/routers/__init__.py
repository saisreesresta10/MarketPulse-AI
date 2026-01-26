"""
API Routers Package

Modular API routers for different functional areas.
"""

from .insights import router as insights_router
from .recommendations import router as recommendations_router
from .data import router as data_router
from .workflows import router as workflows_router

__all__ = [
    'insights_router',
    'recommendations_router', 
    'data_router',
    'workflows_router'
]