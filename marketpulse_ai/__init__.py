"""
MarketPulse AI - AI-powered decision-support copilot for India's MRP-based retail ecosystem.

This package provides comprehensive retail analytics and decision support while ensuring
compliance with India's Maximum Retail Price (MRP) regulations.
"""

__version__ = "0.1.0"
__author__ = "MarketPulse AI Team"
__description__ = "AI-powered decision-support copilot for Indian retail"

from .core.models import (
    SalesDataPoint,
    DemandPattern,
    ExplainableInsight,
    RiskAssessment,
    Scenario,
    ComplianceResult,
)

__all__ = [
    "SalesDataPoint",
    "DemandPattern", 
    "ExplainableInsight",
    "RiskAssessment",
    "Scenario",
    "ComplianceResult",
]