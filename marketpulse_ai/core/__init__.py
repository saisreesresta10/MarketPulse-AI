"""
Core module containing fundamental data models and interfaces for MarketPulse AI.
"""

from .models import (
    SalesDataPoint,
    DemandPattern,
    ExplainableInsight,
    RiskAssessment,
    Scenario,
    ComplianceResult,
)
from .interfaces import (
    DataProcessorInterface,
    InsightGeneratorInterface,
    RiskAssessorInterface,
    ComplianceValidatorInterface,
    ScenarioAnalyzerInterface,
    DecisionSupportEngineInterface,
)

__all__ = [
    "SalesDataPoint",
    "DemandPattern",
    "ExplainableInsight", 
    "RiskAssessment",
    "Scenario",
    "ComplianceResult",
    "DataProcessorInterface",
    "InsightGeneratorInterface",
    "RiskAssessorInterface",
    "ComplianceValidatorInterface",
    "ScenarioAnalyzerInterface",
    "DecisionSupportEngineInterface",
]