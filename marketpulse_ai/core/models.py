"""
Core data models for MarketPulse AI using Pydantic for validation.

These models define the fundamental data structures used throughout the system
for sales data, insights, risk assessments, and compliance validation.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ConfidenceLevel(str, Enum):
    """Confidence levels for insights and predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    """Risk levels for inventory and demand assessments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance status for MRP regulation validation."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"


class SalesDataPoint(BaseModel):
    """
    Represents a single sales data point with validation.
    
    This model captures essential sales information including product details,
    quantities, pricing, and temporal context for analysis.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the data point")
    product_id: str = Field(..., min_length=1, description="Product identifier")
    product_name: str = Field(..., min_length=1, description="Product name")
    category: str = Field(..., min_length=1, description="Product category")
    mrp: Decimal = Field(..., gt=0, description="Maximum Retail Price in INR")
    selling_price: Decimal = Field(..., gt=0, description="Actual selling price in INR")
    quantity_sold: int = Field(..., ge=0, description="Quantity sold")
    sale_date: date = Field(..., description="Date of sale")
    store_location: str = Field(..., min_length=1, description="Store location identifier")
    seasonal_event: Optional[str] = Field(None, description="Associated seasonal event or festival")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    
    @field_validator('selling_price')
    @classmethod
    def validate_selling_price_against_mrp(cls, v, info):
        """Ensure selling price does not exceed MRP (Indian regulation compliance)."""
        if 'mrp' in info.data and v > info.data['mrp']:
            raise ValueError(f"Selling price {v} cannot exceed MRP {info.data['mrp']}")
        return v
    
    @field_validator('sale_date')
    @classmethod
    def validate_sale_date(cls, v):
        """Ensure sale date is not in the future."""
        if v > date.today():
            raise ValueError("Sale date cannot be in the future")
        return v

    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            UUID: str,
        }
    )


class DemandPattern(BaseModel):
    """
    Represents identified demand patterns from sales data analysis.
    
    Captures seasonal trends, cyclical patterns, and demand characteristics
    for inventory planning and decision support.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the pattern")
    product_id: str = Field(..., min_length=1, description="Associated product identifier")
    pattern_type: str = Field(..., min_length=1, description="Type of pattern (seasonal, cyclical, trend)")
    description: str = Field(..., min_length=1, description="Human-readable pattern description")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in pattern identification")
    seasonal_factors: Dict[str, float] = Field(default_factory=dict, description="Seasonal adjustment factors")
    trend_direction: Optional[str] = Field(None, description="Overall trend direction (increasing, decreasing, stable)")
    volatility_score: float = Field(..., ge=0, le=1, description="Demand volatility score (0-1)")
    supporting_data_points: int = Field(..., gt=0, description="Number of data points supporting this pattern")
    date_range_start: date = Field(..., description="Start date of pattern analysis")
    date_range_end: date = Field(..., description="End date of pattern analysis")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Pattern creation timestamp")
    
    @field_validator('date_range_end')
    @classmethod
    def validate_date_range(cls, v, info):
        """Ensure end date is after start date."""
        if 'date_range_start' in info.data and v <= info.data['date_range_start']:
            raise ValueError("End date must be after start date")
        return v

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
        }
    )


class ExplainableInsight(BaseModel):
    """
    Represents an explainable insight generated from data analysis.
    
    Provides human-readable explanations with supporting evidence and
    confidence levels for business decision support.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the insight")
    title: str = Field(..., min_length=1, description="Insight title")
    description: str = Field(..., min_length=1, description="Detailed insight description")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in the insight")
    
    @property
    def confidence(self) -> float:
        """Get numeric confidence value from confidence level."""
        confidence_mapping = {
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.HIGH: 0.9
        }
        return confidence_mapping.get(self.confidence_level, 0.5)
    supporting_evidence: List[str] = Field(..., min_length=1, description="Supporting evidence points")
    key_factors: List[str] = Field(..., min_length=1, description="Key influencing factors")
    business_impact: str = Field(..., min_length=1, description="Potential business impact")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    data_sources: List[str] = Field(..., min_length=1, description="Data sources used")
    related_products: List[str] = Field(default_factory=list, description="Related product identifiers")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Insight creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Insight expiration timestamp")
    
    @field_validator('expires_at')
    @classmethod
    def validate_expiration(cls, v, info):
        """Ensure expiration is in the future if provided."""
        if v is not None and 'created_at' in info.data and v <= info.data['created_at']:
            raise ValueError("Expiration date must be after creation date")
        return v

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
        }
    )


class RiskAssessment(BaseModel):
    """
    Represents inventory and demand risk assessment results.
    
    Captures risk levels, scores, and early warning indicators for
    inventory management and decision support.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the assessment")
    product_id: str = Field(..., min_length=1, description="Associated product identifier")
    risk_type: str = Field(..., min_length=1, description="Type of risk (overstock, understock, demand_volatility)")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_score: float = Field(..., ge=0, le=1, description="Numerical risk score (0-1)")
    contributing_factors: List[str] = Field(..., min_length=1, description="Factors contributing to risk")
    seasonal_adjustments: Dict[str, float] = Field(default_factory=dict, description="Seasonal risk adjustments")
    early_warning_triggered: bool = Field(default=False, description="Whether early warning is active")
    mitigation_suggestions: List[str] = Field(default_factory=list, description="Risk mitigation suggestions")
    assessment_date: date = Field(default_factory=date.today, description="Date of assessment")
    valid_until: date = Field(..., description="Assessment validity end date")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Assessment creation timestamp")
    
    @field_validator('valid_until')
    @classmethod
    def validate_validity_period(cls, v, info):
        """Ensure validity period is in the future."""
        if 'assessment_date' in info.data and v <= info.data['assessment_date']:
            raise ValueError("Validity end date must be after assessment date")
        return v

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
        }
    )


class Scenario(BaseModel):
    """
    Represents a what-if scenario for decision support analysis.
    
    Captures scenario parameters, assumptions, and predicted outcomes
    for strategic planning and decision making.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the scenario")
    name: str = Field(..., min_length=1, description="Scenario name")
    description: str = Field(..., min_length=1, description="Scenario description")
    parameters: Dict[str, Any] = Field(..., description="Scenario parameters and assumptions")
    predicted_outcomes: Dict[str, Any] = Field(..., description="Predicted scenario outcomes")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in predictions")
    assumptions: List[str] = Field(..., min_length=1, description="Key assumptions made")
    limitations: List[str] = Field(..., min_length=1, description="Scenario limitations")
    time_horizon: str = Field(..., min_length=1, description="Time horizon for scenario (e.g., '3 months', '1 year')")
    affected_products: List[str] = Field(default_factory=list, description="Products affected by scenario")
    seasonal_considerations: List[str] = Field(default_factory=list, description="Seasonal factors considered")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Scenario creation timestamp")
    
    @model_validator(mode='after')
    def validate_scenario_completeness(self):
        """Ensure scenario has meaningful parameters and outcomes."""
        if not self.parameters:
            raise ValueError("Scenario must have at least one parameter")
        if not self.predicted_outcomes:
            raise ValueError("Scenario must have at least one predicted outcome")
        
        return self

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
        }
    )


class ComplianceResult(BaseModel):
    """
    Represents MRP regulation compliance validation results.
    
    Captures compliance status, violations, and regulatory constraints
    for ensuring legal compliance in recommendations.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the compliance result")
    recommendation_id: Optional[UUID] = Field(None, description="Associated recommendation identifier")
    compliance_status: ComplianceStatus = Field(..., description="Overall compliance status")
    regulations_checked: List[str] = Field(..., min_length=1, description="Regulations validated against")
    violations: List[str] = Field(default_factory=list, description="Identified violations")
    warnings: List[str] = Field(default_factory=list, description="Compliance warnings")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Regulatory constraints")
    validation_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed validation results")
    validator_version: str = Field(..., min_length=1, description="Version of compliance validator used")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")
    
    @model_validator(mode='after')
    def validate_status_consistency(self):
        """Ensure compliance status is consistent with violations."""
        violations = self.violations or []
        
        if self.compliance_status == ComplianceStatus.COMPLIANT and violations:
            raise ValueError("Cannot be compliant with violations present")
        if self.compliance_status == ComplianceStatus.NON_COMPLIANT and not violations:
            raise ValueError("Non-compliant status requires violations")
        
        return self

    model_config = ConfigDict(
        json_encoders={
            UUID: str,
        }
    )