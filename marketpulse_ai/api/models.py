"""
API Request and Response Models

Pydantic models for API request validation and response formatting.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict
from ..core.models import SalesDataPoint, DemandPattern, Scenario, RiskAssessment, ComplianceResult, ExplainableInsight


class PriorityLevel(str, Enum):
    """Priority levels for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AssessmentType(str, Enum):
    """Types of risk assessment."""
    OVERSTOCK = "overstock"
    UNDERSTOCK = "understock"
    BOTH = "both"


class AnalysisType(str, Enum):
    """Types of scenario analysis."""
    INVENTORY = "inventory"
    DISCOUNT = "discount"
    FULL = "full"


# Request Models
class SalesDataRequest(BaseModel):
    """Request model for sales data ingestion."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": [
                    {
                        "product_id": "PROD001",
                        "date": "2024-01-15",
                        "quantity_sold": 150,
                        "revenue": 15000.0,
                        "inventory_level": 500,
                        "price": 100.0
                    }
                ],
                "validate_data": True,
                "store_patterns": True
            }
        }
    )
    
    data: List[Dict[str, Any]] = Field(..., description="List of sales data points")
    validate_data: bool = Field(True, description="Whether to validate data quality")
    store_patterns: bool = Field(True, description="Whether to store extracted patterns")
    
    @field_validator('data')
    @classmethod
    def validate_data_not_empty(cls, v):
        if not v:
            raise ValueError("Sales data cannot be empty")
        return v


class InsightRequest(BaseModel):
    """Request model for insight generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_ids": ["PROD001", "PROD002"],
                "include_seasonal": True,
                "confidence_threshold": 0.8,
                "max_insights": 10
            }
        }
    )
    
    product_ids: Optional[List[str]] = Field(None, description="Optional list of product IDs")
    include_seasonal: bool = Field(True, description="Include seasonal analysis")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_insights: Optional[int] = Field(None, ge=1, le=100, description="Maximum number of insights to return")


class RecommendationRequest(BaseModel):
    """Request model for recommendation generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_ids": ["PROD001", "PROD002"],
                "business_context": {
                    "target_margin": 0.25,
                    "inventory_turnover_target": 12,
                    "seasonal_events": ["Diwali", "Christmas"]
                },
                "priority_filter": "high",
                "include_compliance_check": True,
                "max_recommendations": 20
            }
        }
    )
    
    product_ids: List[str] = Field(..., description="List of product IDs for recommendations")
    business_context: Optional[Dict[str, Any]] = Field(None, description="Additional business context")
    priority_filter: Optional[PriorityLevel] = Field(None, description="Filter by priority level")
    include_compliance_check: bool = Field(True, description="Include compliance validation")
    max_recommendations: Optional[int] = Field(None, ge=1, le=50, description="Maximum recommendations to return")
    
    @field_validator('product_ids')
    @classmethod
    def validate_product_ids_not_empty(cls, v):
        if not v:
            raise ValueError("Product IDs list cannot be empty")
        return v


class ScenarioRequest(BaseModel):
    """Request model for scenario analysis."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "base_parameters": {
                    "product_id": "PROD001",
                    "current_inventory": 1000,
                    "demand_forecast": 200,
                    "seasonal_events": ["Diwali"],
                    "discount_range": [0.1, 0.3]
                },
                "scenario_types": ["optimistic", "pessimistic", "base"],
                "include_seasonal": True,
                "max_scenarios": 5
            }
        }
    )
    
    base_parameters: Dict[str, Any] = Field(..., description="Base scenario parameters")
    scenario_types: Optional[List[str]] = Field(None, description="Types of scenarios to generate")
    include_seasonal: bool = Field(True, description="Include seasonal modeling")
    max_scenarios: Optional[int] = Field(None, ge=1, le=20, description="Maximum scenarios to generate")
    
    @field_validator('base_parameters')
    @classmethod
    def validate_base_parameters_not_empty(cls, v):
        if not v:
            raise ValueError("Base parameters cannot be empty")
        return v


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": "PROD001",
                "current_inventory": 500,
                "assessment_type": "both",
                "include_seasonal_adjustment": True,
                "upcoming_events": ["Diwali", "Christmas"]
            }
        }
    )
    
    product_id: str = Field(..., description="Product ID to assess")
    current_inventory: int = Field(..., ge=0, description="Current inventory level")
    assessment_type: AssessmentType = Field(AssessmentType.BOTH, description="Type of assessment")
    include_seasonal_adjustment: bool = Field(True, description="Include seasonal risk adjustments")
    upcoming_events: Optional[List[str]] = Field(None, description="Upcoming seasonal events")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance validation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendation": {
                    "product_id": "PROD001",
                    "discount_percentage": 0.15,
                    "new_price": 85.0,
                    "original_price": 100.0,
                    "strategy": "seasonal_discount"
                },
                "product_category": "electronics",
                "include_explanations": True
            }
        }
    )
    
    recommendation: Dict[str, Any] = Field(..., description="Recommendation to validate")
    product_category: Optional[str] = Field(None, description="Product category for specific rules")
    include_explanations: bool = Field(True, description="Include constraint explanations")
    
    @field_validator('recommendation')
    @classmethod
    def validate_recommendation_not_empty(cls, v):
        if not v:
            raise ValueError("Recommendation cannot be empty")
        return v


class BulkOperationRequest(BaseModel):
    """Request model for bulk operations."""
    operation_type: str = Field(..., description="Type of bulk operation")
    items: List[Dict[str, Any]] = Field(..., description="List of items to process")
    batch_size: Optional[int] = Field(10, ge=1, le=100, description="Batch processing size")
    continue_on_error: bool = Field(False, description="Continue processing if individual items fail")
    
    @field_validator('items')
    @classmethod
    def validate_items_not_empty(cls, v):
        if not v:
            raise ValueError("Items list cannot be empty")
        return v


# Response Models
class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
    execution_time_ms: Optional[float] = Field(None, description="Request execution time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    success: bool = Field(True, description="Whether the request was successful")
    data: List[Any] = Field(..., description="Response data items")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    message: Optional[str] = Field(None, description="Response message")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")


class BulkOperationResponse(BaseModel):
    """Bulk operation response model."""
    success: bool = Field(..., description="Whether the overall operation was successful")
    total_items: int = Field(..., description="Total number of items processed")
    successful_items: int = Field(..., description="Number of successfully processed items")
    failed_items: int = Field(..., description="Number of failed items")
    results: List[Dict[str, Any]] = Field(..., description="Individual item results")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details for failed items")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")


# Specialized Response Models
class InsightResponse(BaseModel):
    """Response model for insight generation."""
    insights: List[Dict[str, Any]] = Field(..., description="Generated insights")
    total_patterns_analyzed: int = Field(..., description="Number of patterns analyzed")
    insights_generated: int = Field(..., description="Number of insights generated")
    confidence_threshold: float = Field(..., description="Applied confidence threshold")
    seasonal_analysis_included: bool = Field(..., description="Whether seasonal analysis was included")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[Dict[str, Any]] = Field(..., description="Generated recommendations")
    total_products: int = Field(..., description="Number of products analyzed")
    compliance_validated: bool = Field(..., description="Whether compliance was validated")
    priority_distribution: Dict[str, int] = Field(..., description="Distribution of recommendations by priority")


class ScenarioResponse(BaseModel):
    """Response model for scenario analysis."""
    scenarios: List[Dict[str, Any]] = Field(..., description="Generated scenarios")
    total_scenarios: int = Field(..., description="Number of scenarios generated")
    base_parameters: Dict[str, Any] = Field(..., description="Base parameters used")
    seasonal_modeling_included: bool = Field(..., description="Whether seasonal modeling was included")


class RiskResponse(BaseModel):
    """Response model for risk assessment."""
    product_id: str = Field(..., description="Product ID assessed")
    current_inventory: int = Field(..., description="Current inventory level")
    assessment_type: str = Field(..., description="Type of assessment performed")
    risk_assessments: Dict[str, Any] = Field(..., description="Risk assessment results")
    demand_volatility: float = Field(..., description="Demand volatility score")
    seasonal_adjustments_applied: bool = Field(..., description="Whether seasonal adjustments were applied")


class ComplianceResponse(BaseModel):
    """Response model for compliance validation."""
    compliance_result: Dict[str, Any] = Field(..., description="Compliance validation result")
    regulatory_constraints: Optional[Dict[str, Any]] = Field(None, description="Applicable regulatory constraints")
    constraint_explanations: Optional[Dict[str, Any]] = Field(None, description="Human-readable constraint explanations")
    recommendation: Dict[str, Any] = Field(..., description="Original recommendation validated")


# Query parameter models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class FilterParams(BaseModel):
    """Common filter parameters."""
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    product_categories: Optional[List[str]] = Field(None, description="Product category filters")
    priority_levels: Optional[List[PriorityLevel]] = Field(None, description="Priority level filters")
    confidence_min: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    confidence_max: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum confidence threshold")
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        if v and info.data.get('start_date') and v < info.data['start_date']:
            raise ValueError("End date must be after start date")
        return v


class SortParams(BaseModel):
    """Sorting parameters."""
    sort_by: str = Field("timestamp", description="Field to sort by")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order (asc or desc)")