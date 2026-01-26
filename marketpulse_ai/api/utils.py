"""
API Utility Functions

Helper functions for request validation, response formatting, and error handling.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4
from functools import wraps

from fastapi import HTTPException, Request
from pydantic import ValidationError
import logging

from .models import APIResponse, ErrorResponse, PaginatedResponse, PaginationParams

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid4())


def create_success_response(
    data: Any = None,
    message: str = None,
    request_id: str = None,
    execution_time_ms: float = None
) -> APIResponse:
    """Create a standardized success response."""
    return APIResponse(
        success=True,
        data=data,
        message=message,
        request_id=request_id or generate_request_id(),
        execution_time_ms=execution_time_ms
    )


def create_error_response(
    error: str,
    message: str,
    details: Dict[str, Any] = None,
    request_id: str = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error,
        message=message,
        details=details,
        request_id=request_id or generate_request_id()
    )


def create_paginated_response(
    data: List[Any],
    pagination_params: PaginationParams,
    total_count: int,
    message: str = None,
    request_id: str = None
) -> PaginatedResponse:
    """Create a paginated response."""
    total_pages = (total_count + pagination_params.page_size - 1) // pagination_params.page_size
    
    pagination_info = {
        "current_page": pagination_params.page,
        "page_size": pagination_params.page_size,
        "total_items": total_count,
        "total_pages": total_pages,
        "has_next": pagination_params.page < total_pages,
        "has_previous": pagination_params.page > 1
    }
    
    return PaginatedResponse(
        data=data,
        pagination=pagination_info,
        message=message,
        request_id=request_id or generate_request_id()
    )


def validate_product_ids(product_ids: List[str]) -> None:
    """Validate product ID format and constraints."""
    if not product_ids:
        raise HTTPException(status_code=400, detail="Product IDs list cannot be empty")
    
    if len(product_ids) > 100:
        raise HTTPException(status_code=400, detail="Too many product IDs (maximum 100 allowed)")
    
    for product_id in product_ids:
        if not product_id or not isinstance(product_id, str):
            raise HTTPException(status_code=400, detail=f"Invalid product ID: {product_id}")
        
        if len(product_id) > 50:
            raise HTTPException(status_code=400, detail=f"Product ID too long: {product_id}")


def validate_date_range(start_date: Optional[datetime], end_date: Optional[datetime]) -> None:
    """Validate date range parameters."""
    if start_date and end_date:
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Check if date range is reasonable (not more than 2 years)
        if (end_date - start_date).days > 730:
            raise HTTPException(status_code=400, detail="Date range too large (maximum 2 years)")


def validate_confidence_threshold(confidence: float) -> None:
    """Validate confidence threshold value."""
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")


def validate_pagination_params(page: int, page_size: int) -> None:
    """Validate pagination parameters."""
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")
    
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")


def handle_validation_error(error: ValidationError) -> HTTPException:
    """Convert Pydantic validation error to HTTP exception."""
    error_details = []
    for err in error.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in err["loc"]),
            "message": err["msg"],
            "type": err["type"]
        })
    
    return HTTPException(
        status_code=422,
        detail={
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": error_details
        }
    )


def sanitize_input(data: Any) -> Any:
    """Sanitize input data to prevent injection attacks."""
    if isinstance(data, str):
        # Basic sanitization - remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        for char in dangerous_chars:
            data = data.replace(char, '')
        return data.strip()
    
    elif isinstance(data, dict):
        return {key: sanitize_input(value) for key, value in data.items()}
    
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    
    return data


def log_api_request(request: Request, request_id: str) -> None:
    """Log API request details."""
    logger.info(
        f"API Request - ID: {request_id}, Method: {request.method}, "
        f"URL: {request.url}, Client: {request.client.host if request.client else 'unknown'}"
    )


def log_api_response(request_id: str, status_code: int, execution_time_ms: float) -> None:
    """Log API response details."""
    logger.info(
        f"API Response - ID: {request_id}, Status: {status_code}, "
        f"Execution Time: {execution_time_ms:.2f}ms"
    )


def timing_middleware(func: Callable) -> Callable:
    """Decorator to measure execution time of API endpoints."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Add execution time to response if it's an APIResponse
            if hasattr(result, 'execution_time_ms'):
                result.execution_time_ms = execution_time_ms
            
            return result
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"API endpoint failed after {execution_time_ms:.2f}ms: {str(e)}")
            raise
    
    return wrapper


def format_error_message(error: Exception) -> str:
    """Format error message for API responses."""
    error_type = type(error).__name__
    error_message = str(error)
    
    # Don't expose internal error details in production
    if "Internal" in error_type or "Database" in error_type:
        return "An internal error occurred. Please try again later."
    
    return error_message


def extract_filters_from_request(request_params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate filter parameters from request."""
    filters = {}
    
    # Date filters
    if 'start_date' in request_params and request_params['start_date']:
        filters['start_date'] = request_params['start_date']
    
    if 'end_date' in request_params and request_params['end_date']:
        filters['end_date'] = request_params['end_date']
    
    # Product filters
    if 'product_ids' in request_params and request_params['product_ids']:
        filters['product_ids'] = request_params['product_ids']
    
    if 'product_categories' in request_params and request_params['product_categories']:
        filters['product_categories'] = request_params['product_categories']
    
    # Confidence filters
    if 'confidence_min' in request_params and request_params['confidence_min'] is not None:
        filters['confidence_min'] = request_params['confidence_min']
    
    if 'confidence_max' in request_params and request_params['confidence_max'] is not None:
        filters['confidence_max'] = request_params['confidence_max']
    
    # Priority filters
    if 'priority_levels' in request_params and request_params['priority_levels']:
        filters['priority_levels'] = request_params['priority_levels']
    
    return filters


def apply_pagination(data: List[Any], pagination_params: PaginationParams) -> List[Any]:
    """Apply pagination to a list of data."""
    start_index = pagination_params.offset
    end_index = start_index + pagination_params.page_size
    return data[start_index:end_index]


def validate_business_context(context: Dict[str, Any]) -> None:
    """Validate business context parameters."""
    if not isinstance(context, dict):
        raise HTTPException(status_code=400, detail="Business context must be a dictionary")
    
    # Validate specific business context fields
    if 'target_margin' in context:
        margin = context['target_margin']
        if not isinstance(margin, (int, float)) or margin < 0 or margin > 1:
            raise HTTPException(status_code=400, detail="Target margin must be between 0 and 1")
    
    if 'inventory_turnover_target' in context:
        turnover = context['inventory_turnover_target']
        if not isinstance(turnover, (int, float)) or turnover <= 0:
            raise HTTPException(status_code=400, detail="Inventory turnover target must be positive")
    
    if 'seasonal_events' in context:
        events = context['seasonal_events']
        if not isinstance(events, list):
            raise HTTPException(status_code=400, detail="Seasonal events must be a list")


def validate_scenario_parameters(parameters: Dict[str, Any]) -> None:
    """Validate scenario parameters."""
    required_fields = ['product_id', 'current_inventory', 'demand_forecast']
    
    for field in required_fields:
        if field not in parameters:
            raise HTTPException(status_code=400, detail=f"Missing required parameter: {field}")
    
    # Validate numeric fields
    if parameters.get('current_inventory', 0) < 0:
        raise HTTPException(status_code=400, detail="Current inventory cannot be negative")
    
    if parameters.get('demand_forecast', 0) < 0:
        raise HTTPException(status_code=400, detail="Demand forecast cannot be negative")
    
    # Validate discount range if provided
    if 'discount_range' in parameters:
        discount_range = parameters['discount_range']
        if not isinstance(discount_range, list) or len(discount_range) != 2:
            raise HTTPException(status_code=400, detail="Discount range must be a list of two values")
        
        min_discount, max_discount = discount_range
        if min_discount < 0 or max_discount > 1 or min_discount >= max_discount:
            raise HTTPException(status_code=400, detail="Invalid discount range values")


def check_rate_limit(request: Request, max_requests_per_minute: int = 60) -> None:
    """Basic rate limiting check (in production, use Redis or similar)."""
    # This is a simplified rate limiting implementation
    # In production, you would use Redis or a proper rate limiting service
    client_ip = request.client.host if request.client else "unknown"
    
    # For now, just log the request - implement proper rate limiting as needed
    logger.debug(f"Rate limit check for client {client_ip}")


def format_component_error(component_name: str, error: Exception) -> str:
    """Format component-specific error messages."""
    error_message = str(error)
    
    component_context = {
        "DataProcessor": "data processing",
        "RiskAssessor": "risk assessment",
        "ComplianceValidator": "compliance validation",
        "InsightGenerator": "insight generation",
        "DecisionSupportEngine": "recommendation generation",
        "ScenarioAnalyzer": "scenario analysis"
    }
    
    context = component_context.get(component_name, "system operation")
    return f"Error in {context}: {error_message}"