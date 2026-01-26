"""
Data API Router

Endpoints for data ingestion and management.
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from ...components import DataProcessor
from ...core.models import SalesDataPoint
from ..models import SalesDataRequest, APIResponse
from ..utils import (
    create_success_response, validate_product_ids, timing_middleware, 
    format_component_error
)

router = APIRouter(prefix="/data", tags=["data"])


async def get_data_processor() -> DataProcessor:
    """Dependency to get DataProcessor component."""
    from ..main import components, initialize_components_for_testing
    
    if "data_processor" not in components:
        initialize_components_for_testing()
    
    if "data_processor" not in components:
        raise HTTPException(status_code=500, detail="DataProcessor component not available")
    
    return components["data_processor"]


@router.post("/ingest", response_model=APIResponse)
@timing_middleware
async def ingest_sales_data(
    request: SalesDataRequest,
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """
    Ingest sales data for processing and analysis.
    
    - **data**: List of sales data points to ingest
    - **validate_data**: Whether to validate data quality
    - **store_patterns**: Whether to store extracted patterns
    """
    request_id = str(uuid4())
    
    try:
        # Convert request data to SalesDataPoint objects
        sales_data = []
        validation_errors = []
        
        for i, item in enumerate(request.data):
            try:
                sales_data.append(SalesDataPoint(**item))
            except Exception as e:
                validation_errors.append({
                    "index": i,
                    "data": item,
                    "error": str(e)
                })
        
        if validation_errors and request.validate_data:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Data validation failed",
                    "validation_errors": validation_errors
                }
            )
        
        # Ingest data
        result = await data_processor.ingest_sales_data(sales_data)
        
        # Store patterns if requested
        if request.store_patterns:
            patterns = await data_processor.extract_demand_patterns()
            if patterns:
                await data_processor.store_patterns(patterns)
                result["patterns_stored"] = len(patterns)
        
        return create_success_response(
            data={
                **result,
                "ingestion_summary": {
                    "total_records": len(request.data),
                    "valid_records": len(sales_data),
                    "validation_errors": len(validation_errors),
                    "patterns_extracted": request.store_patterns
                }
            },
            message=f"Successfully ingested {len(sales_data)} sales data points",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DataProcessor", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/patterns", response_model=APIResponse)
@timing_middleware
async def get_demand_patterns(
    product_ids: Optional[List[str]] = Query(None, description="Filter by product IDs"),
    include_seasonal: bool = Query(True, description="Include seasonal correlation"),
    limit: int = Query(50, ge=1, le=200, description="Maximum patterns to return"),
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """
    Retrieve demand patterns from processed data.
    
    - **product_ids**: Optional filter by product IDs
    - **include_seasonal**: Include seasonal correlation data
    - **limit**: Maximum number of patterns to return
    """
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        if product_ids:
            validate_product_ids(product_ids)
        
        # Extract demand patterns
        patterns = await data_processor.extract_demand_patterns(product_ids)
        
        if not patterns:
            return create_success_response(
                data={
                    "patterns": [],
                    "total_patterns": 0,
                    "seasonal_correlation_included": include_seasonal
                },
                message="No patterns found for the specified criteria",
                request_id=request_id
            )
        
        # Add seasonal correlation if requested
        if include_seasonal:
            patterns = await data_processor.correlate_seasonal_events(patterns)
        
        # Apply limit
        limited_patterns = patterns[:limit]
        
        return create_success_response(
            data={
                "patterns": [pattern.dict() for pattern in limited_patterns],
                "total_patterns": len(patterns),
                "patterns_returned": len(limited_patterns),
                "seasonal_correlation_included": include_seasonal,
                "filters_applied": {
                    "product_ids": product_ids,
                    "limit": limit
                }
            },
            message=f"Retrieved {len(limited_patterns)} demand patterns",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DataProcessor", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/seasonal-analysis", response_model=APIResponse)
@timing_middleware
async def get_seasonal_analysis(
    product_ids: Optional[List[str]] = Query(None, description="Filter by product IDs"),
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """
    Generate comprehensive seasonal analysis report.
    
    - **product_ids**: Optional filter by product IDs
    """
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        if product_ids:
            validate_product_ids(product_ids)
        
        # Generate seasonal analysis report
        report = await data_processor.generate_seasonal_analysis_report(product_ids)
        
        return create_success_response(
            data=report,
            message=f"Generated seasonal analysis report for {len(product_ids) if product_ids else 'all'} products",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DataProcessor", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/integrate-market-signals", response_model=APIResponse)
@timing_middleware
async def integrate_market_signals(
    external_data: Dict[str, Any],
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """
    Integrate external market signals into analysis.
    
    - **external_data**: Dictionary containing external market data
    """
    request_id = str(uuid4())
    
    try:
        if not external_data:
            raise HTTPException(status_code=400, detail="External data cannot be empty")
        
        # Integrate market signals
        result = await data_processor.integrate_market_signals(external_data)
        
        return create_success_response(
            data=result,
            message="Successfully integrated external market signals",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DataProcessor", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/quality-report", response_model=APIResponse)
@timing_middleware
async def get_data_quality_report(
    product_ids: Optional[List[str]] = Query(None, description="Filter by product IDs"),
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """
    Generate data quality assessment report.
    
    - **product_ids**: Optional filter by product IDs
    """
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        if product_ids:
            validate_product_ids(product_ids)
        
        # This would be implemented in the DataProcessor
        # For now, return a mock quality report
        quality_report = {
            "data_completeness": 0.95,
            "data_accuracy": 0.92,
            "data_consistency": 0.88,
            "missing_values": {
                "quantity_sold": 0.02,
                "revenue": 0.01,
                "inventory_level": 0.05
            },
            "outliers_detected": 12,
            "duplicate_records": 3,
            "quality_score": 0.91,
            "recommendations": [
                "Address missing inventory level data",
                "Review outlier records for accuracy",
                "Remove duplicate entries"
            ]
        }
        
        return create_success_response(
            data=quality_report,
            message="Generated data quality assessment report",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DataProcessor", e)
        raise HTTPException(status_code=500, detail=error_msg)