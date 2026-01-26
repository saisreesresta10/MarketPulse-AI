"""
Insights API Router

Endpoints for insight generation and retrieval.
"""

from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from ...components import InsightGenerator, DataProcessor
from ...core.models import DemandPattern
from ..models import InsightRequest, APIResponse, PaginationParams, FilterParams
from ..utils import (
    create_success_response, validate_product_ids, validate_confidence_threshold,
    timing_middleware, format_component_error
)

router = APIRouter(prefix="/insights", tags=["insights"])


async def get_insight_generator() -> InsightGenerator:
    """Dependency to get InsightGenerator component."""
    from ..main import components, initialize_components_for_testing
    
    if "insight_generator" not in components:
        initialize_components_for_testing()
    
    if "insight_generator" not in components:
        raise HTTPException(status_code=500, detail="InsightGenerator component not available")
    
    return components["insight_generator"]


async def get_data_processor() -> DataProcessor:
    """Dependency to get DataProcessor component."""
    from ..main import components, initialize_components_for_testing
    
    if "data_processor" not in components:
        initialize_components_for_testing()
    
    if "data_processor" not in components:
        raise HTTPException(status_code=500, detail="DataProcessor component not available")
    
    return components["data_processor"]


@router.post("/generate", response_model=APIResponse)
@timing_middleware
async def generate_insights(
    request: InsightRequest,
    insight_generator: InsightGenerator = Depends(get_insight_generator),
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """
    Generate insights from demand patterns.
    
    - **product_ids**: Optional list of product IDs to analyze
    - **include_seasonal**: Include seasonal analysis in insights
    - **confidence_threshold**: Minimum confidence threshold for insights
    - **max_insights**: Maximum number of insights to return
    """
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        if request.product_ids:
            validate_product_ids(request.product_ids)
        validate_confidence_threshold(request.confidence_threshold)
        
        # Extract demand patterns
        patterns = await data_processor.extract_demand_patterns(request.product_ids)
        
        if not patterns:
            return create_success_response(
                data={
                    "insights": [],
                    "total_patterns_analyzed": 0,
                    "insights_generated": 0,
                    "confidence_threshold": request.confidence_threshold,
                    "seasonal_analysis_included": request.include_seasonal
                },
                message="No patterns found for the specified criteria",
                request_id=request_id
            )
        
        # Add seasonal correlation if requested
        if request.include_seasonal:
            patterns = await data_processor.correlate_seasonal_events(patterns)
        
        # Generate insights
        insights = await insight_generator.generate_insights(patterns)
        
        # Filter by confidence threshold
        filtered_insights = [
            insight for insight in insights 
            if insight.confidence >= request.confidence_threshold
        ]
        
        # Apply max insights limit if specified
        if request.max_insights:
            filtered_insights = filtered_insights[:request.max_insights]
        
        return create_success_response(
            data={
                "insights": [insight.dict() for insight in filtered_insights],
                "total_patterns_analyzed": len(patterns),
                "insights_generated": len(filtered_insights),
                "confidence_threshold": request.confidence_threshold,
                "seasonal_analysis_included": request.include_seasonal
            },
            message=f"Generated {len(filtered_insights)} insights from {len(patterns)} patterns",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("InsightGenerator", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/{product_id}", response_model=APIResponse)
@timing_middleware
async def get_product_insights(
    product_id: str = Path(..., description="Product ID to get insights for"),
    include_seasonal: bool = Query(True, description="Include seasonal analysis"),
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    insight_generator: InsightGenerator = Depends(get_insight_generator),
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """Get insights for a specific product."""
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        validate_product_ids([product_id])
        validate_confidence_threshold(confidence_threshold)
        
        # Extract patterns for specific product
        patterns = await data_processor.extract_demand_patterns([product_id])
        
        if not patterns:
            raise HTTPException(
                status_code=404, 
                detail=f"No patterns found for product {product_id}"
            )
        
        # Add seasonal correlation if requested
        if include_seasonal:
            patterns = await data_processor.correlate_seasonal_events(patterns)
        
        # Generate insights for the product
        insights = []
        for pattern in patterns:
            insight = await insight_generator.explain_pattern(pattern)
            if insight.confidence >= confidence_threshold:
                insights.append(insight)
        
        return create_success_response(
            data={
                "product_id": product_id,
                "insights": [insight.dict() for insight in insights],
                "patterns_analyzed": len(patterns),
                "confidence_threshold": confidence_threshold,
                "seasonal_analysis_included": include_seasonal
            },
            message=f"Retrieved {len(insights)} insights for product {product_id}",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("InsightGenerator", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/{product_id}/key-factors", response_model=APIResponse)
@timing_middleware
async def get_key_factors(
    product_id: str = Path(..., description="Product ID to analyze key factors for"),
    insight_generator: InsightGenerator = Depends(get_insight_generator),
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """Get key factors influencing demand patterns for a specific product."""
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        validate_product_ids([product_id])
        
        # Extract patterns for specific product
        patterns = await data_processor.extract_demand_patterns([product_id])
        
        if not patterns:
            raise HTTPException(
                status_code=404, 
                detail=f"No patterns found for product {product_id}"
            )
        
        # Get key factors for each pattern
        all_key_factors = []
        for pattern in patterns:
            key_factors = await insight_generator.identify_key_factors(pattern)
            all_key_factors.extend(key_factors)
        
        # Remove duplicates and sort by frequency
        factor_counts = {}
        for factor in all_key_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        sorted_factors = sorted(
            factor_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return create_success_response(
            data={
                "product_id": product_id,
                "key_factors": [
                    {"factor": factor, "frequency": count} 
                    for factor, count in sorted_factors
                ],
                "patterns_analyzed": len(patterns),
                "total_factors_identified": len(all_key_factors),
                "unique_factors": len(sorted_factors)
            },
            message=f"Identified {len(sorted_factors)} key factors for product {product_id}",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("InsightGenerator", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/batch-generate", response_model=APIResponse)
@timing_middleware
async def batch_generate_insights(
    product_ids: List[str],
    include_seasonal: bool = Query(True, description="Include seasonal analysis"),
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    max_insights_per_product: Optional[int] = Query(None, ge=1, le=20, description="Max insights per product"),
    insight_generator: InsightGenerator = Depends(get_insight_generator),
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """Generate insights for multiple products in batch."""
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        validate_product_ids(product_ids)
        validate_confidence_threshold(confidence_threshold)
        
        batch_results = {}
        total_insights = 0
        total_patterns = 0
        
        for product_id in product_ids:
            try:
                # Extract patterns for this product
                patterns = await data_processor.extract_demand_patterns([product_id])
                
                if not patterns:
                    batch_results[product_id] = {
                        "insights": [],
                        "patterns_analyzed": 0,
                        "error": None
                    }
                    continue
                
                # Add seasonal correlation if requested
                if include_seasonal:
                    patterns = await data_processor.correlate_seasonal_events(patterns)
                
                # Generate insights
                insights = []
                for pattern in patterns:
                    insight = await insight_generator.explain_pattern(pattern)
                    if insight.confidence >= confidence_threshold:
                        insights.append(insight)
                
                # Apply per-product limit if specified
                if max_insights_per_product:
                    insights = insights[:max_insights_per_product]
                
                batch_results[product_id] = {
                    "insights": [insight.dict() for insight in insights],
                    "patterns_analyzed": len(patterns),
                    "error": None
                }
                
                total_insights += len(insights)
                total_patterns += len(patterns)
                
            except Exception as e:
                batch_results[product_id] = {
                    "insights": [],
                    "patterns_analyzed": 0,
                    "error": str(e)
                }
        
        successful_products = sum(1 for result in batch_results.values() if result["error"] is None)
        
        return create_success_response(
            data={
                "batch_results": batch_results,
                "summary": {
                    "total_products": len(product_ids),
                    "successful_products": successful_products,
                    "failed_products": len(product_ids) - successful_products,
                    "total_insights_generated": total_insights,
                    "total_patterns_analyzed": total_patterns,
                    "confidence_threshold": confidence_threshold,
                    "seasonal_analysis_included": include_seasonal
                }
            },
            message=f"Batch processed {len(product_ids)} products, generated {total_insights} insights",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("InsightGenerator", e)
        raise HTTPException(status_code=500, detail=error_msg)