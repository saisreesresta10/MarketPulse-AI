"""
Recommendations API Router

Endpoints for generating and managing business recommendations.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from ...components import DecisionSupportEngine
from ..models import RecommendationRequest, APIResponse, PriorityLevel
from ..utils import (
    create_success_response, validate_product_ids, validate_business_context,
    timing_middleware, format_component_error
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


async def get_decision_support_engine() -> DecisionSupportEngine:
    """Dependency to get DecisionSupportEngine component."""
    from ..main import components, initialize_components_for_testing
    
    if "decision_support_engine" not in components:
        initialize_components_for_testing()
    
    if "decision_support_engine" not in components:
        raise HTTPException(status_code=500, detail="DecisionSupportEngine component not available")
    
    return components["decision_support_engine"]


@router.post("/generate", response_model=APIResponse)
@timing_middleware
async def generate_recommendations(
    request: RecommendationRequest,
    decision_engine: DecisionSupportEngine = Depends(get_decision_support_engine)
):
    """
    Generate comprehensive business recommendations.
    
    - **product_ids**: List of product IDs for recommendations
    - **business_context**: Additional business context
    - **priority_filter**: Filter by priority level (high, medium, low)
    - **include_compliance_check**: Include compliance validation
    - **max_recommendations**: Maximum recommendations to return
    """
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        validate_product_ids(request.product_ids)
        if request.business_context:
            validate_business_context(request.business_context)
        
        # Prepare request for decision engine
        engine_request = {
            "product_ids": request.product_ids,
            "business_context": request.business_context or {},
            "priority_filter": request.priority_filter.value if request.priority_filter else None,
            "include_compliance_check": request.include_compliance_check,
            "max_recommendations": request.max_recommendations
        }
        
        # Generate recommendations
        recommendations_result = await decision_engine.generate_recommendations(engine_request)
        
        # Extract recommendations and apply filters
        recommendations = recommendations_result.get("recommendations", [])
        
        # Apply priority filter if specified
        if request.priority_filter:
            recommendations = [
                rec for rec in recommendations 
                if rec.get("priority", "").lower() == request.priority_filter.value
            ]
        
        # Apply max recommendations limit
        if request.max_recommendations:
            recommendations = recommendations[:request.max_recommendations]
        
        # Calculate priority distribution
        priority_distribution = {"high": 0, "medium": 0, "low": 0}
        for rec in recommendations:
            priority = rec.get("priority", "medium").lower()
            if priority in priority_distribution:
                priority_distribution[priority] += 1
        
        return create_success_response(
            data={
                "recommendations": recommendations,
                "total_products": len(request.product_ids),
                "recommendations_generated": len(recommendations),
                "compliance_validated": request.include_compliance_check,
                "priority_distribution": priority_distribution,
                "business_context_applied": bool(request.business_context),
                "filters_applied": {
                    "priority_filter": request.priority_filter.value if request.priority_filter else None,
                    "max_recommendations": request.max_recommendations
                }
            },
            message=f"Generated {len(recommendations)} recommendations for {len(request.product_ids)} products",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DecisionSupportEngine", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/optimize-discount", response_model=APIResponse)
@timing_middleware
async def optimize_discount_strategy(
    product_ids: List[str],
    business_context: Optional[Dict[str, Any]] = None,
    decision_engine: DecisionSupportEngine = Depends(get_decision_support_engine)
):
    """
    Optimize discount strategy for specified products.
    
    - **product_ids**: List of product IDs to optimize discounts for
    - **business_context**: Optional business context for optimization
    """
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        validate_product_ids(product_ids)
        if business_context:
            validate_business_context(business_context)
        
        # Optimize discount strategy
        strategy = await decision_engine.optimize_discount_strategy(product_ids)
        
        # Enhance with business context if provided
        if business_context:
            strategy["business_context_applied"] = business_context
            strategy["optimization_constraints"] = {
                "target_margin": business_context.get("target_margin"),
                "inventory_turnover_target": business_context.get("inventory_turnover_target"),
                "seasonal_events": business_context.get("seasonal_events", [])
            }
        
        return create_success_response(
            data=strategy,
            message=f"Optimized discount strategy for {len(product_ids)} products",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DecisionSupportEngine", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/{product_id}/impact", response_model=APIResponse)
@timing_middleware
async def assess_recommendation_impact(
    product_id: str = Path(..., description="Product ID to assess impact for"),
    recommendation_data: Optional[Dict[str, Any]] = None,
    decision_engine: DecisionSupportEngine = Depends(get_decision_support_engine)
):
    """Assess business impact of recommendations for a specific product."""
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        validate_product_ids([product_id])
        
        # If no recommendation data provided, generate default recommendation
        if not recommendation_data:
            rec_request = {"product_ids": [product_id]}
            recommendations = await decision_engine.generate_recommendations(rec_request)
            if not recommendations.get("recommendations"):
                raise HTTPException(
                    status_code=404, 
                    detail=f"No recommendations found for product {product_id}"
                )
            recommendation_data = recommendations["recommendations"][0]
        
        # Assess business impact
        impact = await decision_engine.assess_business_impact(recommendation_data)
        
        return create_success_response(
            data={
                "product_id": product_id,
                "recommendation": recommendation_data,
                "impact_assessment": impact
            },
            message=f"Assessed business impact for product {product_id}",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DecisionSupportEngine", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/prioritize", response_model=APIResponse)
@timing_middleware
async def prioritize_recommendations(
    recommendations: List[Dict[str, Any]],
    decision_engine: DecisionSupportEngine = Depends(get_decision_support_engine)
):
    """
    Prioritize a list of recommendations by impact and urgency.
    
    - **recommendations**: List of recommendations to prioritize
    """
    request_id = str(uuid4())
    
    try:
        if not recommendations:
            raise HTTPException(status_code=400, detail="Recommendations list cannot be empty")
        
        if len(recommendations) > 100:
            raise HTTPException(status_code=400, detail="Too many recommendations (maximum 100 allowed)")
        
        # Prioritize recommendations
        prioritized_recommendations = await decision_engine.prioritize_recommendations(recommendations)
        
        # Calculate priority statistics
        priority_stats = {"high": 0, "medium": 0, "low": 0}
        impact_scores = []
        
        for rec in prioritized_recommendations:
            priority = rec.get("priority", "medium").lower()
            if priority in priority_stats:
                priority_stats[priority] += 1
            
            if "impact_score" in rec:
                impact_scores.append(rec["impact_score"])
        
        avg_impact_score = sum(impact_scores) / len(impact_scores) if impact_scores else 0
        
        return create_success_response(
            data={
                "prioritized_recommendations": prioritized_recommendations,
                "original_count": len(recommendations),
                "prioritized_count": len(prioritized_recommendations),
                "priority_distribution": priority_stats,
                "average_impact_score": round(avg_impact_score, 3),
                "ranking_criteria": ["impact_score", "urgency", "compliance_status", "business_value"]
            },
            message=f"Prioritized {len(prioritized_recommendations)} recommendations",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DecisionSupportEngine", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/validate-pipeline", response_model=APIResponse)
@timing_middleware
async def validate_recommendation_pipeline(
    recommendation: Dict[str, Any],
    decision_engine: DecisionSupportEngine = Depends(get_decision_support_engine)
):
    """
    Validate recommendation through complete compliance pipeline.
    
    - **recommendation**: Recommendation to validate
    """
    request_id = str(uuid4())
    
    try:
        if not recommendation:
            raise HTTPException(status_code=400, detail="Recommendation cannot be empty")
        
        # Validate through complete pipeline
        compliance_result = await decision_engine.validate_recommendation_pipeline(recommendation)
        
        return create_success_response(
            data={
                "recommendation": recommendation,
                "compliance_result": compliance_result.dict(),
                "validation_status": "PASSED" if compliance_result.is_compliant else "FAILED",
                "pipeline_stages": [
                    "mrp_compliance_check",
                    "discount_limits_validation", 
                    "pricing_strategy_validation",
                    "regulatory_constraints_check"
                ]
            },
            message=f"Pipeline validation {'passed' if compliance_result.is_compliant else 'failed'}",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DecisionSupportEngine", e)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/search", response_model=APIResponse)
@timing_middleware
async def search_recommendations(
    query: str = Query(..., description="Search query"),
    product_ids: Optional[List[str]] = Query(None, description="Filter by product IDs"),
    priority: Optional[PriorityLevel] = Query(None, description="Filter by priority level"),
    min_impact_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum impact score"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    decision_engine: DecisionSupportEngine = Depends(get_decision_support_engine)
):
    """
    Search and filter recommendations based on criteria.
    
    - **query**: Search query for recommendation content
    - **product_ids**: Optional filter by product IDs
    - **priority**: Optional filter by priority level
    - **min_impact_score**: Optional minimum impact score filter
    - **limit**: Maximum number of results to return
    """
    request_id = str(uuid4())
    
    try:
        # Validate inputs
        if product_ids:
            validate_product_ids(product_ids)
        
        # For this demo, we'll generate recommendations and then filter them
        # In a real implementation, this would search a recommendations database
        
        search_request = {
            "product_ids": product_ids or [],
            "business_context": {"search_query": query}
        }
        
        # Generate recommendations to search through
        recommendations_result = await decision_engine.generate_recommendations(search_request)
        recommendations = recommendations_result.get("recommendations", [])
        
        # Apply filters
        filtered_recommendations = []
        for rec in recommendations:
            # Priority filter
            if priority and rec.get("priority", "").lower() != priority.value:
                continue
            
            # Impact score filter
            if min_impact_score and rec.get("impact_score", 0) < min_impact_score:
                continue
            
            # Text search in recommendation content
            rec_text = str(rec).lower()
            if query.lower() in rec_text:
                filtered_recommendations.append(rec)
        
        # Apply limit
        filtered_recommendations = filtered_recommendations[:limit]
        
        return create_success_response(
            data={
                "recommendations": filtered_recommendations,
                "search_query": query,
                "total_found": len(filtered_recommendations),
                "filters_applied": {
                    "product_ids": product_ids,
                    "priority": priority.value if priority else None,
                    "min_impact_score": min_impact_score
                },
                "search_metadata": {
                    "total_searched": len(recommendations),
                    "results_returned": len(filtered_recommendations),
                    "limit_applied": limit
                }
            },
            message=f"Found {len(filtered_recommendations)} recommendations matching search criteria",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = format_component_error("DecisionSupportEngine", e)
        raise HTTPException(status_code=500, detail=error_msg)