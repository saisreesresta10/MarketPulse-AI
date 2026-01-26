"""
Workflow API Endpoints

Provides end-to-end workflow endpoints using the MarketPulse orchestrator
for complete retailer insights, recommendations, and scenario analysis.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from ...core.models import SalesDataPoint
from ..models import APIResponse

router = APIRouter(prefix="/workflows", tags=["workflows"])


class RetailerInsightsRequest(BaseModel):
    """Request model for retailer insights workflow."""
    retailer_id: str = Field(..., description="ID of the retailer")
    sales_data: List[Dict[str, Any]] = Field(..., description="Historical sales data")
    include_risk_assessment: bool = Field(True, description="Include risk assessment")
    include_compliance_check: bool = Field(True, description="Include compliance validation")


class RecommendationsRequest(BaseModel):
    """Request model for recommendations workflow."""
    retailer_id: str = Field(..., description="ID of the retailer")
    product_ids: List[str] = Field(..., description="List of product IDs")
    business_context: Optional[Dict[str, Any]] = Field(None, description="Additional business context")


class ScenarioAnalysisRequest(BaseModel):
    """Request model for scenario analysis workflow."""
    retailer_id: str = Field(..., description="ID of the retailer")
    base_scenario: Dict[str, Any] = Field(..., description="Base scenario parameters")
    scenario_variations: List[Dict[str, Any]] = Field(..., description="Scenario variations to analyze")


class FeedbackRequest(BaseModel):
    """Request model for feedback processing."""
    retailer_id: str = Field(..., description="ID of the retailer")
    feedback_data: Dict[str, Any] = Field(..., description="Feedback data")


def get_orchestrator():
    """Dependency to get the orchestrator component."""
    from ..main import component_manager
    orchestrator = component_manager.get_component("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    return orchestrator


@router.post("/retailer-insights", response_model=APIResponse)
async def generate_retailer_insights(
    request: RetailerInsightsRequest,
    orchestrator = Depends(get_orchestrator)
):
    """
    Generate comprehensive retailer insights using end-to-end workflow.
    
    This endpoint processes sales data through the complete pipeline:
    1. Data validation and processing
    2. Demand pattern extraction
    3. Insight generation
    4. Risk assessment (optional)
    5. Compliance validation (optional)
    """
    request_id = str(uuid4())
    
    try:
        # Convert sales data to SalesDataPoint objects
        sales_data_points = []
        for data_dict in request.sales_data:
            try:
                # Create SalesDataPoint from dictionary with correct field mapping
                sales_point = SalesDataPoint(
                    product_id=data_dict["product_id"],
                    product_name=data_dict["product_name"],
                    category=data_dict["category"],
                    mrp=Decimal(str(data_dict["mrp"])),
                    selling_price=Decimal(str(data_dict["selling_price"])),
                    quantity_sold=data_dict["quantity_sold"],
                    sale_date=datetime.fromisoformat(data_dict["date"].replace("Z", "+00:00")).date(),
                    store_location=data_dict["store_location"],
                    seasonal_event=data_dict.get("seasonal_event")
                )
                sales_data_points.append(sales_point)
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid sales data format: {str(e)}"
                )
        
        # Execute workflow
        workflow_result = await orchestrator.generate_retailer_insights(
            retailer_id=request.retailer_id,
            sales_data=sales_data_points,
            include_risk_assessment=request.include_risk_assessment,
            include_compliance_check=request.include_compliance_check
        )
        
        if not workflow_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow failed: {workflow_result.error_message}"
            )
        
        response = APIResponse(
            success=True,
            data=workflow_result.data,
            message="Retailer insights generated successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.post("/recommendations", response_model=APIResponse)
async def generate_recommendations(
    request: RecommendationsRequest,
    orchestrator = Depends(get_orchestrator)
):
    """
    Generate comprehensive recommendations using end-to-end workflow.
    
    This endpoint generates recommendations through the complete pipeline:
    1. Product data retrieval
    2. Recommendation generation
    3. Compliance validation
    4. Prioritization and ranking
    """
    request_id = str(uuid4())
    
    try:
        # Execute workflow
        workflow_result = await orchestrator.generate_recommendations(
            retailer_id=request.retailer_id,
            product_ids=request.product_ids,
            business_context=request.business_context
        )
        
        if not workflow_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow failed: {workflow_result.error_message}"
            )
        
        response = APIResponse(
            success=True,
            data=workflow_result.data,
            message="Recommendations generated successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.post("/scenario-analysis", response_model=APIResponse)
async def analyze_scenarios(
    request: ScenarioAnalysisRequest,
    orchestrator = Depends(get_orchestrator)
):
    """
    Perform comprehensive scenario analysis using end-to-end workflow.
    
    This endpoint analyzes scenarios through the complete pipeline:
    1. Scenario parameter validation
    2. Scenario generation and analysis
    3. Outcome comparison
    4. Recommendation generation
    """
    request_id = str(uuid4())
    
    try:
        # Execute workflow
        workflow_result = await orchestrator.analyze_scenarios(
            retailer_id=request.retailer_id,
            base_scenario=request.base_scenario,
            scenario_variations=request.scenario_variations
        )
        
        if not workflow_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow failed: {workflow_result.error_message}"
            )
        
        response = APIResponse(
            success=True,
            data=workflow_result.data,
            message="Scenario analysis completed successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze scenarios: {str(e)}")


@router.post("/feedback", response_model=APIResponse)
async def process_feedback(
    request: FeedbackRequest,
    orchestrator = Depends(get_orchestrator)
):
    """
    Process retailer feedback and trigger learning updates.
    
    This endpoint processes feedback through:
    1. Feedback collection and validation
    2. Learning signal extraction
    3. Model update triggering (if needed)
    """
    request_id = str(uuid4())
    
    try:
        # Execute feedback processing
        result = await orchestrator.process_feedback_and_learn(
            retailer_id=request.retailer_id,
            feedback_data=request.feedback_data
        )
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Feedback processing failed: {result.get('error_message')}"
            )
        
        response = APIResponse(
            success=True,
            data=result,
            message="Feedback processed successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")


@router.get("/status/{workflow_id}", response_model=APIResponse)
async def get_workflow_status(
    workflow_id: str,
    orchestrator = Depends(get_orchestrator)
):
    """Get the status of a workflow execution."""
    request_id = str(uuid4())
    
    try:
        status = orchestrator.get_workflow_status(workflow_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        response = APIResponse(
            success=True,
            data=status,
            message="Workflow status retrieved successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.get("/system/health", response_model=APIResponse)
async def get_system_health(
    orchestrator = Depends(get_orchestrator)
):
    """Get overall system health status."""
    request_id = str(uuid4())
    
    try:
        health_status = orchestrator.get_system_health()
        
        response = APIResponse(
            success=True,
            data=health_status,
            message="System health retrieved successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


# Example workflow endpoints for common use cases
@router.post("/quick-insights/{retailer_id}", response_model=APIResponse)
async def quick_insights(
    retailer_id: str,
    product_ids: List[str],
    orchestrator = Depends(get_orchestrator)
):
    """
    Quick insights endpoint for common use case.
    
    Generates basic insights for specified products without full workflow complexity.
    """
    request_id = str(uuid4())
    
    try:
        # This would be a simplified workflow for quick results
        # For now, we'll use the full recommendations workflow
        workflow_result = await orchestrator.generate_recommendations(
            retailer_id=retailer_id,
            product_ids=product_ids,
            business_context={"quick_mode": True}
        )
        
        if not workflow_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Quick insights failed: {workflow_result.error_message}"
            )
        
        # Simplify the response for quick insights
        simplified_data = {
            "retailer_id": retailer_id,
            "insights_count": len(workflow_result.data.get("recommendations", [])),
            "top_recommendations": workflow_result.data.get("recommendations", [])[:3],
            "execution_time": workflow_result.execution_time_seconds
        }
        
        response = APIResponse(
            success=True,
            data=simplified_data,
            message="Quick insights generated successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quick insights: {str(e)}")