"""
MarketPulse AI System Orchestrator

Main orchestration layer that coordinates all components and manages
end-to-end workflows for retailer insights, recommendations, and scenarios.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from uuid import uuid4

from .core.models import (
    SalesDataPoint, DemandPattern, ExplainableInsight, 
    RiskAssessment, Scenario, ComplianceResult
)
from .components import (
    DataProcessor, RiskAssessor, ComplianceValidator,
    InsightGenerator, DecisionSupportEngine, ScenarioAnalyzer,
    ModelUpdater, FeedbackLearner
)
from .storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    workflow_id: str
    workflow_type: str
    success: bool
    data: Dict[str, Any]
    execution_time_seconds: float
    error_message: Optional[str] = None


class MarketPulseOrchestrator:
    """
    Main orchestrator for MarketPulse AI system.
    
    Coordinates all components to provide end-to-end workflows:
    - Retailer insight generation
    - Recommendation generation and validation
    - Scenario analysis and reporting
    """
    
    def __init__(
        self,
        data_processor: DataProcessor,
        risk_assessor: RiskAssessor,
        compliance_validator: ComplianceValidator,
        insight_generator: InsightGenerator,
        decision_support_engine: DecisionSupportEngine,
        scenario_analyzer: ScenarioAnalyzer,
        model_updater: ModelUpdater,
        feedback_learner: FeedbackLearner,
        storage_manager: StorageManager
    ):
        self.data_processor = data_processor
        self.risk_assessor = risk_assessor
        self.compliance_validator = compliance_validator
        self.insight_generator = insight_generator
        self.decision_support_engine = decision_support_engine
        self.scenario_analyzer = scenario_analyzer
        self.model_updater = model_updater
        self.feedback_learner = feedback_learner
        self.storage_manager = storage_manager
        
        # Workflow tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.completed_workflows: Dict[str, WorkflowResult] = {}
    
    async def generate_retailer_insights(
        self,
        retailer_id: str,
        sales_data: List[SalesDataPoint],
        include_risk_assessment: bool = True,
        include_compliance_check: bool = True
    ) -> WorkflowResult:
        """
        Complete workflow for generating retailer insights.
        
        Args:
            retailer_id: ID of the retailer
            sales_data: Historical sales data
            include_risk_assessment: Whether to include risk assessment
            include_compliance_check: Whether to include compliance validation
            
        Returns:
            Workflow result with generated insights
        """
        workflow_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting retailer insights workflow {workflow_id} for retailer {retailer_id}")
            
            # Track workflow
            self.active_workflows[workflow_id] = {
                "type": "retailer_insights",
                "retailer_id": retailer_id,
                "started_at": start_time,
                "status": "processing"
            }
            
            # Step 1: Process and validate sales data
            logger.info(f"Processing {len(sales_data)} sales data points")
            processed_data = []
            for data_point in sales_data:
                try:
                    validated_data = await self.data_processor.validate_sales_data(data_point)
                    processed_data.append(validated_data)
                except Exception as e:
                    logger.warning(f"Skipping invalid data point: {e}")
            
            if not processed_data:
                raise ValueError("No valid sales data points provided")
            
            # Step 2: Extract demand patterns
            logger.info("Extracting demand patterns")
            demand_patterns = await self.data_processor.extract_demand_patterns(processed_data)
            
            # Step 3: Generate insights from patterns
            logger.info("Generating insights from demand patterns")
            insights = []
            for pattern in demand_patterns:
                insight = await self.insight_generator.generate_insight(pattern)
                insights.append(insight)
            
            # Step 4: Risk assessment (if requested)
            risk_assessments = []
            if include_risk_assessment:
                logger.info("Performing risk assessment")
                for pattern in demand_patterns:
                    # Use a default inventory value if not available in pattern
                    current_inventory = getattr(pattern, 'current_inventory', 100)  # Default to 100 units
                    risk_assessment = await self.risk_assessor.assess_inventory_risk(
                        pattern, current_inventory=current_inventory
                    )
                    risk_assessments.append(risk_assessment)
            
            # Step 5: Compliance validation (if requested)
            compliance_results = []
            if include_compliance_check:
                logger.info("Validating compliance")
                for insight in insights:
                    compliance_result = await self.compliance_validator.validate_insight_compliance(insight)
                    compliance_results.append(compliance_result)
            
            # Step 6: Compile final results
            workflow_result = {
                "retailer_id": retailer_id,
                "insights": [insight.model_dump() for insight in insights],
                "demand_patterns": [pattern.model_dump() for pattern in demand_patterns],
                "risk_assessments": [risk.model_dump() for risk in risk_assessments],
                "compliance_results": [comp.model_dump() for comp in compliance_results],
                "data_points_processed": len(processed_data),
                "insights_generated": len(insights),
                "workflow_metadata": {
                    "workflow_id": workflow_id,
                    "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "components_used": ["data_processor", "insight_generator"] + 
                                     (["risk_assessor"] if include_risk_assessment else []) +
                                     (["compliance_validator"] if include_compliance_check else [])
                }
            }
            
            # Create workflow result
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type="retailer_insights",
                success=True,
                data=workflow_result,
                execution_time_seconds=execution_time
            )
            
            # Store completed workflow
            self.completed_workflows[workflow_id] = result
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            logger.info(f"Retailer insights workflow {workflow_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Retailer insights workflow {workflow_id} failed: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type="retailer_insights",
                success=False,
                data={},
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
            
            self.completed_workflows[workflow_id] = result
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            return result
    
    async def generate_recommendations(
        self,
        retailer_id: str,
        product_ids: List[str],
        business_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Complete workflow for generating recommendations.
        
        Args:
            retailer_id: ID of the retailer
            product_ids: List of product IDs to generate recommendations for
            business_context: Additional business context
            
        Returns:
            Workflow result with recommendations
        """
        workflow_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting recommendations workflow {workflow_id} for retailer {retailer_id}")
            
            # Track workflow
            self.active_workflows[workflow_id] = {
                "type": "recommendations",
                "retailer_id": retailer_id,
                "product_ids": product_ids,
                "started_at": start_time,
                "status": "processing"
            }
            
            # Step 1: Get current data for products
            logger.info(f"Retrieving data for {len(product_ids)} products")
            product_data = {}
            for product_id in product_ids:
                try:
                    # Get sales data and patterns for the product
                    sales_data = await self.storage_manager.get_sales_data_by_product(product_id)
                    if sales_data:
                        patterns = await self.data_processor.extract_demand_patterns(sales_data)
                        product_data[product_id] = {
                            "sales_data": sales_data,
                            "demand_patterns": patterns
                        }
                except Exception as e:
                    logger.warning(f"Could not retrieve data for product {product_id}: {e}")
            
            if not product_data:
                raise ValueError("No valid product data found")
            
            # Step 2: Generate recommendations for each product
            logger.info("Generating recommendations")
            recommendations = []
            for product_id, data in product_data.items():
                try:
                    # Use decision support engine to generate recommendations
                    product_recommendations = await self.decision_support_engine.generate_recommendations(
                        demand_patterns=data["demand_patterns"],
                        business_context=business_context or {}
                    )
                    
                    for rec in product_recommendations:
                        rec["product_id"] = product_id
                        recommendations.append(rec)
                        
                except Exception as e:
                    logger.warning(f"Could not generate recommendations for product {product_id}: {e}")
            
            # Step 3: Validate all recommendations for compliance
            logger.info("Validating recommendation compliance")
            validated_recommendations = []
            for rec in recommendations:
                try:
                    # Create a proper insight for compliance validation
                    mock_insight = ExplainableInsight(
                        insight_id=str(uuid4()),
                        pattern_id=rec.get("pattern_id", "unknown"),
                        title=rec.get("title", "Recommendation"),
                        description=rec.get("description", "Generated recommendation"),
                        confidence_level="medium",
                        supporting_evidence=["recommendation_analysis"],
                        key_factors=["business_context"],
                        business_impact=rec.get("business_impact", "Potential improvement"),
                        data_sources=["recommendation_engine"]
                    )
                    
                    compliance_result = await self.compliance_validator.validate_insight_compliance(mock_insight)
                    
                    if compliance_result.is_compliant:
                        validated_recommendations.append(rec)
                    else:
                        logger.warning(f"Recommendation failed compliance: {compliance_result.violations}")
                        
                except Exception as e:
                    logger.warning(f"Could not validate recommendation compliance: {e}")
                    # Include recommendation anyway if validation fails
                    validated_recommendations.append(rec)
            
            # Step 4: Prioritize and rank recommendations
            logger.info("Prioritizing recommendations")
            prioritized_recommendations = self._prioritize_recommendations(validated_recommendations)
            
            # Step 5: Compile final results
            workflow_result = {
                "retailer_id": retailer_id,
                "recommendations": prioritized_recommendations,
                "products_analyzed": len(product_data),
                "recommendations_generated": len(recommendations),
                "recommendations_validated": len(validated_recommendations),
                "workflow_metadata": {
                    "workflow_id": workflow_id,
                    "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                    "business_context": business_context
                }
            }
            
            # Create workflow result
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type="recommendations",
                success=True,
                data=workflow_result,
                execution_time_seconds=execution_time
            )
            
            # Store completed workflow
            self.completed_workflows[workflow_id] = result
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            logger.info(f"Recommendations workflow {workflow_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Recommendations workflow {workflow_id} failed: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type="recommendations",
                success=False,
                data={},
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
            
            self.completed_workflows[workflow_id] = result
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            return result
    
    async def analyze_scenarios(
        self,
        retailer_id: str,
        base_scenario: Dict[str, Any],
        scenario_variations: List[Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Complete workflow for scenario analysis.
        
        Args:
            retailer_id: ID of the retailer
            base_scenario: Base scenario parameters
            scenario_variations: List of scenario variations to analyze
            
        Returns:
            Workflow result with scenario analysis
        """
        workflow_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting scenario analysis workflow {workflow_id} for retailer {retailer_id}")
            
            # Track workflow
            self.active_workflows[workflow_id] = {
                "type": "scenario_analysis",
                "retailer_id": retailer_id,
                "started_at": start_time,
                "status": "processing"
            }
            
            # Step 1: Validate base scenario
            logger.info("Validating base scenario")
            validated_base = await self._validate_scenario_parameters(base_scenario)
            
            # Step 2: Generate scenarios
            logger.info(f"Generating {len(scenario_variations)} scenario variations")
            scenarios = []
            
            # Add base scenario
            base_scenario_obj = await self.scenario_analyzer.generate_scenario(
                scenario_type="base",
                parameters=validated_base
            )
            scenarios.append(base_scenario_obj)
            
            # Generate variations
            for i, variation in enumerate(scenario_variations):
                try:
                    validated_variation = await self._validate_scenario_parameters(variation)
                    scenario_obj = await self.scenario_analyzer.generate_scenario(
                        scenario_type=f"variation_{i+1}",
                        parameters=validated_variation
                    )
                    scenarios.append(scenario_obj)
                except Exception as e:
                    logger.warning(f"Could not generate scenario variation {i+1}: {e}")
            
            # Step 3: Analyze each scenario
            logger.info("Analyzing scenarios")
            scenario_analyses = []
            for scenario in scenarios:
                try:
                    analysis = await self.scenario_analyzer.analyze_scenario_impact(scenario)
                    scenario_analyses.append({
                        "scenario": scenario.model_dump(),
                        "analysis": analysis
                    })
                except Exception as e:
                    logger.warning(f"Could not analyze scenario {scenario.scenario_id}: {e}")
            
            # Step 4: Compare scenarios
            logger.info("Comparing scenario outcomes")
            comparison_result = self._compare_scenarios(scenario_analyses)
            
            # Step 5: Generate recommendations based on scenario analysis
            logger.info("Generating scenario-based recommendations")
            scenario_recommendations = await self._generate_scenario_recommendations(
                scenario_analyses, comparison_result
            )
            
            # Step 6: Compile final results
            workflow_result = {
                "retailer_id": retailer_id,
                "base_scenario": validated_base,
                "scenarios_analyzed": len(scenarios),
                "scenario_analyses": scenario_analyses,
                "scenario_comparison": comparison_result,
                "recommendations": scenario_recommendations,
                "workflow_metadata": {
                    "workflow_id": workflow_id,
                    "execution_time": (datetime.now(timezone.utc) - start_time).total_seconds()
                }
            }
            
            # Create workflow result
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type="scenario_analysis",
                success=True,
                data=workflow_result,
                execution_time_seconds=execution_time
            )
            
            # Store completed workflow
            self.completed_workflows[workflow_id] = result
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            logger.info(f"Scenario analysis workflow {workflow_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Scenario analysis workflow {workflow_id} failed: {e}")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type="scenario_analysis",
                success=False,
                data={},
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
            
            self.completed_workflows[workflow_id] = result
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            return result
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize recommendations based on impact and confidence."""
        def get_priority_score(rec):
            impact = rec.get("impact_score", 0.5)
            confidence = rec.get("confidence_score", 0.5)
            urgency = rec.get("urgency_score", 0.5)
            return (impact * 0.4) + (confidence * 0.3) + (urgency * 0.3)
        
        # Sort by priority score (highest first)
        prioritized = sorted(recommendations, key=get_priority_score, reverse=True)
        
        # Add priority rank
        for i, rec in enumerate(prioritized):
            rec["priority_rank"] = i + 1
            rec["priority_score"] = get_priority_score(rec)
        
        return prioritized
    
    async def _validate_scenario_parameters(self, scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize scenario parameters."""
        validated = scenario_params.copy()
        
        # Ensure required parameters exist
        required_params = ["product_id", "time_horizon_days"]
        for param in required_params:
            if param not in validated:
                raise ValueError(f"Missing required scenario parameter: {param}")
        
        # Validate parameter ranges
        if validated.get("discount_percentage", 0) < 0 or validated.get("discount_percentage", 0) > 100:
            raise ValueError("Discount percentage must be between 0 and 100")
        
        if validated.get("time_horizon_days", 0) <= 0:
            raise ValueError("Time horizon must be positive")
        
        return validated
    
    def _compare_scenarios(self, scenario_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare scenario outcomes and identify best options."""
        if not scenario_analyses:
            return {}
        
        # Extract key metrics for comparison
        metrics = []
        for analysis in scenario_analyses:
            scenario_data = analysis["scenario"]
            analysis_data = analysis["analysis"]
            
            metrics.append({
                "scenario_id": scenario_data["scenario_id"],
                "scenario_type": scenario_data["scenario_type"],
                "expected_revenue": analysis_data.get("expected_revenue", 0),
                "risk_score": analysis_data.get("risk_score", 0.5),
                "confidence": analysis_data.get("confidence", 0.5)
            })
        
        # Find best scenarios by different criteria
        best_revenue = max(metrics, key=lambda x: x["expected_revenue"])
        lowest_risk = min(metrics, key=lambda x: x["risk_score"])
        highest_confidence = max(metrics, key=lambda x: x["confidence"])
        
        # Calculate overall best (weighted score)
        for metric in metrics:
            revenue_norm = metric["expected_revenue"] / max(m["expected_revenue"] for m in metrics) if max(m["expected_revenue"] for m in metrics) > 0 else 0
            risk_norm = 1 - (metric["risk_score"] / max(m["risk_score"] for m in metrics)) if max(m["risk_score"] for m in metrics) > 0 else 1
            confidence_norm = metric["confidence"]
            
            metric["overall_score"] = (revenue_norm * 0.4) + (risk_norm * 0.3) + (confidence_norm * 0.3)
        
        best_overall = max(metrics, key=lambda x: x["overall_score"])
        
        return {
            "total_scenarios": len(scenario_analyses),
            "best_revenue_scenario": best_revenue,
            "lowest_risk_scenario": lowest_risk,
            "highest_confidence_scenario": highest_confidence,
            "best_overall_scenario": best_overall,
            "scenario_metrics": metrics
        }
    
    async def _generate_scenario_recommendations(
        self,
        scenario_analyses: List[Dict[str, Any]],
        comparison_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on scenario analysis."""
        recommendations = []
        
        if not scenario_analyses or not comparison_result:
            return recommendations
        
        # Recommend best overall scenario
        best_overall = comparison_result.get("best_overall_scenario")
        if best_overall:
            recommendations.append({
                "type": "scenario_recommendation",
                "title": "Recommended Scenario",
                "description": f"Based on analysis, scenario {best_overall['scenario_id']} offers the best overall outcome",
                "scenario_id": best_overall["scenario_id"],
                "confidence": best_overall["confidence"],
                "expected_outcome": {
                    "revenue": best_overall["expected_revenue"],
                    "risk_score": best_overall["risk_score"],
                    "overall_score": best_overall["overall_score"]
                }
            })
        
        # Recommend risk mitigation if high-risk scenarios exist
        high_risk_scenarios = [
            m for m in comparison_result.get("scenario_metrics", [])
            if m["risk_score"] > 0.7
        ]
        
        if high_risk_scenarios:
            recommendations.append({
                "type": "risk_mitigation",
                "title": "Risk Mitigation Needed",
                "description": f"{len(high_risk_scenarios)} scenarios show high risk levels",
                "high_risk_scenarios": [s["scenario_id"] for s in high_risk_scenarios],
                "mitigation_suggestions": [
                    "Consider reducing discount percentages",
                    "Implement gradual rollout strategy",
                    "Monitor inventory levels closely"
                ]
            })
        
        return recommendations
    
    async def process_feedback_and_learn(
        self,
        retailer_id: str,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process retailer feedback and trigger learning updates."""
        try:
            logger.info(f"Processing feedback from retailer {retailer_id}")
            
            # Extract feedback parameters
            feedback_type = feedback_data.get("type", "general")
            target_id = feedback_data.get("target_id", "unknown")
            
            # Remove these from feedback_data to avoid duplicate keyword arguments
            feedback_params = feedback_data.copy()
            feedback_params.pop("type", None)
            feedback_params.pop("target_id", None)
            
            # Collect feedback
            feedback_id = await self.feedback_learner.collect_feedback(
                retailer_id=retailer_id,
                feedback_type=feedback_type,
                target_id=target_id,
                **feedback_params
            )
            
            # Check if model updates are needed
            if feedback_data.get("trigger_model_update", False):
                # Integrate new data if provided
                if "new_sales_data" in feedback_data:
                    integration_result = await self.model_updater.integrate_new_data(
                        feedback_data["new_sales_data"]
                    )
                    logger.info(f"Integrated new data: {integration_result}")
            
            return {
                "feedback_id": feedback_id,
                "status": "processed",
                "message": "Feedback processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": "active",
                "type": workflow["type"],
                "started_at": workflow["started_at"].isoformat(),
                "current_status": workflow.get("status", "processing")
            }
        
        # Check completed workflows
        if workflow_id in self.completed_workflows:
            result = self.completed_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "type": result.workflow_type,
                "success": result.success,
                "execution_time_seconds": result.execution_time_seconds,
                "error_message": result.error_message
            }
        
        return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        return {
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "components_status": {
                "data_processor": "healthy",
                "risk_assessor": "healthy",
                "compliance_validator": "healthy",
                "insight_generator": "healthy",
                "decision_support_engine": "healthy",
                "scenario_analyzer": "healthy",
                "model_updater": "healthy",
                "feedback_learner": "healthy"
            },
            "system_ready": True
        }