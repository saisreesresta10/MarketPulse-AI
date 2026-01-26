"""
Integration Tests for End-to-End Workflows

Tests complete workflows including component interactions, data flow,
and error propagation across the entire MarketPulse AI system.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone, timedelta, date
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from decimal import Decimal

from marketpulse_ai.core.models import SalesDataPoint, DemandPattern, ExplainableInsight, ConfidenceLevel
from marketpulse_ai.orchestrator import MarketPulseOrchestrator, WorkflowResult
from marketpulse_ai.api.main import ComponentManager


class TestWorkflowIntegration:
    """Test complete workflow integration."""
    
    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Create orchestrator with mock components for testing."""
        # Create mock components
        data_processor = AsyncMock()
        risk_assessor = AsyncMock()
        compliance_validator = AsyncMock()
        insight_generator = AsyncMock()
        decision_support_engine = AsyncMock()
        scenario_analyzer = AsyncMock()
        model_updater = AsyncMock()
        feedback_learner = AsyncMock()
        storage_manager = AsyncMock()
        
        # Configure mock behaviors
        data_processor.validate_sales_data.return_value = SalesDataPoint(
            product_id="PROD_001",
            product_name="Test Product",
            category="Electronics",
            mrp=Decimal("150.0"),
            selling_price=Decimal("100.0"),
            quantity_sold=10,
            sale_date=date.today(),
            store_location="Test_Store"
        )
        
        data_processor.extract_demand_patterns.return_value = [
            DemandPattern(
                product_id="PROD_001",
                pattern_type="seasonal",
                description="Strong seasonal demand pattern detected",
                confidence_level=ConfidenceLevel.HIGH,
                trend_direction="increasing",
                volatility_score=0.3,
                supporting_data_points=50,
                date_range_start=date.today() - timedelta(days=90),
                date_range_end=date.today()
            )
        ]
        
        insight_generator.generate_insight.return_value = ExplainableInsight(
            title="Seasonal Demand Pattern",
            description="Strong seasonal demand pattern detected for this product",
            confidence_level=ConfidenceLevel.HIGH,
            supporting_evidence=["Historical sales data", "Seasonal correlation"],
            key_factors=["seasonal_event", "market_conditions"],
            business_impact="Potential 30% revenue increase during peak seasons",
            data_sources=["sales_data", "seasonal_calendar"]
        )
        
        risk_assessor.assess_inventory_risk.return_value = MagicMock(
            risk_id="risk_001",
            product_id="PROD_001",
            risk_type="understock",
            risk_score=0.3,
            risk_level="low"
        )
        
        compliance_validator.validate_insight_compliance.return_value = MagicMock(
            compliance_id="comp_001",
            is_compliant=True,
            violations=[],
            recommendations=[]
        )
        
        return MarketPulseOrchestrator(
            data_processor=data_processor,
            risk_assessor=risk_assessor,
            compliance_validator=compliance_validator,
            insight_generator=insight_generator,
            decision_support_engine=decision_support_engine,
            scenario_analyzer=scenario_analyzer,
            model_updater=model_updater,
            feedback_learner=feedback_learner,
            storage_manager=storage_manager
        )
    
    @pytest.fixture
    def sample_sales_data(self):
        """Sample sales data for testing."""
        return [
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Festival Electronics",
                category="Electronics",
                mrp=Decimal("200.0"),
                selling_price=Decimal("150.0"),
                quantity_sold=25,
                sale_date=date.today() - timedelta(days=30),
                store_location="Mumbai_Store_01",
                seasonal_event="Festival Sale"
            ),
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Festival Electronics",
                category="Electronics",
                mrp=Decimal("200.0"),
                selling_price=Decimal("150.0"),
                quantity_sold=18,
                sale_date=date.today() - timedelta(days=20),
                store_location="Mumbai_Store_01"
            ),
            SalesDataPoint(
                product_id="PROD_002",
                product_name="Premium Goods",
                category="Home & Garden",
                mrp=Decimal("250.0"),
                selling_price=Decimal("200.0"),
                quantity_sold=32,
                sale_date=date.today() - timedelta(days=15),
                store_location="Delhi_Store_02"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_retailer_insights_workflow_success(self, orchestrator, sample_sales_data):
        """Test successful retailer insights workflow."""
        # Execute workflow
        result = await orchestrator.generate_retailer_insights(
            retailer_id="RETAILER_001",
            sales_data=sample_sales_data,
            include_risk_assessment=True,
            include_compliance_check=True
        )
        
        # Verify workflow success
        assert result.success is True
        assert result.workflow_type == "retailer_insights"
        assert result.execution_time_seconds > 0
        assert result.error_message is None
        
        # Verify workflow data
        assert result.data["retailer_id"] == "RETAILER_001"
        assert result.data["data_points_processed"] == len(sample_sales_data)
        assert result.data["insights_generated"] > 0
        assert "insights" in result.data
        assert "demand_patterns" in result.data
        assert "risk_assessments" in result.data
        assert "compliance_results" in result.data
        
        # Verify component interactions
        orchestrator.data_processor.validate_sales_data.assert_called()
        orchestrator.data_processor.extract_demand_patterns.assert_called()
        orchestrator.insight_generator.generate_insight.assert_called()
        orchestrator.risk_assessor.assess_inventory_risk.assert_called()
        orchestrator.compliance_validator.validate_insight_compliance.assert_called()
    
    @pytest.mark.asyncio
    async def test_retailer_insights_workflow_partial_options(self, orchestrator, sample_sales_data):
        """Test retailer insights workflow with partial options."""
        # Execute workflow without risk assessment
        result = await orchestrator.generate_retailer_insights(
            retailer_id="RETAILER_002",
            sales_data=sample_sales_data,
            include_risk_assessment=False,
            include_compliance_check=True
        )
        
        # Verify workflow success
        assert result.success is True
        assert result.data["retailer_id"] == "RETAILER_002"
        
        # Verify risk assessment was skipped
        assert len(result.data["risk_assessments"]) == 0
        orchestrator.risk_assessor.assess_inventory_risk.assert_not_called()
        
        # Verify compliance check was performed
        assert len(result.data["compliance_results"]) > 0
        orchestrator.compliance_validator.validate_insight_compliance.assert_called()
    
    @pytest.mark.asyncio
    async def test_retailer_insights_workflow_data_validation_error(self, orchestrator):
        """Test retailer insights workflow with data validation errors."""
        # Configure mock to raise validation error
        orchestrator.data_processor.validate_sales_data.side_effect = ValueError("Invalid data format")
        
        # Execute workflow with valid data structure but mock will raise error
        valid_data = [
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Test Product",
                category="Electronics",
                mrp=Decimal("150.0"),
                selling_price=Decimal("100.0"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test_Store"
            )
        ]
        
        result = await orchestrator.generate_retailer_insights(
            retailer_id="RETAILER_003",
            sales_data=valid_data
        )
        
        # Verify workflow failure
        assert result.success is False
        assert result.error_message is not None
        assert "No valid sales data points provided" in result.error_message
    
    @pytest.mark.asyncio
    async def test_recommendations_workflow_success(self, orchestrator):
        """Test successful recommendations workflow."""
        # Configure mock storage manager
        orchestrator.storage_manager.get_sales_data_by_product.return_value = [
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Test Product",
                category="Electronics",
                mrp=Decimal("150.0"),
                selling_price=Decimal("100.0"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test_Store"
            )
        ]
        
        # Configure mock decision support engine
        orchestrator.decision_support_engine.generate_recommendations.return_value = [
            {
                "recommendation_id": "rec_001",
                "title": "Optimize pricing strategy",
                "description": "Adjust prices based on demand patterns",
                "confidence_score": 0.85,
                "impact_score": 0.9,
                "urgency_score": 0.7
            }
        ]
        
        # Execute workflow
        result = await orchestrator.generate_recommendations(
            retailer_id="RETAILER_001",
            product_ids=["PROD_001", "PROD_002"],
            business_context={"target_margin": 0.3}
        )
        
        # Verify workflow success
        assert result.success is True
        assert result.workflow_type == "recommendations"
        assert result.data["retailer_id"] == "RETAILER_001"
        assert result.data["products_analyzed"] > 0
        assert "recommendations" in result.data
        
        # Verify component interactions
        orchestrator.storage_manager.get_sales_data_by_product.assert_called()
        orchestrator.decision_support_engine.generate_recommendations.assert_called()
        orchestrator.compliance_validator.validate_insight_compliance.assert_called()
    
    @pytest.mark.asyncio
    async def test_recommendations_workflow_no_data(self, orchestrator):
        """Test recommendations workflow with no product data."""
        # Configure mock to return no data
        orchestrator.storage_manager.get_sales_data_by_product.return_value = []
        
        # Execute workflow
        result = await orchestrator.generate_recommendations(
            retailer_id="RETAILER_004",
            product_ids=["NONEXISTENT_PROD"],
            business_context={}
        )
        
        # Verify workflow failure
        assert result.success is False
        assert "No valid product data found" in result.error_message
    
    @pytest.mark.asyncio
    async def test_scenario_analysis_workflow_success(self, orchestrator):
        """Test successful scenario analysis workflow."""
        # Configure mock scenario analyzer
        mock_scenario = MagicMock()
        mock_scenario.scenario_id = "scenario_001"
        mock_scenario.scenario_type = "base"
        mock_scenario.model_dump.return_value = {
            "scenario_id": "scenario_001",
            "scenario_type": "base",
            "parameters": {"discount_percentage": 0}
        }
        
        orchestrator.scenario_analyzer.generate_scenario.return_value = mock_scenario
        orchestrator.scenario_analyzer.analyze_scenario_impact.return_value = {
            "expected_revenue": 50000,
            "risk_score": 0.2,
            "confidence": 0.8
        }
        
        # Execute workflow
        base_scenario = {
            "product_id": "PROD_001",
            "time_horizon_days": 30,
            "current_inventory": 100,
            "discount_percentage": 0
        }
        
        scenario_variations = [
            {
                "product_id": "PROD_001",
                "time_horizon_days": 30,
                "current_inventory": 100,
                "discount_percentage": 10
            }
        ]
        
        result = await orchestrator.analyze_scenarios(
            retailer_id="RETAILER_001",
            base_scenario=base_scenario,
            scenario_variations=scenario_variations
        )
        
        # Verify workflow success
        assert result.success is True
        assert result.workflow_type == "scenario_analysis"
        assert result.data["retailer_id"] == "RETAILER_001"
        assert result.data["scenarios_analyzed"] > 0
        assert "scenario_analyses" in result.data
        assert "scenario_comparison" in result.data
        assert "recommendations" in result.data
        
        # Verify component interactions
        orchestrator.scenario_analyzer.generate_scenario.assert_called()
        orchestrator.scenario_analyzer.analyze_scenario_impact.assert_called()
    
    @pytest.mark.asyncio
    async def test_scenario_analysis_workflow_invalid_parameters(self, orchestrator):
        """Test scenario analysis workflow with invalid parameters."""
        # Execute workflow with invalid parameters
        invalid_base_scenario = {
            "product_id": "PROD_001",
            "time_horizon_days": -5,  # Invalid negative time horizon
            "discount_percentage": 150  # Invalid discount > 100%
        }
        
        result = await orchestrator.analyze_scenarios(
            retailer_id="RETAILER_005",
            base_scenario=invalid_base_scenario,
            scenario_variations=[]
        )
        
        # Verify workflow failure
        assert result.success is False
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_feedback_processing_workflow(self, orchestrator):
        """Test feedback processing and learning workflow."""
        # Configure mock feedback learner
        orchestrator.feedback_learner.collect_feedback.return_value = "feedback_001"
        
        # Configure mock model updater
        orchestrator.model_updater.integrate_new_data.return_value = {
            "data_points_processed": 5,
            "models_affected": ["demand_forecasting"],
            "update_operations_triggered": ["update_001"]
        }
        
        # Execute feedback processing
        feedback_data = {
            "type": "recommendation_rating",
            "target_id": "rec_001",
            "rating": 4.5,
            "trigger_model_update": True,
            "new_sales_data": [
                SalesDataPoint(
                    product_id="PROD_001",
                    product_name="Updated Product",
                    category="Electronics",
                    mrp=Decimal("150.0"),
                    selling_price=Decimal("120.0"),
                    quantity_sold=15,
                    sale_date=date.today(),
                    store_location="Test_Store"
                )
            ]
        }
        
        result = await orchestrator.process_feedback_and_learn(
            retailer_id="RETAILER_001",
            feedback_data=feedback_data
        )
        
        # Verify processing success
        assert result["status"] == "processed"
        assert result["feedback_id"] == "feedback_001"
        
        # Verify component interactions
        orchestrator.feedback_learner.collect_feedback.assert_called()
        orchestrator.model_updater.integrate_new_data.assert_called()
    
    @pytest.mark.asyncio
    async def test_workflow_error_propagation(self, orchestrator, sample_sales_data):
        """Test error propagation across workflow components."""
        # Configure mock to raise error in middle of workflow
        orchestrator.insight_generator.generate_insight.side_effect = RuntimeError("Insight generation failed")
        
        # Execute workflow
        result = await orchestrator.generate_retailer_insights(
            retailer_id="RETAILER_006",
            sales_data=sample_sales_data
        )
        
        # Verify error propagation
        assert result.success is False
        assert "Insight generation failed" in result.error_message
        
        # Verify partial execution (data processing should have been called)
        orchestrator.data_processor.validate_sales_data.assert_called()
        orchestrator.data_processor.extract_demand_patterns.assert_called()
        
        # Verify workflow was properly tracked
        assert result.workflow_id is not None
        assert result.execution_time_seconds > 0
    
    def test_workflow_status_tracking(self, orchestrator):
        """Test workflow status tracking functionality."""
        # Initially no workflows
        assert len(orchestrator.active_workflows) == 0
        assert len(orchestrator.completed_workflows) == 0
        
        # Test getting status for non-existent workflow
        status = orchestrator.get_workflow_status("nonexistent")
        assert status is None
    
    def test_system_health_monitoring(self, orchestrator):
        """Test system health monitoring."""
        health_status = orchestrator.get_system_health()
        
        # Verify health status structure
        assert "active_workflows" in health_status
        assert "completed_workflows" in health_status
        assert "components_status" in health_status
        assert "system_ready" in health_status
        
        # Verify component status
        components_status = health_status["components_status"]
        expected_components = [
            "data_processor", "risk_assessor", "compliance_validator",
            "insight_generator", "decision_support_engine", "scenario_analyzer",
            "model_updater", "feedback_learner"
        ]
        
        for component in expected_components:
            assert component in components_status
            assert components_status[component] == "healthy"


class TestComponentManagerIntegration:
    """Test component manager integration."""
    
    @pytest.mark.asyncio
    async def test_component_initialization(self):
        """Test component manager initialization."""
        component_manager = ComponentManager()
        
        # Test initialization in testing mode
        await component_manager.initialize_components(testing_mode=True)
        
        # Verify components are initialized
        assert component_manager.initialized is True
        assert len(component_manager.components) > 0
        
        # Verify key components exist
        expected_components = [
            "data_processor", "risk_assessor", "compliance_validator",
            "insight_generator", "decision_support_engine", "scenario_analyzer",
            "orchestrator"
        ]
        
        for component_name in expected_components:
            component = component_manager.get_component(component_name)
            assert component is not None
    
    @pytest.mark.asyncio
    async def test_component_shutdown(self):
        """Test component manager shutdown."""
        component_manager = ComponentManager()
        await component_manager.initialize_components(testing_mode=True)
        
        # Test shutdown
        await component_manager.shutdown_components()
        
        # Verify shutdown was called (components should still exist but be shut down)
        assert component_manager.initialized is True  # Still initialized, just shut down
    
    def test_get_all_components(self):
        """Test getting all components."""
        component_manager = ComponentManager()
        
        # Initially empty
        all_components = component_manager.get_all_components()
        assert len(all_components) == 0
    
    def test_get_nonexistent_component(self):
        """Test getting non-existent component."""
        component_manager = ComponentManager()
        
        component = component_manager.get_component("nonexistent")
        assert component is None


class TestWorkflowDataFlow:
    """Test data flow between workflow components."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components with realistic data flow."""
        components = {}
        
        # Mock data processor
        data_processor = AsyncMock()
        sample_pattern = DemandPattern(
            product_id="PROD_001",
            pattern_type="seasonal",
            description="Test pattern",
            confidence_level=ConfidenceLevel.HIGH,
            seasonal_factors={"winter": 1.2},
            trend_direction="increasing",
            volatility_score=0.3,
            supporting_data_points=50,
            date_range_start=date.today() - timedelta(days=90),
            date_range_end=date.today()
        )
        data_processor.extract_demand_patterns.return_value = [sample_pattern]
        components["data_processor"] = data_processor
        
        # Mock insight generator
        insight_generator = AsyncMock()
        sample_insight = ExplainableInsight(
            title="Strong seasonal demand detected",
            description="Strong seasonal demand detected for this product",
            confidence_level=ConfidenceLevel.HIGH,
            supporting_evidence=["Historical data"],
            key_factors=["seasonal_event"],
            business_impact="Potential revenue increase",
            data_sources=["sales_data"]
        )
        insight_generator.generate_insight.return_value = sample_insight
        components["insight_generator"] = insight_generator
        
        return components
    
    @pytest.mark.asyncio
    async def test_data_flow_pattern_to_insight(self, mock_components):
        """Test data flow from pattern extraction to insight generation."""
        data_processor = mock_components["data_processor"]
        insight_generator = mock_components["insight_generator"]
        
        # Simulate data flow
        sample_data = [
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Test Product",
                category="Electronics",
                mrp=Decimal("150.0"),
                selling_price=Decimal("100.0"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test_Store"
            )
        ]
        
        # Extract patterns
        patterns = await data_processor.extract_demand_patterns(sample_data)
        assert len(patterns) == 1
        
        # Generate insights from patterns
        insights = []
        for pattern in patterns:
            insight = await insight_generator.generate_insight(pattern)
            insights.append(insight)
        
        assert len(insights) == 1
        # Verify method calls
        data_processor.extract_demand_patterns.assert_called_once_with(sample_data)
        insight_generator.generate_insight.assert_called_once_with(patterns[0])
    
    @pytest.mark.asyncio
    async def test_error_handling_in_data_flow(self, mock_components):
        """Test error handling in component data flow."""
        data_processor = mock_components["data_processor"]
        insight_generator = mock_components["insight_generator"]
        
        # Configure error in insight generation
        insight_generator.generate_insight.side_effect = ValueError("Processing error")
        
        # Simulate data flow with error
        sample_data = [
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Test Product",
                category="Electronics",
                mrp=Decimal("150.0"),
                selling_price=Decimal("100.0"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test_Store"
            )
        ]
        
        # Extract patterns (should succeed)
        patterns = await data_processor.extract_demand_patterns(sample_data)
        assert len(patterns) == 1
        
        # Generate insights (should fail)
        with pytest.raises(ValueError, match="Processing error"):
            await insight_generator.generate_insight(patterns[0])


class TestWorkflowPerformance:
    """Test workflow performance and timing."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_timing(self):
        """Test that workflow execution times are recorded."""
        # Create orchestrator with fast mock components
        orchestrator = MarketPulseOrchestrator(
            data_processor=AsyncMock(),
            risk_assessor=AsyncMock(),
            compliance_validator=AsyncMock(),
            insight_generator=AsyncMock(),
            decision_support_engine=AsyncMock(),
            scenario_analyzer=AsyncMock(),
            model_updater=AsyncMock(),
            feedback_learner=AsyncMock(),
            storage_manager=AsyncMock()
        )
        
        # Configure fast responses
        orchestrator.data_processor.validate_sales_data.return_value = SalesDataPoint(
            product_id="PROD_001",
            product_name="Test Product",
            category="Electronics",
            mrp=Decimal("150.0"),
            selling_price=Decimal("100.0"),
            quantity_sold=10,
            sale_date=date.today(),
            store_location="Test_Store"
        )
        orchestrator.data_processor.extract_demand_patterns.return_value = []
        
        # Execute workflow
        sample_data = [
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Test Product",
                category="Electronics",
                mrp=Decimal("150.0"),
                selling_price=Decimal("100.0"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test_Store"
            )
        ]
        
        result = await orchestrator.generate_retailer_insights(
            retailer_id="RETAILER_PERF",
            sales_data=sample_data
        )
        
        # Verify timing is recorded
        assert result.execution_time_seconds > 0
        assert result.execution_time_seconds < 10  # Should be fast with mocks
        
        # Verify workflow metadata includes timing
        metadata = result.data.get("workflow_metadata", {})
        assert "execution_time" in metadata
        assert metadata["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """Test concurrent workflow execution."""
        # Create orchestrator
        orchestrator = MarketPulseOrchestrator(
            data_processor=AsyncMock(),
            risk_assessor=AsyncMock(),
            compliance_validator=AsyncMock(),
            insight_generator=AsyncMock(),
            decision_support_engine=AsyncMock(),
            scenario_analyzer=AsyncMock(),
            model_updater=AsyncMock(),
            feedback_learner=AsyncMock(),
            storage_manager=AsyncMock()
        )
        
        # Configure mock responses
        orchestrator.data_processor.validate_sales_data.return_value = SalesDataPoint(
            product_id="PROD_001",
            product_name="Test Product",
            category="Electronics",
            mrp=Decimal("150.0"),
            selling_price=Decimal("100.0"),
            quantity_sold=10,
            sale_date=date.today(),
            store_location="Test_Store"
        )
        orchestrator.data_processor.extract_demand_patterns.return_value = []
        
        # Execute multiple workflows concurrently
        sample_data = [
            SalesDataPoint(
                product_id="PROD_001",
                product_name="Test Product",
                category="Electronics",
                mrp=Decimal("150.0"),
                selling_price=Decimal("100.0"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test_Store"
            )
        ]
        
        tasks = []
        for i in range(3):
            task = orchestrator.generate_retailer_insights(
                retailer_id=f"RETAILER_{i}",
                sales_data=sample_data
            )
            tasks.append(task)
        
        # Wait for all workflows to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all workflows completed successfully
        assert len(results) == 3
        for result in results:
            assert result.success is True
            assert result.execution_time_seconds > 0
        
        # Verify unique workflow IDs
        workflow_ids = [result.workflow_id for result in results]
        assert len(set(workflow_ids)) == 3  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])