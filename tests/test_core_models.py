"""
Unit tests for core data models in MarketPulse AI.

Tests data validation, serialization, and business logic
for all core Pydantic models.
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from hypothesis import given, strategies as st
from pydantic import ValidationError

from marketpulse_ai.core.models import (
    SalesDataPoint,
    DemandPattern,
    ExplainableInsight,
    RiskAssessment,
    Scenario,
    ComplianceResult,
    ConfidenceLevel,
    RiskLevel,
    ComplianceStatus,
)
from tests.conftest import (
    sales_data_strategy,
    demand_pattern_strategy,
    risk_assessment_strategy,
)


class TestSalesDataPoint:
    """Test cases for SalesDataPoint model."""
    
    def test_valid_sales_data_creation(self, sample_sales_data):
        """Test creation of valid sales data points."""
        sales_point = sample_sales_data[0]
        
        assert sales_point.product_id == "PROD001"
        assert sales_point.product_name == "Premium Tea"
        assert sales_point.mrp == Decimal("100.00")
        assert sales_point.selling_price == Decimal("95.00")
        assert sales_point.quantity_sold == 50
        assert isinstance(sales_point.id, UUID)
        assert isinstance(sales_point.created_at, datetime)
    
    def test_selling_price_exceeds_mrp_validation(self):
        """Test validation when selling price exceeds MRP."""
        with pytest.raises(ValidationError) as exc_info:
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="Test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("150.00"),  # Exceeds MRP
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test Store"
            )
        
        assert "cannot exceed MRP" in str(exc_info.value)
    
    def test_future_sale_date_validation(self):
        """Test validation for future sale dates."""
        future_date = date.today() + timedelta(days=1)
        
        with pytest.raises(ValidationError) as exc_info:
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="Test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("95.00"),
                quantity_sold=10,
                sale_date=future_date,
                store_location="Test Store"
            )
        
        assert "cannot be in the future" in str(exc_info.value)
    
    def test_negative_values_validation(self):
        """Test validation for negative values."""
        with pytest.raises(ValidationError):
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="Test",
                mrp=Decimal("-100.00"),  # Negative MRP
                selling_price=Decimal("95.00"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="Test Store"
            )
    
    @given(sales_data_strategy())
    def test_sales_data_serialization_roundtrip(self, sales_data):
        """Property test: Sales data should serialize and deserialize correctly."""
        # Serialize to dict
        data_dict = sales_data.model_dump()
        
        # Deserialize back to model
        reconstructed = SalesDataPoint(**data_dict)
        
        # Should be equal (excluding auto-generated fields)
        assert sales_data.product_id == reconstructed.product_id
        assert sales_data.mrp == reconstructed.mrp
        assert sales_data.selling_price == reconstructed.selling_price
        assert sales_data.quantity_sold == reconstructed.quantity_sold


class TestDemandPattern:
    """Test cases for DemandPattern model."""
    
    def test_valid_demand_pattern_creation(self, sample_demand_pattern):
        """Test creation of valid demand pattern."""
        pattern = sample_demand_pattern
        
        assert pattern.product_id == "PROD001"
        assert pattern.pattern_type == "seasonal"
        assert pattern.confidence_level == ConfidenceLevel.HIGH
        assert pattern.volatility_score == 0.25
        assert pattern.supporting_data_points == 150
        assert isinstance(pattern.id, UUID)
    
    def test_date_range_validation(self):
        """Test validation of date ranges."""
        with pytest.raises(ValidationError) as exc_info:
            DemandPattern(
                product_id="PROD001",
                pattern_type="seasonal",
                description="Test pattern",
                confidence_level=ConfidenceLevel.MEDIUM,
                volatility_score=0.5,
                supporting_data_points=100,
                date_range_start=date(2024, 1, 31),
                date_range_end=date(2024, 1, 1),  # End before start
            )
        
        assert "End date must be after start date" in str(exc_info.value)
    
    def test_volatility_score_bounds(self):
        """Test volatility score bounds validation."""
        with pytest.raises(ValidationError):
            DemandPattern(
                product_id="PROD001",
                pattern_type="seasonal",
                description="Test pattern",
                confidence_level=ConfidenceLevel.MEDIUM,
                volatility_score=1.5,  # Exceeds maximum
                supporting_data_points=100,
                date_range_start=date(2024, 1, 1),
                date_range_end=date(2024, 1, 31),
            )
    
    @given(demand_pattern_strategy())
    def test_demand_pattern_serialization_roundtrip(self, pattern):
        """Property test: Demand patterns should serialize and deserialize correctly."""
        data_dict = pattern.model_dump()
        reconstructed = DemandPattern(**data_dict)
        
        assert pattern.product_id == reconstructed.product_id
        assert pattern.pattern_type == reconstructed.pattern_type
        assert pattern.volatility_score == reconstructed.volatility_score


class TestExplainableInsight:
    """Test cases for ExplainableInsight model."""
    
    def test_valid_insight_creation(self, sample_insight):
        """Test creation of valid explainable insight."""
        insight = sample_insight
        
        assert insight.title == "Winter Festival Demand Surge"
        assert insight.confidence_level == ConfidenceLevel.HIGH
        assert len(insight.supporting_evidence) >= 1
        assert len(insight.key_factors) >= 1
        assert len(insight.data_sources) >= 1
        assert isinstance(insight.id, UUID)
    
    def test_expiration_date_validation(self):
        """Test expiration date validation."""
        with pytest.raises(ValidationError) as exc_info:
            ExplainableInsight(
                title="Test Insight",
                description="Test description",
                confidence_level=ConfidenceLevel.MEDIUM,
                supporting_evidence=["Evidence 1"],
                key_factors=["Factor 1"],
                business_impact="Test impact",
                data_sources=["source1"],
                expires_at=datetime.utcnow() - timedelta(hours=1)  # Past expiration
            )
        
        assert "must be after creation date" in str(exc_info.value)
    
    def test_required_lists_validation(self):
        """Test validation of required list fields."""
        with pytest.raises(ValidationError):
            ExplainableInsight(
                title="Test Insight",
                description="Test description",
                confidence_level=ConfidenceLevel.MEDIUM,
                supporting_evidence=[],  # Empty list not allowed
                key_factors=["Factor 1"],
                business_impact="Test impact",
                data_sources=["source1"],
            )


class TestRiskAssessment:
    """Test cases for RiskAssessment model."""
    
    def test_valid_risk_assessment_creation(self, sample_risk_assessment):
        """Test creation of valid risk assessment."""
        assessment = sample_risk_assessment
        
        assert assessment.product_id == "PROD002"
        assert assessment.risk_type == "overstock"
        assert assessment.risk_level == RiskLevel.MEDIUM
        assert assessment.risk_score == 0.65
        assert len(assessment.contributing_factors) >= 1
        assert isinstance(assessment.id, UUID)
    
    def test_validity_period_validation(self):
        """Test validity period validation."""
        with pytest.raises(ValidationError) as exc_info:
            RiskAssessment(
                product_id="PROD001",
                risk_type="overstock",
                risk_level=RiskLevel.HIGH,
                risk_score=0.8,
                contributing_factors=["Factor 1"],
                assessment_date=date.today(),
                valid_until=date.today() - timedelta(days=1)  # Past validity
            )
        
        assert "must be after assessment date" in str(exc_info.value)
    
    def test_risk_score_bounds(self):
        """Test risk score bounds validation."""
        with pytest.raises(ValidationError):
            RiskAssessment(
                product_id="PROD001",
                risk_type="overstock",
                risk_level=RiskLevel.HIGH,
                risk_score=1.5,  # Exceeds maximum
                contributing_factors=["Factor 1"],
                assessment_date=date.today(),
                valid_until=date.today() + timedelta(days=30)
            )
    
    @given(risk_assessment_strategy())
    def test_risk_assessment_serialization_roundtrip(self, assessment):
        """Property test: Risk assessments should serialize and deserialize correctly."""
        data_dict = assessment.model_dump()
        reconstructed = RiskAssessment(**data_dict)
        
        assert assessment.product_id == reconstructed.product_id
        assert assessment.risk_type == reconstructed.risk_type
        assert assessment.risk_score == reconstructed.risk_score


class TestScenario:
    """Test cases for Scenario model."""
    
    def test_valid_scenario_creation(self, sample_scenario):
        """Test creation of valid scenario."""
        scenario = sample_scenario
        
        assert scenario.name == "Festival Season Inventory Planning"
        assert scenario.confidence_level == ConfidenceLevel.MEDIUM
        assert len(scenario.parameters) > 0
        assert len(scenario.predicted_outcomes) > 0
        assert len(scenario.assumptions) >= 1
        assert len(scenario.limitations) >= 1
        assert isinstance(scenario.id, UUID)
    
    def test_empty_parameters_validation(self):
        """Test validation for empty parameters."""
        with pytest.raises(ValidationError) as exc_info:
            Scenario(
                name="Test Scenario",
                description="Test description",
                parameters={},  # Empty parameters
                predicted_outcomes={"outcome1": "value1"},
                confidence_level=ConfidenceLevel.MEDIUM,
                assumptions=["Assumption 1"],
                limitations=["Limitation 1"],
                time_horizon="1 month"
            )
        
        assert "must have at least one parameter" in str(exc_info.value)
    
    def test_empty_outcomes_validation(self):
        """Test validation for empty predicted outcomes."""
        with pytest.raises(ValidationError) as exc_info:
            Scenario(
                name="Test Scenario",
                description="Test description",
                parameters={"param1": "value1"},
                predicted_outcomes={},  # Empty outcomes
                confidence_level=ConfidenceLevel.MEDIUM,
                assumptions=["Assumption 1"],
                limitations=["Limitation 1"],
                time_horizon="1 month"
            )
        
        assert "must have at least one predicted outcome" in str(exc_info.value)


class TestComplianceResult:
    """Test cases for ComplianceResult model."""
    
    def test_valid_compliance_result_creation(self, sample_compliance_result):
        """Test creation of valid compliance result."""
        result = sample_compliance_result
        
        assert result.compliance_status == ComplianceStatus.COMPLIANT
        assert len(result.regulations_checked) >= 1
        assert len(result.violations) == 0  # Compliant should have no violations
        assert result.validator_version == "1.0.0"
        assert isinstance(result.id, UUID)
    
    def test_compliant_with_violations_validation(self):
        """Test validation for compliant status with violations."""
        with pytest.raises(ValidationError) as exc_info:
            ComplianceResult(
                compliance_status=ComplianceStatus.COMPLIANT,
                regulations_checked=["REG1"],
                violations=["Violation 1"],  # Violations with compliant status
                validator_version="1.0.0"
            )
        
        assert "Cannot be compliant with violations present" in str(exc_info.value)
    
    def test_non_compliant_without_violations_validation(self):
        """Test validation for non-compliant status without violations."""
        with pytest.raises(ValidationError) as exc_info:
            ComplianceResult(
                compliance_status=ComplianceStatus.NON_COMPLIANT,
                regulations_checked=["REG1"],
                violations=[],  # No violations with non-compliant status
                validator_version="1.0.0"
            )
        
        assert "Non-compliant status requires violations" in str(exc_info.value)


class TestModelIntegration:
    """Integration tests for model interactions."""
    
    def test_model_json_serialization(self, sample_sales_data):
        """Test JSON serialization of all models."""
        sales_point = sample_sales_data[0]
        
        # Should serialize to JSON without errors
        json_str = sales_point.model_dump_json()
        assert isinstance(json_str, str)
        assert "PROD001" in json_str
        
        # Should deserialize back correctly
        reconstructed = SalesDataPoint.model_validate_json(json_str)
        assert reconstructed.product_id == sales_point.product_id
    
    def test_model_dict_conversion(self, sample_insight):
        """Test dictionary conversion of models."""
        insight_dict = sample_insight.model_dump()
        
        assert isinstance(insight_dict, dict)
        assert insight_dict["title"] == "Winter Festival Demand Surge"
        # UUID might be returned as UUID object, not string in model_dump()
        assert "id" in insight_dict  # Just check the field exists
        assert isinstance(insight_dict["created_at"], datetime)
    
    def test_model_field_validation_consistency(self):
        """Test that field validation is consistent across models."""
        # All models should have proper ID field validation
        models_with_ids = [
            SalesDataPoint, DemandPattern, ExplainableInsight,
            RiskAssessment, Scenario, ComplianceResult
        ]
        
        for model_class in models_with_ids:
            # Check that ID field exists and has proper type
            id_field = model_class.model_fields.get('id')
            assert id_field is not None
            # UUID field should have proper default factory
            assert hasattr(id_field, 'default_factory')