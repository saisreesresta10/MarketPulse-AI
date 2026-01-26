"""
Unit tests for Risk Assessor component.

Tests overstock risk assessment, understock risk assessment, demand volatility
calculation, seasonal adjustments, and early warning generation.
"""

import pytest
import asyncio
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

import numpy as np

from marketpulse_ai.components.risk_assessor import (
    RiskAssessor, 
    RiskCalculationError, 
    InsufficientDataError
)
from marketpulse_ai.core.models import (
    SalesDataPoint, 
    RiskAssessment, 
    RiskLevel, 
    DemandPattern,
    ConfidenceLevel
)

logger = logging.getLogger(__name__)


class TestRiskAssessor:
    """Test suite for RiskAssessor component."""
    
    @pytest.fixture
    def risk_assessor(self):
        """Create a RiskAssessor instance for testing."""
        return RiskAssessor()
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for testing."""
        base_date = date.today() - timedelta(days=365)
        sales_data = []
        
        # Generate 12 months of sales data with some seasonality
        for i in range(365):
            current_date = base_date + timedelta(days=i)
            
            # Base quantity with some seasonality (higher in Oct-Dec)
            base_qty = 50
            if current_date.month in [10, 11, 12]:
                seasonal_boost = 1.5
            elif current_date.month in [6, 7, 8]:
                seasonal_boost = 0.7
            else:
                seasonal_boost = 1.0
            
            # Add some randomness
            quantity = int(base_qty * seasonal_boost * (0.8 + np.random.random() * 0.4))
            
            sales_data.append(SalesDataPoint(
                product_id="TEST_PRODUCT_001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("950.00"),
                quantity_sold=quantity,
                sale_date=current_date,
                store_location="TEST_STORE"
            ))
        
        return sales_data
    
    @pytest.fixture
    def sample_demand_patterns(self):
        """Create sample demand patterns for testing."""
        pattern = DemandPattern(
            product_id="TEST_PRODUCT_001",
            pattern_type="seasonal",
            description="Test seasonal pattern",
            confidence_level=ConfidenceLevel.HIGH,
            seasonal_factors={
                'diwali': 1.8,
                'summer': 0.7,
                'winter': 1.2
            },
            trend_direction="stable",
            volatility_score=0.3,
            supporting_data_points=100,
            date_range_start=date.today() - timedelta(days=365),
            date_range_end=date.today()
        )
        return {"TEST_PRODUCT_001": [pattern]}
    
    @pytest.mark.asyncio
    async def test_assess_overstock_risk_high_inventory(self, risk_assessor, sample_sales_data):
        """Test overstock risk assessment with high inventory levels."""
        # Set up test data
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Test with very high inventory (5x average demand)
        high_inventory = 250  # Much higher than average ~50
        
        assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", high_inventory)
        
        # Assertions
        assert assessment.product_id == "TEST_PRODUCT_001"
        assert assessment.risk_type == "overstock"
        assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert assessment.risk_score > 0.5
        assert len(assessment.contributing_factors) > 0
        assert len(assessment.mitigation_suggestions) > 0
        assert "inventory" in str(assessment.contributing_factors).lower()
    
    @pytest.mark.asyncio
    async def test_assess_overstock_risk_normal_inventory(self, risk_assessor, sample_sales_data):
        """Test overstock risk assessment with normal inventory levels."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Test with normal inventory (close to average demand)
        normal_inventory = 60
        
        assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", normal_inventory)
        
        # Assertions
        assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        assert assessment.risk_score < 0.6
    
    @pytest.mark.asyncio
    async def test_assess_understock_risk_low_inventory(self, risk_assessor, sample_sales_data):
        """Test understock risk assessment with low inventory levels."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Test with very low inventory
        low_inventory = 10  # Much lower than average ~50
        
        assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", low_inventory)
        
        # Assertions
        assert assessment.product_id == "TEST_PRODUCT_001"
        assert assessment.risk_type == "understock"
        assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert assessment.risk_score > 0.5
        assert len(assessment.contributing_factors) > 0
        assert len(assessment.mitigation_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_assess_understock_risk_adequate_inventory(self, risk_assessor, sample_sales_data):
        """Test understock risk assessment with adequate inventory levels."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Test with adequate inventory
        adequate_inventory = 100
        
        assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", adequate_inventory)
        
        # Assertions
        assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        assert assessment.risk_score < 0.6
    
    @pytest.mark.asyncio
    async def test_calculate_demand_volatility(self, risk_assessor, sample_sales_data):
        """Test demand volatility calculation."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        volatility = await risk_assessor.calculate_demand_volatility("TEST_PRODUCT_001")
        
        # Assertions
        assert 0.0 <= volatility <= 1.0
        assert isinstance(volatility, float)
    
    @pytest.mark.asyncio
    async def test_calculate_demand_volatility_high_variance(self, risk_assessor):
        """Test demand volatility calculation with high variance data."""
        # Create high variance sales data
        high_variance_data = []
        base_date = date.today() - timedelta(days=100)
        
        for i in range(100):
            # Alternate between very high and very low quantities
            quantity = 100 if i % 2 == 0 else 10
            
            high_variance_data.append(SalesDataPoint(
                product_id="HIGH_VARIANCE_PRODUCT",
                product_name="High Variance Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=quantity,
                sale_date=base_date + timedelta(days=i),
                store_location="TEST_STORE"
            ))
        
        risk_assessor.set_sales_data(high_variance_data)
        
        volatility = await risk_assessor.calculate_demand_volatility("HIGH_VARIANCE_PRODUCT")
        
        # High variance should result in high volatility score
        assert volatility > 0.5
    
    @pytest.mark.asyncio
    async def test_adjust_for_seasonal_events(self, risk_assessor, sample_sales_data, sample_demand_patterns):
        """Test seasonal adjustment of risk assessments."""
        risk_assessor.set_sales_data(sample_sales_data)
        risk_assessor.set_demand_patterns(sample_demand_patterns)
        
        # Create base assessment
        base_assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", 100)
        
        # Test adjustment for high-demand season (should reduce overstock risk)
        upcoming_events = ["diwali"]
        adjusted_assessment = await risk_assessor.adjust_for_seasonal_events(
            base_assessment, upcoming_events
        )
        
        # Assertions
        assert adjusted_assessment.product_id == base_assessment.product_id
        assert len(adjusted_assessment.seasonal_adjustments) > 0
        assert "diwali" in adjusted_assessment.seasonal_adjustments
        # For overstock risk, high-demand season should reduce risk
        assert adjusted_assessment.risk_score <= base_assessment.risk_score
    
    @pytest.mark.asyncio
    async def test_adjust_for_seasonal_events_understock(self, risk_assessor, sample_sales_data, sample_demand_patterns):
        """Test seasonal adjustment for understock risk."""
        risk_assessor.set_sales_data(sample_sales_data)
        risk_assessor.set_demand_patterns(sample_demand_patterns)
        
        # Create base understock assessment
        base_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", 20)
        
        # Test adjustment for high-demand season (should increase understock risk)
        upcoming_events = ["diwali"]
        adjusted_assessment = await risk_assessor.adjust_for_seasonal_events(
            base_assessment, upcoming_events
        )
        
        # For understock risk, high-demand season should increase risk
        assert adjusted_assessment.risk_score >= base_assessment.risk_score
    
    @pytest.mark.asyncio
    async def test_generate_early_warnings(self, risk_assessor, sample_sales_data):
        """Test early warning generation."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Create assessments with different risk levels
        high_risk_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", 5)
        low_risk_assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", 60)
        
        assessments = [high_risk_assessment, low_risk_assessment]
        
        # Generate early warnings
        warned_assessments = await risk_assessor.generate_early_warnings(assessments)
        
        # Assertions
        assert len(warned_assessments) == 2
        
        # High risk should trigger warning
        high_risk_result = next(a for a in warned_assessments if a.risk_type == "understock")
        if high_risk_result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            assert high_risk_result.early_warning_triggered
        
        # Low risk should not trigger warning
        low_risk_result = next(a for a in warned_assessments if a.risk_type == "overstock")
        if low_risk_result.risk_level == RiskLevel.LOW:
            assert not low_risk_result.early_warning_triggered
    
    @pytest.mark.asyncio
    async def test_insufficient_data_error(self, risk_assessor):
        """Test error handling for insufficient data."""
        # Set minimal data (less than required)
        minimal_data = [
            SalesDataPoint(
                product_id="MINIMAL_PRODUCT",
                product_name="Minimal Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="TEST_STORE"
            )
        ]
        
        risk_assessor.set_sales_data(minimal_data)
        
        # Should raise RiskCalculationError (which wraps InsufficientDataError)
        with pytest.raises(RiskCalculationError):
            await risk_assessor.assess_overstock_risk("MINIMAL_PRODUCT", 50)
    
    @pytest.mark.asyncio
    async def test_nonexistent_product_error(self, risk_assessor, sample_sales_data):
        """Test error handling for nonexistent product."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Should raise RiskCalculationError (which wraps InsufficientDataError) for nonexistent product
        with pytest.raises(RiskCalculationError):
            await risk_assessor.assess_overstock_risk("NONEXISTENT_PRODUCT", 50)
    
    @pytest.mark.asyncio
    async def test_risk_assessment_caching(self, risk_assessor, sample_sales_data):
        """Test that risk assessments are properly cached."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Perform assessment
        assessment1 = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", 100)
        
        # Check that assessment is cached
        cache_key = "overstock_TEST_PRODUCT_001_100"
        assert cache_key in risk_assessor.risk_cache
        assert risk_assessor.risk_cache[cache_key].product_id == "TEST_PRODUCT_001"
    
    @pytest.mark.asyncio
    async def test_detect_upcoming_seasonal_events(self, risk_assessor):
        """Test automatic detection of upcoming seasonal events."""
        upcoming_events = await risk_assessor._detect_upcoming_seasonal_events()
        
        # Should return a list of events
        assert isinstance(upcoming_events, list)
        
        # Events should be strings
        for event in upcoming_events:
            assert isinstance(event, str)
        
        # Should detect at least some seasonal events (depending on current date)
        # Note: This test may vary based on when it's run
        logger.info(f"Detected upcoming events: {upcoming_events}")
    
    @pytest.mark.asyncio
    async def test_enhanced_seasonal_adjustments(self, risk_assessor, sample_sales_data, sample_demand_patterns):
        """Test enhanced seasonal adjustment functionality."""
        risk_assessor.set_sales_data(sample_sales_data)
        risk_assessor.set_demand_patterns(sample_demand_patterns)
        
        # Test with both manual and auto-detected events
        manual_events = ["diwali", "christmas"]
        
        # Get seasonal adjustments
        adjustments = await risk_assessor._calculate_seasonal_adjustments("TEST_PRODUCT_001", manual_events)
        
        # Should include manual events
        assert "diwali" in adjustments
        assert "christmas" in adjustments
        
        # Should have reasonable adjustment factors
        for event, factor in adjustments.items():
            assert 0.1 <= factor <= 3.0  # Reasonable range
    
    @pytest.mark.asyncio
    async def test_seasonal_risk_impact_assessment(self, risk_assessor, sample_sales_data, sample_demand_patterns):
        """Test comprehensive seasonal risk impact assessment."""
        risk_assessor.set_sales_data(sample_sales_data)
        risk_assessor.set_demand_patterns(sample_demand_patterns)
        
        # Perform comprehensive seasonal assessment
        analysis = await risk_assessor.assess_seasonal_risk_impact("TEST_PRODUCT_001", 100)
        
        # Verify analysis structure
        assert "product_id" in analysis
        assert "current_inventory" in analysis
        assert "upcoming_events" in analysis
        assert "base_assessments" in analysis
        assert "seasonal_adjusted_assessments" in analysis
        assert "seasonal_impact_metrics" in analysis
        assert "seasonal_recommendations" in analysis
        assert "risk_summary" in analysis
        
        # Verify base assessments
        assert "overstock" in analysis["base_assessments"]
        assert "understock" in analysis["base_assessments"]
        
        # Verify seasonal adjustments were applied
        assert "overstock" in analysis["seasonal_adjusted_assessments"]
        assert "understock" in analysis["seasonal_adjusted_assessments"]
        
        # Verify metrics
        metrics = analysis["seasonal_impact_metrics"]
        assert "max_demand_multiplier" in metrics
        assert "min_demand_multiplier" in metrics
        assert "seasonal_volatility" in metrics
        
        # Verify recommendations are provided
        assert isinstance(analysis["seasonal_recommendations"], list)
        assert len(analysis["seasonal_recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_early_warning_seasonal_triggers(self, risk_assessor, sample_sales_data, sample_demand_patterns):
        """Test enhanced early warning system with seasonal triggers."""
        risk_assessor.set_sales_data(sample_sales_data)
        risk_assessor.set_demand_patterns(sample_demand_patterns)
        
        # Create assessment with seasonal adjustments
        base_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", 20)
        
        # Apply seasonal adjustments for high-demand event
        seasonal_assessment = await risk_assessor.adjust_for_seasonal_events(
            base_assessment, ["diwali"]  # High-impact event
        )
        
        # Test early warning logic
        should_warn = await risk_assessor._should_trigger_early_warning(seasonal_assessment)
        
        # Should trigger warning for seasonal high-demand event with medium+ risk
        if seasonal_assessment.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            assert should_warn
    
    @pytest.mark.asyncio
    async def test_seasonal_mitigation_suggestions(self, risk_assessor, sample_sales_data, sample_demand_patterns):
        """Test seasonal-specific mitigation suggestions."""
        risk_assessor.set_sales_data(sample_sales_data)
        risk_assessor.set_demand_patterns(sample_demand_patterns)
        
        # Create understock assessment with seasonal adjustments
        base_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", 15)
        seasonal_assessment = await risk_assessor.adjust_for_seasonal_events(
            base_assessment, ["diwali", "christmas"]
        )
        
        # Generate warning suggestions
        warning_suggestions = await risk_assessor._generate_warning_mitigation_suggestions(seasonal_assessment)
        
        # Should include seasonal-specific suggestions
        suggestions_text = " ".join(warning_suggestions).lower()
        assert any(keyword in suggestions_text for keyword in ["seasonal", "festival", "diwali", "christmas"])
        
        # Should include specific seasonal actions
        assert any(keyword in suggestions_text for keyword in ["seasonal alert", "increased demand", "festival"])
    
    @pytest.mark.asyncio
    async def test_time_based_adjustment(self, risk_assessor):
        """Test time-based adjustment of seasonal factors."""
        # Test proximity adjustment
        base_factor = 1.5
        adjusted_factor = await risk_assessor._apply_time_based_adjustment("diwali", base_factor)
        
        # Should return a reasonable adjustment
        assert 0.5 <= adjusted_factor <= 3.0
        
        # High-impact events should get proximity boost
        assert adjusted_factor >= base_factor  # Should be same or higher due to proximity
    
    @pytest.mark.asyncio
    async def test_seasonal_impact_metrics_calculation(self, risk_assessor, sample_sales_data):
        """Test calculation of seasonal impact metrics."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        upcoming_events = ["diwali", "summer", "christmas"]
        metrics = await risk_assessor._calculate_seasonal_impact_metrics("TEST_PRODUCT_001", 100, upcoming_events)
        
        # Verify metrics structure
        assert "max_demand_multiplier" in metrics
        assert "min_demand_multiplier" in metrics
        assert "average_demand_multiplier" in metrics
        assert "seasonal_volatility" in metrics
        assert "high_impact_events" in metrics
        assert "low_impact_events" in metrics
        assert "neutral_events" in metrics
        
        # Verify metrics values are reasonable
        assert metrics["max_demand_multiplier"] >= metrics["min_demand_multiplier"]
        assert 0.0 <= metrics["seasonal_volatility"] <= 2.0
        
        # Verify event categorization
        all_categorized_events = (metrics["high_impact_events"] + 
                                metrics["low_impact_events"] + 
                                metrics["neutral_events"])
        assert len(all_categorized_events) >= len(upcoming_events)  # May include auto-detected events
    
    @pytest.mark.asyncio
    async def test_seasonal_recommendations_generation(self, risk_assessor, sample_sales_data):
        """Test generation of seasonal recommendations."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Create mock assessments
        overstock_assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", 200)
        understock_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", 200)
        
        # Create mock seasonal metrics
        seasonal_metrics = {
            'max_demand_multiplier': 1.8,
            'min_demand_multiplier': 0.7,
            'average_demand_multiplier': 1.2,
            'seasonal_volatility': 0.4,
            'high_impact_events': ['diwali', 'christmas'],
            'low_impact_events': ['summer'],
            'neutral_events': ['winter'],
            'inventory_coverage_at_peak': 0.8,
            'additional_inventory_needed': 50
        }
        
        recommendations = await risk_assessor._generate_seasonal_recommendations(
            overstock_assessment, understock_assessment, seasonal_metrics
        )
        
        # Should generate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include seasonal strategy recommendations
        recommendations_text = " ".join(recommendations).lower()
        assert "seasonal" in recommendations_text
        
        # Should mention high-impact events
        assert any(event in recommendations_text for event in ['diwali', 'christmas'])
    
    @pytest.mark.asyncio
    async def test_calendar_based_event_detection_edge_cases(self, risk_assessor):
        """Test edge cases in calendar-based event detection."""
        # Test the detection logic doesn't crash with edge dates
        upcoming_events = await risk_assessor._detect_upcoming_seasonal_events()
        
        # Should handle year boundaries and leap years gracefully
        assert isinstance(upcoming_events, list)
        
        # All detected events should be valid strings
        for event in upcoming_events:
            assert isinstance(event, str)
            assert len(event) > 0
    
    @pytest.mark.asyncio
    async def test_volatility_calculation_edge_cases(self, risk_assessor):
        """Test volatility calculation with edge cases."""
        # Test with constant values (zero volatility)
        constant_data = []
        base_date = date.today() - timedelta(days=30)
        
        for i in range(30):
            constant_data.append(SalesDataPoint(
                product_id="CONSTANT_PRODUCT",
                product_name="Constant Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=50,  # Same quantity every day
                sale_date=base_date + timedelta(days=i),
                store_location="TEST_STORE"
            ))
        
        risk_assessor.set_sales_data(constant_data)
        
        volatility = await risk_assessor.calculate_demand_volatility("CONSTANT_PRODUCT")
        
        # Constant values should result in very low volatility
        assert volatility < 0.1
    
    @pytest.mark.asyncio
    async def test_risk_level_boundaries(self, risk_assessor, sample_sales_data):
        """Test risk level boundary conditions."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Test various inventory levels to check boundary conditions
        inventory_levels = [1, 10, 50, 100, 200, 500]
        
        for inventory in inventory_levels:
            overstock_assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", inventory)
            understock_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", inventory)
            
            # Risk levels should be valid
            assert overstock_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            assert understock_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            # Risk scores should be in valid range
            assert 0.0 <= overstock_assessment.risk_score <= 1.0
            assert 0.0 <= understock_assessment.risk_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_mitigation_suggestions_quality(self, risk_assessor, sample_sales_data):
        """Test that mitigation suggestions are meaningful and actionable."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Test high-risk scenarios
        high_overstock_assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", 300)
        high_understock_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", 5)
        
        # Overstock suggestions should mention promotions, discounts, or inventory reduction
        overstock_suggestions = " ".join(high_overstock_assessment.mitigation_suggestions).lower()
        assert any(keyword in overstock_suggestions for keyword in ["promotion", "discount", "reduce", "clearance"])
        
        # Understock suggestions should mention replenishment, procurement, or inventory increase
        understock_suggestions = " ".join(high_understock_assessment.mitigation_suggestions).lower()
        assert any(keyword in understock_suggestions for keyword in ["replenish", "procur", "increase", "expedite"])
    
    @pytest.mark.asyncio
    async def test_contributing_factors_accuracy(self, risk_assessor, sample_sales_data):
        """Test that contributing factors accurately reflect the risk situation."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        # Test extreme overstock
        extreme_overstock_assessment = await risk_assessor.assess_overstock_risk("TEST_PRODUCT_001", 500)
        
        factors_text = " ".join(extreme_overstock_assessment.contributing_factors).lower()
        
        # Should mention high inventory levels
        assert any(keyword in factors_text for keyword in ["inventory", "excess", "coverage"])
        
        # Test extreme understock
        extreme_understock_assessment = await risk_assessor.assess_understock_risk("TEST_PRODUCT_001", 2)
        
        understock_factors_text = " ".join(extreme_understock_assessment.contributing_factors).lower()
        
        # Should mention low inventory or stockout risk
        assert any(keyword in understock_factors_text for keyword in ["service", "stockout", "probability", "days"])
    
    def test_risk_assessor_initialization(self):
        """Test RiskAssessor initialization and configuration."""
        assessor = RiskAssessor()
        
        # Check default initialization
        assert assessor.sales_data == []
        assert assessor.demand_patterns == {}
        assert assessor.risk_cache == {}
        assert assessor.storage_manager is None
        
        # Check risk thresholds are properly set
        assert 'overstock' in assessor.risk_thresholds
        assert 'understock' in assessor.risk_thresholds
        assert 'volatility' in assessor.risk_thresholds
    
    def test_set_sales_data(self, risk_assessor, sample_sales_data):
        """Test setting sales data."""
        risk_assessor.set_sales_data(sample_sales_data)
        
        assert len(risk_assessor.sales_data) == len(sample_sales_data)
        assert risk_assessor.sales_data[0].product_id == "TEST_PRODUCT_001"
    
    def test_set_demand_patterns(self, risk_assessor, sample_demand_patterns):
        """Test setting demand patterns."""
        risk_assessor.set_demand_patterns(sample_demand_patterns)
        
        assert len(risk_assessor.demand_patterns) == 1
        assert "TEST_PRODUCT_001" in risk_assessor.demand_patterns
    
    def test_set_storage_manager(self, risk_assessor):
        """Test setting storage manager."""
        mock_storage = Mock()
        risk_assessor.set_storage_manager(mock_storage)
        
        assert risk_assessor.storage_manager == mock_storage


class TestRiskAssessorEdgeCases:
    """Test edge cases and error conditions for RiskAssessor."""
    
    @pytest.fixture
    def risk_assessor(self):
        return RiskAssessor()
    
    @pytest.mark.asyncio
    async def test_zero_inventory_overstock(self, risk_assessor):
        """Test overstock assessment with zero inventory."""
        # Create minimal sales data
        sales_data = []
        base_date = date.today() - timedelta(days=30)
        
        for i in range(10):
            sales_data.append(SalesDataPoint(
                product_id="ZERO_INV_PRODUCT",
                product_name="Zero Inventory Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=10,
                sale_date=base_date + timedelta(days=i*3),
                store_location="TEST_STORE"
            ))
        
        risk_assessor.set_sales_data(sales_data)
        
        # Zero inventory should result in low overstock risk
        assessment = await risk_assessor.assess_overstock_risk("ZERO_INV_PRODUCT", 0)
        assert assessment.risk_level == RiskLevel.LOW
        assert assessment.risk_score < 0.3
    
    @pytest.mark.asyncio
    async def test_zero_inventory_understock(self, risk_assessor):
        """Test understock assessment with zero inventory."""
        # Create sales data
        sales_data = []
        base_date = date.today() - timedelta(days=30)
        
        for i in range(10):
            sales_data.append(SalesDataPoint(
                product_id="ZERO_INV_PRODUCT",
                product_name="Zero Inventory Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=10,
                sale_date=base_date + timedelta(days=i*3),
                store_location="TEST_STORE"
            ))
        
        risk_assessor.set_sales_data(sales_data)
        
        # Zero inventory should result in high understock risk
        assessment = await risk_assessor.assess_understock_risk("ZERO_INV_PRODUCT", 0)
        assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert assessment.risk_score > 0.7
    
    @pytest.mark.asyncio
    async def test_single_sale_volatility(self, risk_assessor):
        """Test volatility calculation with minimal data points."""
        # Create data with only 5 sales (minimum for volatility calculation)
        minimal_sales = []
        base_date = date.today() - timedelta(days=10)
        
        for i in range(5):
            minimal_sales.append(SalesDataPoint(
                product_id="MINIMAL_SALES_PRODUCT",
                product_name="Minimal Sales Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=10 + i,  # Slight variation
                sale_date=base_date + timedelta(days=i*2),
                store_location="TEST_STORE"
            ))
        
        risk_assessor.set_sales_data(minimal_sales)
        
        # Should not raise error and return reasonable volatility
        volatility = await risk_assessor.calculate_demand_volatility("MINIMAL_SALES_PRODUCT")
        assert 0.0 <= volatility <= 1.0
    
    @pytest.mark.asyncio
    async def test_empty_seasonal_events(self, risk_assessor):
        """Test seasonal adjustment with empty events list."""
        # Create basic sales data
        sales_data = []
        base_date = date.today() - timedelta(days=30)
        
        for i in range(10):
            sales_data.append(SalesDataPoint(
                product_id="EMPTY_EVENTS_PRODUCT",
                product_name="Empty Events Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=20,
                sale_date=base_date + timedelta(days=i*3),
                store_location="TEST_STORE"
            ))
        
        risk_assessor.set_sales_data(sales_data)
        
        base_assessment = await risk_assessor.assess_overstock_risk("EMPTY_EVENTS_PRODUCT", 50)
        
        # Empty events list should now auto-detect seasonal events (enhanced behavior)
        adjusted_assessment = await risk_assessor.adjust_for_seasonal_events(base_assessment, [])
        
        # Should have auto-detected some seasonal events
        assert len(adjusted_assessment.seasonal_adjustments) > 0
        
        # Risk level should be reasonable
        assert adjusted_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        # Risk score should be within valid range
        assert 0.0 <= adjusted_assessment.risk_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])