"""
Property-Based Tests for Risk Assessor Component

Property tests validating universal correctness properties for risk assessment,
seasonal adjustments, demand volatility calculations, and early warning systems.

**Property 4: Comprehensive Risk Assessment**
**Property 5: Seasonal Risk Adjustment**
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

import pytest
import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from uuid import uuid4

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite
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


# Hypothesis strategies for generating test data
@composite
def sales_data_point_strategy(draw):
    """Generate valid SalesDataPoint instances for risk assessment testing."""
    product_id = draw(st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"))
    product_name = draw(st.text(min_size=5, max_size=50, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "))
    category = draw(st.sampled_from(['electronics', 'clothing', 'food', 'books', 'home', 'sports', 'beauty']))
    
    # Generate realistic price ranges
    mrp_value = draw(st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    mrp = Decimal(f"{mrp_value:.2f}")
    
    selling_price_value = draw(st.floats(min_value=5.0, max_value=float(mrp), allow_nan=False, allow_infinity=False))
    selling_price = Decimal(f"{selling_price_value:.2f}")
    
    # Generate realistic quantity ranges
    quantity_sold = draw(st.integers(min_value=1, max_value=500))
    
    # Generate dates within reasonable range (last 2 years)
    base_date = date.today() - timedelta(days=730)
    days_offset = draw(st.integers(min_value=0, max_value=729))
    sale_date = base_date + timedelta(days=days_offset)
    
    store_location = draw(st.text(min_size=3, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"))
    
    seasonal_event = draw(st.one_of(
        st.none(),
        st.sampled_from(['diwali', 'holi', 'eid', 'christmas', 'new_year', 'summer', 'winter', 'monsoon'])
    ))
    
    return SalesDataPoint(
        product_id=product_id,
        product_name=product_name,
        category=category,
        mrp=mrp,
        selling_price=selling_price,
        quantity_sold=quantity_sold,
        sale_date=sale_date,
        store_location=store_location,
        seasonal_event=seasonal_event
    )


@composite
def sales_data_list_strategy(draw):
    """Generate lists of SalesDataPoint instances with sufficient data for risk assessment."""
    # Ensure we have enough data points for meaningful risk assessment
    return draw(st.lists(sales_data_point_strategy(), min_size=5, max_size=100))


@composite
def inventory_level_strategy(draw):
    """Generate realistic inventory levels for testing."""
    return draw(st.integers(min_value=0, max_value=1000))


@composite
def seasonal_factors_strategy(draw):
    """Generate realistic seasonal factors for demand patterns."""
    events = ['diwali', 'holi', 'christmas', 'eid', 'summer', 'winter', 'monsoon', 'new_year']
    num_events = draw(st.integers(min_value=0, max_value=len(events)))
    
    if num_events == 0:
        return {}
    
    selected_events = draw(st.lists(st.sampled_from(events), min_size=num_events, max_size=num_events, unique=True))
    factors = {}
    
    for event in selected_events:
        # Generate realistic seasonal factors (0.2 to 3.0 range)
        factor = draw(st.floats(min_value=0.2, max_value=3.0, allow_nan=False, allow_infinity=False))
        factors[event] = factor
    
    return factors


@composite
def demand_pattern_strategy(draw):
    """Generate valid DemandPattern instances for risk assessment."""
    product_id = draw(st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"))
    pattern_type = draw(st.sampled_from(['seasonal', 'trend', 'cyclical', 'volatile', 'stable']))
    description = draw(st.text(min_size=10, max_size=100, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ."))
    confidence_level = draw(st.sampled_from([ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]))
    
    seasonal_factors = draw(seasonal_factors_strategy())
    trend_direction = draw(st.sampled_from(['increasing', 'decreasing', 'stable']))
    volatility_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    # Generate date range
    start_date = date.today() - timedelta(days=365)
    end_date = date.today()
    
    return DemandPattern(
        product_id=product_id,
        pattern_type=pattern_type,
        description=description,
        confidence_level=confidence_level,
        seasonal_factors=seasonal_factors,
        trend_direction=trend_direction,
        volatility_score=volatility_score,
        supporting_data_points=draw(st.integers(min_value=10, max_value=1000)),
        date_range_start=start_date,
        date_range_end=end_date
    )


@composite
def upcoming_events_strategy(draw):
    """Generate lists of upcoming seasonal events."""
    events = ['diwali', 'holi', 'christmas', 'eid', 'summer', 'winter', 'monsoon', 'new_year', 'wedding_season']
    return draw(st.lists(st.sampled_from(events), min_size=0, max_size=5, unique=True))


class TestRiskAssessorProperties:
    """
    Property-based tests for Risk Assessor component.
    
    **Property 4: Comprehensive Risk Assessment**
    **Property 5: Seasonal Risk Adjustment**
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """
    
    @given(sales_data_list_strategy(), inventory_level_strategy())
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_comprehensive_overstock_risk_assessment(self, sales_data, inventory_level):
        """
        **Property 4.1: Comprehensive Overstock Risk Assessment**
        **Validates: Requirements 3.1, 3.3, 3.5**
        
        Property: For any combination of sales data and inventory level, overstock risk assessment
        should produce consistent, valid risk assessments with proper risk scores, levels, and factors.
        """
        # Ensure we have a consistent product ID for all sales data
        if not sales_data:
            return
        
        product_id = sales_data[0].product_id
        for sale in sales_data:
            sale.product_id = product_id
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(sales_data)
        
        try:
            # Property: Overstock assessment should work for any valid inputs
            assessment = await risk_assessor.assess_overstock_risk(product_id, inventory_level)
            
            # Property: Assessment should be a valid RiskAssessment instance
            assert isinstance(assessment, RiskAssessment)
            assert assessment.product_id == product_id
            assert assessment.risk_type == "overstock"
            
            # Property: Risk level should be valid
            assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            # Property: Risk score should be in valid range
            assert 0.0 <= assessment.risk_score <= 1.0
            assert isinstance(assessment.risk_score, float)
            
            # Property: Contributing factors should be provided
            assert isinstance(assessment.contributing_factors, list)
            assert len(assessment.contributing_factors) > 0
            
            # Property: Mitigation suggestions should be provided
            assert isinstance(assessment.mitigation_suggestions, list)
            assert len(assessment.mitigation_suggestions) > 0
            
            # Property: Valid until date should be in the future
            assert assessment.valid_until > date.today()
            
            # Property: Higher inventory should generally lead to higher overstock risk
            if inventory_level > 0:
                # Test with double inventory
                high_inventory_assessment = await risk_assessor.assess_overstock_risk(product_id, inventory_level * 2)
                assert high_inventory_assessment.risk_score >= assessment.risk_score
            
            # Property: Risk assessment should be deterministic for same inputs
            assessment_repeat = await risk_assessor.assess_overstock_risk(product_id, inventory_level)
            assert assessment_repeat.risk_score == assessment.risk_score
            assert assessment_repeat.risk_level == assessment.risk_level
            
        except (RiskCalculationError, InsufficientDataError):
            # Property: Insufficient data should be handled gracefully
            # This is acceptable behavior for insufficient data scenarios
            pass
    
    @given(sales_data_list_strategy(), inventory_level_strategy())
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_comprehensive_understock_risk_assessment(self, sales_data, inventory_level):
        """
        **Property 4.2: Comprehensive Understock Risk Assessment**
        **Validates: Requirements 3.2, 3.3, 3.5**
        
        Property: For any combination of sales data and inventory level, understock risk assessment
        should produce consistent, valid risk assessments with appropriate risk calculations.
        """
        # Ensure we have a consistent product ID for all sales data
        if not sales_data:
            return
        
        product_id = sales_data[0].product_id
        for sale in sales_data:
            sale.product_id = product_id
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(sales_data)
        
        try:
            # Property: Understock assessment should work for any valid inputs
            assessment = await risk_assessor.assess_understock_risk(product_id, inventory_level)
            
            # Property: Assessment should be a valid RiskAssessment instance
            assert isinstance(assessment, RiskAssessment)
            assert assessment.product_id == product_id
            assert assessment.risk_type == "understock"
            
            # Property: Risk level should be valid
            assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            # Property: Risk score should be in valid range
            assert 0.0 <= assessment.risk_score <= 1.0
            assert isinstance(assessment.risk_score, float)
            
            # Property: Contributing factors should be provided
            assert isinstance(assessment.contributing_factors, list)
            assert len(assessment.contributing_factors) > 0
            
            # Property: Mitigation suggestions should be provided
            assert isinstance(assessment.mitigation_suggestions, list)
            assert len(assessment.mitigation_suggestions) > 0
            
            # Property: Valid until date should be in the future
            assert assessment.valid_until > date.today()
            
            # Property: Lower inventory should generally lead to higher understock risk
            if inventory_level > 1:
                # Test with half inventory
                low_inventory_assessment = await risk_assessor.assess_understock_risk(product_id, inventory_level // 2)
                assert low_inventory_assessment.risk_score >= assessment.risk_score
            
            # Property: Zero inventory should result in high understock risk
            if inventory_level == 0:
                assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                assert assessment.risk_score > 0.5
            
        except (RiskCalculationError, InsufficientDataError):
            # Property: Insufficient data should be handled gracefully
            pass
    
    @given(sales_data_list_strategy())
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_demand_volatility_calculation_consistency(self, sales_data):
        """
        **Property 4.3: Demand Volatility Calculation Consistency**
        **Validates: Requirements 3.3**
        
        Property: For any sales data, demand volatility calculation should produce
        consistent, bounded volatility scores that reflect actual demand variation.
        """
        # Ensure we have a consistent product ID and sufficient data
        if len(sales_data) < 5:
            return
        
        product_id = sales_data[0].product_id
        for sale in sales_data:
            sale.product_id = product_id
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(sales_data)
        
        try:
            # Property: Volatility calculation should work for any sufficient data
            volatility = await risk_assessor.calculate_demand_volatility(product_id)
            
            # Property: Volatility should be in valid range
            assert 0.0 <= volatility <= 1.0
            assert isinstance(volatility, float)
            
            # Property: Volatility calculation should be deterministic
            volatility_repeat = await risk_assessor.calculate_demand_volatility(product_id)
            assert abs(volatility - volatility_repeat) < 1e-10  # Should be identical
            
            # Property: Constant demand should result in low volatility
            constant_sales = []
            base_date = date.today() - timedelta(days=30)
            for i in range(10):
                constant_sales.append(SalesDataPoint(
                    product_id="CONSTANT_PRODUCT",
                    product_name="Constant Product",
                    category="test",
                    mrp=Decimal("100.00"),
                    selling_price=Decimal("90.00"),
                    quantity_sold=50,  # Same quantity every time
                    sale_date=base_date + timedelta(days=i*3),
                    store_location="TEST_STORE"
                ))
            
            risk_assessor.set_sales_data(constant_sales)
            constant_volatility = await risk_assessor.calculate_demand_volatility("CONSTANT_PRODUCT")
            assert constant_volatility < 0.2  # Should be very low for constant demand
            
        except (RiskCalculationError, InsufficientDataError):
            # Property: Insufficient data should be handled gracefully
            pass
    
    @given(sales_data_list_strategy(), upcoming_events_strategy())
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_seasonal_risk_adjustment_consistency(self, sales_data, upcoming_events):
        """
        **Property 5.1: Seasonal Risk Adjustment Consistency**
        **Validates: Requirements 3.4**
        
        Property: For any risk assessment and upcoming seasonal events, seasonal adjustments
        should modify risk scores appropriately while maintaining assessment validity.
        """
        # Ensure we have a consistent product ID and sufficient data
        if len(sales_data) < 5:
            return
        
        product_id = sales_data[0].product_id
        for sale in sales_data:
            sale.product_id = product_id
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(sales_data)
        
        try:
            # Create base assessment
            base_assessment = await risk_assessor.assess_overstock_risk(product_id, 100)
            
            # Property: Seasonal adjustment should work for any events list
            adjusted_assessment = await risk_assessor.adjust_for_seasonal_events(
                base_assessment, upcoming_events
            )
            
            # Property: Adjusted assessment should maintain core properties
            assert isinstance(adjusted_assessment, RiskAssessment)
            assert adjusted_assessment.product_id == base_assessment.product_id
            assert adjusted_assessment.risk_type == base_assessment.risk_type
            
            # Property: Risk score should remain in valid range after adjustment
            assert 0.0 <= adjusted_assessment.risk_score <= 1.0
            
            # Property: Risk level should be valid after adjustment
            assert adjusted_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            # Property: Seasonal adjustments should be recorded if events provided
            if upcoming_events:
                assert hasattr(adjusted_assessment, 'seasonal_adjustments')
                if adjusted_assessment.seasonal_adjustments:
                    assert isinstance(adjusted_assessment.seasonal_adjustments, dict)
                    # All adjustment factors should be reasonable
                    for event, factor in adjusted_assessment.seasonal_adjustments.items():
                        assert 0.1 <= factor <= 5.0  # Reasonable seasonal adjustment range
            
            # Property: Contributing factors should be updated with seasonal information
            if upcoming_events and adjusted_assessment.seasonal_adjustments:
                factors_text = " ".join(adjusted_assessment.contributing_factors).lower()
                # Should mention at least one seasonal event if adjustments were made
                significant_adjustments = [event for event, factor in adjusted_assessment.seasonal_adjustments.items() 
                                         if abs(factor - 1.0) > 0.1]
                if significant_adjustments:
                    assert any(event.lower() in factors_text for event in significant_adjustments)
            
            # Property: Empty events list should still return valid assessment
            empty_events_assessment = await risk_assessor.adjust_for_seasonal_events(
                base_assessment, []
            )
            assert isinstance(empty_events_assessment, RiskAssessment)
            assert 0.0 <= empty_events_assessment.risk_score <= 1.0
            
        except (RiskCalculationError, InsufficientDataError):
            # Property: Insufficient data should be handled gracefully
            pass
    
    @given(st.lists(sales_data_list_strategy(), min_size=1, max_size=3))
    @settings(max_examples=5, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_early_warning_generation_consistency(self, sales_data_lists):
        """
        **Property 4.4: Early Warning Generation Consistency**
        **Validates: Requirements 3.5**
        
        Property: For any list of risk assessments, early warning generation should
        consistently identify high-risk situations and provide appropriate warnings.
        """
        if not sales_data_lists:
            return
        
        risk_assessor = RiskAssessor()
        assessments = []
        
        # Create multiple assessments from different sales data
        for i, sales_data in enumerate(sales_data_lists):
            if len(sales_data) < 5:
                continue
            
            product_id = f"PRODUCT_{i}"
            for sale in sales_data:
                sale.product_id = product_id
            
            risk_assessor.set_sales_data(sales_data)
            
            try:
                # Create both overstock and understock assessments with different risk levels
                overstock_assessment = await risk_assessor.assess_overstock_risk(product_id, 500)  # High inventory
                understock_assessment = await risk_assessor.assess_understock_risk(product_id, 5)   # Low inventory
                
                assessments.extend([overstock_assessment, understock_assessment])
                
            except (RiskCalculationError, InsufficientDataError):
                continue
        
        if not assessments:
            return
        
        # Property: Early warning generation should work for any assessment list
        warned_assessments = await risk_assessor.generate_early_warnings(assessments)
        
        # Property: Output should have same number of assessments
        assert len(warned_assessments) == len(assessments)
        
        # Property: All assessments should remain valid after warning processing
        for assessment in warned_assessments:
            assert isinstance(assessment, RiskAssessment)
            assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            assert 0.0 <= assessment.risk_score <= 1.0
            
            # Property: Early warning flag should be boolean
            assert isinstance(assessment.early_warning_triggered, bool)
            
            # Property: High and critical risk assessments should more likely trigger warnings
            if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                # Not guaranteed, but high-risk should have higher probability of warnings
                pass
            
            # Property: If warning is triggered, additional suggestions should be provided
            if assessment.early_warning_triggered:
                # Should have warning-specific mitigation suggestions
                assert len(assessment.mitigation_suggestions) > 0
                suggestions_text = " ".join(assessment.mitigation_suggestions).lower()
                assert any(keyword in suggestions_text for keyword in ["urgent", "immediate", "emergency", "alert"])
    
    @given(sales_data_list_strategy(), inventory_level_strategy())
    @settings(max_examples=5, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_seasonal_risk_impact_assessment_completeness(self, sales_data, inventory_level):
        """
        **Property 5.2: Seasonal Risk Impact Assessment Completeness**
        **Validates: Requirements 3.4, 3.5**
        
        Property: For any product and inventory level, comprehensive seasonal risk impact
        assessment should provide complete analysis including base assessments, seasonal
        adjustments, impact metrics, and recommendations.
        """
        # Ensure we have sufficient data and consistent product ID
        if len(sales_data) < 5:
            return
        
        product_id = sales_data[0].product_id
        for sale in sales_data:
            sale.product_id = product_id
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(sales_data)
        
        try:
            # Property: Seasonal risk impact assessment should work for any valid inputs
            analysis = await risk_assessor.assess_seasonal_risk_impact(product_id, inventory_level)
            
            # Property: Analysis should be a comprehensive dictionary
            assert isinstance(analysis, dict)
            
            # Property: Analysis should contain all required sections
            required_sections = [
                'product_id', 'current_inventory', 'assessment_date', 'upcoming_events',
                'base_assessments', 'seasonal_adjusted_assessments', 'seasonal_impact_metrics',
                'seasonal_recommendations', 'risk_summary'
            ]
            for section in required_sections:
                assert section in analysis
            
            # Property: Product ID and inventory should match inputs
            assert analysis['product_id'] == product_id
            assert analysis['current_inventory'] == inventory_level
            
            # Property: Assessment date should be today
            assert analysis['assessment_date'] == date.today().isoformat()
            
            # Property: Upcoming events should be a list
            assert isinstance(analysis['upcoming_events'], list)
            
            # Property: Base assessments should contain both overstock and understock
            base_assessments = analysis['base_assessments']
            assert isinstance(base_assessments, dict)
            assert 'overstock' in base_assessments
            assert 'understock' in base_assessments
            assert isinstance(base_assessments['overstock'], RiskAssessment)
            assert isinstance(base_assessments['understock'], RiskAssessment)
            
            # Property: Seasonal adjusted assessments should have same structure
            seasonal_assessments = analysis['seasonal_adjusted_assessments']
            assert isinstance(seasonal_assessments, dict)
            assert 'overstock' in seasonal_assessments
            assert 'understock' in seasonal_assessments
            assert isinstance(seasonal_assessments['overstock'], RiskAssessment)
            assert isinstance(seasonal_assessments['understock'], RiskAssessment)
            
            # Property: Seasonal impact metrics should contain required metrics
            metrics = analysis['seasonal_impact_metrics']
            assert isinstance(metrics, dict)
            required_metrics = [
                'max_demand_multiplier', 'min_demand_multiplier', 'average_demand_multiplier',
                'seasonal_volatility', 'high_impact_events', 'low_impact_events', 'neutral_events'
            ]
            for metric in required_metrics:
                assert metric in metrics
            
            # Property: Demand multipliers should be reasonable
            assert 0.1 <= metrics['max_demand_multiplier'] <= 5.0
            assert 0.1 <= metrics['min_demand_multiplier'] <= 5.0
            assert metrics['min_demand_multiplier'] <= metrics['max_demand_multiplier']
            
            # Property: Seasonal volatility should be non-negative
            assert metrics['seasonal_volatility'] >= 0.0
            
            # Property: Event categorizations should be lists
            assert isinstance(metrics['high_impact_events'], list)
            assert isinstance(metrics['low_impact_events'], list)
            assert isinstance(metrics['neutral_events'], list)
            
            # Property: Seasonal recommendations should be a list of strings
            recommendations = analysis['seasonal_recommendations']
            assert isinstance(recommendations, list)
            for recommendation in recommendations:
                assert isinstance(recommendation, str)
                assert len(recommendation) > 0
            
            # Property: Risk summary should contain required fields
            risk_summary = analysis['risk_summary']
            assert isinstance(risk_summary, dict)
            assert 'highest_risk_type' in risk_summary
            assert 'seasonal_adjustment_significant' in risk_summary
            assert 'early_warning_required' in risk_summary
            
            # Property: Highest risk type should be valid
            assert risk_summary['highest_risk_type'] in ['overstock', 'understock']
            
            # Property: Boolean flags should be boolean
            assert isinstance(risk_summary['seasonal_adjustment_significant'], bool)
            assert isinstance(risk_summary['early_warning_required'], bool)
            
        except (RiskCalculationError, InsufficientDataError):
            # Property: Insufficient data should be handled gracefully
            pass
    
    @given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_risk_level_boundary_consistency(self, inventory1, inventory2):
        """
        **Property 4.5: Risk Level Boundary Consistency**
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property: For any two inventory levels, risk assessments should show consistent
        ordering relationships (higher inventory = higher overstock risk, lower understock risk).
        """
        # Create consistent sales data for testing
        sales_data = []
        base_date = date.today() - timedelta(days=60)
        
        for i in range(20):  # Sufficient data for risk assessment
            sales_data.append(SalesDataPoint(
                product_id="BOUNDARY_TEST_PRODUCT",
                product_name="Boundary Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=50 + (i % 10),  # Some variation
                sale_date=base_date + timedelta(days=i*3),
                store_location="TEST_STORE"
            ))
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(sales_data)
        
        try:
            # Property: Risk assessments should work for any inventory levels
            assessment1_over = await risk_assessor.assess_overstock_risk("BOUNDARY_TEST_PRODUCT", inventory1)
            assessment1_under = await risk_assessor.assess_understock_risk("BOUNDARY_TEST_PRODUCT", inventory1)
            
            assessment2_over = await risk_assessor.assess_overstock_risk("BOUNDARY_TEST_PRODUCT", inventory2)
            assessment2_under = await risk_assessor.assess_understock_risk("BOUNDARY_TEST_PRODUCT", inventory2)
            
            # Property: All assessments should be valid
            for assessment in [assessment1_over, assessment1_under, assessment2_over, assessment2_under]:
                assert isinstance(assessment, RiskAssessment)
                assert 0.0 <= assessment.risk_score <= 1.0
                assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            # Property: Higher inventory should generally mean higher overstock risk
            if inventory1 > inventory2:
                assert assessment1_over.risk_score >= assessment2_over.risk_score
            elif inventory2 > inventory1:
                assert assessment2_over.risk_score >= assessment1_over.risk_score
            
            # Property: Lower inventory should generally mean higher understock risk
            if inventory1 < inventory2:
                assert assessment1_under.risk_score >= assessment2_under.risk_score
            elif inventory2 < inventory1:
                assert assessment2_under.risk_score >= assessment1_under.risk_score
            
            # Property: Zero inventory should result in high understock risk
            if inventory1 == 0:
                assert assessment1_under.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                assert assessment1_under.risk_score > 0.5
            if inventory2 == 0:
                assert assessment2_under.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                assert assessment2_under.risk_score > 0.5
            
        except (RiskCalculationError, InsufficientDataError):
            # Property: Insufficient data should be handled gracefully
            pass


# Edge case and error handling property tests
class TestRiskAssessorEdgeCaseProperties:
    """Property tests for edge cases and error handling in Risk Assessor."""
    
    @given(st.lists(sales_data_point_strategy(), min_size=0, max_size=2))
    @settings(max_examples=5, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_insufficient_data_handling(self, minimal_sales_data):
        """
        **Property 4.6: Insufficient Data Handling**
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property: For insufficient sales data, the risk assessor should handle
        the situation gracefully with appropriate error messages.
        """
        if minimal_sales_data:
            product_id = minimal_sales_data[0].product_id
            for sale in minimal_sales_data:
                sale.product_id = product_id
        else:
            product_id = "EMPTY_DATA_PRODUCT"
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(minimal_sales_data)
        
        # Property: Insufficient data should raise appropriate exceptions
        with pytest.raises((RiskCalculationError, InsufficientDataError)):
            await risk_assessor.assess_overstock_risk(product_id, 100)
        
        with pytest.raises((RiskCalculationError, InsufficientDataError)):
            await risk_assessor.assess_understock_risk(product_id, 100)
        
        if len(minimal_sales_data) < 5:
            with pytest.raises((RiskCalculationError, InsufficientDataError)):
                await risk_assessor.calculate_demand_volatility(product_id)
    
    @given(st.text(min_size=1, max_size=50, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"))
    @settings(max_examples=5, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_nonexistent_product_handling(self, nonexistent_product_id):
        """
        **Property 4.7: Nonexistent Product Handling**
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        Property: For nonexistent products, the risk assessor should handle
        the situation gracefully with appropriate error messages.
        """
        # Create some sales data for other products
        sales_data = []
        base_date = date.today() - timedelta(days=30)
        
        for i in range(10):
            sales_data.append(SalesDataPoint(
                product_id="EXISTING_PRODUCT",
                product_name="Existing Product",
                category="electronics",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=10,
                sale_date=base_date + timedelta(days=i*3),
                store_location="TEST_STORE"
            ))
        
        risk_assessor = RiskAssessor()
        risk_assessor.set_sales_data(sales_data)
        
        # Property: Nonexistent products should raise appropriate exceptions
        with pytest.raises((RiskCalculationError, InsufficientDataError)):
            await risk_assessor.assess_overstock_risk(nonexistent_product_id, 100)
        
        with pytest.raises((RiskCalculationError, InsufficientDataError)):
            await risk_assessor.assess_understock_risk(nonexistent_product_id, 100)
        
        with pytest.raises((RiskCalculationError, InsufficientDataError)):
            await risk_assessor.calculate_demand_volatility(nonexistent_product_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])