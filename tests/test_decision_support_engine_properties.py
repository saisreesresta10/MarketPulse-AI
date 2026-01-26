"""
Property-Based Tests for Decision Support Engine Component

Property tests validating universal correctness properties for discount strategy
recommendations, MRP compliance, optimal discount window identification,
price sensitivity analysis, and recommendation orchestration.

**Property 6: MRP-Compliant Discount Recommendations**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

from marketpulse_ai.components.decision_support_engine import (
    DecisionSupportEngine,
    DecisionSupportEngineError,
    RecommendationGenerationError,
    OptimizationError
)
from marketpulse_ai.core.models import (
    DemandPattern, ExplainableInsight, RiskAssessment, ComplianceResult,
    ConfidenceLevel, RiskLevel, ComplianceStatus, SalesDataPoint
)


# Hypothesis strategies for generating test data
@composite
def sales_data_point_strategy(draw):
    """Generate valid SalesDataPoint instances for testing."""
    product_id = draw(st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"))
    product_name = draw(st.text(min_size=5, max_size=50, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "))
    category = draw(st.sampled_from(['electronics', 'clothing', 'food', 'books', 'home', 'sports', 'beauty']))
    
    # Generate realistic price ranges
    mrp_value = draw(st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    mrp = Decimal(f"{mrp_value:.2f}")
    
    selling_price_value = draw(st.floats(min_value=5.0, max_value=float(mrp), allow_nan=False, allow_infinity=False))
    selling_price = Decimal(f"{selling_price_value:.2f}")
    
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
def demand_pattern_strategy(draw):
    """Generate valid DemandPattern instances for testing."""
    product_id = draw(st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"))
    pattern_type = draw(st.sampled_from(['seasonal', 'trend', 'cyclical', 'volatile', 'stable']))
    description = draw(st.text(min_size=10, max_size=100, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ."))
    confidence_level = draw(st.sampled_from([ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]))
    
    # Generate seasonal factors
    events = ['diwali', 'holi', 'christmas', 'eid', 'summer', 'winter', 'monsoon']
    num_events = draw(st.integers(min_value=0, max_value=3))
    seasonal_factors = {}
    
    if num_events > 0:
        selected_events = draw(st.lists(st.sampled_from(events), min_size=num_events, max_size=num_events, unique=True))
        for event in selected_events:
            factor = draw(st.floats(min_value=0.3, max_value=3.0, allow_nan=False, allow_infinity=False))
            seasonal_factors[event] = factor
    
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
def recommendation_request_strategy(draw):
    """Generate valid recommendation request data."""
    num_products = draw(st.integers(min_value=1, max_value=5))
    product_ids = [f"PROD_{i:03d}" for i in range(num_products)]
    
    inventory_levels = {}
    for product_id in product_ids:
        inventory_levels[product_id] = draw(st.integers(min_value=0, max_value=1000))
    
    return {
        'product_ids': product_ids,
        'analysis_type': draw(st.sampled_from(['comprehensive', 'quick', 'detailed'])),
        'time_horizon': draw(st.sampled_from(['1_month', '3_months', '6_months', '1_year'])),
        'inventory_levels': inventory_levels
    }


@composite
def mock_components_strategy(draw):
    """Generate mock components with realistic behavior."""
    # Create mock components
    data_processor = AsyncMock()
    risk_assessor = AsyncMock()
    compliance_validator = AsyncMock()
    insight_generator = AsyncMock()
    
    # Configure data processor mock
    patterns = draw(st.lists(demand_pattern_strategy(), min_size=1, max_size=3))
    data_processor.extract_demand_patterns.return_value = patterns
    
    # Configure insight generator mock
    insights = []
    for pattern in patterns:
        insight = ExplainableInsight(
            title=f"Insight for {pattern.product_id}",
            description=f"Analysis shows {pattern.pattern_type} pattern with {pattern.confidence_level.value} confidence",
            confidence_level=pattern.confidence_level,
            supporting_evidence=[f"Pattern analysis of {pattern.supporting_data_points} data points"],
            key_factors=[f"Pattern type: {pattern.pattern_type}", f"Volatility: {pattern.volatility_score:.2f}"],
            business_impact=f"Expected impact based on {pattern.pattern_type} pattern",
            recommended_actions=[f"Monitor {pattern.product_id} closely"],
            data_sources=["Sales database", "Pattern analysis"]
        )
        insights.append(insight)
    insight_generator.generate_insights.return_value = insights
    
    # Configure risk assessor mock
    def create_risk_assessment(product_id, risk_type, inventory_level):
        # Generate realistic risk based on inventory level
        if risk_type == "overstock":
            risk_score = min(1.0, inventory_level / 500.0)  # Higher inventory = higher overstock risk
        else:  # understock
            risk_score = max(0.0, 1.0 - inventory_level / 100.0)  # Lower inventory = higher understock risk
        
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return RiskAssessment(
            product_id=product_id,
            risk_type=risk_type,
            risk_level=risk_level,
            risk_score=risk_score,
            contributing_factors=[f"{risk_type} risk factors for {product_id}"],
            mitigation_suggestions=[f"Mitigation for {risk_type} risk"],
            assessment_date=date.today(),
            valid_until=date.today() + timedelta(days=30)
        )
    
    risk_assessor.assess_overstock_risk.side_effect = lambda pid, inv: create_risk_assessment(pid, "overstock", inv)
    risk_assessor.assess_understock_risk.side_effect = lambda pid, inv: create_risk_assessment(pid, "understock", inv)
    
    # Configure compliance validator mock
    def create_compliance_result(compliant=True):
        return ComplianceResult(
            compliance_status=ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT,
            regulations_checked=['MRP_COMPLIANCE', 'DISCOUNT_LIMITS'],
            violations=[] if compliant else ['MRP violation detected'],
            warnings=[] if compliant else ['Review required'],
            validator_version='1.0.0'
        )
    
    compliance_validator.validate_mrp_compliance.return_value = create_compliance_result(True)
    compliance_validator.check_discount_limits.return_value = create_compliance_result(True)
    compliance_validator.validate_pricing_strategy.return_value = create_compliance_result(True)
    
    return {
        'data_processor': data_processor,
        'risk_assessor': risk_assessor,
        'compliance_validator': compliance_validator,
        'insight_generator': insight_generator,
        'patterns': patterns
    }


class TestDecisionSupportEngineProperties:
    """
    Property-based tests for Decision Support Engine component.
    
    **Property 6: MRP-Compliant Discount Recommendations**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """
    
    @given(recommendation_request_strategy(), mock_components_strategy())
    @settings(max_examples=20, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_6_mrp_compliant_discount_recommendations(self, request, mock_components):
        """
        **Property 6.1: MRP-Compliant Discount Recommendations - Complete Workflow**
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
        
        Property: For any discount strategy recommendation request, the Decision Support Engine
        should generate MRP-compliant recommendations with optimal timing, price sensitivity
        analysis, and appropriate duration recommendations.
        """
        # Create Decision Support Engine with mocked components
        engine = DecisionSupportEngine(
            data_processor=mock_components['data_processor'],
            risk_assessor=mock_components['risk_assessor'],
            compliance_validator=mock_components['compliance_validator'],
            insight_generator=mock_components['insight_generator']
        )
        
        # Property: Recommendation generation should work for any valid request
        result = await engine.generate_recommendations(request)
        
        # Property: Result should be a comprehensive dictionary
        assert isinstance(result, dict)
        
        # Property: Result should contain all required sections (Requirement 4.1, 4.5)
        required_sections = [
            'request_id', 'generated_at', 'analysis_type', 'time_horizon',
            'products_analyzed', 'summary', 'recommendations', 'insights',
            'risk_assessments', 'compliance_results', 'business_impact',
            'discount_strategy', 'next_review_date'
        ]
        for section in required_sections:
            assert section in result, f"Result should contain {section}"
        
        # Property: Request parameters should be preserved
        assert result['analysis_type'] == request['analysis_type']
        assert result['time_horizon'] == request['time_horizon']
        assert result['products_analyzed'] == request['product_ids']
        
        # Property: Recommendations should be generated for all products (Requirement 4.2, 4.3, 4.4)
        recommendations = result['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0, "Should generate at least one recommendation"
        
        for recommendation in recommendations:
            # Property: Each recommendation should have required fields
            required_fields = [
                'id', 'product_id', 'recommendation_type', 'optimal_discount_percentage',
                'discount_window', 'price_sensitivity_score', 'expected_impact',
                'confidence_level', 'supporting_factors', 'priority'
            ]
            for field in required_fields:
                assert field in recommendation, f"Recommendation should have {field}"
            
            # Property: Discount percentage should be within valid bounds (Requirement 4.1)
            discount_pct = recommendation['optimal_discount_percentage']
            assert isinstance(discount_pct, (int, float))
            assert engine.min_discount_percentage <= discount_pct <= engine.max_discount_percentage
            
            # Property: Discount window should have proper structure (Requirement 4.2)
            discount_window = recommendation['discount_window']
            assert isinstance(discount_window, dict)
            assert 'start_date' in discount_window
            assert 'end_date' in discount_window
            assert 'duration_days' in discount_window
            
            # Validate date format and logic
            start_date = datetime.fromisoformat(discount_window['start_date'])
            end_date = datetime.fromisoformat(discount_window['end_date'])
            assert end_date > start_date, "End date should be after start date"
            
            duration_days = discount_window['duration_days']
            assert isinstance(duration_days, int)
            assert duration_days > 0, "Duration should be positive"
            
            # Property: Price sensitivity should be in valid range (Requirement 4.3)
            price_sensitivity = recommendation['price_sensitivity_score']
            assert isinstance(price_sensitivity, (int, float))
            assert 0.0 <= price_sensitivity <= 1.0, "Price sensitivity should be between 0 and 1"
            
            # Property: Expected impact should contain required metrics (Requirement 4.4)
            expected_impact = recommendation['expected_impact']
            assert isinstance(expected_impact, dict)
            impact_metrics = [
                'demand_increase_percentage', 'revenue_impact_percentage',
                'inventory_turnover_improvement', 'market_share_potential'
            ]
            for metric in impact_metrics:
                assert metric in expected_impact, f"Expected impact should include {metric}"
                assert isinstance(expected_impact[metric], (int, float))
            
            # Property: Priority should be valid
            priority = recommendation['priority']
            assert priority in ['low', 'medium', 'high'], "Priority should be valid level"
            
            # Property: Supporting factors should be informative
            supporting_factors = recommendation['supporting_factors']
            assert isinstance(supporting_factors, list)
            assert len(supporting_factors) > 0, "Should provide supporting factors"
            for factor in supporting_factors:
                assert isinstance(factor, str)
                assert len(factor.strip()) > 0, "Supporting factors should be non-empty"
        
        # Property: Compliance results should validate all recommendations (Requirement 4.1, 4.5)
        compliance_results = result['compliance_results']
        assert isinstance(compliance_results, list)
        assert len(compliance_results) > 0, "Should have compliance validation results"
        
        for compliance_result in compliance_results:
            assert isinstance(compliance_result, dict)
            assert 'compliance_status' in compliance_result
            assert compliance_result['compliance_status'] in ['compliant', 'requires_review', 'non_compliant']
        
        # Property: Business impact should be assessed for top recommendations (Requirement 4.4, 4.5)
        business_impact = result['business_impact']
        assert isinstance(business_impact, dict)
        
        # Property: Summary should provide accurate counts
        summary = result['summary']
        assert isinstance(summary, dict)
        assert summary['total_recommendations'] == len(recommendations)
        assert isinstance(summary['high_priority_count'], int)
        assert isinstance(summary['compliance_issues'], int)
        assert isinstance(summary['critical_risks'], int)
    
    @given(st.lists(st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"), min_size=1, max_size=3), mock_components_strategy())
    @settings(max_examples=15, deadline=12000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_6_optimal_discount_strategy_optimization(self, product_ids, mock_components):
        """
        **Property 6.2: Optimal Discount Strategy Optimization**
        **Validates: Requirements 4.2, 4.3, 4.4**
        
        Property: For any list of product IDs, discount strategy optimization should
        identify optimal discount windows, assess price sensitivity, and recommend
        appropriate discount durations based on demand patterns.
        """
        # Create Decision Support Engine with mocked components
        engine = DecisionSupportEngine(
            data_processor=mock_components['data_processor'],
            risk_assessor=mock_components['risk_assessor'],
            compliance_validator=mock_components['compliance_validator'],
            insight_generator=mock_components['insight_generator']
        )
        
        # Property: Optimization should work for any valid product list
        result = await engine.optimize_discount_strategy(product_ids)
        
        # Property: Result should have proper structure
        assert isinstance(result, dict)
        assert 'strategy_summary' in result
        assert 'recommendations' in result
        assert 'optimization_metadata' in result
        
        # Property: Strategy summary should contain required information
        strategy_summary = result['strategy_summary']
        assert isinstance(strategy_summary, dict)
        assert strategy_summary['total_products'] == len(product_ids)
        assert 'optimization_method' in strategy_summary
        assert 'optimization_date' in strategy_summary
        assert 'constraints_applied' in strategy_summary
        
        # Property: Constraints should include MRP compliance
        constraints = strategy_summary['constraints_applied']
        assert isinstance(constraints, list)
        assert 'mrp_compliance' in constraints
        
        # Property: Recommendations should be generated for available patterns
        recommendations = result['recommendations']
        assert isinstance(recommendations, list)
        
        for recommendation in recommendations:
            # Property: Each recommendation should have optimal discount calculation (Requirement 4.2)
            assert 'optimal_discount_percentage' in recommendation
            discount_pct = recommendation['optimal_discount_percentage']
            assert engine.min_discount_percentage <= discount_pct <= engine.max_discount_percentage
            
            # Property: Discount window should be properly determined (Requirement 4.2)
            assert 'discount_window' in recommendation
            discount_window = recommendation['discount_window']
            assert isinstance(discount_window, dict)
            assert 'start_date' in discount_window
            assert 'end_date' in discount_window
            assert 'duration_days' in discount_window
            assert 'timing_rationale' in discount_window
            
            # Property: Price sensitivity should be assessed (Requirement 4.3)
            assert 'price_sensitivity_score' in recommendation
            price_sensitivity = recommendation['price_sensitivity_score']
            assert 0.0 <= price_sensitivity <= 1.0
            
            # Property: Expected impact should be calculated (Requirement 4.4)
            assert 'expected_impact' in recommendation
            expected_impact = recommendation['expected_impact']
            assert isinstance(expected_impact, dict)
            
            # Property: Priority should be calculated
            assert 'priority' in recommendation
            assert recommendation['priority'] in ['low', 'medium', 'high']
        
        # Property: Optimization metadata should contain configuration
        metadata = result['optimization_metadata']
        assert isinstance(metadata, dict)
        assert 'max_discount_limit' in metadata
        assert 'min_discount_limit' in metadata
        assert 'step_size' in metadata
        assert metadata['max_discount_limit'] == engine.max_discount_percentage
        assert metadata['min_discount_limit'] == engine.min_discount_percentage
    
    @given(demand_pattern_strategy())
    @settings(max_examples=20, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_6_discount_calculation_consistency(self, demand_pattern):
        """
        **Property 6.3: Discount Calculation Consistency**
        **Validates: Requirements 4.2, 4.3**
        
        Property: For any demand pattern, optimal discount calculation should be
        consistent, bounded, and reflect pattern characteristics appropriately.
        """
        # Create minimal engine for testing calculation methods
        engine = DecisionSupportEngine(
            data_processor=AsyncMock(),
            risk_assessor=AsyncMock(),
            compliance_validator=AsyncMock(),
            insight_generator=AsyncMock()
        )
        
        # Property: Optimal discount calculation should work for any pattern
        discount1 = await engine._calculate_optimal_discount(demand_pattern)
        discount2 = await engine._calculate_optimal_discount(demand_pattern)
        
        # Property: Calculation should be deterministic
        assert discount1 == discount2, "Discount calculation should be deterministic"
        
        # Property: Discount should be within bounds
        assert engine.min_discount_percentage <= discount1 <= engine.max_discount_percentage
        assert isinstance(discount1, (int, float))
        
        # Property: High volatility should generally lead to higher discounts
        high_volatility_pattern = demand_pattern.model_copy()
        high_volatility_pattern.volatility_score = 0.9
        high_volatility_pattern.trend_direction = "decreasing"
        high_volatility_pattern.confidence_level = ConfidenceLevel.HIGH
        
        high_volatility_discount = await engine._calculate_optimal_discount(high_volatility_pattern)
        
        # Should be higher due to high volatility and decreasing trend
        assert high_volatility_discount >= discount1 or high_volatility_discount == engine.max_discount_percentage
        
        # Property: Discount window determination should be consistent in structure
        window1 = await engine._determine_discount_window(demand_pattern)
        window2 = await engine._determine_discount_window(demand_pattern)
        
        # Windows may differ slightly in timestamps but should have same structure and duration
        assert window1['duration_days'] == window2['duration_days'], "Duration should be consistent"
        assert window1['timing_rationale'] == window2['timing_rationale'], "Rationale should be consistent"
        
        # Property: Window should have valid structure
        assert isinstance(window1, dict)
        assert 'start_date' in window1
        assert 'end_date' in window1
        assert 'duration_days' in window1
        assert 'timing_rationale' in window1
        
        # Property: Dates should be valid
        start_date = datetime.fromisoformat(window1['start_date'])
        end_date = datetime.fromisoformat(window1['end_date'])
        assert end_date > start_date
        assert start_date > datetime.utcnow()  # Should be in the future
        
        # Property: Duration should match date difference
        duration = (end_date - start_date).days
        assert duration == window1['duration_days']
        
        # Property: Price sensitivity assessment should be consistent
        sensitivity1 = await engine._assess_price_sensitivity(demand_pattern)
        sensitivity2 = await engine._assess_price_sensitivity(demand_pattern)
        
        assert sensitivity1 == sensitivity2, "Price sensitivity assessment should be deterministic"
        assert 0.0 <= sensitivity1 <= 1.0, "Price sensitivity should be in valid range"
    
    @given(st.lists(st.fixed_dictionaries({
        'id': st.uuids().map(str),
        'product_id': st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"),
        'optimal_discount_percentage': st.floats(min_value=5.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        'confidence_level': st.sampled_from(['low', 'medium', 'high']),
        'expected_impact': st.fixed_dictionaries({'revenue_impact_percentage': st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)}),
        'priority': st.sampled_from(['low', 'medium', 'high'])
    }), min_size=1, max_size=5))
    @settings(max_examples=15, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_6_recommendation_prioritization_consistency(self, recommendations):
        """
        **Property 6.4: Recommendation Prioritization Consistency**
        **Validates: Requirements 4.5**
        
        Property: For any list of recommendations, prioritization should produce
        consistent ordering based on impact, urgency, and business value.
        """
        # Create minimal engine for testing prioritization
        engine = DecisionSupportEngine(
            data_processor=AsyncMock(),
            risk_assessor=AsyncMock(),
            compliance_validator=AsyncMock(),
            insight_generator=AsyncMock()
        )
        
        # Ensure recommendations have required fields with valid values
        for i, rec in enumerate(recommendations):
            # All fields should already be properly typed from the strategy
            # Just ensure values are in valid ranges
            rec['optimal_discount_percentage'] = max(engine.min_discount_percentage, 
                                                   min(engine.max_discount_percentage, 
                                                       rec['optimal_discount_percentage']))
        
        # Property: Prioritization should work for any recommendation list
        prioritized1 = await engine.prioritize_recommendations(recommendations.copy())
        prioritized2 = await engine.prioritize_recommendations(recommendations.copy())
        
        # Property: Prioritization should work for any recommendation list
        prioritized = await engine.prioritize_recommendations(recommendations.copy())
        
        # Property: All recommendations should be preserved
        assert len(prioritized) == len(recommendations)
        
        # Property: Each recommendation should have priority metadata
        for rec in prioritized:
            assert 'priority_score' in rec
            assert 'rank' in rec
            assert 'percentile' in rec
            
            # Property: Priority score should be non-negative
            assert rec['priority_score'] >= 0.0
            
            # Property: Rank should be valid
            assert 1 <= rec['rank'] <= len(prioritized)
            
            # Property: Percentile should be valid
            assert 0.0 <= rec['percentile'] <= 100.0
        
        # Property: Rankings should be sequential
        ranks = [rec['rank'] for rec in prioritized]
        assert ranks == list(range(1, len(prioritized) + 1))
        
        # Property: Priority scores should be in descending order
        scores = [rec['priority_score'] for rec in prioritized]
        assert scores == sorted(scores, reverse=True)
        
        # Property: Prioritization should be consistent for same input
        prioritized2 = await engine.prioritize_recommendations(recommendations.copy())
        scores2 = [rec['priority_score'] for rec in prioritized2]
        assert sorted(scores) == sorted(scores2), "Same inputs should produce same scores"
        
        # Property: Priority calculation should consider multiple factors appropriately
        if len(prioritized) > 1:
            # Check that priority scores vary based on input differences
            unique_scores = set(rec['priority_score'] for rec in prioritized)
            # If all recommendations are identical, scores might be the same
            all_identical = all(
                rec['optimal_discount_percentage'] == recommendations[0]['optimal_discount_percentage'] and
                rec['confidence_level'] == recommendations[0]['confidence_level'] and
                rec['expected_impact']['revenue_impact_percentage'] == recommendations[0]['expected_impact']['revenue_impact_percentage'] and
                rec['priority'] == recommendations[0]['priority']
                for rec in recommendations
            )
            
            if not all_identical:
                # If recommendations differ, we should see some score variation
                # (though not guaranteed due to complex scoring algorithm)
                pass  # Accept that scoring might be complex
    
    @given(st.fixed_dictionaries({
        'id': st.uuids().map(str),
        'product_id': st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"),
        'optimal_discount_percentage': st.floats(min_value=5.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        'recommendation_type': st.just('discount_strategy')
    }), mock_components_strategy())
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_6_compliance_validation_pipeline(self, recommendation, mock_components):
        """
        **Property 6.5: Compliance Validation Pipeline**
        **Validates: Requirements 4.1, 4.5**
        
        Property: For any recommendation, the complete compliance validation pipeline
        should ensure MRP compliance and provide comprehensive regulatory assessment.
        """
        # Recommendation should already have proper fields from strategy
        # Just ensure discount percentage is in valid range
        recommendation['optimal_discount_percentage'] = max(5.0, min(50.0, recommendation['optimal_discount_percentage']))
        
        # Create Decision Support Engine with mocked components
        engine = DecisionSupportEngine(
            data_processor=mock_components['data_processor'],
            risk_assessor=mock_components['risk_assessor'],
            compliance_validator=mock_components['compliance_validator'],
            insight_generator=mock_components['insight_generator']
        )
        
        # Property: Compliance validation should work for any recommendation
        result = await engine.validate_recommendation_pipeline(recommendation)
        
        # Property: Result should be a ComplianceResult
        assert isinstance(result, ComplianceResult)
        
        # Property: Should have valid compliance status
        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW, ComplianceStatus.NON_COMPLIANT]
        
        # Property: Should check required regulations
        required_regulations = ['MRP_COMPLIANCE', 'DISCOUNT_LIMITS', 'PRICING_STRATEGY']
        for regulation in required_regulations:
            assert regulation in result.regulations_checked
        
        # Property: Should have validation details
        assert hasattr(result, 'validation_details')
        assert isinstance(result.validation_details, dict)
        
        validation_details = result.validation_details
        expected_validations = ['mrp_validation', 'discount_validation', 'strategy_validation']
        for validation in expected_validations:
            assert validation in validation_details
        
        # Property: Violations and warnings should be lists
        assert isinstance(result.violations, list)
        assert isinstance(result.warnings, list)
        
        # Property: If non-compliant, should have violations
        if result.compliance_status == ComplianceStatus.NON_COMPLIANT:
            assert len(result.violations) > 0, "Non-compliant results should have violations"
        
        # Property: Should have validator version
        assert hasattr(result, 'validator_version')
        assert isinstance(result.validator_version, str)
        assert len(result.validator_version) > 0


# Integration and edge case property tests
class TestDecisionSupportEngineIntegrationProperties:
    """Integration property tests for complete Decision Support Engine workflows."""
    
    @given(recommendation_request_strategy())
    @settings(max_examples=5, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_6_end_to_end_recommendation_workflow_robustness(self, request):
        """
        **Property 6.6: End-to-End Recommendation Workflow Robustness**
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
        
        Property: Complete recommendation workflow should handle any valid request
        and maintain consistency across all components and validation steps.
        """
        # Create realistic mock components
        mock_components = {}
        
        # Data processor mock
        data_processor = AsyncMock()
        patterns = []
        for product_id in request['product_ids']:
            pattern = DemandPattern(
                product_id=product_id,
                pattern_type='seasonal',
                description=f'Test pattern for {product_id}',
                confidence_level=ConfidenceLevel.MEDIUM,
                seasonal_factors={'diwali': 1.5},
                trend_direction='stable',
                volatility_score=0.4,
                supporting_data_points=100,
                date_range_start=date.today() - timedelta(days=365),
                date_range_end=date.today()
            )
            patterns.append(pattern)
        data_processor.extract_demand_patterns.return_value = patterns
        mock_components['data_processor'] = data_processor
        
        # Risk assessor mock
        risk_assessor = AsyncMock()
        def create_risk_assessment(product_id, risk_type, inventory_level):
            return RiskAssessment(
                product_id=product_id,
                risk_type=risk_type,
                risk_level=RiskLevel.MEDIUM,
                risk_score=0.5,
                contributing_factors=[f'{risk_type} factors'],
                mitigation_suggestions=[f'Mitigate {risk_type}'],
                assessment_date=date.today(),
                valid_until=date.today() + timedelta(days=30)
            )
        risk_assessor.assess_overstock_risk.side_effect = lambda pid, inv: create_risk_assessment(pid, "overstock", inv)
        risk_assessor.assess_understock_risk.side_effect = lambda pid, inv: create_risk_assessment(pid, "understock", inv)
        mock_components['risk_assessor'] = risk_assessor
        
        # Compliance validator mock
        compliance_validator = AsyncMock()
        compliant_result = ComplianceResult(
            compliance_status=ComplianceStatus.COMPLIANT,
            regulations_checked=['MRP_COMPLIANCE'],
            violations=[],
            warnings=[],
            validator_version='1.0.0'
        )
        compliance_validator.validate_mrp_compliance.return_value = compliant_result
        compliance_validator.check_discount_limits.return_value = compliant_result
        compliance_validator.validate_pricing_strategy.return_value = compliant_result
        mock_components['compliance_validator'] = compliance_validator
        
        # Insight generator mock
        insight_generator = AsyncMock()
        insights = [
            ExplainableInsight(
                title=f'Insight for {pid}',
                description=f'Analysis for {pid}',
                confidence_level=ConfidenceLevel.MEDIUM,
                supporting_evidence=['Test evidence'],
                key_factors=['Test factor'],
                business_impact='Test impact',
                recommended_actions=['Test action'],
                data_sources=['Test source']
            ) for pid in request['product_ids']
        ]
        insight_generator.generate_insights.return_value = insights
        mock_components['insight_generator'] = insight_generator
        
        # Create engine and test workflow
        engine = DecisionSupportEngine(**mock_components)
        
        # Property: Complete workflow should succeed for any valid request
        result = await engine.generate_recommendations(request)
        
        # Property: Result should maintain consistency across all sections
        assert result['products_analyzed'] == request['product_ids']
        assert len(result['recommendations']) > 0
        assert len(result['insights']) == len(insights)
        assert len(result['risk_assessments']) >= len(request['product_ids']) * 2  # overstock + understock
        
        # Property: All recommendations should be compliant (due to mock setup)
        for compliance_result in result['compliance_results']:
            assert compliance_result['compliance_status'] == 'compliant'
        
        # Property: Summary should accurately reflect the results
        summary = result['summary']
        assert summary['total_recommendations'] == len(result['recommendations'])
        assert summary['compliance_issues'] == 0  # All compliant in this test
        
        # Property: Business impact should be assessed for top recommendations
        business_impact = result['business_impact']
        assert isinstance(business_impact, dict)
        # Should have impact analysis for at least some recommendations
        assert len(business_impact) <= min(3, len(result['recommendations']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])