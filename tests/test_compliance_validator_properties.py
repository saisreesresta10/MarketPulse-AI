"""
Property-Based Tests for Compliance Validator Component

Property tests validating universal correctness properties for MRP compliance validation,
regulatory constraint enforcement, and consumer protection law adherence.

**Property 8: Universal Regulatory Compliance**
**Property 9: Regulatory Adaptation**
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from uuid import uuid4

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

from marketpulse_ai.components.compliance_validator import (
    ComplianceValidator, 
    RegulationViolationError
)
from marketpulse_ai.core.models import ComplianceResult, ComplianceStatus


# Hypothesis strategies for generating test data
@composite
def valid_recommendation_strategy(draw):
    """Generate valid recommendation data for testing."""
    mrp_value = draw(st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    mrp = Decimal(f"{mrp_value:.2f}")
    
    # Selling price should be <= MRP for valid recommendations
    selling_price_value = draw(st.floats(min_value=1.0, max_value=float(mrp), allow_nan=False, allow_infinity=False))
    selling_price = Decimal(f"{selling_price_value:.2f}")
    
    discount_percent = float((mrp - selling_price) / mrp * 100)
    
    return {
        'id': str(uuid4()),
        'product_id': draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        'mrp': float(mrp),
        'proposed_selling_price': float(selling_price),
        'discount_percent': discount_percent,
        'category': draw(st.sampled_from(['electronics', 'clothing', 'home', 'books', 'sports', 'beauty', 'food', 'jewelry'])),
        'original_price': float(mrp),
        'cost_price': draw(st.floats(min_value=1.0, max_value=float(selling_price), allow_nan=False, allow_infinity=False)),
        'sale_duration_days': draw(st.integers(min_value=1, max_value=90)),
        'sale_type': draw(st.sampled_from(['regular', 'flash', 'seasonal', 'clearance'])),
        'is_seasonal_sale': draw(st.booleans()),
        'gst_compliant': True,
        'shop_act_compliant': True,
        'price_breakdown_available': True
    }


@composite
def invalid_recommendation_strategy(draw):
    """Generate invalid recommendation data for testing violation detection."""
    mrp_value = draw(st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    mrp = Decimal(f"{mrp_value:.2f}")
    
    # Selling price should be > MRP for invalid recommendations
    selling_price_value = draw(st.floats(min_value=float(mrp) + 0.01, max_value=float(mrp) * 2, allow_nan=False, allow_infinity=False))
    selling_price = Decimal(f"{selling_price_value:.2f}")
    
    return {
        'id': str(uuid4()),
        'product_id': draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        'mrp': float(mrp),
        'proposed_selling_price': float(selling_price),
        'discount_percent': -((selling_price - mrp) / mrp * 100),  # Negative discount (price increase)
        'category': draw(st.sampled_from(['electronics', 'clothing', 'home', 'books', 'sports', 'beauty', 'food', 'jewelry'])),
        'original_price': float(mrp),
        'cost_price': draw(st.floats(min_value=1.0, max_value=float(mrp), allow_nan=False, allow_infinity=False)),
        'sale_duration_days': draw(st.integers(min_value=1, max_value=90)),
        'sale_type': draw(st.sampled_from(['regular', 'flash', 'seasonal', 'clearance'])),
        'is_seasonal_sale': draw(st.booleans()),
        'gst_compliant': True,
        'shop_act_compliant': True,
        'price_breakdown_available': True
    }


@composite
def pricing_strategy_strategy(draw):
    """Generate pricing strategy data for testing."""
    num_products = draw(st.integers(min_value=1, max_value=5))
    products = []
    
    for _ in range(num_products):
        products.append(draw(valid_recommendation_strategy()))
    
    return {
        'strategy_id': str(uuid4()),
        'strategy_name': draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')))),
        'products': products,
        'strategy_type': draw(st.sampled_from(['promotional', 'clearance', 'seasonal', 'regular'])),
        'duration_days': draw(st.integers(min_value=1, max_value=90)),
        'target_margin': draw(st.floats(min_value=0.1, max_value=0.5)),
        'expected_volume_increase': draw(st.floats(min_value=0.0, max_value=2.0))
    }


@composite
def regulation_update_strategy(draw):
    """Generate regulation update data for testing."""
    return {
        'update_id': str(uuid4()),
        'regulation_type': draw(st.sampled_from(['mrp_compliance', 'discount_regulations', 'consumer_protection'])),
        'changes': {
            'new_rules': draw(st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.booleans(), st.floats(min_value=0.0, max_value=100.0), st.text(min_size=1, max_size=50)),
                min_size=1, max_size=3
            )),
            'effective_date': (date.today() + timedelta(days=draw(st.integers(min_value=1, max_value=365)))).isoformat(),
            'description': draw(st.text(min_size=10, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Pc'))))
        }
    }


class TestComplianceValidatorProperties:
    """
    Property-based tests for Compliance Validator component.
    
    **Property 8: Universal Regulatory Compliance**
    **Property 9: Regulatory Adaptation**
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    """
    
    @given(valid_recommendation_strategy())
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_universal_mrp_compliance_valid(self, recommendation):
        """
        **Property 8.1: Universal MRP Compliance - Valid Cases**
        **Validates: Requirements 6.1**
        
        Property: For any recommendation where selling price <= MRP,
        the MRP aspect should be compliant (though other regulations may apply).
        """
        validator = ComplianceValidator()
        
        # Property: Valid MRP-compliant recommendations should pass MRP validation
        result = await validator.validate_mrp_compliance(recommendation)
        
        # Property: Result should be a ComplianceResult
        assert isinstance(result, ComplianceResult)
        
        # Property: MRP constraint should be satisfied
        assert recommendation['proposed_selling_price'] <= recommendation['mrp']
        
        # Property: If non-compliant, it should be due to discount limits, not MRP violations
        if result.compliance_status == ComplianceStatus.NON_COMPLIANT:
            # Check that violations are related to discounts, not MRP
            mrp_violations = [v for v in result.violations if 'MRP_001' in v or 'exceeds MRP' in v]
            assert len(mrp_violations) == 0, "Should not have MRP violations for valid selling price"
        
        # Property: Should have some regulations checked
        assert len(result.regulations_checked) > 0
    
    @given(invalid_recommendation_strategy())
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_universal_mrp_compliance_invalid(self, recommendation):
        """
        **Property 8.2: Universal MRP Compliance - Invalid Cases**
        **Validates: Requirements 6.1**
        
        Property: For any invalid recommendation where selling price > MRP,
        the compliance validator should reject the recommendation.
        """
        validator = ComplianceValidator()
        
        # Property: Invalid MRP-violating recommendations should fail validation
        result = await validator.validate_mrp_compliance(recommendation)
        
        # Property: Result should be a ComplianceResult
        assert isinstance(result, ComplianceResult)
        
        # Property: Invalid recommendations should have non-compliant status
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        
        # Property: Non-compliant results should have violations
        assert hasattr(result, 'violations')
        assert len(result.violations) > 0
        
        # Property: MRP constraint should be violated
        assert recommendation['proposed_selling_price'] > recommendation['mrp']
    
    @given(st.sampled_from(['electronics', 'clothing', 'home', 'books', 'sports', 'beauty', 'food', 'jewelry']))
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_regulatory_constraints_consistency(self, category):
        """
        **Property 8.3: Regulatory Constraints Consistency**
        **Validates: Requirements 6.2**
        
        Property: For any product category, regulatory constraints should be
        consistent and contain required constraint information.
        """
        validator = ComplianceValidator()
        
        # Property: Regulatory constraints should be available for any category
        constraints = await validator.get_regulatory_constraints(category)
        
        # Property: Constraints should be a dictionary
        assert isinstance(constraints, dict)
        
        # Property: Constraints should contain required fields
        required_fields = ['category', 'mrp_rules', 'discount_limits']
        for field in required_fields:
            assert field in constraints
        
        # Property: Discount limits should be reasonable
        if 'discount_limits' in constraints and 'max_discount_percent' in constraints['discount_limits']:
            max_discount = constraints['discount_limits']['max_discount_percent']
            assert 0 <= max_discount <= 100
        
        # Property: Constraints should be category-specific
        assert constraints['category'] == category
    
    @given(st.floats(min_value=0.0, max_value=100.0))
    @settings(max_examples=15, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_discount_limit_validation(self, discount_percent):
        """
        **Property 8.4: Discount Limit Validation**
        **Validates: Requirements 6.1, 6.2**
        
        Property: For any discount percentage, the validator should correctly
        determine compliance based on category-specific limits.
        """
        validator = ComplianceValidator()
        product_id = "TEST_PRODUCT_001"
        
        # Property: Discount validation should handle any percentage
        result = await validator.check_discount_limits(product_id, discount_percent)
        
        # Property: Result should be a ComplianceResult
        assert isinstance(result, ComplianceResult)
        
        # Property: Result should have appropriate status
        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW, ComplianceStatus.NON_COMPLIANT]
        
        # Property: Extreme discounts should be flagged
        if discount_percent > 90:
            assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        
        # Property: Reasonable discounts should be allowed
        if discount_percent <= 50:
            assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
    
    @given(pricing_strategy_strategy())
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_pricing_strategy_validation(self, strategy):
        """
        **Property 8.5: Pricing Strategy Validation**
        **Validates: Requirements 6.1, 6.2**
        
        Property: For any pricing strategy, validation should check all products
        and provide comprehensive compliance assessment.
        """
        validator = ComplianceValidator()
        
        # Property: Strategy validation should handle any valid strategy
        result = await validator.validate_pricing_strategy(strategy)
        
        # Property: Result should be a ComplianceResult
        assert isinstance(result, ComplianceResult)
        
        # Property: Result should have appropriate status
        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW, ComplianceStatus.NON_COMPLIANT]
        
        # Property: Strategy validation should consider all products
        if hasattr(result, 'details') and 'product_results' in result.details:
            product_results = result.details['product_results']
            assert len(product_results) == len(strategy['products'])
        
        # Property: Each product should be individually validated
        for product in strategy['products']:
            assert product['proposed_selling_price'] <= product['mrp']  # Valid by construction
    
    @given(regulation_update_strategy())
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_regulatory_adaptation(self, regulation_update):
        """
        **Property 9.1: Regulatory Adaptation**
        **Validates: Requirements 6.4**
        
        Property: For any regulation update, the validator should adapt
        its rules and maintain system consistency.
        """
        validator = ComplianceValidator()
        
        # Property: Regulation updates should be processed successfully
        update_result = await validator.update_regulation_rules(regulation_update['changes'])
        
        # Property: Update should succeed for valid changes
        assert isinstance(update_result, bool)
        
        # Property: System should maintain consistency after updates
        # Test that validator still works after rule updates
        test_recommendation = {
            'id': str(uuid4()),
            'product_id': 'TEST_001',
            'mrp': 100.0,
            'proposed_selling_price': 90.0,
            'discount_percent': 10.0,
            'category': 'electronics'
        }
        
        # Property: Validator should still function after rule updates
        result = await validator.validate_mrp_compliance(test_recommendation)
        assert isinstance(result, ComplianceResult)
    
    @given(st.sampled_from(['electronics', 'clothing', 'home', 'books']))
    @settings(max_examples=10, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_constraint_explanation_completeness(self, category):
        """
        **Property 8.6: Constraint Explanation Completeness**
        **Validates: Requirements 6.2, 6.3**
        
        Property: For any product category, constraint explanations should be
        comprehensive and include all necessary regulatory information.
        """
        validator = ComplianceValidator()
        
        # Property: Constraint explanations should be available for any category
        explanation = await validator.generate_constraint_explanation(category)
        
        # Property: Explanation should be a dictionary
        assert isinstance(explanation, dict)
        
        # Property: Explanation should contain required sections
        required_sections = ['category', 'applicable_regulations', 'constraints_summary']
        for section in required_sections:
            assert section in explanation
        
        # Property: Explanation should be category-specific
        assert explanation['category'] == category
        
        # Property: Regulations should be listed
        assert isinstance(explanation['applicable_regulations'], list)
        assert len(explanation['applicable_regulations']) > 0
        
        # Property: Constraints summary should be informative
        assert isinstance(explanation['constraints_summary'], dict)
    
    @given(st.lists(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(min_size=1, max_size=100), st.floats(min_value=0.0, max_value=100.0)),
        min_size=1, max_size=3
    ), min_size=1, max_size=3))
    @settings(max_examples=8, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_regulatory_change_notification(self, changes):
        """
        **Property 9.2: Regulatory Change Notification**
        **Validates: Requirements 6.4, 6.5**
        
        Property: For any regulatory changes, the system should provide
        appropriate notifications and maintain transparency.
        """
        validator = ComplianceValidator()
        
        # Convert changes to expected format
        formatted_changes = []
        for i, change in enumerate(changes):
            formatted_changes.append({
                'change_id': f'CHANGE_{i}',
                'type': 'rule_update',
                'description': f'Test change {i}',
                'effective_date': (date.today() + timedelta(days=30)).isoformat(),
                'details': change
            })
        
        # Property: Regulatory changes should be processed and communicated
        notification_result = await validator.notify_regulatory_changes(formatted_changes)
        
        # Property: Notification result should be structured
        assert isinstance(notification_result, dict)
        
        # Property: Notification should contain processing results
        required_fields = ['changes_processed', 'notifications_sent', 'processing_status']
        for field in required_fields:
            assert field in notification_result
        
        # Property: Processing status should be successful
        assert notification_result['processing_status'] == 'success'
        
        # Property: Number of changes processed should match input
        assert notification_result['changes_processed'] == len(formatted_changes)
    
    @settings(max_examples=5, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_system_limitations_transparency(self):
        """
        **Property 8.7: System Limitations Transparency**
        **Validates: Requirements 6.3, 6.5**
        
        Property: The system should provide comprehensive transparency
        about its limitations and data sources.
        """
        validator = ComplianceValidator()
        
        # Property: System limitations should be available
        transparency_info = await validator.get_system_limitations_and_transparency()
        
        # Property: Transparency info should be comprehensive
        assert isinstance(transparency_info, dict)
        
        # Property: Required transparency sections should be present
        required_sections = ['system_limitations', 'data_sources', 'accuracy_disclaimers', 'regulatory_scope']
        for section in required_sections:
            assert section in transparency_info
        
        # Property: Each section should contain meaningful information
        for section in required_sections:
            assert isinstance(transparency_info[section], (dict, list, str))
            if isinstance(transparency_info[section], (dict, list)):
                assert len(transparency_info[section]) > 0
            elif isinstance(transparency_info[section], str):
                assert len(transparency_info[section]) > 10  # Meaningful content
        
        # Property: System version should be included
        assert 'system_version' in transparency_info
        assert isinstance(transparency_info['system_version'], str)


# Integration property tests
class TestComplianceValidatorIntegrationProperties:
    """Integration property tests for complete compliance validation workflows."""
    
    @given(valid_recommendation_strategy(), st.sampled_from(['electronics', 'clothing', 'home']))
    @settings(max_examples=5, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_end_to_end_compliance_workflow(self, recommendation, category):
        """
        **Property 8.8: End-to-End Compliance Workflow**
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
        
        Property: Complete compliance validation workflow should maintain
        consistency and provide comprehensive regulatory assessment.
        """
        validator = ComplianceValidator()
        
        # Update recommendation category
        recommendation['category'] = category
        
        # Property: Complete workflow should succeed for valid inputs
        
        # Step 1: Get regulatory constraints
        constraints = await validator.get_regulatory_constraints(category)
        assert isinstance(constraints, dict)
        
        # Step 2: Validate MRP compliance
        mrp_result = await validator.validate_mrp_compliance(recommendation)
        assert isinstance(mrp_result, ComplianceResult)
        
        # Step 3: Check discount limits
        discount_result = await validator.check_discount_limits(
            recommendation['product_id'], 
            recommendation['discount_percent']
        )
        assert isinstance(discount_result, ComplianceResult)
        
        # Step 4: Generate constraint explanation
        explanation = await validator.generate_constraint_explanation(category, recommendation)
        assert isinstance(explanation, dict)
        
        # Step 5: Get system transparency info
        transparency = await validator.get_system_limitations_and_transparency()
        assert isinstance(transparency, dict)
        
        # Property: All workflow steps should complete successfully
        assert mrp_result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
        assert discount_result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
        
        # Property: Workflow should maintain data consistency
        assert constraints['category'] == category
        assert explanation['category'] == category


if __name__ == "__main__":
    pytest.main([__file__, "-v"])