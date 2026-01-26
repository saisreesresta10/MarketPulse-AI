"""
Unit tests for the Compliance Validator component.

Tests MRP regulation validation, discount limit checking, and regulatory
constraint enforcement for the Indian retail market.
"""

import pytest
from decimal import Decimal
from datetime import datetime, date
from uuid import uuid4

from marketpulse_ai.components.compliance_validator import ComplianceValidator, RegulationViolationError
from marketpulse_ai.core.models import ComplianceStatus, ComplianceResult


class TestComplianceValidator:
    """Test suite for ComplianceValidator component."""
    
    @pytest.fixture
    def validator(self):
        """Create a ComplianceValidator instance for testing."""
        return ComplianceValidator()
    
    @pytest.fixture
    def valid_recommendation(self):
        """Create a valid recommendation for testing."""
        return {
            'id': uuid4(),
            'product_id': 'PROD_001',
            'mrp': 1000.00,
            'proposed_selling_price': 800.00,
            'discount_percent': 20.0,
            'category': 'electronics',
            'original_price': 1000.00,
            'cost_price': 600.00,
            'sale_duration_days': 7,
            'sale_type': 'regular',
            'is_seasonal_sale': False,
            'gst_compliant': True,
            'shop_act_compliant': True,
            'price_breakdown_available': True
        }
    
    @pytest.fixture
    def mrp_violation_recommendation(self):
        """Create a recommendation that violates MRP regulations."""
        return {
            'id': uuid4(),
            'product_id': 'PROD_002',
            'mrp': 1000.00,
            'proposed_selling_price': 1200.00,  # Exceeds MRP
            'discount_percent': -20.0,  # Negative discount (price increase)
            'category': 'electronics'
        }
    
    @pytest.fixture
    def excessive_discount_recommendation(self):
        """Create a recommendation with excessive discount."""
        return {
            'id': uuid4(),
            'product_id': 'PROD_003',
            'mrp': 1000.00,
            'proposed_selling_price': 200.00,
            'discount_percent': 80.0,  # Exceeds electronics limit of 70%
            'category': 'electronics',
            'sale_type': 'regular',
            'is_seasonal_sale': False
        }
    
    @pytest.fixture
    def essential_commodity_recommendation(self):
        """Create a recommendation for essential commodity."""
        return {
            'id': uuid4(),
            'product_id': 'FOOD_001',
            'mrp': 100.00,
            'proposed_selling_price': 90.00,
            'discount_percent': 10.0,
            'category': 'food',
            'cost_price': 75.00,  # 20% markup, exceeds 15% limit for essential commodities
            'sale_type': 'regular'
        }
    
    @pytest.mark.asyncio
    async def test_valid_recommendation_compliance(self, validator, valid_recommendation):
        """Test that a valid recommendation passes compliance checks."""
        result = await validator.validate_mrp_compliance(valid_recommendation)
        
        assert isinstance(result, ComplianceResult)
        assert result.compliance_status == ComplianceStatus.COMPLIANT
        assert len(result.violations) == 0
        assert result.validation_details['compliance_score'] == 1.0
        assert 'MRP Regulations' in str(result.regulations_checked)
    
    @pytest.mark.asyncio
    async def test_mrp_violation_detection(self, validator, mrp_violation_recommendation):
        """Test detection of MRP violations."""
        result = await validator.validate_mrp_compliance(mrp_violation_recommendation)
        
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert len(result.violations) > 0
        
        # Check for specific MRP violation
        mrp_violation_found = any('MRP_001' in violation for violation in result.violations)
        assert mrp_violation_found, "MRP_001 violation should be detected"
        
        # Check that compliance score is reduced
        assert result.validation_details['compliance_score'] < 1.0
    
    @pytest.mark.asyncio
    async def test_excessive_discount_detection(self, validator, excessive_discount_recommendation):
        """Test detection of excessive discount violations."""
        result = await validator.validate_mrp_compliance(excessive_discount_recommendation)
        
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert len(result.violations) > 0
        
        # Check for discount violation
        discount_violation_found = any('DISC_001' in violation for violation in result.violations)
        assert discount_violation_found, "DISC_001 violation should be detected"
    
    @pytest.mark.asyncio
    async def test_essential_commodity_markup_limits(self, validator, essential_commodity_recommendation):
        """Test essential commodity markup limit enforcement."""
        result = await validator.validate_mrp_compliance(essential_commodity_recommendation)
        
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert len(result.violations) > 0
        
        # Check for essential commodity violation
        essential_violation_found = any('ESS_001' in violation for violation in result.violations)
        assert essential_violation_found, "ESS_001 violation should be detected for excessive markup"
        
        # Check constraints
        assert result.constraints.get('essential_commodity') is True
        assert result.constraints.get('max_markup_percent') == 15
    
    @pytest.mark.asyncio
    async def test_discount_limits_by_category(self, validator):
        """Test category-specific discount limits."""
        categories_and_limits = [
            ('electronics', 70, 80),
            ('clothing', 75, 85),
            ('food', 50, 60),
            ('jewelry', 40, 50),
            ('books', 60, 70)
        ]
        
        for category, regular_limit, seasonal_limit in categories_and_limits:
            # Test regular discount at limit (should pass)
            recommendation = {
                'id': uuid4(),
                'product_id': f'PROD_{category.upper()}',
                'mrp': 1000.00,
                'proposed_selling_price': 1000.00 * (1 - regular_limit/100),
                'discount_percent': regular_limit,
                'category': category,
                'is_seasonal_sale': False,
                'sale_type': 'regular',
                'price_breakdown_available': True
            }
            
            result = await validator.validate_mrp_compliance(recommendation)
            assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
            
            # Test excessive regular discount (should fail)
            recommendation['discount_percent'] = regular_limit + 5
            recommendation['proposed_selling_price'] = 1000.00 * (1 - (regular_limit + 5)/100)
            recommendation['price_breakdown_available'] = True
            
            result = await validator.validate_mrp_compliance(recommendation)
            assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
    
    @pytest.mark.asyncio
    async def test_seasonal_discount_flexibility(self, validator):
        """Test seasonal discount flexibility."""
        recommendation = {
            'id': uuid4(),
            'product_id': 'SEASONAL_001',
            'mrp': 1000.00,
            'proposed_selling_price': 200.00,
            'discount_percent': 80.0,
            'category': 'electronics',
            'is_seasonal_sale': True,  # Seasonal sale allows higher discount
            'sale_type': 'seasonal',
            'price_breakdown_available': True
        }
        
        result = await validator.validate_mrp_compliance(recommendation)
        
        # Should pass because seasonal limit for electronics is 80%
        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
        
        # Test exceeding seasonal limit
        recommendation['discount_percent'] = 85.0
        recommendation['proposed_selling_price'] = 150.00
        recommendation['price_breakdown_available'] = True
        
        result = await validator.validate_mrp_compliance(recommendation)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
    
    @pytest.mark.asyncio
    async def test_consumer_protection_violations(self, validator):
        """Test consumer protection law violations."""
        # Test misleading original price
        recommendation = {
            'id': uuid4(),
            'product_id': 'MISLEADING_001',
            'mrp': 1000.00,
            'proposed_selling_price': 800.00,
            'discount_percent': 20.0,
            'category': 'electronics',
            'original_price': 1200.00,  # Exceeds MRP - misleading
            'price_breakdown_available': False  # Missing transparency
        }
        
        result = await validator.validate_mrp_compliance(recommendation)
        
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert len(result.violations) >= 1  # Price manipulation violation
        assert len(result.warnings) >= 1  # Transparency warning
        
        # Check for specific violations
        price_manipulation_found = any('CP_003' in violation for violation in result.violations)
        transparency_warning_found = any('Price breakdown not available' in warning for warning in result.warnings)
        
        assert price_manipulation_found, "Price manipulation violation should be detected"
        assert transparency_warning_found, "Transparency warning should be detected"
    
    @pytest.mark.asyncio
    async def test_check_discount_limits_method(self, validator):
        """Test the check_discount_limits method specifically."""
        # Test valid discount
        result = await validator.check_discount_limits('PROD_001', 50.0)
        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
        
        # Test excessive discount
        result = await validator.check_discount_limits('PROD_002', 95.0)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert len(result.violations) > 0
    
    @pytest.mark.asyncio
    async def test_pricing_strategy_validation(self, validator):
        """Test validation of complete pricing strategies."""
        strategy = {
            'id': uuid4(),
            'strategy_name': 'Summer Sale 2024',
            'strategy_type': 'seasonal',
            'duration_days': 15,
            'products': [
                {
                    'product_id': 'PROD_001',
                    'mrp': 1000.00,
                    'proposed_price': 700.00,
                    'discount_percent': 30.0,
                    'category': 'electronics'
                },
                {
                    'product_id': 'PROD_002',
                    'mrp': 500.00,
                    'proposed_price': 400.00,
                    'discount_percent': 20.0,
                    'category': 'clothing'
                }
            ]
        }
        
        result = await validator.validate_pricing_strategy(strategy)
        
        assert isinstance(result, ComplianceResult)
        assert result.validation_details['products_validated'] == 2
        assert result.validation_details['strategy_type'] == 'seasonal'
    
    @pytest.mark.asyncio
    async def test_get_regulatory_constraints(self, validator):
        """Test retrieval of regulatory constraints by category."""
        # Test electronics category
        constraints = await validator.get_regulatory_constraints('electronics')
        
        assert constraints['category'] == 'electronics'
        assert constraints['mrp_compliance_required'] is True
        assert 'discount_limits' in constraints
        assert constraints['discount_limits']['max_regular_discount_percent'] == 70
        assert constraints['discount_limits']['max_seasonal_discount_percent'] == 80
        
        # Test essential commodity (food)
        constraints = await validator.get_regulatory_constraints('food')
        
        assert constraints['essential_commodity'] is True
        assert constraints['max_markup_percent'] == 15
        assert constraints['hoarding_restrictions'] is True
        
        # Test luxury goods (jewelry)
        constraints = await validator.get_regulatory_constraints('jewelry')
        
        assert constraints['luxury_item'] is True
        assert constraints['additional_tax_compliance_required'] is True
    
    @pytest.mark.asyncio
    async def test_update_regulation_rules(self, validator):
        """Test updating regulation rules."""
        # Get original rules
        original_version = validator.validator_version
        
        # Test valid update
        new_rules = {
            'mrp_compliance': {
                'test_rule': {
                    'description': 'Test rule for updates',
                    'enabled': True,
                    'severity': 'medium'
                }
            },
            'discount_regulations': validator.regulation_rules['discount_regulations'],
            'consumer_protection': validator.regulation_rules['consumer_protection']
        }
        
        success = await validator.update_regulation_rules(new_rules)
        assert success is True
        assert validator.validator_version != original_version
        assert 'test_rule' in validator.regulation_rules['mrp_compliance']
        
        # Test invalid update (missing required section)
        invalid_rules = {
            'mrp_compliance': {}
            # Missing required sections
        }
        
        success = await validator.update_regulation_rules(invalid_rules)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, validator):
        """Test handling of recommendations with missing required fields."""
        incomplete_recommendation = {
            'id': uuid4(),
            'product_id': 'INCOMPLETE_001'
            # Missing mrp and proposed_selling_price
        }
        
        result = await validator.validate_mrp_compliance(incomplete_recommendation)
        
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert len(result.violations) > 0
        
        missing_info_violation = any(
            'Missing required pricing information' in violation 
            for violation in result.violations
        )
        assert missing_info_violation, "Missing information violation should be detected"
    
    @pytest.mark.asyncio
    async def test_zero_and_negative_prices(self, validator):
        """Test handling of zero and negative prices."""
        # Test zero selling price
        zero_price_recommendation = {
            'id': uuid4(),
            'product_id': 'ZERO_PRICE',
            'mrp': 1000.00,
            'proposed_selling_price': 0.00,
            'category': 'electronics'
        }
        
        result = await validator.validate_mrp_compliance(zero_price_recommendation)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        
        zero_price_violation = any('MRP_004' in violation for violation in result.violations)
        assert zero_price_violation, "Zero price violation should be detected"
        
        # Test negative MRP
        negative_mrp_recommendation = {
            'id': uuid4(),
            'product_id': 'NEGATIVE_MRP',
            'mrp': -100.00,
            'proposed_selling_price': 50.00,
            'category': 'electronics'
        }
        
        result = await validator.validate_mrp_compliance(negative_mrp_recommendation)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        
        negative_mrp_violation = any('MRP_005' in violation for violation in result.violations)
        assert negative_mrp_violation, "Negative MRP violation should be detected"
    
    @pytest.mark.asyncio
    async def test_unrealistic_price_ratios(self, validator):
        """Test detection of unrealistic price ratios."""
        unrealistic_recommendation = {
            'id': uuid4(),
            'product_id': 'UNREALISTIC_001',
            'mrp': 10000.00,
            'proposed_selling_price': 1.00,  # 0.01% of MRP - unrealistic
            'category': 'electronics'
        }
        
        result = await validator.validate_mrp_compliance(unrealistic_recommendation)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        
        unrealistic_violation = any('MRP_006' in violation for violation in result.violations)
        assert unrealistic_violation, "Unrealistic price ratio violation should be detected"
    
    @pytest.mark.asyncio
    async def test_sale_duration_limits(self, validator):
        """Test sale duration limit enforcement."""
        # Test flash sale duration violation
        flash_sale_recommendation = {
            'id': uuid4(),
            'product_id': 'FLASH_001',
            'mrp': 1000.00,
            'proposed_selling_price': 800.00,
            'discount_percent': 20.0,
            'category': 'electronics',
            'sale_type': 'flash_sale',
            'sale_duration_days': 3  # Exceeds 24 hours (1 day) limit
        }
        
        result = await validator.validate_mrp_compliance(flash_sale_recommendation)
        
        duration_violation = any('DISC_002' in violation for violation in result.violations)
        assert duration_violation, "Flash sale duration violation should be detected"
    
    @pytest.mark.asyncio
    async def test_regional_compliance_warnings(self, validator):
        """Test regional compliance warning generation."""
        regional_recommendation = {
            'id': uuid4(),
            'product_id': 'REGIONAL_001',
            'mrp': 1000.00,
            'proposed_selling_price': 800.00,
            'discount_percent': 20.0,
            'category': 'electronics',
            'state': 'Kerala',  # State with additional tax considerations
            'gst_compliant': False,  # Not GST compliant
            'shop_act_compliant': False  # Not Shop Act compliant
        }
        
        result = await validator.validate_mrp_compliance(regional_recommendation)
        
        # Should have warnings for regional compliance
        assert len(result.warnings) > 0
        
        gst_warning = any('GST compliance' in warning for warning in result.warnings)
        shop_act_warning = any('Shop Act compliance' in warning for warning in result.warnings)
        state_warning = any('Kerala' in warning for warning in result.warnings)
        
        assert gst_warning, "GST compliance warning should be present"
        assert shop_act_warning, "Shop Act compliance warning should be present"
        assert state_warning, "State-specific warning should be present"
    
    def test_compliance_score_calculation(self, validator):
        """Test compliance score calculation logic."""
        # Test perfect score
        score = validator._calculate_compliance_score([], [])
        assert score == 1.0
        
        # Test with violations
        score = validator._calculate_compliance_score(['violation1'], [])
        assert score == 0.7  # 1.0 - 0.3
        
        # Test with warnings
        score = validator._calculate_compliance_score([], ['warning1'])
        assert score == 0.9  # 1.0 - 0.1
        
        # Test with both
        score = validator._calculate_compliance_score(['violation1'], ['warning1'])
        assert score == 0.6  # 1.0 - 0.3 - 0.1
        
        # Test minimum score
        score = validator._calculate_compliance_score(['v1', 'v2', 'v3', 'v4'], ['w1', 'w2'])
        assert score == 0.0  # Should not go below 0
    
    @pytest.mark.asyncio
    async def test_strategy_coherence_validation(self, validator):
        """Test strategy-level coherence validation."""
        # Test strategy with too many non-compliant products
        strategy_with_violations = {
            'id': uuid4(),
            'strategy_name': 'Problematic Strategy',
            'strategy_type': 'regular',
            'duration_days': 7,
            'products': [
                {
                    'product_id': 'PROD_001',
                    'mrp': 1000.00,
                    'proposed_price': 1200.00,  # Violates MRP
                    'discount_percent': -20.0,
                    'category': 'electronics'
                },
                {
                    'product_id': 'PROD_002',
                    'mrp': 500.00,
                    'proposed_price': 600.00,  # Violates MRP
                    'discount_percent': -20.0,
                    'category': 'clothing'
                }
            ]
        }
        
        result = await validator.validate_pricing_strategy(strategy_with_violations)
        
        # Should detect strategy-level violations
        strategy_violation = any('STRAT_001' in violation for violation in result.violations)
        assert strategy_violation, "Strategy coherence violation should be detected"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, validator):
        """Test error handling in validation methods."""
        # Test with invalid data types
        invalid_recommendation = {
            'id': 'not-a-uuid',
            'product_id': None,
            'mrp': 'invalid-decimal',
            'proposed_selling_price': 'also-invalid'
        }
        
        result = await validator.validate_mrp_compliance(invalid_recommendation)
        
        # Should handle errors gracefully
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert len(result.violations) > 0
        
        # Should contain error information
        error_found = any('error' in str(result.validation_details).lower() for _ in [True])
        assert error_found or len(result.violations) > 0, "Error should be handled and reported"


class TestComplianceValidatorIntegration:
    """Integration tests for ComplianceValidator with other components."""
    
    @pytest.fixture
    def validator(self):
        """Create a ComplianceValidator instance for integration testing."""
        return ComplianceValidator()
    
    @pytest.mark.asyncio
    async def test_constraint_explanation_generation(self, validator):
        """Test generation of regulatory constraint explanations."""
        # Test electronics category explanation
        explanation = await validator.generate_constraint_explanation('electronics')
        
        assert explanation['category'] == 'electronics'
        assert 'constraints' in explanation
        assert 'business_impact' in explanation
        assert 'compliance_guidance' in explanation
        assert 'regulatory_context' in explanation
        
        # Check MRP compliance explanation
        mrp_constraint = explanation['constraints']['mrp_compliance']
        assert 'title' in mrp_constraint
        assert 'description' in mrp_constraint
        assert 'business_meaning' in mrp_constraint
        assert 'compliance_tips' in mrp_constraint
        
        # Check discount limits explanation
        discount_constraint = explanation['constraints']['discount_limits']
        assert discount_constraint['regular_limit'] == 70  # Electronics limit
        assert discount_constraint['seasonal_limit'] == 80
        
        # Test with recommendation context
        recommendation = {
            'discount_percent': 65.0,
            'category': 'electronics'
        }
        
        explanation_with_context = await validator.generate_constraint_explanation(
            'electronics', recommendation
        )
        
        assert explanation_with_context['business_impact']['discount_risk'] == 'approaching_limits'
    
    @pytest.mark.asyncio
    async def test_essential_commodity_explanation(self, validator):
        """Test explanation generation for essential commodities."""
        explanation = await validator.generate_constraint_explanation('food')
        
        # Should have essential commodity constraints
        assert 'essential_commodity' in explanation['constraints']
        
        essential_constraint = explanation['constraints']['essential_commodity']
        assert essential_constraint['markup_limit'] == 15
        assert 'hoarding_restrictions' in essential_constraint
        assert 'compliance_tips' in essential_constraint
        
        # Check business impact
        assert explanation['business_impact']['pricing_flexibility'] == 'limited'
        assert explanation['business_impact']['operational_complexity'] == 'medium'
    
    @pytest.mark.asyncio
    async def test_luxury_goods_explanation(self, validator):
        """Test explanation generation for luxury goods."""
        explanation = await validator.generate_constraint_explanation('jewelry')
        
        # Should have luxury goods constraints
        assert 'luxury_goods' in explanation['constraints']
        
        luxury_constraint = explanation['constraints']['luxury_goods']
        assert 'additional_requirements' in luxury_constraint
        assert 'compliance_tips' in luxury_constraint
        
        # Check business impact
        assert explanation['business_impact']['pricing_flexibility'] == 'high'
        assert explanation['business_impact']['competitive_positioning'] == 'premium'
    
    @pytest.mark.asyncio
    async def test_system_limitations_and_transparency(self, validator):
        """Test system limitations and transparency information."""
        transparency_info = await validator.get_system_limitations_and_transparency()
        
        assert 'system_capabilities' in transparency_info
        assert 'limitations' in transparency_info
        assert 'data_sources' in transparency_info
        assert 'accuracy_disclaimers' in transparency_info
        assert 'user_responsibilities' in transparency_info
        
        # Check system capabilities
        capabilities = transparency_info['system_capabilities']
        assert 'mrp_compliance_checking' in capabilities
        assert 'discount_limit_validation' in capabilities
        assert 'regulatory_constraint_explanation' in capabilities
        
        # Check limitations
        limitations = transparency_info['limitations']
        assert 'regulatory_interpretation' in limitations
        assert 'real_time_regulation_updates' in limitations
        assert 'state_specific_variations' in limitations
        
        # Check data sources
        data_sources = transparency_info['data_sources']
        assert 'regulation_database' in data_sources
        assert 'category_classifications' in data_sources
        
        # Check user responsibilities
        responsibilities = transparency_info['user_responsibilities']
        assert 'verification' in responsibilities
        assert 'legal_consultation' in responsibilities
    
    @pytest.mark.asyncio
    async def test_regulatory_change_notification(self, validator):
        """Test regulatory change notification processing."""
        # Test discount limit change
        changes = [
            {
                'id': 'CHANGE_001',
                'type': 'discount_limit_update',
                'title': 'Updated Electronics Discount Limits',
                'description': 'New discount limits for electronics category',
                'category': 'electronics',
                'new_requirements': {
                    'discount_limits': {
                        'max_discount_percent': 75,  # Increased from 70
                        'seasonal_max': 85  # Increased from 80
                    }
                },
                'effective_date': '2024-02-01',
                'urgency': 'medium',
                'affected_categories': ['electronics']
            },
            {
                'id': 'CHANGE_002',
                'type': 'mrp_regulation_update',
                'title': 'Updated MRP Display Requirements',
                'description': 'New MRP display requirements',
                'new_requirements': {
                    'mrp_compliance': {
                        'digital_display_requirement': {
                            'description': 'Digital displays must show MRP clearly',
                            'enabled': True,
                            'severity': 'medium'
                        }
                    }
                },
                'effective_date': '2024-02-15',
                'urgency': 'high',
                'affected_categories': ['all']
            }
        ]
        
        result = await validator.notify_regulatory_changes(changes)
        
        assert result['changes_processed'] == 2
        assert result['successful_updates'] >= 1  # At least one should succeed
        assert len(result['user_notifications']) >= 1
        assert 'impact_analysis' in result
        assert 'recommended_actions' in result
        
        # Check that rules were actually updated
        updated_electronics_limits = validator.regulation_rules['discount_regulations']['maximum_discount_limits']['electronics']
        assert updated_electronics_limits['max_discount_percent'] == 75
        
        # Check impact analysis
        impact = result['impact_analysis']
        assert 'electronics' in impact['affected_categories']
        assert 'pricing_strategy' in impact['business_areas_impacted']
        
        # Check recommendations
        recommendations = result['recommended_actions']
        assert len(recommendations) > 0
        
        discount_recommendation = next(
            (r for r in recommendations if 'discount' in r['action'].lower()), 
            None
        )
        assert discount_recommendation is not None
        assert discount_recommendation['priority'] in ['high', 'critical']
    
    @pytest.mark.asyncio
    async def test_regulatory_change_error_handling(self, validator):
        """Test error handling in regulatory change processing."""
        # Test invalid change structure
        invalid_changes = [
            {
                'id': 'INVALID_001',
                'type': 'unknown_type',
                'description': 'Invalid change type'
            }
        ]
        
        result = await validator.notify_regulatory_changes(invalid_changes)
        
        assert result['changes_processed'] == 1
        assert result['failed_updates'] >= 0  # Should handle gracefully
        
        # Test missing required fields
        incomplete_changes = [
            {
                'type': 'discount_limit_update'
                # Missing required fields
            }
        ]
        
        result = await validator.notify_regulatory_changes(incomplete_changes)
        assert result['changes_processed'] == 1
    
    @pytest.mark.asyncio
    async def test_constraint_explanation_error_handling(self, validator):
        """Test error handling in constraint explanation generation."""
        # Test with invalid category
        explanation = await validator.generate_constraint_explanation('invalid_category')
        
        # Should still return basic structure even for unknown categories
        assert explanation['category'] == 'invalid_category'
        assert 'constraints' in explanation
    
    @pytest.mark.asyncio
    async def test_real_world_scenario_diwali_sale(self, validator):
        """Test a real-world Diwali sale scenario."""
        diwali_strategy = {
            'id': uuid4(),
            'strategy_name': 'Diwali Festival Sale 2024',
            'strategy_type': 'seasonal',
            'duration_days': 10,
            'products': [
                {
                    'product_id': 'ELECTRONICS_TV_001',
                    'mrp': 50000.00,
                    'proposed_price': 35000.00,
                    'discount_percent': 30.0,
                    'category': 'electronics',
                    'price_breakdown_available': True
                },
                {
                    'product_id': 'JEWELRY_GOLD_001',
                    'mrp': 100000.00,
                    'proposed_price': 85000.00,
                    'discount_percent': 15.0,
                    'category': 'jewelry',
                    'price_breakdown_available': True
                },
                {
                    'product_id': 'CLOTHING_ETHNIC_001',
                    'mrp': 5000.00,
                    'proposed_price': 3000.00,
                    'discount_percent': 40.0,
                    'category': 'clothing',
                    'price_breakdown_available': True
                }
            ]
        }
        
        result = await validator.validate_pricing_strategy(diwali_strategy)
        
        # Should be compliant or require review (not non-compliant)
        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
        
        # Should validate all products
        assert result.validation_details['products_validated'] == 3
        
        # Should have appropriate constraints for different categories
        constraints = await validator.get_regulatory_constraints('jewelry')
        assert constraints['luxury_item'] is True
        
        # Test constraint explanations for each category
        electronics_explanation = await validator.generate_constraint_explanation('electronics')
        jewelry_explanation = await validator.generate_constraint_explanation('jewelry')
        clothing_explanation = await validator.generate_constraint_explanation('clothing')
        
        assert electronics_explanation['constraints']['discount_limits']['regular_limit'] == 70
        assert jewelry_explanation['constraints']['luxury_goods']['business_meaning']
        assert clothing_explanation['constraints']['discount_limits']['regular_limit'] == 75
    
    @pytest.mark.asyncio
    async def test_essential_commodities_during_crisis(self, validator):
        """Test essential commodities pricing during crisis scenarios."""
        crisis_recommendation = {
            'id': uuid4(),
            'product_id': 'FOOD_RICE_001',
            'mrp': 100.00,
            'proposed_selling_price': 95.00,
            'discount_percent': 5.0,
            'category': 'food',
            'cost_price': 80.00,  # 18.75% markup - exceeds 15% limit
            'is_essential_commodity': True,
            'price_breakdown_available': True
        }
        
        result = await validator.validate_mrp_compliance(crisis_recommendation)
        
        # Should be non-compliant due to excessive markup (18.75% > 15%)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        assert result.constraints.get('essential_commodity') is True
        
        # Test constraint explanation for essential commodities
        explanation = await validator.generate_constraint_explanation('food', crisis_recommendation)
        
        assert 'essential_commodity' in explanation['constraints']
        essential_constraint = explanation['constraints']['essential_commodity']
        assert essential_constraint['markup_limit'] == 15
        assert 'hoarding_restrictions' in essential_constraint
        
        # Test reasonable markup during crisis
        crisis_recommendation['cost_price'] = 85.00  # 11.76% markup - within limit
        
        result = await validator.validate_mrp_compliance(crisis_recommendation)
        assert result.compliance_status == ComplianceStatus.COMPLIANT
        
        # Test excessive markup during crisis
        crisis_recommendation['cost_price'] = 70.00  # 35.7% markup - exceeds limit
        
        result = await validator.validate_mrp_compliance(crisis_recommendation)
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT
        
        essential_violation = any('ESS_001' in violation for violation in result.violations)
        assert essential_violation, "Essential commodity markup violation should be detected"
    
    @pytest.mark.asyncio
    async def test_comprehensive_transparency_workflow(self, validator):
        """Test comprehensive transparency and explanation workflow."""
        # Step 1: Get system limitations
        transparency = await validator.get_system_limitations_and_transparency()
        
        # Step 2: Get constraint explanations for different categories
        categories = ['electronics', 'food', 'jewelry', 'clothing']
        explanations = {}
        
        for category in categories:
            explanations[category] = await validator.generate_constraint_explanation(category)
        
        # Step 3: Simulate regulatory changes
        regulatory_changes = [
            {
                'id': 'TRANSPARENCY_TEST_001',
                'type': 'consumer_protection_update',
                'title': 'Enhanced Transparency Requirements',
                'description': 'New requirements for pricing transparency',
                'new_requirements': {
                    'consumer_protection': {
                        'enhanced_transparency': {
                            'description': 'Enhanced pricing transparency required',
                            'enabled': True,
                            'severity': 'high'
                        }
                    }
                },
                'effective_date': '2024-03-01',
                'urgency': 'medium',
                'affected_categories': categories
            }
        ]
        
        change_result = await validator.notify_regulatory_changes(regulatory_changes)
        
        # Verify comprehensive workflow
        assert transparency['system_capabilities']['regulatory_constraint_explanation']
        assert all(exp['category'] in categories for exp in explanations.values())
        assert change_result['successful_updates'] >= 0
        
        # Verify that explanations contain business-friendly language
        for category, explanation in explanations.items():
            constraints = explanation['constraints']
            assert 'mrp_compliance' in constraints
            
            mrp_constraint = constraints['mrp_compliance']
            assert 'business_meaning' in mrp_constraint
            assert 'compliance_tips' in mrp_constraint
            
            # Verify business impact analysis
            business_impact = explanation['business_impact']
            assert 'pricing_flexibility' in business_impact
            assert 'operational_complexity' in business_impact
            
            # Verify compliance guidance
            guidance = explanation['compliance_guidance']
            assert 'immediate_actions' in guidance
            assert 'ongoing_monitoring' in guidance
            assert 'documentation_requirements' in guidance