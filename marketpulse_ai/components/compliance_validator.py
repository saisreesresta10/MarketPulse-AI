"""
Compliance Validator component for MarketPulse AI.

This module implements MRP (Maximum Retail Price) regulation compliance checking,
violation detection, and regulatory constraint validation for the Indian retail market.
"""

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4

from ..core.interfaces import ComplianceValidatorInterface
from ..core.models import ComplianceResult, ComplianceStatus, SalesDataPoint
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class RegulationViolationError(Exception):
    """Raised when a regulation violation is detected."""
    pass


class ComplianceValidator(ComplianceValidatorInterface):
    """
    Implementation of MRP regulation compliance validation.
    
    Handles validation of recommendations against Indian retail regulations,
    MRP constraints, and consumer protection laws with configurable rule engine.
    """
    
    def __init__(self, settings=None):
        """Initialize the compliance validator with configuration."""
        self.settings = settings  # Don't call get_settings() if None is passed
        self.regulation_rules = self._load_default_regulation_rules()
        self.validator_version = "1.0.0"
        logger.info("Compliance validator initialized with MRP regulation rules")
    
    def _load_default_regulation_rules(self) -> Dict[str, Any]:
        """
        Load default MRP regulation rules for Indian retail market.
        
        Returns:
            Dictionary containing regulation rules and constraints
        """
        return {
            # Core MRP Regulations
            'mrp_compliance': {
                'max_selling_price_rule': {
                    'description': 'Selling price cannot exceed MRP',
                    'enabled': True,
                    'severity': 'critical',
                    'violation_code': 'MRP_001'
                },
                'mrp_display_requirement': {
                    'description': 'MRP must be clearly displayed on product',
                    'enabled': True,
                    'severity': 'high',
                    'violation_code': 'MRP_002'
                },
                'inclusive_of_taxes': {
                    'description': 'MRP must be inclusive of all taxes',
                    'enabled': True,
                    'severity': 'critical',
                    'violation_code': 'MRP_003'
                }
            },
            
            # Discount Regulations
            'discount_regulations': {
                'maximum_discount_limits': {
                    'electronics': {'max_discount_percent': 70, 'seasonal_max': 80},
                    'clothing': {'max_discount_percent': 75, 'seasonal_max': 85},
                    'food': {'max_discount_percent': 50, 'seasonal_max': 60},
                    'jewelry': {'max_discount_percent': 40, 'seasonal_max': 50},
                    'books': {'max_discount_percent': 60, 'seasonal_max': 70},
                    'default': {'max_discount_percent': 60, 'seasonal_max': 70}
                },
                'discount_duration_limits': {
                    'flash_sale_max_hours': 24,
                    'seasonal_sale_max_days': 30,
                    'clearance_sale_max_days': 90
                },
                'discount_frequency_limits': {
                    'min_days_between_sales': 7,
                    'max_sales_per_month': 4
                }
            },
            
            # Consumer Protection Laws
            'consumer_protection': {
                'misleading_pricing': {
                    'description': 'Pricing must not be misleading to consumers',
                    'enabled': True,
                    'severity': 'high',
                    'violation_code': 'CP_001'
                },
                'false_discount_claims': {
                    'description': 'Discount claims must be genuine and verifiable',
                    'enabled': True,
                    'severity': 'high',
                    'violation_code': 'CP_002'
                },
                'price_manipulation': {
                    'description': 'Artificial price inflation before discounts prohibited',
                    'enabled': True,
                    'severity': 'critical',
                    'violation_code': 'CP_003'
                }
            },
            
            # Category-Specific Regulations
            'category_specific': {
                'essential_commodities': {
                    'categories': ['food', 'medicine', 'fuel'],
                    'price_control_applicable': True,
                    'hoarding_restrictions': True,
                    'max_markup_percent': 15
                },
                'luxury_goods': {
                    'categories': ['jewelry', 'luxury_electronics', 'premium_clothing'],
                    'additional_tax_compliance': True,
                    'import_duty_considerations': True
                },
                'seasonal_goods': {
                    'categories': ['festival_items', 'seasonal_clothing', 'holiday_decorations'],
                    'seasonal_pricing_flexibility': True,
                    'pre_season_discount_restrictions': True
                }
            },
            
            # Regional Compliance
            'regional_regulations': {
                'state_specific_taxes': {
                    'gst_compliance_required': True,
                    'local_tax_variations': True
                },
                'local_trading_laws': {
                    'shop_act_compliance': True,
                    'local_authority_permissions': True
                }
            },
            
            # Validation Thresholds
            'validation_thresholds': {
                'price_variance_alert_percent': 5.0,
                'discount_anomaly_threshold': 0.8,
                'compliance_confidence_minimum': 0.7
            }
        }
    
    async def validate_mrp_compliance(self, recommendation: Dict[str, Any]) -> ComplianceResult:
        """
        Validate recommendation against MRP regulations.
        
        Args:
            recommendation: Recommendation to validate containing pricing information
            
        Returns:
            Compliance validation result with detailed findings
        """
        logger.info(f"Validating MRP compliance for recommendation: {recommendation.get('id', 'unknown')}")
        
        violations = []
        warnings = []
        constraints = {}
        validation_details = {}
        
        try:
            # Extract pricing information from recommendation
            product_id = recommendation.get('product_id')
            mrp = recommendation.get('mrp')
            proposed_price = recommendation.get('proposed_selling_price')
            discount_percent = recommendation.get('discount_percent', 0)
            category = recommendation.get('category', 'default')
            
            # Validate required fields
            if not product_id or mrp is None or proposed_price is None:
                violations.append("Missing required pricing information (product_id, mrp, proposed_selling_price)")
                return self._create_compliance_result(
                    recommendation_id=recommendation.get('id'),
                    status=ComplianceStatus.NON_COMPLIANT,
                    violations=violations,
                    warnings=warnings,
                    constraints=constraints,
                    validation_details=validation_details
                )
            
            # Convert to Decimal for precise calculations
            mrp_decimal = Decimal(str(mrp))
            proposed_price_decimal = Decimal(str(proposed_price))
            
            # Core MRP Compliance Checks
            mrp_violations = await self._check_mrp_violations(
                mrp_decimal, proposed_price_decimal, product_id
            )
            violations.extend(mrp_violations)
            
            # Discount Limit Checks
            discount_violations, discount_warnings = await self._check_discount_limits(
                category, discount_percent, recommendation
            )
            violations.extend(discount_violations)
            warnings.extend(discount_warnings)
            
            # Consumer Protection Checks
            consumer_violations, consumer_warnings = await self._check_consumer_protection(
                recommendation, mrp_decimal, proposed_price_decimal
            )
            violations.extend(consumer_violations)
            warnings.extend(consumer_warnings)
            
            # Category-Specific Checks
            category_violations, category_constraints = await self._check_category_specific_rules(
                category, recommendation
            )
            violations.extend(category_violations)
            constraints.update(category_constraints)
            
            # Regional Compliance Checks
            regional_warnings = await self._check_regional_compliance(recommendation)
            warnings.extend(regional_warnings)
            
            # Compile validation details
            validation_details = {
                'mrp_original': float(mrp_decimal),
                'proposed_price': float(proposed_price_decimal),
                'discount_percent': discount_percent,
                'category': category,
                'validation_timestamp': datetime.utcnow().isoformat(),
                'rules_checked': list(self.regulation_rules.keys()),
                'compliance_score': self._calculate_compliance_score(violations, warnings)
            }
            
            # Determine overall compliance status
            if violations:
                status = ComplianceStatus.NON_COMPLIANT
            elif warnings:
                status = ComplianceStatus.REQUIRES_REVIEW
            else:
                status = ComplianceStatus.COMPLIANT
            
            logger.info(f"MRP compliance validation complete. Status: {status}, Violations: {len(violations)}, Warnings: {len(warnings)}")
            
            return self._create_compliance_result(
                recommendation_id=recommendation.get('id'),
                status=status,
                violations=violations,
                warnings=warnings,
                constraints=constraints,
                validation_details=validation_details
            )
            
        except Exception as e:
            logger.error(f"Error during MRP compliance validation: {e}")
            violations.append(f"Validation error: {str(e)}")
            
            return self._create_compliance_result(
                recommendation_id=recommendation.get('id'),
                status=ComplianceStatus.NON_COMPLIANT,
                violations=violations,
                warnings=warnings,
                constraints=constraints,
                validation_details={'error': str(e)}
            )
    
    async def _check_mrp_violations(self, mrp: Decimal, proposed_price: Decimal, product_id: str) -> List[str]:
        """
        Check for core MRP regulation violations.
        
        Args:
            mrp: Maximum Retail Price
            proposed_price: Proposed selling price
            product_id: Product identifier
            
        Returns:
            List of MRP violations found
        """
        violations = []
        
        # Check if selling price exceeds MRP
        if proposed_price > mrp:
            violation_msg = (
                f"MRP_001: Proposed selling price {proposed_price} exceeds MRP {mrp} "
                f"for product {product_id}. This violates Indian MRP regulations."
            )
            violations.append(violation_msg)
            logger.warning(violation_msg)
        
        # Check for zero or negative prices
        if proposed_price <= 0:
            violations.append(f"MRP_004: Invalid selling price {proposed_price} for product {product_id}")
        
        if mrp <= 0:
            violations.append(f"MRP_005: Invalid MRP {mrp} for product {product_id}")
        
        # Check for unrealistic price differences (potential data errors)
        if mrp > 0 and proposed_price > 0:
            price_ratio = proposed_price / mrp
            if price_ratio < Decimal('0.01'):  # Less than 1% of MRP
                violations.append(
                    f"MRP_006: Proposed price {proposed_price} is unrealistically low "
                    f"compared to MRP {mrp} (ratio: {price_ratio:.4f})"
                )
        
        return violations
    
    async def _check_discount_limits(self, category: str, discount_percent: float, 
                                   recommendation: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Check discount limits against regulations.
        
        Args:
            category: Product category
            discount_percent: Proposed discount percentage
            recommendation: Full recommendation context
            
        Returns:
            Tuple of (violations, warnings)
        """
        violations = []
        warnings = []
        
        # Get category-specific discount limits
        discount_rules = self.regulation_rules['discount_regulations']['maximum_discount_limits']
        category_limits = discount_rules.get(category.lower(), discount_rules['default'])
        
        max_discount = category_limits['max_discount_percent']
        seasonal_max = category_limits['seasonal_max']
        
        # Check if this is a seasonal sale
        is_seasonal = recommendation.get('is_seasonal_sale', False)
        applicable_max = seasonal_max if is_seasonal else max_discount
        
        # Check discount limits
        if discount_percent > applicable_max:
            violation_msg = (
                f"DISC_001: Proposed discount {discount_percent}% exceeds maximum allowed "
                f"{applicable_max}% for category '{category}' "
                f"({'seasonal' if is_seasonal else 'regular'} sale)"
            )
            violations.append(violation_msg)
        
        # Check discount duration limits
        sale_duration = recommendation.get('sale_duration_days', 0)
        duration_limits = self.regulation_rules['discount_regulations']['discount_duration_limits']
        
        sale_type = recommendation.get('sale_type', 'regular')
        if sale_type == 'flash_sale' and sale_duration > duration_limits['flash_sale_max_hours'] / 24:
            violations.append(
                f"DISC_002: Flash sale duration {sale_duration} days exceeds maximum "
                f"{duration_limits['flash_sale_max_hours']} hours"
            )
        elif sale_type == 'seasonal' and sale_duration > duration_limits['seasonal_sale_max_days']:
            violations.append(
                f"DISC_003: Seasonal sale duration {sale_duration} days exceeds maximum "
                f"{duration_limits['seasonal_sale_max_days']} days"
            )
        elif sale_type == 'clearance' and sale_duration > duration_limits['clearance_sale_max_days']:
            violations.append(
                f"DISC_004: Clearance sale duration {sale_duration} days exceeds maximum "
                f"{duration_limits['clearance_sale_max_days']} days"
            )
        
        # Warning for high discounts (even if within limits)
        if discount_percent > max_discount * 0.8:  # 80% of maximum
            warnings.append(
                f"High discount warning: {discount_percent}% discount is approaching "
                f"maximum limit of {applicable_max}% for category '{category}'"
            )
        
        return violations, warnings
    
    async def _check_consumer_protection(self, recommendation: Dict[str, Any], 
                                       mrp: Decimal, proposed_price: Decimal) -> Tuple[List[str], List[str]]:
        """
        Check consumer protection law compliance.
        
        Args:
            recommendation: Recommendation to validate
            mrp: Maximum Retail Price
            proposed_price: Proposed selling price
            
        Returns:
            Tuple of (violations, warnings)
        """
        violations = []
        warnings = []
        
        # Check for misleading pricing patterns
        original_price = recommendation.get('original_price')
        if original_price:
            original_price_decimal = Decimal(str(original_price))
            
            # Check if "original price" is artificially inflated
            if original_price_decimal > mrp:
                violations.append(
                    f"CP_003: Original price {original_price} exceeds MRP {mrp}, "
                    "indicating potential price manipulation before discount"
                )
            
            # Check for fake discount claims
            if original_price_decimal <= proposed_price:
                violations.append(
                    f"CP_002: False discount claim - original price {original_price} "
                    f"is not higher than proposed price {proposed_price}"
                )
        
        # Check for unrealistic discount claims
        discount_percent = recommendation.get('discount_percent', 0)
        if discount_percent > 90:
            violations.append(
                f"CP_001: Unrealistic discount claim of {discount_percent}% "
                "may be misleading to consumers"
            )
        
        # Check pricing transparency requirements (only warn, don't make it a violation)
        if not recommendation.get('price_breakdown_available', True):  # Default to True
            warnings.append(
                "Price breakdown not available for transparency compliance"
            )
        
        return violations, warnings
    
    async def _check_category_specific_rules(self, category: str, 
                                           recommendation: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Check category-specific regulatory rules.
        
        Args:
            category: Product category
            recommendation: Recommendation to validate
            
        Returns:
            Tuple of (violations, constraints)
        """
        violations = []
        constraints = {}
        
        category_rules = self.regulation_rules['category_specific']
        
        # Check essential commodities rules
        essential_categories = category_rules['essential_commodities']['categories']
        if category.lower() in essential_categories:
            max_markup = category_rules['essential_commodities']['max_markup_percent']
            
            # Calculate markup if cost price is available
            cost_price = recommendation.get('cost_price')
            proposed_price = recommendation.get('proposed_selling_price')
            
            if cost_price and proposed_price and float(cost_price) > 0:
                markup_percent = ((float(proposed_price) - float(cost_price)) / float(cost_price)) * 100
                
                if markup_percent > max_markup:
                    violations.append(
                        f"ESS_001: Markup {markup_percent:.2f}% exceeds maximum allowed "
                        f"{max_markup}% for essential commodity category '{category}'"
                    )
            
            constraints['essential_commodity'] = True
            constraints['max_markup_percent'] = max_markup
            constraints['hoarding_restrictions'] = True
        
        # Check luxury goods compliance
        luxury_categories = category_rules['luxury_goods']['categories']
        if category.lower() in luxury_categories:
            constraints['luxury_item'] = True
            constraints['additional_tax_compliance_required'] = True
            
            # Check for import duty considerations
            if recommendation.get('is_imported', False):
                constraints['import_duty_applicable'] = True
        
        # Check seasonal goods rules
        seasonal_categories = category_rules['seasonal_goods']['categories']
        if category.lower() in seasonal_categories:
            constraints['seasonal_item'] = True
            constraints['seasonal_pricing_flexibility'] = True
            
            # Check pre-season discount restrictions
            if recommendation.get('is_pre_season_sale', False):
                discount_percent = recommendation.get('discount_percent', 0)
                if discount_percent > 30:  # Pre-season discount limit
                    violations.append(
                        f"SEAS_001: Pre-season discount {discount_percent}% exceeds "
                        "maximum allowed 30% for seasonal goods"
                    )
        
        return violations, constraints
    
    async def _check_regional_compliance(self, recommendation: Dict[str, Any]) -> List[str]:
        """
        Check regional compliance requirements.
        
        Args:
            recommendation: Recommendation to validate
            
        Returns:
            List of regional compliance warnings
        """
        warnings = []
        
        # Check GST compliance requirements
        if not recommendation.get('gst_compliant', True):
            warnings.append(
                "Regional compliance: GST compliance verification required"
            )
        
        # Check state-specific tax considerations
        state = recommendation.get('state')
        if state and state.lower() in ['kerala', 'west bengal', 'delhi']:
            warnings.append(
                f"Regional compliance: Additional state-specific tax rules may apply in {state}"
            )
        
        # Check local trading law compliance
        if not recommendation.get('shop_act_compliant', True):
            warnings.append(
                "Regional compliance: Shop Act compliance verification required"
            )
        
        return warnings
    
    def _calculate_compliance_score(self, violations: List[str], warnings: List[str]) -> float:
        """
        Calculate overall compliance score.
        
        Args:
            violations: List of violations found
            warnings: List of warnings found
            
        Returns:
            Compliance score between 0 and 1
        """
        base_score = 1.0
        
        # Deduct for violations (more severe)
        violation_penalty = len(violations) * 0.3
        
        # Deduct for warnings (less severe)
        warning_penalty = len(warnings) * 0.1
        
        final_score = max(0.0, base_score - violation_penalty - warning_penalty)
        return round(final_score, 3)
    
    def _create_compliance_result(self, recommendation_id: Optional[Union[UUID, str]], status: ComplianceStatus,
                                violations: List[str], warnings: List[str], 
                                constraints: Dict[str, Any], validation_details: Dict[str, Any]) -> ComplianceResult:
        """
        Create a ComplianceResult object.
        
        Args:
            recommendation_id: ID of the recommendation being validated
            status: Compliance status
            violations: List of violations found
            warnings: List of warnings found
            constraints: Regulatory constraints
            validation_details: Detailed validation information
            
        Returns:
            ComplianceResult object
        """
        regulations_checked = [
            'MRP Regulations (Legal Metrology Act)',
            'Consumer Protection Act 2019',
            'Competition Act 2002',
            'Goods and Services Tax Act',
            'Essential Commodities Act'
        ]
        
        # Handle invalid UUID gracefully
        valid_recommendation_id = None
        if recommendation_id:
            try:
                if isinstance(recommendation_id, str):
                    valid_recommendation_id = UUID(recommendation_id)
                else:
                    valid_recommendation_id = recommendation_id
            except (ValueError, TypeError):
                # If UUID is invalid, set to None and add to validation details
                validation_details['invalid_recommendation_id'] = str(recommendation_id)
        
        return ComplianceResult(
            recommendation_id=valid_recommendation_id,
            compliance_status=status,
            regulations_checked=regulations_checked,
            violations=violations,
            warnings=warnings,
            constraints=constraints,
            validation_details=validation_details,
            validator_version=self.validator_version
        )
    
    async def check_discount_limits(self, product_id: str, proposed_discount: float) -> ComplianceResult:
        """
        Check if proposed discount complies with regulations.
        
        Args:
            product_id: Product identifier
            proposed_discount: Proposed discount percentage
            
        Returns:
            Compliance result for discount proposal
        """
        logger.info(f"Checking discount limits for product {product_id}: {proposed_discount}%")
        
        # Create a minimal recommendation for validation
        recommendation = {
            'id': uuid4(),
            'product_id': product_id,
            'discount_percent': proposed_discount,
            'category': 'default',  # Will need to be enhanced with actual category lookup
            'sale_type': 'regular'
        }
        
        violations, warnings = await self._check_discount_limits('default', proposed_discount, recommendation)
        
        status = ComplianceStatus.COMPLIANT
        if violations:
            status = ComplianceStatus.NON_COMPLIANT
        elif warnings:
            status = ComplianceStatus.REQUIRES_REVIEW
        
        validation_details = {
            'product_id': product_id,
            'proposed_discount': proposed_discount,
            'validation_type': 'discount_limits_only',
            'validation_timestamp': datetime.utcnow().isoformat()
        }
        
        return self._create_compliance_result(
            recommendation_id=recommendation['id'],
            status=status,
            violations=violations,
            warnings=warnings,
            constraints={},
            validation_details=validation_details
        )
    
    async def validate_pricing_strategy(self, strategy: Dict[str, Any]) -> ComplianceResult:
        """
        Validate entire pricing strategy for compliance.
        
        Args:
            strategy: Pricing strategy to validate
            
        Returns:
            Comprehensive compliance validation result
        """
        logger.info(f"Validating pricing strategy: {strategy.get('strategy_name', 'unnamed')}")
        
        violations = []
        warnings = []
        constraints = {}
        validation_details = {}
        
        try:
            # Validate strategy structure
            required_fields = ['products', 'strategy_type', 'duration_days']
            missing_fields = [field for field in required_fields if field not in strategy]
            
            if missing_fields:
                violations.append(f"Missing required strategy fields: {missing_fields}")
                return self._create_compliance_result(
                    recommendation_id=strategy.get('id'),
                    status=ComplianceStatus.NON_COMPLIANT,
                    violations=violations,
                    warnings=warnings,
                    constraints=constraints,
                    validation_details={'error': 'Invalid strategy structure'}
                )
            
            # Validate each product in the strategy
            products = strategy['products']
            product_results = []
            
            for product in products:
                # Create recommendation for each product
                recommendation = {
                    'id': uuid4(),
                    'product_id': product.get('product_id'),
                    'mrp': product.get('mrp'),
                    'proposed_selling_price': product.get('proposed_price'),
                    'discount_percent': product.get('discount_percent', 0),
                    'category': product.get('category', 'default'),
                    'sale_duration_days': strategy.get('duration_days'),
                    'sale_type': strategy.get('strategy_type', 'regular')
                }
                
                # Validate individual product
                product_result = await self.validate_mrp_compliance(recommendation)
                product_results.append({
                    'product_id': product.get('product_id'),
                    'compliance_status': product_result.compliance_status,
                    'violations': product_result.violations,
                    'warnings': product_result.warnings
                })
                
                # Aggregate violations and warnings
                violations.extend([f"Product {product.get('product_id')}: {v}" for v in product_result.violations])
                warnings.extend([f"Product {product.get('product_id')}: {w}" for w in product_result.warnings])
            
            # Strategy-level validations
            strategy_violations = await self._validate_strategy_coherence(strategy, product_results)
            violations.extend(strategy_violations)
            
            # Determine overall status
            if violations:
                status = ComplianceStatus.NON_COMPLIANT
            elif warnings:
                status = ComplianceStatus.REQUIRES_REVIEW
            else:
                status = ComplianceStatus.COMPLIANT
            
            validation_details = {
                'strategy_name': strategy.get('strategy_name'),
                'strategy_type': strategy.get('strategy_type'),
                'products_validated': len(products),
                'duration_days': strategy.get('duration_days'),
                'product_results': product_results,
                'validation_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Pricing strategy validation complete. Status: {status}")
            
            return self._create_compliance_result(
                recommendation_id=strategy.get('id'),
                status=status,
                violations=violations,
                warnings=warnings,
                constraints=constraints,
                validation_details=validation_details
            )
            
        except Exception as e:
            logger.error(f"Error validating pricing strategy: {e}")
            violations.append(f"Strategy validation error: {str(e)}")
            
            return self._create_compliance_result(
                recommendation_id=strategy.get('id'),
                status=ComplianceStatus.NON_COMPLIANT,
                violations=violations,
                warnings=warnings,
                constraints=constraints,
                validation_details={'error': str(e)}
            )
    
    async def _validate_strategy_coherence(self, strategy: Dict[str, Any], 
                                         product_results: List[Dict[str, Any]]) -> List[str]:
        """
        Validate strategy-level coherence and consistency.
        
        Args:
            strategy: Pricing strategy
            product_results: Individual product validation results
            
        Returns:
            List of strategy-level violations
        """
        violations = []
        
        # Check for consistent strategy application
        non_compliant_products = [
            r for r in product_results 
            if r['compliance_status'] == ComplianceStatus.NON_COMPLIANT
        ]
        
        if len(non_compliant_products) > len(product_results) * 0.5:  # More than 50% non-compliant
            violations.append(
                f"STRAT_001: Strategy has {len(non_compliant_products)} non-compliant products "
                f"out of {len(product_results)} total products (>50% failure rate)"
            )
        
        # Check strategy duration consistency
        duration_days = strategy.get('duration_days', 0)
        strategy_type = strategy.get('strategy_type', 'regular')
        
        duration_limits = self.regulation_rules['discount_regulations']['discount_duration_limits']
        
        if strategy_type == 'flash_sale' and duration_days > 1:
            violations.append(
                f"STRAT_002: Flash sale strategy duration {duration_days} days "
                "exceeds typical flash sale duration of 1 day"
            )
        
        # Check for conflicting product categories in same strategy
        categories = set()
        for product in strategy.get('products', []):
            categories.add(product.get('category', 'default'))
        
        if len(categories) > 5:  # Too many diverse categories
            violations.append(
                f"STRAT_003: Strategy spans {len(categories)} different categories, "
                "which may create regulatory complexity"
            )
        
        return violations
    
    async def get_regulatory_constraints(self, product_category: str) -> Dict[str, Any]:
        """
        Get regulatory constraints for a product category.
        
        Args:
            product_category: Product category to check
            
        Returns:
            Dictionary of applicable regulatory constraints
        """
        logger.info(f"Retrieving regulatory constraints for category: {product_category}")
        
        constraints = {
            'category': product_category,
            'mrp_compliance_required': True,
            'consumer_protection_applicable': True,
            'retrieved_at': datetime.utcnow().isoformat()
        }
        
        # Get discount limits
        discount_rules = self.regulation_rules['discount_regulations']['maximum_discount_limits']
        category_limits = discount_rules.get(product_category.lower(), discount_rules['default'])
        
        constraints['discount_limits'] = {
            'max_regular_discount_percent': category_limits['max_discount_percent'],
            'max_seasonal_discount_percent': category_limits['seasonal_max']
        }
        
        # Check category-specific rules
        category_rules = self.regulation_rules['category_specific']
        
        # Essential commodities
        if product_category.lower() in category_rules['essential_commodities']['categories']:
            constraints['essential_commodity'] = True
            constraints['price_control_applicable'] = True
            constraints['max_markup_percent'] = category_rules['essential_commodities']['max_markup_percent']
            constraints['hoarding_restrictions'] = True
        
        # Luxury goods
        if product_category.lower() in category_rules['luxury_goods']['categories']:
            constraints['luxury_item'] = True
            constraints['additional_tax_compliance_required'] = True
            constraints['import_duty_considerations'] = True
        
        # Seasonal goods
        if product_category.lower() in category_rules['seasonal_goods']['categories']:
            constraints['seasonal_item'] = True
            constraints['seasonal_pricing_flexibility'] = True
            constraints['pre_season_discount_restrictions'] = True
        
        # Duration limits
        constraints['duration_limits'] = self.regulation_rules['discount_regulations']['discount_duration_limits']
        
        # Frequency limits
        constraints['frequency_limits'] = self.regulation_rules['discount_regulations']['discount_frequency_limits']
        
        logger.info(f"Retrieved {len(constraints)} regulatory constraints for category {product_category}")
        return constraints
    
    async def update_regulation_rules(self, new_rules: Dict[str, Any]) -> bool:
        """
        Update regulation rules when regulations change.
        
        Args:
            new_rules: Updated regulation rules
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            logger.info("Updating regulation rules")
            
            # Validate new rules structure
            required_sections = ['mrp_compliance', 'discount_regulations', 'consumer_protection']
            
            for section in required_sections:
                if section not in new_rules:
                    logger.error(f"Missing required section in new rules: {section}")
                    return False
            
            # Backup current rules
            backup_rules = self.regulation_rules.copy()
            
            try:
                # Update rules
                self.regulation_rules.update(new_rules)
                
                # Update validator version
                self.validator_version = f"{self.validator_version}.{datetime.utcnow().strftime('%Y%m%d')}"
                
                logger.info(f"Regulation rules updated successfully. New version: {self.validator_version}")
                return True
                
            except Exception as e:
                # Restore backup on failure
                self.regulation_rules = backup_rules
                logger.error(f"Failed to update regulation rules, restored backup: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating regulation rules: {e}")
            return False

    async def generate_constraint_explanation(self, product_category: str, 
                                            recommendation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate human-readable explanations of regulatory constraints.
        
        This method provides clear, business-friendly explanations of why certain
        regulatory constraints apply and what they mean for pricing decisions.
        
        Args:
            product_category: Product category to explain constraints for
            recommendation: Optional recommendation context for specific explanations
            
        Returns:
            Dictionary containing detailed constraint explanations
        """
        logger.info(f"Generating constraint explanation for category: {product_category}")
        
        explanation = {
            'category': product_category,
            'generated_at': datetime.utcnow().isoformat(),
            'explanation_version': self.validator_version,
            'constraints': {},
            'business_impact': {},
            'compliance_guidance': {},
            'regulatory_context': {}
        }
        
        try:
            # Get base constraints
            constraints = await self.get_regulatory_constraints(product_category)
            
            # Generate MRP compliance explanation
            explanation['constraints']['mrp_compliance'] = {
                'title': 'Maximum Retail Price (MRP) Compliance',
                'description': (
                    'Under the Legal Metrology Act 2009, all packaged commodities must display '
                    'the Maximum Retail Price (MRP) inclusive of all taxes. The selling price '
                    'cannot exceed this MRP under any circumstances.'
                ),
                'business_meaning': (
                    'You cannot sell any product above its printed MRP. This is a legal requirement '
                    'that protects consumers from price exploitation and ensures price transparency.'
                ),
                'penalties': (
                    'Violations can result in fines up to ₹25,000 for first offense and up to '
                    '₹1,00,000 for subsequent offenses, plus potential imprisonment.'
                ),
                'compliance_tips': [
                    'Always verify MRP is clearly printed on product packaging',
                    'Ensure all taxes (GST, local taxes) are included in MRP calculation',
                    'Never sell above MRP even during high demand periods',
                    'Maintain proper documentation of MRP compliance'
                ]
            }
            
            # Generate discount limit explanations
            discount_limits = constraints.get('discount_limits', {})
            explanation['constraints']['discount_limits'] = {
                'title': f'Discount Limits for {product_category.title()} Category',
                'description': (
                    f'Regular discounts for {product_category} products are limited to '
                    f'{discount_limits.get("max_regular_discount_percent", 60)}% to prevent '
                    'predatory pricing and maintain fair competition.'
                ),
                'regular_limit': discount_limits.get('max_regular_discount_percent', 60),
                'seasonal_limit': discount_limits.get('max_seasonal_discount_percent', 70),
                'business_meaning': (
                    'These limits ensure you can offer competitive discounts while preventing '
                    'unsustainable pricing that could harm market competition or indicate '
                    'misleading original pricing.'
                ),
                'seasonal_flexibility': (
                    f'During recognized seasonal sales (festivals, end-of-season), you can '
                    f'offer up to {discount_limits.get("max_seasonal_discount_percent", 70)}% '
                    'discount to clear inventory and meet consumer expectations.'
                ),
                'compliance_tips': [
                    'Document the reason for discounts (clearance, seasonal, promotional)',
                    'Ensure original prices were genuine and not artificially inflated',
                    'Maintain consistent discount policies across similar products',
                    'Consider market positioning when setting discount levels'
                ]
            }
            
            # Generate category-specific explanations
            if constraints.get('essential_commodity'):
                explanation['constraints']['essential_commodity'] = {
                    'title': 'Essential Commodity Regulations',
                    'description': (
                        f'{product_category.title()} is classified as an essential commodity under '
                        'the Essential Commodities Act 1955, which imposes stricter price controls '
                        'to ensure affordability and prevent hoarding.'
                    ),
                    'markup_limit': constraints.get('max_markup_percent', 15),
                    'business_meaning': (
                        f'Your markup on cost price cannot exceed {constraints.get("max_markup_percent", 15)}% '
                        'for essential commodities. This ensures these necessary items remain '
                        'affordable for all consumers.'
                    ),
                    'hoarding_restrictions': (
                        'You cannot stockpile essential commodities beyond normal business requirements. '
                        'Excessive inventory may be considered hoarding and is punishable by law.'
                    ),
                    'compliance_tips': [
                        'Maintain detailed cost and pricing records',
                        'Ensure markup calculations include all legitimate business costs',
                        'Monitor inventory levels to avoid hoarding accusations',
                        'Stay updated on government price notifications'
                    ]
                }
            
            if constraints.get('luxury_item'):
                explanation['constraints']['luxury_goods'] = {
                    'title': 'Luxury Goods Compliance',
                    'description': (
                        f'{product_category.title()} items are classified as luxury goods, '
                        'which may be subject to additional tax compliance requirements and '
                        'import duty considerations.'
                    ),
                    'business_meaning': (
                        'Luxury goods often have complex tax structures including higher GST rates, '
                        'import duties, and may require additional documentation for compliance.'
                    ),
                    'additional_requirements': [
                        'Higher GST rates may apply (18% or 28%)',
                        'Import duty compliance for imported luxury items',
                        'Additional documentation for high-value transactions',
                        'Potential wealth tax implications for customers'
                    ],
                    'compliance_tips': [
                        'Verify correct GST classification and rates',
                        'Maintain import documentation for foreign goods',
                        'Provide detailed invoices for high-value sales',
                        'Consider customer tax implications in pricing strategy'
                    ]
                }
            
            if constraints.get('seasonal_item'):
                explanation['constraints']['seasonal_goods'] = {
                    'title': 'Seasonal Goods Flexibility',
                    'description': (
                        f'{product_category.title()} items are recognized as seasonal goods, '
                        'allowing for greater pricing flexibility during appropriate seasons '
                        'while maintaining consumer protection.'
                    ),
                    'business_meaning': (
                        'You have more flexibility in pricing seasonal items, especially during '
                        'peak seasons and clearance periods, but must avoid misleading practices.'
                    ),
                    'seasonal_considerations': [
                        'Higher discounts allowed during end-of-season clearance',
                        'Pre-season discount restrictions to prevent market manipulation',
                        'Festival and holiday pricing flexibility',
                        'Inventory clearance allowances'
                    ],
                    'compliance_tips': [
                        'Clearly communicate seasonal nature of pricing',
                        'Avoid excessive pre-season discounts',
                        'Document seasonal pricing strategies',
                        'Ensure clearance sales are genuine inventory reduction'
                    ]
                }
            
            # Generate business impact analysis
            explanation['business_impact'] = self._generate_business_impact_analysis(
                product_category, constraints, recommendation
            )
            
            # Generate compliance guidance
            explanation['compliance_guidance'] = self._generate_compliance_guidance(
                product_category, constraints, recommendation
            )
            
            # Generate regulatory context
            explanation['regulatory_context'] = self._generate_regulatory_context(product_category)
            
            logger.info(f"Generated comprehensive constraint explanation for {product_category}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating constraint explanation: {e}")
            return {
                'category': product_category,
                'error': f"Failed to generate explanation: {str(e)}",
                'generated_at': datetime.utcnow().isoformat()
            }

    def _generate_business_impact_analysis(self, category: str, constraints: Dict[str, Any], 
                                         recommendation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate business impact analysis for regulatory constraints."""
        impact = {
            'pricing_flexibility': 'moderate',
            'competitive_positioning': 'standard',
            'inventory_management': 'standard',
            'operational_complexity': 'low'
        }
        
        # Adjust based on category characteristics
        if constraints.get('essential_commodity'):
            impact['pricing_flexibility'] = 'limited'
            impact['operational_complexity'] = 'medium'
            impact['inventory_management'] = 'restricted'
            impact['social_responsibility'] = 'high'
        
        if constraints.get('luxury_item'):
            impact['pricing_flexibility'] = 'high'
            impact['competitive_positioning'] = 'premium'
            impact['operational_complexity'] = 'high'
            impact['tax_complexity'] = 'high'
        
        if constraints.get('seasonal_item'):
            impact['pricing_flexibility'] = 'seasonal'
            impact['inventory_management'] = 'time-sensitive'
            impact['demand_volatility'] = 'high'
        
        # Add specific recommendations if available
        if recommendation:
            discount_percent = recommendation.get('discount_percent', 0)
            max_discount = constraints.get('discount_limits', {}).get('max_regular_discount_percent', 60)
            
            if discount_percent > max_discount * 0.8:
                impact['discount_risk'] = 'approaching_limits'
            elif discount_percent > max_discount:
                impact['discount_risk'] = 'exceeds_limits'
            else:
                impact['discount_risk'] = 'within_limits'
        
        return impact

    def _generate_compliance_guidance(self, category: str, constraints: Dict[str, Any], 
                                    recommendation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate specific compliance guidance."""
        guidance = {
            'immediate_actions': [],
            'ongoing_monitoring': [],
            'documentation_requirements': [],
            'risk_mitigation': []
        }
        
        # Base compliance actions
        guidance['immediate_actions'].extend([
            'Verify MRP compliance for all products',
            'Review current discount levels against category limits',
            'Ensure price transparency in customer communications'
        ])
        
        guidance['ongoing_monitoring'].extend([
            'Monitor regulatory updates and changes',
            'Track competitor pricing for market positioning',
            'Review compliance status monthly'
        ])
        
        guidance['documentation_requirements'].extend([
            'Maintain MRP compliance records',
            'Document discount rationale and duration',
            'Keep cost and pricing calculation records'
        ])
        
        # Category-specific guidance
        if constraints.get('essential_commodity'):
            guidance['immediate_actions'].append('Calculate and verify markup percentages')
            guidance['ongoing_monitoring'].append('Monitor government price notifications')
            guidance['risk_mitigation'].append('Maintain reasonable inventory levels to avoid hoarding accusations')
        
        if constraints.get('luxury_item'):
            guidance['immediate_actions'].append('Verify GST classification and rates')
            guidance['documentation_requirements'].append('Maintain import duty and tax compliance records')
            guidance['risk_mitigation'].append('Consider customer tax implications in pricing')
        
        if constraints.get('seasonal_item'):
            guidance['immediate_actions'].append('Plan seasonal pricing strategy in advance')
            guidance['ongoing_monitoring'].append('Track seasonal demand patterns')
            guidance['risk_mitigation'].append('Avoid excessive pre-season discounting')
        
        return guidance

    def _generate_regulatory_context(self, category: str) -> Dict[str, Any]:
        """Generate regulatory context and background information."""
        context = {
            'primary_regulations': [
                {
                    'name': 'Legal Metrology Act 2009',
                    'relevance': 'MRP display and compliance requirements',
                    'authority': 'Ministry of Consumer Affairs, Food & Public Distribution'
                },
                {
                    'name': 'Consumer Protection Act 2019',
                    'relevance': 'Consumer rights and protection from unfair practices',
                    'authority': 'Ministry of Consumer Affairs, Food & Public Distribution'
                },
                {
                    'name': 'Competition Act 2002',
                    'relevance': 'Prevention of anti-competitive pricing practices',
                    'authority': 'Competition Commission of India (CCI)'
                }
            ],
            'enforcement_agencies': [
                'Legal Metrology Department (State-level)',
                'Consumer Protection Authorities',
                'Competition Commission of India',
                'Goods and Services Tax Department'
            ],
            'recent_updates': {
                'last_checked': datetime.utcnow().isoformat(),
                'version': self.validator_version,
                'note': 'Regulations are subject to change. Always verify current requirements.'
            }
        }
        
        # Add category-specific regulatory context
        if category.lower() in ['food', 'medicine', 'fuel']:
            context['primary_regulations'].append({
                'name': 'Essential Commodities Act 1955',
                'relevance': 'Price control and anti-hoarding measures',
                'authority': 'Ministry of Consumer Affairs, Food & Public Distribution'
            })
        
        return context

    async def get_system_limitations_and_transparency(self) -> Dict[str, Any]:
        """
        Provide comprehensive information about system limitations and data sources.
        
        This method ensures transparency about what the system can and cannot do,
        data sources used, and limitations in regulatory compliance checking.
        
        Returns:
            Dictionary containing system limitations and transparency information
        """
        logger.info("Generating system limitations and transparency information")
        
        transparency_info = {
            'generated_at': datetime.utcnow().isoformat(),
            'validator_version': self.validator_version,
            'system_capabilities': {},
            'limitations': {},
            'data_sources': {},
            'accuracy_disclaimers': {},
            'user_responsibilities': {},
            'update_information': {}
        }
        
        # System capabilities
        transparency_info['system_capabilities'] = {
            'mrp_compliance_checking': {
                'description': 'Validates selling prices against Maximum Retail Price regulations',
                'accuracy': 'High for basic MRP violations',
                'coverage': 'All product categories with MRP information'
            },
            'discount_limit_validation': {
                'description': 'Checks discount percentages against category-specific limits',
                'accuracy': 'High for standard categories',
                'coverage': 'Electronics, clothing, food, jewelry, books, and general categories'
            },
            'consumer_protection_checks': {
                'description': 'Identifies potential misleading pricing practices',
                'accuracy': 'Medium - requires human judgment for complex cases',
                'coverage': 'Basic misleading pricing patterns'
            },
            'category_specific_rules': {
                'description': 'Applies special rules for essential commodities and luxury goods',
                'accuracy': 'High for defined categories',
                'coverage': 'Essential commodities, luxury goods, seasonal items'
            },
            'regulatory_constraint_explanation': {
                'description': 'Provides human-readable explanations of regulatory requirements',
                'accuracy': 'High for documented regulations',
                'coverage': 'Major Indian retail regulations'
            }
        }
        
        # System limitations
        transparency_info['limitations'] = {
            'regulatory_interpretation': {
                'limitation': 'Cannot provide legal advice or definitive regulatory interpretation',
                'impact': 'Users must consult legal experts for complex compliance questions',
                'mitigation': 'System provides guidance based on documented regulations only'
            },
            'real_time_regulation_updates': {
                'limitation': 'Regulation updates are not real-time',
                'impact': 'Recent regulatory changes may not be immediately reflected',
                'mitigation': 'Regular updates are performed, but users should verify current regulations'
            },
            'state_specific_variations': {
                'limitation': 'Limited coverage of state-specific regulatory variations',
                'impact': 'Some local regulations may not be fully covered',
                'mitigation': 'General warnings provided for states with known variations'
            },
            'complex_business_scenarios': {
                'limitation': 'May not handle highly complex or unusual business scenarios',
                'impact': 'Edge cases may require manual review',
                'mitigation': 'System flags uncertain cases for human review'
            },
            'market_context_analysis': {
                'limitation': 'Does not analyze broader market context or competitive dynamics',
                'impact': 'Compliance checking is done in isolation from market conditions',
                'mitigation': 'Users should consider market factors separately'
            },
            'enforcement_prediction': {
                'limitation': 'Cannot predict enforcement actions or regulatory priorities',
                'impact': 'Compliance does not guarantee immunity from regulatory scrutiny',
                'mitigation': 'System focuses on documented requirements only'
            }
        }
        
        # Data sources
        transparency_info['data_sources'] = {
            'regulation_database': {
                'source': 'Indian government regulatory documents and official publications',
                'last_updated': datetime.utcnow().isoformat(),
                'update_frequency': 'Monthly review, immediate updates for major changes',
                'reliability': 'High - based on official sources'
            },
            'category_classifications': {
                'source': 'GST classification codes and industry standards',
                'reliability': 'High for standard categories, medium for edge cases',
                'limitations': 'May not cover all niche product categories'
            },
            'discount_limits': {
                'source': 'Industry best practices and regulatory guidance',
                'reliability': 'Medium - based on general guidelines',
                'limitations': 'Actual limits may vary by specific circumstances'
            },
            'essential_commodity_lists': {
                'source': 'Essential Commodities Act notifications and state government orders',
                'reliability': 'High for central notifications, medium for state variations',
                'limitations': 'State-specific variations may not be fully covered'
            }
        }
        
        # Accuracy disclaimers
        transparency_info['accuracy_disclaimers'] = {
            'general_disclaimer': (
                'This system provides guidance based on documented regulations and industry '
                'best practices. It is not a substitute for professional legal advice.'
            ),
            'regulation_changes': (
                'Regulations are subject to change without notice. Users should verify '
                'current requirements with appropriate authorities.'
            ),
            'interpretation_limits': (
                'Regulatory interpretation can be complex and context-dependent. The system '
                'provides general guidance that may not apply to all specific situations.'
            ),
            'enforcement_variations': (
                'Regulatory enforcement may vary by jurisdiction and circumstances. '
                'Compliance with system recommendations does not guarantee regulatory approval.'
            )
        }
        
        # User responsibilities
        transparency_info['user_responsibilities'] = {
            'verification': (
                'Users are responsible for verifying current regulations and ensuring '
                'compliance with all applicable laws and requirements.'
            ),
            'legal_consultation': (
                'Users should consult qualified legal professionals for complex compliance '
                'questions or when in doubt about regulatory requirements.'
            ),
            'ongoing_monitoring': (
                'Users must monitor regulatory changes and update their compliance practices '
                'accordingly. The system is a tool, not a complete compliance solution.'
            ),
            'context_consideration': (
                'Users should consider their specific business context, market conditions, '
                'and circumstances when applying system recommendations.'
            ),
            'record_keeping': (
                'Users are responsible for maintaining appropriate documentation and records '
                'to demonstrate compliance with applicable regulations.'
            )
        }
        
        # Update information
        transparency_info['update_information'] = {
            'current_version': self.validator_version,
            'last_major_update': '2024-01-01',  # This would be dynamically set
            'update_schedule': 'Monthly reviews with immediate updates for critical changes',
            'notification_method': 'Version number changes indicate updates',
            'change_log_availability': 'Contact system administrator for detailed change logs'
        }
        
        logger.info("Generated comprehensive system limitations and transparency information")
        return transparency_info

    async def notify_regulatory_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and communicate regulatory changes to users.
        
        This method handles notifications about regulatory changes, updates internal
        rules, and provides guidance on how changes affect existing recommendations.
        
        Args:
            changes: List of regulatory changes to process
            
        Returns:
            Dictionary containing change processing results and user notifications
        """
        logger.info(f"Processing {len(changes)} regulatory changes")
        
        notification_result = {
            'processed_at': datetime.utcnow().isoformat(),
            'changes_processed': len(changes),
            'successful_updates': 0,
            'failed_updates': 0,
            'user_notifications': [],
            'system_updates': [],
            'impact_analysis': {},
            'recommended_actions': []
        }
        
        try:
            for change in changes:
                try:
                    # Process individual change
                    change_result = await self._process_regulatory_change(change)
                    
                    if change_result['success']:
                        notification_result['successful_updates'] += 1
                        notification_result['system_updates'].append(change_result)
                        
                        # Generate user notification
                        user_notification = self._generate_change_notification(change, change_result)
                        notification_result['user_notifications'].append(user_notification)
                        
                    else:
                        notification_result['failed_updates'] += 1
                        logger.error(f"Failed to process regulatory change: {change_result.get('error')}")
                        
                except Exception as e:
                    notification_result['failed_updates'] += 1
                    logger.error(f"Error processing regulatory change: {e}")
            
            # Generate impact analysis
            notification_result['impact_analysis'] = self._analyze_change_impact(changes)
            
            # Generate recommended actions
            notification_result['recommended_actions'] = self._generate_change_recommendations(changes)
            
            logger.info(f"Processed regulatory changes: {notification_result['successful_updates']} successful, "
                       f"{notification_result['failed_updates']} failed")
            
            return notification_result
            
        except Exception as e:
            logger.error(f"Error processing regulatory changes: {e}")
            notification_result['error'] = str(e)
            return notification_result

    async def _process_regulatory_change(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single regulatory change."""
        result = {
            'change_id': change.get('id', 'unknown'),
            'success': False,
            'updates_made': [],
            'error': None
        }
        
        try:
            change_type = change.get('type')
            affected_area = change.get('affected_area')
            new_requirements = change.get('new_requirements', {})
            
            if change_type == 'discount_limit_update':
                # Update discount limits
                category = change.get('category', 'default')
                new_limits = new_requirements.get('discount_limits', {})
                
                if category in self.regulation_rules['discount_regulations']['maximum_discount_limits']:
                    self.regulation_rules['discount_regulations']['maximum_discount_limits'][category].update(new_limits)
                    result['updates_made'].append(f"Updated discount limits for {category}")
                
            elif change_type == 'mrp_regulation_update':
                # Update MRP regulations
                mrp_updates = new_requirements.get('mrp_compliance', {})
                self.regulation_rules['mrp_compliance'].update(mrp_updates)
                result['updates_made'].append("Updated MRP compliance rules")
                
            elif change_type == 'essential_commodity_update':
                # Update essential commodity rules
                essential_updates = new_requirements.get('essential_commodities', {})
                self.regulation_rules['category_specific']['essential_commodities'].update(essential_updates)
                result['updates_made'].append("Updated essential commodity regulations")
                
            elif change_type == 'consumer_protection_update':
                # Update consumer protection rules
                cp_updates = new_requirements.get('consumer_protection', {})
                self.regulation_rules['consumer_protection'].update(cp_updates)
                result['updates_made'].append("Updated consumer protection regulations")
            
            # Update validator version to reflect changes
            if result['updates_made']:
                self.validator_version = f"{self.validator_version}.{datetime.utcnow().strftime('%Y%m%d%H%M')}"
                result['success'] = True
                
        except Exception as e:
            result['error'] = str(e)
            
        return result

    def _generate_change_notification(self, change: Dict[str, Any], 
                                    change_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate user notification for regulatory change."""
        notification = {
            'notification_id': uuid4(),
            'change_id': change.get('id'),
            'title': change.get('title', 'Regulatory Update'),
            'description': change.get('description', 'A regulatory change has been processed'),
            'effective_date': change.get('effective_date'),
            'urgency': change.get('urgency', 'medium'),
            'affected_categories': change.get('affected_categories', []),
            'impact_summary': change.get('impact_summary', 'Impact assessment pending'),
            'required_actions': change.get('required_actions', []),
            'system_updates': change_result.get('updates_made', []),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Add specific guidance based on change type
        change_type = change.get('type')
        if change_type == 'discount_limit_update':
            notification['guidance'] = (
                'Review your current discount strategies to ensure compliance with updated limits. '
                'Adjust any ongoing promotions that may now exceed the new limits.'
            )
        elif change_type == 'mrp_regulation_update':
            notification['guidance'] = (
                'Verify that all your products comply with updated MRP regulations. '
                'Review pricing strategies and ensure proper MRP display compliance.'
            )
        elif change_type == 'essential_commodity_update':
            notification['guidance'] = (
                'If you sell essential commodities, review your markup calculations and '
                'inventory levels to ensure compliance with updated regulations.'
            )
        
        return notification

    def _analyze_change_impact(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the overall impact of regulatory changes."""
        impact = {
            'overall_severity': 'low',
            'affected_categories': set(),
            'business_areas_impacted': set(),
            'compliance_complexity_change': 'no_change',
            'immediate_action_required': False
        }
        
        high_impact_changes = 0
        
        for change in changes:
            # Track affected categories
            affected_cats = change.get('affected_categories', [])
            impact['affected_categories'].update(affected_cats)
            
            # Track business areas
            change_type = change.get('type', '')
            if 'discount' in change_type:
                impact['business_areas_impacted'].add('pricing_strategy')
            if 'mrp' in change_type:
                impact['business_areas_impacted'].add('mrp_compliance')
            if 'essential' in change_type:
                impact['business_areas_impacted'].add('essential_commodities')
            
            # Assess severity
            urgency = change.get('urgency', 'medium')
            if urgency in ['high', 'critical']:
                high_impact_changes += 1
                impact['immediate_action_required'] = True
        
        # Determine overall severity
        if high_impact_changes > 0:
            impact['overall_severity'] = 'high' if high_impact_changes > 2 else 'medium'
        
        # Assess complexity change
        if len(changes) > 5:
            impact['compliance_complexity_change'] = 'increased'
        elif any(change.get('simplifies_compliance', False) for change in changes):
            impact['compliance_complexity_change'] = 'decreased'
        
        # Convert sets to lists for JSON serialization
        impact['affected_categories'] = list(impact['affected_categories'])
        impact['business_areas_impacted'] = list(impact['business_areas_impacted'])
        
        return impact

    def _generate_change_recommendations(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommended actions based on regulatory changes."""
        recommendations = []
        
        # Group changes by type for better recommendations
        change_types = {}
        for change in changes:
            change_type = change.get('type', 'unknown')
            if change_type not in change_types:
                change_types[change_type] = []
            change_types[change_type].append(change)
        
        # Generate type-specific recommendations
        for change_type, type_changes in change_types.items():
            if change_type == 'discount_limit_update':
                recommendations.append({
                    'priority': 'high',
                    'action': 'Review Current Discount Strategies',
                    'description': (
                        'Audit all current and planned discount campaigns to ensure compliance '
                        'with updated discount limits. Adjust any promotions that exceed new limits.'
                    ),
                    'timeline': 'Within 7 days',
                    'affected_changes': len(type_changes)
                })
                
            elif change_type == 'mrp_regulation_update':
                recommendations.append({
                    'priority': 'critical',
                    'action': 'Verify MRP Compliance',
                    'description': (
                        'Conduct comprehensive review of all product MRP compliance. '
                        'Ensure pricing systems reflect updated MRP requirements.'
                    ),
                    'timeline': 'Immediate',
                    'affected_changes': len(type_changes)
                })
                
            elif change_type == 'essential_commodity_update':
                recommendations.append({
                    'priority': 'high',
                    'action': 'Review Essential Commodity Pricing',
                    'description': (
                        'Recalculate markup percentages for essential commodities. '
                        'Verify inventory levels comply with anti-hoarding regulations.'
                    ),
                    'timeline': 'Within 3 days',
                    'affected_changes': len(type_changes)
                })
        
        # Add general recommendations
        if len(changes) > 0:
            recommendations.extend([
                {
                    'priority': 'medium',
                    'action': 'Update Staff Training',
                    'description': (
                        'Brief relevant staff on regulatory changes and updated compliance requirements. '
                        'Ensure customer-facing teams understand new constraints.'
                    ),
                    'timeline': 'Within 14 days',
                    'affected_changes': len(changes)
                },
                {
                    'priority': 'medium',
                    'action': 'Review Documentation',
                    'description': (
                        'Update internal compliance documentation and procedures to reflect '
                        'regulatory changes. Ensure audit trails are maintained.'
                    ),
                    'timeline': 'Within 30 days',
                    'affected_changes': len(changes)
                }
            ])
        
        return recommendations