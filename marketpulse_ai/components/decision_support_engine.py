"""
Decision Support Engine for MarketPulse AI.

This module implements the main orchestration engine that coordinates all components
to generate comprehensive business recommendations for discount strategies and
inventory management decisions.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from ..core.interfaces import DecisionSupportEngineInterface
from ..core.models import (
    SalesDataPoint, DemandPattern, ExplainableInsight, RiskAssessment, 
    ComplianceResult, ComplianceStatus, RiskLevel, ConfidenceLevel
)
from .data_processor import DataProcessor
from .risk_assessor import RiskAssessor
from .compliance_validator import ComplianceValidator
from .insight_generator import InsightGenerator


class DecisionSupportEngineError(Exception):
    """Base exception for Decision Support Engine errors."""
    pass


class RecommendationGenerationError(DecisionSupportEngineError):
    """Raised when recommendation generation fails."""
    pass


class OptimizationError(DecisionSupportEngineError):
    """Raised when discount optimization fails."""
    pass


class DecisionSupportEngine(DecisionSupportEngineInterface):
    """
    Main decision support orchestration engine.
    
    Coordinates all MarketPulse AI components to generate comprehensive
    business recommendations with compliance validation and risk assessment.
    """
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 risk_assessor: RiskAssessor,
                 compliance_validator: ComplianceValidator,
                 insight_generator: InsightGenerator):
        """
        Initialize Decision Support Engine with component dependencies.
        
        Args:
            data_processor: Data processing component
            risk_assessor: Risk assessment component
            compliance_validator: Compliance validation component
            insight_generator: Insight generation component
        """
        self.data_processor = data_processor
        self.risk_assessor = risk_assessor
        self.compliance_validator = compliance_validator
        self.insight_generator = insight_generator
        self.logger = logging.getLogger(__name__)
        
        # Configuration for discount optimization
        self.max_discount_percentage = 50.0  # Maximum allowed discount
        self.min_discount_percentage = 5.0   # Minimum meaningful discount
        self.discount_step_size = 5.0        # Discount optimization step size
        
    async def generate_recommendations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive business recommendations.
        
        Args:
            request: Request parameters containing product_ids, analysis_type, etc.
            
        Returns:
            Dictionary containing recommendations and supporting analysis
            
        Raises:
            RecommendationGenerationError: If recommendation generation fails
        """
        try:
            self.logger.info(f"Generating recommendations for request: {request}")
            
            # Extract request parameters
            product_ids = request.get('product_ids', [])
            analysis_type = request.get('analysis_type', 'comprehensive')
            time_horizon = request.get('time_horizon', '3_months')
            
            if not product_ids:
                raise RecommendationGenerationError("No product IDs provided in request")
            
            # Step 1: Extract demand patterns for products
            patterns = await self.data_processor.extract_demand_patterns(product_ids)
            if not patterns:
                raise RecommendationGenerationError(f"No demand patterns found for products: {product_ids}")
            
            # Step 2: Generate insights from patterns
            insights = await self.insight_generator.generate_insights(patterns)
            
            # Step 3: Assess risks for each product
            risk_assessments = []
            for product_id in product_ids:
                # Get current inventory (would come from request in real implementation)
                current_inventory = request.get('inventory_levels', {}).get(product_id, 100)
                
                overstock_risk = await self.risk_assessor.assess_overstock_risk(product_id, current_inventory)
                understock_risk = await self.risk_assessor.assess_understock_risk(product_id, current_inventory)
                
                risk_assessments.extend([overstock_risk, understock_risk])
            
            # Step 4: Generate discount strategy recommendations
            discount_recommendations = await self.optimize_discount_strategy(product_ids)
            
            # Step 5: Validate all recommendations for compliance
            compliance_results = []
            for recommendation in discount_recommendations.get('recommendations', []):
                compliance_result = await self.validate_recommendation_pipeline(recommendation)
                compliance_results.append(compliance_result)
            
            # Step 6: Prioritize recommendations
            all_recommendations = discount_recommendations.get('recommendations', [])
            prioritized_recommendations = await self.prioritize_recommendations(all_recommendations)
            
            # Step 7: Assess business impact
            business_impact = {}
            for recommendation in prioritized_recommendations[:3]:  # Top 3 recommendations
                impact = await self.assess_business_impact(recommendation)
                business_impact[recommendation['id']] = impact
            
            # Compile comprehensive response
            response = {
                'request_id': str(uuid4()),
                'generated_at': datetime.utcnow().isoformat(),
                'analysis_type': analysis_type,
                'time_horizon': time_horizon,
                'products_analyzed': product_ids,
                'summary': {
                    'total_recommendations': len(prioritized_recommendations),
                    'high_priority_count': len([r for r in prioritized_recommendations if r.get('priority') == 'high']),
                    'compliance_issues': len([c for c in compliance_results if c.compliance_status != ComplianceStatus.COMPLIANT]),
                    'critical_risks': len([r for r in risk_assessments if r.risk_level == RiskLevel.CRITICAL])
                },
                'recommendations': prioritized_recommendations,
                'insights': [self._serialize_insight(insight) for insight in insights],
                'risk_assessments': [self._serialize_risk_assessment(risk) for risk in risk_assessments],
                'compliance_results': [self._serialize_compliance_result(result) for result in compliance_results],
                'business_impact': business_impact,
                'discount_strategy': discount_recommendations.get('strategy_summary', {}),
                'next_review_date': (datetime.utcnow() + timedelta(days=30)).isoformat()
            }
            
            self.logger.info(f"Successfully generated {len(prioritized_recommendations)} recommendations")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
            raise RecommendationGenerationError(f"Recommendation generation failed: {str(e)}")
    
    async def optimize_discount_strategy(self, product_ids: List[str]) -> Dict[str, Any]:
        """
        Optimize discount strategy for specified products.
        
        Args:
            product_ids: List of product identifiers
            
        Returns:
            Optimized discount strategy recommendations
            
        Raises:
            OptimizationError: If optimization fails
        """
        try:
            self.logger.info(f"Optimizing discount strategy for products: {product_ids}")
            
            recommendations = []
            strategy_summary = {
                'total_products': len(product_ids),
                'optimization_method': 'demand_pattern_based',
                'optimization_date': datetime.utcnow().isoformat(),
                'constraints_applied': ['mrp_compliance', 'seasonal_adjustment', 'risk_mitigation']
            }
            
            for product_id in product_ids:
                try:
                    # Get demand patterns for the product
                    patterns = await self.data_processor.extract_demand_patterns([product_id])
                    if not patterns:
                        self.logger.warning(f"No patterns found for product {product_id}, skipping")
                        continue
                    
                    pattern = patterns[0]  # Use the first pattern
                    
                    # Calculate optimal discount based on demand pattern
                    optimal_discount = await self._calculate_optimal_discount(pattern)
                    
                    # Determine discount window based on seasonal factors
                    discount_window = await self._determine_discount_window(pattern)
                    
                    # Assess price sensitivity
                    price_sensitivity = await self._assess_price_sensitivity(pattern)
                    
                    # Create recommendation
                    recommendation = {
                        'id': str(uuid4()),
                        'product_id': product_id,
                        'recommendation_type': 'discount_strategy',
                        'optimal_discount_percentage': optimal_discount,
                        'discount_window': discount_window,
                        'price_sensitivity_score': price_sensitivity,
                        'expected_impact': await self._calculate_expected_impact(pattern, optimal_discount),
                        'confidence_level': pattern.confidence_level.value,
                        'supporting_factors': [
                            f"Demand pattern: {pattern.pattern_type}",
                            f"Volatility score: {pattern.volatility_score:.2f}",
                            f"Trend direction: {pattern.trend_direction or 'stable'}"
                        ],
                        'generated_at': datetime.utcnow().isoformat(),
                        'priority': await self._calculate_recommendation_priority(pattern, optimal_discount)
                    }
                    
                    recommendations.append(recommendation)
                    
                except Exception as e:
                    self.logger.error(f"Failed to optimize discount for product {product_id}: {str(e)}")
                    continue
            
            if not recommendations:
                raise OptimizationError("No valid discount recommendations could be generated")
            
            strategy_summary['successful_optimizations'] = len(recommendations)
            strategy_summary['average_discount'] = sum(r['optimal_discount_percentage'] for r in recommendations) / len(recommendations)
            
            return {
                'strategy_summary': strategy_summary,
                'recommendations': recommendations,
                'optimization_metadata': {
                    'max_discount_limit': self.max_discount_percentage,
                    'min_discount_limit': self.min_discount_percentage,
                    'step_size': self.discount_step_size
                }
            }
            
        except Exception as e:
            self.logger.error(f"Discount strategy optimization failed: {str(e)}")
            raise OptimizationError(f"Optimization failed: {str(e)}")
    
    async def assess_business_impact(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess potential business impact of recommendations.
        
        Args:
            recommendation: Recommendation to assess
            
        Returns:
            Business impact analysis results
        """
        try:
            product_id = recommendation.get('product_id')
            discount_percentage = recommendation.get('optimal_discount_percentage', 0)
            
            # Get current risk assessments
            current_inventory = 100  # Would come from inventory system
            overstock_risk = await self.risk_assessor.assess_overstock_risk(product_id, current_inventory)
            
            # Calculate potential revenue impact
            revenue_impact = await self._calculate_revenue_impact(product_id, discount_percentage)
            
            # Calculate inventory impact
            inventory_impact = await self._calculate_inventory_impact(product_id, discount_percentage)
            
            # Assess market positioning impact
            market_impact = await self._assess_market_positioning_impact(product_id, discount_percentage)
            
            return {
                'recommendation_id': recommendation.get('id'),
                'product_id': product_id,
                'revenue_impact': revenue_impact,
                'inventory_impact': inventory_impact,
                'market_positioning_impact': market_impact,
                'risk_mitigation': {
                    'overstock_risk_reduction': max(0, overstock_risk.risk_score - 0.2),
                    'demand_stimulation_potential': min(1.0, discount_percentage / 20.0)
                },
                'implementation_complexity': 'low',  # Discount strategies are typically low complexity
                'time_to_impact': '1-2 weeks',
                'confidence_level': recommendation.get('confidence_level', 'medium'),
                'assessed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Business impact assessment failed: {str(e)}")
            return {
                'error': f"Impact assessment failed: {str(e)}",
                'recommendation_id': recommendation.get('id'),
                'assessed_at': datetime.utcnow().isoformat()
            }
    
    async def prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize recommendations by impact and urgency.
        
        Args:
            recommendations: List of recommendations to prioritize
            
        Returns:
            Prioritized list of recommendations
        """
        try:
            # Calculate priority scores for each recommendation
            scored_recommendations = []
            
            for recommendation in recommendations:
                priority_score = await self._calculate_priority_score(recommendation)
                recommendation['priority_score'] = priority_score
                recommendation['priority'] = self._get_priority_level(priority_score)
                scored_recommendations.append(recommendation)
            
            # Sort by priority score (highest first)
            prioritized = sorted(scored_recommendations, key=lambda x: x['priority_score'], reverse=True)
            
            # Add ranking information
            for i, recommendation in enumerate(prioritized):
                recommendation['rank'] = i + 1
                recommendation['percentile'] = ((len(prioritized) - i) / len(prioritized)) * 100
            
            self.logger.info(f"Prioritized {len(prioritized)} recommendations")
            return prioritized
            
        except Exception as e:
            self.logger.error(f"Recommendation prioritization failed: {str(e)}")
            return recommendations  # Return original list if prioritization fails
    
    async def validate_recommendation_pipeline(self, recommendation: Dict[str, Any]) -> ComplianceResult:
        """
        Validate recommendation through complete compliance pipeline.
        
        Args:
            recommendation: Recommendation to validate
            
        Returns:
            Comprehensive validation result
        """
        try:
            product_id = recommendation.get('product_id')
            discount_percentage = recommendation.get('optimal_discount_percentage', 0)
            
            # Validate MRP compliance
            mrp_result = await self.compliance_validator.validate_mrp_compliance(recommendation)
            
            # Check discount limits
            discount_result = await self.compliance_validator.check_discount_limits(product_id, discount_percentage)
            
            # Validate overall pricing strategy
            strategy = {
                'product_id': product_id,
                'discount_percentage': discount_percentage,
                'recommendation_type': recommendation.get('recommendation_type', 'discount_strategy')
            }
            strategy_result = await self.compliance_validator.validate_pricing_strategy(strategy)
            
            # Combine results
            combined_violations = []
            combined_warnings = []
            
            for result in [mrp_result, discount_result, strategy_result]:
                combined_violations.extend(result.violations)
                combined_warnings.extend(result.warnings)
            
            # Determine overall compliance status
            if combined_violations:
                overall_status = ComplianceStatus.NON_COMPLIANT
            elif combined_warnings:
                overall_status = ComplianceStatus.REQUIRES_REVIEW
            else:
                overall_status = ComplianceStatus.COMPLIANT
            
            return ComplianceResult(
                recommendation_id=UUID(recommendation.get('id')),
                compliance_status=overall_status,
                regulations_checked=['MRP_COMPLIANCE', 'DISCOUNT_LIMITS', 'PRICING_STRATEGY'],
                violations=combined_violations,
                warnings=combined_warnings,
                constraints={
                    'max_discount_allowed': self.max_discount_percentage,
                    'mrp_compliance_required': True,
                    'seasonal_restrictions': []
                },
                validation_details={
                    'mrp_validation': self._serialize_compliance_result(mrp_result),
                    'discount_validation': self._serialize_compliance_result(discount_result),
                    'strategy_validation': self._serialize_compliance_result(strategy_result)
                },
                validator_version='1.0.0'
            )
            
        except Exception as e:
            self.logger.error(f"Compliance validation failed: {str(e)}")
            return ComplianceResult(
                compliance_status=ComplianceStatus.REQUIRES_REVIEW,
                regulations_checked=['ERROR_OCCURRED'],
                violations=[f"Validation error: {str(e)}"],
                validator_version='1.0.0'
            )
    
    # Helper methods for internal calculations
    
    async def _calculate_optimal_discount(self, pattern: DemandPattern) -> float:
        """Calculate optimal discount percentage based on demand pattern."""
        base_discount = 15.0  # Base discount percentage
        
        # Adjust based on volatility (higher volatility = higher discount to stimulate demand)
        volatility_adjustment = pattern.volatility_score * 10.0
        
        # Adjust based on trend direction
        trend_adjustment = 0.0
        if pattern.trend_direction == 'decreasing':
            trend_adjustment = 10.0  # Higher discount for declining products
        elif pattern.trend_direction == 'increasing':
            trend_adjustment = -5.0  # Lower discount for growing products
        
        # Adjust based on confidence level
        confidence_adjustment = 0.0
        if pattern.confidence_level == ConfidenceLevel.HIGH:
            confidence_adjustment = 5.0
        elif pattern.confidence_level == ConfidenceLevel.LOW:
            confidence_adjustment = -5.0
        
        optimal_discount = base_discount + volatility_adjustment + trend_adjustment + confidence_adjustment
        
        # Ensure within bounds
        return max(self.min_discount_percentage, min(self.max_discount_percentage, optimal_discount))
    
    async def _determine_discount_window(self, pattern: DemandPattern) -> Dict[str, Any]:
        """Determine optimal discount timing window."""
        # Default window
        start_date = datetime.utcnow() + timedelta(days=7)
        duration_days = 30
        
        # Adjust based on seasonal factors
        if pattern.seasonal_factors:
            # Find the highest seasonal factor in the next 3 months
            max_factor = max(pattern.seasonal_factors.values())
            if max_factor > 1.2:  # High seasonal demand expected
                duration_days = 14  # Shorter window during high demand
            elif max_factor < 0.8:  # Low seasonal demand expected
                duration_days = 45  # Longer window during low demand
        
        return {
            'start_date': start_date.isoformat(),
            'end_date': (start_date + timedelta(days=duration_days)).isoformat(),
            'duration_days': duration_days,
            'timing_rationale': f"Based on seasonal factors and {pattern.pattern_type} pattern"
        }
    
    async def _assess_price_sensitivity(self, pattern: DemandPattern) -> float:
        """Assess price sensitivity score (0-1, higher = more sensitive)."""
        base_sensitivity = 0.5
        
        # Higher volatility suggests higher price sensitivity
        volatility_factor = pattern.volatility_score * 0.3
        
        # Seasonal patterns may indicate lower price sensitivity
        seasonal_factor = -0.1 if pattern.seasonal_factors else 0.0
        
        sensitivity = base_sensitivity + volatility_factor + seasonal_factor
        return max(0.0, min(1.0, sensitivity))
    
    async def _calculate_expected_impact(self, pattern: DemandPattern, discount_percentage: float) -> Dict[str, Any]:
        """Calculate expected impact of discount strategy."""
        # Simple impact model based on price elasticity
        price_elasticity = await self._assess_price_sensitivity(pattern)
        
        # Expected demand increase (simplified model)
        demand_increase_percentage = discount_percentage * price_elasticity * 2.0
        
        # Expected revenue impact (accounting for lower price but higher volume)
        revenue_multiplier = (1 - discount_percentage / 100) * (1 + demand_increase_percentage / 100)
        revenue_impact_percentage = (revenue_multiplier - 1) * 100
        
        return {
            'demand_increase_percentage': round(demand_increase_percentage, 2),
            'revenue_impact_percentage': round(revenue_impact_percentage, 2),
            'inventory_turnover_improvement': round(demand_increase_percentage * 0.8, 2),
            'market_share_potential': round(discount_percentage * 0.5, 2)
        }
    
    async def _calculate_recommendation_priority(self, pattern: DemandPattern, discount_percentage: float) -> str:
        """Calculate priority level for recommendation."""
        score = 0
        
        # Higher discount suggests higher priority
        if discount_percentage > 25:
            score += 3
        elif discount_percentage > 15:
            score += 2
        else:
            score += 1
        
        # Higher volatility suggests higher priority
        if pattern.volatility_score > 0.7:
            score += 3
        elif pattern.volatility_score > 0.4:
            score += 2
        else:
            score += 1
        
        # Confidence level affects priority
        if pattern.confidence_level == ConfidenceLevel.HIGH:
            score += 2
        elif pattern.confidence_level == ConfidenceLevel.MEDIUM:
            score += 1
        
        # Convert score to priority level
        if score >= 7:
            return 'high'
        elif score >= 4:
            return 'medium'
        else:
            return 'low'
    
    async def _calculate_priority_score(self, recommendation: Dict[str, Any]) -> float:
        """Calculate numerical priority score for sorting."""
        score = 0.0
        
        # Discount percentage factor (0-30 points)
        discount = recommendation.get('optimal_discount_percentage', 0)
        score += min(30, discount * 0.6)
        
        # Confidence level factor (0-20 points)
        confidence = recommendation.get('confidence_level', 'medium')
        confidence_scores = {'high': 20, 'medium': 10, 'low': 5}
        score += confidence_scores.get(confidence, 10)
        
        # Expected impact factor (0-25 points)
        expected_impact = recommendation.get('expected_impact', {})
        revenue_impact = expected_impact.get('revenue_impact_percentage', 0)
        score += min(25, abs(revenue_impact) * 0.5)
        
        # Priority level factor (0-25 points)
        priority = recommendation.get('priority', 'medium')
        priority_scores = {'high': 25, 'medium': 15, 'low': 5}
        score += priority_scores.get(priority, 15)
        
        return score
    
    def _get_priority_level(self, score: float) -> str:
        """Convert numerical score to priority level."""
        if score >= 70:
            return 'high'
        elif score >= 40:
            return 'medium'
        else:
            return 'low'
    
    async def _calculate_revenue_impact(self, product_id: str, discount_percentage: float) -> Dict[str, Any]:
        """Calculate potential revenue impact."""
        # Simplified revenue impact calculation
        base_revenue = 10000  # Would come from sales data in real implementation
        
        # Estimate demand elasticity
        demand_elasticity = 1.5  # Simplified assumption
        
        # Calculate volume increase
        volume_increase = discount_percentage * demand_elasticity / 100
        
        # Calculate revenue impact
        new_price_multiplier = (100 - discount_percentage) / 100
        new_volume_multiplier = 1 + volume_increase
        revenue_multiplier = new_price_multiplier * new_volume_multiplier
        
        revenue_change = (revenue_multiplier - 1) * base_revenue
        
        return {
            'base_revenue_estimate': base_revenue,
            'volume_increase_percentage': round(volume_increase * 100, 2),
            'revenue_change_amount': round(revenue_change, 2),
            'revenue_change_percentage': round((revenue_multiplier - 1) * 100, 2),
            'break_even_volume_increase': round(discount_percentage / (100 - discount_percentage) * 100, 2)
        }
    
    async def _calculate_inventory_impact(self, product_id: str, discount_percentage: float) -> Dict[str, Any]:
        """Calculate potential inventory impact."""
        # Simplified inventory impact calculation
        current_inventory = 100  # Would come from inventory system
        
        # Estimate inventory turnover improvement
        turnover_improvement = discount_percentage * 0.8  # Simplified model
        
        # Calculate days to clear inventory
        base_days_to_clear = 30  # Simplified assumption
        new_days_to_clear = base_days_to_clear / (1 + turnover_improvement / 100)
        
        return {
            'current_inventory_estimate': current_inventory,
            'turnover_improvement_percentage': round(turnover_improvement, 2),
            'days_to_clear_reduction': round(base_days_to_clear - new_days_to_clear, 1),
            'inventory_risk_reduction': 'medium' if discount_percentage > 20 else 'low',
            'optimal_reorder_timing': f"{int(new_days_to_clear - 5)} days"
        }
    
    async def _assess_market_positioning_impact(self, product_id: str, discount_percentage: float) -> Dict[str, Any]:
        """Assess market positioning impact."""
        return {
            'competitive_advantage': 'temporary' if discount_percentage < 20 else 'significant',
            'brand_perception_risk': 'low' if discount_percentage < 25 else 'medium',
            'customer_acquisition_potential': 'high' if discount_percentage > 15 else 'medium',
            'market_share_impact': round(discount_percentage * 0.3, 2),
            'positioning_strategy': 'value_focused' if discount_percentage > 20 else 'quality_focused'
        }
    
    # Serialization helper methods
    
    def _serialize_insight(self, insight: ExplainableInsight) -> Dict[str, Any]:
        """Serialize ExplainableInsight to dictionary."""
        return {
            'id': str(insight.id),
            'title': insight.title,
            'description': insight.description,
            'confidence_level': insight.confidence_level.value,
            'supporting_evidence': insight.supporting_evidence,
            'key_factors': insight.key_factors,
            'business_impact': insight.business_impact,
            'recommended_actions': insight.recommended_actions,
            'data_sources': insight.data_sources,
            'related_products': insight.related_products,
            'created_at': insight.created_at.isoformat(),
            'expires_at': insight.expires_at.isoformat() if insight.expires_at else None
        }
    
    def _serialize_risk_assessment(self, risk: RiskAssessment) -> Dict[str, Any]:
        """Serialize RiskAssessment to dictionary."""
        return {
            'id': str(risk.id),
            'product_id': risk.product_id,
            'risk_type': risk.risk_type,
            'risk_level': risk.risk_level.value,
            'risk_score': risk.risk_score,
            'contributing_factors': risk.contributing_factors,
            'seasonal_adjustments': risk.seasonal_adjustments,
            'early_warning_triggered': risk.early_warning_triggered,
            'mitigation_suggestions': risk.mitigation_suggestions,
            'assessment_date': risk.assessment_date.isoformat(),
            'valid_until': risk.valid_until.isoformat(),
            'created_at': risk.created_at.isoformat()
        }
    
    def _serialize_compliance_result(self, result: ComplianceResult) -> Dict[str, Any]:
        """Serialize ComplianceResult to dictionary."""
        return {
            'id': str(result.id),
            'recommendation_id': str(result.recommendation_id) if result.recommendation_id else None,
            'compliance_status': result.compliance_status.value,
            'regulations_checked': result.regulations_checked,
            'violations': result.violations,
            'warnings': result.warnings,
            'constraints': result.constraints,
            'validation_details': result.validation_details,
            'validator_version': result.validator_version,
            'checked_at': result.checked_at.isoformat()
        }