"""
Scenario Analyzer for MarketPulse AI.

This module implements what-if scenario analysis and modeling capabilities
for strategic decision making and business planning.
"""

import asyncio
import logging
import random
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from ..core.interfaces import ScenarioAnalyzerInterface
from ..core.models import Scenario, ConfidenceLevel
from .data_processor import DataProcessor
from .risk_assessor import RiskAssessor


class ScenarioAnalysisError(Exception):
    """Base exception for Scenario Analyzer errors."""
    pass


class ScenarioGenerationError(ScenarioAnalysisError):
    """Raised when scenario generation fails."""
    pass


class ScenarioAnalyzer(ScenarioAnalyzerInterface):
    """
    What-if scenario analysis and modeling component.
    
    Generates and analyzes business scenarios to support strategic
    decision making with inventory management and discount strategies.
    """
    
    def __init__(self, data_processor: Optional[DataProcessor] = None, 
                 risk_assessor: Optional[RiskAssessor] = None):
        """
        Initialize Scenario Analyzer with optional component dependencies.
        
        Args:
            data_processor: Optional data processing component for historical analysis
            risk_assessor: Optional risk assessment component for risk modeling
        """
        self.data_processor = data_processor
        self.risk_assessor = risk_assessor
        self.logger = logging.getLogger(__name__)
        
        # Scenario generation configuration
        self.max_scenarios_per_request = 10
        self.default_time_horizons = ['1_month', '3_months', '6_months', '1_year']
        self.seasonal_events = [
            'diwali', 'holi', 'eid', 'christmas', 'new_year', 'valentine',
            'monsoon', 'summer', 'winter', 'back_to_school'
        ]
        
        # Market condition variations
        self.market_conditions = [
            'optimistic', 'pessimistic', 'stable', 'volatile', 'recession', 'growth'
        ]
        
        self.logger.info("Scenario Analyzer initialized")
    
    async def generate_scenarios(self, base_parameters: Dict[str, Any]) -> List[Scenario]:
        """
        Generate multiple what-if scenarios from base parameters.
        
        Args:
            base_parameters: Base scenario parameters including products, time_horizon, etc.
            
        Returns:
            List of generated scenarios with variations
            
        Raises:
            ScenarioGenerationError: If scenario generation fails
        """
        try:
            self.logger.info(f"Generating scenarios from base parameters: {base_parameters}")
            
            # Validate base parameters
            if not base_parameters:
                raise ScenarioGenerationError("Base parameters cannot be empty")
            
            product_ids = base_parameters.get('product_ids', [])
            if not product_ids:
                raise ScenarioGenerationError("Product IDs must be provided")
            
            time_horizon = base_parameters.get('time_horizon', '3_months')
            scenario_count = min(base_parameters.get('scenario_count', 5), self.max_scenarios_per_request)
            
            scenarios = []
            
            # Generate base scenario
            base_scenario = await self._create_base_scenario(base_parameters)
            scenarios.append(base_scenario)
            
            # Generate variation scenarios
            for i in range(scenario_count - 1):
                variation_scenario = await self._create_variation_scenario(base_parameters, i + 1)
                scenarios.append(variation_scenario)
            
            self.logger.info(f"Generated {len(scenarios)} scenarios successfully")
            return scenarios
            
        except Exception as e:
            self.logger.error(f"Scenario generation failed: {str(e)}")
            raise ScenarioGenerationError(f"Failed to generate scenarios: {str(e)}")
    
    async def predict_inventory_outcomes(self, scenario: Scenario) -> Dict[str, Any]:
        """
        Predict inventory outcomes for a given scenario.
        
        Args:
            scenario: Scenario to analyze
            
        Returns:
            Dictionary of predicted inventory outcomes
        """
        try:
            self.logger.info(f"Predicting inventory outcomes for scenario: {scenario.name}")
            
            # Extract scenario parameters
            products = scenario.parameters.get('products', [])
            market_condition = scenario.parameters.get('market_condition', 'stable')
            demand_multiplier = scenario.parameters.get('demand_multiplier', 1.0)
            seasonal_factors = scenario.parameters.get('seasonal_factors', {})
            
            inventory_predictions = {}
            
            for product in products:
                product_id = product.get('id', 'unknown')
                current_inventory = product.get('current_inventory', 100)
                
                # Predict demand based on scenario parameters
                predicted_demand = await self._predict_demand(
                    product_id, demand_multiplier, market_condition, seasonal_factors
                )
                
                # Calculate inventory outcomes
                inventory_outcome = await self._calculate_inventory_outcome(
                    current_inventory, predicted_demand, scenario.time_horizon
                )
                
                inventory_predictions[product_id] = inventory_outcome
            
            # Aggregate predictions
            aggregated_outcomes = await self._aggregate_inventory_predictions(inventory_predictions)
            
            return {
                'scenario_id': str(scenario.id),
                'scenario_name': scenario.name,
                'time_horizon': scenario.time_horizon,
                'product_predictions': inventory_predictions,
                'aggregated_outcomes': aggregated_outcomes,
                'confidence_level': scenario.confidence_level.value,
                'predicted_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Inventory outcome prediction failed: {str(e)}")
            return {
                'error': f"Prediction failed: {str(e)}",
                'scenario_id': str(scenario.id),
                'predicted_at': datetime.utcnow().isoformat()
            }
    
    async def analyze_discount_impact(self, scenario: Scenario) -> Dict[str, Any]:
        """
        Analyze impact of discount strategies in a scenario.
        
        Args:
            scenario: Scenario with discount parameters
            
        Returns:
            Dictionary of predicted discount impacts
        """
        try:
            self.logger.info(f"Analyzing discount impact for scenario: {scenario.name}")
            
            # Extract discount parameters
            discount_strategy = scenario.parameters.get('discount_strategy', {})
            products = scenario.parameters.get('products', [])
            market_condition = scenario.parameters.get('market_condition', 'stable')
            
            discount_impacts = {}
            
            for product in products:
                product_id = product.get('id', 'unknown')
                discount_percentage = discount_strategy.get(product_id, 0.0)
                
                if discount_percentage > 0:
                    # Analyze discount impact for this product
                    impact_analysis = await self._analyze_product_discount_impact(
                        product_id, discount_percentage, market_condition, scenario.time_horizon
                    )
                    discount_impacts[product_id] = impact_analysis
            
            # Calculate overall impact
            overall_impact = await self._calculate_overall_discount_impact(discount_impacts)
            
            return {
                'scenario_id': str(scenario.id),
                'scenario_name': scenario.name,
                'discount_strategy': discount_strategy,
                'product_impacts': discount_impacts,
                'overall_impact': overall_impact,
                'market_condition': market_condition,
                'confidence_level': scenario.confidence_level.value,
                'analyzed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Discount impact analysis failed: {str(e)}")
            return {
                'error': f"Analysis failed: {str(e)}",
                'scenario_id': str(scenario.id),
                'analyzed_at': datetime.utcnow().isoformat()
            }
    
    async def model_seasonal_effects(self, scenario: Scenario, seasonal_events: List[str]) -> Scenario:
        """
        Model seasonal effects on scenario outcomes.
        
        Args:
            scenario: Base scenario to enhance
            seasonal_events: List of seasonal events to consider
            
        Returns:
            Enhanced scenario with seasonal modeling
        """
        try:
            self.logger.info(f"Modeling seasonal effects for scenario: {scenario.name}")
            
            # Create enhanced scenario
            enhanced_scenario = scenario.model_copy()
            enhanced_scenario.id = uuid4()  # New ID for enhanced scenario
            enhanced_scenario.name = f"{scenario.name} (Seasonal Enhanced)"
            enhanced_scenario.description = f"{scenario.description} Enhanced with seasonal modeling for: {', '.join(seasonal_events)}"
            
            # Add seasonal considerations
            enhanced_scenario.seasonal_considerations = seasonal_events
            
            # Calculate seasonal factors
            seasonal_factors = await self._calculate_seasonal_factors(seasonal_events, scenario.time_horizon)
            
            # Update scenario parameters with seasonal factors
            enhanced_parameters = enhanced_scenario.parameters.copy()
            enhanced_parameters['seasonal_factors'] = seasonal_factors
            enhanced_parameters['seasonal_events'] = seasonal_events
            
            # Adjust demand multipliers based on seasonal effects
            if 'demand_multiplier' in enhanced_parameters:
                base_multiplier = enhanced_parameters['demand_multiplier']
                seasonal_multiplier = max(seasonal_factors.values()) if seasonal_factors else 1.0
                enhanced_parameters['demand_multiplier'] = base_multiplier * seasonal_multiplier
            
            enhanced_scenario.parameters = enhanced_parameters
            
            # Update predicted outcomes with seasonal effects
            enhanced_outcomes = enhanced_scenario.predicted_outcomes.copy()
            enhanced_outcomes['seasonal_adjustments'] = seasonal_factors
            enhanced_outcomes['seasonal_impact_summary'] = await self._summarize_seasonal_impact(seasonal_factors)
            
            enhanced_scenario.predicted_outcomes = enhanced_outcomes
            
            # Add seasonal limitations
            seasonal_limitations = [
                f"Seasonal modeling based on historical patterns for {', '.join(seasonal_events)}",
                "Actual seasonal impact may vary due to market conditions",
                "External factors not accounted for in seasonal modeling"
            ]
            enhanced_scenario.limitations.extend(seasonal_limitations)
            
            # Adjust confidence level based on seasonal data availability
            if len(seasonal_events) > 3:
                # Lower confidence with too many seasonal factors
                if enhanced_scenario.confidence_level == ConfidenceLevel.HIGH:
                    enhanced_scenario.confidence_level = ConfidenceLevel.MEDIUM
                elif enhanced_scenario.confidence_level == ConfidenceLevel.MEDIUM:
                    enhanced_scenario.confidence_level = ConfidenceLevel.LOW
            
            enhanced_scenario.created_at = datetime.utcnow()
            
            self.logger.info(f"Enhanced scenario with seasonal effects: {enhanced_scenario.name}")
            return enhanced_scenario
            
        except Exception as e:
            self.logger.error(f"Seasonal modeling failed: {str(e)}")
            # Return original scenario if enhancement fails
            return scenario
    
    async def validate_scenario_assumptions(self, scenario: Scenario) -> List[str]:
        """
        Validate and identify limitations in scenario assumptions.
        
        Args:
            scenario: Scenario to validate
            
        Returns:
            List of identified limitations and concerns
        """
        try:
            self.logger.info(f"Validating assumptions for scenario: {scenario.name}")
            
            limitations = []
            
            # Validate time horizon
            time_horizon = scenario.time_horizon
            if time_horizon not in self.default_time_horizons:
                limitations.append(f"Non-standard time horizon '{time_horizon}' may have limited historical data")
            
            # Validate parameters
            parameters = scenario.parameters
            
            # Check product coverage
            products = parameters.get('products', [])
            if len(products) > 10:
                limitations.append("Large number of products may reduce prediction accuracy")
            elif len(products) == 0:
                limitations.append("No products specified - scenario may not be actionable")
            
            # Check market conditions
            market_condition = parameters.get('market_condition', 'stable')
            if market_condition in ['recession', 'volatile']:
                limitations.append(f"'{market_condition}' market conditions increase prediction uncertainty")
            
            # Check demand multipliers
            demand_multiplier = parameters.get('demand_multiplier', 1.0)
            if demand_multiplier > 2.0:
                limitations.append("High demand multiplier (>2.0) may not be realistic")
            elif demand_multiplier < 0.5:
                limitations.append("Low demand multiplier (<0.5) may indicate extreme conditions")
            
            # Check discount strategies
            discount_strategy = parameters.get('discount_strategy', {})
            for product_id, discount in discount_strategy.items():
                if discount > 50:
                    limitations.append(f"High discount ({discount}%) for {product_id} may impact brand perception")
                elif discount < 0:
                    limitations.append(f"Negative discount for {product_id} is not realistic")
            
            # Check seasonal factors
            seasonal_factors = parameters.get('seasonal_factors', {})
            for event, factor in seasonal_factors.items():
                if factor > 3.0:
                    limitations.append(f"Extreme seasonal factor ({factor}) for {event} may be unrealistic")
                elif factor < 0.1:
                    limitations.append(f"Very low seasonal factor ({factor}) for {event} may indicate data issues")
            
            # Validate confidence level consistency
            confidence_level = scenario.confidence_level
            assumption_count = len(scenario.assumptions)
            if confidence_level == ConfidenceLevel.HIGH and assumption_count > 5:
                limitations.append("High confidence with many assumptions may be overoptimistic")
            elif confidence_level == ConfidenceLevel.LOW and assumption_count < 2:
                limitations.append("Low confidence with few assumptions may be overly pessimistic")
            
            # Check data availability (if data processor is available)
            if self.data_processor:
                try:
                    # This would check if we have sufficient historical data
                    data_limitations = await self._validate_data_availability(products)
                    limitations.extend(data_limitations)
                except Exception as e:
                    limitations.append(f"Could not validate data availability: {str(e)}")
            
            # Add existing limitations from scenario
            limitations.extend(scenario.limitations)
            
            # Remove duplicates and sort
            unique_limitations = list(set(limitations))
            unique_limitations.sort()
            
            self.logger.info(f"Identified {len(unique_limitations)} limitations for scenario validation")
            return unique_limitations
            
        except Exception as e:
            self.logger.error(f"Scenario validation failed: {str(e)}")
            return [f"Validation error: {str(e)}"]
    
    # Helper methods for scenario generation and analysis
    
    async def _create_base_scenario(self, base_parameters: Dict[str, Any]) -> Scenario:
        """Create base scenario from parameters."""
        product_ids = base_parameters.get('product_ids', [])
        time_horizon = base_parameters.get('time_horizon', '3_months')
        
        # Create base scenario parameters
        scenario_parameters = {
            'products': [{'id': pid, 'current_inventory': 100} for pid in product_ids],
            'market_condition': 'stable',
            'demand_multiplier': 1.0,
            'discount_strategy': {},
            'seasonal_factors': {},
            'analysis_type': base_parameters.get('analysis_type', 'comprehensive')
        }
        
        # Create base predicted outcomes
        predicted_outcomes = {
            'revenue_impact': 0.0,
            'inventory_turnover': 1.0,
            'market_share_change': 0.0,
            'risk_level': 'medium'
        }
        
        return Scenario(
            name="Base Scenario",
            description="Baseline scenario with current market conditions and no strategic changes",
            parameters=scenario_parameters,
            predicted_outcomes=predicted_outcomes,
            confidence_level=ConfidenceLevel.MEDIUM,
            assumptions=[
                "Market conditions remain stable",
                "No significant external disruptions",
                "Current inventory levels maintained"
            ],
            limitations=[
                "Based on historical data patterns",
                "Does not account for unexpected market changes"
            ],
            time_horizon=time_horizon,
            affected_products=product_ids
        )
    
    async def _create_variation_scenario(self, base_parameters: Dict[str, Any], variation_index: int) -> Scenario:
        """Create variation scenario with different parameters."""
        product_ids = base_parameters.get('product_ids', [])
        time_horizon = base_parameters.get('time_horizon', '3_months')
        
        # Define scenario variations
        variations = [
            {
                'name': 'Optimistic Growth',
                'description': 'Scenario with increased demand and favorable market conditions',
                'market_condition': 'growth',
                'demand_multiplier': 1.3,
                'confidence_level': ConfidenceLevel.MEDIUM,
                'assumptions': [
                    "Market growth continues",
                    "Consumer confidence remains high",
                    "No major competitive threats"
                ]
            },
            {
                'name': 'Economic Downturn',
                'description': 'Scenario with reduced demand due to economic challenges',
                'market_condition': 'recession',
                'demand_multiplier': 0.7,
                'confidence_level': ConfidenceLevel.LOW,
                'assumptions': [
                    "Economic conditions worsen",
                    "Consumer spending decreases",
                    "Price sensitivity increases"
                ]
            },
            {
                'name': 'Aggressive Discounting',
                'description': 'Scenario with significant discount strategies to boost sales',
                'market_condition': 'stable',
                'demand_multiplier': 1.1,
                'discount_strategy': {pid: 25.0 for pid in product_ids},
                'confidence_level': ConfidenceLevel.HIGH,
                'assumptions': [
                    "Discounts drive demand effectively",
                    "Competitors do not match discounts",
                    "Brand perception remains positive"
                ]
            },
            {
                'name': 'Seasonal Peak',
                'description': 'Scenario during peak seasonal demand periods',
                'market_condition': 'optimistic',
                'demand_multiplier': 1.5,
                'seasonal_factors': {'diwali': 1.8, 'christmas': 1.4},
                'confidence_level': ConfidenceLevel.HIGH,
                'assumptions': [
                    "Seasonal patterns repeat",
                    "Inventory can meet increased demand",
                    "Supply chain remains stable"
                ]
            }
        ]
        
        # Select variation based on index
        variation = variations[variation_index % len(variations)]
        
        # Create scenario parameters
        scenario_parameters = {
            'products': [{'id': pid, 'current_inventory': 100} for pid in product_ids],
            'market_condition': variation['market_condition'],
            'demand_multiplier': variation['demand_multiplier'],
            'discount_strategy': variation.get('discount_strategy', {}),
            'seasonal_factors': variation.get('seasonal_factors', {}),
            'analysis_type': base_parameters.get('analysis_type', 'comprehensive')
        }
        
        # Create predicted outcomes based on variation
        predicted_outcomes = await self._calculate_variation_outcomes(variation, product_ids)
        
        return Scenario(
            name=variation['name'],
            description=variation['description'],
            parameters=scenario_parameters,
            predicted_outcomes=predicted_outcomes,
            confidence_level=variation['confidence_level'],
            assumptions=variation['assumptions'],
            limitations=[
                "Based on historical patterns and assumptions",
                "External factors may impact actual outcomes",
                "Market conditions may change unexpectedly"
            ],
            time_horizon=time_horizon,
            affected_products=product_ids
        )
    
    async def _calculate_variation_outcomes(self, variation: Dict[str, Any], product_ids: List[str]) -> Dict[str, Any]:
        """Calculate predicted outcomes for scenario variation."""
        demand_multiplier = variation.get('demand_multiplier', 1.0)
        discount_strategy = variation.get('discount_strategy', {})
        market_condition = variation.get('market_condition', 'stable')
        
        # Calculate revenue impact
        revenue_impact = (demand_multiplier - 1.0) * 100  # Percentage change
        
        # Adjust for discounts
        if discount_strategy:
            avg_discount = sum(discount_strategy.values()) / len(discount_strategy)
            discount_impact = -avg_discount * 0.8  # Discount reduces revenue but increases volume
            revenue_impact += discount_impact
        
        # Calculate inventory turnover
        inventory_turnover = demand_multiplier * 1.2  # Higher demand = higher turnover
        
        # Calculate market share change
        market_share_change = 0.0
        if market_condition == 'growth':
            market_share_change = 2.0
        elif market_condition == 'recession':
            market_share_change = -1.5
        elif discount_strategy:
            market_share_change = len(discount_strategy) * 0.5
        
        # Determine risk level
        risk_level = 'medium'
        if market_condition in ['recession', 'volatile']:
            risk_level = 'high'
        elif market_condition in ['growth', 'optimistic']:
            risk_level = 'low'
        
        return {
            'revenue_impact': round(revenue_impact, 2),
            'inventory_turnover': round(inventory_turnover, 2),
            'market_share_change': round(market_share_change, 2),
            'risk_level': risk_level,
            'demand_change': round((demand_multiplier - 1.0) * 100, 2)
        }
    
    async def _predict_demand(self, product_id: str, demand_multiplier: float, 
                            market_condition: str, seasonal_factors: Dict[str, float]) -> Dict[str, Any]:
        """Predict demand for a product based on scenario parameters."""
        # Base demand (would come from historical data in real implementation)
        base_demand = 100
        
        # Apply demand multiplier
        adjusted_demand = base_demand * demand_multiplier
        
        # Apply market condition adjustments
        market_adjustments = {
            'growth': 1.2,
            'optimistic': 1.1,
            'stable': 1.0,
            'pessimistic': 0.9,
            'recession': 0.7,
            'volatile': 0.85
        }
        
        market_multiplier = market_adjustments.get(market_condition, 1.0)
        adjusted_demand *= market_multiplier
        
        # Apply seasonal factors
        seasonal_multiplier = 1.0
        if seasonal_factors:
            seasonal_multiplier = max(seasonal_factors.values())
        
        final_demand = adjusted_demand * seasonal_multiplier
        
        return {
            'base_demand': base_demand,
            'demand_multiplier': demand_multiplier,
            'market_multiplier': market_multiplier,
            'seasonal_multiplier': seasonal_multiplier,
            'predicted_demand': round(final_demand, 2),
            'demand_change_percentage': round(((final_demand / base_demand) - 1) * 100, 2)
        }
    
    async def _calculate_inventory_outcome(self, current_inventory: int, predicted_demand: Dict[str, Any], 
                                         time_horizon: str) -> Dict[str, Any]:
        """Calculate inventory outcomes based on predicted demand."""
        demand = predicted_demand['predicted_demand']
        
        # Time horizon multipliers
        time_multipliers = {
            '1_month': 1.0,
            '3_months': 3.0,
            '6_months': 6.0,
            '1_year': 12.0
        }
        
        time_multiplier = time_multipliers.get(time_horizon, 3.0)
        total_demand = demand * time_multiplier
        
        # Calculate inventory outcomes
        inventory_coverage_days = (current_inventory / demand) * 30 if demand > 0 else float('inf')
        stockout_risk = 'low' if current_inventory > total_demand else 'high'
        overstock_risk = 'high' if current_inventory > total_demand * 2 else 'low'
        
        # Calculate reorder recommendations
        optimal_inventory = total_demand * 1.2  # 20% buffer
        reorder_quantity = max(0, optimal_inventory - current_inventory)
        
        return {
            'current_inventory': current_inventory,
            'predicted_demand': demand,
            'total_demand_period': round(total_demand, 2),
            'inventory_coverage_days': round(inventory_coverage_days, 1),
            'stockout_risk': stockout_risk,
            'overstock_risk': overstock_risk,
            'optimal_inventory': round(optimal_inventory, 2),
            'reorder_quantity': round(reorder_quantity, 2)
        }
    
    async def _aggregate_inventory_predictions(self, inventory_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate inventory predictions across all products."""
        if not inventory_predictions:
            return {}
        
        total_current_inventory = sum(pred['current_inventory'] for pred in inventory_predictions.values())
        total_predicted_demand = sum(pred['predicted_demand'] for pred in inventory_predictions.values())
        total_reorder_quantity = sum(pred['reorder_quantity'] for pred in inventory_predictions.values())
        
        # Calculate risk distribution
        stockout_risks = [pred['stockout_risk'] for pred in inventory_predictions.values()]
        overstock_risks = [pred['overstock_risk'] for pred in inventory_predictions.values()]
        
        high_stockout_count = stockout_risks.count('high')
        high_overstock_count = overstock_risks.count('high')
        
        return {
            'total_products': len(inventory_predictions),
            'total_current_inventory': total_current_inventory,
            'total_predicted_demand': round(total_predicted_demand, 2),
            'total_reorder_quantity': round(total_reorder_quantity, 2),
            'high_stockout_risk_products': high_stockout_count,
            'high_overstock_risk_products': high_overstock_count,
            'overall_inventory_health': 'good' if high_stockout_count == 0 and high_overstock_count == 0 else 'needs_attention'
        }
    
    async def _analyze_product_discount_impact(self, product_id: str, discount_percentage: float, 
                                             market_condition: str, time_horizon: str) -> Dict[str, Any]:
        """Analyze discount impact for a specific product."""
        # Price elasticity assumptions (would come from historical data)
        price_elasticity = 1.5  # Simplified assumption
        
        # Calculate demand increase from discount
        demand_increase = discount_percentage * price_elasticity / 100
        
        # Calculate revenue impact
        price_reduction = discount_percentage / 100
        volume_increase = demand_increase
        revenue_multiplier = (1 - price_reduction) * (1 + volume_increase)
        revenue_impact = (revenue_multiplier - 1) * 100
        
        # Market condition adjustments
        market_adjustments = {
            'growth': 1.2,
            'recession': 0.8,
            'stable': 1.0,
            'volatile': 0.9
        }
        
        market_multiplier = market_adjustments.get(market_condition, 1.0)
        adjusted_revenue_impact = revenue_impact * market_multiplier
        
        # Calculate other impacts
        margin_impact = -discount_percentage  # Direct margin reduction
        market_share_impact = discount_percentage * 0.3  # Simplified model
        
        return {
            'product_id': product_id,
            'discount_percentage': discount_percentage,
            'demand_increase_percentage': round(demand_increase * 100, 2),
            'revenue_impact_percentage': round(adjusted_revenue_impact, 2),
            'margin_impact_percentage': round(margin_impact, 2),
            'market_share_impact_percentage': round(market_share_impact, 2),
            'price_elasticity': price_elasticity,
            'market_condition_adjustment': market_multiplier
        }
    
    async def _calculate_overall_discount_impact(self, discount_impacts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall impact across all discounted products."""
        if not discount_impacts:
            return {}
        
        # Aggregate impacts
        total_products = len(discount_impacts)
        avg_discount = sum(impact['discount_percentage'] for impact in discount_impacts.values()) / total_products
        avg_revenue_impact = sum(impact['revenue_impact_percentage'] for impact in discount_impacts.values()) / total_products
        avg_margin_impact = sum(impact['margin_impact_percentage'] for impact in discount_impacts.values()) / total_products
        total_market_share_impact = sum(impact['market_share_impact_percentage'] for impact in discount_impacts.values())
        
        # Determine overall strategy effectiveness
        effectiveness = 'high' if avg_revenue_impact > 5 else 'medium' if avg_revenue_impact > 0 else 'low'
        
        return {
            'total_discounted_products': total_products,
            'average_discount_percentage': round(avg_discount, 2),
            'average_revenue_impact_percentage': round(avg_revenue_impact, 2),
            'average_margin_impact_percentage': round(avg_margin_impact, 2),
            'total_market_share_impact_percentage': round(total_market_share_impact, 2),
            'strategy_effectiveness': effectiveness,
            'break_even_volume_increase': round(avg_discount / (100 - avg_discount) * 100, 2)
        }
    
    async def _calculate_seasonal_factors(self, seasonal_events: List[str], time_horizon: str) -> Dict[str, float]:
        """Calculate seasonal adjustment factors for events."""
        seasonal_multipliers = {
            'diwali': 1.8,
            'christmas': 1.5,
            'holi': 1.3,
            'eid': 1.4,
            'new_year': 1.2,
            'valentine': 1.1,
            'monsoon': 0.8,
            'summer': 0.9,
            'winter': 1.1,
            'back_to_school': 1.3
        }
        
        # Time horizon adjustments
        time_adjustments = {
            '1_month': 1.0,
            '3_months': 0.9,
            '6_months': 0.8,
            '1_year': 0.7
        }
        
        time_adjustment = time_adjustments.get(time_horizon, 0.8)
        
        factors = {}
        for event in seasonal_events:
            base_factor = seasonal_multipliers.get(event, 1.0)
            adjusted_factor = 1.0 + (base_factor - 1.0) * time_adjustment
            factors[event] = round(adjusted_factor, 2)
        
        return factors
    
    async def _summarize_seasonal_impact(self, seasonal_factors: Dict[str, float]) -> Dict[str, Any]:
        """Summarize seasonal impact from factors."""
        if not seasonal_factors:
            return {'impact': 'none', 'summary': 'No seasonal factors applied'}
        
        max_factor = max(seasonal_factors.values())
        min_factor = min(seasonal_factors.values())
        avg_factor = sum(seasonal_factors.values()) / len(seasonal_factors)
        
        # Determine impact level
        if max_factor > 1.5:
            impact_level = 'high'
        elif max_factor > 1.2:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        # Find most impactful events
        max_event = max(seasonal_factors.items(), key=lambda x: x[1])
        
        return {
            'impact': impact_level,
            'max_factor': max_factor,
            'min_factor': min_factor,
            'average_factor': round(avg_factor, 2),
            'most_impactful_event': max_event[0],
            'most_impactful_factor': max_event[1],
            'summary': f"Seasonal impact ranges from {min_factor:.1f}x to {max_factor:.1f}x, with {max_event[0]} having the highest impact"
        }
    
    async def _validate_data_availability(self, products: List[Dict[str, Any]]) -> List[str]:
        """Validate data availability for scenario analysis."""
        limitations = []
        
        if not self.data_processor:
            limitations.append("No data processor available - using simplified assumptions")
            return limitations
        
        try:
            product_ids = [p.get('id', '') for p in products]
            
            # Check if we have demand patterns for products
            patterns = await self.data_processor.extract_demand_patterns(product_ids)
            
            if not patterns:
                limitations.append("No historical demand patterns available - predictions may be less accurate")
            elif len(patterns) < len(product_ids):
                missing_count = len(product_ids) - len(patterns)
                limitations.append(f"Limited historical data for {missing_count} products")
            
            # Check data recency (simplified check)
            for pattern in patterns:
                if pattern.date_range_end < date.today() - timedelta(days=90):
                    limitations.append(f"Historical data for {pattern.product_id} is more than 90 days old")
        
        except Exception as e:
            limitations.append(f"Could not validate historical data: {str(e)}")
        
        return limitations