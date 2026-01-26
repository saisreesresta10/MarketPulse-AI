"""
Risk Assessor component for MarketPulse AI.

This module implements inventory and demand risk assessment functionality
including overstock risk, understock risk, and demand volatility calculations.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
import statistics
import calendar

import numpy as np
import pandas as pd
from scipy import stats

from ..core.interfaces import RiskAssessorInterface
from ..core.models import (
    SalesDataPoint, 
    RiskAssessment, 
    RiskLevel, 
    DemandPattern,
    ConfidenceLevel
)

logger = logging.getLogger(__name__)


class RiskCalculationError(Exception):
    """Raised when risk calculation fails."""
    pass


class InsufficientDataError(Exception):
    """Raised when insufficient data is available for risk assessment."""
    pass


class RiskAssessor(RiskAssessorInterface):
    """
    Implementation of inventory and demand risk assessment.
    
    Provides comprehensive risk analysis including overstock, understock,
    and demand volatility assessments with seasonal adjustments.
    """
    
    def __init__(self, settings=None):
        """Initialize the risk assessor with configuration."""
        self.settings = settings
        self.sales_data: List[SalesDataPoint] = []
        self.demand_patterns: Dict[str, List[DemandPattern]] = {}
        self.risk_cache: Dict[str, RiskAssessment] = {}
        self.storage_manager = None
        
        # Risk calculation parameters
        self.risk_thresholds = {
            'overstock': {
                'low': 1.2,      # 20% above average demand
                'medium': 1.5,   # 50% above average demand
                'high': 2.0,     # 100% above average demand
                'critical': 3.0  # 200% above average demand
            },
            'understock': {
                'low': 0.8,      # 20% below average demand
                'medium': 0.6,   # 40% below average demand
                'high': 0.4,     # 60% below average demand
                'critical': 0.2  # 80% below average demand
            },
            'volatility': {
                'low': 0.2,      # CV < 20%
                'medium': 0.4,   # CV < 40%
                'high': 0.6,     # CV < 60%
                'critical': 0.8  # CV >= 60%
            }
        }
    
    def set_storage_manager(self, storage_manager):
        """
        Set the storage manager for persistent data operations.
        
        Args:
            storage_manager: StorageManager instance for data persistence
        """
        self.storage_manager = storage_manager
        logger.info("Storage manager configured for risk assessor")
    
    def set_sales_data(self, sales_data: List[SalesDataPoint]):
        """
        Set sales data for risk assessment calculations.
        
        Args:
            sales_data: List of sales data points
        """
        self.sales_data = sales_data
        logger.info(f"Sales data set with {len(sales_data)} records")
    
    def set_demand_patterns(self, patterns: Dict[str, List[DemandPattern]]):
        """
        Set demand patterns for enhanced risk assessment.
        
        Args:
            patterns: Dictionary mapping product IDs to demand patterns
        """
        self.demand_patterns = patterns
        logger.info(f"Demand patterns set for {len(patterns)} products")
    
    async def assess_overstock_risk(self, product_id: str, current_inventory: int) -> RiskAssessment:
        """
        Assess overstock risk for a specific product.
        
        Args:
            product_id: Product identifier to assess
            current_inventory: Current inventory level
            
        Returns:
            Risk assessment for overstock scenario
            
        Raises:
            InsufficientDataError: If insufficient data for assessment
            RiskCalculationError: If calculation fails
        """
        try:
            logger.info(f"Assessing overstock risk for product {product_id}")
            
            # Get product sales data
            product_sales = await self._get_product_sales_data(product_id)
            if len(product_sales) < 3:
                raise InsufficientDataError(f"Insufficient sales data for product {product_id}")
            
            # Calculate demand statistics
            demand_stats = await self._calculate_demand_statistics(product_sales)
            
            # Calculate overstock risk metrics
            risk_metrics = await self._calculate_overstock_metrics(
                current_inventory, demand_stats, product_sales
            )
            
            # Determine risk level
            risk_level = await self._determine_overstock_risk_level(risk_metrics)
            
            # Calculate risk score (0-1 scale)
            risk_score = await self._calculate_overstock_risk_score(risk_metrics)
            
            # Identify contributing factors
            contributing_factors = await self._identify_overstock_factors(
                risk_metrics, demand_stats, product_sales
            )
            
            # Generate mitigation suggestions
            mitigation_suggestions = await self._generate_overstock_mitigation_suggestions(
                risk_level, risk_metrics, current_inventory
            )
            
            # Create risk assessment
            assessment = RiskAssessment(
                product_id=product_id,
                risk_type="overstock",
                risk_level=risk_level,
                risk_score=risk_score,
                contributing_factors=contributing_factors,
                mitigation_suggestions=mitigation_suggestions,
                valid_until=date.today() + timedelta(days=30)  # Valid for 30 days
            )
            
            # Cache the assessment
            cache_key = f"overstock_{product_id}_{current_inventory}"
            self.risk_cache[cache_key] = assessment
            
            logger.info(f"Overstock risk assessment complete for {product_id}: {risk_level.value}")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess overstock risk for {product_id}: {e}")
            raise RiskCalculationError(f"Overstock risk assessment failed: {e}")
    
    async def assess_understock_risk(self, product_id: str, current_inventory: int) -> RiskAssessment:
        """
        Assess understock risk for a specific product.
        
        Args:
            product_id: Product identifier to assess
            current_inventory: Current inventory level
            
        Returns:
            Risk assessment for understock scenario
            
        Raises:
            InsufficientDataError: If insufficient data for assessment
            RiskCalculationError: If calculation fails
        """
        try:
            logger.info(f"Assessing understock risk for product {product_id}")
            
            # Get product sales data
            product_sales = await self._get_product_sales_data(product_id)
            if len(product_sales) < 3:
                raise InsufficientDataError(f"Insufficient sales data for product {product_id}")
            
            # Calculate demand statistics
            demand_stats = await self._calculate_demand_statistics(product_sales)
            
            # Calculate understock risk metrics
            risk_metrics = await self._calculate_understock_metrics(
                current_inventory, demand_stats, product_sales
            )
            
            # Determine risk level
            risk_level = await self._determine_understock_risk_level(risk_metrics)
            
            # Calculate risk score (0-1 scale)
            risk_score = await self._calculate_understock_risk_score(risk_metrics)
            
            # Identify contributing factors
            contributing_factors = await self._identify_understock_factors(
                risk_metrics, demand_stats, product_sales
            )
            
            # Generate mitigation suggestions
            mitigation_suggestions = await self._generate_understock_mitigation_suggestions(
                risk_level, risk_metrics, current_inventory
            )
            
            # Create risk assessment
            assessment = RiskAssessment(
                product_id=product_id,
                risk_type="understock",
                risk_level=risk_level,
                risk_score=risk_score,
                contributing_factors=contributing_factors,
                mitigation_suggestions=mitigation_suggestions,
                valid_until=date.today() + timedelta(days=30)
            )
            
            # Cache the assessment
            cache_key = f"understock_{product_id}_{current_inventory}"
            self.risk_cache[cache_key] = assessment
            
            logger.info(f"Understock risk assessment complete for {product_id}: {risk_level.value}")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess understock risk for {product_id}: {e}")
            raise RiskCalculationError(f"Understock risk assessment failed: {e}")
    
    async def calculate_demand_volatility(self, product_id: str) -> float:
        """
        Calculate demand volatility score for a product.
        
        Args:
            product_id: Product identifier to analyze
            
        Returns:
            Volatility score between 0 and 1 (higher = more volatile)
            
        Raises:
            InsufficientDataError: If insufficient data for calculation
            RiskCalculationError: If calculation fails
        """
        try:
            logger.info(f"Calculating demand volatility for product {product_id}")
            
            # Get product sales data
            product_sales = await self._get_product_sales_data(product_id)
            if len(product_sales) < 5:
                raise InsufficientDataError(f"Insufficient sales data for volatility calculation: {product_id}")
            
            # Calculate various volatility measures
            volatility_measures = await self._calculate_volatility_measures(product_sales)
            
            # Combine measures into single volatility score
            volatility_score = await self._combine_volatility_measures(volatility_measures)
            
            logger.info(f"Demand volatility calculated for {product_id}: {volatility_score:.3f}")
            return volatility_score
            
        except Exception as e:
            logger.error(f"Failed to calculate demand volatility for {product_id}: {e}")
            raise RiskCalculationError(f"Volatility calculation failed: {e}")
    
    async def adjust_for_seasonal_events(self, assessment: RiskAssessment, 
                                       upcoming_events: List[str]) -> RiskAssessment:
        """
        Adjust risk assessment for upcoming seasonal events.
        
        Args:
            assessment: Base risk assessment to adjust
            upcoming_events: List of upcoming seasonal events
            
        Returns:
            Adjusted risk assessment
        """
        try:
            logger.info(f"Adjusting risk assessment for seasonal events: {upcoming_events}")
            
            # Get seasonal adjustment factors
            seasonal_adjustments = await self._calculate_seasonal_adjustments(
                assessment.product_id, upcoming_events
            )
            
            # Apply adjustments to risk score
            adjusted_score = await self._apply_seasonal_adjustments(
                assessment.risk_score, seasonal_adjustments, assessment.risk_type
            )
            
            # Recalculate risk level based on adjusted score
            adjusted_level = await self._score_to_risk_level(adjusted_score)
            
            # Update contributing factors
            updated_factors = assessment.contributing_factors.copy()
            for event in upcoming_events:
                if event in seasonal_adjustments:
                    factor_impact = seasonal_adjustments[event]
                    if abs(factor_impact - 1.0) > 0.1:  # Significant impact
                        updated_factors.append(f"Upcoming {event} (impact: {factor_impact:.2f})")
            
            # Create adjusted assessment
            adjusted_assessment = assessment.model_copy()
            adjusted_assessment.risk_score = adjusted_score
            adjusted_assessment.risk_level = adjusted_level
            adjusted_assessment.contributing_factors = updated_factors
            adjusted_assessment.seasonal_adjustments = seasonal_adjustments
            
            logger.info(f"Seasonal adjustment complete. Risk level: {assessment.risk_level.value} -> {adjusted_level.value}")
            return adjusted_assessment
            
        except Exception as e:
            logger.error(f"Failed to adjust for seasonal events: {e}")
            return assessment  # Return original assessment if adjustment fails
    
    async def generate_early_warnings(self, assessments: List[RiskAssessment]) -> List[RiskAssessment]:
        """
        Generate early warning alerts for high-risk situations.
        
        Args:
            assessments: List of risk assessments to evaluate
            
        Returns:
            List of assessments with early warning flags
        """
        try:
            logger.info(f"Generating early warnings for {len(assessments)} assessments")
            
            updated_assessments = []
            
            for assessment in assessments:
                # Check if early warning should be triggered
                should_warn = await self._should_trigger_early_warning(assessment)
                
                if should_warn:
                    # Create updated assessment with warning
                    updated_assessment = assessment.model_copy()
                    updated_assessment.early_warning_triggered = True
                    
                    # Add warning-specific mitigation suggestions
                    warning_suggestions = await self._generate_warning_mitigation_suggestions(assessment)
                    updated_assessment.mitigation_suggestions.extend(warning_suggestions)
                    
                    updated_assessments.append(updated_assessment)
                    logger.warning(f"Early warning triggered for {assessment.product_id}: {assessment.risk_type}")
                else:
                    updated_assessments.append(assessment)
            
            warnings_count = sum(1 for a in updated_assessments if a.early_warning_triggered)
            logger.info(f"Early warning generation complete. {warnings_count} warnings triggered")
            
            return updated_assessments
            
        except Exception as e:
            logger.error(f"Failed to generate early warnings: {e}")
            return assessments  # Return original assessments if warning generation fails
    
    async def assess_seasonal_risk_impact(self, product_id: str, current_inventory: int, 
                                        days_ahead: int = 60) -> Dict[str, Any]:
        """
        Comprehensive seasonal risk impact assessment.
        
        Args:
            product_id: Product identifier to assess
            current_inventory: Current inventory level
            days_ahead: Number of days to look ahead for seasonal events
            
        Returns:
            Comprehensive seasonal risk analysis
        """
        try:
            logger.info(f"Assessing seasonal risk impact for product {product_id}")
            
            # Get base risk assessments
            overstock_assessment = await self.assess_overstock_risk(product_id, current_inventory)
            understock_assessment = await self.assess_understock_risk(product_id, current_inventory)
            
            # Detect upcoming seasonal events
            upcoming_events = await self._detect_upcoming_seasonal_events()
            
            # Apply seasonal adjustments
            adjusted_overstock = await self.adjust_for_seasonal_events(overstock_assessment, upcoming_events)
            adjusted_understock = await self.adjust_for_seasonal_events(understock_assessment, upcoming_events)
            
            # Calculate seasonal impact metrics
            seasonal_impact = await self._calculate_seasonal_impact_metrics(
                product_id, current_inventory, upcoming_events
            )
            
            # Generate seasonal recommendations
            seasonal_recommendations = await self._generate_seasonal_recommendations(
                adjusted_overstock, adjusted_understock, seasonal_impact
            )
            
            # Compile comprehensive analysis
            analysis = {
                'product_id': product_id,
                'current_inventory': current_inventory,
                'assessment_date': date.today().isoformat(),
                'upcoming_events': upcoming_events,
                'base_assessments': {
                    'overstock': overstock_assessment,
                    'understock': understock_assessment
                },
                'seasonal_adjusted_assessments': {
                    'overstock': adjusted_overstock,
                    'understock': adjusted_understock
                },
                'seasonal_impact_metrics': seasonal_impact,
                'seasonal_recommendations': seasonal_recommendations,
                'risk_summary': {
                    'highest_risk_type': 'overstock' if adjusted_overstock.risk_score > adjusted_understock.risk_score else 'understock',
                    'seasonal_adjustment_significant': any(abs(factor - 1.0) > 0.2 for factor in 
                                                         (adjusted_overstock.seasonal_adjustments.values() if adjusted_overstock.seasonal_adjustments else [])),
                    'early_warning_required': adjusted_overstock.early_warning_triggered or adjusted_understock.early_warning_triggered
                }
            }
            
            logger.info(f"Seasonal risk impact assessment complete for {product_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to assess seasonal risk impact for {product_id}: {e}")
            raise RiskCalculationError(f"Seasonal risk impact assessment failed: {e}")
    
    async def _calculate_seasonal_impact_metrics(self, product_id: str, current_inventory: int, 
                                               upcoming_events: List[str]) -> Dict[str, Any]:
        """Calculate detailed seasonal impact metrics."""
        metrics = {}
        
        # Get seasonal adjustments
        seasonal_adjustments = await self._calculate_seasonal_adjustments(product_id, upcoming_events)
        
        if seasonal_adjustments:
            # Calculate impact statistics
            adjustment_values = list(seasonal_adjustments.values())
            metrics['max_demand_multiplier'] = max(adjustment_values)
            metrics['min_demand_multiplier'] = min(adjustment_values)
            metrics['average_demand_multiplier'] = sum(adjustment_values) / len(adjustment_values)
            metrics['seasonal_volatility'] = np.std(adjustment_values) if len(adjustment_values) > 1 else 0.0
            
            # Categorize events by impact
            metrics['high_impact_events'] = [event for event, factor in seasonal_adjustments.items() if factor > 1.5]
            metrics['low_impact_events'] = [event for event, factor in seasonal_adjustments.items() if factor < 0.8]
            metrics['neutral_events'] = [event for event, factor in seasonal_adjustments.items() if 0.8 <= factor <= 1.5]
            
            # Calculate inventory adequacy for seasonal peaks
            if metrics['max_demand_multiplier'] > 1.0:
                # Get product sales data for baseline calculation
                product_sales = await self._get_product_sales_data(product_id)
                if product_sales:
                    demand_stats = await self._calculate_demand_statistics(product_sales)
                    peak_demand_estimate = demand_stats['mean'] * metrics['max_demand_multiplier']
                    metrics['inventory_coverage_at_peak'] = current_inventory / peak_demand_estimate if peak_demand_estimate > 0 else float('inf')
                    metrics['additional_inventory_needed'] = max(0, peak_demand_estimate * 1.2 - current_inventory)  # 20% buffer
                else:
                    metrics['inventory_coverage_at_peak'] = None
                    metrics['additional_inventory_needed'] = None
            else:
                metrics['inventory_coverage_at_peak'] = None
                metrics['additional_inventory_needed'] = None
        else:
            # No seasonal events detected
            metrics = {
                'max_demand_multiplier': 1.0,
                'min_demand_multiplier': 1.0,
                'average_demand_multiplier': 1.0,
                'seasonal_volatility': 0.0,
                'high_impact_events': [],
                'low_impact_events': [],
                'neutral_events': [],
                'inventory_coverage_at_peak': None,
                'additional_inventory_needed': None
            }
        
        return metrics
    
    async def _generate_seasonal_recommendations(self, overstock_assessment: RiskAssessment, 
                                               understock_assessment: RiskAssessment, 
                                               seasonal_metrics: Dict[str, Any]) -> List[str]:
        """Generate comprehensive seasonal recommendations."""
        recommendations = []
        
        # High-level seasonal strategy recommendations
        if seasonal_metrics['high_impact_events']:
            recommendations.append(f"SEASONAL STRATEGY: Prepare for high-demand events: {', '.join(seasonal_metrics['high_impact_events'])}")
            
            if seasonal_metrics['additional_inventory_needed'] and seasonal_metrics['additional_inventory_needed'] > 0:
                recommendations.append(f"INVENTORY ACTION: Consider increasing inventory by {seasonal_metrics['additional_inventory_needed']:.0f} units for seasonal peak")
        
        if seasonal_metrics['low_impact_events']:
            recommendations.append(f"SEASONAL STRATEGY: Plan inventory reduction for low-demand periods: {', '.join(seasonal_metrics['low_impact_events'])}")
        
        # Risk-specific recommendations
        if overstock_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("OVERSTOCK ALERT: Implement pre-seasonal clearance strategy")
            if seasonal_metrics['low_impact_events']:
                recommendations.append("Consider timing promotions with low-demand seasonal periods")
        
        if understock_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("UNDERSTOCK ALERT: Expedite seasonal inventory buildup")
            if seasonal_metrics['high_impact_events']:
                recommendations.append("Prioritize inventory for high-impact seasonal events")
        
        # Seasonal volatility recommendations
        if seasonal_metrics['seasonal_volatility'] > 0.3:
            recommendations.append("HIGH SEASONAL VOLATILITY: Implement flexible inventory management")
            recommendations.append("Consider dynamic safety stock adjustments based on seasonal patterns")
        
        # Coverage recommendations
        if seasonal_metrics['inventory_coverage_at_peak'] is not None:
            if seasonal_metrics['inventory_coverage_at_peak'] < 0.5:
                recommendations.append("CRITICAL: Current inventory insufficient for seasonal peak demand")
            elif seasonal_metrics['inventory_coverage_at_peak'] < 1.0:
                recommendations.append("WARNING: Current inventory may be insufficient for seasonal peak")
        
        # General seasonal management recommendations
        recommendations.append("Implement weekly seasonal risk monitoring")
        recommendations.append("Review and update seasonal demand forecasts regularly")
        
        return recommendations
    
    # Private helper methods
    
    async def _get_product_sales_data(self, product_id: str) -> List[SalesDataPoint]:
        """Get sales data for a specific product."""
        # First try to get from memory
        product_sales = [sale for sale in self.sales_data if sale.product_id == product_id]
        
        # If insufficient data and storage manager available, try to load more
        if len(product_sales) < 10 and self.storage_manager:
            try:
                stored_sales = await self.storage_manager.retrieve_sales_data(
                    product_ids=[product_id],
                    limit=1000
                )
                if stored_sales:
                    product_sales.extend(stored_sales)
                    # Remove duplicates
                    seen_ids = set()
                    unique_sales = []
                    for sale in product_sales:
                        if sale.id not in seen_ids:
                            seen_ids.add(sale.id)
                            unique_sales.append(sale)
                    product_sales = unique_sales
            except Exception as e:
                logger.warning(f"Failed to load additional sales data from storage: {e}")
        
        # Sort by date
        product_sales.sort(key=lambda x: x.sale_date)
        return product_sales
    
    async def _calculate_demand_statistics(self, sales_data: List[SalesDataPoint]) -> Dict[str, float]:
        """Calculate comprehensive demand statistics."""
        quantities = [sale.quantity_sold for sale in sales_data]
        
        stats_dict = {
            'mean': np.mean(quantities),
            'median': np.median(quantities),
            'std': np.std(quantities),
            'min': np.min(quantities),
            'max': np.max(quantities),
            'q25': np.percentile(quantities, 25),
            'q75': np.percentile(quantities, 75),
            'cv': np.std(quantities) / np.mean(quantities) if np.mean(quantities) > 0 else 0,
            'skewness': stats.skew(quantities),
            'kurtosis': stats.kurtosis(quantities)
        }
        
        # Calculate trend
        if len(sales_data) >= 5:
            dates = [(sale.sale_date - sales_data[0].sale_date).days for sale in sales_data]
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates, quantities)
            stats_dict['trend_slope'] = slope
            stats_dict['trend_r_squared'] = r_value ** 2
        else:
            stats_dict['trend_slope'] = 0
            stats_dict['trend_r_squared'] = 0
        
        # Calculate recent vs historical comparison
        if len(sales_data) >= 6:
            recent_data = quantities[-len(quantities)//3:]  # Last third
            historical_data = quantities[:len(quantities)//3]  # First third
            
            stats_dict['recent_mean'] = np.mean(recent_data)
            stats_dict['historical_mean'] = np.mean(historical_data)
            stats_dict['trend_ratio'] = np.mean(recent_data) / np.mean(historical_data) if np.mean(historical_data) > 0 else 1
        else:
            stats_dict['recent_mean'] = stats_dict['mean']
            stats_dict['historical_mean'] = stats_dict['mean']
            stats_dict['trend_ratio'] = 1.0
        
        return stats_dict
    
    async def _calculate_overstock_metrics(self, current_inventory: int, 
                                         demand_stats: Dict[str, float], 
                                         sales_data: List[SalesDataPoint]) -> Dict[str, float]:
        """Calculate overstock-specific risk metrics."""
        metrics = {}
        
        # Days of inventory coverage
        daily_demand = demand_stats['mean'] / 30  # Approximate daily demand
        if daily_demand > 0:
            metrics['days_of_coverage'] = current_inventory / daily_demand
        else:
            metrics['days_of_coverage'] = float('inf')
        
        # Inventory to demand ratio
        metrics['inventory_demand_ratio'] = current_inventory / demand_stats['mean'] if demand_stats['mean'] > 0 else float('inf')
        
        # Excess inventory calculation
        safety_stock = demand_stats['mean'] + 2 * demand_stats['std']  # 2 sigma safety stock
        metrics['excess_inventory'] = max(0, current_inventory - safety_stock)
        metrics['excess_ratio'] = metrics['excess_inventory'] / current_inventory if current_inventory > 0 else 0
        
        # Seasonal adjustment factor
        current_month = date.today().month
        seasonal_sales = [sale.quantity_sold for sale in sales_data if sale.sale_date.month == current_month]
        if seasonal_sales:
            seasonal_mean = np.mean(seasonal_sales)
            metrics['seasonal_adjustment'] = seasonal_mean / demand_stats['mean'] if demand_stats['mean'] > 0 else 1
        else:
            metrics['seasonal_adjustment'] = 1.0
        
        # Trend-adjusted demand
        if demand_stats['trend_slope'] > 0:
            # Increasing trend - higher future demand expected
            metrics['trend_adjusted_demand'] = demand_stats['mean'] * (1 + abs(demand_stats['trend_slope']) * 0.1)
        else:
            # Decreasing trend - lower future demand expected
            metrics['trend_adjusted_demand'] = demand_stats['mean'] * (1 - abs(demand_stats['trend_slope']) * 0.1)
        
        # Volatility impact
        metrics['volatility_multiplier'] = 1 + demand_stats['cv']  # Higher volatility increases overstock risk
        
        return metrics
    
    async def _calculate_understock_metrics(self, current_inventory: int, 
                                          demand_stats: Dict[str, float], 
                                          sales_data: List[SalesDataPoint]) -> Dict[str, float]:
        """Calculate understock-specific risk metrics."""
        metrics = {}
        
        # Days until stockout
        daily_demand = demand_stats['mean'] / 30
        if daily_demand > 0:
            metrics['days_until_stockout'] = current_inventory / daily_demand
        else:
            metrics['days_until_stockout'] = float('inf')
        
        # Service level calculation
        safety_stock = demand_stats['std'] * 1.65  # 95% service level
        metrics['required_safety_stock'] = safety_stock
        metrics['current_service_level'] = min(1.0, current_inventory / (demand_stats['mean'] + safety_stock)) if (demand_stats['mean'] + safety_stock) > 0 else 0
        
        # Stockout probability using normal distribution
        if demand_stats['std'] > 0:
            z_score = (current_inventory - demand_stats['mean']) / demand_stats['std']
            metrics['stockout_probability'] = 1 - stats.norm.cdf(z_score)
        else:
            metrics['stockout_probability'] = 0 if current_inventory >= demand_stats['mean'] else 1
        
        # Demand surge risk
        max_demand = demand_stats['max']
        metrics['surge_coverage'] = current_inventory / max_demand if max_demand > 0 else float('inf')
        
        # Seasonal adjustment
        current_month = date.today().month
        seasonal_sales = [sale.quantity_sold for sale in sales_data if sale.sale_date.month == current_month]
        if seasonal_sales:
            seasonal_mean = np.mean(seasonal_sales)
            metrics['seasonal_demand_multiplier'] = seasonal_mean / demand_stats['mean'] if demand_stats['mean'] > 0 else 1
        else:
            metrics['seasonal_demand_multiplier'] = 1.0
        
        # Trend impact
        if demand_stats['trend_slope'] > 0:
            # Increasing trend increases understock risk
            metrics['trend_risk_multiplier'] = 1 + abs(demand_stats['trend_slope']) * 0.2
        else:
            # Decreasing trend reduces understock risk
            metrics['trend_risk_multiplier'] = max(0.5, 1 - abs(demand_stats['trend_slope']) * 0.2)
        
        return metrics
    
    async def _determine_overstock_risk_level(self, metrics: Dict[str, float]) -> RiskLevel:
        """Determine overstock risk level based on metrics."""
        # Primary indicator: inventory to demand ratio
        inventory_ratio = metrics.get('inventory_demand_ratio', 1.0)
        
        # Adjust for excess inventory
        excess_ratio = metrics.get('excess_ratio', 0.0)
        
        # Adjust for seasonal factors
        seasonal_adj = metrics.get('seasonal_adjustment', 1.0)
        adjusted_ratio = inventory_ratio / seasonal_adj
        
        # Determine base risk level
        if adjusted_ratio >= self.risk_thresholds['overstock']['critical']:
            base_level = RiskLevel.CRITICAL
        elif adjusted_ratio >= self.risk_thresholds['overstock']['high']:
            base_level = RiskLevel.HIGH
        elif adjusted_ratio >= self.risk_thresholds['overstock']['medium']:
            base_level = RiskLevel.MEDIUM
        else:
            base_level = RiskLevel.LOW
        
        # Escalate if high excess inventory
        if excess_ratio > 0.5 and base_level == RiskLevel.MEDIUM:
            base_level = RiskLevel.HIGH
        elif excess_ratio > 0.3 and base_level == RiskLevel.LOW:
            base_level = RiskLevel.MEDIUM
        
        return base_level
    
    async def _determine_understock_risk_level(self, metrics: Dict[str, float]) -> RiskLevel:
        """Determine understock risk level based on metrics."""
        # Primary indicators
        stockout_prob = metrics.get('stockout_probability', 0.0)
        service_level = metrics.get('current_service_level', 1.0)
        days_until_stockout = metrics.get('days_until_stockout', float('inf'))
        
        # Adjust for seasonal and trend factors
        seasonal_multiplier = metrics.get('seasonal_demand_multiplier', 1.0)
        trend_multiplier = metrics.get('trend_risk_multiplier', 1.0)
        
        # Calculate composite risk score
        risk_score = stockout_prob * seasonal_multiplier * trend_multiplier
        
        # Determine risk level
        if risk_score >= 0.3 or service_level < 0.5 or days_until_stockout < 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.15 or service_level < 0.7 or days_until_stockout < 14:
            return RiskLevel.HIGH
        elif risk_score >= 0.05 or service_level < 0.85 or days_until_stockout < 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _calculate_overstock_risk_score(self, metrics: Dict[str, float]) -> float:
        """Calculate normalized overstock risk score (0-1)."""
        # Normalize inventory ratio to 0-1 scale
        inventory_ratio = metrics.get('inventory_demand_ratio', 1.0)
        ratio_score = min(1.0, (inventory_ratio - 1.0) / 2.0)  # Normalize around 1.0, cap at 1.0
        
        # Excess inventory component
        excess_score = metrics.get('excess_ratio', 0.0)
        
        # Days of coverage component (normalize around 90 days)
        days_coverage = metrics.get('days_of_coverage', 30)
        coverage_score = min(1.0, max(0.0, (days_coverage - 30) / 120))  # 30-150 days range
        
        # Volatility component
        volatility_mult = metrics.get('volatility_multiplier', 1.0)
        volatility_score = min(1.0, (volatility_mult - 1.0) / 2.0)
        
        # Weighted combination
        risk_score = (
            ratio_score * 0.4 +
            excess_score * 0.3 +
            coverage_score * 0.2 +
            volatility_score * 0.1
        )
        
        return min(1.0, max(0.0, risk_score))
    
    async def _calculate_understock_risk_score(self, metrics: Dict[str, float]) -> float:
        """Calculate normalized understock risk score (0-1)."""
        # Stockout probability component
        stockout_prob = metrics.get('stockout_probability', 0.0)
        
        # Service level component (inverted - lower service level = higher risk)
        service_level = metrics.get('current_service_level', 1.0)
        service_score = 1.0 - service_level
        
        # Days until stockout component
        days_until_stockout = metrics.get('days_until_stockout', float('inf'))
        if days_until_stockout == float('inf'):
            days_score = 0.0
        else:
            days_score = max(0.0, 1.0 - days_until_stockout / 60)  # 60 days = no risk
        
        # Seasonal and trend adjustments
        seasonal_mult = metrics.get('seasonal_demand_multiplier', 1.0)
        trend_mult = metrics.get('trend_risk_multiplier', 1.0)
        adjustment_factor = min(2.0, seasonal_mult * trend_mult)
        
        # Weighted combination
        base_score = (
            stockout_prob * 0.4 +
            service_score * 0.3 +
            days_score * 0.3
        )
        
        # Apply adjustments
        risk_score = base_score * adjustment_factor
        
        return min(1.0, max(0.0, risk_score))
    
    async def _identify_overstock_factors(self, metrics: Dict[str, float], 
                                        demand_stats: Dict[str, float], 
                                        sales_data: List[SalesDataPoint]) -> List[str]:
        """Identify factors contributing to overstock risk."""
        factors = []
        
        # High inventory levels
        inventory_ratio = metrics.get('inventory_demand_ratio', 1.0)
        if inventory_ratio > 2.0:
            factors.append(f"Inventory level {inventory_ratio:.1f}x above average demand")
        elif inventory_ratio > 1.5:
            factors.append(f"Inventory level {inventory_ratio:.1f}x above average demand")
        
        # Excess inventory
        excess_ratio = metrics.get('excess_ratio', 0.0)
        if excess_ratio > 0.3:
            factors.append(f"Excess inventory: {excess_ratio:.1%} above safety stock")
        elif excess_ratio > 0.1:
            factors.append(f"Moderate excess inventory: {excess_ratio:.1%} above safety stock")
        
        # Declining demand trend
        if demand_stats.get('trend_slope', 0) < -0.1:
            factors.append("Declining demand trend detected")
        elif demand_stats.get('trend_slope', 0) < -0.05:
            factors.append("Slight declining demand trend detected")
        
        # High demand volatility
        if demand_stats.get('cv', 0) > 0.5:
            factors.append(f"High demand volatility (CV: {demand_stats['cv']:.1%})")
        elif demand_stats.get('cv', 0) > 0.3:
            factors.append(f"Moderate demand volatility (CV: {demand_stats['cv']:.1%})")
        
        # Seasonal factors
        seasonal_adj = metrics.get('seasonal_adjustment', 1.0)
        if seasonal_adj < 0.8:
            factors.append("Currently in low-demand season")
        elif seasonal_adj < 0.9:
            factors.append("Currently in below-average demand period")
        
        # Long coverage period
        days_coverage = metrics.get('days_of_coverage', 30)
        if days_coverage > 120:
            factors.append(f"Inventory covers {days_coverage:.0f} days of demand")
        elif days_coverage > 60:
            factors.append(f"Inventory covers {days_coverage:.0f} days of demand")
        
        # Ensure we always have at least one factor
        if not factors:
            if inventory_ratio > 1.0:
                factors.append(f"Current inventory level is {inventory_ratio:.1f}x average demand")
            else:
                factors.append("Low overstock risk - inventory levels are appropriate")
        
        return factors
    
    async def _identify_understock_factors(self, metrics: Dict[str, float], 
                                         demand_stats: Dict[str, float], 
                                         sales_data: List[SalesDataPoint]) -> List[str]:
        """Identify factors contributing to understock risk."""
        factors = []
        
        # Low inventory levels
        service_level = metrics.get('current_service_level', 1.0)
        if service_level < 0.8:
            factors.append(f"Low service level: {service_level:.1%}")
        elif service_level < 0.9:
            factors.append(f"Below-target service level: {service_level:.1%}")
        
        # High stockout probability
        stockout_prob = metrics.get('stockout_probability', 0.0)
        if stockout_prob > 0.2:
            factors.append(f"High stockout probability: {stockout_prob:.1%}")
        elif stockout_prob > 0.1:
            factors.append(f"Moderate stockout probability: {stockout_prob:.1%}")
        
        # Increasing demand trend
        if demand_stats.get('trend_slope', 0) > 0.1:
            factors.append("Increasing demand trend detected")
        elif demand_stats.get('trend_slope', 0) > 0.05:
            factors.append("Slight increasing demand trend detected")
        
        # Seasonal demand increase
        seasonal_mult = metrics.get('seasonal_demand_multiplier', 1.0)
        if seasonal_mult > 1.2:
            factors.append("Currently in high-demand season")
        elif seasonal_mult > 1.1:
            factors.append("Currently in above-average demand period")
        
        # Short time to stockout
        days_until_stockout = metrics.get('days_until_stockout', float('inf'))
        if days_until_stockout < 30:
            factors.append(f"Potential stockout in {days_until_stockout:.0f} days")
        elif days_until_stockout < 60:
            factors.append(f"Limited inventory coverage: {days_until_stockout:.0f} days")
        
        # High demand volatility
        if demand_stats.get('cv', 0) > 0.4:
            factors.append(f"High demand volatility increases stockout risk")
        elif demand_stats.get('cv', 0) > 0.3:
            factors.append(f"Moderate demand volatility affects inventory planning")
        
        # Recent demand surge
        if demand_stats.get('trend_ratio', 1.0) > 1.3:
            factors.append("Recent demand increase detected")
        elif demand_stats.get('trend_ratio', 1.0) > 1.2:
            factors.append("Recent uptick in demand observed")
        
        # Ensure we always have at least one factor
        if not factors:
            if service_level < 1.0:
                factors.append(f"Current service level is {service_level:.1%}")
            else:
                factors.append("Low understock risk - inventory levels are adequate")
        
        return factors
    
    async def _generate_overstock_mitigation_suggestions(self, risk_level: RiskLevel, 
                                                       metrics: Dict[str, float], 
                                                       current_inventory: int) -> List[str]:
        """Generate mitigation suggestions for overstock risk."""
        suggestions = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            suggestions.append("Consider implementing promotional pricing to accelerate sales")
            suggestions.append("Evaluate discount strategies within MRP compliance limits")
            
            excess_inventory = metrics.get('excess_inventory', 0)
            if excess_inventory > 0:
                suggestions.append(f"Reduce inventory by approximately {excess_inventory:.0f} units")
        
        if risk_level == RiskLevel.CRITICAL:
            suggestions.append("Implement emergency clearance strategies")
            suggestions.append("Consider alternative sales channels or bulk sales")
        
        # Seasonal suggestions
        seasonal_adj = metrics.get('seasonal_adjustment', 1.0)
        if seasonal_adj < 0.8:
            suggestions.append("Plan inventory reduction before seasonal demand recovery")
        
        # General suggestions
        suggestions.append("Review demand forecasting accuracy")
        suggestions.append("Implement dynamic inventory management")
        
        if metrics.get('volatility_multiplier', 1.0) > 1.5:
            suggestions.append("Implement safety stock optimization for high volatility")
        
        return suggestions
    
    async def _generate_understock_mitigation_suggestions(self, risk_level: RiskLevel, 
                                                        metrics: Dict[str, float], 
                                                        current_inventory: int) -> List[str]:
        """Generate mitigation suggestions for understock risk."""
        suggestions = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            suggestions.append("Expedite inventory replenishment orders")
            
            required_safety = metrics.get('required_safety_stock', 0)
            suggestions.append(f"Increase inventory to maintain {required_safety:.0f} units safety stock")
        
        if risk_level == RiskLevel.CRITICAL:
            suggestions.append("Implement emergency procurement procedures")
            suggestions.append("Consider alternative suppliers for faster delivery")
        
        # Seasonal suggestions
        seasonal_mult = metrics.get('seasonal_demand_multiplier', 1.0)
        if seasonal_mult > 1.2:
            suggestions.append("Increase inventory levels for seasonal demand surge")
        
        # Trend-based suggestions
        if metrics.get('trend_risk_multiplier', 1.0) > 1.2:
            suggestions.append("Adjust reorder points for increasing demand trend")
        
        # Service level suggestions
        service_level = metrics.get('current_service_level', 1.0)
        if service_level < 0.9:
            target_inventory = current_inventory / service_level * 0.95  # Target 95% service level
            additional_needed = target_inventory - current_inventory
            suggestions.append(f"Increase inventory by {additional_needed:.0f} units to achieve 95% service level")
        
        # General suggestions
        suggestions.append("Review and optimize reorder points")
        suggestions.append("Implement demand sensing for early warning")
        
        return suggestions
    
    async def _calculate_volatility_measures(self, sales_data: List[SalesDataPoint]) -> Dict[str, float]:
        """Calculate various volatility measures."""
        quantities = [sale.quantity_sold for sale in sales_data]
        
        measures = {}
        
        # Coefficient of variation
        mean_qty = np.mean(quantities)
        std_qty = np.std(quantities)
        measures['cv'] = std_qty / mean_qty if mean_qty > 0 else 0
        
        # Mean absolute deviation
        measures['mad'] = np.mean([abs(q - mean_qty) for q in quantities])
        measures['mad_normalized'] = measures['mad'] / mean_qty if mean_qty > 0 else 0
        
        # Interquartile range
        q75 = np.percentile(quantities, 75)
        q25 = np.percentile(quantities, 25)
        measures['iqr'] = q75 - q25
        measures['iqr_normalized'] = measures['iqr'] / mean_qty if mean_qty > 0 else 0
        
        # Range-based volatility
        max_qty = np.max(quantities)
        min_qty = np.min(quantities)
        measures['range'] = max_qty - min_qty
        measures['range_normalized'] = measures['range'] / mean_qty if mean_qty > 0 else 0
        
        # Rolling volatility (if sufficient data)
        if len(quantities) >= 10:
            window_size = min(7, len(quantities) // 3)
            rolling_stds = []
            for i in range(window_size, len(quantities)):
                window_data = quantities[i-window_size:i]
                rolling_stds.append(np.std(window_data))
            
            measures['rolling_volatility'] = np.mean(rolling_stds)
            measures['rolling_volatility_normalized'] = measures['rolling_volatility'] / mean_qty if mean_qty > 0 else 0
        else:
            measures['rolling_volatility'] = std_qty
            measures['rolling_volatility_normalized'] = measures['cv']
        
        # Seasonal volatility
        monthly_data = {}
        for sale in sales_data:
            month = sale.sale_date.month
            if month not in monthly_data:
                monthly_data[month] = []
            monthly_data[month].append(sale.quantity_sold)
        
        if len(monthly_data) >= 3:
            monthly_means = [np.mean(data) for data in monthly_data.values()]
            measures['seasonal_volatility'] = np.std(monthly_means) / np.mean(monthly_means) if np.mean(monthly_means) > 0 else 0
        else:
            measures['seasonal_volatility'] = measures['cv']
        
        return measures
    
    async def _combine_volatility_measures(self, measures: Dict[str, float]) -> float:
        """Combine multiple volatility measures into single score."""
        # Weight different measures
        weights = {
            'cv': 0.3,
            'mad_normalized': 0.2,
            'iqr_normalized': 0.2,
            'rolling_volatility_normalized': 0.2,
            'seasonal_volatility': 0.1
        }
        
        # Calculate weighted average
        volatility_score = 0.0
        total_weight = 0.0
        
        for measure, weight in weights.items():
            if measure in measures:
                volatility_score += measures[measure] * weight
                total_weight += weight
        
        # Normalize to 0-1 scale
        if total_weight > 0:
            volatility_score = volatility_score / total_weight
        
        # Cap at 1.0 and apply non-linear scaling for better distribution
        volatility_score = min(1.0, volatility_score)
        
        # Apply sigmoid-like transformation for better scaling
        if volatility_score > 0:
            volatility_score = 2 / (1 + np.exp(-5 * volatility_score)) - 1
        
        return max(0.0, min(1.0, volatility_score))
    
    async def _calculate_seasonal_adjustments(self, product_id: str, 
                                            upcoming_events: List[str]) -> Dict[str, float]:
        """Calculate seasonal adjustment factors for upcoming events."""
        adjustments = {}
        
        # Get product demand patterns if available
        product_patterns = self.demand_patterns.get(product_id, [])
        
        # Enhanced seasonal factors with time-based adjustments
        default_factors = {
            'diwali': 1.8,
            'holi': 1.4,
            'eid': 1.6,
            'christmas': 1.3,
            'durga_puja': 1.6,
            'navratri': 1.4,
            'ganesh_chaturthi': 1.5,
            'summer': 0.85,
            'monsoon': 0.75,
            'winter': 1.1,
            'wedding_season': 1.7,
            'back_to_school': 1.4,
            'new_year': 1.2,
            'valentine_day': 1.3,
            'mother_day': 1.2,
            'father_day': 1.1,
            'raksha_bandhan': 1.3,
            'karva_chauth': 1.2,
            'dhanteras': 1.5,
            'bhai_dooj': 1.2
        }
        
        # Only auto-detect events if no specific events are provided
        if not upcoming_events:
            auto_detected_events = await self._detect_upcoming_seasonal_events()
            all_events = auto_detected_events
        else:
            # Use only the explicitly provided events
            all_events = upcoming_events
        
        for event in all_events:
            # Check if we have pattern-specific data
            pattern_factor = None
            for pattern in product_patterns:
                if hasattr(pattern, 'seasonal_factors') and event in pattern.seasonal_factors:
                    pattern_factor = pattern.seasonal_factors[event]
                    break
            
            # Use pattern factor if available, otherwise default
            if pattern_factor is not None:
                base_factor = pattern_factor
            else:
                # Try to match event with default factors
                base_factor = None
                for default_event, factor in default_factors.items():
                    if default_event.lower() in event.lower():
                        base_factor = factor
                        break
                
                if base_factor is None:
                    # No match found, use neutral factor
                    base_factor = 1.0
            
            # Apply time-based adjustment (closer events have higher impact)
            time_adjusted_factor = await self._apply_time_based_adjustment(event, base_factor)
            adjustments[event] = time_adjusted_factor
        
        return adjustments
    
    async def _detect_upcoming_seasonal_events(self) -> List[str]:
        """Automatically detect upcoming seasonal events based on current date."""
        current_date = date.today()
        upcoming_events = []
        
        # Define seasonal events with their typical dates (month, day ranges)
        seasonal_calendar = {
            'summer': [(4, 1, 6, 30)],  # April to June
            'monsoon': [(7, 1, 9, 30)],  # July to September
            'winter': [(11, 1, 2, 28)],  # November to February
            'diwali': [(10, 15, 11, 15)],  # Mid October to Mid November (varies)
            'holi': [(2, 20, 3, 20)],  # Late February to Mid March (varies)
            'christmas': [(12, 15, 12, 31)],  # Mid to End December
            'new_year': [(12, 25, 1, 7)],  # Late December to Early January
            'durga_puja': [(9, 15, 10, 15)],  # Mid September to Mid October
            'navratri': [(9, 15, 10, 15)],  # Similar to Durga Puja
            'ganesh_chaturthi': [(8, 15, 9, 15)],  # Mid August to Mid September
            'eid': [(5, 1, 6, 30), (7, 1, 8, 31)],  # Varies, approximate ranges
            'raksha_bandhan': [(7, 15, 8, 31)],  # Mid July to End August
            'karva_chauth': [(10, 15, 11, 15)],  # Mid October to Mid November
            'dhanteras': [(10, 10, 11, 10)],  # Mid October to Early November
            'wedding_season': [(11, 1, 2, 28), (4, 1, 6, 30)],  # Winter and Spring
            'back_to_school': [(6, 1, 7, 31)],  # June to July
            'valentine_day': [(2, 1, 2, 20)],  # Early to Mid February
            'mother_day': [(5, 1, 5, 15)],  # Early May
            'father_day': [(6, 15, 6, 30)]  # Mid June
        }
        
        # Check for events in the next 60 days
        for days_ahead in range(1, 61):
            check_date = current_date + timedelta(days=days_ahead)
            
            for event_name, date_ranges in seasonal_calendar.items():
                for date_range in date_ranges:
                    start_month, start_day, end_month, end_day = date_range
                    
                    # Handle year boundary crossing
                    if start_month > end_month:  # Crosses year boundary
                        if ((check_date.month == start_month and check_date.day >= start_day) or
                            (check_date.month == end_month and check_date.day <= end_day) or
                            (start_month < check_date.month < 12) or
                            (1 <= check_date.month < end_month)):
                            if event_name not in upcoming_events:
                                upcoming_events.append(event_name)
                    else:  # Same year
                        if ((check_date.month == start_month and check_date.day >= start_day) or
                            (check_date.month == end_month and check_date.day <= end_day) or
                            (start_month < check_date.month < end_month)):
                            if event_name not in upcoming_events:
                                upcoming_events.append(event_name)
        
        return upcoming_events
    
    async def _apply_time_based_adjustment(self, event: str, base_factor: float) -> float:
        """Apply time-based adjustment to seasonal factors based on proximity."""
        # For now, return base factor - can be enhanced with specific event timing
        # In a real implementation, this would calculate days until event and adjust accordingly
        
        # Simple proximity adjustment - events closer in time have higher impact
        current_date = date.today()
        
        # Estimate event proximity (simplified - in real implementation would use actual dates)
        proximity_multiplier = 1.0
        
        # High-impact events get stronger proximity effects
        high_impact_events = ['diwali', 'christmas', 'eid', 'wedding_season']
        if any(high_event in event.lower() for high_event in high_impact_events):
            proximity_multiplier = 1.1  # 10% boost for high-impact events
        
        return base_factor * proximity_multiplier
    
    async def _apply_seasonal_adjustments(self, base_score: float, 
                                        adjustments: Dict[str, float], 
                                        risk_type: str) -> float:
        """Apply seasonal adjustments to risk score."""
        if not adjustments:
            return base_score
        
        # Calculate combined adjustment factor more conservatively
        combined_factor = 1.0
        for event, factor in adjustments.items():
            if risk_type == "overstock":
                # For overstock: lower seasonal demand increases risk, but more conservatively
                if factor < 1.0:
                    # Low demand season - increases overstock risk
                    # Use a more conservative adjustment: 0.85 becomes 1.18, 0.7 becomes 1.43
                    risk_multiplier = 1.0 + (1.0 - factor) * 0.5  # Dampen the effect
                    combined_factor *= min(risk_multiplier, 1.5)  # Cap at 50% increase
                else:
                    # High demand season - reduces overstock risk
                    # Use conservative reduction: 1.5 becomes 0.83, 2.0 becomes 0.75
                    risk_reducer = 1.0 / (1.0 + (factor - 1.0) * 0.5)  # Dampen the effect
                    combined_factor *= max(risk_reducer, 0.7)  # Cap at 30% reduction
            else:  # understock
                # For understock: higher seasonal demand increases risk
                # Apply factor more conservatively
                if factor > 1.0:
                    # High demand season - increase understock risk
                    risk_multiplier = 1.0 + (factor - 1.0) * 0.7  # Dampen the effect
                    combined_factor *= min(risk_multiplier, 2.0)  # Cap at 100% increase
                else:
                    # Low demand season - reduce understock risk
                    combined_factor *= max(factor, 0.5)  # Cap at 50% reduction
        
        # Apply adjustment
        adjusted_score = base_score * combined_factor
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, adjusted_score))
    
    async def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level."""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _should_trigger_early_warning(self, assessment: RiskAssessment) -> bool:
        """Determine if early warning should be triggered."""
        # Trigger for high and critical risk levels
        if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return True
        
        # Trigger for medium risk with high score
        if assessment.risk_level == RiskLevel.MEDIUM and assessment.risk_score > 0.5:
            return True
        
        # Enhanced seasonal-based early warning triggers
        if assessment.seasonal_adjustments:
            # Check for significant seasonal impact
            max_seasonal_impact = max(assessment.seasonal_adjustments.values())
            min_seasonal_impact = min(assessment.seasonal_adjustments.values())
            
            # Trigger warning if seasonal events significantly increase risk
            if assessment.risk_type == "understock" and max_seasonal_impact > 1.5:
                # High-demand season approaching with medium+ risk
                if assessment.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                    return True
            
            elif assessment.risk_type == "overstock" and min_seasonal_impact < 0.8:
                # Low-demand season approaching with medium+ risk
                if assessment.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                    return True
        
        # Trigger for specific risk types and conditions
        if assessment.risk_type == "understock":
            # Check for specific understock warning conditions
            for factor in assessment.contributing_factors:
                if "stockout in" in factor.lower() and any(word in factor for word in ["7", "14"]):
                    return True
                # New: Check for seasonal demand surge warnings
                if "high-demand season" in factor.lower() or "seasonal demand" in factor.lower():
                    return True
        
        # Check for seasonal event proximity warnings
        for factor in assessment.contributing_factors:
            if "upcoming" in factor.lower() and any(event in factor.lower() for event in 
                ['diwali', 'christmas', 'eid', 'wedding_season', 'holi']):
                # Major seasonal event approaching
                if assessment.risk_score > 0.4:  # Lower threshold for seasonal events
                    return True
        
        return False
    
    async def _generate_warning_mitigation_suggestions(self, assessment: RiskAssessment) -> List[str]:
        """Generate warning-specific mitigation suggestions."""
        suggestions = []
        
        if assessment.risk_type == "overstock":
            suggestions.append("URGENT: Implement immediate promotional strategies")
            suggestions.append("Consider emergency discount programs within MRP limits")
            
            # Seasonal-specific overstock suggestions
            if assessment.seasonal_adjustments:
                low_demand_events = [event for event, factor in assessment.seasonal_adjustments.items() 
                                   if factor < 0.9]
                if low_demand_events:
                    suggestions.append(f"SEASONAL ALERT: Prepare for reduced demand during {', '.join(low_demand_events)}")
                    suggestions.append("Consider pre-seasonal clearance sales to reduce inventory")
                    
        elif assessment.risk_type == "understock":
            suggestions.append("URGENT: Expedite emergency procurement")
            suggestions.append("Activate backup suppliers immediately")
            
            # Seasonal-specific understock suggestions
            if assessment.seasonal_adjustments:
                high_demand_events = [event for event, factor in assessment.seasonal_adjustments.items() 
                                    if factor > 1.2]
                if high_demand_events:
                    suggestions.append(f"SEASONAL ALERT: Prepare for increased demand during {', '.join(high_demand_events)}")
                    suggestions.append("Consider emergency seasonal inventory buildup")
                    suggestions.append("Negotiate expedited delivery terms with suppliers")
                    
                    # Specific suggestions for major festivals
                    major_festivals = ['diwali', 'christmas', 'eid', 'durga_puja']
                    if any(festival in event.lower() for event in high_demand_events for festival in major_festivals):
                        suggestions.append("FESTIVAL ALERT: Implement festival-specific procurement strategy")
                        suggestions.append("Consider alternative product sourcing for festival demand")
        
        # General seasonal warnings
        if assessment.seasonal_adjustments:
            suggestions.append("Implement enhanced seasonal monitoring")
            suggestions.append("Review seasonal demand forecasts weekly")
        
        suggestions.append("Notify management of high-risk situation")
        suggestions.append("Implement daily monitoring until risk subsides")
        
        return suggestions