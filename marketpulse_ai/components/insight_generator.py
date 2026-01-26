"""
Insight Generator component for MarketPulse AI.

This module implements natural language insight generation from demand patterns,
creating human-readable explanations with supporting evidence and confidence levels.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
import statistics
import re

import numpy as np

from ..core.interfaces import InsightGeneratorInterface
from ..core.models import (
    DemandPattern, 
    ExplainableInsight, 
    ConfidenceLevel,
    SalesDataPoint
)

logger = logging.getLogger(__name__)


class InsightGenerationError(Exception):
    """Raised when insight generation fails."""
    pass


class InsightGenerator(InsightGeneratorInterface):
    """
    Implementation of natural language insight generation.
    
    Converts demand patterns into human-readable insights with supporting
    evidence, confidence levels, and business-friendly explanations.
    """
    
    def __init__(self, settings=None):
        """Initialize the insight generator with configuration."""
        self.settings = settings
        self.sales_data: List[SalesDataPoint] = []
        self.insight_cache: Dict[str, ExplainableInsight] = {}
        self.storage_manager = None
        
        # Business language templates
        self.pattern_templates = self._load_pattern_templates()
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Seasonal event mappings for business context
        self.seasonal_events = {
            'diwali': 'Diwali festival season',
            'holi': 'Holi celebration period',
            'christmas': 'Christmas holiday season',
            'eid_ul_fitr': 'Eid ul-Fitr festivities',
            'durga_puja': 'Durga Puja celebrations',
            'ganesh_chaturthi': 'Ganesh Chaturthi festival',
            'navratri': 'Navratri festival period',
            'wedding_season': 'Indian wedding season',
            'back_to_school': 'back-to-school period',
            'summer': 'summer season',
            'monsoon': 'monsoon season',
            'winter': 'winter season'
        }
    
    def set_sales_data(self, sales_data: List[SalesDataPoint]):
        """
        Set sales data for insight generation context.
        
        Args:
            sales_data: List of sales data points
        """
        self.sales_data = sales_data
        logger.info(f"Sales data set with {len(sales_data)} records for insight generation")
    
    def set_storage_manager(self, storage_manager):
        """
        Set the storage manager for persistent operations.
        
        Args:
            storage_manager: StorageManager instance
        """
        self.storage_manager = storage_manager
        logger.info("Storage manager configured for insight generator")
    
    def _load_pattern_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load business-friendly language templates for different pattern types.
        
        Returns:
            Dictionary of pattern templates
        """
        return {
            'seasonal': {
                'title_template': "{product_name} shows {strength} seasonal demand pattern",
                'description_template': "Sales of {product_name} demonstrate a {strength} seasonal pattern with {peak_description} during {peak_periods}. {trend_description}",
                'evidence_template': "Based on analysis of {data_points} sales records over {time_period}",
                'recommendation_template': "Consider {seasonal_strategy} to optimize inventory and pricing during peak seasons"
            },
            'trending': {
                'title_template': "{product_name} exhibits {direction} demand trend",
                'description_template': "Demand for {product_name} has been {direction} {trend_strength} over the analyzed period. {volatility_description}",
                'evidence_template': "Trend analysis based on {data_points} sales transactions showing {trend_percentage} change",
                'recommendation_template': "Adjust inventory planning to account for {direction} demand trajectory"
            },
            'cyclical': {
                'title_template': "{product_name} demonstrates {cycle_type} demand cycles",
                'description_template': "Sales patterns reveal {cycle_type} demand cycles with {amplitude_description}. {consistency_description}",
                'evidence_template': "Cyclical analysis identified {cycle_count} distinct patterns in {data_points} records",
                'recommendation_template': "Leverage cyclical patterns for strategic inventory and promotional planning"
            },
            'volatile': {
                'title_template': "{product_name} shows {volatility_level} demand volatility",
                'description_template': "Demand variability for {product_name} is {volatility_level}, indicating {business_implication}. {risk_description}",
                'evidence_template': "Volatility analysis based on coefficient of variation of {cv_value} across {data_points} records",
                'recommendation_template': "Implement {volatility_strategy} to manage demand uncertainty"
            },
            'stable': {
                'title_template': "{product_name} maintains stable demand pattern",
                'description_template': "Demand for {product_name} remains consistently stable with {consistency_description}. {predictability_description}",
                'evidence_template': "Stability confirmed through low variance analysis of {data_points} sales records",
                'recommendation_template': "Maintain steady inventory levels with minimal safety stock adjustments"
            }
        }
    
    async def generate_insights(self, patterns: List[DemandPattern]) -> List[ExplainableInsight]:
        """
        Generate explainable insights from demand patterns.
        
        Args:
            patterns: List of demand patterns to analyze
            
        Returns:
            List of generated insights with explanations
        """
        try:
            logger.info(f"Generating insights from {len(patterns)} demand patterns")
            
            if not patterns:
                logger.warning("No patterns provided for insight generation")
                return []
            
            insights = []
            
            for pattern in patterns:
                try:
                    # Generate individual insight for each pattern
                    insight = await self.explain_pattern(pattern)
                    
                    # Calculate and update confidence
                    confidence_score = await self.calculate_confidence(insight)
                    
                    # Update insight with calculated confidence
                    updated_insight = insight.model_copy()
                    updated_insight.confidence_level = self._score_to_confidence_level(confidence_score)
                    
                    insights.append(updated_insight)
                    
                    # Cache the insight
                    cache_key = f"{pattern.product_id}_{pattern.pattern_type}_{pattern.id}"
                    self.insight_cache[cache_key] = updated_insight
                    
                except Exception as e:
                    logger.error(f"Failed to generate insight for pattern {pattern.id}: {e}")
                    continue
            
            # Sort insights by confidence and business impact
            sorted_insights = await self._prioritize_insights(insights)
            
            logger.info(f"Successfully generated {len(sorted_insights)} insights")
            return sorted_insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            raise InsightGenerationError(f"Insight generation failed: {e}")
    
    async def explain_pattern(self, pattern: DemandPattern) -> ExplainableInsight:
        """
        Create detailed explanation for a specific demand pattern.
        
        Args:
            pattern: Demand pattern to explain
            
        Returns:
            Detailed insight with explanation and evidence
        """
        try:
            logger.debug(f"Explaining pattern {pattern.id} for product {pattern.product_id}")
            
            # Determine pattern category and strength
            pattern_category = await self._categorize_pattern(pattern)
            pattern_strength = await self._assess_pattern_strength(pattern)
            
            # Get product context
            product_context = await self._get_product_context(pattern.product_id)
            
            # Identify key factors
            key_factors = await self.identify_key_factors(pattern)
            
            # Generate supporting evidence
            supporting_evidence = await self._generate_supporting_evidence(pattern)
            
            # Create business-friendly description
            description = await self._create_pattern_description(
                pattern, pattern_category, pattern_strength, product_context
            )
            
            # Generate title
            title = await self._create_insight_title(pattern, pattern_category, product_context)
            
            # Generate recommended actions
            recommended_actions = await self._generate_recommended_actions(
                pattern, pattern_category, key_factors
            )
            
            # Determine business impact
            business_impact = await self._assess_business_impact(pattern, pattern_category)
            
            # Get data sources
            data_sources = await self._identify_data_sources(pattern)
            
            # Create the insight
            insight = ExplainableInsight(
                title=title,
                description=description,
                confidence_level=pattern.confidence_level,  # Will be updated later
                supporting_evidence=supporting_evidence,
                key_factors=key_factors,
                business_impact=business_impact,
                recommended_actions=recommended_actions,
                data_sources=data_sources,
                related_products=[pattern.product_id],
                expires_at=datetime.utcnow() + timedelta(days=30)  # Insights valid for 30 days
            )
            
            logger.debug(f"Successfully explained pattern {pattern.id}")
            return insight
            
        except Exception as e:
            logger.error(f"Failed to explain pattern {pattern.id}: {e}")
            raise InsightGenerationError(f"Pattern explanation failed: {e}")
    
    async def calculate_confidence(self, insight: ExplainableInsight) -> float:
        """
        Calculate confidence score for an insight.
        
        Args:
            insight: Insight to evaluate
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            confidence_factors = []
            
            # Evidence quality factor (0.0 - 0.3)
            evidence_score = min(len(insight.supporting_evidence) * 0.1, 0.3)
            confidence_factors.append(evidence_score)
            
            # Key factors identification (0.0 - 0.2)
            factors_score = min(len(insight.key_factors) * 0.05, 0.2)
            confidence_factors.append(factors_score)
            
            # Business impact clarity (0.0 - 0.2)
            impact_score = 0.2 if insight.business_impact and len(insight.business_impact) > 50 else 0.1
            confidence_factors.append(impact_score)
            
            # Data source reliability (0.0 - 0.15)
            data_score = min(len(insight.data_sources) * 0.05, 0.15)
            confidence_factors.append(data_score)
            
            # Actionability (0.0 - 0.15)
            action_score = min(len(insight.recommended_actions) * 0.05, 0.15)
            confidence_factors.append(action_score)
            
            # Base confidence from description quality
            description_score = min(len(insight.description) / 200, 0.1)  # Longer descriptions = higher confidence
            confidence_factors.append(description_score)
            
            # Calculate weighted confidence
            total_confidence = sum(confidence_factors)
            
            # Normalize to 0-1 range
            normalized_confidence = min(total_confidence, 1.0)
            
            logger.debug(f"Calculated confidence score: {normalized_confidence:.3f}")
            return normalized_confidence
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5  # Default medium confidence
    
    async def identify_key_factors(self, pattern: DemandPattern) -> List[str]:
        """
        Identify key factors influencing a demand pattern.
        
        Args:
            pattern: Demand pattern to analyze
            
        Returns:
            List of key influencing factors
        """
        try:
            key_factors = []
            
            # Seasonal factors with enhanced highlighting
            if pattern.seasonal_factors:
                seasonal_factors = await self._analyze_seasonal_factors(pattern.seasonal_factors)
                key_factors.extend(seasonal_factors)
            
            # Trend factors with importance analysis
            if pattern.trend_direction and pattern.trend_direction != "stable":
                trend_factors = await self._analyze_trend_factors(pattern)
                key_factors.extend(trend_factors)
            
            # Volatility factors with business impact
            volatility_factors = await self._analyze_volatility_factors(pattern)
            key_factors.extend(volatility_factors)
            
            # Data quality factors with reliability indicators
            data_factors = await self._analyze_data_quality_factors(pattern)
            key_factors.extend(data_factors)
            
            # Pattern-specific factors with business context
            pattern_factors = await self._analyze_pattern_specific_factors(pattern)
            key_factors.extend(pattern_factors)
            
            # Time period factors with strategic implications
            time_factors = await self._analyze_time_period_factors(pattern)
            key_factors.extend(time_factors)
            
            # Ensure we have at least some factors
            if not key_factors:
                key_factors.append("Standard demand pattern analysis based on historical sales data")
            
            # Prioritize and highlight most important factors
            prioritized_factors = await self._prioritize_key_factors(key_factors, pattern)
            
            logger.debug(f"Identified {len(prioritized_factors)} key factors for pattern {pattern.id}")
            return prioritized_factors[:5]  # Limit to top 5 factors for clarity
            
        except Exception as e:
            logger.error(f"Failed to identify key factors for pattern {pattern.id}: {e}")
            return ["Historical sales data analysis", "Market demand patterns"]
    
    async def _analyze_seasonal_factors(self, seasonal_factors: Dict[str, float]) -> List[str]:
        """Analyze seasonal factors with enhanced highlighting."""
        factors = []
        
        # Sort by impact magnitude for highlighting
        sorted_factors = sorted(seasonal_factors.items(), key=lambda x: abs(x[1] - 1.0), reverse=True)
        
        for event, factor in sorted_factors:
            if abs(factor - 1.0) > 0.15:  # 15% threshold for significance
                event_name = self.seasonal_events.get(event, event.replace('_', ' ').title())
                impact_magnitude = abs(factor - 1.0) * 100
                
                if factor > 1.0:
                    impact_type = "📈 BOOST"
                    business_impact = "revenue opportunity"
                else:
                    impact_type = "📉 DECLINE"
                    business_impact = "inventory optimization opportunity"
                
                # Enhanced highlighting with business context
                if impact_magnitude > 50:
                    priority = "🔥 CRITICAL"
                elif impact_magnitude > 30:
                    priority = "⚠️ HIGH"
                elif impact_magnitude > 15:
                    priority = "📊 MODERATE"
                else:
                    priority = "📋 MINOR"
                
                factor_description = (
                    f"{priority}: {event_name} creates {impact_magnitude:.0f}% demand {impact_type.split()[1].lower()} "
                    f"({business_impact})"
                )
                factors.append(factor_description)
        
        return factors
    
    async def _analyze_trend_factors(self, pattern: DemandPattern) -> List[str]:
        """Analyze trend factors with importance analysis."""
        factors = []
        
        trend_strength = "strong" if pattern.volatility_score < 0.3 else "moderate"
        trend_direction = pattern.trend_direction
        
        # Enhanced trend analysis with business implications
        if trend_direction == "increasing":
            business_implication = "growth opportunity requiring capacity planning"
            strategic_action = "consider market expansion"
        elif trend_direction == "decreasing":
            business_implication = "market challenge requiring intervention"
            strategic_action = "implement retention strategies"
        else:
            business_implication = "stable market conditions"
            strategic_action = "focus on operational efficiency"
        
        factor_description = (
            f"📈 TREND ANALYSIS: {trend_strength.title()} {trend_direction} trend indicates "
            f"{business_implication} - {strategic_action}"
        )
        factors.append(factor_description)
        
        return factors
    
    async def _analyze_volatility_factors(self, pattern: DemandPattern) -> List[str]:
        """Analyze volatility factors with business impact."""
        factors = []
        
        volatility = pattern.volatility_score
        
        if volatility > 0.7:
            risk_level = "🔴 CRITICAL"
            business_impact = "high inventory risk and forecasting challenges"
            recommendation = "implement advanced demand sensing"
        elif volatility > 0.5:
            risk_level = "🟡 HIGH"
            business_impact = "moderate inventory management complexity"
            recommendation = "increase safety stock buffers"
        elif volatility > 0.3:
            risk_level = "🟢 MODERATE"
            business_impact = "manageable demand variability"
            recommendation = "standard inventory practices sufficient"
        else:
            risk_level = "✅ LOW"
            business_impact = "predictable demand enabling efficient operations"
            recommendation = "optimize for cost efficiency"
        
        factor_description = (
            f"{risk_level} VOLATILITY: Demand variability of {volatility:.1f} indicates "
            f"{business_impact} - {recommendation}"
        )
        factors.append(factor_description)
        
        return factors
    
    async def _analyze_data_quality_factors(self, pattern: DemandPattern) -> List[str]:
        """Analyze data quality factors with reliability indicators."""
        factors = []
        
        data_points = pattern.supporting_data_points
        confidence = pattern.confidence_level
        
        if data_points < 10:
            reliability = "⚠️ LIMITED"
            implication = "insights may have lower reliability"
            action = "collect additional data for validation"
        elif data_points < 50:
            reliability = "📊 MODERATE"
            implication = "reasonable confidence in patterns"
            action = "continue monitoring for trend confirmation"
        elif data_points < 100:
            reliability = "✅ GOOD"
            implication = "solid foundation for decision making"
            action = "leverage insights for strategic planning"
        else:
            reliability = "🏆 EXCELLENT"
            implication = "high confidence insights supporting strategic decisions"
            action = "use for long-term business planning"
        
        factor_description = (
            f"{reliability} DATA QUALITY: {data_points} data points with {confidence.value} confidence - "
            f"{implication}, {action}"
        )
        factors.append(factor_description)
        
        return factors
    
    async def _analyze_pattern_specific_factors(self, pattern: DemandPattern) -> List[str]:
        """Analyze pattern-specific factors with business context."""
        factors = []
        
        pattern_type = pattern.pattern_type.lower()
        
        if "seasonal" in pattern_type:
            factor_description = (
                "🎯 PATTERN TYPE: Seasonal business cycle drives primary demand variations - "
                "leverage for promotional timing and inventory planning"
            )
        elif "cyclical" in pattern_type:
            factor_description = (
                "🔄 PATTERN TYPE: Regular cyclical patterns enable predictive inventory planning - "
                "optimize reorder cycles and promotional schedules"
            )
        elif "trend" in pattern_type:
            factor_description = (
                "📊 PATTERN TYPE: Fundamental market trend influences long-term demand direction - "
                "align strategic planning with trend trajectory"
            )
        elif "volatile" in pattern_type:
            factor_description = (
                "⚡ PATTERN TYPE: High variability pattern requires adaptive management - "
                "implement flexible inventory and pricing strategies"
            )
        else:
            factor_description = (
                "📋 PATTERN TYPE: Standard demand pattern suitable for conventional planning - "
                "apply established inventory management practices"
            )
        
        factors.append(factor_description)
        return factors
    
    async def _analyze_time_period_factors(self, pattern: DemandPattern) -> List[str]:
        """Analyze time period factors with strategic implications."""
        factors = []
        
        time_span = (pattern.date_range_end - pattern.date_range_start).days
        
        if time_span > 730:  # 2+ years
            strategic_value = "🏆 STRATEGIC"
            implication = "multi-year analysis provides robust seasonal insights"
            application = "use for long-term strategic planning and capacity decisions"
        elif time_span > 365:  # 1+ year
            strategic_value = "✅ COMPREHENSIVE"
            implication = "full-year analysis captures complete seasonal cycles"
            application = "reliable for annual planning and seasonal strategies"
        elif time_span > 180:  # 6+ months
            strategic_value = "📊 SUBSTANTIAL"
            implication = "sufficient data for seasonal pattern identification"
            application = "suitable for medium-term planning and tactical decisions"
        elif time_span > 90:  # 3+ months
            strategic_value = "📋 MODERATE"
            implication = "adequate for trend identification but limited seasonal insight"
            application = "use for short-term planning with caution on seasonal factors"
        else:
            strategic_value = "⚠️ LIMITED"
            implication = "short-term analysis may not capture full seasonal patterns"
            application = "supplement with additional data for strategic decisions"
        
        factor_description = (
            f"{strategic_value} TIME SCOPE: {time_span} days of analysis - "
            f"{implication}, {application}"
        )
        factors.append(factor_description)
        
        return factors
    
    async def _prioritize_key_factors(self, factors: List[str], pattern: DemandPattern) -> List[str]:
        """Prioritize key factors by business importance and impact."""
        
        def factor_priority_score(factor: str) -> float:
            score = 0.0
            
            # High priority indicators
            if "🔥 CRITICAL" in factor or "🔴 CRITICAL" in factor:
                score += 10.0
            elif "⚠️ HIGH" in factor or "🟡 HIGH" in factor:
                score += 8.0
            elif "📊 MODERATE" in factor or "🟢 MODERATE" in factor:
                score += 6.0
            
            # Business impact indicators
            if "revenue opportunity" in factor:
                score += 5.0
            elif "growth opportunity" in factor:
                score += 4.0
            elif "inventory risk" in factor:
                score += 4.0
            elif "strategic planning" in factor:
                score += 3.0
            
            # Data quality boost
            if "🏆 EXCELLENT" in factor:
                score += 3.0
            elif "✅ GOOD" in factor:
                score += 2.0
            elif "📊 MODERATE" in factor:
                score += 1.0
            
            # Pattern significance
            if "BOOST" in factor or "DECLINE" in factor:
                score += 2.0
            
            # Strategic value
            if "🏆 STRATEGIC" in factor:
                score += 2.0
            elif "✅ COMPREHENSIVE" in factor:
                score += 1.5
            
            return score
        
        # Sort factors by priority score
        prioritized = sorted(factors, key=factor_priority_score, reverse=True)
        
        return prioritized
    
    async def enhance_business_language(self, text: str) -> str:
        """
        Enhance text with business-friendly language processing.
        
        Args:
            text: Text to enhance
            
        Returns:
            Enhanced business-friendly text
        """
        try:
            # Business terminology mapping
            business_terms = {
                'high volatility': 'significant demand variability',
                'low volatility': 'consistent demand patterns',
                'increasing trend': 'growing market demand',
                'decreasing trend': 'declining market conditions',
                'seasonal factor': 'seasonal business driver',
                'data points': 'sales transactions',
                'confidence level': 'reliability indicator',
                'pattern type': 'demand characteristic',
                'risk assessment': 'business risk evaluation',
                'inventory level': 'stock position',
                'demand forecast': 'sales projection'
            }
            
            enhanced_text = text
            
            # Apply business terminology
            for technical_term, business_term in business_terms.items():
                enhanced_text = enhanced_text.replace(technical_term, business_term)
            
            # Add emphasis markers for key business concepts
            emphasis_patterns = {
                r'\b(\d+)%\b': r'**\1%**',  # Highlight percentages
                r'\b(revenue|profit|cost|sales)\b': r'**\1**',  # Highlight financial terms
                r'\b(opportunity|risk|challenge)\b': r'**\1**',  # Highlight business implications
                r'\b(seasonal|festival|holiday)\b': r'**\1**',  # Highlight seasonal terms
            }
            
            import re
            for pattern, replacement in emphasis_patterns.items():
                enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Failed to enhance business language: {e}")
            return text  # Return original text if enhancement fails
    
    async def generate_factor_importance_analysis(self, factors: List[str]) -> Dict[str, Any]:
        """
        Generate detailed factor importance analysis.
        
        Args:
            factors: List of key factors
            
        Returns:
            Dictionary containing factor importance analysis
        """
        try:
            analysis = {
                'total_factors': len(factors),
                'critical_factors': [],
                'high_impact_factors': [],
                'moderate_factors': [],
                'strategic_factors': [],
                'operational_factors': [],
                'importance_summary': '',
                'priority_recommendations': []
            }
            
            # Categorize factors by importance and type
            for factor in factors:
                if "🔥 CRITICAL" in factor or "🔴 CRITICAL" in factor:
                    analysis['critical_factors'].append(factor)
                elif "⚠️ HIGH" in factor or "🟡 HIGH" in factor:
                    analysis['high_impact_factors'].append(factor)
                elif "📊 MODERATE" in factor or "🟢 MODERATE" in factor:
                    analysis['moderate_factors'].append(factor)
                
                if "strategic planning" in factor.lower() or "long-term" in factor.lower():
                    analysis['strategic_factors'].append(factor)
                elif "inventory" in factor.lower() or "operational" in factor.lower():
                    analysis['operational_factors'].append(factor)
            
            # Generate importance summary
            critical_count = len(analysis['critical_factors'])
            high_count = len(analysis['high_impact_factors'])
            
            if critical_count > 0:
                analysis['importance_summary'] = f"Analysis reveals {critical_count} critical factors requiring immediate attention"
                analysis['priority_recommendations'].append("Address critical factors immediately to mitigate business risks")
            elif high_count > 0:
                analysis['importance_summary'] = f"Analysis identifies {high_count} high-impact factors for strategic focus"
                analysis['priority_recommendations'].append("Prioritize high-impact factors in business planning")
            else:
                analysis['importance_summary'] = "Analysis shows moderate factors suitable for standard planning approaches"
                analysis['priority_recommendations'].append("Apply standard business practices with regular monitoring")
            
            # Add strategic vs operational balance
            strategic_count = len(analysis['strategic_factors'])
            operational_count = len(analysis['operational_factors'])
            
            if strategic_count > operational_count:
                analysis['priority_recommendations'].append("Focus on strategic planning and long-term positioning")
            elif operational_count > strategic_count:
                analysis['priority_recommendations'].append("Emphasize operational efficiency and tactical execution")
            else:
                analysis['priority_recommendations'].append("Balance strategic planning with operational excellence")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate factor importance analysis: {e}")
            return {
                'total_factors': len(factors),
                'importance_summary': 'Standard factor analysis completed',
                'priority_recommendations': ['Apply standard business practices']
            }
    
    async def _categorize_pattern(self, pattern: DemandPattern) -> str:
        """Categorize the pattern type for appropriate template selection."""
        # Check for seasonal patterns
        if pattern.seasonal_factors and any(abs(f - 1.0) > 0.3 for f in pattern.seasonal_factors.values()):
            return 'seasonal'
        
        # Check for trending patterns
        if pattern.trend_direction and pattern.trend_direction != "stable":
            return 'trending'
        
        # Check for high volatility
        if pattern.volatility_score > 0.6:
            return 'volatile'
        
        # Check for stable patterns
        if pattern.volatility_score < 0.2:
            return 'stable'
        
        # Default to cyclical if pattern type suggests it
        if 'cyclical' in pattern.pattern_type.lower():
            return 'cyclical'
        
        return 'stable'  # Default fallback
    
    async def _assess_pattern_strength(self, pattern: DemandPattern) -> str:
        """Assess the strength of the pattern."""
        if pattern.confidence_level == ConfidenceLevel.HIGH:
            return "strong"
        elif pattern.confidence_level == ConfidenceLevel.MEDIUM:
            return "moderate"
        else:
            return "weak"
    
    async def _get_product_context(self, product_id: str) -> Dict[str, Any]:
        """Get product context from sales data."""
        product_sales = [sale for sale in self.sales_data if sale.product_id == product_id]
        
        if not product_sales:
            return {
                'product_name': f"Product {product_id}",
                'category': 'general',
                'avg_price': 0,
                'total_sales': 0
            }
        
        return {
            'product_name': product_sales[0].product_name,
            'category': product_sales[0].category,
            'avg_price': statistics.mean([float(sale.selling_price) for sale in product_sales]),
            'total_sales': sum(sale.quantity_sold for sale in product_sales)
        }
    
    async def _generate_supporting_evidence(self, pattern: DemandPattern) -> List[str]:
        """Generate supporting evidence for the pattern."""
        evidence = []
        
        # Data volume evidence
        evidence.append(f"Analysis based on {pattern.supporting_data_points} sales transactions")
        
        # Time period evidence
        time_span = (pattern.date_range_end - pattern.date_range_start).days
        if time_span > 365:
            evidence.append(f"Multi-year analysis covering {time_span} days of sales history")
        else:
            evidence.append(f"Analysis period covers {time_span} days of recent sales data")
        
        # Confidence evidence
        confidence_desc = {
            ConfidenceLevel.HIGH: "High statistical confidence in pattern identification",
            ConfidenceLevel.MEDIUM: "Moderate statistical confidence with clear trend indicators",
            ConfidenceLevel.LOW: "Emerging pattern with limited historical validation"
        }
        evidence.append(confidence_desc.get(pattern.confidence_level, "Pattern identified through statistical analysis"))
        
        # Seasonal evidence
        if pattern.seasonal_factors:
            significant_events = [
                event for event, factor in pattern.seasonal_factors.items() 
                if abs(factor - 1.0) > 0.2
            ]
            if significant_events:
                evidence.append(f"Seasonal correlation identified with {', '.join(significant_events[:3])}")
        
        # Volatility evidence
        if pattern.volatility_score > 0.6:
            evidence.append("High demand variability observed across multiple time periods")
        elif pattern.volatility_score < 0.2:
            evidence.append("Consistent demand levels with minimal variation")
        
        return evidence[:4]  # Limit to 4 pieces of evidence
    
    async def _create_pattern_description(self, pattern: DemandPattern, category: str, 
                                        strength: str, product_context: Dict[str, Any]) -> str:
        """Create a business-friendly pattern description."""
        template = self.pattern_templates.get(category, self.pattern_templates['stable'])
        
        # Prepare template variables
        template_vars = {
            'product_name': product_context['product_name'],
            'strength': strength,
            'direction': pattern.trend_direction or 'stable',
            'data_points': pattern.supporting_data_points,
            'time_period': f"{(pattern.date_range_end - pattern.date_range_start).days} days"
        }
        
        # Category-specific variables
        if category == 'seasonal':
            peak_events = []
            if pattern.seasonal_factors:
                for event, factor in pattern.seasonal_factors.items():
                    if factor > 1.2:  # 20% boost or more
                        peak_events.append(self.seasonal_events.get(event, event))
            
            template_vars.update({
                'peak_description': f"{int((max(pattern.seasonal_factors.values()) - 1) * 100)}% higher sales" if pattern.seasonal_factors else "seasonal variations",
                'peak_periods': ', '.join(peak_events[:3]) if peak_events else "key seasonal periods",
                'trend_description': f"Overall trend is {pattern.trend_direction}" if pattern.trend_direction != 'stable' else "Demand remains seasonally consistent"
            })
        
        elif category == 'trending':
            template_vars.update({
                'trend_strength': strength,
                'trend_percentage': f"{abs(pattern.volatility_score * 100):.0f}%",
                'volatility_description': f"Volatility level is {pattern.volatility_score:.1f}" if pattern.volatility_score > 0.3 else "Trend shows good consistency"
            })
        
        elif category == 'volatile':
            template_vars.update({
                'volatility_level': "high" if pattern.volatility_score > 0.7 else "moderate",
                'business_implication': "challenging demand forecasting" if pattern.volatility_score > 0.7 else "moderate forecasting complexity",
                'risk_description': "Consider flexible inventory strategies" if pattern.volatility_score > 0.7 else "Standard inventory management approaches should suffice",
                'cv_value': f"{pattern.volatility_score:.2f}"
            })
        
        elif category == 'stable':
            template_vars.update({
                'consistency_description': f"low variation (CV: {pattern.volatility_score:.2f})",
                'predictability_description': "This enables reliable demand forecasting and inventory planning"
            })
        
        # Format the description
        try:
            description = template['description_template'].format(**template_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using fallback description")
            description = f"{product_context['product_name']} shows a {strength} {category} demand pattern based on analysis of {pattern.supporting_data_points} sales records."
        
        return description
    
    async def _create_insight_title(self, pattern: DemandPattern, category: str, 
                                  product_context: Dict[str, Any]) -> str:
        """Create a concise insight title."""
        template = self.pattern_templates.get(category, self.pattern_templates['stable'])
        
        template_vars = {
            'product_name': product_context['product_name'],
            'strength': await self._assess_pattern_strength(pattern),
            'direction': pattern.trend_direction or 'stable',
            'cycle_type': 'monthly' if 'monthly' in pattern.pattern_type else 'seasonal',
            'volatility_level': 'high' if pattern.volatility_score > 0.6 else 'moderate'
        }
        
        try:
            title = template['title_template'].format(**template_vars)
        except KeyError:
            title = f"{product_context['product_name']} - {category.title()} Demand Pattern"
        
        return title
    
    async def _generate_recommended_actions(self, pattern: DemandPattern, category: str, 
                                          key_factors: List[str]) -> List[str]:
        """Generate actionable recommendations based on the pattern."""
        recommendations = []
        
        # Category-specific recommendations
        if category == 'seasonal':
            recommendations.append("Plan inventory buildup 2-4 weeks before peak seasonal periods")
            recommendations.append("Consider seasonal pricing strategies to maximize revenue during high-demand periods")
            if pattern.seasonal_factors:
                peak_factor = max(pattern.seasonal_factors.values())
                if peak_factor > 1.5:
                    recommendations.append(f"Increase safety stock by {int((peak_factor - 1) * 100)}% for seasonal peaks")
        
        elif category == 'trending':
            if pattern.trend_direction == 'increasing':
                recommendations.append("Gradually increase inventory levels to meet growing demand")
                recommendations.append("Consider expanding product line or increasing marketing investment")
            elif pattern.trend_direction == 'decreasing':
                recommendations.append("Implement promotional strategies to stimulate demand")
                recommendations.append("Review inventory levels to prevent overstock situations")
        
        elif category == 'volatile':
            recommendations.append("Implement flexible inventory management with shorter reorder cycles")
            recommendations.append("Consider demand sensing technologies for better short-term forecasting")
            recommendations.append("Maintain higher safety stock levels to buffer against demand variability")
        
        elif category == 'stable':
            recommendations.append("Maintain consistent inventory levels with standard reorder points")
            recommendations.append("Focus on operational efficiency and cost optimization")
            recommendations.append("Consider this product for reliable revenue planning")
        
        # General recommendations based on key factors
        if any('limited' in factor.lower() or 'short-term' in factor.lower() for factor in key_factors):
            recommendations.append("Collect additional sales data to improve forecast accuracy")
        
        if any('extensive' in factor.lower() or 'multi-year' in factor.lower() for factor in key_factors):
            recommendations.append("Leverage historical patterns for strategic long-term planning")
        
        # Ensure we have at least 2 recommendations
        if len(recommendations) < 2:
            recommendations.extend([
                "Monitor demand patterns regularly for changes",
                "Align inventory planning with identified demand characteristics"
            ])
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    async def _assess_business_impact(self, pattern: DemandPattern, category: str) -> str:
        """Assess the business impact of the pattern."""
        impact_parts = []
        
        # Revenue impact
        if category == 'seasonal' and pattern.seasonal_factors:
            peak_factor = max(pattern.seasonal_factors.values())
            if peak_factor > 1.5:
                impact_parts.append(f"Seasonal peaks can drive up to {int((peak_factor - 1) * 100)}% revenue increase")
        
        elif category == 'trending':
            if pattern.trend_direction == 'increasing':
                impact_parts.append("Growing demand trend presents revenue expansion opportunities")
            elif pattern.trend_direction == 'decreasing':
                impact_parts.append("Declining trend requires proactive intervention to maintain revenue")
        
        # Inventory impact
        if pattern.volatility_score > 0.6:
            impact_parts.append("High volatility increases inventory management complexity and costs")
        elif pattern.volatility_score < 0.2:
            impact_parts.append("Stable demand enables efficient inventory management and cost control")
        
        # Planning impact
        if pattern.confidence_level == ConfidenceLevel.HIGH:
            impact_parts.append("High pattern confidence enables reliable business planning and forecasting")
        elif pattern.confidence_level == ConfidenceLevel.LOW:
            impact_parts.append("Pattern uncertainty requires flexible planning approaches and contingency strategies")
        
        # Combine impact statements
        if impact_parts:
            return ". ".join(impact_parts) + "."
        else:
            return "This pattern provides valuable insights for inventory and sales planning decisions."
    
    async def _identify_data_sources(self, pattern: DemandPattern) -> List[str]:
        """Identify data sources used for the pattern analysis."""
        sources = [
            "Historical sales transaction data",
            "Product inventory records"
        ]
        
        if pattern.seasonal_factors:
            sources.append("Seasonal calendar and festival data")
        
        if pattern.supporting_data_points > 100:
            sources.append("Comprehensive multi-period sales analysis")
        
        sources.append("MarketPulse AI pattern recognition algorithms")
        
        return sources
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numerical confidence score to ConfidenceLevel enum."""
        if score >= self.confidence_thresholds['high']:
            return ConfidenceLevel.HIGH
        elif score >= self.confidence_thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    async def _prioritize_insights(self, insights: List[ExplainableInsight]) -> List[ExplainableInsight]:
        """Prioritize insights by confidence and business impact."""
        def priority_score(insight: ExplainableInsight) -> float:
            # Base score from confidence level
            confidence_scores = {
                ConfidenceLevel.HIGH: 0.8,
                ConfidenceLevel.MEDIUM: 0.6,
                ConfidenceLevel.LOW: 0.4
            }
            base_score = confidence_scores.get(insight.confidence_level, 0.5)
            
            # Boost for actionable insights
            action_boost = min(len(insight.recommended_actions) * 0.05, 0.2)
            
            # Boost for business impact clarity
            impact_boost = 0.1 if len(insight.business_impact) > 100 else 0.05
            
            # Boost for evidence quality
            evidence_boost = min(len(insight.supporting_evidence) * 0.03, 0.1)
            
            return base_score + action_boost + impact_boost + evidence_boost
        
        # Sort by priority score (descending)
        return sorted(insights, key=priority_score, reverse=True)