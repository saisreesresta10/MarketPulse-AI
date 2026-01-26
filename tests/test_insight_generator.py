"""
Unit tests for the Insight Generator component.

Tests natural language insight generation, pattern explanation, confidence calculation,
and key factor identification functionality.
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

from marketpulse_ai.components.insight_generator import (
    InsightGenerator, 
    InsightGenerationError
)
from marketpulse_ai.core.models import (
    DemandPattern, 
    ExplainableInsight, 
    ConfidenceLevel,
    SalesDataPoint
)


class TestInsightGenerator:
    """Test suite for InsightGenerator component."""
    
    @pytest.fixture
    def insight_generator(self):
        """Create an InsightGenerator instance for testing."""
        return InsightGenerator()
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for testing."""
        base_date = date.today() - timedelta(days=365)
        sales_data = []
        
        for i in range(100):
            current_date = base_date + timedelta(days=i * 3)  # Every 3 days
            
            # Seasonal boost for Diwali (October-November)
            quantity = 50
            if current_date.month in [10, 11]:
                quantity = 80  # 60% boost during Diwali
            
            sales_data.append(SalesDataPoint(
                product_id="TEST_PRODUCT_001",
                product_name="Premium Electronics Item",
                category="electronics",
                mrp=Decimal("2000.00"),
                selling_price=Decimal("1800.00"),
                quantity_sold=quantity,
                sale_date=current_date,
                store_location="MAIN_STORE"
            ))
        
        return sales_data
    
    @pytest.fixture
    def seasonal_pattern(self):
        """Create a seasonal demand pattern for testing."""
        return DemandPattern(
            product_id="TEST_PRODUCT_001",
            pattern_type="seasonal",
            description="Seasonal pattern with Diwali boost",
            confidence_level=ConfidenceLevel.HIGH,
            seasonal_factors={
                'diwali': 1.6,
                'summer': 0.8,
                'winter': 1.1
            },
            trend_direction="stable",
            volatility_score=0.3,
            supporting_data_points=100,
            date_range_start=date.today() - timedelta(days=365),
            date_range_end=date.today()
        )
    
    @pytest.fixture
    def trending_pattern(self):
        """Create a trending demand pattern for testing."""
        return DemandPattern(
            product_id="TEST_PRODUCT_002",
            pattern_type="basic_trend",
            description="Increasing trend pattern",
            confidence_level=ConfidenceLevel.MEDIUM,
            seasonal_factors={},
            trend_direction="increasing",
            volatility_score=0.4,
            supporting_data_points=75,
            date_range_start=date.today() - timedelta(days=180),
            date_range_end=date.today()
        )
    
    @pytest.fixture
    def volatile_pattern(self):
        """Create a volatile demand pattern for testing."""
        return DemandPattern(
            product_id="TEST_PRODUCT_003",
            pattern_type="volatile",
            description="High volatility pattern",
            confidence_level=ConfidenceLevel.LOW,
            seasonal_factors={},
            trend_direction="stable",
            volatility_score=0.8,
            supporting_data_points=50,
            date_range_start=date.today() - timedelta(days=90),
            date_range_end=date.today()
        )
    
    @pytest.mark.asyncio
    async def test_generate_insights_success(self, insight_generator, sample_sales_data, seasonal_pattern, trending_pattern):
        """Test successful insight generation from multiple patterns."""
        insight_generator.set_sales_data(sample_sales_data)
        
        patterns = [seasonal_pattern, trending_pattern]
        insights = await insight_generator.generate_insights(patterns)
        
        # Verify insights were generated
        assert len(insights) == 2
        
        # Check insight structure
        for insight in insights:
            assert isinstance(insight, ExplainableInsight)
            assert insight.title
            assert insight.description
            assert insight.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
            assert len(insight.supporting_evidence) > 0
            assert len(insight.key_factors) > 0
            assert insight.business_impact
            assert len(insight.recommended_actions) > 0
            assert len(insight.data_sources) > 0
    
    @pytest.mark.asyncio
    async def test_generate_insights_empty_patterns(self, insight_generator):
        """Test insight generation with empty pattern list."""
        insights = await insight_generator.generate_insights([])
        
        assert insights == []
    
    @pytest.mark.asyncio
    async def test_explain_seasonal_pattern(self, insight_generator, sample_sales_data, seasonal_pattern):
        """Test explanation of seasonal demand pattern."""
        insight_generator.set_sales_data(sample_sales_data)
        
        insight = await insight_generator.explain_pattern(seasonal_pattern)
        
        # Verify insight content
        assert "seasonal" in insight.title.lower()
        assert "Premium Electronics Item" in insight.title
        assert "diwali" in insight.description.lower() or "seasonal" in insight.description.lower()
        
        # Check for seasonal-specific evidence
        evidence_text = " ".join(insight.supporting_evidence).lower()
        assert "seasonal" in evidence_text or "correlation" in evidence_text
        
        # Check for seasonal recommendations
        recommendations_text = " ".join(insight.recommended_actions).lower()
        assert any(keyword in recommendations_text for keyword in ["seasonal", "inventory", "peak", "buildup"])
        
        # Verify key factors mention seasonality
        factors_text = " ".join(insight.key_factors).lower()
        assert "diwali" in factors_text or "seasonal" in factors_text
    
    @pytest.mark.asyncio
    async def test_explain_trending_pattern(self, insight_generator, sample_sales_data, trending_pattern):
        """Test explanation of trending demand pattern."""
        insight_generator.set_sales_data(sample_sales_data)
        
        insight = await insight_generator.explain_pattern(trending_pattern)
        
        # Verify trend-specific content
        assert "trend" in insight.title.lower() or "increasing" in insight.title.lower()
        assert "increasing" in insight.description.lower() or "trend" in insight.description.lower()
        
        # Check for trend-specific recommendations
        recommendations_text = " ".join(insight.recommended_actions).lower()
        assert any(keyword in recommendations_text for keyword in ["increase", "growing", "expand", "investment"])
    
    @pytest.mark.asyncio
    async def test_explain_volatile_pattern(self, insight_generator, sample_sales_data, volatile_pattern):
        """Test explanation of volatile demand pattern."""
        insight_generator.set_sales_data(sample_sales_data)
        
        insight = await insight_generator.explain_pattern(volatile_pattern)
        
        # Verify volatility-specific content
        assert "volatility" in insight.title.lower() or "volatile" in insight.title.lower()
        assert "volatility" in insight.description.lower() or "variability" in insight.description.lower()
        
        # Check for volatility-specific recommendations
        recommendations_text = " ".join(insight.recommended_actions).lower()
        assert any(keyword in recommendations_text for keyword in ["flexible", "safety stock", "buffer", "variability"])
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_high_quality(self, insight_generator):
        """Test confidence calculation for high-quality insight."""
        high_quality_insight = ExplainableInsight(
            title="Comprehensive Product Analysis",
            description="This is a detailed description with comprehensive analysis of demand patterns, seasonal variations, and business implications that provides substantial value for decision making.",
            confidence_level=ConfidenceLevel.MEDIUM,  # Will be updated
            supporting_evidence=[
                "Analysis based on 200 sales transactions",
                "Multi-year analysis covering 730 days",
                "High statistical confidence in pattern identification",
                "Seasonal correlation identified with diwali, christmas"
            ],
            key_factors=[
                "Diwali creates 60% demand boost",
                "Strong increasing trend in demand",
                "Extensive historical data supporting insights",
                "Seasonal business cycle drives variations"
            ],
            business_impact="Seasonal peaks can drive up to 60% revenue increase. High pattern confidence enables reliable business planning and forecasting.",
            recommended_actions=[
                "Plan inventory buildup before peak seasons",
                "Consider seasonal pricing strategies",
                "Increase safety stock by 60% for peaks",
                "Leverage patterns for strategic planning"
            ],
            data_sources=[
                "Historical sales transaction data",
                "Seasonal calendar and festival data",
                "Comprehensive multi-period analysis"
            ]
        )
        
        confidence = await insight_generator.calculate_confidence(high_quality_insight)
        
        # High-quality insight should have high confidence
        assert confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_low_quality(self, insight_generator):
        """Test confidence calculation for low-quality insight."""
        low_quality_insight = ExplainableInsight(
            title="Basic Analysis",
            description="Simple pattern.",
            confidence_level=ConfidenceLevel.LOW,
            supporting_evidence=["Limited data"],
            key_factors=["Basic trend"],
            business_impact="Some impact",
            recommended_actions=["Monitor"],
            data_sources=["Sales data"]
        )
        
        confidence = await insight_generator.calculate_confidence(low_quality_insight)
        
        # Low-quality insight should have low confidence
        assert confidence < 0.5
    
    @pytest.mark.asyncio
    async def test_identify_key_factors_seasonal(self, insight_generator, seasonal_pattern):
        """Test key factor identification for seasonal patterns."""
        key_factors = await insight_generator.identify_key_factors(seasonal_pattern)
        
        assert len(key_factors) > 0
        assert len(key_factors) <= 5  # Should limit to top 5
        
        # Should identify seasonal factors with enhanced highlighting
        factors_text = " ".join(key_factors).lower()
        assert "diwali" in factors_text or "seasonal" in factors_text
        
        # Should mention the boost percentage with highlighting
        assert any("60%" in factor or "boost" in factor for factor in key_factors)
        
        # Should include priority indicators
        assert any("critical" in factor.lower() or "high" in factor.lower() or "moderate" in factor.lower() 
                  for factor in key_factors)
    
    @pytest.mark.asyncio
    async def test_enhanced_seasonal_factor_analysis(self, insight_generator):
        """Test enhanced seasonal factor analysis with highlighting."""
        seasonal_factors = {
            'diwali': 1.8,    # 80% boost - should be critical
            'summer': 0.7,    # 30% decline - should be high
            'winter': 1.1     # 10% boost - should be minor
        }
        
        factors = await insight_generator._analyze_seasonal_factors(seasonal_factors)
        
        assert len(factors) > 0
        
        # Should prioritize by impact magnitude
        factors_text = " ".join(factors)
        
        # Should include priority indicators
        assert "CRITICAL" in factors_text or "HIGH" in factors_text
        
        # Should include business context
        assert "revenue opportunity" in factors_text or "optimization opportunity" in factors_text
        
        # Should include emojis for visual highlighting
        assert "🔥" in factors_text or "⚠️" in factors_text
    
    @pytest.mark.asyncio
    async def test_trend_factor_analysis(self, insight_generator, trending_pattern):
        """Test enhanced trend factor analysis."""
        factors = await insight_generator._analyze_trend_factors(trending_pattern)
        
        assert len(factors) > 0
        
        factors_text = " ".join(factors)
        
        # Should include trend analysis with business implications
        assert "TREND ANALYSIS" in factors_text
        assert "growth opportunity" in factors_text or "market challenge" in factors_text
        assert "consider" in factors_text or "implement" in factors_text
    
    @pytest.mark.asyncio
    async def test_volatility_factor_analysis(self, insight_generator, volatile_pattern):
        """Test enhanced volatility factor analysis."""
        factors = await insight_generator._analyze_volatility_factors(volatile_pattern)
        
        assert len(factors) > 0
        
        factors_text = " ".join(factors)
        
        # Should include volatility analysis with risk levels
        assert "VOLATILITY" in factors_text
        assert any(level in factors_text for level in ["CRITICAL", "HIGH", "MODERATE", "LOW"])
        
        # Should include business impact and recommendations
        assert "inventory" in factors_text.lower()
        assert "implement" in factors_text.lower() or "increase" in factors_text.lower()
    
    @pytest.mark.asyncio
    async def test_data_quality_factor_analysis(self, insight_generator, seasonal_pattern):
        """Test enhanced data quality factor analysis."""
        factors = await insight_generator._analyze_data_quality_factors(seasonal_pattern)
        
        assert len(factors) > 0
        
        factors_text = " ".join(factors)
        
        # Should include data quality assessment
        assert "DATA QUALITY" in factors_text
        assert any(quality in factors_text for quality in ["LIMITED", "MODERATE", "GOOD", "EXCELLENT"])
        
        # Should include actionable recommendations
        assert "collect" in factors_text.lower() or "leverage" in factors_text.lower() or "use" in factors_text.lower()
    
    @pytest.mark.asyncio
    async def test_factor_prioritization(self, insight_generator):
        """Test factor prioritization by business importance."""
        factors = [
            "🔥 CRITICAL: High impact factor with revenue opportunity",
            "📊 MODERATE: Standard factor with operational impact",
            "⚠️ HIGH: Important factor requiring attention",
            "📋 MINOR: Low impact factor for monitoring"
        ]
        
        # Create a mock pattern for prioritization
        mock_pattern = DemandPattern(
            product_id="TEST",
            pattern_type="test",
            description="Test pattern",
            confidence_level=ConfidenceLevel.HIGH,
            volatility_score=0.5,
            supporting_data_points=100,
            date_range_start=date.today() - timedelta(days=365),
            date_range_end=date.today()
        )
        
        prioritized = await insight_generator._prioritize_key_factors(factors, mock_pattern)
        
        # Should prioritize critical factors first
        assert "CRITICAL" in prioritized[0]
        assert "HIGH" in prioritized[1]
    
    @pytest.mark.asyncio
    async def test_business_language_enhancement(self, insight_generator):
        """Test business-friendly language enhancement."""
        technical_text = "High volatility with increasing trend shows seasonal factor impact on data points"
        
        enhanced_text = await insight_generator.enhance_business_language(technical_text)
        
        # Should replace technical terms with business terms
        assert "growing market demand" in enhanced_text
        assert "**seasonal**" in enhanced_text
        assert "**sales**" in enhanced_text
    
    @pytest.mark.asyncio
    async def test_factor_importance_analysis(self, insight_generator):
        """Test factor importance analysis generation."""
        factors = [
            "🔥 CRITICAL: Revenue opportunity requiring immediate attention",
            "⚠️ HIGH: Strategic planning factor for long-term growth",
            "📊 MODERATE: Operational efficiency consideration",
            "🏆 STRATEGIC: Long-term capacity planning factor"
        ]
        
        analysis = await insight_generator.generate_factor_importance_analysis(factors)
        
        # Should categorize factors properly
        assert analysis['total_factors'] == 4
        assert len(analysis['critical_factors']) == 1
        assert len(analysis['high_impact_factors']) == 1
        assert len(analysis['strategic_factors']) >= 1
        
        # Should provide summary and recommendations
        assert analysis['importance_summary']
        assert len(analysis['priority_recommendations']) > 0
        assert "critical" in analysis['importance_summary'].lower()
    
    @pytest.mark.asyncio
    async def test_enhanced_key_factors_integration(self, insight_generator, sample_sales_data, seasonal_pattern):
        """Test integration of enhanced key factor analysis in insight generation."""
        insight_generator.set_sales_data(sample_sales_data)
        
        insight = await insight_generator.explain_pattern(seasonal_pattern)
        
        # Should include enhanced key factors with highlighting
        factors_text = " ".join(insight.key_factors)
        
        # Should have priority indicators
        assert any(indicator in factors_text for indicator in ["CRITICAL", "HIGH", "MODERATE", "STRATEGIC"])
        
        # Should have business context
        assert any(context in factors_text.lower() for context in ["opportunity", "planning", "efficiency", "growth"])
        
        # Should have visual indicators (emojis)
        assert any(emoji in factors_text for emoji in ["🔥", "⚠️", "📊", "📈", "📉", "✅", "🏆"])
    
    @pytest.mark.asyncio
    async def test_identify_key_factors_trending(self, insight_generator, trending_pattern):
        """Test key factor identification for trending patterns."""
        key_factors = await insight_generator.identify_key_factors(trending_pattern)
        
        assert len(key_factors) > 0
        
        # Should identify trend factors
        factors_text = " ".join(key_factors).lower()
        assert "increasing" in factors_text or "trend" in factors_text
    
    @pytest.mark.asyncio
    async def test_identify_key_factors_volatile(self, insight_generator, volatile_pattern):
        """Test key factor identification for volatile patterns."""
        key_factors = await insight_generator.identify_key_factors(volatile_pattern)
        
        assert len(key_factors) > 0
        
        # Should identify volatility factors
        factors_text = " ".join(key_factors).lower()
        assert any(keyword in factors_text for keyword in ["volatility", "variability", "flexible", "high"])
    
    @pytest.mark.asyncio
    async def test_pattern_categorization(self, insight_generator, seasonal_pattern, trending_pattern, volatile_pattern):
        """Test pattern categorization logic."""
        # Test seasonal categorization
        seasonal_category = await insight_generator._categorize_pattern(seasonal_pattern)
        assert seasonal_category == 'seasonal'
        
        # Test trending categorization
        trending_category = await insight_generator._categorize_pattern(trending_pattern)
        assert trending_category == 'trending'
        
        # Test volatile categorization
        volatile_category = await insight_generator._categorize_pattern(volatile_pattern)
        assert volatile_category == 'volatile'
    
    @pytest.mark.asyncio
    async def test_pattern_strength_assessment(self, insight_generator, seasonal_pattern, trending_pattern, volatile_pattern):
        """Test pattern strength assessment."""
        # High confidence should be strong
        high_strength = await insight_generator._assess_pattern_strength(seasonal_pattern)
        assert high_strength == "strong"
        
        # Medium confidence should be moderate
        medium_strength = await insight_generator._assess_pattern_strength(trending_pattern)
        assert medium_strength == "moderate"
        
        # Low confidence should be weak
        weak_strength = await insight_generator._assess_pattern_strength(volatile_pattern)
        assert weak_strength == "weak"
    
    @pytest.mark.asyncio
    async def test_product_context_extraction(self, insight_generator, sample_sales_data):
        """Test product context extraction from sales data."""
        insight_generator.set_sales_data(sample_sales_data)
        
        context = await insight_generator._get_product_context("TEST_PRODUCT_001")
        
        assert context['product_name'] == "Premium Electronics Item"
        assert context['category'] == "electronics"
        assert context['avg_price'] > 0
        assert context['total_sales'] > 0
    
    @pytest.mark.asyncio
    async def test_product_context_no_data(self, insight_generator):
        """Test product context extraction with no sales data."""
        context = await insight_generator._get_product_context("NONEXISTENT_PRODUCT")
        
        assert "NONEXISTENT_PRODUCT" in context['product_name']
        assert context['category'] == 'general'
        assert context['avg_price'] == 0
        assert context['total_sales'] == 0
    
    @pytest.mark.asyncio
    async def test_supporting_evidence_generation(self, insight_generator, seasonal_pattern):
        """Test supporting evidence generation."""
        evidence = await insight_generator._generate_supporting_evidence(seasonal_pattern)
        
        assert len(evidence) > 0
        assert len(evidence) <= 4  # Should limit to 4 pieces
        
        # Should mention data volume
        evidence_text = " ".join(evidence).lower()
        assert "100" in evidence_text or "transactions" in evidence_text
        
        # Should mention confidence level
        assert any("confidence" in ev.lower() for ev in evidence)
    
    @pytest.mark.asyncio
    async def test_recommended_actions_seasonal(self, insight_generator, seasonal_pattern):
        """Test recommended actions for seasonal patterns."""
        recommendations = await insight_generator._generate_recommended_actions(
            seasonal_pattern, 'seasonal', ["Diwali creates 60% demand boost"]
        )
        
        assert len(recommendations) > 0
        assert len(recommendations) <= 4  # Should limit to 4
        
        # Should include seasonal-specific recommendations
        rec_text = " ".join(recommendations).lower()
        assert any(keyword in rec_text for keyword in ["seasonal", "inventory", "peak", "buildup"])
    
    @pytest.mark.asyncio
    async def test_recommended_actions_trending(self, insight_generator, trending_pattern):
        """Test recommended actions for trending patterns."""
        recommendations = await insight_generator._generate_recommended_actions(
            trending_pattern, 'trending', ["Strong increasing trend in demand"]
        )
        
        assert len(recommendations) > 0
        
        # Should include trend-specific recommendations
        rec_text = " ".join(recommendations).lower()
        assert any(keyword in rec_text for keyword in ["increase", "growing", "expand"])
    
    @pytest.mark.asyncio
    async def test_business_impact_assessment(self, insight_generator, seasonal_pattern):
        """Test business impact assessment."""
        impact = await insight_generator._assess_business_impact(seasonal_pattern, 'seasonal')
        
        assert len(impact) > 0
        assert impact.endswith('.')  # Should be properly formatted
        
        # Should mention revenue impact for seasonal patterns
        assert "revenue" in impact.lower() or "seasonal" in impact.lower()
    
    @pytest.mark.asyncio
    async def test_data_sources_identification(self, insight_generator, seasonal_pattern):
        """Test data sources identification."""
        sources = await insight_generator._identify_data_sources(seasonal_pattern)
        
        assert len(sources) > 0
        
        # Should include basic sources
        sources_text = " ".join(sources).lower()
        assert "sales" in sources_text
        assert "inventory" in sources_text or "transaction" in sources_text
        
        # Should include seasonal sources for seasonal patterns
        assert "seasonal" in sources_text or "festival" in sources_text
    
    def test_confidence_level_conversion(self, insight_generator):
        """Test confidence score to level conversion."""
        # High confidence
        high_level = insight_generator._score_to_confidence_level(0.85)
        assert high_level == ConfidenceLevel.HIGH
        
        # Medium confidence
        medium_level = insight_generator._score_to_confidence_level(0.65)
        assert medium_level == ConfidenceLevel.MEDIUM
        
        # Low confidence
        low_level = insight_generator._score_to_confidence_level(0.35)
        assert low_level == ConfidenceLevel.LOW
    
    @pytest.mark.asyncio
    async def test_insight_prioritization(self, insight_generator):
        """Test insight prioritization logic."""
        # Create insights with different quality levels
        high_quality = ExplainableInsight(
            title="High Quality Insight",
            description="Comprehensive analysis with detailed business implications",
            confidence_level=ConfidenceLevel.HIGH,
            supporting_evidence=["Evidence 1", "Evidence 2", "Evidence 3"],
            key_factors=["Factor 1", "Factor 2"],
            business_impact="Significant business impact with detailed explanation of revenue implications and strategic considerations",
            recommended_actions=["Action 1", "Action 2", "Action 3"],
            data_sources=["Source 1", "Source 2"]
        )
        
        low_quality = ExplainableInsight(
            title="Low Quality Insight",
            description="Basic analysis",
            confidence_level=ConfidenceLevel.LOW,
            supporting_evidence=["Evidence"],
            key_factors=["Factor"],
            business_impact="Some impact",
            recommended_actions=["Action"],
            data_sources=["Source"]
        )
        
        insights = [low_quality, high_quality]  # Intentionally wrong order
        prioritized = await insight_generator._prioritize_insights(insights)
        
        # High quality should come first
        assert prioritized[0].title == "High Quality Insight"
        assert prioritized[1].title == "Low Quality Insight"
    
    @pytest.mark.asyncio
    async def test_insight_caching(self, insight_generator, sample_sales_data, seasonal_pattern):
        """Test insight caching functionality."""
        insight_generator.set_sales_data(sample_sales_data)
        
        # Generate insights (which does the caching)
        insights = await insight_generator.generate_insights([seasonal_pattern])
        
        # Check that insight is cached
        cache_key = f"{seasonal_pattern.product_id}_{seasonal_pattern.pattern_type}_{seasonal_pattern.id}"
        assert cache_key in insight_generator.insight_cache
        assert insight_generator.insight_cache[cache_key].title == insights[0].title
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_pattern(self, insight_generator):
        """Test error handling with edge case pattern data."""
        # Create pattern with minimal valid data but challenging characteristics
        edge_case_pattern = DemandPattern(
            product_id="EDGE_CASE_PRODUCT",
            pattern_type="unknown_type",
            description="Edge case pattern with minimal data",
            confidence_level=ConfidenceLevel.LOW,
            volatility_score=0.5,
            supporting_data_points=1,  # Minimal supporting data
            date_range_start=date.today() - timedelta(days=1),
            date_range_end=date.today()
        )
        
        # Should handle gracefully and not crash
        insight = await insight_generator.explain_pattern(edge_case_pattern)
        
        # Should still generate some insight, even if basic
        assert insight.title
        assert insight.description
        assert len(insight.key_factors) > 0
        assert len(insight.recommended_actions) > 0
    
    def test_set_sales_data(self, insight_generator, sample_sales_data):
        """Test setting sales data."""
        insight_generator.set_sales_data(sample_sales_data)
        
        assert len(insight_generator.sales_data) == len(sample_sales_data)
        assert insight_generator.sales_data[0].product_id == "TEST_PRODUCT_001"
    
    def test_set_storage_manager(self, insight_generator):
        """Test setting storage manager."""
        mock_storage = Mock()
        insight_generator.set_storage_manager(mock_storage)
        
        assert insight_generator.storage_manager == mock_storage
    
    @pytest.mark.asyncio
    async def test_seasonal_event_mapping(self, insight_generator):
        """Test seasonal event name mapping."""
        # Test that seasonal events are properly mapped to business-friendly names
        assert 'diwali' in insight_generator.seasonal_events
        assert insight_generator.seasonal_events['diwali'] == 'Diwali festival season'
        assert 'christmas' in insight_generator.seasonal_events
        assert 'wedding_season' in insight_generator.seasonal_events
    
    @pytest.mark.asyncio
    async def test_template_loading(self, insight_generator):
        """Test pattern template loading."""
        templates = insight_generator.pattern_templates
        
        # Should have templates for all major pattern types
        assert 'seasonal' in templates
        assert 'trending' in templates
        assert 'volatile' in templates
        assert 'stable' in templates
        
        # Each template should have required fields
        for pattern_type, template in templates.items():
            assert 'title_template' in template
            assert 'description_template' in template
            assert 'evidence_template' in template
            assert 'recommendation_template' in template


class TestInsightGeneratorEdgeCases:
    """Test edge cases and error conditions for InsightGenerator."""
    
    @pytest.fixture
    def insight_generator(self):
        return InsightGenerator()
    
    @pytest.mark.asyncio
    async def test_minimal_pattern_data(self, insight_generator):
        """Test insight generation with minimal pattern data."""
        minimal_pattern = DemandPattern(
            product_id="MIN_PRODUCT",
            pattern_type="basic",
            description="Minimal pattern",
            confidence_level=ConfidenceLevel.LOW,
            volatility_score=0.5,
            supporting_data_points=1,
            date_range_start=date.today() - timedelta(days=1),
            date_range_end=date.today()
        )
        
        insight = await insight_generator.explain_pattern(minimal_pattern)
        
        # Should still generate valid insight
        assert insight.title
        assert insight.description
        assert len(insight.key_factors) > 0
        assert len(insight.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_very_high_volatility(self, insight_generator):
        """Test handling of extremely high volatility patterns."""
        high_volatility_pattern = DemandPattern(
            product_id="VOLATILE_PRODUCT",
            pattern_type="volatile",
            description="Extremely volatile pattern",
            confidence_level=ConfidenceLevel.LOW,
            volatility_score=0.95,  # Very high volatility
            supporting_data_points=20,
            date_range_start=date.today() - timedelta(days=30),
            date_range_end=date.today()
        )
        
        insight = await insight_generator.explain_pattern(high_volatility_pattern)
        
        # Should handle extreme volatility appropriately
        assert "volatility" in insight.title.lower() or "volatile" in insight.title.lower()
        
        # Should provide appropriate recommendations for high volatility
        rec_text = " ".join(insight.recommended_actions).lower()
        assert any(keyword in rec_text for keyword in ["flexible", "safety", "buffer", "short-term"])
    
    @pytest.mark.asyncio
    async def test_zero_volatility(self, insight_generator):
        """Test handling of zero volatility (perfectly stable) patterns."""
        zero_volatility_pattern = DemandPattern(
            product_id="STABLE_PRODUCT",
            pattern_type="stable",
            description="Perfectly stable pattern",
            confidence_level=ConfidenceLevel.HIGH,
            volatility_score=0.0,  # Zero volatility
            supporting_data_points=100,
            date_range_start=date.today() - timedelta(days=365),
            date_range_end=date.today()
        )
        
        insight = await insight_generator.explain_pattern(zero_volatility_pattern)
        
        # Should recognize perfect stability
        assert "stable" in insight.title.lower() or "consistent" in insight.description.lower()
        
        # Should provide stability-appropriate recommendations
        rec_text = " ".join(insight.recommended_actions).lower()
        assert any(keyword in rec_text for keyword in ["consistent", "steady", "standard", "reliable"])
    
    @pytest.mark.asyncio
    async def test_future_date_range(self, insight_generator):
        """Test handling of patterns with future date ranges."""
        future_pattern = DemandPattern(
            product_id="FUTURE_PRODUCT",
            pattern_type="forecast",
            description="Future pattern",
            confidence_level=ConfidenceLevel.MEDIUM,
            volatility_score=0.3,
            supporting_data_points=50,
            date_range_start=date.today() + timedelta(days=1),  # Future start
            date_range_end=date.today() + timedelta(days=30)    # Future end
        )
        
        # Should handle gracefully
        insight = await insight_generator.explain_pattern(future_pattern)
        assert insight.title
        assert insight.description
    
    @pytest.mark.asyncio
    async def test_extreme_seasonal_factors(self, insight_generator):
        """Test handling of extreme seasonal factors."""
        extreme_seasonal_pattern = DemandPattern(
            product_id="EXTREME_SEASONAL",
            pattern_type="seasonal",
            description="Extreme seasonal pattern",
            confidence_level=ConfidenceLevel.HIGH,
            seasonal_factors={
                'diwali': 5.0,    # 400% boost - extreme
                'summer': 0.1,    # 90% drop - extreme
                'normal': 1.0
            },
            volatility_score=0.4,
            supporting_data_points=200,
            date_range_start=date.today() - timedelta(days=730),
            date_range_end=date.today()
        )
        
        insight = await insight_generator.explain_pattern(extreme_seasonal_pattern)
        
        # Should handle extreme values appropriately
        assert "seasonal" in insight.title.lower()
        
        # Should mention the extreme boost in factors or description
        factors_and_desc = (insight.description + " " + " ".join(insight.key_factors)).lower()
        assert "400%" in factors_and_desc or "500%" in factors_and_desc or "diwali" in factors_and_desc
    
    @pytest.mark.asyncio
    async def test_empty_seasonal_factors(self, insight_generator):
        """Test handling of patterns with empty seasonal factors."""
        no_seasonal_pattern = DemandPattern(
            product_id="NO_SEASONAL",
            pattern_type="basic_trend",
            description="No seasonal pattern",
            confidence_level=ConfidenceLevel.MEDIUM,
            seasonal_factors={},  # Empty seasonal factors
            trend_direction="stable",
            volatility_score=0.2,
            supporting_data_points=50,
            date_range_start=date.today() - timedelta(days=180),
            date_range_end=date.today()
        )
        
        insight = await insight_generator.explain_pattern(no_seasonal_pattern)
        
        # Should still generate meaningful insight
        assert insight.title
        assert insight.description
        assert len(insight.key_factors) > 0
        
        # Should not mention specific seasonal events
        factors_text = " ".join(insight.key_factors).lower()
        assert not any(event in factors_text for event in ['diwali', 'christmas', 'holi'])