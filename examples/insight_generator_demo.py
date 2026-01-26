"""
Demo script for the Insight Generator component.

This script demonstrates how to use the InsightGenerator to create
human-readable insights from demand patterns.
"""

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal

from marketpulse_ai.components.insight_generator import InsightGenerator
from marketpulse_ai.core.models import (
    DemandPattern, 
    ConfidenceLevel,
    SalesDataPoint
)


async def main():
    """Demonstrate insight generation functionality."""
    print("=== MarketPulse AI - Insight Generator Demo ===\n")
    
    # Initialize the insight generator
    insight_generator = InsightGenerator()
    
    # Create sample sales data for context
    print("1. Creating sample sales data...")
    sales_data = []
    base_date = date.today() - timedelta(days=365)
    
    for i in range(100):
        current_date = base_date + timedelta(days=i * 3)
        
        # Simulate seasonal boost during Diwali (October-November)
        quantity = 45
        if current_date.month in [10, 11]:
            quantity = 75  # 67% boost during Diwali season
        elif current_date.month in [6, 7, 8]:
            quantity = 30  # Lower sales during monsoon
        
        sales_data.append(SalesDataPoint(
            product_id="DEMO_ELECTRONICS_001",
            product_name="Premium Smart TV",
            category="electronics",
            mrp=Decimal("45000.00"),
            selling_price=Decimal("42000.00"),
            quantity_sold=quantity,
            sale_date=current_date,
            store_location="MUMBAI_MAIN"
        ))
    
    insight_generator.set_sales_data(sales_data)
    print(f"   ✓ Created {len(sales_data)} sales records with seasonal patterns\n")
    
    # Create sample demand patterns
    print("2. Creating demand patterns...")
    
    # Seasonal pattern
    seasonal_pattern = DemandPattern(
        product_id="DEMO_ELECTRONICS_001",
        pattern_type="seasonal",
        description="Strong seasonal pattern with festival correlation",
        confidence_level=ConfidenceLevel.HIGH,
        seasonal_factors={
            'diwali': 1.67,      # 67% boost during Diwali
            'monsoon': 0.67,     # 33% drop during monsoon
            'winter': 1.15,      # 15% boost in winter
            'summer': 0.85       # 15% drop in summer
        },
        trend_direction="stable",
        volatility_score=0.35,
        supporting_data_points=100,
        date_range_start=base_date,
        date_range_end=date.today()
    )
    
    # Trending pattern
    trending_pattern = DemandPattern(
        product_id="DEMO_ELECTRONICS_002",
        pattern_type="basic_trend",
        description="Increasing demand trend",
        confidence_level=ConfidenceLevel.MEDIUM,
        seasonal_factors={},
        trend_direction="increasing",
        volatility_score=0.25,
        supporting_data_points=75,
        date_range_start=base_date + timedelta(days=90),
        date_range_end=date.today()
    )
    
    # Volatile pattern
    volatile_pattern = DemandPattern(
        product_id="DEMO_ELECTRONICS_003",
        pattern_type="volatile",
        description="High volatility demand pattern",
        confidence_level=ConfidenceLevel.LOW,
        seasonal_factors={},
        trend_direction="stable",
        volatility_score=0.75,
        supporting_data_points=50,
        date_range_start=base_date + timedelta(days=180),
        date_range_end=date.today()
    )
    
    patterns = [seasonal_pattern, trending_pattern, volatile_pattern]
    print(f"   ✓ Created {len(patterns)} demand patterns (seasonal, trending, volatile)\n")
    
    # Generate insights
    print("3. Generating insights from patterns...")
    insights = await insight_generator.generate_insights(patterns)
    print(f"   ✓ Generated {len(insights)} insights\n")
    
    # Display insights
    print("4. Generated Insights:\n")
    print("=" * 80)
    
    for i, insight in enumerate(insights, 1):
        print(f"\n📊 INSIGHT #{i}")
        print(f"Title: {insight.title}")
        print(f"Confidence: {insight.confidence_level.value.upper()}")
        print(f"\nDescription:")
        print(f"   {insight.description}")
        
        print(f"\n🔍 Key Factors:")
        for factor in insight.key_factors:
            print(f"   • {factor}")
        
        print(f"\n📈 Business Impact:")
        print(f"   {insight.business_impact}")
        
        print(f"\n💡 Recommended Actions:")
        for action in insight.recommended_actions:
            print(f"   • {action}")
        
        print(f"\n📋 Supporting Evidence:")
        for evidence in insight.supporting_evidence:
            print(f"   • {evidence}")
        
        print(f"\n📊 Data Sources:")
        for source in insight.data_sources:
            print(f"   • {source}")
        
        print("\n" + "=" * 80)
    
    # Demonstrate individual pattern explanation
    print("\n5. Detailed Pattern Explanation Demo:")
    print("\n🎯 Explaining Seasonal Pattern in Detail...")
    
    detailed_insight = await insight_generator.explain_pattern(seasonal_pattern)
    confidence_score = await insight_generator.calculate_confidence(detailed_insight)
    
    print(f"\nDetailed Analysis:")
    print(f"   Title: {detailed_insight.title}")
    print(f"   Confidence Score: {confidence_score:.3f}")
    print(f"   Confidence Level: {detailed_insight.confidence_level.value}")
    
    print(f"\n   Key Factors Identified:")
    key_factors = await insight_generator.identify_key_factors(seasonal_pattern)
    for factor in key_factors:
        print(f"   • {factor}")
    
    # Show factor importance analysis
    factor_analysis = await insight_generator.generate_factor_importance_analysis(key_factors)
    print(f"\n   Factor Importance Analysis:")
    print(f"   • Total Factors: {factor_analysis['total_factors']}")
    print(f"   • Critical Factors: {len(factor_analysis['critical_factors'])}")
    print(f"   • High Impact Factors: {len(factor_analysis['high_impact_factors'])}")
    print(f"   • Summary: {factor_analysis['importance_summary']}")
    
    if factor_analysis['priority_recommendations']:
        print(f"   • Priority Recommendations:")
        for rec in factor_analysis['priority_recommendations']:
            print(f"     - {rec}")
    
    # Show business language enhancement
    technical_text = "High volatility with increasing trend shows seasonal factor impact"
    enhanced_text = await insight_generator.enhance_business_language(technical_text)
    print(f"\n   Business Language Enhancement Demo:")
    print(f"   • Original: {technical_text}")
    print(f"   • Enhanced: {enhanced_text}")
    
    print(f"\n   Pattern Category: {await insight_generator._categorize_pattern(seasonal_pattern)}")
    print(f"   Pattern Strength: {await insight_generator._assess_pattern_strength(seasonal_pattern)}")
    
    # Show product context
    product_context = await insight_generator._get_product_context("DEMO_ELECTRONICS_001")
    print(f"\n   Product Context:")
    print(f"   • Name: {product_context['product_name']}")
    print(f"   • Category: {product_context['category']}")
    print(f"   • Average Price: ₹{product_context['avg_price']:,.2f}")
    print(f"   • Total Sales: {product_context['total_sales']} units")
    
    print("\n6. Insight Prioritization Demo:")
    print("\n🏆 Insights Ranked by Priority:")
    
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight.title} ({insight.confidence_level.value})")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   Generated {len(insights)} actionable business insights")
    print(f"   All insights include explanations, evidence, and recommendations")
    print(f"   Insights are prioritized by confidence and business impact")


if __name__ == "__main__":
    asyncio.run(main())