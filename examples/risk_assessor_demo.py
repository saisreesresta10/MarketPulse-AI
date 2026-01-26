"""
Risk Assessor Demo Script

This script demonstrates the Risk Assessor component functionality
including overstock risk, understock risk, demand volatility calculation,
and seasonal adjustments.
"""

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
import numpy as np

from marketpulse_ai.components.risk_assessor import RiskAssessor
from marketpulse_ai.core.models import SalesDataPoint, DemandPattern, ConfidenceLevel


def create_sample_data():
    """Create sample sales data for demonstration."""
    print("📊 Creating sample sales data...")
    
    base_date = date.today() - timedelta(days=180)  # 6 months of data
    sales_data = []
    
    # Create sales data for a smartphone product with seasonal patterns
    for i in range(180):
        current_date = base_date + timedelta(days=i)
        
        # Base demand with seasonal variation
        base_qty = 25
        
        # Festival season boost (Oct-Dec)
        if current_date.month in [10, 11, 12]:
            seasonal_factor = 1.8
        # Summer slowdown (Jun-Aug)
        elif current_date.month in [6, 7, 8]:
            seasonal_factor = 0.7
        else:
            seasonal_factor = 1.0
        
        # Weekly pattern (higher on weekends)
        weekly_factor = 1.3 if current_date.weekday() >= 5 else 1.0
        
        # Random variation
        random_factor = 0.8 + np.random.random() * 0.4
        
        quantity = int(base_qty * seasonal_factor * weekly_factor * random_factor)
        quantity = max(1, quantity)
        
        sales_data.append(SalesDataPoint(
            product_id="SMARTPHONE_001",
            product_name="Premium Smartphone",
            category="electronics",
            mrp=Decimal("50000.00"),
            selling_price=Decimal("45000.00"),
            quantity_sold=quantity,
            sale_date=current_date,
            store_location="MAIN_STORE",
            seasonal_event="diwali" if current_date.month == 11 and current_date.day <= 7 else None
        ))
    
    print(f"✅ Created {len(sales_data)} sales records")
    return sales_data


def create_sample_demand_patterns():
    """Create sample demand patterns for demonstration."""
    pattern = DemandPattern(
        product_id="SMARTPHONE_001",
        pattern_type="seasonal_cyclical",
        description="Smartphone shows strong seasonal pattern with festival peaks",
        confidence_level=ConfidenceLevel.HIGH,
        seasonal_factors={
            'diwali': 1.8,
            'summer': 0.7,
            'winter': 1.2,
            'weekend_boost': 1.3
        },
        trend_direction="stable",
        volatility_score=0.35,
        supporting_data_points=180,
        date_range_start=date.today() - timedelta(days=180),
        date_range_end=date.today()
    )
    
    return {"SMARTPHONE_001": [pattern]}


async def demonstrate_overstock_risk():
    """Demonstrate overstock risk assessment."""
    print("\n🔴 OVERSTOCK RISK ASSESSMENT")
    print("=" * 50)
    
    risk_assessor = RiskAssessor()
    sales_data = create_sample_data()
    demand_patterns = create_sample_demand_patterns()
    
    risk_assessor.set_sales_data(sales_data)
    risk_assessor.set_demand_patterns(demand_patterns)
    
    # Test different inventory levels
    inventory_levels = [50, 150, 300, 500]
    
    for inventory in inventory_levels:
        print(f"\n📦 Testing inventory level: {inventory} units")
        
        assessment = await risk_assessor.assess_overstock_risk("SMARTPHONE_001", inventory)
        
        print(f"   Risk Level: {assessment.risk_level.value.upper()}")
        print(f"   Risk Score: {assessment.risk_score:.3f}")
        print(f"   Contributing Factors:")
        for factor in assessment.contributing_factors[:3]:  # Show top 3
            print(f"     • {factor}")
        print(f"   Key Mitigation Suggestion: {assessment.mitigation_suggestions[0]}")


async def demonstrate_understock_risk():
    """Demonstrate understock risk assessment."""
    print("\n🟡 UNDERSTOCK RISK ASSESSMENT")
    print("=" * 50)
    
    risk_assessor = RiskAssessor()
    sales_data = create_sample_data()
    demand_patterns = create_sample_demand_patterns()
    
    risk_assessor.set_sales_data(sales_data)
    risk_assessor.set_demand_patterns(demand_patterns)
    
    # Test different inventory levels
    inventory_levels = [5, 15, 30, 60]
    
    for inventory in inventory_levels:
        print(f"\n📦 Testing inventory level: {inventory} units")
        
        assessment = await risk_assessor.assess_understock_risk("SMARTPHONE_001", inventory)
        
        print(f"   Risk Level: {assessment.risk_level.value.upper()}")
        print(f"   Risk Score: {assessment.risk_score:.3f}")
        print(f"   Contributing Factors:")
        for factor in assessment.contributing_factors[:3]:  # Show top 3
            print(f"     • {factor}")
        print(f"   Key Mitigation Suggestion: {assessment.mitigation_suggestions[0]}")


async def demonstrate_volatility_calculation():
    """Demonstrate demand volatility calculation."""
    print("\n📈 DEMAND VOLATILITY ANALYSIS")
    print("=" * 50)
    
    risk_assessor = RiskAssessor()
    
    # Create different volatility scenarios
    scenarios = [
        ("Low Volatility", [25, 26, 24, 25, 27, 25, 24, 26, 25, 25] * 10),
        ("Medium Volatility", [20, 30, 25, 35, 15, 40, 25, 20, 35, 30] * 10),
        ("High Volatility", [10, 50, 5, 60, 15, 45, 8, 55, 12, 48] * 10)
    ]
    
    for scenario_name, quantities in scenarios:
        print(f"\n📊 {scenario_name} Scenario:")
        
        # Create sales data for this scenario
        base_date = date.today() - timedelta(days=len(quantities))
        scenario_data = []
        
        for i, qty in enumerate(quantities):
            scenario_data.append(SalesDataPoint(
                product_id=f"PRODUCT_{scenario_name.upper().replace(' ', '_')}",
                product_name=f"{scenario_name} Product",
                category="test",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=qty,
                sale_date=base_date + timedelta(days=i),
                store_location="TEST_STORE"
            ))
        
        risk_assessor.set_sales_data(scenario_data)
        
        volatility = await risk_assessor.calculate_demand_volatility(f"PRODUCT_{scenario_name.upper().replace(' ', '_')}")
        
        print(f"   Volatility Score: {volatility:.3f}")
        print(f"   Mean Quantity: {np.mean(quantities):.1f}")
        print(f"   Std Deviation: {np.std(quantities):.1f}")
        print(f"   Coefficient of Variation: {np.std(quantities)/np.mean(quantities):.3f}")


async def demonstrate_seasonal_adjustments():
    """Demonstrate seasonal risk adjustments."""
    print("\n🌟 SEASONAL RISK ADJUSTMENTS")
    print("=" * 50)
    
    risk_assessor = RiskAssessor()
    sales_data = create_sample_data()
    demand_patterns = create_sample_demand_patterns()
    
    risk_assessor.set_sales_data(sales_data)
    risk_assessor.set_demand_patterns(demand_patterns)
    
    # Get base assessment
    base_assessment = await risk_assessor.assess_overstock_risk("SMARTPHONE_001", 200)
    
    print(f"\n📊 Base Assessment (No Seasonal Events):")
    print(f"   Risk Level: {base_assessment.risk_level.value.upper()}")
    print(f"   Risk Score: {base_assessment.risk_score:.3f}")
    
    # Test different seasonal scenarios
    seasonal_scenarios = [
        (["diwali"], "Festival Season (High Demand)"),
        (["summer"], "Summer Season (Low Demand)"),
        (["diwali", "winter"], "Multiple Events")
    ]
    
    for events, description in seasonal_scenarios:
        print(f"\n🎯 {description}:")
        
        adjusted_assessment = await risk_assessor.adjust_for_seasonal_events(
            base_assessment, events
        )
        
        print(f"   Adjusted Risk Level: {adjusted_assessment.risk_level.value.upper()}")
        print(f"   Adjusted Risk Score: {adjusted_assessment.risk_score:.3f}")
        print(f"   Score Change: {adjusted_assessment.risk_score - base_assessment.risk_score:+.3f}")
        
        if adjusted_assessment.seasonal_adjustments:
            print(f"   Seasonal Factors:")
            for event, factor in adjusted_assessment.seasonal_adjustments.items():
                print(f"     • {event}: {factor:.2f}")


async def demonstrate_early_warnings():
    """Demonstrate early warning system."""
    print("\n⚠️  EARLY WARNING SYSTEM")
    print("=" * 50)
    
    risk_assessor = RiskAssessor()
    sales_data = create_sample_data()
    
    risk_assessor.set_sales_data(sales_data)
    
    # Create various risk scenarios
    assessments = []
    
    # High overstock risk
    high_overstock = await risk_assessor.assess_overstock_risk("SMARTPHONE_001", 600)
    assessments.append(high_overstock)
    
    # High understock risk
    high_understock = await risk_assessor.assess_understock_risk("SMARTPHONE_001", 3)
    assessments.append(high_understock)
    
    # Medium risks
    medium_overstock = await risk_assessor.assess_overstock_risk("SMARTPHONE_001", 150)
    assessments.append(medium_overstock)
    
    medium_understock = await risk_assessor.assess_understock_risk("SMARTPHONE_001", 20)
    assessments.append(medium_understock)
    
    print(f"\n📋 Assessing {len(assessments)} risk scenarios...")
    
    # Generate early warnings
    warned_assessments = await risk_assessor.generate_early_warnings(assessments)
    
    for i, assessment in enumerate(warned_assessments, 1):
        warning_status = "🚨 WARNING TRIGGERED" if assessment.early_warning_triggered else "✅ No Warning"
        
        print(f"\n{i}. {assessment.risk_type.upper()} Risk Assessment:")
        print(f"   Risk Level: {assessment.risk_level.value.upper()}")
        print(f"   Risk Score: {assessment.risk_score:.3f}")
        print(f"   Early Warning: {warning_status}")
        
        if assessment.early_warning_triggered:
            print(f"   🔔 Warning Actions:")
            warning_suggestions = [s for s in assessment.mitigation_suggestions if "URGENT" in s.upper()]
            for suggestion in warning_suggestions[:2]:
                print(f"     • {suggestion}")


async def main():
    """Run the complete Risk Assessor demonstration."""
    print("🎯 MARKETPULSE AI - RISK ASSESSOR DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the Risk Assessor component capabilities:")
    print("• Overstock Risk Assessment")
    print("• Understock Risk Assessment") 
    print("• Demand Volatility Analysis")
    print("• Seasonal Risk Adjustments")
    print("• Early Warning System")
    print("=" * 60)
    
    try:
        await demonstrate_overstock_risk()
        await demonstrate_understock_risk()
        await demonstrate_volatility_calculation()
        await demonstrate_seasonal_adjustments()
        await demonstrate_early_warnings()
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("The Risk Assessor component is working correctly and provides:")
        print("• Comprehensive risk analysis with multiple algorithms")
        print("• Seasonal and trend-aware adjustments")
        print("• Actionable mitigation suggestions")
        print("• Early warning system for critical situations")
        print("• Integration with demand patterns and sales data")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())