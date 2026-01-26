"""
Decision Support Engine Demo Script.

This script demonstrates the capabilities of the Decision Support Engine
by generating comprehensive business recommendations for sample products.
"""

import asyncio
import json
from datetime import datetime, date, timedelta
from decimal import Decimal

from marketpulse_ai.components import (
    DataProcessor, RiskAssessor, ComplianceValidator, 
    InsightGenerator, DecisionSupportEngine
)
from marketpulse_ai.core.models import SalesDataPoint
from marketpulse_ai.storage.factory import StorageFactory


async def create_sample_data():
    """Create sample sales data for demonstration."""
    sample_data = []
    
    # Generate sample data for 3 products over 6 months
    products = [
        {"id": "ELEC001", "name": "Smartphone X1", "category": "Electronics", "mrp": Decimal("25000")},
        {"id": "FASH001", "name": "Designer Kurta", "category": "Fashion", "mrp": Decimal("2500")},
        {"id": "HOME001", "name": "LED TV 55inch", "category": "Home Appliances", "mrp": Decimal("45000")}
    ]
    
    base_date = date.today() - timedelta(days=180)
    
    for product in products:
        for day_offset in range(0, 180, 3):  # Every 3 days
            sale_date = base_date + timedelta(days=day_offset)
            
            # Simulate seasonal variations
            seasonal_multiplier = 1.0
            if sale_date.month in [10, 11]:  # Festival season
                seasonal_multiplier = 1.5
            elif sale_date.month in [6, 7]:  # Monsoon season
                seasonal_multiplier = 0.8
            
            # Simulate different demand patterns for different products
            base_quantity = 10
            if product["category"] == "Electronics":
                base_quantity = 15
            elif product["category"] == "Fashion":
                base_quantity = 25
            
            quantity = max(1, int(base_quantity * seasonal_multiplier))
            selling_price = product["mrp"] * Decimal("0.9")  # 10% discount from MRP
            
            data_point = SalesDataPoint(
                product_id=product["id"],
                product_name=product["name"],
                category=product["category"],
                mrp=product["mrp"],
                selling_price=selling_price,
                quantity_sold=quantity,
                sale_date=sale_date,
                store_location="STORE001",
                seasonal_event="diwali" if sale_date.month == 10 else None
            )
            sample_data.append(data_point)
    
    return sample_data


async def setup_components():
    """Set up all MarketPulse AI components."""
    print("🔧 Setting up MarketPulse AI components...")
    
    # Initialize storage
    factory = StorageFactory()
    storage_manager = factory.get_storage_manager()
    
    # Initialize components
    data_processor = DataProcessor(storage_manager)
    risk_assessor = RiskAssessor(storage_manager)
    compliance_validator = ComplianceValidator()
    insight_generator = InsightGenerator()
    
    # Initialize Decision Support Engine
    decision_engine = DecisionSupportEngine(
        data_processor=data_processor,
        risk_assessor=risk_assessor,
        compliance_validator=compliance_validator,
        insight_generator=insight_generator
    )
    
    print("✅ Components initialized successfully!")
    return decision_engine, data_processor


async def demonstrate_recommendation_generation(decision_engine, product_ids):
    """Demonstrate comprehensive recommendation generation."""
    print("\n🎯 Generating Comprehensive Business Recommendations...")
    print("=" * 60)
    
    # Create recommendation request
    request = {
        'product_ids': product_ids,
        'analysis_type': 'comprehensive',
        'time_horizon': '3_months',
        'inventory_levels': {
            'ELEC001': 150,
            'FASH001': 200,
            'HOME001': 80
        }
    }
    
    try:
        # Generate recommendations
        recommendations = await decision_engine.generate_recommendations(request)
        
        # Display summary
        summary = recommendations['summary']
        print(f"📊 Analysis Summary:")
        print(f"   • Total Recommendations: {summary['total_recommendations']}")
        print(f"   • High Priority: {summary['high_priority_count']}")
        print(f"   • Compliance Issues: {summary['compliance_issues']}")
        print(f"   • Critical Risks: {summary['critical_risks']}")
        
        # Display top recommendations
        print(f"\n🏆 Top Recommendations:")
        for i, rec in enumerate(recommendations['recommendations'][:3], 1):
            print(f"\n   {i}. Product: {rec['product_id']}")
            print(f"      Priority: {rec['priority'].upper()} ({rec.get('priority_score', 0):.1f} points)")
            print(f"      Optimal Discount: {rec['optimal_discount_percentage']:.1f}%")
            print(f"      Expected Revenue Impact: {rec['expected_impact']['revenue_impact_percentage']:+.1f}%")
            print(f"      Confidence: {rec['confidence_level'].title()}")
            
            # Display discount window
            window = rec['discount_window']
            start_date = datetime.fromisoformat(window['start_date']).strftime('%Y-%m-%d')
            end_date = datetime.fromisoformat(window['end_date']).strftime('%Y-%m-%d')
            print(f"      Discount Window: {start_date} to {end_date} ({window['duration_days']} days)")
        
        # Display insights
        print(f"\n💡 Key Insights:")
        for insight in recommendations['insights'][:2]:
            print(f"   • {insight['title']}")
            print(f"     {insight['description']}")
            print(f"     Confidence: {insight['confidence_level'].title()}")
        
        # Display risk assessments
        print(f"\n⚠️  Risk Assessments:")
        critical_risks = [r for r in recommendations['risk_assessments'] if r['risk_level'] == 'critical']
        high_risks = [r for r in recommendations['risk_assessments'] if r['risk_level'] == 'high']
        
        if critical_risks:
            print(f"   🔴 Critical Risks: {len(critical_risks)}")
            for risk in critical_risks[:2]:
                print(f"      • {risk['product_id']}: {risk['risk_type']} (Score: {risk['risk_score']:.2f})")
        
        if high_risks:
            print(f"   🟡 High Risks: {len(high_risks)}")
            for risk in high_risks[:2]:
                print(f"      • {risk['product_id']}: {risk['risk_type']} (Score: {risk['risk_score']:.2f})")
        
        # Display compliance status
        print(f"\n✅ Compliance Status:")
        compliant_count = len([c for c in recommendations['compliance_results'] if c['compliance_status'] == 'compliant'])
        total_checks = len(recommendations['compliance_results'])
        print(f"   • Compliant Recommendations: {compliant_count}/{total_checks}")
        
        non_compliant = [c for c in recommendations['compliance_results'] if c['compliance_status'] == 'non_compliant']
        if non_compliant:
            print(f"   • Compliance Issues Found:")
            for comp in non_compliant[:2]:
                print(f"     - Violations: {', '.join(comp['violations'])}")
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {str(e)}")
        return None


async def demonstrate_discount_optimization(decision_engine, product_ids):
    """Demonstrate discount strategy optimization."""
    print("\n💰 Discount Strategy Optimization...")
    print("=" * 50)
    
    try:
        # Optimize discount strategy
        optimization_result = await decision_engine.optimize_discount_strategy(product_ids)
        
        # Display strategy summary
        strategy = optimization_result['strategy_summary']
        print(f"📈 Optimization Summary:")
        print(f"   • Products Analyzed: {strategy['total_products']}")
        print(f"   • Successful Optimizations: {strategy['successful_optimizations']}")
        print(f"   • Average Discount: {strategy['average_discount']:.1f}%")
        print(f"   • Method: {strategy['optimization_method'].replace('_', ' ').title()}")
        
        # Display individual recommendations
        print(f"\n🎯 Optimized Discount Strategies:")
        for rec in optimization_result['recommendations']:
            print(f"\n   Product: {rec['product_id']}")
            print(f"   • Optimal Discount: {rec['optimal_discount_percentage']:.1f}%")
            print(f"   • Price Sensitivity: {rec['price_sensitivity_score']:.2f}")
            print(f"   • Priority: {rec['priority'].title()}")
            
            # Expected impact
            impact = rec['expected_impact']
            print(f"   • Expected Demand Increase: +{impact['demand_increase_percentage']:.1f}%")
            print(f"   • Expected Revenue Impact: {impact['revenue_impact_percentage']:+.1f}%")
            print(f"   • Inventory Turnover Improvement: +{impact['inventory_turnover_improvement']:.1f}%")
            
            # Supporting factors
            print(f"   • Supporting Factors:")
            for factor in rec['supporting_factors']:
                print(f"     - {factor}")
        
        return optimization_result
        
    except Exception as e:
        print(f"❌ Error optimizing discount strategy: {str(e)}")
        return None


async def demonstrate_business_impact_assessment(decision_engine, recommendations):
    """Demonstrate business impact assessment."""
    print("\n📊 Business Impact Assessment...")
    print("=" * 40)
    
    if not recommendations or not recommendations.get('recommendations'):
        print("❌ No recommendations available for impact assessment")
        return
    
    # Assess impact for top recommendation
    top_recommendation = recommendations['recommendations'][0]
    
    try:
        impact = await decision_engine.assess_business_impact(top_recommendation)
        
        print(f"🎯 Impact Analysis for Product: {top_recommendation['product_id']}")
        print(f"   Discount Strategy: {top_recommendation['optimal_discount_percentage']:.1f}%")
        
        # Revenue impact
        revenue = impact['revenue_impact']
        print(f"\n💰 Revenue Impact:")
        print(f"   • Base Revenue Estimate: ₹{revenue['base_revenue_estimate']:,}")
        print(f"   • Volume Increase: +{revenue['volume_increase_percentage']:.1f}%")
        print(f"   • Revenue Change: ₹{revenue['revenue_change_amount']:+,.0f} ({revenue['revenue_change_percentage']:+.1f}%)")
        print(f"   • Break-even Volume Increase: {revenue['break_even_volume_increase']:.1f}%")
        
        # Inventory impact
        inventory = impact['inventory_impact']
        print(f"\n📦 Inventory Impact:")
        print(f"   • Current Inventory: {inventory['current_inventory_estimate']} units")
        print(f"   • Turnover Improvement: +{inventory['turnover_improvement_percentage']:.1f}%")
        print(f"   • Days to Clear Reduction: -{inventory['days_to_clear_reduction']:.1f} days")
        print(f"   • Risk Reduction: {inventory['inventory_risk_reduction'].title()}")
        print(f"   • Optimal Reorder Timing: {inventory['optimal_reorder_timing']}")
        
        # Market positioning
        market = impact['market_positioning_impact']
        print(f"\n🎯 Market Positioning Impact:")
        print(f"   • Competitive Advantage: {market['competitive_advantage'].title()}")
        print(f"   • Brand Perception Risk: {market['brand_perception_risk'].title()}")
        print(f"   • Customer Acquisition: {market['customer_acquisition_potential'].title()}")
        print(f"   • Market Share Impact: +{market['market_share_impact']:.1f}%")
        print(f"   • Positioning Strategy: {market['positioning_strategy'].replace('_', ' ').title()}")
        
        # Risk mitigation
        risk_mitigation = impact['risk_mitigation']
        print(f"\n🛡️  Risk Mitigation:")
        print(f"   • Overstock Risk Reduction: -{risk_mitigation['overstock_risk_reduction']:.2f}")
        print(f"   • Demand Stimulation Potential: {risk_mitigation['demand_stimulation_potential']:.2f}")
        
        # Implementation details
        print(f"\n⚙️  Implementation:")
        print(f"   • Complexity: {impact['implementation_complexity'].title()}")
        print(f"   • Time to Impact: {impact['time_to_impact']}")
        print(f"   • Confidence Level: {impact['confidence_level'].title()}")
        
    except Exception as e:
        print(f"❌ Error assessing business impact: {str(e)}")


async def main():
    """Main demo function."""
    print("🚀 MarketPulse AI - Decision Support Engine Demo")
    print("=" * 60)
    
    try:
        # Setup components
        decision_engine, data_processor = await setup_components()
        
        # Create and ingest sample data
        print("\n📊 Creating sample sales data...")
        sample_data = await create_sample_data()
        await data_processor.ingest_sales_data(sample_data)
        print(f"✅ Ingested {len(sample_data)} sales data points")
        
        # Extract patterns
        print("\n🔍 Extracting demand patterns...")
        product_ids = ["ELEC001", "FASH001", "HOME001"]
        patterns = await data_processor.extract_demand_patterns(product_ids)
        print(f"✅ Extracted {len(patterns)} demand patterns")
        
        # Demonstrate recommendation generation
        recommendations = await demonstrate_recommendation_generation(decision_engine, product_ids)
        
        # Demonstrate discount optimization
        await demonstrate_discount_optimization(decision_engine, product_ids)
        
        # Demonstrate business impact assessment
        await demonstrate_business_impact_assessment(decision_engine, recommendations)
        
        print("\n" + "=" * 60)
        print("✅ Decision Support Engine Demo Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("• Comprehensive recommendation generation")
        print("• Discount strategy optimization")
        print("• Business impact assessment")
        print("• Risk assessment integration")
        print("• Compliance validation")
        print("• Recommendation prioritization")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())