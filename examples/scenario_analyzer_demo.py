"""
Scenario Analyzer Demo Script.

This script demonstrates the capabilities of the Scenario Analyzer
by generating and analyzing various business scenarios for strategic planning.
"""

import asyncio
import json
from datetime import datetime, date, timedelta

from marketpulse_ai.components import ScenarioAnalyzer, DataProcessor, RiskAssessor
from marketpulse_ai.core.models import ConfidenceLevel
from marketpulse_ai.storage.factory import StorageFactory


async def demonstrate_scenario_generation():
    """Demonstrate scenario generation capabilities."""
    print("\n🎯 Scenario Generation Demo")
    print("=" * 50)
    
    # Initialize Scenario Analyzer
    scenario_analyzer = ScenarioAnalyzer()
    
    # Define base parameters for scenario generation
    base_parameters = {
        'product_ids': ['ELEC001', 'FASH001', 'HOME001'],
        'time_horizon': '6_months',
        'scenario_count': 4,
        'analysis_type': 'comprehensive'
    }
    
    print(f"📊 Generating scenarios for products: {base_parameters['product_ids']}")
    print(f"⏰ Time horizon: {base_parameters['time_horizon']}")
    print(f"🔢 Number of scenarios: {base_parameters['scenario_count']}")
    
    try:
        # Generate scenarios
        scenarios = await scenario_analyzer.generate_scenarios(base_parameters)
        
        print(f"\n✅ Successfully generated {len(scenarios)} scenarios:")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario.name}")
            print(f"   Description: {scenario.description}")
            print(f"   Confidence: {scenario.confidence_level.value.title()}")
            print(f"   Market Condition: {scenario.parameters.get('market_condition', 'N/A')}")
            print(f"   Demand Multiplier: {scenario.parameters.get('demand_multiplier', 'N/A')}")
            
            # Show discount strategy if present
            discount_strategy = scenario.parameters.get('discount_strategy', {})
            if discount_strategy:
                print(f"   Discount Strategy:")
                for product_id, discount in discount_strategy.items():
                    print(f"     • {product_id}: {discount}%")
            
            # Show key assumptions
            if scenario.assumptions:
                print(f"   Key Assumptions:")
                for assumption in scenario.assumptions[:2]:  # Show first 2
                    print(f"     • {assumption}")
            
            # Show predicted outcomes
            outcomes = scenario.predicted_outcomes
            if outcomes:
                print(f"   Predicted Outcomes:")
                print(f"     • Revenue Impact: {outcomes.get('revenue_impact', 0):+.1f}%")
                print(f"     • Inventory Turnover: {outcomes.get('inventory_turnover', 1.0):.1f}x")
                print(f"     • Market Share Change: {outcomes.get('market_share_change', 0):+.1f}%")
                print(f"     • Risk Level: {outcomes.get('risk_level', 'medium').title()}")
        
        return scenarios
        
    except Exception as e:
        print(f"❌ Error generating scenarios: {str(e)}")
        return []


async def demonstrate_inventory_prediction(scenarios):
    """Demonstrate inventory outcome prediction."""
    print("\n📦 Inventory Outcome Prediction Demo")
    print("=" * 50)
    
    if not scenarios:
        print("❌ No scenarios available for inventory prediction")
        return
    
    scenario_analyzer = ScenarioAnalyzer()
    
    # Analyze inventory outcomes for each scenario
    for i, scenario in enumerate(scenarios[:3], 1):  # Analyze first 3 scenarios
        print(f"\n{i}. Analyzing inventory outcomes for: {scenario.name}")
        
        try:
            inventory_outcomes = await scenario_analyzer.predict_inventory_outcomes(scenario)
            
            if 'error' in inventory_outcomes:
                print(f"   ❌ Error: {inventory_outcomes['error']}")
                continue
            
            print(f"   ⏰ Time Horizon: {inventory_outcomes['time_horizon']}")
            print(f"   🎯 Confidence: {inventory_outcomes['confidence_level'].title()}")
            
            # Show product-specific predictions
            product_predictions = inventory_outcomes.get('product_predictions', {})
            print(f"\n   📊 Product-Specific Predictions:")
            
            for product_id, prediction in product_predictions.items():
                print(f"\n     {product_id}:")
                print(f"       • Current Inventory: {prediction['current_inventory']} units")
                print(f"       • Predicted Demand: {prediction['predicted_demand']:.1f} units/month")
                print(f"       • Total Period Demand: {prediction['total_demand_period']:.1f} units")
                print(f"       • Coverage: {prediction['inventory_coverage_days']:.1f} days")
                print(f"       • Stockout Risk: {prediction['stockout_risk'].title()}")
                print(f"       • Overstock Risk: {prediction['overstock_risk'].title()}")
                print(f"       • Recommended Reorder: {prediction['reorder_quantity']:.0f} units")
            
            # Show aggregated outcomes
            aggregated = inventory_outcomes.get('aggregated_outcomes', {})
            if aggregated:
                print(f"\n   📈 Aggregated Outcomes:")
                print(f"       • Total Products: {aggregated['total_products']}")
                print(f"       • Total Current Inventory: {aggregated['total_current_inventory']} units")
                print(f"       • Total Predicted Demand: {aggregated['total_predicted_demand']:.1f} units")
                print(f"       • Total Reorder Needed: {aggregated['total_reorder_quantity']:.0f} units")
                print(f"       • High Stockout Risk Products: {aggregated['high_stockout_risk_products']}")
                print(f"       • High Overstock Risk Products: {aggregated['high_overstock_risk_products']}")
                print(f"       • Overall Health: {aggregated['overall_inventory_health'].replace('_', ' ').title()}")
        
        except Exception as e:
            print(f"   ❌ Error predicting inventory outcomes: {str(e)}")


async def demonstrate_discount_impact_analysis(scenarios):
    """Demonstrate discount strategy impact analysis."""
    print("\n💰 Discount Impact Analysis Demo")
    print("=" * 50)
    
    if not scenarios:
        print("❌ No scenarios available for discount analysis")
        return
    
    scenario_analyzer = ScenarioAnalyzer()
    
    # Find scenarios with discount strategies
    discount_scenarios = [s for s in scenarios if s.parameters.get('discount_strategy')]
    
    if not discount_scenarios:
        print("❌ No scenarios with discount strategies found")
        return
    
    for i, scenario in enumerate(discount_scenarios, 1):
        print(f"\n{i}. Analyzing discount impact for: {scenario.name}")
        
        try:
            discount_analysis = await scenario_analyzer.analyze_discount_impact(scenario)
            
            if 'error' in discount_analysis:
                print(f"   ❌ Error: {discount_analysis['error']}")
                continue
            
            discount_strategy = discount_analysis['discount_strategy']
            print(f"   💸 Discount Strategy: {len(discount_strategy)} products discounted")
            
            # Show product-specific impacts
            product_impacts = discount_analysis.get('product_impacts', {})
            print(f"\n   📊 Product-Specific Impacts:")
            
            for product_id, impact in product_impacts.items():
                print(f"\n     {product_id} (Discount: {impact['discount_percentage']}%):")
                print(f"       • Demand Increase: +{impact['demand_increase_percentage']:.1f}%")
                print(f"       • Revenue Impact: {impact['revenue_impact_percentage']:+.1f}%")
                print(f"       • Margin Impact: {impact['margin_impact_percentage']:+.1f}%")
                print(f"       • Market Share Impact: +{impact['market_share_impact_percentage']:.1f}%")
                print(f"       • Price Elasticity: {impact['price_elasticity']:.1f}")
            
            # Show overall impact
            overall_impact = discount_analysis.get('overall_impact', {})
            if overall_impact:
                print(f"\n   🎯 Overall Strategy Impact:")
                print(f"       • Products Discounted: {overall_impact['total_discounted_products']}")
                print(f"       • Average Discount: {overall_impact['average_discount_percentage']:.1f}%")
                print(f"       • Average Revenue Impact: {overall_impact['average_revenue_impact_percentage']:+.1f}%")
                print(f"       • Average Margin Impact: {overall_impact['average_margin_impact_percentage']:+.1f}%")
                print(f"       • Total Market Share Impact: +{overall_impact['total_market_share_impact_percentage']:.1f}%")
                print(f"       • Strategy Effectiveness: {overall_impact['strategy_effectiveness'].title()}")
                print(f"       • Break-even Volume Increase: {overall_impact['break_even_volume_increase']:.1f}%")
        
        except Exception as e:
            print(f"   ❌ Error analyzing discount impact: {str(e)}")


async def demonstrate_seasonal_modeling(scenarios):
    """Demonstrate seasonal effects modeling."""
    print("\n🌟 Seasonal Effects Modeling Demo")
    print("=" * 50)
    
    if not scenarios:
        print("❌ No scenarios available for seasonal modeling")
        return
    
    scenario_analyzer = ScenarioAnalyzer()
    
    # Take the first scenario and enhance it with seasonal effects
    base_scenario = scenarios[0]
    seasonal_events = ['diwali', 'christmas', 'holi', 'new_year']
    
    print(f"🎯 Enhancing scenario: {base_scenario.name}")
    print(f"🌟 Seasonal events to model: {', '.join(seasonal_events)}")
    
    try:
        enhanced_scenario = await scenario_analyzer.model_seasonal_effects(base_scenario, seasonal_events)
        
        print(f"\n✅ Enhanced scenario created: {enhanced_scenario.name}")
        print(f"📊 Original confidence: {base_scenario.confidence_level.value.title()}")
        print(f"📊 Enhanced confidence: {enhanced_scenario.confidence_level.value.title()}")
        
        # Show seasonal factors
        seasonal_factors = enhanced_scenario.parameters.get('seasonal_factors', {})
        if seasonal_factors:
            print(f"\n🌟 Seasonal Adjustment Factors:")
            for event, factor in seasonal_factors.items():
                impact_desc = "Positive" if factor > 1.0 else "Negative" if factor < 1.0 else "Neutral"
                print(f"   • {event.title()}: {factor:.2f}x ({impact_desc} impact)")
        
        # Show seasonal impact summary
        seasonal_summary = enhanced_scenario.predicted_outcomes.get('seasonal_impact_summary', {})
        if seasonal_summary:
            print(f"\n📈 Seasonal Impact Summary:")
            print(f"   • Impact Level: {seasonal_summary['impact'].title()}")
            print(f"   • Factor Range: {seasonal_summary['min_factor']:.1f}x to {seasonal_summary['max_factor']:.1f}x")
            print(f"   • Average Factor: {seasonal_summary['average_factor']:.2f}x")
            print(f"   • Most Impactful Event: {seasonal_summary['most_impactful_event'].title()} ({seasonal_summary['most_impactful_factor']:.1f}x)")
            print(f"   • Summary: {seasonal_summary['summary']}")
        
        # Show additional limitations
        original_limitations = len(base_scenario.limitations)
        enhanced_limitations = len(enhanced_scenario.limitations)
        additional_limitations = enhanced_limitations - original_limitations
        
        if additional_limitations > 0:
            print(f"\n⚠️  Additional Limitations Added: {additional_limitations}")
            new_limitations = enhanced_scenario.limitations[original_limitations:]
            for limitation in new_limitations[:2]:  # Show first 2 new limitations
                print(f"   • {limitation}")
        
        return enhanced_scenario
        
    except Exception as e:
        print(f"❌ Error modeling seasonal effects: {str(e)}")
        return None


async def demonstrate_scenario_validation(scenarios):
    """Demonstrate scenario assumption validation."""
    print("\n✅ Scenario Validation Demo")
    print("=" * 50)
    
    if not scenarios:
        print("❌ No scenarios available for validation")
        return
    
    scenario_analyzer = ScenarioAnalyzer()
    
    # Validate different types of scenarios
    for i, scenario in enumerate(scenarios[:2], 1):  # Validate first 2 scenarios
        print(f"\n{i}. Validating scenario: {scenario.name}")
        
        try:
            limitations = await scenario_analyzer.validate_scenario_assumptions(scenario)
            
            print(f"   📊 Total limitations identified: {len(limitations)}")
            
            if limitations:
                print(f"\n   ⚠️  Key Limitations:")
                for j, limitation in enumerate(limitations[:5], 1):  # Show first 5
                    print(f"     {j}. {limitation}")
                
                if len(limitations) > 5:
                    print(f"     ... and {len(limitations) - 5} more limitations")
            else:
                print(f"   ✅ No significant limitations identified")
            
            # Categorize limitations
            limitation_text = ' '.join(limitations).lower()
            categories = []
            
            if 'data' in limitation_text or 'historical' in limitation_text:
                categories.append("Data Quality")
            if 'market' in limitation_text or 'condition' in limitation_text:
                categories.append("Market Assumptions")
            if 'discount' in limitation_text or 'pricing' in limitation_text:
                categories.append("Pricing Strategy")
            if 'seasonal' in limitation_text:
                categories.append("Seasonal Modeling")
            if 'confidence' in limitation_text or 'assumption' in limitation_text:
                categories.append("Confidence Level")
            
            if categories:
                print(f"   🏷️  Limitation Categories: {', '.join(categories)}")
        
        except Exception as e:
            print(f"   ❌ Error validating scenario: {str(e)}")


async def demonstrate_scenario_comparison():
    """Demonstrate scenario comparison for decision making."""
    print("\n🔍 Scenario Comparison Demo")
    print("=" * 50)
    
    scenario_analyzer = ScenarioAnalyzer()
    
    # Generate scenarios for comparison
    base_parameters = {
        'product_ids': ['PROD001', 'PROD002'],
        'time_horizon': '3_months',
        'scenario_count': 3,
        'analysis_type': 'comparison'
    }
    
    try:
        scenarios = await scenario_analyzer.generate_scenarios(base_parameters)
        
        print(f"📊 Comparing {len(scenarios)} scenarios for decision making:")
        
        # Create comparison table
        print(f"\n{'Scenario':<20} {'Revenue Impact':<15} {'Risk Level':<12} {'Confidence':<12}")
        print("-" * 60)
        
        for scenario in scenarios:
            outcomes = scenario.predicted_outcomes
            revenue_impact = outcomes.get('revenue_impact', 0)
            risk_level = outcomes.get('risk_level', 'medium')
            confidence = scenario.confidence_level.value
            
            print(f"{scenario.name:<20} {revenue_impact:+8.1f}%      {risk_level:<12} {confidence:<12}")
        
        # Recommend best scenario
        best_scenario = max(scenarios, key=lambda s: (
            s.predicted_outcomes.get('revenue_impact', 0) * 
            (2 if s.confidence_level == ConfidenceLevel.HIGH else 1.5 if s.confidence_level == ConfidenceLevel.MEDIUM else 1)
        ))
        
        print(f"\n🏆 Recommended Scenario: {best_scenario.name}")
        print(f"   Reason: Best combination of revenue impact and confidence level")
        
    except Exception as e:
        print(f"❌ Error in scenario comparison: {str(e)}")


async def main():
    """Main demo function."""
    print("🚀 MarketPulse AI - Scenario Analyzer Demo")
    print("=" * 60)
    
    try:
        # Demonstrate scenario generation
        scenarios = await demonstrate_scenario_generation()
        
        if scenarios:
            # Demonstrate inventory prediction
            await demonstrate_inventory_prediction(scenarios)
            
            # Demonstrate discount impact analysis
            await demonstrate_discount_impact_analysis(scenarios)
            
            # Demonstrate seasonal modeling
            enhanced_scenario = await demonstrate_seasonal_modeling(scenarios)
            
            # Demonstrate scenario validation
            validation_scenarios = scenarios + ([enhanced_scenario] if enhanced_scenario else [])
            await demonstrate_scenario_validation(validation_scenarios)
        
        # Demonstrate scenario comparison
        await demonstrate_scenario_comparison()
        
        print("\n" + "=" * 60)
        print("✅ Scenario Analyzer Demo Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("• Multi-scenario generation with variations")
        print("• Inventory outcome prediction")
        print("• Discount strategy impact analysis")
        print("• Seasonal effects modeling")
        print("• Scenario assumption validation")
        print("• Scenario comparison for decision making")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())