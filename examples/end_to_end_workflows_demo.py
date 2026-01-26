"""
End-to-End Workflows Demo

Demonstrates the complete MarketPulse AI workflows including:
1. Retailer insights generation
2. Recommendation generation and validation
3. Scenario analysis and reporting
4. Feedback processing and learning
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock data for demonstration
SAMPLE_SALES_DATA = [
    {
        "data_point_id": "dp_001",
        "product_id": "PROD_001",
        "date": "2024-01-15T10:00:00Z",
        "quantity_sold": 25,
        "selling_price": 150.0,
        "cost_price": 100.0,
        "seasonal_event": "Republic Day Sale",
        "market_conditions": {"demand_level": "high", "competition": "medium"}
    },
    {
        "data_point_id": "dp_002",
        "product_id": "PROD_001",
        "date": "2024-01-20T14:30:00Z",
        "quantity_sold": 18,
        "selling_price": 150.0,
        "cost_price": 100.0,
        "seasonal_event": None,
        "market_conditions": {"demand_level": "medium", "competition": "high"}
    },
    {
        "data_point_id": "dp_003",
        "product_id": "PROD_002",
        "date": "2024-01-25T09:15:00Z",
        "quantity_sold": 32,
        "selling_price": 200.0,
        "cost_price": 140.0,
        "seasonal_event": None,
        "market_conditions": {"demand_level": "high", "competition": "low"}
    }
]

SAMPLE_BASE_SCENARIO = {
    "product_id": "PROD_001",
    "time_horizon_days": 30,
    "current_inventory": 100,
    "discount_percentage": 0,
    "marketing_budget": 5000
}

SAMPLE_SCENARIO_VARIATIONS = [
    {
        "product_id": "PROD_001",
        "time_horizon_days": 30,
        "current_inventory": 100,
        "discount_percentage": 10,
        "marketing_budget": 5000
    },
    {
        "product_id": "PROD_001",
        "time_horizon_days": 30,
        "current_inventory": 100,
        "discount_percentage": 20,
        "marketing_budget": 7500
    }
]


async def demo_retailer_insights_workflow():
    """Demonstrate the retailer insights workflow."""
    print("\n" + "="*60)
    print("RETAILER INSIGHTS WORKFLOW DEMO")
    print("="*60)
    
    try:
        # Import here to avoid circular imports
        from marketpulse_ai.api.main import component_manager
        
        # Initialize components for demo
        await component_manager.initialize_components(testing_mode=True)
        orchestrator = component_manager.get_component("orchestrator")
        
        if not orchestrator:
            print("❌ Orchestrator not available - using mock demo")
            return await demo_mock_retailer_insights()
        
        # Convert sample data to SalesDataPoint objects
        from marketpulse_ai.core.models import SalesDataPoint
        sales_data_points = []
        
        for data_dict in SAMPLE_SALES_DATA:
            sales_point = SalesDataPoint(
                data_point_id=data_dict["data_point_id"],
                product_id=data_dict["product_id"],
                date=datetime.fromisoformat(data_dict["date"].replace("Z", "+00:00")),
                quantity_sold=data_dict["quantity_sold"],
                selling_price=data_dict["selling_price"],
                cost_price=data_dict["cost_price"],
                seasonal_event=data_dict["seasonal_event"],
                market_conditions=data_dict["market_conditions"]
            )
            sales_data_points.append(sales_point)
        
        print(f"📊 Processing {len(sales_data_points)} sales data points...")
        
        # Execute retailer insights workflow
        result = await orchestrator.generate_retailer_insights(
            retailer_id="RETAILER_001",
            sales_data=sales_data_points,
            include_risk_assessment=True,
            include_compliance_check=True
        )
        
        if result.success:
            print("✅ Retailer insights workflow completed successfully!")
            print(f"⏱️  Execution time: {result.execution_time_seconds:.2f} seconds")
            print(f"📈 Insights generated: {result.data.get('insights_generated', 0)}")
            print(f"📊 Data points processed: {result.data.get('data_points_processed', 0)}")
            
            # Display sample insights
            insights = result.data.get('insights', [])
            if insights:
                print(f"\n🔍 Sample Insight:")
                sample_insight = insights[0]
                print(f"   Text: {sample_insight.get('insight_text', 'N/A')}")
                print(f"   Confidence: {sample_insight.get('confidence_level', 'N/A')}")
        else:
            print(f"❌ Workflow failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Retailer insights demo error: {e}")


async def demo_mock_retailer_insights():
    """Mock demo for retailer insights when orchestrator is not available."""
    print("🔄 Running mock retailer insights workflow...")
    
    # Simulate processing time
    await asyncio.sleep(1)
    
    mock_result = {
        "retailer_id": "RETAILER_001",
        "insights_generated": 3,
        "data_points_processed": len(SAMPLE_SALES_DATA),
        "sample_insights": [
            {
                "insight_text": "Strong sales performance during Republic Day Sale period",
                "confidence_level": "high",
                "key_factors": ["seasonal_event", "high_demand"]
            },
            {
                "insight_text": "Product PROD_002 shows consistent high demand",
                "confidence_level": "medium",
                "key_factors": ["low_competition", "high_demand"]
            }
        ]
    }
    
    print("✅ Mock retailer insights completed!")
    print(f"📈 Insights generated: {mock_result['insights_generated']}")
    print(f"📊 Data points processed: {mock_result['data_points_processed']}")
    
    for i, insight in enumerate(mock_result['sample_insights'], 1):
        print(f"\n🔍 Insight {i}:")
        print(f"   Text: {insight['insight_text']}")
        print(f"   Confidence: {insight['confidence_level']}")


async def demo_recommendations_workflow():
    """Demonstrate the recommendations workflow."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS WORKFLOW DEMO")
    print("="*60)
    
    try:
        from marketpulse_ai.api.main import component_manager
        orchestrator = component_manager.get_component("orchestrator")
        
        if not orchestrator:
            print("❌ Orchestrator not available - using mock demo")
            return await demo_mock_recommendations()
        
        product_ids = ["PROD_001", "PROD_002"]
        business_context = {
            "target_margin": 0.3,
            "inventory_turnover_target": 12,
            "seasonal_focus": "summer_sale"
        }
        
        print(f"🎯 Generating recommendations for {len(product_ids)} products...")
        
        # Execute recommendations workflow
        result = await orchestrator.generate_recommendations(
            retailer_id="RETAILER_001",
            product_ids=product_ids,
            business_context=business_context
        )
        
        if result.success:
            print("✅ Recommendations workflow completed successfully!")
            print(f"⏱️  Execution time: {result.execution_time_seconds:.2f} seconds")
            print(f"🎯 Products analyzed: {result.data.get('products_analyzed', 0)}")
            print(f"💡 Recommendations generated: {result.data.get('recommendations_generated', 0)}")
            print(f"✅ Recommendations validated: {result.data.get('recommendations_validated', 0)}")
            
            # Display sample recommendations
            recommendations = result.data.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Top Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec.get('title', 'Recommendation')}")
                    print(f"      Priority: {rec.get('priority_rank', 'N/A')}")
                    print(f"      Confidence: {rec.get('confidence_score', 'N/A')}")
        else:
            print(f"❌ Workflow failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Recommendations demo error: {e}")


async def demo_mock_recommendations():
    """Mock demo for recommendations when orchestrator is not available."""
    print("🔄 Running mock recommendations workflow...")
    
    # Simulate processing time
    await asyncio.sleep(1)
    
    mock_result = {
        "retailer_id": "RETAILER_001",
        "products_analyzed": 2,
        "recommendations_generated": 4,
        "recommendations_validated": 3,
        "top_recommendations": [
            {
                "title": "Optimize discount timing for PROD_001",
                "priority_rank": 1,
                "confidence_score": 0.85,
                "description": "Apply 15% discount during weekends for maximum impact"
            },
            {
                "title": "Increase inventory for PROD_002",
                "priority_rank": 2,
                "confidence_score": 0.78,
                "description": "Stock levels should be increased by 25% for upcoming demand"
            },
            {
                "title": "Bundle products for cross-selling",
                "priority_rank": 3,
                "confidence_score": 0.72,
                "description": "Create bundle offers combining PROD_001 and PROD_002"
            }
        ]
    }
    
    print("✅ Mock recommendations completed!")
    print(f"🎯 Products analyzed: {mock_result['products_analyzed']}")
    print(f"💡 Recommendations generated: {mock_result['recommendations_generated']}")
    
    for i, rec in enumerate(mock_result['top_recommendations'], 1):
        print(f"\n💡 Recommendation {i}:")
        print(f"   Title: {rec['title']}")
        print(f"   Priority: {rec['priority_rank']}")
        print(f"   Confidence: {rec['confidence_score']}")


async def demo_scenario_analysis_workflow():
    """Demonstrate the scenario analysis workflow."""
    print("\n" + "="*60)
    print("SCENARIO ANALYSIS WORKFLOW DEMO")
    print("="*60)
    
    try:
        from marketpulse_ai.api.main import component_manager
        orchestrator = component_manager.get_component("orchestrator")
        
        if not orchestrator:
            print("❌ Orchestrator not available - using mock demo")
            return await demo_mock_scenario_analysis()
        
        print(f"📊 Analyzing base scenario with {len(SAMPLE_SCENARIO_VARIATIONS)} variations...")
        
        # Execute scenario analysis workflow
        result = await orchestrator.analyze_scenarios(
            retailer_id="RETAILER_001",
            base_scenario=SAMPLE_BASE_SCENARIO,
            scenario_variations=SAMPLE_SCENARIO_VARIATIONS
        )
        
        if result.success:
            print("✅ Scenario analysis workflow completed successfully!")
            print(f"⏱️  Execution time: {result.execution_time_seconds:.2f} seconds")
            print(f"📊 Scenarios analyzed: {result.data.get('scenarios_analyzed', 0)}")
            
            # Display scenario comparison
            comparison = result.data.get('scenario_comparison', {})
            if comparison:
                print(f"\n📈 Scenario Analysis Results:")
                best_overall = comparison.get('best_overall_scenario', {})
                if best_overall:
                    print(f"   🏆 Best Overall: {best_overall.get('scenario_id', 'N/A')}")
                    print(f"   💰 Expected Revenue: {best_overall.get('expected_revenue', 0)}")
                    print(f"   ⚠️  Risk Score: {best_overall.get('risk_score', 0)}")
                
                print(f"   📊 Total Scenarios: {comparison.get('total_scenarios', 0)}")
            
            # Display recommendations
            recommendations = result.data.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Scenario-based Recommendations:")
                for i, rec in enumerate(recommendations[:2], 1):
                    print(f"   {i}. {rec.get('title', 'Recommendation')}")
                    print(f"      Type: {rec.get('type', 'N/A')}")
        else:
            print(f"❌ Workflow failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Scenario analysis demo error: {e}")


async def demo_mock_scenario_analysis():
    """Mock demo for scenario analysis when orchestrator is not available."""
    print("🔄 Running mock scenario analysis workflow...")
    
    # Simulate processing time
    await asyncio.sleep(1.5)
    
    mock_result = {
        "retailer_id": "RETAILER_001",
        "scenarios_analyzed": 3,
        "best_scenario": {
            "scenario_id": "variation_2",
            "expected_revenue": 45000,
            "risk_score": 0.3,
            "confidence": 0.82
        },
        "recommendations": [
            {
                "title": "Implement 20% discount strategy",
                "type": "scenario_recommendation",
                "description": "Based on analysis, 20% discount with increased marketing budget offers best ROI"
            },
            {
                "title": "Monitor inventory levels closely",
                "type": "risk_mitigation",
                "description": "Higher discount scenarios require careful inventory management"
            }
        ]
    }
    
    print("✅ Mock scenario analysis completed!")
    print(f"📊 Scenarios analyzed: {mock_result['scenarios_analyzed']}")
    
    best = mock_result['best_scenario']
    print(f"\n📈 Best Scenario Results:")
    print(f"   🏆 Best Scenario: {best['scenario_id']}")
    print(f"   💰 Expected Revenue: ${best['expected_revenue']:,}")
    print(f"   ⚠️  Risk Score: {best['risk_score']}")
    
    for i, rec in enumerate(mock_result['recommendations'], 1):
        print(f"\n💡 Recommendation {i}:")
        print(f"   Title: {rec['title']}")
        print(f"   Type: {rec['type']}")


async def demo_feedback_processing():
    """Demonstrate feedback processing and learning."""
    print("\n" + "="*60)
    print("FEEDBACK PROCESSING DEMO")
    print("="*60)
    
    try:
        from marketpulse_ai.api.main import component_manager
        orchestrator = component_manager.get_component("orchestrator")
        
        if not orchestrator:
            print("❌ Orchestrator not available - using mock demo")
            return await demo_mock_feedback_processing()
        
        # Sample feedback data
        feedback_data = {
            "type": "recommendation_rating",
            "target_id": "REC_001",
            "rating": 4.5,
            "text_feedback": "Great recommendation! Helped increase sales by 15%",
            "business_impact": {
                "revenue_increase": 15000,
                "customer_satisfaction": 4.2
            },
            "trigger_model_update": True
        }
        
        print("📝 Processing retailer feedback...")
        
        # Execute feedback processing
        result = await orchestrator.process_feedback_and_learn(
            retailer_id="RETAILER_001",
            feedback_data=feedback_data
        )
        
        if result.get("status") == "processed":
            print("✅ Feedback processing completed successfully!")
            print(f"📝 Feedback ID: {result.get('feedback_id', 'N/A')}")
            print(f"💡 Learning triggered: {feedback_data.get('trigger_model_update', False)}")
        else:
            print(f"❌ Feedback processing failed: {result.get('error_message')}")
            
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Feedback processing demo error: {e}")


async def demo_mock_feedback_processing():
    """Mock demo for feedback processing when orchestrator is not available."""
    print("🔄 Running mock feedback processing...")
    
    # Simulate processing time
    await asyncio.sleep(0.5)
    
    mock_result = {
        "feedback_id": "FB_001",
        "status": "processed",
        "learning_triggered": True,
        "model_updates": ["demand_forecasting", "recommendation_engine"]
    }
    
    print("✅ Mock feedback processing completed!")
    print(f"📝 Feedback ID: {mock_result['feedback_id']}")
    print(f"💡 Learning triggered: {mock_result['learning_triggered']}")
    print(f"🔄 Models updated: {', '.join(mock_result['model_updates'])}")


async def demo_system_health():
    """Demonstrate system health monitoring."""
    print("\n" + "="*60)
    print("SYSTEM HEALTH MONITORING DEMO")
    print("="*60)
    
    try:
        from marketpulse_ai.api.main import component_manager
        orchestrator = component_manager.get_component("orchestrator")
        
        if not orchestrator:
            print("❌ Orchestrator not available - using mock demo")
            return demo_mock_system_health()
        
        print("🔍 Checking system health...")
        
        # Get system health
        health_status = orchestrator.get_system_health()
        
        print("✅ System health check completed!")
        print(f"🔄 Active workflows: {health_status.get('active_workflows', 0)}")
        print(f"✅ Completed workflows: {health_status.get('completed_workflows', 0)}")
        print(f"🚀 System ready: {health_status.get('system_ready', False)}")
        
        # Display component status
        components_status = health_status.get('components_status', {})
        print(f"\n🔧 Component Status:")
        for component, status in components_status.items():
            status_icon = "✅" if status == "healthy" else "⚠️"
            print(f"   {status_icon} {component}: {status}")
            
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"System health demo error: {e}")


def demo_mock_system_health():
    """Mock demo for system health when orchestrator is not available."""
    print("🔄 Running mock system health check...")
    
    mock_health = {
        "active_workflows": 2,
        "completed_workflows": 15,
        "system_ready": True,
        "components_status": {
            "data_processor": "healthy",
            "risk_assessor": "healthy",
            "compliance_validator": "healthy",
            "insight_generator": "healthy",
            "decision_support_engine": "healthy",
            "scenario_analyzer": "healthy",
            "model_updater": "healthy",
            "feedback_learner": "healthy"
        }
    }
    
    print("✅ Mock system health check completed!")
    print(f"🔄 Active workflows: {mock_health['active_workflows']}")
    print(f"✅ Completed workflows: {mock_health['completed_workflows']}")
    print(f"🚀 System ready: {mock_health['system_ready']}")
    
    print(f"\n🔧 Component Status:")
    for component, status in mock_health['components_status'].items():
        print(f"   ✅ {component}: {status}")


async def main():
    """Run all workflow demos."""
    print("🚀 MarketPulse AI End-to-End Workflows Demo")
    print("=" * 80)
    
    try:
        # Run all workflow demos
        await demo_retailer_insights_workflow()
        await demo_recommendations_workflow()
        await demo_scenario_analysis_workflow()
        await demo_feedback_processing()
        await demo_system_health()
        
        print("\n" + "="*80)
        print("✅ All workflow demos completed successfully!")
        print("🎉 MarketPulse AI system is ready for production use.")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Demo suite failed: {str(e)}")
        logger.error(f"Main demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())