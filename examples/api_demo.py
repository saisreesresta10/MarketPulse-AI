"""
MarketPulse AI API Demo

Demonstration script showing how to use the MarketPulse AI REST API.
"""

import asyncio
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_VERSION = "v1"


class MarketPulseAPIClient:
    """Client for interacting with MarketPulse AI API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._make_request("GET", "/health")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return self._make_request("GET", f"/api/{API_VERSION}/system/status")
    
    def ingest_sales_data(self, sales_data: List[Dict[str, Any]], 
                         validate_data: bool = True, 
                         store_patterns: bool = True) -> Dict[str, Any]:
        """Ingest sales data for processing."""
        payload = {
            "data": sales_data,
            "validate_data": validate_data,
            "store_patterns": store_patterns
        }
        return self._make_request("POST", f"/api/{API_VERSION}/data/ingest", json=payload)
    
    def generate_insights(self, product_ids: List[str] = None,
                         include_seasonal: bool = True,
                         confidence_threshold: float = 0.7,
                         max_insights: int = None) -> Dict[str, Any]:
        """Generate insights from demand patterns."""
        payload = {
            "product_ids": product_ids,
            "include_seasonal": include_seasonal,
            "confidence_threshold": confidence_threshold
        }
        if max_insights:
            payload["max_insights"] = max_insights
        
        return self._make_request("POST", f"/api/{API_VERSION}/insights/generate", json=payload)
    
    def get_product_insights(self, product_id: str,
                           include_seasonal: bool = True,
                           confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Get insights for a specific product."""
        params = {
            "include_seasonal": include_seasonal,
            "confidence_threshold": confidence_threshold
        }
        return self._make_request("GET", f"/api/{API_VERSION}/insights/{product_id}", params=params)
    
    def generate_recommendations(self, product_ids: List[str],
                               business_context: Dict[str, Any] = None,
                               priority_filter: str = None,
                               include_compliance_check: bool = True,
                               max_recommendations: int = None) -> Dict[str, Any]:
        """Generate business recommendations."""
        payload = {
            "product_ids": product_ids,
            "business_context": business_context or {},
            "include_compliance_check": include_compliance_check
        }
        if priority_filter:
            payload["priority_filter"] = priority_filter
        if max_recommendations:
            payload["max_recommendations"] = max_recommendations
        
        return self._make_request("POST", f"/api/{API_VERSION}/recommendations/generate", json=payload)
    
    def optimize_discount_strategy(self, product_ids: List[str],
                                 business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize discount strategy for products."""
        params = {"product_ids": product_ids}
        if business_context:
            params["business_context"] = business_context
        
        return self._make_request("POST", f"/api/{API_VERSION}/recommendations/optimize-discount", 
                                params=params)


def generate_sample_sales_data() -> List[Dict[str, Any]]:
    """Generate sample sales data for demonstration."""
    products = ["PROD001", "PROD002", "PROD003", "PROD004", "PROD005"]
    sales_data = []
    
    # Generate data for the last 30 days
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        current_date = base_date + timedelta(days=i)
        
        for product_id in products:
            # Simulate different sales patterns
            base_quantity = 100 + (i * 2)  # Growing trend
            seasonal_factor = 1.2 if current_date.weekday() < 5 else 0.8  # Weekday vs weekend
            
            # Add some randomness
            import random
            random_factor = random.uniform(0.8, 1.2)
            
            quantity = int(base_quantity * seasonal_factor * random_factor)
            price = 100.0 + (hash(product_id) % 50)  # Different prices per product
            revenue = quantity * price
            inventory = max(500 - (i * 10), 100)  # Decreasing inventory
            
            sales_data.append({
                "product_id": product_id,
                "date": current_date.strftime("%Y-%m-%d"),
                "quantity_sold": quantity,
                "revenue": revenue,
                "inventory_level": inventory,
                "price": price
            })
    
    return sales_data


def demo_data_ingestion(client: MarketPulseAPIClient):
    """Demonstrate data ingestion functionality."""
    print("\n" + "="*50)
    print("DEMO: Data Ingestion")
    print("="*50)
    
    # Generate sample data
    sales_data = generate_sample_sales_data()
    print(f"Generated {len(sales_data)} sales data points")
    
    # Ingest data
    try:
        result = client.ingest_sales_data(sales_data, validate_data=True, store_patterns=True)
        print(f"✅ Data ingestion successful!")
        print(f"   Records processed: {result['data'].get('records_processed', 'N/A')}")
        print(f"   Message: {result['message']}")
        return True
    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
        return False


def demo_insight_generation(client: MarketPulseAPIClient):
    """Demonstrate insight generation functionality."""
    print("\n" + "="*50)
    print("DEMO: Insight Generation")
    print("="*50)
    
    try:
        # Generate insights for all products
        result = client.generate_insights(
            product_ids=["PROD001", "PROD002", "PROD003"],
            include_seasonal=True,
            confidence_threshold=0.6,
            max_insights=5
        )
        
        print(f"✅ Insight generation successful!")
        print(f"   Patterns analyzed: {result['data']['total_patterns_analyzed']}")
        print(f"   Insights generated: {result['data']['insights_generated']}")
        
        # Display insights
        insights = result['data']['insights']
        for i, insight in enumerate(insights[:3], 1):  # Show first 3
            print(f"\n   Insight {i}:")
            print(f"     Product: {insight['product_id']}")
            print(f"     Text: {insight['insight_text']}")
            print(f"     Confidence: {insight['confidence']:.2f}")
        
        # Get insights for specific product
        print(f"\n   Getting specific insights for PROD001...")
        product_result = client.get_product_insights("PROD001", include_seasonal=True)
        print(f"   ✅ Retrieved {len(product_result['data']['insights'])} insights for PROD001")
        
        return True
    except Exception as e:
        print(f"❌ Insight generation failed: {e}")
        return False


def demo_recommendation_generation(client: MarketPulseAPIClient):
    """Demonstrate recommendation generation functionality."""
    print("\n" + "="*50)
    print("DEMO: Recommendation Generation")
    print("="*50)
    
    try:
        # Generate recommendations with business context
        business_context = {
            "target_margin": 0.25,
            "inventory_turnover_target": 12,
            "seasonal_events": ["Diwali", "Christmas", "New Year"]
        }
        
        result = client.generate_recommendations(
            product_ids=["PROD001", "PROD002", "PROD003"],
            business_context=business_context,
            priority_filter="high",
            include_compliance_check=True,
            max_recommendations=10
        )
        
        print(f"✅ Recommendation generation successful!")
        print(f"   Products analyzed: {result['data']['total_products']}")
        print(f"   Recommendations generated: {result['data']['recommendations_generated']}")
        
        # Display recommendations
        recommendations = result['data']['recommendations']
        for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
            print(f"\n   Recommendation {i}:")
            print(f"     Product: {rec['product_id']}")
            print(f"     Strategy: {rec['strategy']}")
            print(f"     Priority: {rec.get('priority', 'N/A')}")
            print(f"     Impact Score: {rec.get('impact_score', 'N/A')}")
        
        # Optimize discount strategy
        print(f"\n   Optimizing discount strategy...")
        discount_result = client.optimize_discount_strategy(
            product_ids=["PROD001", "PROD002"],
            business_context=business_context
        )
        print(f"   ✅ Discount optimization completed")
        print(f"   Strategy: {discount_result['data'].get('strategy_type', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Recommendation generation failed: {e}")
        return False


def demo_system_status(client: MarketPulseAPIClient):
    """Demonstrate system status functionality."""
    print("\n" + "="*50)
    print("DEMO: System Status")
    print("="*50)
    
    try:
        # Health check
        health = client.health_check()
        print(f"✅ Health check: {health['data']['status']}")
        
        # System status
        status = client.get_system_status()
        print(f"✅ System status retrieved")
        print(f"   API Version: {status['data']['api_version']}")
        print(f"   Components initialized: {status['data']['components_initialized']}")
        
        # Show available endpoints
        endpoints = status['data']['available_endpoints']
        print(f"\n   Available endpoint categories:")
        for category, endpoint_list in endpoints.items():
            print(f"     {category}: {len(endpoint_list)} endpoints")
        
        return True
    except Exception as e:
        print(f"❌ System status check failed: {e}")
        return False


def main():
    """Run the complete API demonstration."""
    print("MarketPulse AI API Demonstration")
    print("=" * 60)
    
    # Initialize client
    client = MarketPulseAPIClient()
    
    # Check if API is running
    try:
        health = client.health_check()
        print(f"✅ API is running: {health['data']['status']}")
    except Exception as e:
        print(f"❌ API is not accessible: {e}")
        print("Please ensure the API server is running on http://localhost:8000")
        return
    
    # Run demonstrations
    demos = [
        ("System Status", demo_system_status),
        ("Data Ingestion", demo_data_ingestion),
        ("Insight Generation", demo_insight_generation),
        ("Recommendation Generation", demo_recommendation_generation)
    ]
    
    results = {}
    for demo_name, demo_func in demos:
        try:
            results[demo_name] = demo_func(client)
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("DEMONSTRATION SUMMARY")
    print("="*60)
    
    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{demo_name}: {status}")
    
    successful_demos = sum(1 for success in results.values() if success)
    total_demos = len(results)
    print(f"\nOverall: {successful_demos}/{total_demos} demos successful")
    
    if successful_demos == total_demos:
        print("🎉 All API demonstrations completed successfully!")
    else:
        print("⚠️  Some demonstrations failed. Check the logs above for details.")


if __name__ == "__main__":
    main()