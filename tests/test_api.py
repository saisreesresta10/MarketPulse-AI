"""
Tests for MarketPulse AI REST API

Test suite for API endpoints and functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json

from marketpulse_ai.api.main import app
from marketpulse_ai.core.models import SalesDataPoint, DemandPattern, ExplainableInsight


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_sales_data():
    """Sample sales data for testing."""
    return [
        {
            "product_id": "PROD001",
            "product_name": "Premium Tea",
            "category": "beverages",
            "mrp": 100.0,
            "selling_price": 95.0,
            "quantity_sold": 150,
            "sale_date": "2024-01-15",
            "store_location": "Mumbai_Central"
        },
        {
            "product_id": "PROD002",
            "product_name": "Organic Rice",
            "category": "groceries", 
            "mrp": 250.0,
            "selling_price": 240.0,
            "quantity_sold": 75,
            "sale_date": "2024-01-15",
            "store_location": "Delhi_CP"
        }
    ]


@pytest.fixture
def sample_insight_request():
    """Sample insight generation request."""
    return {
        "product_ids": ["PROD001", "PROD002"],
        "include_seasonal": True,
        "confidence_threshold": 0.8,
        "max_insights": 10
    }


@pytest.fixture
def sample_recommendation_request():
    """Sample recommendation request."""
    return {
        "product_ids": ["PROD001", "PROD002"],
        "business_context": {
            "target_margin": 0.25,
            "inventory_turnover_target": 12,
            "seasonal_events": ["Diwali", "Christmas"]
        },
        "priority_filter": "high",
        "include_compliance_check": True,
        "max_recommendations": 20
    }


class TestHealthEndpoints:
    """Test health and system status endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]
        assert data["data"]["status"] == "healthy"
        assert "request_id" in data
    
    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "api_version" in data["data"]
        assert "components_initialized" in data["data"]
        assert "available_endpoints" in data["data"]


class TestDataEndpoints:
    """Test data ingestion endpoints."""
    
    @patch('marketpulse_ai.api.main.components')
    def test_data_ingestion_success(self, mock_components, client, sample_sales_data):
        """Test successful data ingestion."""
        # Mock the data processor
        mock_data_processor = AsyncMock()
        mock_data_processor.ingest_sales_data.return_value = {
            "records_processed": 2,
            "validation_passed": True,
            "patterns_extracted": 1
        }
        mock_components.__getitem__.return_value = mock_data_processor
        
        response = client.post(
            "/api/v1/data/ingest",
            json={"data": sample_sales_data, "validate_data": True, "store_patterns": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Successfully ingested" in data["message"]
    
    def test_data_ingestion_empty_data(self, client):
        """Test data ingestion with empty data."""
        response = client.post(
            "/api/v1/data/ingest",
            json={"data": [], "validate_data": True}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_data_ingestion_invalid_data(self, client):
        """Test data ingestion with invalid data format."""
        invalid_data = [{"invalid": "data"}]
        
        response = client.post(
            "/api/v1/data/ingest",
            json={"data": invalid_data, "validate_data": True}
        )
        
        assert response.status_code == 400


class TestInsightEndpoints:
    """Test insight generation endpoints."""
    
    @patch('marketpulse_ai.api.routers.insights.get_insight_generator')
    @patch('marketpulse_ai.api.routers.insights.get_data_processor')
    def test_generate_insights_success(self, mock_get_data_processor, mock_get_insight_generator, client, sample_insight_request):
        """Test successful insight generation."""
        # Mock components
        mock_data_processor = AsyncMock()
        mock_insight_generator = AsyncMock()
        
        mock_get_data_processor.return_value = mock_data_processor
        mock_get_insight_generator.return_value = mock_insight_generator
        
        # Mock demand patterns
        from datetime import date
        from marketpulse_ai.core.models import ConfidenceLevel
        
        mock_pattern = DemandPattern(
            product_id="PROD001",
            pattern_type="seasonal",
            description="Strong seasonal demand pattern",
            confidence_level=ConfidenceLevel.HIGH,
            trend_direction="increasing",
            volatility_score=0.3,
            supporting_data_points=100,
            date_range_start=date(2024, 1, 1),
            date_range_end=date(2024, 12, 31),
            seasonal_factors={"festival_season": 1.5, "price_sensitivity": 0.8}
        )
        mock_data_processor.extract_demand_patterns.return_value = [mock_pattern]
        mock_data_processor.correlate_seasonal_events.return_value = [mock_pattern]
        
        # Mock insights
        mock_insight = ExplainableInsight(
            title="Seasonal Demand Pattern",
            description="Strong seasonal demand pattern detected",
            confidence_level=ConfidenceLevel.HIGH,
            supporting_evidence=["Historical sales data", "Seasonal correlation"],
            key_factors=["festival_season"],
            business_impact="High revenue potential during festival season",
            data_sources=["sales_data", "seasonal_events"]
        )
        mock_insight_generator.generate_insights.return_value = [mock_insight]
        
        response = client.post("/api/v1/insights/generate", json=sample_insight_request)
        
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")
        
        print(f"Mock data processor called: {mock_data_processor.extract_demand_patterns.called}")
        print(f"Mock insight generator called: {mock_insight_generator.generate_insights.called}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["insights"]) == 1
        assert data["data"]["total_patterns_analyzed"] == 1
    
    @patch('marketpulse_ai.api.routers.insights.get_insight_generator')
    @patch('marketpulse_ai.api.routers.insights.get_data_processor')
    def test_get_product_insights(self, mock_get_data_processor, mock_get_insight_generator, client):
        """Test getting insights for specific product."""
        # Mock components
        mock_data_processor = AsyncMock()
        mock_insight_generator = AsyncMock()
        
        mock_get_data_processor.return_value = mock_data_processor
        mock_get_insight_generator.return_value = mock_insight_generator
        
        from datetime import date
        from marketpulse_ai.core.models import ConfidenceLevel
        
        mock_pattern = DemandPattern(
            product_id="PROD001",
            pattern_type="seasonal",
            description="Strong seasonal demand pattern",
            confidence_level=ConfidenceLevel.HIGH,
            trend_direction="increasing",
            volatility_score=0.3,
            supporting_data_points=100,
            date_range_start=date(2024, 1, 1),
            date_range_end=date(2024, 12, 31),
            seasonal_factors={"festival_season": 1.5}
        )
        mock_data_processor.extract_demand_patterns.return_value = [mock_pattern]
        
        mock_insight = ExplainableInsight(
            title="Seasonal Demand Pattern",
            description="Strong seasonal demand pattern detected",
            confidence_level=ConfidenceLevel.HIGH,
            supporting_evidence=["Historical sales data"],
            key_factors=["festival_season"],
            business_impact="High revenue potential",
            data_sources=["sales_data"]
        )
        mock_insight_generator.explain_pattern.return_value = mock_insight
        
        response = client.get("/api/v1/insights/PROD001?include_seasonal=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["product_id"] == "PROD001"
        assert len(data["data"]["insights"]) == 1
    
    @patch('marketpulse_ai.api.main.components')
    def test_get_product_insights_not_found(self, mock_components, client):
        """Test getting insights for non-existent product."""
        mock_data_processor = AsyncMock()
        mock_data_processor.extract_demand_patterns.return_value = []
        mock_components.__getitem__.return_value = mock_data_processor
        
        response = client.get("/api/v1/insights/NONEXISTENT")
        
        assert response.status_code == 404


class TestRecommendationEndpoints:
    """Test recommendation generation endpoints."""
    
    @patch('marketpulse_ai.api.main.components')
    def test_generate_recommendations_success(self, mock_components, client, sample_recommendation_request):
        """Test successful recommendation generation."""
        mock_decision_engine = AsyncMock()
        mock_decision_engine.generate_recommendations.return_value = {
            "recommendations": [
                {
                    "product_id": "PROD001",
                    "strategy": "seasonal_discount",
                    "discount_percentage": 0.15,
                    "priority": "high",
                    "impact_score": 0.85
                }
            ],
            "total_analyzed": 2,
            "compliance_validated": True
        }
        mock_components.__getitem__.return_value = mock_decision_engine
        
        response = client.post("/api/v1/recommendations/generate", json=sample_recommendation_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["recommendations"]) == 1
        assert data["data"]["total_products"] == 2
    
    def test_generate_recommendations_empty_products(self, client):
        """Test recommendation generation with empty product list."""
        request_data = {
            "product_ids": [],
            "business_context": {},
            "include_compliance_check": True
        }
        
        response = client.post("/api/v1/recommendations/generate", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestValidationAndErrorHandling:
    """Test input validation and error handling."""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/insights/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/api/v1/recommendations/generate",
            json={"business_context": {}}  # Missing required product_ids
        )
        
        assert response.status_code == 422
    
    def test_invalid_confidence_threshold(self, client):
        """Test handling of invalid confidence threshold."""
        request_data = {
            "product_ids": ["PROD001"],
            "confidence_threshold": 1.5  # Invalid - should be <= 1.0
        }
        
        response = client.post("/api/v1/insights/generate", json=request_data)
        
        assert response.status_code == 422


class TestResponseFormat:
    """Test API response format consistency."""
    
    def test_success_response_format(self, client):
        """Test that success responses follow standard format."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["success", "data", "message", "request_id", "timestamp"]
        for field in required_fields:
            assert field in data
        
        assert data["success"] is True
        assert isinstance(data["request_id"], str)
        assert isinstance(data["timestamp"], str)
    
    @patch('marketpulse_ai.api.main.components')
    def test_error_response_format(self, mock_components, client):
        """Test that error responses follow standard format."""
        # Mock component to raise an exception
        mock_data_processor = AsyncMock()
        mock_data_processor.ingest_sales_data.side_effect = Exception("Test error")
        mock_components.__getitem__.return_value = mock_data_processor
        
        response = client.post(
            "/api/v1/data/ingest",
            json={"data": [{"product_id": "TEST"}], "validate_data": True}
        )
        
        assert response.status_code == 500
        data = response.json()
        
        # Check error response format
        required_fields = ["success", "error", "message", "request_id", "timestamp"]
        for field in required_fields:
            assert field in data
        
        assert data["success"] is False
        assert isinstance(data["error"], str)
        assert isinstance(data["message"], str)


if __name__ == "__main__":
    pytest.main([__file__])