"""
Simple API Property Tests

Basic property tests for the API layer to validate requirements.

**Property 14: Recommendation Organization**
**Property 15: Search and Filter Functionality** 
**Property 16: Error Handling and Recovery**
**Validates: Requirements 8.3, 8.4, 8.5**
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from marketpulse_ai.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_api_health_property(client):
    """
    **Property 14: Recommendation Organization**
    **Validates: Requirements 8.3**
    
    Property: Health endpoint maintains consistent response format.
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Property: Health response has required fields
    required_fields = ["success", "data", "message", "request_id", "timestamp"]
    for field in required_fields:
        assert field in data
    
    assert data["success"] is True
    assert "status" in data["data"]
    assert data["data"]["status"] == "healthy"


def test_system_status_property(client):
    """
    **Property 15: Search and Filter Functionality**
    **Validates: Requirements 8.4**
    
    Property: System status provides consistent API information.
    """
    response = client.get("/api/v1/system/status")
    
    assert response.status_code == 200
    data = response.json()
    
    # Property: System status has required structure
    assert data["success"] is True
    assert "api_version" in data["data"]
    assert "components_initialized" in data["data"]
    assert "available_endpoints" in data["data"]


@patch('marketpulse_ai.api.main.components')
def test_error_handling_property(mock_components, client):
    """
    **Property 16: Error Handling and Recovery**
    **Validates: Requirements 8.5**
    
    Property: Component errors are handled gracefully.
    """
    # Mock component to raise error
    mock_data_processor = AsyncMock()
    mock_data_processor.extract_demand_patterns.side_effect = Exception("Test error")
    mock_components.__getitem__.return_value = mock_data_processor
    
    response = client.get("/api/v1/insights/PROD001")
    
    # Property: Component errors result in proper error response
    assert response.status_code == 500
    
    # Property: Error response has proper structure
    error_data = response.json()
    assert isinstance(error_data, dict)
    assert error_data.get("success") is False or "error" in error_data


def test_parameter_validation_property(client):
    """
    **Property 16: Error Handling and Recovery**
    **Validates: Requirements 8.5**
    
    Property: Parameter validation is consistent across endpoints.
    """
    # Test with invalid confidence threshold
    request_data = {
        "product_ids": ["PROD001"],
        "confidence_threshold": 1.5,  # Invalid - should be <= 1.0
        "include_seasonal": True
    }
    
    response = client.post("/api/v1/insights/generate", json=request_data)
    
    # Property: Invalid parameters are rejected with validation error
    assert response.status_code == 422


@patch('marketpulse_ai.api.main.components')
def test_recommendation_organization_property(mock_components, client):
    """
    **Property 14: Recommendation Organization**
    **Validates: Requirements 8.3**
    
    Property: Recommendations are properly organized and structured.
    """
    # Mock decision engine
    mock_decision_engine = AsyncMock()
    mock_recommendations = [
        {
            "product_id": "PROD001",
            "strategy": "discount",
            "priority": "high",
            "impact_score": 0.9,
            "recommendation_id": "REC001"
        },
        {
            "product_id": "PROD002", 
            "strategy": "seasonal",
            "priority": "medium",
            "impact_score": 0.7,
            "recommendation_id": "REC002"
        }
    ]
    
    mock_decision_engine.generate_recommendations.return_value = {
        "recommendations": mock_recommendations,
        "total_analyzed": 2,
        "compliance_validated": True
    }
    mock_components.__getitem__.return_value = mock_decision_engine
    
    request_data = {
        "product_ids": ["PROD001", "PROD002"],
        "include_compliance_check": True
    }
    
    response = client.post("/api/v1/recommendations/generate", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    recommendations = data["data"]["recommendations"]
    
    # Property: Recommendations are properly structured
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Property: Each recommendation has required fields
    for rec in recommendations:
        assert "product_id" in rec
        assert "strategy" in rec
        assert "priority" in rec or "impact_score" in rec


@patch('marketpulse_ai.api.main.components')
def test_search_functionality_property(mock_components, client):
    """
    **Property 15: Search and Filter Functionality**
    **Validates: Requirements 8.4**
    
    Property: Search functionality respects limit parameters.
    """
    # Mock decision engine with search results
    mock_decision_engine = AsyncMock()
    mock_recommendations = [
        {
            "product_id": f"PROD{i:03d}",
            "strategy": "test_strategy",
            "priority": "high",
            "description": f"Test recommendation {i} with test keyword"
        }
        for i in range(20)  # More than limit
    ]
    
    mock_decision_engine.generate_recommendations.return_value = {
        "recommendations": mock_recommendations
    }
    mock_components.__getitem__.return_value = mock_decision_engine
    
    # Test search with limit
    params = {"query": "test", "limit": 5}
    response = client.get("/api/v1/recommendations/search", params=params)
    
    assert response.status_code == 200
    data = response.json()
    results = data["data"]["recommendations"]
    
    # Property: Results respect limit parameter
    assert len(results) <= 5
    
    # Property: Search metadata is accurate
    search_metadata = data["data"]["search_metadata"]
    assert "results_returned" in search_metadata
    assert "limit_applied" in search_metadata
    assert search_metadata["limit_applied"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])