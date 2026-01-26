"""
Property-Based Tests for MarketPulse AI API

Property tests validating universal correctness properties for the API layer.

**Validates: Requirements 8.3, 8.4, 8.5**
"""

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st, settings
from unittest.mock import AsyncMock, patch
import json

from marketpulse_ai.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIProperties:
    """
    Property-based tests for API functionality.
    
    **Property 14: Recommendation Organization**
    **Property 15: Search and Filter Functionality** 
    **Property 16: Error Handling and Recovery**
    **Validates: Requirements 8.3, 8.4, 8.5**
    """
    
    def test_health_endpoint_consistency(self, client):
        """Property: Health endpoint always returns consistent format."""
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
    
    def test_system_status_consistency(self, client):
        """Property: System status endpoint returns consistent structure."""
        response = client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Property: System status has required structure
        assert data["success"] is True
        assert "api_version" in data["data"]
        assert "components_initialized" in data["data"]
        assert "available_endpoints" in data["data"]
    
    @patch('marketpulse_ai.api.main.components')
    def test_recommendation_generation_properties(self, mock_components, client):
        """Property: Recommendation generation maintains consistency."""
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
        
        if response.status_code == 200:
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
    def test_search_functionality_properties(self, mock_components, client):
        """Property: Search functionality respects parameters."""
        # Mock decision engine with search results
        mock_decision_engine = AsyncMock()
        mock_recommendations = [
            {
                "product_id": "PROD001",
                "strategy": "test_strategy",
                "priority": "high",
                "description": "Test recommendation with test keyword"
            }
        ]
        
        mock_decision_engine.generate_recommendations.return_value = {
            "recommendations": mock_recommendations
        }
        mock_components.__getitem__.return_value = mock_decision_engine
        
        # Test search with limit
        params = {"query": "test", "limit": 10}
        response = client.get("/api/v1/recommendations/search", params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data["data"]["recommendations"]
            
            # Property: Results respect limit parameter
            assert len(results) <= 10
            
            # Property: Search metadata is accurate
            search_metadata = data["data"]["search_metadata"]
            assert "results_returned" in search_metadata
            assert "limit_applied" in search_metadata
            assert search_metadata["limit_applied"] == 10
    
    @patch('marketpulse_ai.api.main.components')
    def test_error_handling_properties(self, mock_components, client):
        """Property: Errors are handled gracefully."""
        # Mock component to raise error
        mock_data_processor = AsyncMock()
        mock_data_processor.extract_demand_patterns.side_effect = Exception("Test error")
        mock_components.__getitem__.return_value = mock_data_processor
        
        response = client.get("/api/v1/insights/PROD001")
        
        # Property: Component errors result in proper error response
        assert response.status_code == 500
        
        try:
            error_data = response.json()
            # Property: Error response has proper structure
            assert isinstance(error_data, dict)
            assert error_data.get("success") is False or "error" in error_data
        except json.JSONDecodeError:
            # Property: Non-JSON errors still provide response
            assert len(response.text) > 0
    
    def test_parameter_validation_properties(self, client):
        """Property: Parameter validation is consistent."""
        # Test with invalid confidence threshold
        request_data = {
            "product_ids": ["PROD001"],
            "confidence_threshold": 1.5,  # Invalid - should be <= 1.0
            "include_seasonal": True
        }
        
        response = client.post("/api/v1/insights/generate", json=request_data)
        
        # Property: Invalid parameters are rejected with validation error
        assert response.status_code == 422
        
        # Test with valid confidence threshold
        request_data["confidence_threshold"] = 0.8
        response = client.post("/api/v1/insights/generate", json=request_data)
        
        # Property: Valid parameters don't fail validation
        assert response.status_code != 422
    
    @patch('marketpulse_ai.api.main.components')
    def test_data_ingestion_properties(self, mock_components, client):
        """Property: Data ingestion handles various input sizes consistently."""
        mock_data_processor = AsyncMock()
        
        # Test with small dataset
        small_data = [
            {
                "product_id": "PROD001",
                "date": "2024-01-15",
                "quantity_sold": 100,
                "revenue": 1000.0,
                "inventory_level": 500,
                "price": 10.0
            }
        ]
        
        mock_data_processor.ingest_sales_data.return_value = {
            "records_processed": len(small_data),
            "validation_passed": True
        }
        mock_components.__getitem__.return_value = mock_data_processor
        
        request_data = {"data": small_data, "validate_data": True, "store_patterns": True}
        response = client.post("/api/v1/data/ingest", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            # Property: Response reflects input data size
            assert data["success"] is True
            assert "ingestion_summary" in data["data"]
            assert data["data"]["ingestion_summary"]["total_records"] == len(small_data)
    
    def test_response_format_consistency_property(self, client):
        """Property: All successful responses follow consistent format."""
        endpoints = ["/health", "/api/v1/system/status"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            
            if response.status_code == 200:
                data = response.json()
                
                # Property: Success responses have consistent structure
                required_fields = ["success", "data", "message", "request_id", "timestamp"]
                for field in required_fields:
                    assert field in data, f"Missing field {field} in {endpoint}"
                
                assert data["success"] is True
                assert isinstance(data["request_id"], str)
                assert len(data["request_id"]) > 0


# Additional Hypothesis-based property tests
class TestHypothesisProperties:
    """Hypothesis-based property tests for more comprehensive coverage."""
    
    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    @settings(max_examples=10, deadline=5000)
    def test_search_query_safety(self, query, client):
        """Property: Search queries are handled safely regardless of content."""
        with patch('marketpulse_ai.api.main.components') as mock_components:
            mock_decision_engine = AsyncMock()
            mock_decision_engine.generate_recommendations.return_value = {"recommendations": []}
            mock_components.__getitem__.return_value = mock_decision_engine
            
            response = client.get("/api/v1/recommendations/search", params={"query": query})
            
            # Property: Search never crashes regardless of query content
            assert response.status_code in [200, 400, 422]
    
    @given(st.floats(min_value=-10.0, max_value=10.0))
    @settings(max_examples=10, deadline=5000)
    def test_confidence_threshold_validation(self, confidence, client):
        """Property: Confidence threshold validation is consistent."""
        request_data = {
            "product_ids": ["PROD001"],
            "confidence_threshold": confidence,
            "include_seasonal": True
        }
        
        response = client.post("/api/v1/insights/generate", json=request_data)
        
        # Property: Only valid confidence values (0.0-1.0) are accepted
        if 0.0 <= confidence <= 1.0:
            assert response.status_code != 422
        else:
            assert response.status_code == 422
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=5000)
    def test_limit_parameter_consistency(self, limit, client):
        """Property: Limit parameters are consistently respected."""
        with patch('marketpulse_ai.api.main.components') as mock_components:
            mock_decision_engine = AsyncMock()
            # Create more recommendations than the limit
            mock_recommendations = [
                {"product_id": f"PROD{i:03d}", "strategy": "test", "priority": "high"}
                for i in range(limit + 10)
            ]
            mock_decision_engine.generate_recommendations.return_value = {
                "recommendations": mock_recommendations
            }
            mock_components.__getitem__.return_value = mock_decision_engine
            
            response = client.get("/api/v1/recommendations/search", 
                                params={"query": "test", "limit": limit})
            
            if response.status_code == 200:
                data = response.json()
                results = data["data"]["recommendations"]
                
                # Property: Results never exceed specified limit
                assert len(results) <= limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])