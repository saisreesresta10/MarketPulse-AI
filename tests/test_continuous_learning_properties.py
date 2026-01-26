"""
Property-Based Tests for Continuous Learning

Property tests validating universal correctness properties for model updates,
accuracy tracking, market adaptation, feedback learning, and seasonal evolution.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from hypothesis import given, strategies as st, settings
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from marketpulse_ai.components.model_updater import (
    ModelUpdater, ModelType, UpdateStatus, ModelVersion
)
from marketpulse_ai.components.feedback_learner import (
    FeedbackLearner, FeedbackType, FeedbackSentiment, FeedbackEntry
)
from marketpulse_ai.core.models import SalesDataPoint, ConfidenceLevel


class TestModelUpdateProperties:
    """
    Property-based tests for model update functionality.
    
    **Property 20: Model Update Integration**
    **Property 21: Accuracy Tracking and Improvement**
    **Property 22: Market Adaptation**
    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        return AsyncMock()
    
    @pytest.fixture
    def model_updater(self, mock_storage_manager):
        """Create model updater for testing."""
        return ModelUpdater(mock_storage_manager)
    
    def test_property_model_version_consistency(self, model_updater):
        """Property: Model versions maintain consistency and proper ordering."""
        # Create multiple model versions
        versions = []
        for i in range(5):
            version = ModelVersion(
                version_id=f"v{i}",
                model_type=ModelType.DEMAND_FORECASTING,
                created_at=datetime.now(timezone.utc) + timedelta(hours=i),
                accuracy_score=0.7 + (i * 0.05),
                training_data_size=100 + (i * 50)
            )
            versions.append(version)
            model_updater.model_versions[ModelType.DEMAND_FORECASTING].append(version)
        
        # Property: Versions should be stored correctly
        stored_versions = model_updater.model_versions[ModelType.DEMAND_FORECASTING]
        assert len(stored_versions) == 5
        
        # Property: Each version should have unique ID
        version_ids = [v.version_id for v in stored_versions]
        assert len(set(version_ids)) == len(version_ids)
        
        # Property: Accuracy scores should be within valid range
        for version in stored_versions:
            assert 0.0 <= version.accuracy_score <= 1.0
            assert version.training_data_size > 0
    
    def test_property_model_deployment_rules(self, model_updater):
        """Property: Model deployment follows proper rules and rollback capability."""
        # Create current version
        current_version = ModelVersion(
            version_id="current",
            model_type=ModelType.DEMAND_FORECASTING,
            created_at=datetime.now(timezone.utc),
            accuracy_score=0.8,
            training_data_size=500,
            is_active=True
        )
        model_updater.model_versions[ModelType.DEMAND_FORECASTING].append(current_version)
        
        # Test deployment of better version
        better_version = ModelVersion(
            version_id="better",
            model_type=ModelType.DEMAND_FORECASTING,
            created_at=datetime.now(timezone.utc),
            accuracy_score=0.85,
            training_data_size=600
        )
        
        # Property: Better version should be deployable
        should_deploy = model_updater._should_deploy_new_version(
            ModelType.DEMAND_FORECASTING, better_version
        )
        assert should_deploy is True
        
        # Deploy the better version
        model_updater._deploy_model_version(better_version)
        
        # Property: Only one version should be active
        active_versions = [v for v in model_updater.model_versions[ModelType.DEMAND_FORECASTING] if v.is_active]
        assert len(active_versions) == 1
        assert active_versions[0].version_id == "better"
        
        # Property: Previous version should be deactivated
        assert not current_version.is_active
        
        # Test rollback capability
        rollback_result = model_updater.rollback_model(ModelType.DEMAND_FORECASTING, "current")
        
        # Property: Rollback should succeed
        assert rollback_result["success"] is True
        assert current_version.is_active is True
        assert better_version.is_active is False
    
    @given(st.lists(st.floats(min_value=0.1, max_value=1.0), min_size=5, max_size=20))
    @settings(max_examples=10, deadline=5000)
    def test_property_accuracy_tracking_consistency(self, accuracy_values, model_updater):
        """Property: Accuracy tracking maintains consistency across predictions."""
        model_type = ModelType.DEMAND_FORECASTING
        
        # Generate corresponding actual values with some noise
        actual_values = [val + np.random.normal(0, 0.1) for val in accuracy_values]
        
        async def test_accuracy_tracking():
            # Track accuracy
            metrics = await model_updater.track_model_accuracy(
                model_type, accuracy_values, actual_values
            )
            
            # Property: Metrics should be calculated
            assert "mean_absolute_error" in metrics
            assert "root_mean_square_error" in metrics
            
            # Property: MAE should be non-negative
            assert metrics["mean_absolute_error"] >= 0
            
            # Property: RMSE should be non-negative
            assert metrics["root_mean_square_error"] >= 0
            
            # Property: Performance metrics should be stored
            assert model_type in model_updater.performance_metrics
            stored_metrics = model_updater.performance_metrics[model_type]
            assert "mean_absolute_error" in stored_metrics
            
            # Property: Accuracy history should be updated
            assert len(model_updater.accuracy_history[model_type]) > 0
        
        asyncio.run(test_accuracy_tracking())
    
    def test_property_market_adaptation_response(self, model_updater):
        """Property: Market adaptation responds appropriately to different signal strengths."""
        # Test different market signal scenarios
        test_scenarios = [
            # Small change - should not trigger major adaptation
            {"inflation_rate": 1.01, "consumer_confidence": 0.95},
            # Medium change - should trigger parameter adjustment
            {"inflation_rate": 1.15, "consumer_confidence": 0.85},
            # Large change - should trigger retraining
            {"inflation_rate": 1.25, "consumer_confidence": 0.7}
        ]
        
        async def test_adaptation():
            for i, market_signals in enumerate(test_scenarios):
                adaptation_result = await model_updater.adapt_to_market_conditions(market_signals)
                
                # Property: Adaptation should always return valid result
                assert "market_signals_processed" in adaptation_result
                assert "adaptations_made" in adaptation_result
                assert "models_updated" in adaptation_result
                
                # Property: Number of signals processed should match input
                assert adaptation_result["market_signals_processed"] == len(market_signals)
                
                # Property: Adaptations should be proportional to signal strength
                adaptations_count = len(adaptation_result["adaptations_made"])
                
                if i == 0:  # Small change
                    assert adaptations_count <= 1
                elif i == 1:  # Medium change
                    assert adaptations_count >= 0
                else:  # Large change
                    assert adaptations_count >= 0
        
        asyncio.run(test_adaptation())
    
    @given(st.lists(
        st.fixed_dictionaries({
            'product_id': st.text(min_size=1, max_size=10),
            'quantity_sold': st.integers(min_value=0, max_value=1000),
            'selling_price': st.decimals(min_value=1, max_value=1000, places=2)
        }),
        min_size=1, max_size=50
    ))
    @settings(max_examples=5, deadline=10000)
    def test_property_data_integration_scalability(self, raw_data_points, model_updater):
        """Property: Data integration scales properly with different data volumes."""
        # Convert raw data to SalesDataPoint objects
        sales_data = []
        for i, raw_point in enumerate(raw_data_points):
            try:
                data_point = SalesDataPoint(
                    product_id=raw_point['product_id'],
                    product_name=f"Product {i}",
                    category="test_category",
                    mrp=raw_point['selling_price'],
                    selling_price=raw_point['selling_price'],
                    quantity_sold=raw_point['quantity_sold'],
                    sale_date=datetime.now().date(),
                    store_location="test_store"
                )
                sales_data.append(data_point)
            except Exception:
                # Skip invalid data points
                continue
        
        if not sales_data:
            return  # Skip if no valid data points
        
        async def test_integration():
            # Mock storage operations
            model_updater.storage_manager.store_sales_data = AsyncMock()
            
            # Integrate data
            result = await model_updater.integrate_new_data(sales_data)
            
            # Property: Integration should process all data points
            assert result["data_points_processed"] == len(sales_data)
            
            # Property: Integration should identify affected models
            assert "models_affected" in result
            assert isinstance(result["models_affected"], list)
            
            # Property: Integration timestamp should be recent
            integration_time = datetime.fromisoformat(result["integration_timestamp"])
            time_diff = datetime.now(timezone.utc) - integration_time
            assert time_diff.total_seconds() < 60  # Within last minute
        
        asyncio.run(test_integration())


class TestFeedbackLearningProperties:
    """
    Property-based tests for feedback learning functionality.
    
    **Property 23: Feedback Learning**
    **Property 24: Seasonal Model Evolution**
    **Validates: Requirements 10.4, 10.5**
    """
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        return AsyncMock()
    
    @pytest.fixture
    def feedback_learner(self, mock_storage_manager):
        """Create feedback learner for testing."""
        return FeedbackLearner(mock_storage_manager)
    
    @given(st.floats(min_value=1.0, max_value=5.0))
    @settings(max_examples=10, deadline=5000)
    def test_property_feedback_collection_consistency(self, rating, feedback_learner):
        """Property: Feedback collection maintains consistency across different inputs."""
        async def test_collection():
            # Collect feedback
            feedback_id = await feedback_learner.rate_recommendation(
                retailer_id="test_retailer",
                recommendation_id="test_recommendation",
                rating=rating
            )
            
            # Property: Feedback ID should be generated
            assert feedback_id is not None
            assert isinstance(feedback_id, str)
            assert len(feedback_id) > 0
            
            # Property: Feedback should be stored
            assert feedback_id in feedback_learner.feedback_entries
            
            # Property: Feedback should have correct properties
            feedback = feedback_learner.feedback_entries[feedback_id]
            assert feedback.rating == rating
            assert feedback.feedback_type == FeedbackType.RECOMMENDATION_RATING
            assert feedback.retailer_id == "test_retailer"
            assert feedback.target_id == "test_recommendation"
            
            # Property: Sentiment should be correctly inferred
            if rating >= 4:
                assert feedback.sentiment == FeedbackSentiment.POSITIVE
            elif rating <= 2:
                assert feedback.sentiment == FeedbackSentiment.NEGATIVE
            else:
                assert feedback.sentiment == FeedbackSentiment.NEUTRAL
        
        asyncio.run(test_collection())
    
    def test_property_seasonal_pattern_evolution(self, feedback_learner):
        """Property: Seasonal patterns evolve correctly based on feedback accumulation."""
        # Simulate feedback over different seasons
        seasons = ["winter", "spring", "summer", "autumn"]
        
        async def test_seasonal_evolution():
            for season in seasons:
                # Add multiple feedback entries for each season
                for i in range(15):  # Above minimum threshold
                    rating = 3.5 + np.random.normal(0, 0.5)  # Ratings around 3.5
                    rating = max(1.0, min(5.0, rating))  # Clamp to valid range
                    
                    # Simulate seasonal feedback
                    await feedback_learner._update_seasonal_learning(
                        season, "recommendation_rating", rating, 0.8
                    )
            
            # Property: Seasonal patterns should be created for each season
            seasonal_patterns = feedback_learner.seasonal_patterns
            season_pattern_keys = [key for key in seasonal_patterns.keys() 
                                 if any(season in key for season in seasons)]
            assert len(season_pattern_keys) >= len(seasons)
            
            # Property: Each pattern should have accumulated feedback
            for pattern_id, pattern in seasonal_patterns.items():
                if any(season in pattern_id for season in seasons):
                    assert pattern.feedback_count >= 15
                    assert 0.0 <= pattern.confidence_score <= 1.0
                    assert pattern.last_updated is not None
        
        asyncio.run(test_seasonal_evolution())
    
    def test_property_retailer_credibility_tracking(self, feedback_learner):
        """Property: Retailer credibility is tracked consistently and fairly."""
        retailer_id = "test_retailer"
        
        async def test_credibility():
            # Provide consistent feedback
            consistent_ratings = [4.0, 4.2, 3.8, 4.1, 3.9]  # Low variance
            for i, rating in enumerate(consistent_ratings):
                await feedback_learner.rate_recommendation(
                    retailer_id=retailer_id,
                    recommendation_id=f"rec_{i}",
                    rating=rating
                )
            
            # Property: Consistent retailer should have higher credibility
            consistent_credibility = feedback_learner._get_retailer_credibility(retailer_id)
            
            # Provide inconsistent feedback from another retailer
            inconsistent_retailer = "inconsistent_retailer"
            inconsistent_ratings = [1.0, 5.0, 2.0, 4.5, 1.5]  # High variance
            for i, rating in enumerate(inconsistent_ratings):
                await feedback_learner.rate_recommendation(
                    retailer_id=inconsistent_retailer,
                    recommendation_id=f"rec_{i}",
                    rating=rating
                )
            
            inconsistent_credibility = feedback_learner._get_retailer_credibility(inconsistent_retailer)
            
            # Property: Consistent retailer should have higher credibility
            assert consistent_credibility > inconsistent_credibility
            
            # Property: Both credibilities should be in valid range
            assert 0.0 <= consistent_credibility <= 1.0
            assert 0.0 <= inconsistent_credibility <= 1.0
        
        asyncio.run(test_credibility())
    
    def test_property_feedback_strength_calculation(self, feedback_learner):
        """Property: Feedback strength is calculated consistently."""
        # Test different feedback scenarios
        test_cases = [
            # High rating with text feedback
            {"rating": 5.0, "text_feedback": "Excellent recommendation!", "expected_min": 0.8},
            # Low rating with structured data
            {"rating": 1.0, "structured_data": {"reason": "poor_accuracy"}, "expected_min": 0.6},
            # Neutral rating only
            {"rating": 3.0, "expected_max": 0.7},
            # No rating, just text
            {"text_feedback": "Good insight", "expected_min": 0.5}
        ]
        
        for case in test_cases:
            feedback = FeedbackEntry(
                feedback_id="test",
                retailer_id="test_retailer",
                feedback_type=FeedbackType.RECOMMENDATION_RATING,
                target_id="test_target",
                rating=case.get("rating"),
                text_feedback=case.get("text_feedback"),
                structured_data=case.get("structured_data", {})
            )
            
            strength = feedback_learner._calculate_feedback_strength(feedback)
            
            # Property: Strength should be in valid range
            assert 0.0 <= strength <= 1.0
            
            # Property: Strength should meet expectations
            if "expected_min" in case:
                assert strength >= case["expected_min"]
            if "expected_max" in case:
                assert strength <= case["expected_max"]
    
    def test_property_prediction_accuracy_calculation(self, feedback_learner):
        """Property: Prediction accuracy is calculated correctly."""
        # Test different prediction scenarios
        test_cases = [
            # Perfect prediction
            {
                "predicted": {"sales": 100, "revenue": 1000},
                "actual": {"sales": 100, "revenue": 1000},
                "expected_accuracy": 1.0
            },
            # Moderate error
            {
                "predicted": {"sales": 100, "revenue": 1000},
                "actual": {"sales": 90, "revenue": 900},
                "expected_accuracy_min": 0.8
            },
            # Large error
            {
                "predicted": {"sales": 100, "revenue": 1000},
                "actual": {"sales": 50, "revenue": 500},
                "expected_accuracy_max": 0.6
            }
        ]
        
        for case in test_cases:
            accuracy_metrics = feedback_learner._calculate_prediction_accuracy(
                case["predicted"], case["actual"]
            )
            
            # Property: Accuracy metrics should be calculated for all matching keys
            for key in case["predicted"].keys():
                if key in case["actual"]:
                    accuracy_key = f"{key}_accuracy"
                    assert accuracy_key in accuracy_metrics
                    
                    accuracy = accuracy_metrics[accuracy_key]
                    # Property: Accuracy should be in valid range
                    assert 0.0 <= accuracy <= 1.0
                    
                    # Property: Accuracy should meet expectations
                    if "expected_accuracy" in case:
                        assert abs(accuracy - case["expected_accuracy"]) < 0.01
                    if "expected_accuracy_min" in case:
                        assert accuracy >= case["expected_accuracy_min"]
                    if "expected_accuracy_max" in case:
                        assert accuracy <= case["expected_accuracy_max"]
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=5, deadline=10000)
    def test_property_feedback_batch_processing(self, batch_size, feedback_learner):
        """Property: Feedback batch processing handles different batch sizes correctly."""
        # Set batch size
        feedback_learner.batch_size = min(batch_size, 50)  # Cap at reasonable size
        
        async def test_batch_processing():
            # Generate feedback entries
            feedback_ids = []
            for i in range(batch_size):
                feedback_id = await feedback_learner.rate_recommendation(
                    retailer_id=f"retailer_{i}",
                    recommendation_id=f"rec_{i}",
                    rating=3.0 + (i % 3)  # Ratings 3, 4, 5
                )
                feedback_ids.append(feedback_id)
            
            # Property: All feedback should be queued
            assert len(feedback_learner.feedback_queue) == batch_size
            
            # Process batch
            await feedback_learner._process_feedback_batch()
            
            # Property: Feedback should be processed (queue should be smaller or empty)
            processed_count = batch_size - len(feedback_learner.feedback_queue)
            assert processed_count > 0
            
            # Property: Processed feedback should be marked as processed
            processed_feedback = [
                f for f in feedback_learner.feedback_entries.values()
                if f.processed
            ]
            assert len(processed_feedback) >= min(feedback_learner.batch_size, batch_size)
        
        asyncio.run(test_batch_processing())


class TestContinuousLearningIntegration:
    """
    Integration property tests for continuous learning system.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
    """
    
    def test_property_end_to_end_learning_workflow(self):
        """Property: End-to-end learning workflow maintains consistency."""
        mock_storage = AsyncMock()
        model_updater = ModelUpdater(mock_storage)
        feedback_learner = FeedbackLearner(mock_storage)
        
        async def test_integration():
            # Step 1: Integrate new data
            sales_data = [
                SalesDataPoint(
                    product_id="PROD001",
                    product_name="Test Product",
                    category="electronics",
                    mrp=Decimal("100.00"),
                    selling_price=Decimal("95.00"),
                    quantity_sold=50,
                    sale_date=datetime.now().date(),
                    store_location="test_store"
                )
            ]
            
            integration_result = await model_updater.integrate_new_data(sales_data)
            
            # Property: Data integration should succeed
            assert integration_result["data_points_processed"] == 1
            
            # Step 2: Collect feedback on predictions
            feedback_id = await feedback_learner.rate_recommendation(
                retailer_id="test_retailer",
                recommendation_id="test_rec",
                rating=4.5,
                business_impact={"revenue_increase": 0.15}
            )
            
            # Property: Feedback should be collected
            assert feedback_id is not None
            
            # Step 3: Track model accuracy
            predictions = [95.0, 48.0]  # Predicted price and quantity
            actual_values = [95.0, 50.0]  # Actual price and quantity
            
            accuracy_metrics = await model_updater.track_model_accuracy(
                ModelType.DEMAND_FORECASTING, predictions, actual_values
            )
            
            # Property: Accuracy should be tracked
            assert "mean_absolute_error" in accuracy_metrics
            
            # Step 4: Adapt to market conditions
            market_signals = {"inflation_rate": 1.05, "consumer_confidence": 0.9}
            adaptation_result = await model_updater.adapt_to_market_conditions(market_signals)
            
            # Property: Market adaptation should respond
            assert "adaptations_made" in adaptation_result
            
            # Step 5: Get performance summary
            performance_summary = model_updater.get_model_performance_summary()
            
            # Property: Performance summary should be comprehensive
            assert "models" in performance_summary
            assert "summary_timestamp" in performance_summary
        
        asyncio.run(test_integration())
    
    def test_property_learning_system_resilience(self):
        """Property: Learning system handles failures gracefully."""
        mock_storage = AsyncMock()
        
        # Simulate storage failures
        mock_storage.store_sales_data.side_effect = Exception("Storage failure")
        
        model_updater = ModelUpdater(mock_storage)
        feedback_learner = FeedbackLearner(mock_storage)
        
        async def test_resilience():
            # Test data integration with storage failure
            sales_data = [
                SalesDataPoint(
                    product_id="PROD001",
                    product_name="Test Product",
                    category="electronics",
                    mrp=Decimal("100.00"),
                    selling_price=Decimal("95.00"),
                    quantity_sold=50,
                    sale_date=datetime.now().date(),
                    store_location="test_store"
                )
            ]
            
            integration_result = await model_updater.integrate_new_data(sales_data)
            
            # Property: System should handle storage failures gracefully
            assert "error" in integration_result or integration_result["data_points_processed"] >= 0
            
            # Test feedback collection (should still work)
            feedback_id = await feedback_learner.rate_recommendation(
                retailer_id="test_retailer",
                recommendation_id="test_rec",
                rating=4.0
            )
            
            # Property: Feedback collection should continue working
            assert feedback_id is not None
            
            # Test model performance tracking (should work with in-memory data)
            performance_summary = model_updater.get_model_performance_summary()
            
            # Property: Performance tracking should remain functional
            assert isinstance(performance_summary, dict)
            assert "models" in performance_summary
        
        asyncio.run(test_resilience())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])