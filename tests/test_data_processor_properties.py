"""
Property-Based Tests for Data Processor Component

Property tests validating universal correctness properties for data processing,
pattern extraction, seasonal correlation, and data storage functionality.

**Property 1: Comprehensive Data Processing**
**Validates: Requirements 1.1, 1.2, 1.3, 1.5**
"""

import pytest
import tempfile
import json
import csv
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

from marketpulse_ai.components.data_processor import (
    DataProcessor, 
    DataValidationError, 
    DataQualityError
)
from marketpulse_ai.core.models import SalesDataPoint, DemandPattern, ConfidenceLevel


# Hypothesis strategies for generating test data
@composite
def sales_data_point_strategy(draw):
    """Generate valid SalesDataPoint instances."""
    product_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    product_name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'))))
    category = draw(st.sampled_from(['electronics', 'clothing', 'home', 'books', 'sports', 'beauty']))
    
    # Generate realistic price values
    mrp_value = draw(st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    mrp = Decimal(f"{mrp_value:.2f}")
    
    # Selling price should be <= MRP (Indian regulation)
    selling_price_value = draw(st.floats(min_value=1.0, max_value=float(mrp), allow_nan=False, allow_infinity=False))
    selling_price = Decimal(f"{selling_price_value:.2f}")
    
    quantity_sold = draw(st.integers(min_value=1, max_value=1000))
    
    # Generate dates within last 2 years
    base_date = date.today() - timedelta(days=730)
    days_offset = draw(st.integers(min_value=0, max_value=729))
    sale_date = base_date + timedelta(days=days_offset)
    
    store_location = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    
    seasonal_event = draw(st.one_of(
        st.none(),
        st.sampled_from(['diwali', 'holi', 'eid', 'christmas', 'new_year', 'dussehra'])
    ))
    
    return SalesDataPoint(
        product_id=product_id,
        product_name=product_name,
        category=category,
        mrp=mrp,
        selling_price=selling_price,
        quantity_sold=quantity_sold,
        sale_date=sale_date,
        store_location=store_location,
        seasonal_event=seasonal_event
    )


@composite
def sales_data_list_strategy(draw):
    """Generate lists of SalesDataPoint instances."""
    return draw(st.lists(sales_data_point_strategy(), min_size=1, max_size=50))


@composite
def market_signals_strategy(draw):
    """Generate market signals data for integration testing."""
    return {
        'price_trends': {
            'categories': draw(st.dictionaries(
                st.sampled_from(['electronics', 'clothing', 'home']),
                st.floats(min_value=-0.5, max_value=0.5),
                min_size=1, max_size=3
            )),
            'overall_trend': draw(st.floats(min_value=-0.3, max_value=0.3))
        },
        'demand_indicators': {
            'search_volume_change': draw(st.floats(min_value=-0.8, max_value=0.8)),
            'social_sentiment': draw(st.floats(min_value=-1.0, max_value=1.0))
        },
        'seasonal_adjustments': {
            'current_season': draw(st.sampled_from(['spring', 'summer', 'monsoon', 'winter'])),
            'festival_proximity': draw(st.integers(min_value=0, max_value=60))
        },
        'economic_indicators': {
            'inflation_rate': draw(st.floats(min_value=0.0, max_value=0.15)),
            'consumer_confidence': draw(st.floats(min_value=0.3, max_value=1.0))
        }
    }


class TestDataProcessorProperties:
    """
    Property-based tests for Data Processor component.
    
    **Property 1: Comprehensive Data Processing**
    **Validates: Requirements 1.1, 1.2, 1.3, 1.5**
    """
    
    @given(sales_data_list_strategy())
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_data_ingestion_completeness(self, sales_data):
        """
        **Property 1.1: Data Ingestion Completeness**
        **Validates: Requirements 1.1**
        
        Property: For any valid sales data input, the data processor should successfully
        handle the ingestion process and provide accurate processing statistics.
        """
        processor = DataProcessor()
        
        # Property: Data ingestion should handle any valid sales data
        try:
            result = await processor.ingest_sales_data(sales_data)
            
            # Property: Successful ingestion should contain processing statistics
            assert isinstance(result, dict)
            assert 'records_processed' in result
            assert 'records_accepted' in result
            assert 'status' in result
            assert 'quality_report' in result
            
            # Property: Records processed should match input
            assert result['records_processed'] == len(sales_data)
            assert result['status'] == 'success'
            
            # Property: Quality report should contain quality score
            quality_report = result['quality_report']
            assert 'quality_score' in quality_report
            assert 0.0 <= quality_report['quality_score'] <= 1.0
            
            # Property: Accepted records should be stored internally
            assert len(processor.processed_data) == result['records_accepted']
            
            # Property: Accepted records should be <= processed records
            assert result['records_accepted'] <= result['records_processed']
            
        except ValueError as e:
            # Property: Quality failures should be handled gracefully
            assert "Data quality score" in str(e) or "Data ingestion failed" in str(e)
            # This is acceptable behavior for low-quality data
    
    @given(sales_data_list_strategy())
    @settings(max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_pattern_extraction_consistency(self, sales_data):
        """
        **Property 1.2: Pattern Extraction Consistency**
        **Validates: Requirements 1.2**
        
        Property: For any sales data, pattern extraction should produce consistent,
        valid demand patterns with proper confidence levels and seasonal correlations.
        """
        processor = DataProcessor()
        
        # First ingest the data
        await processor.ingest_sales_data(sales_data)
        
        # Property: Pattern extraction should work for any valid data
        patterns = await processor.extract_demand_patterns()
        
        # Property: Patterns should be returned as a list
        assert isinstance(patterns, list)
        
        # Property: Each pattern should be valid
        for pattern in patterns:
            assert isinstance(pattern, DemandPattern)
            assert pattern.product_id is not None
            assert pattern.pattern_type is not None
            assert isinstance(pattern.confidence_level, ConfidenceLevel)
            assert 0.0 <= pattern.strength <= 1.0
            
            # Property: Pattern should have valid time range
            if pattern.start_date and pattern.end_date:
                assert pattern.start_date <= pattern.end_date
        
        # Property: Patterns should be consistent across multiple extractions
        patterns_second = await processor.extract_demand_patterns()
        assert len(patterns) == len(patterns_second)
        
        # Property: Pattern IDs should be consistent
        pattern_ids_first = {p.product_id for p in patterns}
        pattern_ids_second = {p.product_id for p in patterns_second}
        assert pattern_ids_first == pattern_ids_second
    
    @given(sales_data_list_strategy())
    @settings(max_examples=15, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_seasonal_correlation_accuracy(self, sales_data):
        """
        **Property 1.3: Seasonal Correlation Accuracy**
        **Validates: Requirements 1.3**
        
        Property: For any demand patterns, seasonal correlation should enhance patterns
        with accurate seasonal factors and maintain data consistency.
        """
        processor = DataProcessor()
        
        # First ingest data and extract basic patterns
        await processor.ingest_sales_data(sales_data)
        basic_patterns = await processor.extract_basic_patterns(sales_data)
        
        # Property: Seasonal correlation should work for any patterns
        enhanced_patterns = await processor.correlate_seasonal_events(basic_patterns)
        
        # Property: Enhanced patterns should maintain original pattern count
        assert len(enhanced_patterns) == len(basic_patterns)
        
        # Property: Each enhanced pattern should be valid
        for enhanced, original in zip(enhanced_patterns, basic_patterns):
            # Property: Core pattern data should be preserved
            assert enhanced.product_id == original.product_id
            assert enhanced.pattern_type == original.pattern_type
            
            # Property: Seasonal factors should be reasonable
            if hasattr(enhanced, 'seasonal_factors') and enhanced.seasonal_factors:
                for factor_name, factor_value in enhanced.seasonal_factors.items():
                    assert isinstance(factor_value, (int, float))
                    assert 0.0 <= factor_value <= 10.0  # Reasonable seasonal multiplier range
            
            # Property: Confidence should not decrease unreasonably
            original_confidence = original.confidence_level.value if hasattr(original.confidence_level, 'value') else 0.5
            enhanced_confidence = enhanced.confidence_level.value if hasattr(enhanced.confidence_level, 'value') else 0.5
            
            # Allow some confidence adjustment but not dramatic drops
            assert enhanced_confidence >= original_confidence * 0.5
    
    @given(market_signals_strategy())
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_market_signal_integration(self, market_signals):
        """
        **Property 1.4: Market Signal Integration**
        **Validates: Requirements 1.3**
        
        Property: For any external market signals, integration should produce
        valid insights without corrupting existing data.
        """
        processor = DataProcessor()
        
        # Property: Market signal integration should handle any valid signals
        result = await processor.integrate_market_signals(market_signals)
        
        # Property: Result should contain integration insights
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'signals_processed' in result
        assert 'signals_integrated' in result
        assert result['status'] == 'success'
        assert 'market_insights' in result
        assert 'correlation_scores' in result
        
        # Property: Integration should process signals correctly
        assert result['signals_processed'] >= 0
        assert result['signals_integrated'] >= 0
        
        # Property: Correlation scores should be reasonable
        correlation_scores = result['correlation_scores']
        assert isinstance(correlation_scores, dict)
        
        for signal_type, score in correlation_scores.items():
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0
        
        # Property: Market insights should be structured
        market_insights = result['market_insights']
        assert isinstance(market_insights, dict)
        
        # Property: Each insight category should have valid structure
        for category, insights in market_insights.items():
            assert isinstance(insights, dict)
            if 'impact_score' in insights:
                assert 0.0 <= insights['impact_score'] <= 1.0
    
    @given(sales_data_list_strategy())
    @settings(max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_data_storage_integrity(self, sales_data):
        """
        **Property 1.5: Data Storage Integrity**
        **Validates: Requirements 1.5**
        
        Property: For any processed patterns, storage operations should maintain
        data integrity and provide reliable persistence.
        """
        processor = DataProcessor()
        
        # Process data and extract patterns
        await processor.ingest_sales_data(sales_data)
        patterns = await processor.extract_demand_patterns()
        
        # Set up storage manager
        mock_storage_manager = AsyncMock()
        mock_storage_manager.store_patterns.return_value = {
            'status': 'success',
            'stored_count': len(patterns)
        }
        mock_storage_manager.retrieve_demand_patterns.return_value = []
        processor.set_storage_manager(mock_storage_manager)
        
        # Property: Pattern storage should succeed for any valid patterns
        storage_result = await processor.store_patterns(patterns)
        
        # Property: Storage operation should complete successfully
        assert storage_result is True
        
        # Property: Storage manager should be called with correct patterns
        mock_storage_manager.store_patterns.assert_called_once()
        stored_patterns = mock_storage_manager.store_patterns.call_args[0][0]
        
        # Property: Stored patterns should match original patterns
        assert len(stored_patterns) == len(patterns)
        
        for stored, original in zip(stored_patterns, patterns):
            assert stored.product_id == original.product_id
            assert stored.pattern_type == original.pattern_type
            assert stored.confidence_level == original.confidence_level
    
    @given(st.lists(sales_data_point_strategy(), min_size=5, max_size=20))
    @settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_data_quality_validation(self, sales_data):
        """
        **Property 1.6: Data Quality Validation**
        **Validates: Requirements 1.4**
        
        Property: For any sales data, quality validation should identify issues
        and provide accurate quality metrics.
        """
        processor = DataProcessor()
        
        # Property: Quality validation should work for any data
        cleaned_data, quality_report = await processor.validate_data_quality(sales_data)
        
        # Property: Cleaned data should be a subset of original data
        assert len(cleaned_data) <= len(sales_data)
        assert isinstance(cleaned_data, list)
        
        # Property: Quality report should contain required metrics
        assert isinstance(quality_report, dict)
        assert 'total_records' in quality_report
        assert 'valid_records' in quality_report
        assert 'quality_score' in quality_report
        assert 'issues' in quality_report
        
        # Property: Quality metrics should be consistent
        assert quality_report['total_records'] == len(sales_data)
        assert quality_report['valid_records'] == len(cleaned_data)
        assert 0.0 <= quality_report['quality_score'] <= 1.0
        
        # Property: All cleaned records should be valid SalesDataPoint instances
        for record in cleaned_data:
            assert isinstance(record, SalesDataPoint)
            assert record.mrp >= record.selling_price  # MRP compliance
            assert record.quantity_sold > 0
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_csv_loading_robustness(self, filename, tmp_path):
        """
        **Property 1.7: CSV Loading Robustness**
        **Validates: Requirements 1.1**
        
        Property: CSV loading should handle various file conditions gracefully
        and provide appropriate error handling.
        """
        processor = DataProcessor()
        
        # Create a safe filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_filename:
            safe_filename = "test_file"
        
        csv_file = tmp_path / f"{safe_filename}.csv"
        
        # Property: Non-existent files should raise appropriate errors
        with pytest.raises(DataValidationError):
            await processor.load_from_csv(csv_file)
        
        # Property: Empty files should be handled gracefully
        csv_file.write_text("")
        try:
            result = await processor.load_from_csv(csv_file)
            assert isinstance(result, list)
            assert len(result) == 0
        except DataValidationError:
            # Empty files may legitimately cause validation errors
            pass
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(), st.integers(), st.floats(allow_nan=False)),
        min_size=1, max_size=10
    ))
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_api_data_handling(self, api_data):
        """
        **Property 1.8: API Data Handling Robustness**
        **Validates: Requirements 1.1**
        
        Property: API data loading should handle various data structures
        and provide appropriate validation.
        """
        processor = DataProcessor()
        
        # Property: API data loading should not crash on any dictionary input
        try:
            result = await processor.load_from_api(api_data)
            assert isinstance(result, list)
            # If successful, all results should be valid SalesDataPoint instances
            for item in result:
                assert isinstance(item, SalesDataPoint)
        except (DataValidationError, ValueError, TypeError):
            # Invalid data should raise appropriate exceptions
            pass


# Additional integration property tests
class TestDataProcessorIntegrationProperties:
    """Integration property tests for complete data processing workflows."""
    
    @given(sales_data_list_strategy(), market_signals_strategy())
    @settings(max_examples=5, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_end_to_end_processing_workflow(self, sales_data, market_signals):
        """
        **Property 1.9: End-to-End Processing Workflow**
        **Validates: Requirements 1.1, 1.2, 1.3, 1.5**
        
        Property: Complete data processing workflow should maintain consistency
        and data integrity from ingestion to storage.
        """
        processor = DataProcessor()
        
        # Set up storage manager
        storage_manager = AsyncMock()
        storage_manager.store_patterns.return_value = {
            'status': 'success',
            'stored_count': 1
        }
        storage_manager.retrieve_patterns.return_value = []
        storage_manager.retrieve_sales_data.return_value = []
        processor.set_storage_manager(storage_manager)
        
        # Property: Complete workflow should succeed for any valid inputs
        
        # Step 1: Data ingestion
        try:
            ingestion_result = await processor.ingest_sales_data(sales_data)
            assert ingestion_result['status'] == 'success'
        except ValueError:
            # Skip if data quality is too low for this test
            return
        
        # Step 2: Pattern extraction
        patterns = await processor.extract_demand_patterns()
        assert isinstance(patterns, list)
        
        # Step 3: Seasonal correlation
        enhanced_patterns = await processor.correlate_seasonal_events(patterns)
        assert len(enhanced_patterns) == len(patterns)
        
        # Step 4: Market signal integration
        market_result = await processor.integrate_market_signals(market_signals)
        assert market_result['status'] == 'success'
        
        # Step 5: Pattern storage
        storage_result = await processor.store_patterns(enhanced_patterns)
        assert storage_result is True
        
        # Property: Data consistency should be maintained throughout
        assert len(processor.processed_data) == len(sales_data)
        
        # Property: All operations should complete without data corruption
        for original, processed in zip(sales_data, processor.processed_data):
            assert processed.product_id == original.product_id
            assert processed.mrp == original.mrp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])