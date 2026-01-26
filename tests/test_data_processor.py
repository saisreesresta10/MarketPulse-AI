"""
Unit tests for the DataProcessor component.

Tests data ingestion, validation, cleansing, and pattern extraction functionality.
"""

import json
import pytest
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd

from marketpulse_ai.components.data_processor import (
    DataProcessor, 
    DataValidationError, 
    DataQualityError
)
from marketpulse_ai.core.models import SalesDataPoint, DemandPattern, ConfidenceLevel


class TestDataProcessor:
    """Test cases for DataProcessor component."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for testing."""
        return [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product 1",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=date.today() - timedelta(days=1),
                store_location="STORE001"
            ),
            SalesDataPoint(
                product_id="PROD002",
                product_name="Test Product 2",
                category="clothing",
                mrp=Decimal("500.00"),
                selling_price=Decimal("450.00"),
                quantity_sold=3,
                sale_date=date.today() - timedelta(days=2),
                store_location="STORE002",
                seasonal_event="diwali"
            )
        ]
    
    @pytest.fixture
    def csv_test_file(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_content = """product_id,product_name,category,mrp,selling_price,quantity_sold,sale_date,store_location,seasonal_event
PROD001,Test Product 1,electronics,1000.00,900.00,5,2024-01-15,STORE001,
PROD002,Test Product 2,clothing,500.00,450.00,3,2024-01-14,STORE002,diwali"""
        
        csv_file = tmp_path / "test_sales.csv"
        csv_file.write_text(csv_content)
        return csv_file
    
    @pytest.fixture
    def json_test_file(self, tmp_path):
        """Create a temporary JSON file for testing."""
        json_data = {
            "sales_data": [
                {
                    "product_id": "PROD001",
                    "product_name": "Test Product 1",
                    "category": "electronics",
                    "mrp": "1000.00",
                    "selling_price": "900.00",
                    "quantity_sold": 5,
                    "sale_date": "2024-01-15",
                    "store_location": "STORE001"
                },
                {
                    "product_id": "PROD002",
                    "product_name": "Test Product 2",
                    "category": "clothing",
                    "mrp": "500.00",
                    "selling_price": "450.00",
                    "quantity_sold": 3,
                    "sale_date": "2024-01-14",
                    "store_location": "STORE002",
                    "seasonal_event": "diwali"
                }
            ]
        }
        
        json_file = tmp_path / "test_sales.json"
        json_file.write_text(json.dumps(json_data))
        return json_file

    @pytest.mark.asyncio
    async def test_load_from_csv_success(self, processor, csv_test_file):
        """Test successful CSV loading."""
        data = await processor.load_from_csv(csv_test_file)
        
        assert len(data) == 2
        assert data[0].product_id == "PROD001"
        assert data[0].product_name == "Test Product 1"
        assert data[1].seasonal_event == "diwali"
    
    @pytest.mark.asyncio
    async def test_load_from_csv_file_not_found(self, processor):
        """Test CSV loading with non-existent file."""
        with pytest.raises(DataValidationError, match="CSV file not found"):
            await processor.load_from_csv("nonexistent.csv")
    
    @pytest.mark.asyncio
    async def test_load_from_csv_missing_columns(self, processor, tmp_path):
        """Test CSV loading with missing required columns."""
        csv_content = "product_id,product_name\nPROD001,Test Product"
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text(csv_content)
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            await processor.load_from_csv(csv_file)
    
    @pytest.mark.asyncio
    async def test_load_from_json_success(self, processor, json_test_file):
        """Test successful JSON loading."""
        data = await processor.load_from_json(json_test_file)
        
        assert len(data) == 2
        assert data[0].product_id == "PROD001"
        assert data[1].seasonal_event == "diwali"
    
    @pytest.mark.asyncio
    async def test_load_from_json_file_not_found(self, processor):
        """Test JSON loading with non-existent file."""
        with pytest.raises(DataValidationError, match="JSON file not found"):
            await processor.load_from_json("nonexistent.json")
    
    @pytest.mark.asyncio
    async def test_load_from_api_success(self, processor):
        """Test successful API data loading."""
        api_data = {
            "data": [
                {
                    "product_id": "PROD001",
                    "product_name": "Test Product 1",
                    "category": "electronics",
                    "mrp": "1000.00",
                    "selling_price": "900.00",
                    "quantity_sold": 5,
                    "sale_date": "2024-01-15",
                    "store_location": "STORE001"
                }
            ]
        }
        
        data = await processor.load_from_api(api_data)
        
        assert len(data) == 1
        assert data[0].product_id == "PROD001"
    
    @pytest.mark.asyncio
    async def test_load_from_api_invalid_format(self, processor):
        """Test API loading with invalid format."""
        api_data = {"invalid": "format"}
        
        with pytest.raises(DataValidationError, match="Invalid API data format"):
            await processor.load_from_api(api_data)
    
    @pytest.mark.asyncio
    async def test_validate_data_quality_success(self, processor, sample_sales_data):
        """Test successful data quality validation."""
        cleaned_data, quality_report = await processor.validate_data_quality(sample_sales_data)
        
        assert len(cleaned_data) == 2
        assert quality_report['total_records'] == 2
        assert quality_report['valid_records'] == 2
        assert quality_report['quality_score'] == 1.0
    
    @pytest.mark.asyncio
    async def test_validate_data_quality_empty_data(self, processor):
        """Test data quality validation with empty data."""
        with pytest.raises(DataQualityError, match="No data provided"):
            await processor.validate_data_quality([])
    
    @pytest.mark.asyncio
    async def test_validate_data_quality_duplicates(self, processor):
        """Test data quality validation with duplicate records."""
        test_date = date.today() - timedelta(days=1)  # Use past date
        duplicate_data = [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=test_date,
                store_location="STORE001"
            ),
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=test_date,
                store_location="STORE001"
            ),
            # Add more valid records to meet quality threshold
            SalesDataPoint(
                product_id="PROD002",
                product_name="Test Product 2",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=test_date,
                store_location="STORE002"
            ),
            SalesDataPoint(
                product_id="PROD003",
                product_name="Test Product 3",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=test_date,
                store_location="STORE003"
            ),
            SalesDataPoint(
                product_id="PROD004",
                product_name="Test Product 4",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=test_date,
                store_location="STORE004"
            )
        ]
        
        cleaned_data, quality_report = await processor.validate_data_quality(duplicate_data)
        
        assert len(cleaned_data) == 4  # 4 unique records after removing 1 duplicate
        assert quality_report['duplicates_removed'] == 1
        assert quality_report['total_records'] == 5
    
    @pytest.mark.asyncio
    async def test_cleanse_data(self, processor, sample_sales_data):
        """Test data cleansing functionality."""
        # Modify sample data to test cleansing
        sample_sales_data[0].product_name = "  test product 1  "
        sample_sales_data[0].category = "  ELECTRONICS  "
        sample_sales_data[0].store_location = "store001"
        
        cleansed_data = await processor.cleanse_data(sample_sales_data)
        
        assert cleansed_data[0].product_name == "Test Product 1"
        assert cleansed_data[0].category == "electronics"
        assert cleansed_data[0].store_location == "STORE001"
    
    @pytest.mark.asyncio
    async def test_extract_basic_patterns(self, processor):
        """Test basic pattern extraction."""
        # Create test data with trend
        test_data = []
        for i in range(10):
            test_data.append(
                SalesDataPoint(
                    product_id="PROD001",
                    product_name="Test Product",
                    category="electronics",
                    mrp=Decimal("1000.00"),
                    selling_price=Decimal("900.00"),
                    quantity_sold=5 + i,  # Increasing trend
                    sale_date=date.today() - timedelta(days=10-i),
                    store_location="STORE001"
                )
            )
        
        patterns = await processor.extract_basic_patterns(test_data)
        
        assert len(patterns) == 1
        assert patterns[0].product_id == "PROD001"
        assert patterns[0].trend_direction == "increasing"
        assert patterns[0].supporting_data_points == 10
    
    @pytest.mark.asyncio
    async def test_extract_basic_patterns_insufficient_data(self, processor):
        """Test pattern extraction with insufficient data."""
        test_data = [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=date.today(),
                store_location="STORE001"
            )
        ]
        
        patterns = await processor.extract_basic_patterns(test_data)
        
        assert len(patterns) == 0  # Not enough data points
    
    @pytest.mark.asyncio
    async def test_ingest_sales_data_success(self, processor, sample_sales_data):
        """Test successful sales data ingestion."""
        result = await processor.ingest_sales_data(sample_sales_data)
        
        assert result['status'] == 'success'
        assert result['records_processed'] == 2
        assert result['records_accepted'] == 2
        assert result['records_rejected'] == 0
        assert len(processor.processed_data) == 2
    
    @pytest.mark.asyncio
    async def test_extract_demand_patterns(self, processor, sample_sales_data):
        """Test demand pattern extraction."""
        # First ingest data
        await processor.ingest_sales_data(sample_sales_data)
        
        # Extract patterns for all products
        patterns = await processor.extract_demand_patterns()
        
        assert len(patterns) == 0  # Not enough data points per product
        
        # Test with specific product IDs
        patterns = await processor.extract_demand_patterns(["PROD001"])
        assert len(patterns) == 0  # Still not enough data points
    
    @pytest.mark.asyncio
    async def test_correlate_seasonal_events(self, processor):
        """Test seasonal event correlation."""
        # Create test patterns
        test_patterns = [
            DemandPattern(
                product_id="PROD001",
                pattern_type="basic_trend",
                description="Test pattern",
                confidence_level=ConfidenceLevel.MEDIUM,
                volatility_score=0.5,
                supporting_data_points=5,
                date_range_start=date.today() - timedelta(days=30),
                date_range_end=date.today()
            )
        ]
        
        # Add some processed data for correlation
        processor.processed_data = [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=10,
                sale_date=date(2024, 10, 15),  # Diwali season
                store_location="STORE001"
            ),
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=date(2024, 1, 15),  # Non-festival
                store_location="STORE001"
            )
        ]
        
        enhanced_patterns = await processor.correlate_seasonal_events(test_patterns)
        
        assert len(enhanced_patterns) == 1
        assert 'diwali' in enhanced_patterns[0].seasonal_factors
    
    @pytest.mark.asyncio
    async def test_generate_seasonal_analysis_report(self, processor):
        """Test comprehensive seasonal analysis report generation."""
        # Add test data
        test_data = []
        for i in range(24):  # 2 years of monthly data
            month = (i % 12) + 1
            year = 2023 if i < 12 else 2024
            
            # Higher sales during Diwali months (Oct-Nov)
            quantity = 15 if month in [10, 11] else 8
            
            test_data.append(
                SalesDataPoint(
                    product_id="PROD001",
                    product_name="Electronics Item",
                    category="electronics",
                    mrp=Decimal("1000.00"),
                    selling_price=Decimal("900.00"),
                    quantity_sold=quantity,
                    sale_date=date(year, month, 15),
                    store_location="STORE001"
                )
            )
        
        # Ingest the test data
        await processor.ingest_sales_data(test_data)
        
        # Generate report
        report = await processor.generate_seasonal_analysis_report()
        
        assert report['status'] == 'success'
        assert 'seasonal_insights' in report
        assert 'festival_correlations' in report
        assert 'cyclical_patterns' in report
        assert 'category_analysis' in report
        assert 'recommendations' in report
        
        # Check if Diwali is detected as a high-performing event
        if 'diwali' in report['seasonal_insights']:
            diwali_data = report['seasonal_insights']['diwali']
            assert diwali_data['boost_factor'] > 1.0  # Should show boost during Diwali
    
    @pytest.mark.asyncio
    async def test_category_seasonal_correlations(self, processor):
        """Test category-specific seasonal correlations."""
        # Create test data with electronics during Diwali
        test_data = [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Electronics Item",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=20,  # High sales during Diwali
                sale_date=date(2024, 10, 15),
                store_location="STORE001"
            ),
            SalesDataPoint(
                product_id="PROD001",
                product_name="Electronics Item",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=8,  # Normal sales
                sale_date=date(2024, 5, 15),
                store_location="STORE001"
            )
        ]
        
        seasonal_events = await processor._get_seasonal_calendar()
        correlations = await processor._calculate_category_seasonal_correlations(test_data, seasonal_events)
        
        # Electronics should have correlation with Diwali
        assert 'diwali_correlation' in correlations
        # The correlation should be positive (indicating some correlation)
        assert correlations['diwali_correlation'] > 0.5
    
    @pytest.mark.asyncio
    async def test_pre_festival_patterns(self, processor):
        """Test pre-festival buying pattern detection."""
        # Create test data with pre-festival boost
        test_data = [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=15,  # Pre-festival boost (September before Diwali)
                sale_date=date(2024, 9, 15),
                store_location="STORE001"
            ),
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=8,  # Normal sales
                sale_date=date(2024, 5, 15),
                store_location="STORE001"
            )
        ]
        
        seasonal_events = await processor._get_seasonal_calendar()
        pre_festival_patterns = await processor._detect_pre_festival_patterns(test_data, seasonal_events)
        
        # Should detect pre-festival boost for Diwali
        assert len(pre_festival_patterns) > 0
        assert any('diwali' in key for key in pre_festival_patterns.keys())
    
    @pytest.mark.asyncio
    async def test_weather_correlations(self, processor):
        """Test weather-related seasonal correlations."""
        # Create test data across different seasons
        test_data = [
            # Summer sales
            SalesDataPoint(
                product_id="PROD001",
                product_name="Cooling Product",
                category="appliances",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=20,
                sale_date=date(2024, 5, 15),  # Summer
                store_location="STORE001"
            ),
            # Winter sales
            SalesDataPoint(
                product_id="PROD001",
                product_name="Cooling Product",
                category="appliances",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=date(2024, 1, 15),  # Winter
                store_location="STORE001"
            )
        ]
        
        weather_correlations = await processor._calculate_weather_correlations(test_data)
        
        assert 'summer_factor' in weather_correlations
        assert 'winter_factor' in weather_correlations
        assert weather_correlations['summer_factor'] > weather_correlations['winter_factor']
    
    @pytest.mark.asyncio
    async def test_advanced_cyclical_patterns(self, processor):
        """Test advanced cyclical pattern detection."""
        # Create test data with clear monthly pattern
        test_data = []
        for month in range(1, 13):
            # Higher sales in Q4 (Oct-Dec)
            quantity = 15 if month >= 10 else 8
            
            test_data.append(
                SalesDataPoint(
                    product_id="PROD001",
                    product_name="Test Product",
                    category="electronics",
                    mrp=Decimal("1000.00"),
                    selling_price=Decimal("900.00"),
                    quantity_sold=quantity,
                    sale_date=date(2024, month, 15),
                    store_location="STORE001"
                )
            )
        
        cyclical_patterns = await processor._detect_cyclical_patterns(test_data)
        
        assert len(cyclical_patterns) > 0
        # Should detect quarterly pattern
        quarterly_patterns = [p for p in cyclical_patterns if p['type'] == 'quarterly']
        assert len(quarterly_patterns) > 0
        assert quarterly_patterns[0]['peak_quarter'] == 'Q4'
    
    @pytest.mark.asyncio
    async def test_year_over_year_factors(self, processor):
        """Test year-over-year seasonal factor calculation."""
        # Create multi-year test data
        test_data = []
        for year in [2023, 2024]:
            for month in [10, 11]:  # Diwali months
                # Show growth in 2024
                quantity = 12 if year == 2023 else 18
                
                test_data.append(
                    SalesDataPoint(
                        product_id="PROD001",
                        product_name="Test Product",
                        category="electronics",
                        mrp=Decimal("1000.00"),
                        selling_price=Decimal("900.00"),
                        quantity_sold=quantity,
                        sale_date=date(year, month, 15),
                        store_location="STORE001"
                    )
                )
        
        seasonal_events = await processor._get_seasonal_calendar()
        yoy_factors = await processor._calculate_year_over_year_factors(test_data, seasonal_events)
        
        # Should detect positive growth for Diwali
        assert 'diwali_yoy_growth' in yoy_factors
        assert yoy_factors['diwali_yoy_growth'] > 0
    
    @pytest.mark.asyncio
    async def test_integrate_market_signals(self, processor):
        """Test market signal integration."""
        external_data = {
            'synthetic': {
                'price_trends': {'electronics': 'increasing'},
                'demand_indicators': {'seasonal_boost': 1.2}
            },
            'unauthorized_source': {
                'private_data': 'should_be_rejected'
            }
        }
        
        result = await processor.integrate_market_signals(external_data)
        
        assert result['status'] == 'success'
        assert result['signals_processed'] == 1  # Only synthetic source processed
        assert len(result['warnings']) == 1  # Unauthorized source warning
    
    @pytest.mark.asyncio
    async def test_store_patterns(self, processor):
        """Test pattern storage."""
        test_patterns = [
            DemandPattern(
                product_id="PROD001",
                pattern_type="basic_trend",
                description="Test pattern",
                confidence_level=ConfidenceLevel.MEDIUM,
                volatility_score=0.5,
                supporting_data_points=5,
                date_range_start=date.today() - timedelta(days=30),
                date_range_end=date.today()
            )
        ]
        
        result = await processor.store_patterns(test_patterns)
        
        assert result is True
        assert len(processor.patterns_cache) > 0


class TestDataProcessorEdgeCases:
    """Test edge cases and error conditions for DataProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    @pytest.mark.asyncio
    async def test_csv_with_invalid_data_types(self, processor, tmp_path):
        """Test CSV loading with invalid data types."""
        csv_content = """product_id,product_name,category,mrp,selling_price,quantity_sold,sale_date,store_location
PROD001,Test Product,electronics,invalid_price,900.00,5,2024-01-15,STORE001"""
        
        csv_file = tmp_path / "invalid_types.csv"
        csv_file.write_text(csv_content)
        
        # Should skip invalid rows and continue
        data = await processor.load_from_csv(csv_file)
        assert len(data) == 0  # Invalid row skipped
    
    @pytest.mark.asyncio
    async def test_json_with_mixed_date_formats(self, processor, tmp_path):
        """Test JSON loading with various date formats."""
        json_data = [
            {
                "product_id": "PROD001",
                "product_name": "Test Product 1",
                "category": "electronics",
                "mrp": "1000.00",
                "selling_price": "900.00",
                "quantity_sold": 5,
                "sale_date": "2024-01-15T00:00:00",  # ISO format
                "store_location": "STORE001"
            }
        ]
        
        json_file = tmp_path / "mixed_dates.json"
        json_file.write_text(json.dumps(json_data))
        
        data = await processor.load_from_json(json_file)
        assert len(data) == 1
    
    @pytest.mark.asyncio
    async def test_api_data_with_multiple_date_formats(self, processor):
        """Test API data loading with multiple date formats."""
        api_data = [
            {
                "product_id": "PROD001",
                "product_name": "Test Product 1",
                "category": "electronics",
                "mrp": "1000.00",
                "selling_price": "900.00",
                "quantity_sold": 5,
                "sale_date": "15/01/2024",  # DD/MM/YYYY format
                "store_location": "STORE001"
            },
            {
                "product_id": "PROD002",
                "product_name": "Test Product 2",
                "category": "electronics",
                "mrp": "1000.00",
                "selling_price": "900.00",
                "quantity_sold": 5,
                "sale_date": "2024-01-16",  # YYYY-MM-DD format
                "store_location": "STORE001"
            }
        ]
        
        data = await processor.load_from_api(api_data)
        assert len(data) == 2
    
    @pytest.mark.asyncio
    async def test_data_quality_below_threshold(self, processor):
        """Test data quality validation below acceptable threshold."""
        # Create data that will fail validation during quality check (not during model creation)
        # Use valid SalesDataPoint objects but with conditions that will be flagged as quality issues
        test_date = date.today() - timedelta(days=1)
        bad_data = []
        
        # Create 10 records with very high quantities (will be flagged as outliers)
        for i in range(10):
            bad_data.append(
                SalesDataPoint(
                    product_id=f"PROD{i:03d}",
                    product_name="Test Product",
                    category="electronics",
                    mrp=Decimal("1000.00"),
                    selling_price=Decimal("900.00"),
                    quantity_sold=15000,  # Very high quantity - will be flagged as outlier
                    sale_date=test_date,
                    store_location="STORE001"
                )
            )
        
        # Add one valid record
        bad_data.append(
            SalesDataPoint(
                product_id="PROD_VALID",
                product_name="Valid Product",
                category="electronics",
                mrp=Decimal("1000.00"),
                selling_price=Decimal("900.00"),
                quantity_sold=5,
                sale_date=test_date,
                store_location="STORE001"
            )
        )
        
        with pytest.raises(DataQualityError, match="below threshold"):
            await processor.validate_data_quality(bad_data)