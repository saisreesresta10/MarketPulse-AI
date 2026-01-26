"""
Data Processor component for MarketPulse AI.

This module implements data ingestion, validation, and pattern extraction
functionality for sales data analysis and processing.
"""

import csv
import json
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID

import pandas as pd
import numpy as np
from pydantic import ValidationError

from ..core.interfaces import DataProcessorInterface
from ..core.models import SalesDataPoint, DemandPattern, ConfidenceLevel
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataQualityError(Exception):
    """Raised when data quality issues are detected."""
    pass


class DataProcessor(DataProcessorInterface):
    """
    Implementation of data processing and pattern extraction.
    
    Handles data ingestion from multiple sources, validation, cleansing,
    and pattern extraction for sales data analysis.
    """
    
    def __init__(self, settings=None):
        """Initialize the data processor with configuration."""
        self.settings = settings  # Allow injection for testing
        self.processed_data: List[SalesDataPoint] = []
        self.patterns_cache: Dict[str, List[DemandPattern]] = {}
        self.storage_manager = None  # Will be set via dependency injection
    
    def set_storage_manager(self, storage_manager):
        """
        Set the storage manager for persistent data operations.
        
        Args:
            storage_manager: StorageManager instance for data persistence
        """
        self.storage_manager = storage_manager
        logger.info("Storage manager configured for data processor")
        
    async def load_from_csv(self, file_path: Union[str, Path]) -> List[SalesDataPoint]:
        """
        Load sales data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of validated sales data points
            
        Raises:
            DataValidationError: If CSV format or data is invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DataValidationError(f"CSV file not found: {file_path}")
            
            logger.info(f"Loading sales data from CSV: {file_path}")
            
            # Read CSV with pandas for better handling
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = {
                'product_id', 'product_name', 'category', 'mrp', 
                'selling_price', 'quantity_sold', 'sale_date', 'store_location'
            }
            
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
            
            # Convert and validate data
            sales_data = []
            for index, row in df.iterrows():
                try:
                    # Handle date conversion
                    sale_date = pd.to_datetime(row['sale_date']).date()
                    
                    # Create sales data point
                    data_point = SalesDataPoint(
                        product_id=str(row['product_id']),
                        product_name=str(row['product_name']),
                        category=str(row['category']),
                        mrp=Decimal(str(row['mrp'])),
                        selling_price=Decimal(str(row['selling_price'])),
                        quantity_sold=int(row['quantity_sold']),
                        sale_date=sale_date,
                        store_location=str(row['store_location']),
                        seasonal_event=str(row.get('seasonal_event', '')) or None
                    )
                    sales_data.append(data_point)
                    
                except (ValueError, ValidationError, Exception) as e:
                    logger.warning(f"Skipping invalid row {index}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(sales_data)} valid records from CSV")
            return sales_data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise DataValidationError(f"Failed to load CSV data: {e}")
    
    async def load_from_json(self, file_path: Union[str, Path]) -> List[SalesDataPoint]:
        """
        Load sales data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of validated sales data points
            
        Raises:
            DataValidationError: If JSON format or data is invalid
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DataValidationError(f"JSON file not found: {file_path}")
            
            logger.info(f"Loading sales data from JSON: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single object and array formats
            if isinstance(data, dict):
                if 'sales_data' in data:
                    records = data['sales_data']
                else:
                    records = [data]
            elif isinstance(data, list):
                records = data
            else:
                raise DataValidationError("Invalid JSON format: expected object or array")
            
            # Validate and convert records
            sales_data = []
            for index, record in enumerate(records):
                try:
                    # Handle date conversion
                    if isinstance(record.get('sale_date'), str):
                        record['sale_date'] = datetime.fromisoformat(record['sale_date']).date()
                    
                    data_point = SalesDataPoint(**record)
                    sales_data.append(data_point)
                    
                except (ValueError, ValidationError) as e:
                    logger.warning(f"Skipping invalid record {index}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(sales_data)} valid records from JSON")
            return sales_data
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise DataValidationError(f"Failed to load JSON data: {e}")
    
    async def load_from_api(self, api_data: Dict[str, Any]) -> List[SalesDataPoint]:
        """
        Load sales data from API response.
        
        Args:
            api_data: Dictionary containing API response data
            
        Returns:
            List of validated sales data points
            
        Raises:
            DataValidationError: If API data format is invalid
        """
        try:
            logger.info("Processing sales data from API")
            
            # Extract records from API response
            if 'data' in api_data:
                records = api_data['data']
            elif 'sales_data' in api_data:
                records = api_data['sales_data']
            elif isinstance(api_data, list):
                records = api_data
            else:
                raise DataValidationError("Invalid API data format")
            
            # Validate and convert records
            sales_data = []
            for index, record in enumerate(records):
                try:
                    # Handle various date formats from API
                    if 'sale_date' in record:
                        if isinstance(record['sale_date'], str):
                            # Try multiple date formats
                            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                                try:
                                    record['sale_date'] = datetime.strptime(record['sale_date'], fmt).date()
                                    break
                                except ValueError:
                                    continue
                            else:
                                # If no format worked, try ISO format
                                record['sale_date'] = datetime.fromisoformat(record['sale_date']).date()
                    
                    data_point = SalesDataPoint(**record)
                    sales_data.append(data_point)
                    
                except (ValueError, ValidationError) as e:
                    logger.warning(f"Skipping invalid API record {index}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(sales_data)} valid records from API")
            return sales_data
            
        except Exception as e:
            logger.error(f"Error processing API data: {e}")
            raise DataValidationError(f"Failed to process API data: {e}")
    
    async def validate_data_quality(self, data: List[SalesDataPoint]) -> Tuple[List[SalesDataPoint], Dict[str, Any]]:
        """
        Validate and assess data quality.
        
        Args:
            data: List of sales data points to validate
            
        Returns:
            Tuple of (cleaned_data, quality_report)
            
        Raises:
            DataQualityError: If data quality is below acceptable threshold
        """
        logger.info(f"Validating data quality for {len(data)} records")
        
        quality_report = {
            'total_records': len(data),
            'valid_records': 0,
            'issues': [],
            'duplicates_removed': 0,
            'outliers_flagged': 0,
            'missing_data_filled': 0
        }
        
        if not data:
            raise DataQualityError("No data provided for validation")
        
        # Remove duplicates based on product_id, sale_date, and store_location
        seen = set()
        deduplicated_data = []
        
        for record in data:
            key = (record.product_id, record.sale_date, record.store_location)
            if key not in seen:
                seen.add(key)
                deduplicated_data.append(record)
            else:
                quality_report['duplicates_removed'] += 1
        
        # Validate business rules
        cleaned_data = []
        for record in deduplicated_data:
            issues = []
            
            # Check MRP compliance
            if record.selling_price > record.mrp:
                issues.append(f"Selling price {record.selling_price} exceeds MRP {record.mrp}")
            
            # Check for reasonable quantities
            if record.quantity_sold > 10000:  # Configurable threshold
                issues.append(f"Unusually high quantity: {record.quantity_sold}")
                quality_report['outliers_flagged'] += 1
            
            # Check for future dates
            if record.sale_date > date.today():
                issues.append(f"Future sale date: {record.sale_date}")
            
            # Check for very old dates (more than 5 years)
            if record.sale_date < date.today() - timedelta(days=5*365):
                issues.append(f"Very old sale date: {record.sale_date}")
            
            if issues:
                quality_report['issues'].extend(issues)
                logger.warning(f"Data quality issues for record {record.id}: {issues}")
            else:
                cleaned_data.append(record)
                quality_report['valid_records'] += 1
        
        # Calculate quality score
        quality_score = quality_report['valid_records'] / quality_report['total_records']
        quality_report['quality_score'] = quality_score
        
        # Check if quality meets minimum threshold (configurable, default 60% for testing)
        min_quality_threshold = getattr(self, 'min_quality_threshold', 0.6)
        if quality_score < min_quality_threshold:
            raise DataQualityError(
                f"Data quality score {quality_score:.2%} below threshold {min_quality_threshold:.2%}"
            )
        
        logger.info(f"Data quality validation complete. Score: {quality_score:.2%}")
        return cleaned_data, quality_report
    
    async def cleanse_data(self, data: List[SalesDataPoint]) -> List[SalesDataPoint]:
        """
        Cleanse and normalize sales data.
        
        Args:
            data: List of sales data points to cleanse
            
        Returns:
            List of cleansed sales data points
        """
        logger.info(f"Cleansing {len(data)} sales records")
        
        cleansed_data = []
        for record in data:
            # Create a copy to avoid modifying original
            cleansed_record = record.model_copy()
            
            # Normalize text fields
            cleansed_record.product_name = cleansed_record.product_name.strip().title()
            cleansed_record.category = cleansed_record.category.strip().lower()
            cleansed_record.store_location = cleansed_record.store_location.strip().upper()
            
            # Normalize seasonal events
            if cleansed_record.seasonal_event:
                cleansed_record.seasonal_event = cleansed_record.seasonal_event.strip().lower()
            
            cleansed_data.append(cleansed_record)
        
        logger.info(f"Data cleansing complete for {len(cleansed_data)} records")
        return cleansed_data
    
    async def extract_basic_patterns(self, data: List[SalesDataPoint]) -> List[DemandPattern]:
        """
        Extract basic demand patterns from sales data.
        
        Args:
            data: List of sales data points to analyze
            
        Returns:
            List of identified demand patterns
        """
        logger.info(f"Extracting patterns from {len(data)} sales records")
        
        if not data:
            return []
        
        # Group data by product
        product_data = {}
        for record in data:
            if record.product_id not in product_data:
                product_data[record.product_id] = []
            product_data[record.product_id].append(record)
        
        patterns = []
        
        for product_id, records in product_data.items():
            if len(records) < 3:  # Need minimum data points
                continue
            
            # Sort by date
            records.sort(key=lambda x: x.sale_date)
            
            # Calculate basic statistics
            quantities = [r.quantity_sold for r in records]
            mean_quantity = np.mean(quantities)
            std_quantity = np.std(quantities)
            volatility = std_quantity / mean_quantity if mean_quantity > 0 else 1.0
            
            # Determine trend direction
            if len(records) >= 5:
                recent_avg = np.mean(quantities[-3:])
                older_avg = np.mean(quantities[:3])
                
                if recent_avg > older_avg * 1.1:
                    trend_direction = "increasing"
                elif recent_avg < older_avg * 0.9:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "stable"
            
            # Determine confidence based on data points and consistency
            if len(records) >= 10 and volatility < 0.5:
                confidence = ConfidenceLevel.HIGH
            elif len(records) >= 5 and volatility < 0.8:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            
            # Create demand pattern
            pattern = DemandPattern(
                product_id=product_id,
                pattern_type="basic_trend",
                description=f"Basic demand pattern for {records[0].product_name} showing {trend_direction} trend",
                confidence_level=confidence,
                trend_direction=trend_direction,
                volatility_score=min(volatility, 1.0),  # Cap at 1.0
                supporting_data_points=len(records),
                date_range_start=records[0].sale_date,
                date_range_end=records[-1].sale_date
            )
            
            patterns.append(pattern)
        
        logger.info(f"Extracted {len(patterns)} basic demand patterns")
        return patterns
    
    async def ingest_sales_data(self, data: List[SalesDataPoint]) -> Dict[str, Any]:
        """
        Ingest and validate sales data for processing.
        
        Args:
            data: List of sales data points to process
            
        Returns:
            Dictionary containing ingestion results and statistics
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            logger.info(f"Ingesting {len(data)} sales data points")
            
            # Validate data quality
            cleaned_data, quality_report = await self.validate_data_quality(data)
            
            # Cleanse data
            cleansed_data = await self.cleanse_data(cleaned_data)
            
            # Store processed data in memory
            self.processed_data.extend(cleansed_data)
            
            # Store in persistent storage if available
            storage_result = None
            if hasattr(self, 'storage_manager') and self.storage_manager:
                try:
                    storage_result = await self.storage_manager.store_sales_data(cleansed_data)
                    logger.info(f"Stored {storage_result['stored_count']} records in persistent storage")
                except Exception as e:
                    logger.error(f"Failed to store in persistent storage: {e}")
                    # Continue with in-memory storage
            
            # Generate ingestion report
            ingestion_result = {
                'status': 'success',
                'records_processed': len(data),
                'records_accepted': len(cleansed_data),
                'records_rejected': len(data) - len(cleansed_data),
                'quality_report': quality_report,
                'storage_result': storage_result,
                'ingestion_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Data ingestion complete. Accepted: {len(cleansed_data)}, Rejected: {len(data) - len(cleansed_data)}")
            return ingestion_result
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise ValueError(f"Data ingestion failed: {e}")
    
    async def extract_demand_patterns(self, product_ids: Optional[List[str]] = None) -> List[DemandPattern]:
        """
        Extract demand patterns from processed sales data.
        
        Args:
            product_ids: Optional list of product IDs to analyze (None for all)
            
        Returns:
            List of identified demand patterns
        """
        logger.info(f"Extracting demand patterns for {len(product_ids) if product_ids else 'all'} products")
        
        # Try to retrieve from persistent storage first
        if hasattr(self, 'storage_manager') and self.storage_manager:
            try:
                stored_patterns = await self.storage_manager.retrieve_patterns(
                    product_ids=product_ids,
                    use_cache=True
                )
                if stored_patterns:
                    logger.info(f"Retrieved {len(stored_patterns)} patterns from persistent storage")
                    return stored_patterns
            except Exception as e:
                logger.warning(f"Failed to retrieve from persistent storage, falling back to analysis: {e}")
        
        # Filter data if specific products requested
        if product_ids:
            filtered_data = [d for d in self.processed_data if d.product_id in product_ids]
        else:
            filtered_data = self.processed_data
        
        # If no data available, try to load from storage
        if not filtered_data and hasattr(self, 'storage_manager') and self.storage_manager:
            try:
                stored_sales_data = await self.storage_manager.retrieve_sales_data(
                    product_ids=product_ids,
                    limit=10000  # Reasonable limit for analysis
                )
                if stored_sales_data:
                    filtered_data = stored_sales_data
                    logger.info(f"Loaded {len(filtered_data)} sales records from storage for analysis")
            except Exception as e:
                logger.warning(f"Failed to load sales data from storage: {e}")
        
        # Extract patterns
        patterns = await self.extract_basic_patterns(filtered_data)
        
        # Cache patterns for future use
        cache_key = ','.join(sorted(product_ids)) if product_ids else 'all'
        self.patterns_cache[cache_key] = patterns
        
        return patterns
    
    async def correlate_seasonal_events(self, patterns: List[DemandPattern]) -> List[DemandPattern]:
        """
        Correlate demand patterns with seasonal events and festivals.
        
        Args:
            patterns: List of demand patterns to enhance with seasonal correlation
            
        Returns:
            Updated patterns with seasonal correlation data
        """
        logger.info(f"Correlating {len(patterns)} patterns with seasonal events")
        
        # Enhanced Indian festivals and seasons with more detailed calendar data
        seasonal_events = await self._get_seasonal_calendar()
        
        enhanced_patterns = []
        
        for pattern in patterns:
            # Get sales data for this product
            product_data = [d for d in self.processed_data if d.product_id == pattern.product_id]
            
            if not product_data:
                enhanced_patterns.append(pattern)
                continue
            
            # Calculate seasonal factors using advanced correlation analysis
            seasonal_factors = await self._calculate_seasonal_factors(product_data, seasonal_events)
            
            # Detect cyclical patterns
            cyclical_patterns = await self._detect_cyclical_patterns(product_data)
            
            # Calculate festival impact scores
            festival_impacts = await self._calculate_festival_impacts(product_data, seasonal_events)
            
            # Calculate category-specific seasonal correlations
            category_correlations = await self._calculate_category_seasonal_correlations(
                product_data, seasonal_events
            )
            
            # Detect pre-festival buying patterns
            pre_festival_patterns = await self._detect_pre_festival_patterns(
                product_data, seasonal_events
            )
            
            # Calculate weather correlation if applicable
            weather_correlations = await self._calculate_weather_correlations(product_data)
            
            # Update pattern with enhanced seasonal data
            enhanced_pattern = pattern.model_copy()
            enhanced_pattern.seasonal_factors = seasonal_factors
            
            # Add cyclical pattern information
            if cyclical_patterns:
                enhanced_pattern.pattern_type = f"{pattern.pattern_type}_cyclical"
                enhanced_pattern.description += f" with {len(cyclical_patterns)} cyclical patterns detected"
            
            # Update description with significant seasonal correlations
            significant_events = [
                event for event, factor in seasonal_factors.items() 
                if abs(factor - 1.0) > 0.2  # More than 20% deviation from baseline
            ]
            
            if significant_events:
                enhanced_pattern.description += f" with seasonal correlation to {', '.join(significant_events)}"
            
            # Add festival impact data as metadata
            if festival_impacts:
                enhanced_pattern.seasonal_factors.update({
                    f"{event}_impact": impact for event, impact in festival_impacts.items()
                })
            
            # Add category correlations
            if category_correlations:
                enhanced_pattern.seasonal_factors.update({
                    f"category_{key}": value for key, value in category_correlations.items()
                })
            
            # Add pre-festival patterns
            if pre_festival_patterns:
                enhanced_pattern.seasonal_factors.update({
                    f"pre_festival_{key}": value for key, value in pre_festival_patterns.items()
                })
            
            # Add weather correlations
            if weather_correlations:
                enhanced_pattern.seasonal_factors.update({
                    f"weather_{key}": value for key, value in weather_correlations.items()
                })
            
            enhanced_patterns.append(enhanced_pattern)
        
        logger.info(f"Seasonal correlation complete for {len(enhanced_patterns)} patterns")
        return enhanced_patterns
    
    async def _get_seasonal_calendar(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive seasonal calendar for Indian retail market.
        
        Returns:
            Dictionary of seasonal events with detailed information
        """
        return {
            # Major Hindu Festivals
            'diwali': {
                'months': [10, 11], 
                'boost_factor': 1.8,
                'categories': ['electronics', 'clothing', 'jewelry', 'sweets', 'gifts', 'home_decor'],
                'duration_days': 5,
                'pre_festival_boost': 30,  # days before festival
                'regional_variations': {
                    'north': 1.9,
                    'south': 1.7,
                    'west': 1.8,
                    'east': 1.6
                }
            },
            'holi': {
                'months': [3], 
                'boost_factor': 1.4,
                'categories': ['colors', 'sweets', 'clothing', 'beverages'],
                'duration_days': 2,
                'pre_festival_boost': 7,
                'regional_variations': {
                    'north': 1.6,
                    'south': 1.2,
                    'west': 1.4,
                    'east': 1.3
                }
            },
            'durga_puja': {
                'months': [9, 10], 
                'boost_factor': 1.6,
                'categories': ['clothing', 'jewelry', 'food', 'decorations'],
                'duration_days': 10,
                'pre_festival_boost': 15,
                'regional_variations': {
                    'north': 1.3,
                    'south': 1.2,
                    'west': 1.4,
                    'east': 2.0  # Highest in Bengal
                }
            },
            'ganesh_chaturthi': {
                'months': [8, 9], 
                'boost_factor': 1.5,
                'categories': ['sweets', 'decorations', 'electronics', 'flowers'],
                'duration_days': 11,
                'pre_festival_boost': 10,
                'regional_variations': {
                    'north': 1.2,
                    'south': 1.4,
                    'west': 1.8,  # Highest in Maharashtra
                    'east': 1.3
                }
            },
            'navratri': {
                'months': [9, 10], 
                'boost_factor': 1.4,
                'categories': ['clothing', 'jewelry', 'accessories', 'food'],
                'duration_days': 9,
                'pre_festival_boost': 15,
                'regional_variations': {
                    'north': 1.5,
                    'south': 1.2,
                    'west': 1.6,  # High in Gujarat
                    'east': 1.3
                }
            },
            
            # Islamic Festivals
            'eid_ul_fitr': {
                'months': [4, 5, 6],  # Variable based on lunar calendar
                'boost_factor': 1.6,
                'categories': ['clothing', 'food', 'gifts', 'sweets'],
                'duration_days': 3,
                'pre_festival_boost': 15,
                'regional_variations': {
                    'north': 1.7,
                    'south': 1.5,
                    'west': 1.6,
                    'east': 1.4
                }
            },
            'eid_ul_adha': {
                'months': [6, 7, 8],  # Variable based on lunar calendar
                'boost_factor': 1.4,
                'categories': ['clothing', 'food', 'meat'],
                'duration_days': 3,
                'pre_festival_boost': 10,
                'regional_variations': {
                    'north': 1.5,
                    'south': 1.3,
                    'west': 1.4,
                    'east': 1.2
                }
            },
            
            # Christian Festivals
            'christmas': {
                'months': [12], 
                'boost_factor': 1.3,
                'categories': ['gifts', 'food', 'decorations', 'clothing'],
                'duration_days': 1,
                'pre_festival_boost': 20,
                'regional_variations': {
                    'north': 1.2,
                    'south': 1.5,  # Higher in Kerala, Goa
                    'west': 1.3,
                    'east': 1.4
                }
            },
            
            # Seasonal Patterns
            'summer': {
                'months': [4, 5, 6], 
                'boost_factor': 0.85,
                'categories': ['cooling_appliances', 'summer_clothing', 'beverages', 'ice_cream'],
                'duration_days': 90,
                'pre_festival_boost': 0,
                'impact_categories': {
                    'cooling_appliances': 2.5,
                    'summer_clothing': 1.8,
                    'beverages': 1.6,
                    'ice_cream': 2.0
                }
            },
            'monsoon': {
                'months': [7, 8, 9], 
                'boost_factor': 0.75,
                'categories': ['umbrellas', 'rainwear', 'indoor_entertainment', 'hot_beverages'],
                'duration_days': 90,
                'pre_festival_boost': 0,
                'impact_categories': {
                    'umbrellas': 3.0,
                    'rainwear': 2.5,
                    'indoor_entertainment': 1.4,
                    'hot_beverages': 1.3
                }
            },
            'winter': {
                'months': [12, 1, 2], 
                'boost_factor': 1.1,
                'categories': ['winter_clothing', 'heaters', 'blankets', 'hot_beverages'],
                'duration_days': 90,
                'pre_festival_boost': 0,
                'impact_categories': {
                    'winter_clothing': 2.2,
                    'heaters': 2.8,
                    'blankets': 2.0,
                    'hot_beverages': 1.5
                }
            },
            
            # Shopping Seasons
            'wedding_season': {
                'months': [11, 12, 1, 2], 
                'boost_factor': 1.7,
                'categories': ['jewelry', 'clothing', 'gifts', 'electronics', 'home_appliances'],
                'duration_days': 120,
                'pre_festival_boost': 30,
                'impact_categories': {
                    'jewelry': 3.0,
                    'clothing': 2.5,
                    'gifts': 2.0,
                    'electronics': 1.8,
                    'home_appliances': 2.2
                }
            },
            'back_to_school': {
                'months': [6, 7], 
                'boost_factor': 1.4,
                'categories': ['stationery', 'bags', 'uniforms', 'electronics', 'books'],
                'duration_days': 60,
                'pre_festival_boost': 15,
                'impact_categories': {
                    'stationery': 2.5,
                    'bags': 2.8,
                    'uniforms': 3.0,
                    'electronics': 1.6,
                    'books': 2.2
                }
            },
            'harvest_season': {
                'months': [10, 11, 4, 5],  # Kharif and Rabi harvest
                'boost_factor': 1.3,
                'categories': ['agricultural_tools', 'food_grains', 'rural_goods'],
                'duration_days': 60,
                'pre_festival_boost': 10,
                'regional_variations': {
                    'north': 1.5,  # Major agricultural region
                    'south': 1.4,
                    'west': 1.2,
                    'east': 1.3
                }
            }
        }
    
    async def _calculate_seasonal_factors(self, product_data: List[SalesDataPoint], 
                                        seasonal_events: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate seasonal factors using advanced statistical analysis.
        
        Args:
            product_data: Sales data for a specific product
            seasonal_events: Seasonal event definitions
            
        Returns:
            Dictionary of seasonal factors
        """
        seasonal_factors = {}
        
        if not product_data:
            return seasonal_factors
        
        # Calculate baseline (non-seasonal) average
        baseline_sales = []
        seasonal_sales = {}
        
        for event_name, event_info in seasonal_events.items():
            seasonal_sales[event_name] = []
        
        # Categorize sales data
        for record in product_data:
            month = record.sale_date.month
            is_seasonal = False
            
            # Check if this record falls in any seasonal period
            for event_name, event_info in seasonal_events.items():
                if month in event_info['months']:
                    seasonal_sales[event_name].append(record.quantity_sold)
                    is_seasonal = True
            
            if not is_seasonal:
                baseline_sales.append(record.quantity_sold)
        
        # Calculate baseline average
        baseline_avg = np.mean(baseline_sales) if baseline_sales else 1.0
        
        # Calculate seasonal factors with confidence intervals
        for event_name, sales_data in seasonal_sales.items():
            if sales_data and baseline_avg > 0:
                seasonal_avg = np.mean(sales_data)
                seasonal_factor = seasonal_avg / baseline_avg
                
                # Calculate confidence based on data points and consistency
                if len(sales_data) >= 3:
                    std_dev = np.std(sales_data)
                    coefficient_of_variation = std_dev / seasonal_avg if seasonal_avg > 0 else 1.0
                    
                    # Adjust factor based on consistency (lower CV = higher confidence)
                    confidence_multiplier = max(0.5, 1 - coefficient_of_variation)
                    seasonal_factors[event_name] = seasonal_factor * confidence_multiplier + (1 - confidence_multiplier)
                    
                    # Add confidence score
                    seasonal_factors[f"{event_name}_confidence"] = confidence_multiplier
                else:
                    seasonal_factors[event_name] = seasonal_factor
                    seasonal_factors[f"{event_name}_confidence"] = 0.3  # Low confidence
                
                # Add statistical measures
                if len(sales_data) > 1:
                    seasonal_factors[f"{event_name}_volatility"] = std_dev / seasonal_avg if seasonal_avg > 0 else 1.0
                    seasonal_factors[f"{event_name}_data_points"] = len(sales_data)
            else:
                seasonal_factors[event_name] = 1.0
                seasonal_factors[f"{event_name}_confidence"] = 0.0
        
        # Calculate year-over-year growth if sufficient data
        if len(product_data) >= 24:  # At least 2 years of monthly data
            yoy_factors = await self._calculate_year_over_year_factors(product_data, seasonal_events)
            seasonal_factors.update(yoy_factors)
        
        return seasonal_factors
    
    async def _calculate_year_over_year_factors(self, product_data: List[SalesDataPoint], 
                                             seasonal_events: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate year-over-year seasonal factors.
        
        Args:
            product_data: Sales data for analysis
            seasonal_events: Seasonal event definitions
            
        Returns:
            Dictionary of year-over-year factors
        """
        yoy_factors = {}
        
        # Group data by year and month
        yearly_data = {}
        for record in product_data:
            year = record.sale_date.year
            month = record.sale_date.month
            
            if year not in yearly_data:
                yearly_data[year] = {}
            if month not in yearly_data[year]:
                yearly_data[year][month] = []
            
            yearly_data[year][month].append(record.quantity_sold)
        
        # Calculate year-over-year growth for seasonal events
        for event_name, event_info in seasonal_events.items():
            event_months = event_info['months']
            yearly_event_sales = {}
            
            for year, monthly_data in yearly_data.items():
                event_sales = []
                for month in event_months:
                    if month in monthly_data:
                        event_sales.extend(monthly_data[month])
                
                if event_sales:
                    yearly_event_sales[year] = np.mean(event_sales)
            
            # Calculate year-over-year growth
            if len(yearly_event_sales) >= 2:
                years = sorted(yearly_event_sales.keys())
                growth_rates = []
                
                for i in range(1, len(years)):
                    prev_year_sales = yearly_event_sales[years[i-1]]
                    curr_year_sales = yearly_event_sales[years[i]]
                    
                    if prev_year_sales > 0:
                        growth_rate = (curr_year_sales - prev_year_sales) / prev_year_sales
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    avg_growth = np.mean(growth_rates)
                    yoy_factors[f"{event_name}_yoy_growth"] = avg_growth
                    yoy_factors[f"{event_name}_growth_consistency"] = 1 - np.std(growth_rates) if len(growth_rates) > 1 else 1.0
        
        return yoy_factors
    
    async def _detect_cyclical_patterns(self, product_data: List[SalesDataPoint]) -> List[Dict[str, Any]]:
        """
        Detect cyclical patterns in sales data using advanced time series analysis.
        
        Args:
            product_data: Sales data for analysis
            
        Returns:
            List of detected cyclical patterns
        """
        if len(product_data) < 12:  # Need at least a year of data
            return []
        
        # Sort data by date
        sorted_data = sorted(product_data, key=lambda x: x.sale_date)
        
        patterns = []
        
        # Monthly cycle analysis
        monthly_patterns = await self._detect_monthly_cycles(sorted_data)
        patterns.extend(monthly_patterns)
        
        # Quarterly cycle analysis
        quarterly_patterns = await self._detect_quarterly_cycles(sorted_data)
        patterns.extend(quarterly_patterns)
        
        # Weekly cycle analysis (if sufficient daily data)
        if len(sorted_data) >= 52:  # At least a year of weekly data
            weekly_patterns = await self._detect_weekly_cycles(sorted_data)
            patterns.extend(weekly_patterns)
        
        # Seasonal decomposition
        if len(sorted_data) >= 24:  # At least 2 years for trend analysis
            trend_patterns = await self._detect_trend_cycles(sorted_data)
            patterns.extend(trend_patterns)
        
        return patterns
    
    async def _detect_monthly_cycles(self, sorted_data: List[SalesDataPoint]) -> List[Dict[str, Any]]:
        """Detect monthly cyclical patterns."""
        monthly_sales = {}
        for record in sorted_data:
            month = record.sale_date.month
            if month not in monthly_sales:
                monthly_sales[month] = []
            monthly_sales[month].append(record.quantity_sold)
        
        # Calculate monthly averages
        monthly_averages = {}
        for month, sales in monthly_sales.items():
            monthly_averages[month] = np.mean(sales)
        
        patterns = []
        
        if len(monthly_averages) >= 6:  # Need at least 6 months
            # Find peak and trough months
            max_month = max(monthly_averages, key=monthly_averages.get)
            min_month = min(monthly_averages, key=monthly_averages.get)
            
            amplitude = monthly_averages[max_month] / monthly_averages[min_month]
            
            if amplitude > 1.3:  # Significant monthly variation
                patterns.append({
                    'type': 'monthly',
                    'peak_month': max_month,
                    'trough_month': min_month,
                    'amplitude': amplitude,
                    'consistency': self._calculate_pattern_consistency(monthly_averages)
                })
        
        return patterns
    
    async def _detect_quarterly_cycles(self, sorted_data: List[SalesDataPoint]) -> List[Dict[str, Any]]:
        """Detect quarterly cyclical patterns."""
        quarters = {
            'Q1': [1, 2, 3],
            'Q2': [4, 5, 6], 
            'Q3': [7, 8, 9],
            'Q4': [10, 11, 12]
        }
        
        quarterly_sales = {q: [] for q in quarters.keys()}
        
        for record in sorted_data:
            month = record.sale_date.month
            for quarter, months in quarters.items():
                if month in months:
                    quarterly_sales[quarter].append(record.quantity_sold)
                    break
        
        quarterly_averages = {}
        for quarter, sales in quarterly_sales.items():
            if sales:
                quarterly_averages[quarter] = np.mean(sales)
        
        patterns = []
        
        if len(quarterly_averages) >= 3:  # Need at least 3 quarters
            max_quarter = max(quarterly_averages, key=quarterly_averages.get)
            min_quarter = min(quarterly_averages, key=quarterly_averages.get)
            
            amplitude = quarterly_averages[max_quarter] / quarterly_averages[min_quarter]
            
            if amplitude > 1.3:
                patterns.append({
                    'type': 'quarterly',
                    'peak_quarter': max_quarter,
                    'low_quarter': min_quarter,
                    'amplitude': amplitude,
                    'consistency': self._calculate_pattern_consistency(quarterly_averages)
                })
        
        return patterns
    
    async def _detect_weekly_cycles(self, sorted_data: List[SalesDataPoint]) -> List[Dict[str, Any]]:
        """Detect weekly cyclical patterns."""
        weekly_sales = {i: [] for i in range(7)}  # 0=Monday, 6=Sunday
        
        for record in sorted_data:
            weekday = record.sale_date.weekday()
            weekly_sales[weekday].append(record.quantity_sold)
        
        weekly_averages = {}
        for day, sales in weekly_sales.items():
            if sales:
                weekly_averages[day] = np.mean(sales)
        
        patterns = []
        
        if len(weekly_averages) >= 5:  # Need at least 5 days of week
            max_day = max(weekly_averages, key=weekly_averages.get)
            min_day = min(weekly_averages, key=weekly_averages.get)
            
            amplitude = weekly_averages[max_day] / weekly_averages[min_day]
            
            if amplitude > 1.2:  # Weekly patterns can be more subtle
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                patterns.append({
                    'type': 'weekly',
                    'peak_day': day_names[max_day],
                    'low_day': day_names[min_day],
                    'amplitude': amplitude,
                    'consistency': self._calculate_pattern_consistency(weekly_averages)
                })
        
        return patterns
    
    async def _detect_trend_cycles(self, sorted_data: List[SalesDataPoint]) -> List[Dict[str, Any]]:
        """Detect long-term trend cycles."""
        # Group data by year-month for trend analysis
        monthly_data = {}
        for record in sorted_data:
            year_month = f"{record.sale_date.year}-{record.sale_date.month:02d}"
            if year_month not in monthly_data:
                monthly_data[year_month] = []
            monthly_data[year_month].append(record.quantity_sold)
        
        # Calculate monthly totals
        monthly_totals = {}
        for year_month, sales in monthly_data.items():
            monthly_totals[year_month] = sum(sales)
        
        patterns = []
        
        if len(monthly_totals) >= 12:  # At least a year of monthly data
            # Sort by date
            sorted_months = sorted(monthly_totals.keys())
            values = [monthly_totals[month] for month in sorted_months]
            
            # Simple trend detection using linear regression
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            trend_slope = z[0]
            
            # Calculate trend strength
            trend_strength = abs(trend_slope) / np.mean(values) if np.mean(values) > 0 else 0
            
            if trend_strength > 0.05:  # 5% trend strength threshold
                patterns.append({
                    'type': 'trend',
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'strength': trend_strength,
                    'slope': trend_slope,
                    'r_squared': self._calculate_r_squared(x, values, z)
                })
        
        return patterns
    
    def _calculate_pattern_consistency(self, data_dict: Dict[Any, float]) -> float:
        """Calculate consistency score for a pattern."""
        if len(data_dict) < 2:
            return 0.0
        
        values = list(data_dict.values())
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Consistency is inverse of coefficient of variation
        if mean_val > 0:
            cv = std_val / mean_val
            consistency = max(0, 1 - cv)
        else:
            consistency = 0.0
        
        return consistency
    
    def _calculate_r_squared(self, x: np.ndarray, y: List[float], coeffs: np.ndarray) -> float:
        """Calculate R-squared for linear regression."""
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, r_squared)
    
    async def _calculate_festival_impacts(self, product_data: List[SalesDataPoint], 
                                        seasonal_events: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate specific festival impact scores.
        
        Args:
            product_data: Sales data for analysis
            seasonal_events: Seasonal event definitions
            
        Returns:
            Dictionary of festival impact scores
        """
        festival_impacts = {}
        
        for event_name, event_info in seasonal_events.items():
            if 'pre_festival_boost' not in event_info or event_info['pre_festival_boost'] == 0:
                continue
            
            # Find sales during pre-festival period
            pre_festival_sales = []
            normal_sales = []
            
            for record in product_data:
                month = record.sale_date.month
                
                # Check if in festival months
                if month in event_info['months']:
                    pre_festival_sales.append(record.quantity_sold)
                else:
                    normal_sales.append(record.quantity_sold)
            
            if pre_festival_sales and normal_sales:
                pre_festival_avg = np.mean(pre_festival_sales)
                normal_avg = np.mean(normal_sales)
                
                if normal_avg > 0:
                    impact_score = (pre_festival_avg - normal_avg) / normal_avg
                    festival_impacts[event_name] = impact_score
        
        return festival_impacts
    
    async def _calculate_category_seasonal_correlations(self, product_data: List[SalesDataPoint], 
                                                      seasonal_events: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate category-specific seasonal correlations.
        
        Args:
            product_data: Sales data for analysis
            seasonal_events: Seasonal event definitions
            
        Returns:
            Dictionary of category-specific seasonal correlations
        """
        if not product_data:
            return {}
        
        category = product_data[0].category.lower()
        correlations = {}
        
        # Calculate baseline sales for the category
        baseline_sales = np.mean([record.quantity_sold for record in product_data])
        
        for event_name, event_info in seasonal_events.items():
            # Check if this category is relevant for this event
            relevant_categories = event_info.get('categories', [])
            
            if category in relevant_categories or any(cat in category for cat in relevant_categories):
                # Calculate sales during event months
                event_sales = []
                for record in product_data:
                    if record.sale_date.month in event_info['months']:
                        event_sales.append(record.quantity_sold)
                
                if event_sales and baseline_sales > 0:
                    event_avg = np.mean(event_sales)
                    correlation_strength = event_avg / baseline_sales
                    
                    # Apply expected boost factor to determine correlation quality
                    expected_boost = event_info.get('boost_factor', 1.0)
                    correlation_quality = min(correlation_strength / expected_boost, 2.0)  # Cap at 2.0
                    
                    correlations[f"{event_name}_correlation"] = correlation_quality
        
        return correlations
    
    async def _detect_pre_festival_patterns(self, product_data: List[SalesDataPoint], 
                                          seasonal_events: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Detect pre-festival buying patterns.
        
        Args:
            product_data: Sales data for analysis
            seasonal_events: Seasonal event definitions
            
        Returns:
            Dictionary of pre-festival pattern indicators
        """
        pre_festival_patterns = {}
        
        for event_name, event_info in seasonal_events.items():
            pre_festival_days = event_info.get('pre_festival_boost', 0)
            if pre_festival_days == 0:
                continue
            
            # Group sales by proximity to festival months
            pre_festival_sales = []
            regular_sales = []
            
            for record in product_data:
                month = record.sale_date.month
                
                # Check if this is a pre-festival month (month before festival months)
                festival_months = event_info['months']
                pre_festival_months = [(m - 1) if m > 1 else 12 for m in festival_months]
                
                if month in pre_festival_months:
                    pre_festival_sales.append(record.quantity_sold)
                elif month not in festival_months:  # Regular months (not festival or pre-festival)
                    regular_sales.append(record.quantity_sold)
            
            if pre_festival_sales and regular_sales:
                pre_festival_avg = np.mean(pre_festival_sales)
                regular_avg = np.mean(regular_sales)
                
                if regular_avg > 0:
                    # Calculate pre-festival boost ratio
                    boost_ratio = pre_festival_avg / regular_avg
                    pre_festival_patterns[f"{event_name}_pre_boost"] = boost_ratio
                    
                    # Calculate consistency of pre-festival pattern
                    if len(pre_festival_sales) > 1:
                        consistency = 1 - (np.std(pre_festival_sales) / pre_festival_avg)
                        pre_festival_patterns[f"{event_name}_consistency"] = max(consistency, 0)
        
        return pre_festival_patterns
    
    async def _calculate_weather_correlations(self, product_data: List[SalesDataPoint]) -> Dict[str, float]:
        """
        Calculate weather-related seasonal correlations.
        
        Args:
            product_data: Sales data for analysis
            
        Returns:
            Dictionary of weather correlation indicators
        """
        if not product_data:
            return {}
        
        weather_correlations = {}
        
        # Group sales by season based on months
        seasonal_sales = {
            'summer': [],      # April-June
            'monsoon': [],     # July-September  
            'winter': [],      # December-February
            'spring': []       # March, October-November
        }
        
        for record in product_data:
            month = record.sale_date.month
            
            if month in [4, 5, 6]:
                seasonal_sales['summer'].append(record.quantity_sold)
            elif month in [7, 8, 9]:
                seasonal_sales['monsoon'].append(record.quantity_sold)
            elif month in [12, 1, 2]:
                seasonal_sales['winter'].append(record.quantity_sold)
            else:  # March, October, November
                seasonal_sales['spring'].append(record.quantity_sold)
        
        # Calculate seasonal averages and correlations
        all_sales = [record.quantity_sold for record in product_data]
        overall_avg = np.mean(all_sales) if all_sales else 1
        
        for season, sales in seasonal_sales.items():
            if sales and overall_avg > 0:
                seasonal_avg = np.mean(sales)
                correlation = seasonal_avg / overall_avg
                weather_correlations[f"{season}_factor"] = correlation
                
                # Calculate seasonal volatility
                if len(sales) > 1:
                    volatility = np.std(sales) / seasonal_avg if seasonal_avg > 0 else 1
                    weather_correlations[f"{season}_volatility"] = min(volatility, 2.0)
        
        return weather_correlations
    
    async def integrate_market_signals(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate external market signals into analysis.
        
        Args:
            external_data: Dictionary containing external market data
            
        Returns:
            Integration results and updated analysis
        """
        logger.info("Integrating external market signals")
        
        # Validate external data sources (must be synthetic or public)
        allowed_sources = ['synthetic', 'public_api', 'government_data', 'industry_reports', 'economic_indicators']
        
        integration_result = {
            'status': 'success',
            'signals_processed': 0,
            'signals_integrated': 0,
            'warnings': [],
            'market_insights': {},
            'correlation_scores': {}
        }
        
        for source, data in external_data.items():
            if source not in allowed_sources:
                integration_result['warnings'].append(f"Skipping unauthorized data source: {source}")
                continue
            
            integration_result['signals_processed'] += 1
            
            # Process different types of market signals
            if 'price_trends' in data:
                price_insights = await self._process_price_trends(data['price_trends'])
                integration_result['market_insights']['price_trends'] = price_insights
                integration_result['signals_integrated'] += 1
                logger.info(f"Processing price trends from {source}")
            
            if 'demand_indicators' in data:
                demand_insights = await self._process_demand_indicators(data['demand_indicators'])
                integration_result['market_insights']['demand_indicators'] = demand_insights
                integration_result['signals_integrated'] += 1
                logger.info(f"Processing demand indicators from {source}")
            
            if 'seasonal_adjustments' in data:
                seasonal_insights = await self._process_seasonal_adjustments(data['seasonal_adjustments'])
                integration_result['market_insights']['seasonal_adjustments'] = seasonal_insights
                integration_result['signals_integrated'] += 1
                logger.info(f"Processing seasonal adjustments from {source}")
            
            if 'economic_indicators' in data:
                economic_insights = await self._process_economic_indicators(data['economic_indicators'])
                integration_result['market_insights']['economic_indicators'] = economic_insights
                integration_result['signals_integrated'] += 1
                logger.info(f"Processing economic indicators from {source}")
            
            if 'competitor_data' in data:
                competitor_insights = await self._process_competitor_data(data['competitor_data'])
                integration_result['market_insights']['competitor_data'] = competitor_insights
                integration_result['signals_integrated'] += 1
                logger.info(f"Processing competitor data from {source}")
        
        # Calculate correlation scores between market signals and internal data
        if integration_result['market_insights']:
            correlation_scores = await self._calculate_market_correlations(integration_result['market_insights'])
            integration_result['correlation_scores'] = correlation_scores
        
        logger.info(f"Market signal integration complete. Processed: {integration_result['signals_integrated']}")
        return integration_result
    
    async def _process_price_trends(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process price trend market signals.
        
        Args:
            price_data: Price trend data from external source
            
        Returns:
            Processed price trend insights
        """
        insights = {
            'category_trends': {},
            'inflation_impact': 0.0,
            'price_volatility': {}
        }
        
        # Process category-wise price trends
        if 'categories' in price_data:
            for category, trend_data in price_data['categories'].items():
                if isinstance(trend_data, dict):
                    insights['category_trends'][category] = {
                        'direction': trend_data.get('direction', 'stable'),
                        'magnitude': trend_data.get('magnitude', 0.0),
                        'confidence': trend_data.get('confidence', 0.5)
                    }
                else:
                    # Simple string trend
                    insights['category_trends'][category] = {
                        'direction': trend_data,
                        'magnitude': 0.1 if trend_data in ['increasing', 'decreasing'] else 0.0,
                        'confidence': 0.6
                    }
        
        # Process inflation impact
        if 'inflation_rate' in price_data:
            insights['inflation_impact'] = float(price_data['inflation_rate'])
        
        # Process price volatility
        if 'volatility' in price_data:
            insights['price_volatility'] = price_data['volatility']
        
        return insights
    
    async def _process_demand_indicators(self, demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process demand indicator market signals.
        
        Args:
            demand_data: Demand indicator data from external source
            
        Returns:
            Processed demand insights
        """
        insights = {
            'overall_demand': 'stable',
            'category_demand': {},
            'seasonal_multipliers': {},
            'consumer_sentiment': 0.5
        }
        
        # Process overall demand trend
        if 'overall_trend' in demand_data:
            insights['overall_demand'] = demand_data['overall_trend']
        
        # Process category-specific demand
        if 'categories' in demand_data:
            insights['category_demand'] = demand_data['categories']
        
        # Process seasonal multipliers
        if 'seasonal_boost' in demand_data:
            insights['seasonal_multipliers']['general'] = float(demand_data['seasonal_boost'])
        
        if 'seasonal_factors' in demand_data:
            insights['seasonal_multipliers'].update(demand_data['seasonal_factors'])
        
        # Process consumer sentiment
        if 'consumer_sentiment' in demand_data:
            insights['consumer_sentiment'] = float(demand_data['consumer_sentiment'])
        
        return insights
    
    async def _process_seasonal_adjustments(self, seasonal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process seasonal adjustment market signals.
        
        Args:
            seasonal_data: Seasonal adjustment data from external source
            
        Returns:
            Processed seasonal insights
        """
        insights = {
            'festival_adjustments': {},
            'weather_impact': {},
            'calendar_effects': {}
        }
        
        # Process festival adjustments
        if 'festivals' in seasonal_data:
            insights['festival_adjustments'] = seasonal_data['festivals']
        
        # Process weather impact
        if 'weather' in seasonal_data:
            insights['weather_impact'] = seasonal_data['weather']
        
        # Process calendar effects (holidays, weekends, etc.)
        if 'calendar' in seasonal_data:
            insights['calendar_effects'] = seasonal_data['calendar']
        
        return insights
    
    async def _process_economic_indicators(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process economic indicator market signals.
        
        Args:
            economic_data: Economic indicator data from external source
            
        Returns:
            Processed economic insights
        """
        insights = {
            'gdp_growth': 0.0,
            'unemployment_rate': 0.0,
            'consumer_confidence': 0.5,
            'retail_index': 1.0,
            'regional_factors': {}
        }
        
        # Process GDP growth
        if 'gdp_growth' in economic_data:
            insights['gdp_growth'] = float(economic_data['gdp_growth'])
        
        # Process unemployment rate
        if 'unemployment' in economic_data:
            insights['unemployment_rate'] = float(economic_data['unemployment'])
        
        # Process consumer confidence
        if 'consumer_confidence' in economic_data:
            insights['consumer_confidence'] = float(economic_data['consumer_confidence'])
        
        # Process retail index
        if 'retail_index' in economic_data:
            insights['retail_index'] = float(economic_data['retail_index'])
        
        # Process regional factors
        if 'regional' in economic_data:
            insights['regional_factors'] = economic_data['regional']
        
        return insights
    
    async def _process_competitor_data(self, competitor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process competitor market signals.
        
        Args:
            competitor_data: Competitor data from external source
            
        Returns:
            Processed competitor insights
        """
        insights = {
            'pricing_strategies': {},
            'promotion_patterns': {},
            'market_share_trends': {},
            'competitive_pressure': 0.5
        }
        
        # Process pricing strategies
        if 'pricing' in competitor_data:
            insights['pricing_strategies'] = competitor_data['pricing']
        
        # Process promotion patterns
        if 'promotions' in competitor_data:
            insights['promotion_patterns'] = competitor_data['promotions']
        
        # Process market share trends
        if 'market_share' in competitor_data:
            insights['market_share_trends'] = competitor_data['market_share']
        
        # Process competitive pressure
        if 'competitive_pressure' in competitor_data:
            insights['competitive_pressure'] = float(competitor_data['competitive_pressure'])
        
        return insights
    
    async def _calculate_market_correlations(self, market_insights: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate correlation scores between market signals and internal sales data.
        
        Args:
            market_insights: Processed market insights
            
        Returns:
            Dictionary of correlation scores
        """
        correlations = {}
        
        if not self.processed_data:
            return correlations
        
        # Calculate correlation with price trends
        if 'price_trends' in market_insights:
            price_correlation = await self._correlate_with_price_trends(market_insights['price_trends'])
            correlations['price_trends'] = price_correlation
        
        # Calculate correlation with demand indicators
        if 'demand_indicators' in market_insights:
            demand_correlation = await self._correlate_with_demand_indicators(market_insights['demand_indicators'])
            correlations['demand_indicators'] = demand_correlation
        
        # Calculate correlation with economic indicators
        if 'economic_indicators' in market_insights:
            economic_correlation = await self._correlate_with_economic_indicators(market_insights['economic_indicators'])
            correlations['economic_indicators'] = economic_correlation
        
        return correlations
    
    async def _correlate_with_price_trends(self, price_insights: Dict[str, Any]) -> float:
        """Calculate correlation between price trends and sales data."""
        # Group sales by category
        category_sales = {}
        for record in self.processed_data:
            category = record.category
            if category not in category_sales:
                category_sales[category] = []
            category_sales[category].append(record.quantity_sold)
        
        # Calculate correlation score
        correlation_score = 0.0
        matching_categories = 0
        
        for category, sales_data in category_sales.items():
            if category in price_insights.get('category_trends', {}):
                trend_data = price_insights['category_trends'][category]
                
                # Simple correlation: increasing prices should correlate with decreasing sales
                avg_sales = np.mean(sales_data)
                if trend_data['direction'] == 'increasing' and avg_sales < np.median(list(category_sales.values())):
                    correlation_score += 0.7
                elif trend_data['direction'] == 'decreasing' and avg_sales > np.median(list(category_sales.values())):
                    correlation_score += 0.7
                else:
                    correlation_score += 0.3
                
                matching_categories += 1
        
        return correlation_score / matching_categories if matching_categories > 0 else 0.5
    
    async def _correlate_with_demand_indicators(self, demand_insights: Dict[str, Any]) -> float:
        """Calculate correlation between demand indicators and sales data."""
        # Calculate overall sales trend
        if len(self.processed_data) < 2:
            return 0.5
        
        sorted_data = sorted(self.processed_data, key=lambda x: x.sale_date)
        recent_sales = np.mean([d.quantity_sold for d in sorted_data[-len(sorted_data)//3:]])
        older_sales = np.mean([d.quantity_sold for d in sorted_data[:len(sorted_data)//3]])
        
        sales_trend = 'increasing' if recent_sales > older_sales * 1.1 else 'decreasing' if recent_sales < older_sales * 0.9 else 'stable'
        
        # Compare with demand indicators
        overall_demand = demand_insights.get('overall_demand', 'stable')
        
        if sales_trend == overall_demand:
            return 0.8
        elif (sales_trend in ['increasing', 'decreasing'] and overall_demand in ['increasing', 'decreasing']):
            return 0.6
        else:
            return 0.4
    
    async def _correlate_with_economic_indicators(self, economic_insights: Dict[str, Any]) -> float:
        """Calculate correlation between economic indicators and sales data."""
        correlation_score = 0.5  # Base score
        
        # Higher consumer confidence should correlate with higher sales
        consumer_confidence = economic_insights.get('consumer_confidence', 0.5)
        
        if self.processed_data:
            avg_sales = np.mean([d.quantity_sold for d in self.processed_data])
            # Normalize sales to 0-1 scale (rough approximation)
            normalized_sales = min(avg_sales / 100, 1.0)  # Assuming 100 is high sales
            
            # Calculate correlation
            confidence_correlation = 1 - abs(consumer_confidence - normalized_sales)
            correlation_score = (correlation_score + confidence_correlation) / 2
        
        return correlation_score
    
    async def store_patterns(self, patterns: List[DemandPattern]) -> bool:
        """
        Store analyzed patterns for future reference.
        
        Args:
            patterns: List of patterns to store
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            logger.info(f"Storing {len(patterns)} demand patterns")
            
            # Use storage manager if available, otherwise fall back to memory cache
            if hasattr(self, 'storage_manager') and self.storage_manager:
                result = await self.storage_manager.store_patterns(patterns)
                success = result['status'] in ['success', 'partial_success']
                
                if success:
                    logger.info(f"Successfully stored {result['stored_count']} patterns via storage manager")
                else:
                    logger.error(f"Storage manager failed to store patterns: {result.get('errors', [])}")
                
                return success
            else:
                # Fallback to memory cache
                timestamp = datetime.utcnow().isoformat()
                
                for pattern in patterns:
                    cache_key = f"{pattern.product_id}_{timestamp}"
                    self.patterns_cache[cache_key] = [pattern]
                
                logger.info(f"Successfully stored {len(patterns)} patterns in memory cache")
                return True
            
        except Exception as e:
            logger.error(f"Failed to store patterns: {e}")
            return False
    
    async def generate_seasonal_analysis_report(self, product_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive seasonal analysis report.
        
        Args:
            product_ids: Optional list of product IDs to analyze (None for all)
            
        Returns:
            Comprehensive seasonal analysis report
        """
        logger.info("Generating comprehensive seasonal analysis report")
        
        # Filter data if specific products requested
        if product_ids:
            filtered_data = [d for d in self.processed_data if d.product_id in product_ids]
        else:
            filtered_data = self.processed_data
        
        if not filtered_data:
            return {
                'status': 'no_data',
                'message': 'No sales data available for analysis'
            }
        
        # Get seasonal calendar
        seasonal_events = await self._get_seasonal_calendar()
        
        # Extract basic patterns
        basic_patterns = await self.extract_basic_patterns(filtered_data)
        
        # Enhance with seasonal correlation
        enhanced_patterns = await self.correlate_seasonal_events(basic_patterns)
        
        # Generate report
        report = {
            'status': 'success',
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'data_period': {
                'start_date': min(d.sale_date for d in filtered_data).isoformat(),
                'end_date': max(d.sale_date for d in filtered_data).isoformat(),
                'total_records': len(filtered_data)
            },
            'seasonal_insights': {},
            'festival_correlations': {},
            'cyclical_patterns': {},
            'category_analysis': {},
            'recommendations': []
        }
        
        # Analyze seasonal insights
        for event_name, event_info in seasonal_events.items():
            event_sales = []
            for record in filtered_data:
                if record.sale_date.month in event_info['months']:
                    event_sales.append(record.quantity_sold)
            
            if event_sales:
                baseline_sales = [r.quantity_sold for r in filtered_data 
                                if r.sale_date.month not in event_info['months']]
                baseline_avg = np.mean(baseline_sales) if baseline_sales else 1
                
                report['seasonal_insights'][event_name] = {
                    'average_sales': np.mean(event_sales),
                    'boost_factor': np.mean(event_sales) / baseline_avg if baseline_avg > 0 else 1,
                    'expected_boost': event_info['boost_factor'],
                    'performance_ratio': (np.mean(event_sales) / baseline_avg) / event_info['boost_factor'] if baseline_avg > 0 and event_info['boost_factor'] > 0 else 1,
                    'volatility': np.std(event_sales) / np.mean(event_sales) if np.mean(event_sales) > 0 else 0,
                    'data_points': len(event_sales)
                }
        
        # Analyze festival correlations
        for pattern in enhanced_patterns:
            if pattern.seasonal_factors:
                product_correlations = {}
                for factor_name, factor_value in pattern.seasonal_factors.items():
                    if not factor_name.endswith('_confidence') and not factor_name.endswith('_volatility'):
                        product_correlations[factor_name] = factor_value
                
                if product_correlations:
                    report['festival_correlations'][pattern.product_id] = product_correlations
        
        # Analyze cyclical patterns
        all_cyclical_patterns = []
        for record in filtered_data:
            product_data = [d for d in filtered_data if d.product_id == record.product_id]
            if len(product_data) >= 12:
                cyclical = await self._detect_cyclical_patterns(product_data)
                if cyclical:
                    all_cyclical_patterns.extend(cyclical)
        
        report['cyclical_patterns'] = {
            'total_patterns': len(all_cyclical_patterns),
            'pattern_types': list(set(p['type'] for p in all_cyclical_patterns)),
            'patterns': all_cyclical_patterns[:10]  # Limit to top 10
        }
        
        # Category analysis
        categories = set(record.category for record in filtered_data)
        for category in categories:
            category_data = [r for r in filtered_data if r.category == category]
            category_seasonal_factors = {}
            
            for event_name, event_info in seasonal_events.items():
                if category.lower() in [cat.lower() for cat in event_info.get('categories', [])]:
                    event_sales = [r.quantity_sold for r in category_data 
                                 if r.sale_date.month in event_info['months']]
                    non_event_sales = [r.quantity_sold for r in category_data 
                                     if r.sale_date.month not in event_info['months']]
                    
                    if event_sales and non_event_sales:
                        factor = np.mean(event_sales) / np.mean(non_event_sales)
                        category_seasonal_factors[event_name] = factor
            
            if category_seasonal_factors:
                report['category_analysis'][category] = category_seasonal_factors
        
        # Generate recommendations
        recommendations = []
        
        # High-performing seasonal events
        high_performers = [
            event for event, data in report['seasonal_insights'].items()
            if data['performance_ratio'] > 1.2
        ]
        if high_performers:
            recommendations.append({
                'type': 'opportunity',
                'title': 'Strong Seasonal Performance',
                'description': f"Products show strong performance during {', '.join(high_performers)}. Consider increasing inventory and marketing efforts.",
                'events': high_performers
            })
        
        # Underperforming seasonal events
        underperformers = [
            event for event, data in report['seasonal_insights'].items()
            if data['performance_ratio'] < 0.8
        ]
        if underperformers:
            recommendations.append({
                'type': 'improvement',
                'title': 'Seasonal Underperformance',
                'description': f"Products underperform during {', '.join(underperformers)}. Review pricing, inventory, and marketing strategies.",
                'events': underperformers
            })
        
        # High volatility events
        volatile_events = [
            event for event, data in report['seasonal_insights'].items()
            if data['volatility'] > 0.5
        ]
        if volatile_events:
            recommendations.append({
                'type': 'risk_management',
                'title': 'High Seasonal Volatility',
                'description': f"High sales volatility during {', '.join(volatile_events)}. Implement flexible inventory and pricing strategies.",
                'events': volatile_events
            })
        
        report['recommendations'] = recommendations
        
        logger.info(f"Seasonal analysis report generated with {len(enhanced_patterns)} patterns analyzed")
        return report