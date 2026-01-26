"""
Tests for MarketPulse AI storage functionality.

Tests data storage, retrieval, encryption, and caching operations
with proper error handling and data integrity validation.
"""

import pytest
import tempfile
import os
from datetime import datetime, date, timedelta
from decimal import Decimal
from uuid import uuid4

from marketpulse_ai.storage.models import (
    SalesDataModel, DemandPatternModel, CacheEntryModel, AuditLogModel
)
from marketpulse_ai.storage.encryption import EncryptionManager, EncryptionError
from marketpulse_ai.storage.storage_manager import StorageManager, StorageError
from marketpulse_ai.storage.factory import StorageFactory, get_storage_factory
from marketpulse_ai.config.database import DatabaseConfig, DatabaseManager
from marketpulse_ai.config.security import SecurityConfig
from marketpulse_ai.core.models import SalesDataPoint, DemandPattern, ConfidenceLevel


class TestEncryptionManager:
    """Test encryption functionality."""
    
    @pytest.fixture
    def security_config(self):
        """Create security config for testing."""
        return SecurityConfig(
            secret_key="test-secret-key-32-characters-long",
            enable_data_encryption=True,
            encryption_key=None  # Will be auto-generated
        )
    
    @pytest.fixture
    def encryption_manager(self, security_config):
        """Create encryption manager for testing."""
        return EncryptionManager(security_config)
    
    def test_encryption_initialization(self, encryption_manager):
        """Test encryption manager initialization."""
        assert encryption_manager.is_encryption_enabled()
        
        info = encryption_manager.get_encryption_info()
        assert info['encryption_enabled'] is True
        assert info['cipher_available'] is True
        assert 'Fernet' in info['algorithm']
    
    def test_encrypt_decrypt_string(self, encryption_manager):
        """Test string encryption and decryption."""
        original_value = "test string value"
        
        # Encrypt
        encrypted = encryption_manager.encrypt_value(original_value)
        assert encrypted != original_value
        assert isinstance(encrypted, str)
        
        # Decrypt
        decrypted = encryption_manager.decrypt_value(encrypted, str)
        assert decrypted == original_value
    
    def test_encrypt_decrypt_decimal(self, encryption_manager):
        """Test Decimal encryption and decryption."""
        original_value = Decimal('123.45')
        
        # Encrypt
        encrypted = encryption_manager.encrypt_value(original_value)
        assert encrypted != str(original_value)
        
        # Decrypt
        decrypted = encryption_manager.decrypt_value(encrypted, Decimal)
        assert decrypted == original_value
        assert isinstance(decrypted, Decimal)
    
    def test_encrypt_decrypt_dict(self, encryption_manager):
        """Test dictionary field encryption."""
        original_dict = {
            'product_id': 'PROD001',
            'mrp': Decimal('100.00'),
            'selling_price': Decimal('85.00'),
            'quantity': 5
        }
        
        # Encrypt specific fields
        encrypted_dict = encryption_manager.encrypt_dict(
            original_dict, ['mrp', 'selling_price']
        )
        
        # Check encrypted fields exist and original fields are removed
        assert 'mrp_encrypted' in encrypted_dict
        assert 'selling_price_encrypted' in encrypted_dict
        assert 'mrp' not in encrypted_dict
        assert 'selling_price' not in encrypted_dict
        assert encrypted_dict['product_id'] == 'PROD001'  # Unencrypted field preserved
        
        # Decrypt fields
        decrypted_dict = encryption_manager.decrypt_dict(
            encrypted_dict,
            {'mrp_encrypted': Decimal, 'selling_price_encrypted': Decimal}
        )
        
        # Check decrypted values
        assert decrypted_dict['mrp'] == original_dict['mrp']
        assert decrypted_dict['selling_price'] == original_dict['selling_price']
        assert 'mrp_encrypted' not in decrypted_dict
        assert 'selling_price_encrypted' not in decrypted_dict
    
    def test_encryption_disabled(self):
        """Test behavior when encryption is disabled."""
        security_config = SecurityConfig(
            secret_key="test-secret-key-32-characters-long",
            enable_data_encryption=False
        )
        
        encryption_manager = EncryptionManager(security_config)
        assert not encryption_manager.is_encryption_enabled()
        
        # Should raise error when trying to encrypt
        with pytest.raises(EncryptionError):
            encryption_manager.encrypt_value("test")
    
    def test_invalid_token_decryption(self, encryption_manager):
        """Test decryption with invalid token."""
        with pytest.raises(EncryptionError, match="Invalid encryption token"):
            encryption_manager.decrypt_value("invalid-encrypted-data", str)


class TestStorageManager:
    """Test storage manager functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Clean up with retry logic for Windows
        try:
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError:
            # File might be locked on Windows, ignore for tests
            pass
    
    @pytest.fixture
    def db_config(self, temp_db_path):
        """Create database config for testing."""
        return DatabaseConfig(url=f"sqlite:///{temp_db_path}")
    
    @pytest.fixture
    def db_manager(self, db_config):
        """Create database manager for testing."""
        manager = DatabaseManager(db_config)
        try:
            manager.create_tables()
        except Exception:
            # Tables might already exist, ignore the error
            pass
        yield manager
        # Close connections properly
        if hasattr(manager, 'engine'):
            manager.engine.dispose()
    
    @pytest.fixture
    def security_config(self):
        """Create security config for testing."""
        from cryptography.fernet import Fernet
        return SecurityConfig(
            secret_key="test-secret-key-32-characters-long",
            enable_data_encryption=True,
            encryption_key=Fernet.generate_key().decode()  # Generate a valid key
        )
    
    @pytest.fixture
    def storage_manager(self, db_manager, security_config):
        """Create storage manager for testing."""
        return StorageManager(db_manager, security_config)
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create sample sales data for testing."""
        return [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Test Product 1",
                category="electronics",
                mrp=Decimal('100.00'),
                selling_price=Decimal('85.00'),
                quantity_sold=5,
                sale_date=date.today(),
                store_location="STORE001"
            ),
            SalesDataPoint(
                product_id="PROD002",
                product_name="Test Product 2",
                category="clothing",
                mrp=Decimal('50.00'),
                selling_price=Decimal('45.00'),
                quantity_sold=3,
                sale_date=date.today() - timedelta(days=1),
                store_location="STORE002"
            )
        ]
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample demand patterns for testing."""
        return [
            DemandPattern(
                product_id="PROD001",
                pattern_type="seasonal",
                description="Test seasonal pattern",
                confidence_level=ConfidenceLevel.HIGH,
                seasonal_factors={"diwali": 1.5, "summer": 0.8},
                trend_direction="increasing",
                volatility_score=0.3,
                supporting_data_points=10,
                date_range_start=date.today() - timedelta(days=30),
                date_range_end=date.today()
            )
        ]
    
    @pytest.mark.asyncio
    async def test_store_sales_data(self, storage_manager, sample_sales_data):
        """Test storing sales data."""
        result = await storage_manager.store_sales_data(sample_sales_data)
        
        assert result['status'] == 'success'
        assert result['stored_count'] == 2
        assert result['failed_count'] == 0
        assert result['total_count'] == 2
    
    @pytest.mark.asyncio
    async def test_retrieve_sales_data(self, storage_manager, sample_sales_data):
        """Test retrieving sales data."""
        # Store data first
        await storage_manager.store_sales_data(sample_sales_data)
        
        # Retrieve all data
        retrieved_data = await storage_manager.retrieve_sales_data()
        
        assert len(retrieved_data) == 2
        assert retrieved_data[0].product_id in ["PROD001", "PROD002"]
        assert isinstance(retrieved_data[0].mrp, Decimal)
        assert isinstance(retrieved_data[0].selling_price, Decimal)
    
    @pytest.mark.asyncio
    async def test_retrieve_sales_data_with_filters(self, storage_manager, sample_sales_data):
        """Test retrieving sales data with filters."""
        # Store data first
        await storage_manager.store_sales_data(sample_sales_data)
        
        # Filter by product ID
        filtered_data = await storage_manager.retrieve_sales_data(product_ids=["PROD001"])
        assert len(filtered_data) == 1
        assert filtered_data[0].product_id == "PROD001"
        
        # Filter by category
        category_data = await storage_manager.retrieve_sales_data(categories=["electronics"])
        assert len(category_data) == 1
        assert category_data[0].category == "electronics"
        
        # Filter by date range
        date_range = (date.today() - timedelta(days=2), date.today())
        date_filtered = await storage_manager.retrieve_sales_data(date_range=date_range)
        assert len(date_filtered) == 2
    
    @pytest.mark.asyncio
    async def test_store_patterns(self, storage_manager, sample_patterns):
        """Test storing demand patterns."""
        result = await storage_manager.store_patterns(sample_patterns)
        
        assert result['status'] == 'success'
        assert result['stored_count'] == 1
        assert result['failed_count'] == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_patterns(self, storage_manager, sample_patterns):
        """Test retrieving demand patterns."""
        # Store patterns first
        await storage_manager.store_patterns(sample_patterns)
        
        # Retrieve all patterns
        retrieved_patterns = await storage_manager.retrieve_patterns()
        
        assert len(retrieved_patterns) == 1
        assert retrieved_patterns[0].product_id == "PROD001"
        assert retrieved_patterns[0].pattern_type == "seasonal"
        assert retrieved_patterns[0].seasonal_factors["diwali"] == 1.5
    
    @pytest.mark.asyncio
    async def test_pattern_caching(self, storage_manager, sample_patterns):
        """Test pattern caching functionality."""
        # Store patterns
        await storage_manager.store_patterns(sample_patterns)
        
        # First retrieval (should cache)
        patterns1 = await storage_manager.retrieve_patterns(use_cache=True)
        
        # Second retrieval (should use cache)
        patterns2 = await storage_manager.retrieve_patterns(use_cache=True)
        
        assert len(patterns1) == len(patterns2) == 1
        assert patterns1[0].id == patterns2[0].id
    
    @pytest.mark.asyncio
    async def test_cache_cleanup(self, storage_manager):
        """Test cache cleanup functionality."""
        # This would require creating expired cache entries
        # For now, just test that the method runs without error
        result = await storage_manager.cleanup_expired_cache()
        assert 'deleted_count' in result
    
    @pytest.mark.asyncio
    async def test_storage_statistics(self, storage_manager, sample_sales_data, sample_patterns):
        """Test storage statistics retrieval."""
        # Store some data
        await storage_manager.store_sales_data(sample_sales_data)
        await storage_manager.store_patterns(sample_patterns)
        
        # Get statistics
        stats = await storage_manager.get_storage_statistics()
        
        assert 'sales_data_count' in stats
        assert 'pattern_count' in stats
        assert 'cache_entries' in stats
        assert 'encryption_info' in stats
        assert 'database_health' in stats
        
        assert stats['sales_data_count'] >= 2
        assert stats['pattern_count'] >= 1


class TestStorageFactory:
    """Test storage factory functionality."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = StorageFactory()
        assert factory is not None
    
    def test_create_database_config(self):
        """Test database config creation."""
        factory = StorageFactory()
        db_config = factory.create_database_config()
        
        assert isinstance(db_config, DatabaseConfig)
        assert db_config.url.startswith('sqlite:///')
    
    def test_create_security_config(self):
        """Test security config creation."""
        factory = StorageFactory()
        security_config = factory.create_security_config()
        
        assert isinstance(security_config, SecurityConfig)
        assert security_config.enable_data_encryption is True
    
    def test_get_managers(self):
        """Test manager creation."""
        factory = StorageFactory()
        
        # Test database manager
        db_manager = factory.get_database_manager()
        assert isinstance(db_manager, DatabaseManager)
        
        # Test encryption manager
        encryption_manager = factory.get_encryption_manager()
        assert isinstance(encryption_manager, EncryptionManager)
        
        # Test storage manager
        storage_manager = factory.get_storage_manager()
        assert isinstance(storage_manager, StorageManager)
    
    def test_singleton_behavior(self):
        """Test that managers are singletons."""
        factory = StorageFactory()
        
        # Get managers twice
        db_manager1 = factory.get_database_manager()
        db_manager2 = factory.get_database_manager()
        
        # Should be the same instance
        assert db_manager1 is db_manager2
    
    def test_global_factory(self):
        """Test global factory function."""
        factory1 = get_storage_factory()
        factory2 = get_storage_factory()
        
        # Should be the same instance
        assert factory1 is factory2
    
    def test_configure_data_processor(self):
        """Test data processor configuration."""
        from marketpulse_ai.components.data_processor import DataProcessor
        
        factory = StorageFactory()
        data_processor = DataProcessor()
        
        # Configure with storage
        factory.configure_data_processor(data_processor)
        
        # Should have storage manager set
        assert hasattr(data_processor, 'storage_manager')
        assert data_processor.storage_manager is not None
    
    def test_storage_health_check(self):
        """Test storage health check."""
        factory = StorageFactory()
        health = factory.check_storage_health()
        
        assert 'database' in health
        assert 'encryption' in health
        assert 'storage_manager' in health
        assert 'overall' in health


class TestIntegration:
    """Integration tests for storage functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Clean up with retry logic for Windows
        try:
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError:
            # File might be locked on Windows, ignore for tests
            pass
    
    @pytest.fixture
    def configured_data_processor(self, temp_db_path):
        """Create configured data processor for testing."""
        from marketpulse_ai.components.data_processor import DataProcessor
        from marketpulse_ai.config.settings import Settings
        from marketpulse_ai.storage.factory import StorageFactory
        from cryptography.fernet import Fernet
        
        # Create test settings with required fields
        settings = Settings(
            secret_key="test-secret-key-32-characters-long",
            database_url=f"sqlite:///{temp_db_path}",
            enable_data_encryption=True,
            encryption_key=Fernet.generate_key().decode()
        )
        
        # Create factory with test settings
        factory = StorageFactory(settings)
        
        # Create and configure data processor
        data_processor = DataProcessor()
        factory.configure_data_processor(data_processor)
        
        return data_processor
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self, configured_data_processor):
        """Test complete data flow from ingestion to retrieval."""
        # Create sample data
        sample_data = [
            SalesDataPoint(
                product_id="PROD001",
                product_name="Integration Test Product",
                category="test_category",
                mrp=Decimal('200.00'),
                selling_price=Decimal('180.00'),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="TEST_STORE"
            )
        ]
        
        # Ingest data
        ingestion_result = await configured_data_processor.ingest_sales_data(sample_data)
        assert ingestion_result['status'] == 'success'
        assert ingestion_result['records_accepted'] == 1
        
        # Extract patterns
        patterns = await configured_data_processor.extract_demand_patterns(['PROD001'])
        
        # Should have at least basic patterns
        assert isinstance(patterns, list)
        
        # Store patterns
        store_result = await configured_data_processor.store_patterns(patterns)
        assert store_result is True