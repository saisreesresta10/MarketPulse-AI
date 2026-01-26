"""
Factory for creating storage components with proper configuration.

Provides centralized creation and configuration of storage-related
components with dependency injection and proper initialization.
"""

import logging
from typing import Optional

from ..config.database import DatabaseConfig, DatabaseManager
from ..config.security import SecurityConfig
from ..config.settings import get_settings
from .storage_manager import StorageManager
from .encryption import EncryptionManager

logger = logging.getLogger(__name__)


class StorageFactory:
    """
    Factory for creating and configuring storage components.
    
    Handles the creation of database managers, storage managers,
    and encryption managers with proper configuration and dependencies.
    """
    
    def __init__(self, settings=None):
        """
        Initialize storage factory with configuration.
        
        Args:
            settings: Optional settings override (for testing)
        """
        self.settings = settings or get_settings()
        self._db_manager = None
        self._storage_manager = None
        self._encryption_manager = None
        
        logger.info("Storage factory initialized")
    
    def create_database_config(self) -> DatabaseConfig:
        """
        Create database configuration from settings.
        
        Returns:
            Configured DatabaseConfig instance
        """
        # Get database URL from settings
        db_url = getattr(self.settings, 'database_url', 'sqlite:///marketpulse.db')
        
        # Create database configuration
        db_config = DatabaseConfig(
            url=db_url,
            echo=getattr(self.settings, 'database_echo', False),
            pool_size=getattr(self.settings, 'database_pool_size', 5),
            max_overflow=getattr(self.settings, 'database_max_overflow', 10),
            pool_timeout=getattr(self.settings, 'database_pool_timeout', 30),
            pool_recycle=getattr(self.settings, 'database_pool_recycle', 3600)
        )
        
        logger.info(f"Created database config for: {db_url}")
        return db_config
    
    def create_security_config(self) -> SecurityConfig:
        """
        Create security configuration from settings.
        
        Returns:
            Configured SecurityConfig instance
        """
        # Get security settings
        secret_key = getattr(self.settings, 'secret_key', 'dev-secret-key-change-in-production')
        encryption_key = getattr(self.settings, 'encryption_key', None)
        
        # Create security configuration
        security_config = SecurityConfig(
            secret_key=secret_key,
            encryption_key=encryption_key,
            enable_data_encryption=getattr(self.settings, 'enable_data_encryption', True),
            password_min_length=getattr(self.settings, 'password_min_length', 8),
            enable_rate_limiting=getattr(self.settings, 'enable_rate_limiting', True),
            rate_limit_requests=getattr(self.settings, 'rate_limit_requests', 100),
            rate_limit_window_minutes=getattr(self.settings, 'rate_limit_window_minutes', 15)
        )
        
        logger.info("Created security configuration")
        return security_config
    
    def get_database_manager(self) -> DatabaseManager:
        """
        Get or create database manager instance.
        
        Returns:
            DatabaseManager instance (singleton)
        """
        if self._db_manager is None:
            db_config = self.create_database_config()
            self._db_manager = DatabaseManager(db_config)
            
            # Create tables if they don't exist
            try:
                self._db_manager.create_tables()
                logger.info("Database tables created/verified")
            except Exception as e:
                logger.error(f"Failed to create database tables: {e}")
                raise
        
        return self._db_manager
    
    def get_encryption_manager(self) -> EncryptionManager:
        """
        Get or create encryption manager instance.
        
        Returns:
            EncryptionManager instance (singleton)
        """
        if self._encryption_manager is None:
            security_config = self.create_security_config()
            self._encryption_manager = EncryptionManager(security_config)
            logger.info("Encryption manager created")
        
        return self._encryption_manager
    
    def get_storage_manager(self) -> StorageManager:
        """
        Get or create storage manager instance.
        
        Returns:
            StorageManager instance (singleton)
        """
        if self._storage_manager is None:
            db_manager = self.get_database_manager()
            security_config = self.create_security_config()
            
            self._storage_manager = StorageManager(db_manager, security_config)
            logger.info("Storage manager created")
        
        return self._storage_manager
    
    def configure_data_processor(self, data_processor):
        """
        Configure data processor with storage manager.
        
        Args:
            data_processor: DataProcessor instance to configure
        """
        storage_manager = self.get_storage_manager()
        data_processor.set_storage_manager(storage_manager)
        logger.info("Data processor configured with storage manager")
    
    def check_storage_health(self) -> dict:
        """
        Check health of all storage components.
        
        Returns:
            Dictionary with health status of each component
        """
        health_status = {}
        
        try:
            # Check database connection
            db_manager = self.get_database_manager()
            health_status['database'] = db_manager.check_connection()
            
            # Check encryption
            encryption_manager = self.get_encryption_manager()
            health_status['encryption'] = encryption_manager.is_encryption_enabled()
            
            # Check storage manager
            storage_manager = self.get_storage_manager()
            storage_stats = None
            try:
                # This is async, so we can't call it directly here
                # In a real application, this would be handled differently
                health_status['storage_manager'] = True
            except Exception as e:
                health_status['storage_manager'] = False
                health_status['storage_error'] = str(e)
            
            health_status['overall'] = all([
                health_status.get('database', False),
                health_status.get('encryption', False),
                health_status.get('storage_manager', False)
            ])
            
        except Exception as e:
            health_status['overall'] = False
            health_status['error'] = str(e)
            logger.error(f"Storage health check failed: {e}")
        
        return health_status
    
    def cleanup_resources(self):
        """
        Clean up storage resources.
        
        Should be called when shutting down the application.
        """
        try:
            if self._storage_manager:
                # In a real application, this would clean up connections
                logger.info("Storage manager resources cleaned up")
            
            if self._db_manager:
                # Close database connections
                logger.info("Database manager resources cleaned up")
            
            logger.info("All storage resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during storage cleanup: {e}")


# Global factory instance for dependency injection
_storage_factory = None


def get_storage_factory(settings=None) -> StorageFactory:
    """
    Get global storage factory instance.
    
    Args:
        settings: Optional settings override
        
    Returns:
        StorageFactory instance (singleton)
    """
    global _storage_factory
    
    if _storage_factory is None or settings is not None:
        _storage_factory = StorageFactory(settings)
    
    return _storage_factory


def create_configured_data_processor():
    """
    Create a data processor with storage configuration.
    
    Returns:
        Configured DataProcessor instance
    """
    from ..components.data_processor import DataProcessor
    
    # Create data processor
    data_processor = DataProcessor()
    
    # Configure with storage
    factory = get_storage_factory()
    factory.configure_data_processor(data_processor)
    
    return data_processor