"""
Database configuration for MarketPulse AI.

Provides database connection management, migration support,
and environment-specific database settings.
"""

from typing import Dict, Any, Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    """
    Database configuration with validation and connection management.
    
    Handles different database engines and provides secure connection
    settings for various deployment environments.
    """
    
    url: str = Field(..., description="Database connection URL")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=5, ge=1, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, ge=1, description="Pool connection timeout")
    pool_recycle: int = Field(default=3600, ge=1, description="Pool connection recycle time")
    connect_args: Dict[str, Any] = Field(default_factory=dict, description="Additional connection arguments")
    
    @field_validator('url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format and security."""
        if not v.startswith(('sqlite:///', 'postgresql://', 'mysql://', 'oracle://')):
            raise ValueError("Unsupported database URL format")
        
        # Security check for production databases
        if v.startswith('sqlite:///') and not v.endswith(('.db', '.sqlite', '.sqlite3')):
            raise ValueError("SQLite database must have proper file extension")
        
        return v
    
    @field_validator('connect_args')
    @classmethod
    def set_sqlite_connect_args(cls, v, info):
        """Set SQLite-specific connection arguments."""
        url = info.data.get('url', '')
        if url.startswith('sqlite:///'):
            # SQLite-specific settings for better performance and reliability
            v.update({
                'check_same_thread': False,  # Allow multi-threading
                'poolclass': StaticPool,     # Use static pool for SQLite
            })
        return v
    
    def create_engine_config(self) -> Dict[str, Any]:
        """
        Create SQLAlchemy engine configuration.
        
        Returns:
            Dictionary with engine configuration parameters
        """
        config = {
            'echo': self.echo,
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_timeout': self.pool_timeout,
            'pool_recycle': self.pool_recycle,
            'connect_args': self.connect_args,
        }
        
        # SQLite doesn't support connection pooling
        if self.url.startswith('sqlite:///'):
            config.pop('pool_size', None)
            config.pop('max_overflow', None)
            config.pop('pool_timeout', None)
            config.pop('pool_recycle', None)
        
        return config
    
    def create_engine(self):
        """
        Create SQLAlchemy engine with configuration.
        
        Returns:
            Configured SQLAlchemy engine
        """
        return create_engine(self.url, **self.create_engine_config())
    
    def create_session_factory(self):
        """
        Create SQLAlchemy session factory.
        
        Returns:
            Configured sessionmaker class
        """
        engine = self.create_engine()
        return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# SQLAlchemy declarative base for model definitions
Base = declarative_base()

# Metadata for schema management
metadata = MetaData()


class DatabaseManager:
    """
    Database connection and session management.
    
    Provides centralized database connection handling with proper
    resource management and connection pooling.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager with configuration.
        
        Args:
            config: Database configuration instance
        """
        self.config = config
        self.engine = config.create_engine()
        self.SessionLocal = config.create_session_factory()
    
    def get_session(self):
        """
        Get database session with proper resource management.
        
        Yields:
            SQLAlchemy database session
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables from models."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables (use with caution)."""
        Base.metadata.drop_all(bind=self.engine)
    
    def check_connection(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.engine.connect() as connection:
                connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get database connection information.
        
        Returns:
            Dictionary with connection details (excluding sensitive data)
        """
        return {
            'driver': self.engine.driver,
            'dialect': self.engine.dialect.name,
            'pool_size': getattr(self.engine.pool, 'size', None),
            'checked_out': getattr(self.engine.pool, 'checkedout', None),
            'overflow': getattr(self.engine.pool, 'overflow', None),
        }