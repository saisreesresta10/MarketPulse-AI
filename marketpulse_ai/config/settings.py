"""
Main configuration settings for MarketPulse AI.

Handles environment-specific configuration with validation and
secure defaults for different deployment environments.
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Optional, List, Dict, Any

from pydantic import Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Main application settings with environment-specific configuration.
    
    Uses Pydantic BaseSettings for automatic environment variable loading
    and validation with secure defaults.
    """
    
    # Application settings
    app_name: str = Field(default="MarketPulse AI", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Deployment environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # API settings
    api_host: str = Field(default="localhost", description="API host address")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port number")
    api_prefix: str = Field(default="/api/v1", description="API URL prefix")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], description="CORS allowed origins")
    
    # Database settings
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    database_echo: bool = Field(default=False, description="Enable database query logging")
    database_pool_size: int = Field(default=5, ge=1, description="Database connection pool size")
    database_max_overflow: int = Field(default=10, ge=0, description="Database connection pool overflow")
    
    # Security settings
    secret_key: str = Field(..., min_length=32, description="Secret key for encryption")
    encryption_key: Optional[str] = Field(default=None, description="Data encryption key")
    encryption_algorithm: str = Field(default="HS256", description="Encryption algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="Access token expiration time")
    
    # Data processing settings
    max_batch_size: int = Field(default=1000, ge=1, description="Maximum batch size for data processing")
    data_retention_days: int = Field(default=365, ge=1, description="Data retention period in days")
    enable_data_encryption: bool = Field(default=True, description="Enable data encryption at rest")
    
    # AI/ML settings
    model_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_scenarios_per_request: int = Field(default=10, ge=1, le=50, description="Maximum scenarios per analysis request")
    enable_property_testing: bool = Field(default=True, description="Enable property-based testing")
    property_test_iterations: int = Field(default=100, ge=10, description="Property test iterations")
    
    # Compliance settings
    mrp_compliance_strict: bool = Field(default=True, description="Enable strict MRP compliance checking")
    regulation_update_check_hours: int = Field(default=24, ge=1, description="Hours between regulation update checks")
    
    # Performance settings
    request_timeout_seconds: int = Field(default=30, ge=1, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(default=100, ge=1, description="Maximum concurrent requests")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, ge=1, description="Cache TTL in seconds")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    
    @model_validator(mode='after')
    def validate_environment_settings(self):
        """Validate environment-specific settings."""
        if self.environment == Environment.PRODUCTION:
            # Production environment validations
            if self.debug:
                raise ValueError("Debug mode must be disabled in production")
            if self.database_echo:
                raise ValueError("Database query logging must be disabled in production")
            if "http://localhost:3000" in self.cors_origins:
                raise ValueError("Localhost origins not allowed in production")
        return self
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v, info):
        """Set default database URL based on environment if not provided."""
        if v is None:
            env = info.data.get('environment', Environment.DEVELOPMENT)
            if env == Environment.TESTING:
                return "sqlite:///./test_marketpulse.db"
            elif env == Environment.DEVELOPMENT:
                return "sqlite:///./marketpulse.db"
            else:
                raise ValueError("Database URL must be provided for staging/production environments")
        return v

    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "echo": self.database_echo,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
        }
    
    @property
    def security_config(self) -> Dict[str, Any]:
        """Get security configuration dictionary."""
        return {
            "secret_key": self.secret_key,
            "encryption_key": self.encryption_key,
            "encryption_algorithm": self.encryption_algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "enable_data_encryption": self.enable_data_encryption,
        }
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        return {
            "level": self.log_level,
            "format": self.log_format,
            "enable_audit": self.enable_audit_logging,
        }

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="MARKETPULSE_",
        # Field aliases for common environment variables
        env_nested_delimiter="__"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses LRU cache to ensure settings are loaded only once per application
    lifecycle, improving performance and consistency.
    
    Returns:
        Configured Settings instance
    """
    return Settings()