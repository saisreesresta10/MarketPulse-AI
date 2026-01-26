"""
Unit tests for configuration management in MarketPulse AI.

Tests settings validation, environment handling, and configuration
loading for different deployment environments.
"""

import pytest
from pydantic import ValidationError

from marketpulse_ai.config.settings import Settings, Environment
from marketpulse_ai.config.database import DatabaseConfig
from marketpulse_ai.config.security import SecurityConfig


class TestSettings:
    """Test cases for Settings configuration."""
    
    def test_default_settings_creation(self):
        """Test creation of settings with default values."""
        settings = Settings(
            secret_key="test_secret_key_with_sufficient_length_for_security",
            environment=Environment.DEVELOPMENT,  # Explicitly set for test
            debug=False  # Explicitly set to test default behavior
        )
        
        assert settings.app_name == "MarketPulse AI"
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.api_port == 8000
        assert settings.debug == False
        assert settings.enable_property_testing == True
    
    def test_environment_specific_validation(self):
        """Test environment-specific validation rules."""
        # Production environment should not allow debug mode
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                environment=Environment.PRODUCTION,
                debug=True,  # Not allowed in production
                secret_key="test_secret_key_with_sufficient_length_for_security",
                cors_origins=["https://production.example.com"]  # Valid production CORS
            )
        
        assert "Debug mode must be disabled in production" in str(exc_info.value)
    
    def test_database_url_defaults(self):
        """Test database URL defaults based on environment."""
        # Development environment
        dev_settings = Settings(
            environment=Environment.DEVELOPMENT,
            secret_key="test_secret_key_with_sufficient_length_for_security"
        )
        assert "marketpulse.db" in dev_settings.database_url
        
        # Testing environment
        test_settings = Settings(
            environment=Environment.TESTING,
            secret_key="test_secret_key_with_sufficient_length_for_security"
        )
        assert "test_marketpulse.db" in test_settings.database_url


class TestDatabaseConfig:
    """Test cases for DatabaseConfig."""
    
    def test_valid_database_config_creation(self):
        """Test creation of valid database configuration."""
        config = DatabaseConfig(
            url="sqlite:///./test.db",
            echo=False,
            pool_size=5
        )
        
        assert config.url == "sqlite:///./test.db"
        assert config.echo == False
        assert config.pool_size == 5
    
    def test_invalid_database_url_validation(self):
        """Test validation of invalid database URLs."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(
                url="invalid://database/url"  # Unsupported protocol
            )
        
        assert "Unsupported database URL format" in str(exc_info.value)
    
    def test_engine_config_creation(self):
        """Test SQLAlchemy engine configuration creation."""
        config = DatabaseConfig(
            url="sqlite:///./test.db",
            pool_size=10,
            max_overflow=20
        )
        
        engine_config = config.create_engine_config()
        
        # SQLite should not have pool settings
        assert 'pool_size' not in engine_config
        assert 'connect_args' in engine_config


class TestSecurityConfig:
    """Test cases for SecurityConfig."""
    
    def test_valid_security_config_creation(self):
        """Test creation of valid security configuration."""
        config = SecurityConfig(
            secret_key="test_secret_key_with_sufficient_length_for_security",
            encryption_algorithm="HS256",
            enable_data_encryption=True
        )
        
        assert len(config.secret_key) >= 32
        assert config.encryption_algorithm == "HS256"
        assert config.enable_data_encryption == True
        assert config.encryption_key is not None  # Should be auto-generated
    
    def test_short_secret_key_validation(self):
        """Test validation of short secret keys."""
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(
                secret_key="short_key"  # Too short
            )
        
        assert "at least 32 characters" in str(exc_info.value)
    
    def test_invalid_algorithm_validation(self):
        """Test validation of invalid algorithms."""
        with pytest.raises(ValidationError) as exc_info:
            SecurityConfig(
                secret_key="test_secret_key_with_sufficient_length_for_security",
                encryption_algorithm="INVALID_ALGO"
            )
        
        assert "Algorithm must be one of" in str(exc_info.value)
    
    def test_password_requirements(self):
        """Test password requirements configuration."""
        config = SecurityConfig(
            secret_key="test_secret_key_with_sufficient_length_for_security"
        )
        
        requirements = config.get_password_requirements()
        
        assert requirements['min_length'] >= 8
        assert requirements['require_uppercase'] == True
        assert requirements['require_lowercase'] == True
        assert requirements['require_numbers'] == True
        assert requirements['require_special'] == True


class TestConfigurationIntegration:
    """Integration tests for configuration components."""
    
    def test_settings_database_config_integration(self):
        """Test integration between Settings and DatabaseConfig."""
        settings = Settings(
            secret_key="test_secret_key_with_sufficient_length_for_security",
            database_url="sqlite:///./integration_test.db"
        )
        
        db_config = DatabaseConfig(**settings.database_config)
        
        assert db_config.url == settings.database_url
        assert db_config.echo == settings.database_echo
    
    def test_settings_security_config_integration(self):
        """Test integration between Settings and SecurityConfig."""
        settings = Settings(
            secret_key="test_secret_key_with_sufficient_length_for_security",
            encryption_algorithm="HS512"
        )
        
        security_config = SecurityConfig(**settings.security_config)
        
        assert security_config.secret_key == settings.secret_key
        assert security_config.encryption_algorithm == settings.encryption_algorithm
    
    def test_environment_configuration_consistency(self):
        """Test that configuration is consistent across environments."""
        environments = [Environment.DEVELOPMENT, Environment.TESTING, Environment.STAGING]
        
        for env in environments:
            settings = Settings(
                environment=env,
                secret_key="test_secret_key_with_sufficient_length_for_security"
            )
            
            # All environments should have valid configuration
            assert settings.environment == env
            assert len(settings.secret_key) >= 32
            assert settings.database_url is not None