"""
Security configuration for MarketPulse AI.

Provides encryption, authentication, and data protection settings
with secure defaults and validation.
"""

import secrets
from typing import Dict, Any, Optional, List
from cryptography.fernet import Fernet
from passlib.context import CryptContext

from pydantic import BaseModel, Field, field_validator, model_validator


class SecurityConfig(BaseModel):
    """
    Security configuration with encryption and authentication settings.
    
    Manages cryptographic keys, password hashing, and security policies
    for data protection and user authentication.
    """
    
    secret_key: str = Field(..., min_length=32, description="Main application secret key")
    encryption_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, ge=1, description="Refresh token expiration")
    
    # Password security
    password_min_length: int = Field(default=8, ge=6, description="Minimum password length")
    password_require_uppercase: bool = Field(default=True, description="Require uppercase letters")
    password_require_lowercase: bool = Field(default=True, description="Require lowercase letters")
    password_require_numbers: bool = Field(default=True, description="Require numbers")
    password_require_special: bool = Field(default=True, description="Require special characters")
    
    # Data encryption
    enable_data_encryption: bool = Field(default=True, description="Enable data encryption at rest")
    encryption_key: Optional[str] = Field(default=None, description="Data encryption key")
    
    # API security
    enable_rate_limiting: bool = Field(default=True, description="Enable API rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per rate limit window")
    rate_limit_window_minutes: int = Field(default=15, ge=1, description="Rate limit window in minutes")
    
    # CORS settings
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="Allowed HTTP methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed HTTP headers")
    
    # Security headers
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    hsts_max_age: int = Field(default=31536000, ge=0, description="HSTS max age in seconds")
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        """Validate secret key strength."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        
        # Check for sufficient entropy (basic check)
        if v.isalnum() and len(set(v)) < 10:
            raise ValueError("Secret key must have sufficient entropy")
        
        return v
    
    @model_validator(mode='after')
    def validate_encryption_key(self):
        """Generate or validate encryption key."""
        if self.enable_data_encryption:
            if self.encryption_key is None:
                # Generate a new Fernet key if none provided
                self.encryption_key = Fernet.generate_key().decode()
            else:
                # Validate provided key
                try:
                    Fernet(self.encryption_key.encode())
                except Exception:
                    raise ValueError("Invalid encryption key format")
        return self
    
    @field_validator('encryption_algorithm')
    @classmethod
    def validate_algorithm(cls, v):
        """Validate JWT algorithm."""
        allowed_algorithms = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512']
        if v not in allowed_algorithms:
            raise ValueError(f"Algorithm must be one of: {allowed_algorithms}")
        return v
    
    def create_password_context(self) -> CryptContext:
        """
        Create password hashing context.
        
        Returns:
            Configured PassLib CryptContext for password hashing
        """
        return CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12,  # Strong hashing rounds
        )
    
    def create_fernet_cipher(self) -> Optional[Fernet]:
        """
        Create Fernet cipher for data encryption.
        
        Returns:
            Fernet cipher instance if encryption is enabled, None otherwise
        """
        if self.enable_data_encryption and self.encryption_key:
            return Fernet(self.encryption_key.encode())
        return None
    
    def get_password_requirements(self) -> Dict[str, Any]:
        """
        Get password requirements for validation.
        
        Returns:
            Dictionary with password validation requirements
        """
        return {
            'min_length': self.password_min_length,
            'require_uppercase': self.password_require_uppercase,
            'require_lowercase': self.password_require_lowercase,
            'require_numbers': self.password_require_numbers,
            'require_special': self.password_require_special,
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """
        Get CORS configuration.
        
        Returns:
            Dictionary with CORS settings
        """
        return {
            'allow_credentials': self.cors_allow_credentials,
            'allow_methods': self.cors_allow_methods,
            'allow_headers': self.cors_allow_headers,
        }
    
    def get_security_headers(self) -> Dict[str, str]:
        """
        Get security headers configuration.
        
        Returns:
            Dictionary with security headers
        """
        if not self.enable_security_headers:
            return {}
        
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': f'max-age={self.hsts_max_age}; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Content-Security-Policy': "default-src 'self'",
        }
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """
        Get rate limiting configuration.
        
        Returns:
            Dictionary with rate limiting settings
        """
        return {
            'enabled': self.enable_rate_limiting,
            'requests': self.rate_limit_requests,
            'window_minutes': self.rate_limit_window_minutes,
        }


def generate_secret_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure secret key.
    
    Args:
        length: Length of the secret key in bytes
        
    Returns:
        URL-safe base64 encoded secret key
    """
    return secrets.token_urlsafe(length)


def validate_password_strength(password: str, requirements: Dict[str, Any]) -> List[str]:
    """
    Validate password against security requirements.
    
    Args:
        password: Password to validate
        requirements: Password requirements dictionary
        
    Returns:
        List of validation errors (empty if password is valid)
    """
    errors = []
    
    if len(password) < requirements.get('min_length', 8):
        errors.append(f"Password must be at least {requirements['min_length']} characters long")
    
    if requirements.get('require_uppercase', True) and not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if requirements.get('require_lowercase', True) and not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if requirements.get('require_numbers', True) and not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one number")
    
    if requirements.get('require_special', True):
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            errors.append("Password must contain at least one special character")
    
    return errors