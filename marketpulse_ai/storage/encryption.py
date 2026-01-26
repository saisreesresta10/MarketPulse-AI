"""
Encryption manager for MarketPulse AI.

Provides data encryption and decryption functionality for sensitive
information like pricing data and personal information.
"""

import json
import logging
from typing import Any, Optional, Union
from decimal import Decimal

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel

from ..config.security import SecurityConfig

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption/decryption operations fail."""
    pass


class EncryptionManager:
    """
    Manages data encryption and decryption operations.
    
    Provides secure encryption for sensitive data fields using
    industry-standard Fernet encryption with proper key management.
    """
    
    def __init__(self, security_config: SecurityConfig):
        """
        Initialize encryption manager with security configuration.
        
        Args:
            security_config: Security configuration with encryption settings
        """
        self.security_config = security_config
        self._cipher = None
        
        if security_config.enable_data_encryption:
            try:
                self._cipher = security_config.create_fernet_cipher()
                if self._cipher is None:
                    raise EncryptionError("Failed to create Fernet cipher")
                logger.info("Encryption manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize encryption: {e}")
                raise EncryptionError(f"Encryption initialization failed: {e}")
        else:
            logger.warning("Data encryption is disabled")
    
    def is_encryption_enabled(self) -> bool:
        """
        Check if encryption is enabled and available.
        
        Returns:
            True if encryption is enabled and cipher is available
        """
        return self._cipher is not None
    
    def encrypt_value(self, value: Any) -> str:
        """
        Encrypt a value for secure storage.
        
        Args:
            value: Value to encrypt (will be JSON serialized)
            
        Returns:
            Base64 encoded encrypted string
            
        Raises:
            EncryptionError: If encryption fails or is not available
        """
        if not self.is_encryption_enabled():
            raise EncryptionError("Encryption is not enabled or available")
        
        try:
            # Convert value to JSON string for encryption
            if isinstance(value, Decimal):
                json_str = str(value)
            elif isinstance(value, (dict, list)):
                json_str = json.dumps(value, default=str)
            else:
                json_str = json.dumps(value)
            
            # Encrypt the JSON string
            encrypted_bytes = self._cipher.encrypt(json_str.encode('utf-8'))
            encrypted_str = encrypted_bytes.decode('utf-8')
            
            logger.debug(f"Successfully encrypted value of type {type(value).__name__}")
            return encrypted_str
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt value: {e}")
    
    def decrypt_value(self, encrypted_value: str, expected_type: type = str) -> Any:
        """
        Decrypt a value from secure storage.
        
        Args:
            encrypted_value: Base64 encoded encrypted string
            expected_type: Expected type of the decrypted value
            
        Returns:
            Decrypted and deserialized value
            
        Raises:
            EncryptionError: If decryption fails or is not available
        """
        if not self.is_encryption_enabled():
            raise EncryptionError("Encryption is not enabled or available")
        
        try:
            # Decrypt the string
            decrypted_bytes = self._cipher.decrypt(encrypted_value.encode('utf-8'))
            json_str = decrypted_bytes.decode('utf-8')
            
            # Deserialize based on expected type
            if expected_type == Decimal:
                return Decimal(json_str.strip('"'))
            elif expected_type in (dict, list):
                return json.loads(json_str)
            elif expected_type == str:
                # Handle both plain strings and JSON strings
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return json_str
            else:
                # Try to deserialize as JSON first, then convert to expected type
                try:
                    deserialized = json.loads(json_str)
                    return expected_type(deserialized)
                except (json.JSONDecodeError, ValueError):
                    return expected_type(json_str)
            
        except InvalidToken:
            logger.error("Invalid encryption token - data may be corrupted")
            raise EncryptionError("Invalid encryption token - data may be corrupted")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt value: {e}")
    
    def encrypt_dict(self, data_dict: dict, fields_to_encrypt: list) -> dict:
        """
        Encrypt specific fields in a dictionary.
        
        Args:
            data_dict: Dictionary containing data to encrypt
            fields_to_encrypt: List of field names to encrypt
            
        Returns:
            Dictionary with specified fields encrypted
            
        Raises:
            EncryptionError: If encryption fails
        """
        if not self.is_encryption_enabled():
            logger.warning("Encryption not enabled - returning original data")
            return data_dict.copy()
        
        try:
            encrypted_dict = data_dict.copy()
            
            for field in fields_to_encrypt:
                if field in encrypted_dict and encrypted_dict[field] is not None:
                    encrypted_dict[f"{field}_encrypted"] = self.encrypt_value(encrypted_dict[field])
                    # Remove original unencrypted field
                    del encrypted_dict[field]
            
            logger.debug(f"Successfully encrypted {len(fields_to_encrypt)} fields")
            return encrypted_dict
            
        except Exception as e:
            logger.error(f"Dictionary encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt dictionary fields: {e}")
    
    def decrypt_dict(self, encrypted_dict: dict, fields_to_decrypt: dict) -> dict:
        """
        Decrypt specific fields in a dictionary.
        
        Args:
            encrypted_dict: Dictionary containing encrypted data
            fields_to_decrypt: Dictionary mapping encrypted field names to expected types
                              e.g., {'mrp_encrypted': Decimal, 'selling_price_encrypted': Decimal}
            
        Returns:
            Dictionary with specified fields decrypted
            
        Raises:
            EncryptionError: If decryption fails
        """
        if not self.is_encryption_enabled():
            logger.warning("Encryption not enabled - returning original data")
            return encrypted_dict.copy()
        
        try:
            decrypted_dict = encrypted_dict.copy()
            
            for encrypted_field, expected_type in fields_to_decrypt.items():
                if encrypted_field in decrypted_dict and decrypted_dict[encrypted_field] is not None:
                    # Decrypt the field
                    decrypted_value = self.decrypt_value(decrypted_dict[encrypted_field], expected_type)
                    
                    # Add decrypted field with original name (remove '_encrypted' suffix)
                    original_field = encrypted_field.replace('_encrypted', '')
                    decrypted_dict[original_field] = decrypted_value
                    
                    # Remove encrypted field
                    del decrypted_dict[encrypted_field]
            
            logger.debug(f"Successfully decrypted {len(fields_to_decrypt)} fields")
            return decrypted_dict
            
        except Exception as e:
            logger.error(f"Dictionary decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt dictionary fields: {e}")
    
    def encrypt_pydantic_model(self, model: BaseModel, fields_to_encrypt: list) -> dict:
        """
        Encrypt specific fields from a Pydantic model.
        
        Args:
            model: Pydantic model instance
            fields_to_encrypt: List of field names to encrypt
            
        Returns:
            Dictionary with model data and encrypted fields
            
        Raises:
            EncryptionError: If encryption fails
        """
        try:
            # Convert model to dictionary
            model_dict = model.model_dump()
            
            # Encrypt specified fields
            return self.encrypt_dict(model_dict, fields_to_encrypt)
            
        except Exception as e:
            logger.error(f"Pydantic model encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt Pydantic model: {e}")
    
    def decrypt_to_pydantic_model(self, encrypted_dict: dict, model_class: type, 
                                 fields_to_decrypt: dict) -> BaseModel:
        """
        Decrypt dictionary data and create Pydantic model instance.
        
        Args:
            encrypted_dict: Dictionary containing encrypted data
            model_class: Pydantic model class to create
            fields_to_decrypt: Dictionary mapping encrypted field names to expected types
            
        Returns:
            Pydantic model instance with decrypted data
            
        Raises:
            EncryptionError: If decryption or model creation fails
        """
        try:
            # Decrypt the dictionary
            decrypted_dict = self.decrypt_dict(encrypted_dict, fields_to_decrypt)
            
            # Create and return model instance
            return model_class(**decrypted_dict)
            
        except Exception as e:
            logger.error(f"Pydantic model decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt to Pydantic model: {e}")
    
    def rotate_encryption_key(self, new_key: str) -> bool:
        """
        Rotate the encryption key (for key management).
        
        Args:
            new_key: New encryption key
            
        Returns:
            True if key rotation successful
            
        Note:
            This is a placeholder for key rotation functionality.
            In production, this would require re-encrypting all existing data.
        """
        try:
            # Validate new key
            test_cipher = Fernet(new_key.encode())
            
            # In production, this would:
            # 1. Decrypt all existing data with old key
            # 2. Re-encrypt with new key
            # 3. Update configuration
            # 4. Update cipher instance
            
            logger.warning("Key rotation not fully implemented - requires data migration")
            return False
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False
    
    def get_encryption_info(self) -> dict:
        """
        Get information about encryption status and configuration.
        
        Returns:
            Dictionary with encryption information (no sensitive data)
        """
        return {
            'encryption_enabled': self.is_encryption_enabled(),
            'cipher_available': self._cipher is not None,
            'algorithm': 'Fernet (AES 128 in CBC mode)',
            'key_derivation': 'PBKDF2 with SHA256',
        }