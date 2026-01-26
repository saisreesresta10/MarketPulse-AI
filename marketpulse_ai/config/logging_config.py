"""
Logging configuration for MarketPulse AI.

Provides structured logging with audit trails, security logging,
and environment-specific log levels and formats.
"""

import logging
import logging.config
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    """
    Logging configuration with structured logging and audit support.
    
    Manages log levels, formats, handlers, and audit trail configuration
    for comprehensive application monitoring and compliance.
    """
    
    level: str = Field(default="INFO", description="Default logging level")
    format: str = Field(default="json", description="Log format (json or text)")
    enable_audit: bool = Field(default=True, description="Enable audit logging")
    enable_security_logging: bool = Field(default=True, description="Enable security event logging")
    
    # File logging settings
    log_file_path: Optional[str] = Field(default="logs/marketpulse.log", description="Main log file path")
    audit_file_path: Optional[str] = Field(default="logs/audit.log", description="Audit log file path")
    security_file_path: Optional[str] = Field(default="logs/security.log", description="Security log file path")
    max_file_size_mb: int = Field(default=10, ge=1, description="Maximum log file size in MB")
    backup_count: int = Field(default=5, ge=1, description="Number of backup log files to keep")
    
    # Console logging
    enable_console_logging: bool = Field(default=True, description="Enable console logging")
    console_level: str = Field(default="INFO", description="Console logging level")
    
    # Structured logging
    include_timestamp: bool = Field(default=True, description="Include timestamp in logs")
    include_correlation_id: bool = Field(default=True, description="Include correlation ID in logs")
    include_user_context: bool = Field(default=True, description="Include user context in logs")
    
    @field_validator('level', 'console_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level values."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('format')
    @classmethod
    def validate_log_format(cls, v):
        """Validate log format values."""
        valid_formats = ['json', 'text']
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v.lower()
    
    def create_log_directories(self):
        """Create log directories if they don't exist."""
        for path in [self.log_file_path, self.audit_file_path, self.security_file_path]:
            if path:
                log_dir = Path(path).parent
                log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get complete logging configuration dictionary.
        
        Returns:
            Dictionary compatible with logging.config.dictConfig
        """
        self.create_log_directories()
        
        formatters = self._get_formatters()
        handlers = self._get_handlers()
        loggers = self._get_loggers()
        
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'loggers': loggers,
            'root': {
                'level': self.level,
                'handlers': ['console'] if self.enable_console_logging else [],
            }
        }
    
    def _get_formatters(self) -> Dict[str, Any]:
        """Get log formatters configuration."""
        if self.format == 'json':
            return {
                'json': {
                    'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'audit_json': {
                    'format': '{"timestamp": "%(asctime)s", "event_type": "audit", "level": "%(levelname)s", "message": "%(message)s", "user_id": "%(user_id)s", "action": "%(action)s", "resource": "%(resource)s"}',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'security_json': {
                    'format': '{"timestamp": "%(asctime)s", "event_type": "security", "level": "%(levelname)s", "message": "%(message)s", "source_ip": "%(source_ip)s", "user_agent": "%(user_agent)s"}',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            }
        else:
            return {
                'text': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'audit_text': {
                    'format': '%(asctime)s - AUDIT - %(levelname)s - %(message)s - User: %(user_id)s - Action: %(action)s - Resource: %(resource)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'security_text': {
                    'format': '%(asctime)s - SECURITY - %(levelname)s - %(message)s - IP: %(source_ip)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            }
    
    def _get_handlers(self) -> Dict[str, Any]:
        """Get log handlers configuration."""
        handlers = {}
        
        # Console handler
        if self.enable_console_logging:
            formatter_key = 'json' if self.format == 'json' else 'text'
            handlers['console'] = {
                'class': 'logging.StreamHandler',
                'level': self.console_level,
                'formatter': formatter_key,
                'stream': 'ext://sys.stdout'
            }
        
        # File handler for main logs
        if self.log_file_path:
            formatter_key = 'json' if self.format == 'json' else 'text'
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': self.level,
                'formatter': formatter_key,
                'filename': self.log_file_path,
                'maxBytes': self.max_file_size_mb * 1024 * 1024,
                'backupCount': self.backup_count,
                'encoding': 'utf-8'
            }
        
        # Audit log handler
        if self.enable_audit and self.audit_file_path:
            formatter_key = 'audit_json' if self.format == 'json' else 'audit_text'
            handlers['audit'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': formatter_key,
                'filename': self.audit_file_path,
                'maxBytes': self.max_file_size_mb * 1024 * 1024,
                'backupCount': self.backup_count,
                'encoding': 'utf-8'
            }
        
        # Security log handler
        if self.enable_security_logging and self.security_file_path:
            formatter_key = 'security_json' if self.format == 'json' else 'security_text'
            handlers['security'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'WARNING',
                'formatter': formatter_key,
                'filename': self.security_file_path,
                'maxBytes': self.max_file_size_mb * 1024 * 1024,
                'backupCount': self.backup_count,
                'encoding': 'utf-8'
            }
        
        return handlers
    
    def _get_loggers(self) -> Dict[str, Any]:
        """Get loggers configuration."""
        loggers = {
            'marketpulse_ai': {
                'level': self.level,
                'handlers': ['file'] if self.log_file_path else [],
                'propagate': True
            }
        }
        
        # Audit logger
        if self.enable_audit:
            loggers['marketpulse_ai.audit'] = {
                'level': 'INFO',
                'handlers': ['audit'] if self.audit_file_path else [],
                'propagate': False
            }
        
        # Security logger
        if self.enable_security_logging:
            loggers['marketpulse_ai.security'] = {
                'level': 'WARNING',
                'handlers': ['security'] if self.security_file_path else [],
                'propagate': False
            }
        
        return loggers


class AuditLogger:
    """
    Specialized logger for audit events.
    
    Provides structured audit logging for compliance and security monitoring
    with standardized audit event formats.
    """
    
    def __init__(self, logger_name: str = 'marketpulse_ai.audit'):
        """
        Initialize audit logger.
        
        Args:
            logger_name: Name of the audit logger
        """
        self.logger = logging.getLogger(logger_name)
    
    def log_user_action(self, user_id: str, action: str, resource: str, 
                       details: Optional[Dict[str, Any]] = None, success: bool = True):
        """
        Log user action for audit trail.
        
        Args:
            user_id: User identifier
            action: Action performed
            resource: Resource affected
            details: Additional details
            success: Whether action was successful
        """
        level = logging.INFO if success else logging.WARNING
        message = f"User action: {action} on {resource}"
        
        extra = {
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        if details:
            extra.update(details)
        
        self.logger.log(level, message, extra=extra)
    
    def log_data_access(self, user_id: str, data_type: str, operation: str, 
                       record_count: Optional[int] = None):
        """
        Log data access for compliance monitoring.
        
        Args:
            user_id: User identifier
            data_type: Type of data accessed
            operation: Operation performed (read, write, delete)
            record_count: Number of records affected
        """
        message = f"Data access: {operation} {data_type}"
        
        extra = {
            'user_id': user_id,
            'action': f'data_{operation}',
            'resource': data_type,
            'record_count': record_count,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        self.logger.info(message, extra=extra)
    
    def log_system_event(self, event_type: str, description: str, 
                        severity: str = 'info', details: Optional[Dict[str, Any]] = None):
        """
        Log system events for monitoring.
        
        Args:
            event_type: Type of system event
            description: Event description
            severity: Event severity (info, warning, error)
            details: Additional event details
        """
        level_map = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }
        
        level = level_map.get(severity.lower(), logging.INFO)
        
        extra = {
            'user_id': 'system',
            'action': event_type,
            'resource': 'system',
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        if details:
            extra.update(details)
        
        self.logger.log(level, description, extra=extra)


class SecurityLogger:
    """
    Specialized logger for security events.
    
    Provides structured security logging for threat detection and
    security incident monitoring.
    """
    
    def __init__(self, logger_name: str = 'marketpulse_ai.security'):
        """
        Initialize security logger.
        
        Args:
            logger_name: Name of the security logger
        """
        self.logger = logging.getLogger(logger_name)
    
    def log_authentication_event(self, user_id: str, event_type: str, 
                                source_ip: str, user_agent: str, success: bool = True):
        """
        Log authentication events.
        
        Args:
            user_id: User identifier
            event_type: Type of authentication event (login, logout, failed_login)
            source_ip: Source IP address
            user_agent: User agent string
            success: Whether authentication was successful
        """
        level = logging.INFO if success else logging.WARNING
        message = f"Authentication {event_type}: {user_id}"
        
        extra = {
            'user_id': user_id,
            'event_type': event_type,
            'source_ip': source_ip,
            'user_agent': user_agent,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        self.logger.log(level, message, extra=extra)
    
    def log_security_violation(self, violation_type: str, description: str, 
                              source_ip: str, user_id: Optional[str] = None, 
                              severity: str = 'warning'):
        """
        Log security violations and threats.
        
        Args:
            violation_type: Type of security violation
            description: Violation description
            source_ip: Source IP address
            user_id: User identifier if known
            severity: Violation severity
        """
        level_map = {
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }
        
        level = level_map.get(severity.lower(), logging.WARNING)
        
        extra = {
            'user_id': user_id or 'unknown',
            'violation_type': violation_type,
            'source_ip': source_ip,
            'user_agent': '',
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        self.logger.log(level, f"Security violation: {description}", extra=extra)