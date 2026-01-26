"""
Storage manager for MarketPulse AI.

Provides high-level storage operations with encryption, caching,
and audit logging for all data persistence needs.
"""

import json
import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc, asc, func

from ..config.database import DatabaseManager
from ..config.security import SecurityConfig
from ..core.models import (
    SalesDataPoint, DemandPattern, ExplainableInsight, 
    RiskAssessment, Scenario, ComplianceResult
)
from .models import (
    SalesDataModel, DemandPatternModel, InsightModel,
    RiskAssessmentModel, ScenarioModel, ComplianceResultModel,
    CacheEntryModel, AuditLogModel
)
from .encryption import EncryptionManager, EncryptionError

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operations fail."""
    pass


class StorageManager:
    """
    High-level storage manager for MarketPulse AI.
    
    Provides encrypted storage, caching, and retrieval operations
    with comprehensive audit logging and data lifecycle management.
    """
    
    def __init__(self, db_manager: DatabaseManager, security_config: SecurityConfig):
        """
        Initialize storage manager with database and security configuration.
        
        Args:
            db_manager: Database manager instance
            security_config: Security configuration for encryption
        """
        self.db_manager = db_manager
        self.security_config = security_config
        self.encryption_manager = EncryptionManager(security_config)
        
        # Cache configuration
        self.cache_ttl_hours = 24  # Default cache TTL
        self.max_cache_entries = 10000  # Maximum cache entries
        
        logger.info("Storage manager initialized successfully")
    
    def _get_session(self) -> Session:
        """Get database session with proper error handling."""
        try:
            return next(self.db_manager.get_session())
        except Exception as e:
            logger.error(f"Failed to get database session: {e}")
            raise StorageError(f"Database connection failed: {e}")
    
    def _log_operation(self, session: Session, operation_type: str, table_name: str,
                      record_id: Optional[str] = None, success: bool = True,
                      error_message: Optional[str] = None, user_id: Optional[str] = None,
                      operation_details: Optional[Dict[str, Any]] = None):
        """
        Log database operation for audit trail.
        
        Args:
            session: Database session
            operation_type: Type of operation (CREATE, READ, UPDATE, DELETE)
            table_name: Name of affected table
            record_id: ID of affected record
            success: Whether operation was successful
            error_message: Error message if operation failed
            user_id: User performing the operation
            operation_details: Additional operation context
        """
        try:
            audit_log = AuditLogModel(
                operation_type=operation_type,
                table_name=table_name,
                record_id=record_id,
                user_id=user_id,
                success=success,
                error_message=error_message,
                operation_details=operation_details,
                timestamp=datetime.utcnow()
            )
            session.add(audit_log)
            # Note: Don't commit here - let the calling method handle transaction
            
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
            # Don't raise exception for audit logging failures
    
    # Sales Data Operations
    
    async def store_sales_data(self, sales_data: List[SalesDataPoint], 
                              user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store sales data with encryption for sensitive fields.
        
        Args:
            sales_data: List of sales data points to store
            user_id: User performing the operation
            
        Returns:
            Dictionary with storage results and statistics
            
        Raises:
            StorageError: If storage operation fails
        """
        session = self._get_session()
        
        try:
            stored_count = 0
            failed_count = 0
            errors = []
            
            for data_point in sales_data:
                try:
                    # Encrypt sensitive fields
                    encrypted_data = self.encryption_manager.encrypt_pydantic_model(
                        data_point, ['mrp', 'selling_price']
                    )
                    
                    # Create database model
                    db_model = SalesDataModel(
                        id=str(data_point.id),
                        product_id=data_point.product_id,
                        product_name=data_point.product_name,
                        category=data_point.category,
                        mrp_encrypted=encrypted_data.get('mrp_encrypted'),
                        selling_price_encrypted=encrypted_data.get('selling_price_encrypted'),
                        quantity_sold=data_point.quantity_sold,
                        sale_date=data_point.sale_date,
                        store_location=data_point.store_location,
                        seasonal_event=data_point.seasonal_event,
                        created_at=data_point.created_at
                    )
                    
                    session.add(db_model)
                    stored_count += 1
                    
                    # Log operation
                    self._log_operation(
                        session, 'CREATE', 'sales_data', str(data_point.id),
                        success=True, user_id=user_id,
                        operation_details={'product_id': data_point.product_id}
                    )
                    
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Failed to store sales data {data_point.id}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    
                    # Log failed operation
                    self._log_operation(
                        session, 'CREATE', 'sales_data', str(data_point.id),
                        success=False, error_message=str(e), user_id=user_id
                    )
            
            # Commit transaction
            session.commit()
            
            result = {
                'status': 'success' if failed_count == 0 else 'partial_success',
                'stored_count': stored_count,
                'failed_count': failed_count,
                'total_count': len(sales_data),
                'errors': errors,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Stored {stored_count}/{len(sales_data)} sales data points")
            return result
            
        except Exception as e:
            session.rollback()
            logger.error(f"Sales data storage failed: {e}")
            raise StorageError(f"Failed to store sales data: {e}")
        finally:
            session.close()
    
    async def retrieve_sales_data(self, product_ids: Optional[List[str]] = None,
                                 date_range: Optional[Tuple[date, date]] = None,
                                 categories: Optional[List[str]] = None,
                                 limit: Optional[int] = None,
                                 user_id: Optional[str] = None) -> List[SalesDataPoint]:
        """
        Retrieve sales data with decryption and filtering.
        
        Args:
            product_ids: Optional list of product IDs to filter
            date_range: Optional tuple of (start_date, end_date)
            categories: Optional list of categories to filter
            limit: Optional limit on number of records
            user_id: User performing the operation
            
        Returns:
            List of decrypted sales data points
            
        Raises:
            StorageError: If retrieval operation fails
        """
        session = self._get_session()
        
        try:
            # Build query
            query = session.query(SalesDataModel)
            
            # Apply filters
            if product_ids:
                query = query.filter(SalesDataModel.product_id.in_(product_ids))
            
            if date_range:
                start_date, end_date = date_range
                query = query.filter(
                    and_(
                        SalesDataModel.sale_date >= start_date,
                        SalesDataModel.sale_date <= end_date
                    )
                )
            
            if categories:
                query = query.filter(SalesDataModel.category.in_(categories))
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Order by date (most recent first)
            query = query.order_by(desc(SalesDataModel.sale_date))
            
            # Execute query
            db_results = query.all()
            
            # Decrypt and convert to Pydantic models
            sales_data = []
            for db_model in db_results:
                try:
                    # Prepare encrypted data for decryption
                    encrypted_data = {
                        'id': db_model.id,
                        'product_id': db_model.product_id,
                        'product_name': db_model.product_name,
                        'category': db_model.category,
                        'mrp_encrypted': db_model.mrp_encrypted,
                        'selling_price_encrypted': db_model.selling_price_encrypted,
                        'quantity_sold': db_model.quantity_sold,
                        'sale_date': db_model.sale_date,
                        'store_location': db_model.store_location,
                        'seasonal_event': db_model.seasonal_event,
                        'created_at': db_model.created_at
                    }
                    
                    # Decrypt sensitive fields
                    decrypted_data = self.encryption_manager.decrypt_dict(
                        encrypted_data,
                        {'mrp_encrypted': Decimal, 'selling_price_encrypted': Decimal}
                    )
                    
                    # Create Pydantic model
                    sales_point = SalesDataPoint(**decrypted_data)
                    sales_data.append(sales_point)
                    
                except Exception as e:
                    logger.error(f"Failed to decrypt sales data {db_model.id}: {e}")
                    continue
            
            # Log operation
            self._log_operation(
                session, 'READ', 'sales_data', success=True, user_id=user_id,
                operation_details={
                    'filter_count': len(db_results),
                    'product_ids': product_ids[:5] if product_ids else None,  # Log first 5
                    'date_range': [str(date_range[0]), str(date_range[1])] if date_range else None
                }
            )
            
            logger.info(f"Retrieved {len(sales_data)} sales data points")
            return sales_data
            
        except Exception as e:
            logger.error(f"Sales data retrieval failed: {e}")
            raise StorageError(f"Failed to retrieve sales data: {e}")
        finally:
            session.close()
    
    # Pattern Storage Operations
    
    async def store_patterns(self, patterns: List[DemandPattern], 
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Store demand patterns with caching.
        
        Args:
            patterns: List of demand patterns to store
            user_id: User performing the operation
            
        Returns:
            Dictionary with storage results
            
        Raises:
            StorageError: If storage operation fails
        """
        session = self._get_session()
        
        try:
            stored_count = 0
            failed_count = 0
            errors = []
            
            for pattern in patterns:
                try:
                    # Create database model
                    db_model = DemandPatternModel(
                        id=str(pattern.id),
                        product_id=pattern.product_id,
                        pattern_type=pattern.pattern_type,
                        description=pattern.description,
                        confidence_level=pattern.confidence_level.value,
                        seasonal_factors=pattern.seasonal_factors,
                        trend_direction=pattern.trend_direction,
                        volatility_score=pattern.volatility_score,
                        supporting_data_points=pattern.supporting_data_points,
                        date_range_start=pattern.date_range_start,
                        date_range_end=pattern.date_range_end,
                        created_at=pattern.created_at
                    )
                    
                    session.add(db_model)
                    stored_count += 1
                    
                    # Cache the pattern
                    await self._cache_pattern(pattern)
                    
                    # Log operation
                    self._log_operation(
                        session, 'CREATE', 'demand_patterns', str(pattern.id),
                        success=True, user_id=user_id,
                        operation_details={'product_id': pattern.product_id, 'pattern_type': pattern.pattern_type}
                    )
                    
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Failed to store pattern {pattern.id}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    
                    # Log failed operation
                    self._log_operation(
                        session, 'CREATE', 'demand_patterns', str(pattern.id),
                        success=False, error_message=str(e), user_id=user_id
                    )
            
            # Commit transaction
            session.commit()
            
            result = {
                'status': 'success' if failed_count == 0 else 'partial_success',
                'stored_count': stored_count,
                'failed_count': failed_count,
                'total_count': len(patterns),
                'errors': errors,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Stored {stored_count}/{len(patterns)} demand patterns")
            return result
            
        except Exception as e:
            session.rollback()
            logger.error(f"Pattern storage failed: {e}")
            raise StorageError(f"Failed to store patterns: {e}")
        finally:
            session.close()
    
    async def retrieve_patterns(self, product_ids: Optional[List[str]] = None,
                              pattern_types: Optional[List[str]] = None,
                              confidence_levels: Optional[List[str]] = None,
                              date_range: Optional[Tuple[date, date]] = None,
                              use_cache: bool = True,
                              user_id: Optional[str] = None) -> List[DemandPattern]:
        """
        Retrieve demand patterns with caching support.
        
        Args:
            product_ids: Optional list of product IDs to filter
            pattern_types: Optional list of pattern types to filter
            confidence_levels: Optional list of confidence levels to filter
            date_range: Optional tuple of (start_date, end_date)
            use_cache: Whether to use cached results
            user_id: User performing the operation
            
        Returns:
            List of demand patterns
            
        Raises:
            StorageError: If retrieval operation fails
        """
        # Try cache first if enabled
        if use_cache:
            cache_key = self._generate_cache_key('patterns', {
                'product_ids': product_ids,
                'pattern_types': pattern_types,
                'confidence_levels': confidence_levels,
                'date_range': date_range
            })
            
            cached_patterns = await self._get_cached_patterns(cache_key)
            if cached_patterns:
                logger.info(f"Retrieved {len(cached_patterns)} patterns from cache")
                return cached_patterns
        
        session = self._get_session()
        
        try:
            # Build query
            query = session.query(DemandPatternModel)
            
            # Apply filters
            if product_ids:
                query = query.filter(DemandPatternModel.product_id.in_(product_ids))
            
            if pattern_types:
                query = query.filter(DemandPatternModel.pattern_type.in_(pattern_types))
            
            if confidence_levels:
                query = query.filter(DemandPatternModel.confidence_level.in_(confidence_levels))
            
            if date_range:
                start_date, end_date = date_range
                query = query.filter(
                    and_(
                        DemandPatternModel.date_range_start >= start_date,
                        DemandPatternModel.date_range_end <= end_date
                    )
                )
            
            # Order by creation date (most recent first)
            query = query.order_by(desc(DemandPatternModel.created_at))
            
            # Execute query
            db_results = query.all()
            
            # Convert to Pydantic models
            patterns = []
            for db_model in db_results:
                try:
                    pattern = DemandPattern(
                        id=UUID(db_model.id),
                        product_id=db_model.product_id,
                        pattern_type=db_model.pattern_type,
                        description=db_model.description,
                        confidence_level=db_model.confidence_level,
                        seasonal_factors=db_model.seasonal_factors or {},
                        trend_direction=db_model.trend_direction,
                        volatility_score=db_model.volatility_score,
                        supporting_data_points=db_model.supporting_data_points,
                        date_range_start=db_model.date_range_start,
                        date_range_end=db_model.date_range_end,
                        created_at=db_model.created_at
                    )
                    patterns.append(pattern)
                    
                except Exception as e:
                    logger.error(f"Failed to convert pattern {db_model.id}: {e}")
                    continue
            
            # Cache results if caching is enabled
            if use_cache and patterns:
                await self._cache_patterns(cache_key, patterns)
            
            # Log operation
            self._log_operation(
                session, 'READ', 'demand_patterns', success=True, user_id=user_id,
                operation_details={
                    'result_count': len(patterns),
                    'product_ids': product_ids[:5] if product_ids else None,
                    'pattern_types': pattern_types
                }
            )
            
            logger.info(f"Retrieved {len(patterns)} demand patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern retrieval failed: {e}")
            raise StorageError(f"Failed to retrieve patterns: {e}")
        finally:
            session.close()
    
    # Caching Operations
    
    def _generate_cache_key(self, cache_type: str, parameters: Dict[str, Any]) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            cache_type: Type of cached data
            parameters: Parameters used for filtering
            
        Returns:
            Generated cache key
        """
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(parameters, sort_keys=True, default=str)
        return f"{cache_type}:{hash(sorted_params)}"
    
    async def _cache_pattern(self, pattern: DemandPattern) -> bool:
        """
        Cache a single pattern.
        
        Args:
            pattern: Pattern to cache
            
        Returns:
            True if caching successful
        """
        try:
            cache_key = f"pattern:{pattern.id}"
            encrypted_data = self.encryption_manager.encrypt_value(pattern.model_dump())
            
            session = self._get_session()
            try:
                cache_entry = CacheEntryModel(
                    cache_key=cache_key,
                    cache_type='pattern',
                    data_encrypted=encrypted_data,
                    expires_at=datetime.utcnow() + timedelta(hours=self.cache_ttl_hours)
                )
                
                session.merge(cache_entry)  # Use merge to handle duplicates
                session.commit()
                return True
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to cache pattern {pattern.id}: {e}")
            return False
    
    async def _cache_patterns(self, cache_key: str, patterns: List[DemandPattern]) -> bool:
        """
        Cache a list of patterns.
        
        Args:
            cache_key: Cache key for the pattern list
            patterns: List of patterns to cache
            
        Returns:
            True if caching successful
        """
        try:
            patterns_data = [pattern.model_dump() for pattern in patterns]
            encrypted_data = self.encryption_manager.encrypt_value(patterns_data)
            
            session = self._get_session()
            try:
                cache_entry = CacheEntryModel(
                    cache_key=cache_key,
                    cache_type='patterns',
                    data_encrypted=encrypted_data,
                    expires_at=datetime.utcnow() + timedelta(hours=self.cache_ttl_hours)
                )
                
                session.merge(cache_entry)
                session.commit()
                return True
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to cache patterns: {e}")
            return False
    
    async def _get_cached_patterns(self, cache_key: str) -> Optional[List[DemandPattern]]:
        """
        Retrieve patterns from cache.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            List of cached patterns or None if not found/expired
        """
        session = self._get_session()
        
        try:
            cache_entry = session.query(CacheEntryModel).filter(
                and_(
                    CacheEntryModel.cache_key == cache_key,
                    CacheEntryModel.expires_at > datetime.utcnow()
                )
            ).first()
            
            if not cache_entry:
                return None
            
            # Update access statistics
            cache_entry.access_count += 1
            cache_entry.last_accessed = datetime.utcnow()
            session.commit()
            
            # Decrypt and deserialize patterns
            patterns_data = self.encryption_manager.decrypt_value(cache_entry.data_encrypted, list)
            patterns = [DemandPattern(**pattern_dict) for pattern_dict in patterns_data]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached patterns: {e}")
            return None
        finally:
            session.close()
    
    # Cache Management
    
    async def cleanup_expired_cache(self) -> Dict[str, int]:
        """
        Clean up expired cache entries.
        
        Returns:
            Dictionary with cleanup statistics
        """
        session = self._get_session()
        
        try:
            # Delete expired entries
            deleted_count = session.query(CacheEntryModel).filter(
                CacheEntryModel.expires_at <= datetime.utcnow()
            ).delete()
            
            session.commit()
            
            logger.info(f"Cleaned up {deleted_count} expired cache entries")
            return {'deleted_count': deleted_count}
            
        except Exception as e:
            session.rollback()
            logger.error(f"Cache cleanup failed: {e}")
            return {'deleted_count': 0, 'error': str(e)}
        finally:
            session.close()
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics and health information.
        
        Returns:
            Dictionary with storage statistics
        """
        session = self._get_session()
        
        try:
            stats = {}
            
            # Sales data statistics
            sales_count = session.query(func.count(SalesDataModel.id)).scalar()
            stats['sales_data_count'] = sales_count
            
            # Pattern statistics
            pattern_count = session.query(func.count(DemandPatternModel.id)).scalar()
            stats['pattern_count'] = pattern_count
            
            # Cache statistics
            cache_count = session.query(func.count(CacheEntryModel.cache_key)).scalar()
            expired_cache_count = session.query(func.count(CacheEntryModel.cache_key)).filter(
                CacheEntryModel.expires_at <= datetime.utcnow()
            ).scalar()
            
            stats['cache_entries'] = cache_count
            stats['expired_cache_entries'] = expired_cache_count
            
            # Audit log statistics
            audit_count = session.query(func.count(AuditLogModel.id)).scalar()
            stats['audit_log_count'] = audit_count
            
            # Encryption information
            stats['encryption_info'] = self.encryption_manager.get_encryption_info()
            
            # Database health
            stats['database_health'] = self.db_manager.check_connection()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    # Data Lifecycle Management
    
    async def archive_old_data(self, retention_days: int = 365) -> Dict[str, int]:
        """
        Archive old data based on retention policy.
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            Dictionary with archival statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        session = self._get_session()
        
        try:
            # Archive old sales data (in production, this would move to archive storage)
            old_sales_count = session.query(func.count(SalesDataModel.id)).filter(
                SalesDataModel.created_at < cutoff_date
            ).scalar()
            
            # Archive old patterns
            old_patterns_count = session.query(func.count(DemandPatternModel.id)).filter(
                DemandPatternModel.created_at < cutoff_date
            ).scalar()
            
            # For now, just return counts (actual archival would be implemented based on requirements)
            logger.info(f"Found {old_sales_count} old sales records and {old_patterns_count} old patterns for archival")
            
            return {
                'old_sales_data': old_sales_count,
                'old_patterns': old_patterns_count,
                'cutoff_date': cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data archival check failed: {e}")
            return {'error': str(e)}
        finally:
            session.close()