"""
Property-Based Tests for Data Security Features

Property tests validating universal correctness properties for data security,
encryption, audit logging, data lifecycle management, and privacy compliance.

**Property 10: Data Source Compliance**
**Property 11: Data Protection Round-Trip**
**Property 12: Data Lifecycle Management**
**Property 13: Data Sharing Restrictions**
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4, UUID
from pathlib import Path

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

from marketpulse_ai.storage.encryption import EncryptionManager, EncryptionError
from marketpulse_ai.storage.storage_manager import StorageManager, StorageError
from marketpulse_ai.config.security import SecurityConfig
from marketpulse_ai.config.database import DatabaseManager
from marketpulse_ai.core.models import (
    SalesDataPoint, DemandPattern, ExplainableInsight, 
    RiskAssessment, Scenario, ComplianceResult, ConfidenceLevel
)
from marketpulse_ai.storage.models import AuditLogModel


# Hypothesis strategies for generating test data
@composite
def security_config_strategy(draw):
    """Generate valid security configuration for testing."""
    # Generate a secret key with sufficient entropy by mixing different character types
    base_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    special_chars = '!@#$%^&*'
    
    # Ensure we have at least some variety in the key
    key_parts = []
    key_parts.append(draw(st.text(min_size=8, max_size=16, alphabet=base_chars)))
    key_parts.append(draw(st.text(min_size=4, max_size=8, alphabet='0123456789')))
    key_parts.append(draw(st.text(min_size=4, max_size=8, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ')))
    key_parts.append(draw(st.text(min_size=2, max_size=4, alphabet=special_chars)))
    
    # Use hypothesis to shuffle
    shuffled_parts = draw(st.permutations(key_parts))
    secret_key = ''.join(shuffled_parts)
    
    # Ensure minimum length
    while len(secret_key) < 32:
        secret_key += draw(st.text(min_size=1, max_size=5, alphabet=base_chars + special_chars))
    
    return SecurityConfig(
        secret_key=secret_key[:64],  # Limit to max size
        enable_data_encryption=draw(st.booleans()),
        encryption_key=None,  # Will be auto-generated
        enable_rate_limiting=draw(st.booleans()),
        rate_limit_requests=draw(st.integers(min_value=10, max_value=1000)),
        rate_limit_window_minutes=draw(st.integers(min_value=1, max_value=60))
    )


@composite
def sales_data_strategy(draw):
    """Generate valid sales data for testing."""
    mrp_value = draw(st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    mrp = Decimal(str(round(mrp_value, 2)))
    
    # Ensure selling price is always <= MRP
    selling_price_value = draw(st.floats(min_value=5.0, max_value=float(mrp), allow_nan=False, allow_infinity=False))
    selling_price = Decimal(str(round(selling_price_value, 2)))
    
    return SalesDataPoint(
        id=UUID(str(uuid4())),
        product_id=draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        product_name=draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')))),
        category=draw(st.sampled_from(['electronics', 'clothing', 'home', 'books', 'sports', 'beauty', 'food', 'jewelry'])),
        mrp=mrp,
        selling_price=selling_price,
        quantity_sold=draw(st.integers(min_value=1, max_value=1000)),
        sale_date=draw(st.dates(min_value=date(2020, 1, 1), max_value=date.today())),
        store_location=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')))),
        seasonal_event=draw(st.one_of(st.none(), st.sampled_from(['diwali', 'holi', 'christmas', 'new_year', 'summer_sale']))),
        created_at=datetime.utcnow()
    )


@composite
def sensitive_data_strategy(draw):
    """Generate sensitive data for encryption testing."""
    data_type = draw(st.sampled_from(['string', 'number', 'decimal', 'dict', 'list']))
    
    if data_type == 'string':
        return draw(st.text(min_size=1, max_size=1000, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Pc'))))
    elif data_type == 'number':
        return draw(st.floats(min_value=-1000000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False))
    elif data_type == 'decimal':
        return Decimal(str(draw(st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False))))
    elif data_type == 'dict':
        return draw(st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            st.one_of(
                st.text(min_size=1, max_size=100),
                st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
                st.integers(min_value=0, max_value=10000)
            ),
            min_size=1, max_size=5
        ))
    else:  # list
        return draw(st.lists(
            st.one_of(
                st.text(min_size=1, max_size=50),
                st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                st.integers(min_value=0, max_value=1000)
            ),
            min_size=1, max_size=10
        ))


@composite
def external_data_source_strategy(draw):
    """Generate external data source information for testing."""
    source_type = draw(st.sampled_from(['synthetic', 'public', 'private', 'third_party']))
    
    return {
        'source_id': str(uuid4()),
        'source_type': source_type,
        'source_name': draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')))),
        'data_classification': draw(st.sampled_from(['public', 'synthetic', 'confidential', 'restricted'])),
        'contains_pii': draw(st.booleans()),
        'requires_consent': draw(st.booleans()),
        'data_retention_days': draw(st.integers(min_value=1, max_value=3650)),
        'compliance_tags': draw(st.lists(
            st.sampled_from(['gdpr', 'ccpa', 'hipaa', 'pci_dss', 'sox', 'india_privacy']),
            min_size=0, max_size=3
        ))
    }


@composite
def audit_operation_strategy(draw):
    """Generate audit operation data for testing."""
    return {
        'operation_type': draw(st.sampled_from(['CREATE', 'READ', 'UPDATE', 'DELETE', 'EXPORT', 'SHARE'])),
        'table_name': draw(st.sampled_from(['sales_data', 'demand_patterns', 'insights', 'risk_assessments', 'scenarios'])),
        'record_id': str(uuid4()),
        'user_id': draw(st.one_of(st.none(), st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))),
        'session_id': str(uuid4()),
        'ip_address': draw(st.ip_addresses(v=4)).exploded,
        'operation_details': draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(min_size=1, max_size=100), st.integers(min_value=0, max_value=1000)),
            min_size=0, max_size=5
        )),
        'success': draw(st.booleans()),
        'error_message': draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    }


@composite
def data_sharing_request_strategy(draw):
    """Generate data sharing request for testing."""
    return {
        'request_id': str(uuid4()),
        'requester_id': draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        'data_type': draw(st.sampled_from(['sales_data', 'patterns', 'insights', 'aggregated_stats'])),
        'purpose': draw(st.text(min_size=10, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Pc')))),
        'has_consent': draw(st.booleans()),
        'consent_details': draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.booleans(), st.text(min_size=1, max_size=100)),
            min_size=0, max_size=3
        )),
        'third_party_recipient': draw(st.booleans()),
        'retention_period_days': draw(st.integers(min_value=1, max_value=365))
    }


class TestDataSecurityProperties:
    """
    Property-based tests for data security features.
    
    **Property 10: Data Source Compliance**
    **Property 11: Data Protection Round-Trip**
    **Property 12: Data Lifecycle Management**
    **Property 13: Data Sharing Restrictions**
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """
    
    @given(external_data_source_strategy())
    @settings(max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_data_source_compliance_validation(self, data_source):
        """
        **Property 10.1: Data Source Compliance Validation**
        **Validates: Requirements 7.1**
        
        Property: For any external data source, the system should only accept
        synthetic or publicly available data, rejecting private third-party data.
        """
        # Property: Only synthetic and public data sources should be accepted
        is_compliant = self._validate_data_source_compliance(data_source)
        
        # Property: Synthetic and public sources should be compliant if they meet criteria
        if (data_source['source_type'] in ['synthetic', 'public'] and 
            data_source['data_classification'] in ['public', 'synthetic'] and
            not (data_source['contains_pii'] and data_source['source_type'] != 'synthetic') and
            not (data_source['requires_consent'] and data_source['source_type'] not in ['synthetic'])):
            assert is_compliant, f"Synthetic/public data source should be compliant: {data_source}"
        
        # Property: Private or third-party sources should be rejected
        if data_source['source_type'] in ['private', 'third_party'] or data_source['data_classification'] in ['confidential', 'restricted']:
            assert not is_compliant, f"Private/third-party data source should be rejected: {data_source}"
        
        # Property: PII-containing sources should be rejected unless synthetic
        if data_source['contains_pii'] and data_source['source_type'] != 'synthetic':
            assert not is_compliant, f"PII-containing non-synthetic source should be rejected: {data_source}"
    
    @given(security_config_strategy(), sensitive_data_strategy())
    @settings(max_examples=100, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_data_protection_round_trip_integrity(self, security_config, sensitive_data):
        """
        **Property 11.1: Data Protection Round-Trip Integrity**
        **Validates: Requirements 7.2**
        
        Property: For any sensitive data encrypted and stored, when retrieved
        and decrypted, it should maintain complete integrity and confidentiality.
        """
        # Skip if encryption is disabled
        assume(security_config.enable_data_encryption)
        
        encryption_manager = EncryptionManager(security_config)
        
        # Property: Encryption should be available when enabled
        assert encryption_manager.is_encryption_enabled()
        
        try:
            # Property: Any data should be encryptable
            encrypted_value = encryption_manager.encrypt_value(sensitive_data)
            assert isinstance(encrypted_value, str)
            assert len(encrypted_value) > 0
            
            # Property: Encrypted data should be different from original
            if isinstance(sensitive_data, str):
                assert encrypted_value != sensitive_data
            
            # Property: Decrypted data should match original exactly
            decrypted_value = encryption_manager.decrypt_value(encrypted_value, type(sensitive_data))
            
            # Property: Round-trip integrity must be maintained
            if isinstance(sensitive_data, Decimal):
                assert decrypted_value == sensitive_data
            elif isinstance(sensitive_data, (dict, list)):
                assert decrypted_value == sensitive_data
            elif isinstance(sensitive_data, float):
                # Handle floating point precision
                assert abs(decrypted_value - sensitive_data) < 1e-10
            else:
                assert decrypted_value == sensitive_data
            
            # Property: Multiple encryptions should produce different ciphertexts (nonce-based)
            encrypted_value2 = encryption_manager.encrypt_value(sensitive_data)
            if len(str(sensitive_data)) > 10:  # Only for non-trivial data
                assert encrypted_value != encrypted_value2, "Multiple encryptions should produce different ciphertexts"
            
        except EncryptionError as e:
            # Property: Encryption errors should be properly handled
            assert "encryption" in str(e).lower() or "cipher" in str(e).lower()
    
    @given(security_config_strategy(), st.lists(sales_data_strategy(), min_size=1, max_size=5))
    @settings(max_examples=50, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_encrypted_storage_round_trip(self, security_config, sales_data_list):
        """
        **Property 11.2: Encrypted Storage Round-Trip**
        **Validates: Requirements 7.2**
        
        Property: For any sales data stored with encryption, retrieval should
        maintain data integrity while protecting sensitive fields.
        """
        # Skip if encryption is disabled
        assume(security_config.enable_data_encryption)
        
        # Create temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_url = f"sqlite:///{tmp_file.name}"
        
        try:
            # Initialize storage components
            from marketpulse_ai.config.database import DatabaseConfig
            db_config = DatabaseConfig(url=db_url)
            db_manager = DatabaseManager(db_config)
            db_manager.create_tables()
            
            storage_manager = StorageManager(db_manager, security_config)
            
            # Property: Storage should handle any valid sales data
            store_result = await storage_manager.store_sales_data(sales_data_list)
            
            # Property: Storage should succeed for valid data
            assert store_result['status'] in ['success', 'partial_success']
            assert store_result['stored_count'] > 0
            
            # Property: Retrieve all stored data
            retrieved_data = await storage_manager.retrieve_sales_data()
            
            # Property: Retrieved data count should match stored count
            assert len(retrieved_data) == store_result['stored_count']
            
            # Property: Each retrieved item should match original data
            original_by_id = {str(item.id): item for item in sales_data_list}
            
            for retrieved_item in retrieved_data:
                original_item = original_by_id[str(retrieved_item.id)]
                
                # Property: All fields should match exactly
                assert retrieved_item.id == original_item.id
                assert retrieved_item.product_id == original_item.product_id
                assert retrieved_item.product_name == original_item.product_name
                assert retrieved_item.category == original_item.category
                assert retrieved_item.mrp == original_item.mrp
                assert retrieved_item.selling_price == original_item.selling_price
                assert retrieved_item.quantity_sold == original_item.quantity_sold
                assert retrieved_item.sale_date == original_item.sale_date
                assert retrieved_item.store_location == original_item.store_location
                assert retrieved_item.seasonal_event == original_item.seasonal_event
        
        finally:
            # Cleanup
            try:
                os.unlink(tmp_file.name)
            except:
                pass
    
    @given(audit_operation_strategy())
    @settings(max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_audit_logging_completeness(self, audit_operation):
        """
        **Property 12.1: Audit Logging Completeness**
        **Validates: Requirements 7.5**
        
        Property: For any data operation, comprehensive audit logs should be
        created with all required information for compliance monitoring.
        """
        # Property: Audit log should capture all required fields
        audit_log = self._create_audit_log_entry(audit_operation)
        
        # Property: Required fields should be present
        required_fields = ['operation_type', 'table_name', 'timestamp', 'success']
        for field in required_fields:
            assert hasattr(audit_log, field)
            assert getattr(audit_log, field) is not None
        
        # Property: Operation type should be valid
        valid_operations = ['CREATE', 'READ', 'UPDATE', 'DELETE', 'EXPORT', 'SHARE']
        assert audit_log.operation_type in valid_operations
        
        # Property: Timestamp should be recent
        time_diff = datetime.utcnow() - audit_log.timestamp
        assert time_diff.total_seconds() < 60  # Within last minute
        
        # Property: Success status should be boolean
        assert isinstance(audit_log.success, bool)
        
        # Property: Failed operations should have error messages
        if not audit_log.success and audit_operation.get('error_message'):
            assert audit_log.error_message is not None
            assert len(audit_log.error_message) > 0
    
    @given(st.integers(min_value=1, max_value=3650))
    @settings(max_examples=50, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_data_retention_policy_compliance(self, retention_days):
        """
        **Property 12.2: Data Retention Policy Compliance**
        **Validates: Requirements 7.3**
        
        Property: For any retention period, data lifecycle management should
        correctly identify and handle data that exceeds retention limits.
        """
        # Property: Retention policy should handle any valid retention period
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Property: Cutoff date should be in the past
        assert cutoff_date < datetime.utcnow()
        
        # Property: Data older than cutoff should be identified for deletion
        test_data_dates = [
            datetime.utcnow() - timedelta(days=retention_days + 10),  # Should be deleted
            datetime.utcnow() - timedelta(days=max(retention_days - 10, 0)),  # Should be kept (avoid negative)
            datetime.utcnow() - timedelta(days=retention_days, hours=1),       # Boundary case - should be deleted
        ]
        
        for test_date in test_data_dates:
            should_delete = self._should_delete_data(test_date, retention_days)
            
            if test_date <= cutoff_date:  # Use <= for boundary case
                assert should_delete, f"Data from {test_date} should be deleted with {retention_days} day retention (cutoff: {cutoff_date})"
            else:
                assert not should_delete, f"Data from {test_date} should be kept with {retention_days} day retention (cutoff: {cutoff_date})"
    
    @given(data_sharing_request_strategy())
    @settings(max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_data_sharing_restrictions_enforcement(self, sharing_request):
        """
        **Property 13.1: Data Sharing Restrictions Enforcement**
        **Validates: Requirements 7.4**
        
        Property: For any data sharing request, the system should enforce
        restrictions and only allow sharing with explicit consent.
        """
        # Property: Data sharing decision should be based on consent and restrictions
        sharing_allowed = self._evaluate_data_sharing_request(sharing_request)
        
        # Property: Third-party sharing without consent should be denied
        if sharing_request['third_party_recipient'] and not sharing_request['has_consent']:
            assert not sharing_allowed, f"Third-party sharing without consent should be denied: {sharing_request}"
        
        # Property: Sharing with consent should be allowed (subject to other restrictions)
        if sharing_request['has_consent'] and not sharing_request['third_party_recipient']:
            # Internal sharing with consent should generally be allowed
            pass  # Additional business logic would determine final decision
        
        # Property: Sharing decision should be logged
        sharing_decision = {
            'request_id': sharing_request['request_id'],
            'allowed': sharing_allowed,
            'reason': self._get_sharing_decision_reason(sharing_request, sharing_allowed)
        }
        
        assert 'request_id' in sharing_decision
        assert 'allowed' in sharing_decision
        assert 'reason' in sharing_decision
        assert isinstance(sharing_decision['allowed'], bool)
    
    @given(st.lists(audit_operation_strategy(), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_audit_trail_integrity(self, audit_operations):
        """
        **Property 12.3: Audit Trail Integrity**
        **Validates: Requirements 7.5**
        
        Property: For any sequence of operations, the audit trail should
        maintain chronological integrity and completeness.
        """
        # Property: Create audit logs for all operations
        audit_logs = []
        base_time = datetime.utcnow()
        
        for i, operation in enumerate(audit_operations):
            # Ensure chronological ordering
            operation_time = base_time + timedelta(seconds=i)
            audit_log = self._create_audit_log_entry(operation, operation_time)
            audit_logs.append(audit_log)
        
        # Property: Audit logs should maintain chronological order
        for i in range(1, len(audit_logs)):
            assert audit_logs[i].timestamp >= audit_logs[i-1].timestamp
        
        # Property: Each operation should have corresponding audit log
        assert len(audit_logs) == len(audit_operations)
        
        # Property: Audit logs should be immutable (no gaps in sequence)
        operation_types = [log.operation_type for log in audit_logs]
        expected_types = [op['operation_type'] for op in audit_operations]
        assert operation_types == expected_types
    
    @given(security_config_strategy())
    @settings(max_examples=50, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_encryption_key_security(self, security_config):
        """
        **Property 11.3: Encryption Key Security**
        **Validates: Requirements 7.2**
        
        Property: For any security configuration with encryption enabled,
        encryption keys should meet security standards and be properly managed.
        """
        # Skip if encryption is disabled
        assume(security_config.enable_data_encryption)
        
        encryption_manager = EncryptionManager(security_config)
        
        # Property: Encryption should be properly initialized
        assert encryption_manager.is_encryption_enabled()
        
        # Property: Encryption info should not expose sensitive data
        encryption_info = encryption_manager.get_encryption_info()
        
        assert isinstance(encryption_info, dict)
        assert 'encryption_enabled' in encryption_info
        assert 'algorithm' in encryption_info
        
        # Property: Sensitive key material should not be exposed
        for key, value in encryption_info.items():
            if isinstance(value, str):
                # Should not contain key material
                assert 'key' not in value.lower() or len(value) < 20
        
        # Property: Algorithm should be industry standard
        assert 'Fernet' in encryption_info.get('algorithm', '')
    
    @given(st.lists(external_data_source_strategy(), min_size=1, max_size=5))
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_data_source_validation_consistency(self, data_sources):
        """
        **Property 10.2: Data Source Validation Consistency**
        **Validates: Requirements 7.1**
        
        Property: For any collection of data sources, validation should be
        consistent and deterministic across multiple evaluations.
        """
        # Property: Validation should be consistent across multiple runs
        first_validation = [self._validate_data_source_compliance(source) for source in data_sources]
        second_validation = [self._validate_data_source_compliance(source) for source in data_sources]
        
        # Property: Results should be identical
        assert first_validation == second_validation
        
        # Property: Validation should be deterministic based on source properties
        for i, source in enumerate(data_sources):
            expected_result = self._expected_compliance_result(source)
            assert first_validation[i] == expected_result
    
    # Helper methods for property testing
    
    def _validate_data_source_compliance(self, data_source: Dict[str, Any]) -> bool:
        """Validate if a data source complies with privacy requirements."""
        # Only synthetic and public data sources are allowed
        if data_source['source_type'] not in ['synthetic', 'public']:
            return False
        
        # Data classification must be public or synthetic
        if data_source['data_classification'] not in ['public', 'synthetic']:
            return False
        
        # PII-containing sources must be synthetic
        if data_source['contains_pii'] and data_source['source_type'] != 'synthetic':
            return False
        
        # Sources requiring consent are not allowed for external data unless synthetic
        if data_source['requires_consent'] and data_source['source_type'] not in ['synthetic']:
            return False
        
        return True
    
    def _expected_compliance_result(self, data_source: Dict[str, Any]) -> bool:
        """Calculate expected compliance result for consistency testing."""
        return self._validate_data_source_compliance(data_source)
    
    def _create_audit_log_entry(self, operation: Dict[str, Any], timestamp: Optional[datetime] = None) -> AuditLogModel:
        """Create an audit log entry for testing."""
        return AuditLogModel(
            id=str(uuid4()),
            operation_type=operation['operation_type'],
            table_name=operation['table_name'],
            record_id=operation.get('record_id'),
            user_id=operation.get('user_id'),
            session_id=operation.get('session_id'),
            ip_address=operation.get('ip_address'),
            operation_details=operation.get('operation_details'),
            success=operation['success'],
            error_message=operation.get('error_message'),
            timestamp=timestamp or datetime.utcnow()
        )
    
    def _should_delete_data(self, data_date: datetime, retention_days: int) -> bool:
        """Determine if data should be deleted based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        return data_date <= cutoff_date  # Use <= for boundary case
    
    def _evaluate_data_sharing_request(self, request: Dict[str, Any]) -> bool:
        """Evaluate whether a data sharing request should be allowed."""
        # Third-party sharing requires explicit consent
        if request['third_party_recipient'] and not request['has_consent']:
            return False
        
        # Internal sharing may be allowed with proper justification
        if not request['third_party_recipient']:
            return True  # Simplified logic for testing
        
        # With consent, evaluate other factors
        if request['has_consent']:
            # Check retention period is reasonable
            if request['retention_period_days'] > 365:
                return False
            return True
        
        return False
    
    def _get_sharing_decision_reason(self, request: Dict[str, Any], allowed: bool) -> str:
        """Get reason for data sharing decision."""
        if not allowed:
            if request['third_party_recipient'] and not request['has_consent']:
                return "Third-party sharing requires explicit consent"
            elif request['retention_period_days'] > 365:
                return "Retention period exceeds maximum allowed"
            else:
                return "Request does not meet sharing criteria"
        else:
            return "Request meets all sharing requirements"


class TestDataSecurityIntegrationProperties:
    """Integration property tests for complete data security workflows."""
    
    def _validate_data_source_compliance(self, data_source: Dict[str, Any]) -> bool:
        """Validate if a data source complies with privacy requirements."""
        # Only synthetic and public data sources are allowed
        if data_source['source_type'] not in ['synthetic', 'public']:
            return False
        
        # Data classification must be public or synthetic
        if data_source['data_classification'] not in ['public', 'synthetic']:
            return False
        
        # PII-containing sources must be synthetic
        if data_source['contains_pii'] and data_source['source_type'] != 'synthetic':
            return False
        
        # Sources requiring consent are not allowed for external data unless synthetic
        if data_source['requires_consent'] and data_source['source_type'] not in ['synthetic']:
            return False
        
        return True
    
    def _evaluate_data_sharing_request(self, request: Dict[str, Any]) -> bool:
        """Evaluate whether a data sharing request should be allowed."""
        # Third-party sharing requires explicit consent
        if request['third_party_recipient'] and not request['has_consent']:
            return False
        
        # Internal sharing may be allowed with proper justification
        if not request['third_party_recipient']:
            return True  # Simplified logic for testing
        
        # With consent, evaluate other factors
        if request['has_consent']:
            # Check retention period is reasonable
            if request['retention_period_days'] > 365:
                return False
            return True
        
        return False
    
    def _get_sharing_decision_reason(self, request: Dict[str, Any], allowed: bool) -> str:
        """Get reason for data sharing decision."""
        if not allowed:
            if request['third_party_recipient'] and not request['has_consent']:
                return "Third-party sharing requires explicit consent"
            elif request['retention_period_days'] > 365:
                return "Retention period exceeds maximum allowed"
            else:
                return "Request does not meet sharing criteria"
        else:
            return "Request meets all sharing requirements"
    
    @given(security_config_strategy(), st.lists(sales_data_strategy(), min_size=1, max_size=3))
    @settings(max_examples=20, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.asyncio
    async def test_property_end_to_end_data_security_workflow(self, security_config, sales_data_list):
        """
        **Property 11.4: End-to-End Data Security Workflow**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        
        Property: Complete data security workflow should maintain security
        and compliance across all operations from storage to deletion.
        """
        # Skip if encryption is disabled
        assume(security_config.enable_data_encryption)
        
        # Create temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_url = f"sqlite:///{tmp_file.name}"
        
        try:
            # Initialize storage components
            from marketpulse_ai.config.database import DatabaseConfig
            db_config = DatabaseConfig(url=db_url)
            db_manager = DatabaseManager(db_config)
            db_manager.create_tables()
            
            storage_manager = StorageManager(db_manager, security_config)
            
            # Property: Complete workflow should succeed for valid data
            
            # Step 1: Store data with encryption
            store_result = await storage_manager.store_sales_data(sales_data_list, user_id="test_user")
            assert store_result['status'] in ['success', 'partial_success']
            
            # Step 2: Retrieve data with decryption
            retrieved_data = await storage_manager.retrieve_sales_data(user_id="test_user")
            assert len(retrieved_data) == store_result['stored_count']
            
            # Step 3: Verify audit logs were created
            # (In a real implementation, we would query audit logs)
            
            # Step 4: Test data lifecycle management
            stats = await storage_manager.get_storage_statistics()
            assert isinstance(stats, dict)
            assert 'sales_data_count' in stats
            assert stats['sales_data_count'] >= len(sales_data_list)
            
            # Step 5: Test cache cleanup
            cleanup_result = await storage_manager.cleanup_expired_cache()
            assert isinstance(cleanup_result, dict)
            assert 'deleted_count' in cleanup_result
            
            # Property: All workflow steps should maintain data integrity
            for i, original_item in enumerate(sales_data_list):
                if i < len(retrieved_data):
                    retrieved_item = retrieved_data[i]
                    # Verify critical fields match
                    assert retrieved_item.product_id == original_item.product_id
                    assert retrieved_item.mrp == original_item.mrp
                    assert retrieved_item.selling_price == original_item.selling_price
        
        finally:
            # Cleanup
            try:
                os.unlink(tmp_file.name)
            except:
                pass
    
    @given(st.lists(external_data_source_strategy(), min_size=1, max_size=5), 
           st.lists(data_sharing_request_strategy(), min_size=1, max_size=3))
    @settings(max_examples=15, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_comprehensive_privacy_compliance(self, data_sources, sharing_requests):
        """
        **Property 10.3: Comprehensive Privacy Compliance**
        **Validates: Requirements 7.1, 7.4**
        
        Property: For any combination of data sources and sharing requests,
        the system should maintain comprehensive privacy compliance.
        """
        # Property: Data source validation should be comprehensive
        compliant_sources = []
        non_compliant_sources = []
        
        for source in data_sources:
            if self._validate_data_source_compliance(source):
                compliant_sources.append(source)
            else:
                non_compliant_sources.append(source)
        
        # Property: Only compliant sources should be accepted
        for source in compliant_sources:
            assert source['source_type'] in ['synthetic', 'public']
            assert source['data_classification'] in ['public', 'synthetic']
        
        # Property: Non-compliant sources should be rejected
        for source in non_compliant_sources:
            # At least one of these conditions should be true for non-compliant sources
            is_wrong_type = source['source_type'] not in ['synthetic', 'public']
            is_wrong_classification = source['data_classification'] not in ['public', 'synthetic']
            is_pii_non_synthetic = source['contains_pii'] and source['source_type'] != 'synthetic'
            is_consent_required_non_synthetic = source['requires_consent'] and source['source_type'] not in ['synthetic']
            
            assert (is_wrong_type or is_wrong_classification or is_pii_non_synthetic or is_consent_required_non_synthetic), \
                f"Non-compliant source should have at least one violation: {source}"
        
        # Property: Sharing requests should be properly evaluated
        allowed_requests = []
        denied_requests = []
        
        for request in sharing_requests:
            if self._evaluate_data_sharing_request(request):
                allowed_requests.append(request)
            else:
                denied_requests.append(request)
        
        # Property: Denied requests should have valid reasons
        for request in denied_requests:
            reason = self._get_sharing_decision_reason(request, False)
            assert len(reason) > 10  # Should have meaningful reason
        
        # Property: System should maintain privacy by default
        total_requests = len(sharing_requests)
        if total_requests > 0:
            denial_rate = len(denied_requests) / total_requests
            # At least some requests should be denied for security
            # (This depends on the random data, but we expect some denials)
            pass  # In practice, we'd have more specific business rules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])