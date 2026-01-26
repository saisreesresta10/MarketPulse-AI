"""
Database models for MarketPulse AI storage.

SQLAlchemy models for persisting sales data, patterns, insights,
and other analysis results with proper indexing and relationships.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Date, Text, Boolean,
    JSON, DECIMAL, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from ..config.database import Base


class SalesDataModel(Base):
    """
    Database model for sales data points.
    
    Stores individual sales transactions with proper indexing
    for efficient querying and analysis.
    """
    __tablename__ = 'sales_data'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Product information
    product_id = Column(String(100), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    
    # Pricing information (encrypted)
    mrp_encrypted = Column(Text, nullable=False)  # Encrypted MRP
    selling_price_encrypted = Column(Text, nullable=False)  # Encrypted selling price
    
    # Sales information
    quantity_sold = Column(Integer, nullable=False)
    sale_date = Column(Date, nullable=False, index=True)
    store_location = Column(String(100), nullable=False, index=True)
    seasonal_event = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_product_date', 'product_id', 'sale_date'),
        Index('idx_category_date', 'category', 'sale_date'),
        Index('idx_store_date', 'store_location', 'sale_date'),
        Index('idx_seasonal_event', 'seasonal_event'),
    )
    
    def __repr__(self):
        return f"<SalesDataModel(id={self.id}, product_id={self.product_id}, sale_date={self.sale_date})>"


class DemandPatternModel(Base):
    """
    Database model for demand patterns.
    
    Stores identified patterns from sales data analysis with
    seasonal factors and confidence metrics.
    """
    __tablename__ = 'demand_patterns'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Pattern information
    product_id = Column(String(100), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    confidence_level = Column(String(20), nullable=False)
    
    # Pattern characteristics
    seasonal_factors = Column(JSON, nullable=True)  # JSON field for seasonal factors
    trend_direction = Column(String(20), nullable=True)
    volatility_score = Column(Float, nullable=False)
    supporting_data_points = Column(Integer, nullable=False)
    
    # Date range
    date_range_start = Column(Date, nullable=False)
    date_range_end = Column(Date, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_product_pattern', 'product_id', 'pattern_type'),
        Index('idx_confidence_level', 'confidence_level'),
        Index('idx_date_range', 'date_range_start', 'date_range_end'),
    )
    
    def __repr__(self):
        return f"<DemandPatternModel(id={self.id}, product_id={self.product_id}, pattern_type={self.pattern_type})>"


class InsightModel(Base):
    """
    Database model for explainable insights.
    
    Stores generated insights with supporting evidence and
    business recommendations.
    """
    __tablename__ = 'insights'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Insight information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    confidence_level = Column(String(20), nullable=False, index=True)
    
    # Supporting information
    supporting_evidence = Column(JSON, nullable=False)  # List of evidence points
    key_factors = Column(JSON, nullable=False)  # List of key factors
    business_impact = Column(Text, nullable=False)
    recommended_actions = Column(JSON, nullable=True)  # List of recommendations
    data_sources = Column(JSON, nullable=False)  # List of data sources
    related_products = Column(JSON, nullable=True)  # List of product IDs
    
    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True, index=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_insight_confidence_expires', 'confidence_level', 'expires_at'),
        Index('idx_insight_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<InsightModel(id={self.id}, title={self.title[:50]}...)>"


class RiskAssessmentModel(Base):
    """
    Database model for risk assessments.
    
    Stores inventory and demand risk evaluations with
    mitigation suggestions and validity periods.
    """
    __tablename__ = 'risk_assessments'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Risk information
    product_id = Column(String(100), nullable=False, index=True)
    risk_type = Column(String(50), nullable=False, index=True)
    risk_level = Column(String(20), nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    
    # Risk details
    contributing_factors = Column(JSON, nullable=False)  # List of factors
    seasonal_adjustments = Column(JSON, nullable=True)  # Seasonal adjustments
    early_warning_triggered = Column(Boolean, default=False, nullable=False, index=True)
    mitigation_suggestions = Column(JSON, nullable=True)  # List of suggestions
    
    # Validity
    assessment_date = Column(Date, nullable=False, index=True)
    valid_until = Column(Date, nullable=False, index=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_product_risk_type', 'product_id', 'risk_type'),
        Index('idx_risk_level_score', 'risk_level', 'risk_score'),
        Index('idx_early_warning', 'early_warning_triggered', 'valid_until'),
    )
    
    def __repr__(self):
        return f"<RiskAssessmentModel(id={self.id}, product_id={self.product_id}, risk_type={self.risk_type})>"


class ScenarioModel(Base):
    """
    Database model for what-if scenarios.
    
    Stores scenario parameters, assumptions, and predicted
    outcomes for decision support analysis.
    """
    __tablename__ = 'scenarios'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Scenario information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    confidence_level = Column(String(20), nullable=False, index=True)
    time_horizon = Column(String(50), nullable=False)
    
    # Scenario data
    parameters = Column(JSON, nullable=False)  # Scenario parameters
    predicted_outcomes = Column(JSON, nullable=False)  # Predicted results
    assumptions = Column(JSON, nullable=False)  # List of assumptions
    limitations = Column(JSON, nullable=False)  # List of limitations
    affected_products = Column(JSON, nullable=True)  # List of product IDs
    seasonal_considerations = Column(JSON, nullable=True)  # Seasonal factors
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_scenario_confidence_horizon', 'confidence_level', 'time_horizon'),
        Index('idx_scenario_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ScenarioModel(id={self.id}, name={self.name})>"


class ComplianceResultModel(Base):
    """
    Database model for compliance validation results.
    
    Stores MRP regulation compliance checks with violations
    and regulatory constraints.
    """
    __tablename__ = 'compliance_results'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Associated recommendation (optional)
    recommendation_id = Column(String(36), nullable=True, index=True)
    
    # Compliance information
    compliance_status = Column(String(20), nullable=False, index=True)
    regulations_checked = Column(JSON, nullable=False)  # List of regulations
    violations = Column(JSON, nullable=True)  # List of violations
    warnings = Column(JSON, nullable=True)  # List of warnings
    constraints = Column(JSON, nullable=True)  # Regulatory constraints
    validation_details = Column(JSON, nullable=True)  # Detailed results
    
    # Validation metadata
    validator_version = Column(String(20), nullable=False)
    checked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_compliance_status', 'compliance_status'),
        Index('idx_compliance_recommendation_compliance', 'recommendation_id', 'compliance_status'),
        Index('idx_compliance_checked_at', 'checked_at'),
    )
    
    def __repr__(self):
        return f"<ComplianceResultModel(id={self.id}, status={self.compliance_status})>"


class CacheEntryModel(Base):
    """
    Database model for caching analysis results.
    
    Provides efficient caching of computed results to avoid
    redundant calculations and improve performance.
    """
    __tablename__ = 'cache_entries'
    
    # Primary key
    cache_key = Column(String(255), primary_key=True)
    
    # Cache data
    cache_type = Column(String(50), nullable=False, index=True)
    data_encrypted = Column(Text, nullable=False)  # Encrypted cached data
    
    # Cache metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_cache_type_expires', 'cache_type', 'expires_at'),
        Index('idx_cache_expires_at', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<CacheEntryModel(key={self.cache_key}, type={self.cache_type})>"


class AuditLogModel(Base):
    """
    Database model for audit logging.
    
    Tracks all data operations for security and compliance
    monitoring as required by data privacy regulations.
    """
    __tablename__ = 'audit_logs'
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    
    # Operation information
    operation_type = Column(String(50), nullable=False, index=True)  # CREATE, READ, UPDATE, DELETE
    table_name = Column(String(100), nullable=False, index=True)
    record_id = Column(String(36), nullable=True, index=True)
    
    # User and session information
    user_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    
    # Operation details
    operation_details = Column(JSON, nullable=True)  # Additional operation context
    success = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    
    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_operation_table', 'operation_type', 'table_name'),
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_success_timestamp', 'success', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<AuditLogModel(id={self.id}, operation={self.operation_type}, table={self.table_name})>"