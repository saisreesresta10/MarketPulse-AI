"""
Pytest configuration and shared fixtures for MarketPulse AI tests.

Provides common test fixtures, configuration, and utilities
for consistent testing across all test modules.
"""

import os
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from uuid import uuid4

import pytest
from hypothesis import settings, Verbosity
from hypothesis.strategies import composite, integers, text, decimals, dates, booleans, lists, dictionaries, sampled_from

from marketpulse_ai.config.settings import Settings, Environment
from marketpulse_ai.config.database import DatabaseConfig, DatabaseManager
from marketpulse_ai.core.models import (
    SalesDataPoint,
    DemandPattern,
    ExplainableInsight,
    RiskAssessment,
    Scenario,
    ComplianceResult,
    ConfidenceLevel,
    RiskLevel,
    ComplianceStatus,
)


# Hypothesis configuration for property-based testing
settings.register_profile("default", max_examples=100, deadline=5000)
settings.register_profile("ci", max_examples=1000, deadline=10000)
settings.register_profile("dev", max_examples=10, deadline=1000, verbosity=Verbosity.verbose)

# Load profile based on environment
profile = os.getenv("HYPOTHESIS_PROFILE", "default")
settings.load_profile(profile)


@pytest.fixture(scope="session")
def test_settings():
    """
    Provide test-specific settings configuration.
    
    Returns:
        Settings instance configured for testing
    """
    return Settings(
        environment=Environment.TESTING,
        debug=True,
        database_url="sqlite:///./test_marketpulse.db",
        database_echo=False,
        secret_key="test_secret_key_with_sufficient_length_for_security",
        log_level="DEBUG",
        enable_property_testing=True,
        property_test_iterations=10,  # Reduced for faster tests
    )


@pytest.fixture(scope="session")
def test_db_config(test_settings):
    """
    Provide test database configuration.
    
    Args:
        test_settings: Test settings fixture
        
    Returns:
        DatabaseConfig instance for testing
    """
    return DatabaseConfig(
        url=test_settings.database_url,
        echo=test_settings.database_echo,
        pool_size=1,  # Single connection for tests
        max_overflow=0,
    )


@pytest.fixture(scope="function")
def db_manager(test_db_config):
    """
    Provide database manager with clean database for each test.
    
    Args:
        test_db_config: Test database configuration
        
    Yields:
        DatabaseManager instance with clean database
    """
    # Use temporary file for each test
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        test_config = DatabaseConfig(
            url=f"sqlite:///{tmp_file.name}",
            echo=False,
            pool_size=1,
            max_overflow=0,
        )
        
        manager = DatabaseManager(test_config)
        manager.create_tables()
        
        yield manager
        
        # Cleanup
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


@pytest.fixture
def sample_sales_data():
    """
    Provide sample sales data for testing.
    
    Returns:
        List of SalesDataPoint instances
    """
    return [
        SalesDataPoint(
            product_id="PROD001",
            product_name="Premium Tea",
            category="Beverages",
            mrp=Decimal("100.00"),
            selling_price=Decimal("95.00"),
            quantity_sold=50,
            sale_date=date(2024, 1, 15),
            store_location="Mumbai_Central",
            seasonal_event="Republic Day"
        ),
        SalesDataPoint(
            product_id="PROD002",
            product_name="Organic Rice",
            category="Groceries",
            mrp=Decimal("250.00"),
            selling_price=Decimal("240.00"),
            quantity_sold=25,
            sale_date=date(2024, 1, 16),
            store_location="Delhi_CP",
        ),
        SalesDataPoint(
            product_id="PROD001",
            product_name="Premium Tea",
            category="Beverages",
            mrp=Decimal("100.00"),
            selling_price=Decimal("90.00"),
            quantity_sold=75,
            sale_date=date(2024, 1, 20),
            store_location="Bangalore_MG",
            seasonal_event="Pongal"
        ),
    ]


@pytest.fixture
def sample_demand_pattern():
    """
    Provide sample demand pattern for testing.
    
    Returns:
        DemandPattern instance
    """
    return DemandPattern(
        product_id="PROD001",
        pattern_type="seasonal",
        description="Increased demand during winter festivals",
        confidence_level=ConfidenceLevel.HIGH,
        seasonal_factors={"winter": 1.3, "summer": 0.8, "monsoon": 1.0},
        trend_direction="increasing",
        volatility_score=0.25,
        supporting_data_points=150,
        date_range_start=date(2023, 1, 1),
        date_range_end=date(2024, 1, 31),
    )


@pytest.fixture
def sample_insight():
    """
    Provide sample explainable insight for testing.
    
    Returns:
        ExplainableInsight instance
    """
    return ExplainableInsight(
        title="Winter Festival Demand Surge",
        description="Premium Tea shows 30% higher demand during winter festivals",
        confidence_level=ConfidenceLevel.HIGH,
        supporting_evidence=[
            "Historical sales data shows consistent 25-35% increase",
            "Correlation with festival calendar is 0.85",
            "Pattern observed across 3 consecutive years"
        ],
        key_factors=[
            "Festival season timing",
            "Weather conditions",
            "Cultural preferences"
        ],
        business_impact="Potential 30% revenue increase with proper inventory planning",
        recommended_actions=[
            "Increase inventory by 25% before winter festivals",
            "Plan promotional campaigns for festival periods"
        ],
        data_sources=["sales_history", "festival_calendar", "weather_data"],
        related_products=["PROD001", "PROD003"]
    )


@pytest.fixture
def sample_risk_assessment():
    """
    Provide sample risk assessment for testing.
    
    Returns:
        RiskAssessment instance
    """
    return RiskAssessment(
        product_id="PROD002",
        risk_type="overstock",
        risk_level=RiskLevel.MEDIUM,
        risk_score=0.65,
        contributing_factors=[
            "Declining demand trend",
            "High current inventory",
            "Seasonal demand reduction"
        ],
        seasonal_adjustments={"summer": 0.7, "monsoon": 0.9},
        early_warning_triggered=True,
        mitigation_suggestions=[
            "Implement promotional pricing",
            "Explore alternative sales channels"
        ],
        assessment_date=date.today(),
        valid_until=date.today() + timedelta(days=30)
    )


@pytest.fixture
def sample_scenario():
    """
    Provide sample scenario for testing.
    
    Returns:
        Scenario instance
    """
    return Scenario(
        name="Festival Season Inventory Planning",
        description="What-if analysis for Diwali season inventory management",
        parameters={
            "inventory_increase_percent": 25,
            "discount_percent": 10,
            "festival_duration_days": 15
        },
        predicted_outcomes={
            "expected_sales_increase": 30,
            "inventory_turnover": 1.8,
            "profit_margin_impact": -2
        },
        confidence_level=ConfidenceLevel.MEDIUM,
        assumptions=[
            "Historical demand patterns continue",
            "No major supply chain disruptions",
            "Competitor pricing remains stable"
        ],
        limitations=[
            "Based on 3 years of historical data",
            "Does not account for new market entrants",
            "Weather impact not fully modeled"
        ],
        time_horizon="3 months",
        affected_products=["PROD001", "PROD003", "PROD005"],
        seasonal_considerations=["Diwali", "Dhanteras", "Karva Chauth"]
    )


@pytest.fixture
def sample_compliance_result():
    """
    Provide sample compliance result for testing.
    
    Returns:
        ComplianceResult instance
    """
    return ComplianceResult(
        recommendation_id=uuid4(),
        compliance_status=ComplianceStatus.COMPLIANT,
        regulations_checked=["MRP_ACT_2009", "CONSUMER_PROTECTION_ACT_2019"],
        violations=[],
        warnings=["Discount approaching maximum allowed limit"],
        constraints={
            "max_discount_percent": 15,
            "min_selling_price": 85.0,
            "mrp_display_required": True
        },
        validation_details={
            "mrp_compliance": True,
            "discount_compliance": True,
            "labeling_compliance": True
        },
        validator_version="1.0.0"
    )


# Hypothesis strategies for property-based testing

@composite
def sales_data_strategy(draw):
    """
    Hypothesis strategy for generating valid SalesDataPoint instances.
    
    Args:
        draw: Hypothesis draw function
        
    Returns:
        Generated SalesDataPoint instance
    """
    mrp = draw(decimals(min_value=1, max_value=10000, places=2))
    selling_price = draw(decimals(min_value=1, max_value=mrp, places=2))
    
    return SalesDataPoint(
        product_id=draw(text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
        product_name=draw(text(min_size=1, max_size=100)),
        category=draw(text(min_size=1, max_size=50)),
        mrp=mrp,
        selling_price=selling_price,
        quantity_sold=draw(integers(min_value=0, max_value=1000)),
        sale_date=draw(dates(min_value=date(2020, 1, 1), max_value=date.today())),
        store_location=draw(text(min_size=1, max_size=50)),
        seasonal_event=draw(text(min_size=0, max_size=50)) or None,
    )


@composite
def demand_pattern_strategy(draw):
    """
    Hypothesis strategy for generating valid DemandPattern instances.
    
    Args:
        draw: Hypothesis draw function
        
    Returns:
        Generated DemandPattern instance
    """
    start_date = draw(dates(min_value=date(2020, 1, 1), max_value=date(2023, 12, 30)))
    end_date = draw(dates(min_value=start_date + timedelta(days=1), max_value=date.today()))
    
    return DemandPattern(
        product_id=draw(text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
        pattern_type=draw(text(min_size=1, max_size=50)),
        description=draw(text(min_size=1, max_size=200)),
        confidence_level=draw(sampled_from([ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH])),
        seasonal_factors=draw(dictionaries(text(min_size=1, max_size=20), decimals(min_value=Decimal('0.10'), max_value=Decimal('2.00'), places=2))),
        trend_direction=draw(text(min_size=0, max_size=20)) or None,
        volatility_score=draw(decimals(min_value=Decimal('0.00'), max_value=Decimal('1.00'), places=2)),
        supporting_data_points=draw(integers(min_value=1, max_value=10000)),
        date_range_start=start_date,
        date_range_end=end_date,
    )


@composite
def risk_assessment_strategy(draw):
    """
    Hypothesis strategy for generating valid RiskAssessment instances.
    
    Args:
        draw: Hypothesis draw function
        
    Returns:
        Generated RiskAssessment instance
    """
    assessment_date = draw(dates(min_value=date(2024, 1, 1), max_value=date.today() - timedelta(days=1)))
    valid_until = draw(dates(min_value=assessment_date + timedelta(days=1), max_value=assessment_date + timedelta(days=365)))
    
    return RiskAssessment(
        product_id=draw(text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
        risk_type=draw(text(min_size=1, max_size=50)),
        risk_level=draw(sampled_from([RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL])),
        risk_score=draw(decimals(min_value=Decimal('0.00'), max_value=Decimal('1.00'), places=2)),
        contributing_factors=draw(lists(text(min_size=1, max_size=100), min_size=1, max_size=10)),
        seasonal_adjustments=draw(dictionaries(text(min_size=1, max_size=20), decimals(min_value=Decimal('0.10'), max_value=Decimal('2.00'), places=2))),
        early_warning_triggered=draw(booleans()),
        mitigation_suggestions=draw(lists(text(min_size=1, max_size=100), min_size=0, max_size=5)),
        assessment_date=assessment_date,
        valid_until=valid_until,
    )


# Test utilities

def assert_model_equality(model1, model2, exclude_fields=None):
    """
    Assert that two Pydantic models are equal, optionally excluding fields.
    
    Args:
        model1: First model instance
        model2: Second model instance
        exclude_fields: List of field names to exclude from comparison
    """
    exclude_fields = exclude_fields or []
    
    dict1 = model1.dict(exclude=set(exclude_fields))
    dict2 = model2.dict(exclude=set(exclude_fields))
    
    assert dict1 == dict2, f"Models differ: {dict1} != {dict2}"


def create_test_data_batch(model_class, count=10, **kwargs):
    """
    Create a batch of test data instances.
    
    Args:
        model_class: Pydantic model class to create
        count: Number of instances to create
        **kwargs: Additional arguments for model creation
        
    Returns:
        List of model instances
    """
    instances = []
    for i in range(count):
        # Add variation to avoid duplicate data
        instance_kwargs = kwargs.copy()
        if hasattr(model_class, 'id'):
            instance_kwargs['id'] = uuid4()
        
        instances.append(model_class(**instance_kwargs))
    
    return instances