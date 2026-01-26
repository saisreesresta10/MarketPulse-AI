"""
Integration tests for Risk Assessor with Data Processor.

Tests the complete workflow of data processing followed by risk assessment.
"""

import pytest
import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import numpy as np

from marketpulse_ai.components.data_processor import DataProcessor
from marketpulse_ai.components.risk_assessor import RiskAssessor
from marketpulse_ai.core.models import SalesDataPoint, RiskLevel


class TestRiskAssessorIntegration:
    """Integration tests for Risk Assessor with Data Processor."""
    
    @pytest.fixture
    def data_processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    @pytest.fixture
    def risk_assessor(self):
        """Create a RiskAssessor instance for testing."""
        return RiskAssessor()
    
    @pytest.fixture
    def sample_sales_data(self):
        """Create comprehensive sample sales data for integration testing."""
        base_date = date.today() - timedelta(days=365)
        sales_data = []
        
        # Generate sales data for multiple products with different patterns
        products = [
            {"id": "ELECTRONICS_001", "name": "Smartphone", "category": "electronics", "base_qty": 30},
            {"id": "CLOTHING_001", "name": "Winter Jacket", "category": "clothing", "base_qty": 20},
            {"id": "FOOD_001", "name": "Premium Tea", "category": "food", "base_qty": 100}
        ]
        
        for product in products:
            for i in range(365):
                current_date = base_date + timedelta(days=i)
                
                # Different seasonal patterns for different products
                if product["category"] == "electronics":
                    # Electronics peak during festivals
                    if current_date.month in [10, 11, 12]:
                        seasonal_boost = 2.0
                    else:
                        seasonal_boost = 1.0
                elif product["category"] == "clothing":
                    # Clothing peaks in winter
                    if current_date.month in [11, 12, 1, 2]:
                        seasonal_boost = 1.8
                    elif current_date.month in [6, 7, 8]:
                        seasonal_boost = 0.5
                    else:
                        seasonal_boost = 1.0
                else:  # food
                    # Food has moderate seasonal variation
                    if current_date.month in [10, 11, 12]:
                        seasonal_boost = 1.3
                    else:
                        seasonal_boost = 1.0
                
                # Add some randomness and weekly patterns
                weekly_factor = 1.2 if current_date.weekday() < 5 else 0.8  # Weekdays vs weekends
                random_factor = 0.7 + np.random.random() * 0.6  # ±30% variation
                
                quantity = int(product["base_qty"] * seasonal_boost * weekly_factor * random_factor)
                quantity = max(1, quantity)  # Ensure at least 1 unit sold
                
                sales_data.append(SalesDataPoint(
                    product_id=product["id"],
                    product_name=product["name"],
                    category=product["category"],
                    mrp=Decimal("1000.00"),
                    selling_price=Decimal("900.00"),
                    quantity_sold=quantity,
                    sale_date=current_date,
                    store_location="MAIN_STORE",
                    seasonal_event="diwali" if current_date.month == 11 and current_date.day <= 7 else None
                ))
        
        return sales_data
    
    @pytest.mark.asyncio
    async def test_complete_risk_assessment_workflow(self, data_processor, risk_assessor, sample_sales_data):
        """Test complete workflow from data processing to risk assessment."""
        
        # Step 1: Process sales data
        ingestion_result = await data_processor.ingest_sales_data(sample_sales_data)
        
        assert ingestion_result['status'] == 'success'
        assert ingestion_result['records_accepted'] > 0
        
        # Step 2: Extract demand patterns
        patterns = await data_processor.extract_demand_patterns()
        
        assert len(patterns) > 0
        
        # Step 3: Enhance patterns with seasonal correlation
        enhanced_patterns = await data_processor.correlate_seasonal_events(patterns)
        
        assert len(enhanced_patterns) == len(patterns)
        
        # Step 4: Set up risk assessor with processed data
        risk_assessor.set_sales_data(data_processor.processed_data)
        
        # Convert patterns to the format expected by risk assessor
        patterns_dict = {}
        for pattern in enhanced_patterns:
            if pattern.product_id not in patterns_dict:
                patterns_dict[pattern.product_id] = []
            patterns_dict[pattern.product_id].append(pattern)
        
        risk_assessor.set_demand_patterns(patterns_dict)
        
        # Step 5: Perform risk assessments for different scenarios
        
        # Test overstock scenario
        overstock_assessment = await risk_assessor.assess_overstock_risk("ELECTRONICS_001", 500)
        
        assert overstock_assessment.product_id == "ELECTRONICS_001"
        assert overstock_assessment.risk_type == "overstock"
        assert overstock_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert 0.0 <= overstock_assessment.risk_score <= 1.0
        assert len(overstock_assessment.contributing_factors) > 0
        assert len(overstock_assessment.mitigation_suggestions) > 0
        
        # Test understock scenario
        understock_assessment = await risk_assessor.assess_understock_risk("CLOTHING_001", 5)
        
        assert understock_assessment.product_id == "CLOTHING_001"
        assert understock_assessment.risk_type == "understock"
        assert understock_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert 0.0 <= understock_assessment.risk_score <= 1.0
        assert len(understock_assessment.contributing_factors) > 0
        assert len(understock_assessment.mitigation_suggestions) > 0
        
        # Step 6: Test seasonal adjustments
        upcoming_events = ["diwali", "winter"]
        
        adjusted_overstock = await risk_assessor.adjust_for_seasonal_events(
            overstock_assessment, upcoming_events
        )
        
        assert len(adjusted_overstock.seasonal_adjustments) > 0
        assert "diwali" in adjusted_overstock.seasonal_adjustments or "winter" in adjusted_overstock.seasonal_adjustments
        
        adjusted_understock = await risk_assessor.adjust_for_seasonal_events(
            understock_assessment, upcoming_events
        )
        
        assert len(adjusted_understock.seasonal_adjustments) > 0
        
        # Step 7: Test early warning generation
        assessments = [adjusted_overstock, adjusted_understock]
        warned_assessments = await risk_assessor.generate_early_warnings(assessments)
        
        assert len(warned_assessments) == 2
        
        # At least one assessment should have early warning if risk is high enough
        high_risk_assessments = [a for a in warned_assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risk_assessments:
            assert any(a.early_warning_triggered for a in high_risk_assessments)
    
    @pytest.mark.asyncio
    async def test_volatility_calculation_integration(self, data_processor, risk_assessor, sample_sales_data):
        """Test demand volatility calculation with processed data."""
        
        # Process data
        await data_processor.ingest_sales_data(sample_sales_data)
        risk_assessor.set_sales_data(data_processor.processed_data)
        
        # Calculate volatility for each product
        products = ["ELECTRONICS_001", "CLOTHING_001", "FOOD_001"]
        
        for product_id in products:
            volatility = await risk_assessor.calculate_demand_volatility(product_id)
            
            assert 0.0 <= volatility <= 1.0
            assert isinstance(volatility, float)
            
            # Electronics should have higher volatility due to seasonal spikes
            if product_id == "ELECTRONICS_001":
                # Should have some volatility due to seasonal patterns
                assert volatility > 0.1
    
    @pytest.mark.asyncio
    async def test_seasonal_analysis_integration(self, data_processor, risk_assessor, sample_sales_data):
        """Test integration of seasonal analysis with risk assessment."""
        
        # Process data and generate seasonal analysis
        await data_processor.ingest_sales_data(sample_sales_data)
        
        seasonal_report = await data_processor.generate_seasonal_analysis_report()
        
        assert seasonal_report['status'] == 'success'
        assert 'seasonal_insights' in seasonal_report
        assert 'recommendations' in seasonal_report
        
        # Set up risk assessor
        risk_assessor.set_sales_data(data_processor.processed_data)
        
        # Test risk assessment during different seasons
        # High inventory during low season should be higher overstock risk
        winter_jacket_overstock = await risk_assessor.assess_overstock_risk("CLOTHING_001", 200)
        
        # Adjust for summer season (low demand for winter clothing)
        summer_adjusted = await risk_assessor.adjust_for_seasonal_events(
            winter_jacket_overstock, ["summer"]
        )
        
        # Summer adjustment should increase overstock risk for winter clothing
        # (or at least not decrease it significantly)
        assert summer_adjusted.risk_score >= winter_jacket_overstock.risk_score * 0.8
    
    @pytest.mark.asyncio
    async def test_multi_product_risk_assessment(self, data_processor, risk_assessor, sample_sales_data):
        """Test risk assessment across multiple products."""
        
        # Process data
        await data_processor.ingest_sales_data(sample_sales_data)
        risk_assessor.set_sales_data(data_processor.processed_data)
        
        # Assess risks for all products
        products = ["ELECTRONICS_001", "CLOTHING_001", "FOOD_001"]
        inventory_levels = [100, 50, 300]  # Different inventory levels
        
        overstock_assessments = []
        understock_assessments = []
        
        for product_id, inventory in zip(products, inventory_levels):
            # Test both overstock and understock scenarios
            overstock = await risk_assessor.assess_overstock_risk(product_id, inventory * 3)  # High inventory
            understock = await risk_assessor.assess_understock_risk(product_id, inventory // 10)  # Low inventory
            
            overstock_assessments.append(overstock)
            understock_assessments.append(understock)
        
        # Generate early warnings for all assessments
        all_assessments = overstock_assessments + understock_assessments
        warned_assessments = await risk_assessor.generate_early_warnings(all_assessments)
        
        assert len(warned_assessments) == len(all_assessments)
        
        # Verify each assessment has valid properties
        for assessment in warned_assessments:
            assert assessment.product_id in products
            assert assessment.risk_type in ["overstock", "understock"]
            assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            assert 0.0 <= assessment.risk_score <= 1.0
            assert len(assessment.contributing_factors) > 0
            assert len(assessment.mitigation_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, data_processor, risk_assessor):
        """Test error handling in integrated workflow."""
        
        # Test with minimal data
        minimal_data = [
            SalesDataPoint(
                product_id="MINIMAL_PRODUCT",
                product_name="Minimal Product",
                category="test",
                mrp=Decimal("100.00"),
                selling_price=Decimal("90.00"),
                quantity_sold=10,
                sale_date=date.today(),
                store_location="TEST_STORE"
            )
        ]
        
        # This should work for data processing
        ingestion_result = await data_processor.ingest_sales_data(minimal_data)
        assert ingestion_result['status'] == 'success'
        
        # But should fail for risk assessment due to insufficient data
        risk_assessor.set_sales_data(data_processor.processed_data)
        
        from marketpulse_ai.components.risk_assessor import RiskCalculationError
        
        with pytest.raises(RiskCalculationError):
            await risk_assessor.assess_overstock_risk("MINIMAL_PRODUCT", 50)


if __name__ == "__main__":
    pytest.main([__file__])