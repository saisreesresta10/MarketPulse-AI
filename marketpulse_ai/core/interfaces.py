"""
Core interfaces defining contracts for MarketPulse AI components.

These abstract base classes define the expected behavior and methods
for each major component in the system architecture.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from .models import (
    SalesDataPoint,
    DemandPattern,
    ExplainableInsight,
    RiskAssessment,
    Scenario,
    ComplianceResult,
)


class DataProcessorInterface(ABC):
    """
    Interface for data processing and pattern extraction components.
    
    Defines methods for ingesting, validating, and analyzing sales data
    to extract meaningful patterns and trends.
    """
    
    @abstractmethod
    async def ingest_sales_data(self, data: List[SalesDataPoint]) -> Dict[str, Any]:
        """
        Ingest and validate sales data for processing.
        
        Args:
            data: List of sales data points to process
            
        Returns:
            Dictionary containing ingestion results and statistics
            
        Raises:
            ValueError: If data validation fails
        """
        pass
    
    @abstractmethod
    async def extract_demand_patterns(self, product_ids: Optional[List[str]] = None) -> List[DemandPattern]:
        """
        Extract demand patterns from processed sales data.
        
        Args:
            product_ids: Optional list of product IDs to analyze (None for all)
            
        Returns:
            List of identified demand patterns
        """
        pass
    
    @abstractmethod
    async def correlate_seasonal_events(self, patterns: List[DemandPattern]) -> List[DemandPattern]:
        """
        Correlate demand patterns with seasonal events and festivals.
        
        Args:
            patterns: List of demand patterns to enhance with seasonal correlation
            
        Returns:
            Updated patterns with seasonal correlation data
        """
        pass
    
    @abstractmethod
    async def integrate_market_signals(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate external market signals into analysis.
        
        Args:
            external_data: Dictionary containing external market data
            
        Returns:
            Integration results and updated analysis
        """
        pass
    
    @abstractmethod
    async def store_patterns(self, patterns: List[DemandPattern]) -> bool:
        """
        Store analyzed patterns for future reference.
        
        Args:
            patterns: List of patterns to store
            
        Returns:
            True if storage successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_seasonal_analysis_report(self, product_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive seasonal analysis report.
        
        Args:
            product_ids: Optional list of product IDs to analyze (None for all)
            
        Returns:
            Comprehensive seasonal analysis report
        """
        pass


class InsightGeneratorInterface(ABC):
    """
    Interface for generating explainable insights from data analysis.
    
    Defines methods for creating human-readable insights with supporting
    evidence and confidence levels.
    """
    
    @abstractmethod
    async def generate_insights(self, patterns: List[DemandPattern]) -> List[ExplainableInsight]:
        """
        Generate explainable insights from demand patterns.
        
        Args:
            patterns: List of demand patterns to analyze
            
        Returns:
            List of generated insights with explanations
        """
        pass
    
    @abstractmethod
    async def explain_pattern(self, pattern: DemandPattern) -> ExplainableInsight:
        """
        Create detailed explanation for a specific demand pattern.
        
        Args:
            pattern: Demand pattern to explain
            
        Returns:
            Detailed insight with explanation and evidence
        """
        pass
    
    @abstractmethod
    async def calculate_confidence(self, insight: ExplainableInsight) -> float:
        """
        Calculate confidence score for an insight.
        
        Args:
            insight: Insight to evaluate
            
        Returns:
            Confidence score between 0 and 1
        """
        pass
    
    @abstractmethod
    async def identify_key_factors(self, pattern: DemandPattern) -> List[str]:
        """
        Identify key factors influencing a demand pattern.
        
        Args:
            pattern: Demand pattern to analyze
            
        Returns:
            List of key influencing factors
        """
        pass


class RiskAssessorInterface(ABC):
    """
    Interface for inventory and demand risk assessment.
    
    Defines methods for identifying, calculating, and reporting
    various types of inventory and business risks.
    """
    
    @abstractmethod
    async def assess_overstock_risk(self, product_id: str, current_inventory: int) -> RiskAssessment:
        """
        Assess overstock risk for a specific product.
        
        Args:
            product_id: Product identifier to assess
            current_inventory: Current inventory level
            
        Returns:
            Risk assessment for overstock scenario
        """
        pass
    
    @abstractmethod
    async def assess_understock_risk(self, product_id: str, current_inventory: int) -> RiskAssessment:
        """
        Assess understock risk for a specific product.
        
        Args:
            product_id: Product identifier to assess
            current_inventory: Current inventory level
            
        Returns:
            Risk assessment for understock scenario
        """
        pass
    
    @abstractmethod
    async def calculate_demand_volatility(self, product_id: str) -> float:
        """
        Calculate demand volatility score for a product.
        
        Args:
            product_id: Product identifier to analyze
            
        Returns:
            Volatility score between 0 and 1
        """
        pass
    
    @abstractmethod
    async def adjust_for_seasonal_events(self, assessment: RiskAssessment, upcoming_events: List[str]) -> RiskAssessment:
        """
        Adjust risk assessment for upcoming seasonal events.
        
        Args:
            assessment: Base risk assessment to adjust
            upcoming_events: List of upcoming seasonal events
            
        Returns:
            Adjusted risk assessment
        """
        pass
    
    @abstractmethod
    async def generate_early_warnings(self, assessments: List[RiskAssessment]) -> List[RiskAssessment]:
        """
        Generate early warning alerts for high-risk situations.
        
        Args:
            assessments: List of risk assessments to evaluate
            
        Returns:
            List of assessments with early warning flags
        """
        pass


class ComplianceValidatorInterface(ABC):
    """
    Interface for MRP regulation compliance validation.
    
    Defines methods for validating recommendations against Indian
    retail regulations and MRP constraints.
    """
    
    @abstractmethod
    async def validate_mrp_compliance(self, recommendation: Dict[str, Any]) -> ComplianceResult:
        """
        Validate recommendation against MRP regulations.
        
        Args:
            recommendation: Recommendation to validate
            
        Returns:
            Compliance validation result
        """
        pass
    
    @abstractmethod
    async def check_discount_limits(self, product_id: str, proposed_discount: float) -> ComplianceResult:
        """
        Check if proposed discount complies with regulations.
        
        Args:
            product_id: Product identifier
            proposed_discount: Proposed discount percentage
            
        Returns:
            Compliance result for discount proposal
        """
        pass
    
    @abstractmethod
    async def validate_pricing_strategy(self, strategy: Dict[str, Any]) -> ComplianceResult:
        """
        Validate entire pricing strategy for compliance.
        
        Args:
            strategy: Pricing strategy to validate
            
        Returns:
            Comprehensive compliance validation result
        """
        pass
    
    @abstractmethod
    async def get_regulatory_constraints(self, product_category: str) -> Dict[str, Any]:
        """
        Get regulatory constraints for a product category.
        
        Args:
            product_category: Product category to check
            
        Returns:
            Dictionary of applicable regulatory constraints
        """
        pass
    
    @abstractmethod
    async def update_regulation_rules(self, new_rules: Dict[str, Any]) -> bool:
        """
        Update regulation rules when regulations change.
        
        Args:
            new_rules: Updated regulation rules
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_constraint_explanation(self, product_category: str, 
                                            recommendation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate human-readable explanations of regulatory constraints.
        
        Args:
            product_category: Product category to explain constraints for
            recommendation: Optional recommendation context for specific explanations
            
        Returns:
            Dictionary containing detailed constraint explanations
        """
        pass
    
    @abstractmethod
    async def get_system_limitations_and_transparency(self) -> Dict[str, Any]:
        """
        Provide comprehensive information about system limitations and data sources.
        
        Returns:
            Dictionary containing system limitations and transparency information
        """
        pass
    
    @abstractmethod
    async def notify_regulatory_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and communicate regulatory changes to users.
        
        Args:
            changes: List of regulatory changes to process
            
        Returns:
            Dictionary containing change processing results and user notifications
        """
        pass


class ScenarioAnalyzerInterface(ABC):
    """
    Interface for what-if scenario analysis and modeling.
    
    Defines methods for generating and analyzing business scenarios
    to support strategic decision making.
    """
    
    @abstractmethod
    async def generate_scenarios(self, base_parameters: Dict[str, Any]) -> List[Scenario]:
        """
        Generate multiple what-if scenarios from base parameters.
        
        Args:
            base_parameters: Base scenario parameters
            
        Returns:
            List of generated scenarios with variations
        """
        pass
    
    @abstractmethod
    async def predict_inventory_outcomes(self, scenario: Scenario) -> Dict[str, Any]:
        """
        Predict inventory outcomes for a given scenario.
        
        Args:
            scenario: Scenario to analyze
            
        Returns:
            Dictionary of predicted inventory outcomes
        """
        pass
    
    @abstractmethod
    async def analyze_discount_impact(self, scenario: Scenario) -> Dict[str, Any]:
        """
        Analyze impact of discount strategies in a scenario.
        
        Args:
            scenario: Scenario with discount parameters
            
        Returns:
            Dictionary of predicted discount impacts
        """
        pass
    
    @abstractmethod
    async def model_seasonal_effects(self, scenario: Scenario, seasonal_events: List[str]) -> Scenario:
        """
        Model seasonal effects on scenario outcomes.
        
        Args:
            scenario: Base scenario to enhance
            seasonal_events: List of seasonal events to consider
            
        Returns:
            Enhanced scenario with seasonal modeling
        """
        pass
    
    @abstractmethod
    async def validate_scenario_assumptions(self, scenario: Scenario) -> List[str]:
        """
        Validate and identify limitations in scenario assumptions.
        
        Args:
            scenario: Scenario to validate
            
        Returns:
            List of identified limitations and concerns
        """
        pass


class DecisionSupportEngineInterface(ABC):
    """
    Interface for the main decision support orchestration engine.
    
    Defines methods for coordinating all components to generate
    comprehensive business recommendations.
    """
    
    @abstractmethod
    async def generate_recommendations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive business recommendations.
        
        Args:
            request: Request parameters for recommendations
            
        Returns:
            Dictionary containing recommendations and supporting analysis
        """
        pass
    
    @abstractmethod
    async def optimize_discount_strategy(self, product_ids: List[str]) -> Dict[str, Any]:
        """
        Optimize discount strategy for specified products.
        
        Args:
            product_ids: List of product identifiers
            
        Returns:
            Optimized discount strategy recommendations
        """
        pass
    
    @abstractmethod
    async def assess_business_impact(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess potential business impact of recommendations.
        
        Args:
            recommendation: Recommendation to assess
            
        Returns:
            Business impact analysis results
        """
        pass
    
    @abstractmethod
    async def prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize recommendations by impact and urgency.
        
        Args:
            recommendations: List of recommendations to prioritize
            
        Returns:
            Prioritized list of recommendations
        """
        pass
    
    @abstractmethod
    async def validate_recommendation_pipeline(self, recommendation: Dict[str, Any]) -> ComplianceResult:
        """
        Validate recommendation through complete compliance pipeline.
        
        Args:
            recommendation: Recommendation to validate
            
        Returns:
            Comprehensive validation result
        """
        pass