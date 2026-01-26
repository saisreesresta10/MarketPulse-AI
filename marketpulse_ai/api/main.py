"""
MarketPulse AI REST API

FastAPI-based REST API providing endpoints for insights, recommendations,
scenarios, and system management.
"""

from datetime import datetime
from typing import Dict, Any
from uuid import uuid4
import logging
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from cryptography.fernet import Fernet

from ..components import (
    DataProcessor, RiskAssessor, ComplianceValidator, 
    InsightGenerator, DecisionSupportEngine, ScenarioAnalyzer
)
from ..config.settings import get_settings
from ..storage.factory import StorageFactory
from .models import APIResponse, ErrorResponse
from .routers import (
    insights_router, recommendations_router, data_router, workflows_router
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MarketPulse AI API",
    description="AI-powered retail insights and decision support system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
components: Dict[str, Any] = {}

class ComponentManager:
    """Manages component initialization and dependency injection."""
    
    def __init__(self):
        self.components = {}
        self.initialized = False
    
    async def initialize_components(self, testing_mode: bool = False):
        """Initialize all system components with proper dependency injection."""
        if self.initialized:
            return
        
        try:
            logger.info("Initializing MarketPulse AI components...")
            
            # Initialize core infrastructure
            from ..storage.storage_manager import StorageManager
            from ..config.database import DatabaseManager
            from ..config.security import SecurityConfig
            from ..components.model_updater import ModelUpdater
            from ..components.feedback_learner import FeedbackLearner
            from .load_manager import LoadManager
            from .backup_manager import BackupManager
            
            # Database and security setup
            if testing_mode:
                from ..config.database import DatabaseConfig
                db_config = DatabaseConfig(url="sqlite:///test_marketpulse.db")
                db_manager = DatabaseManager(db_config)
            else:
                settings = get_settings()
                from ..config.database import DatabaseConfig
                db_config = DatabaseConfig(url=settings.database_url)
                db_manager = DatabaseManager(db_config)
            
            security_config = SecurityConfig(
                secret_key="test_secret_key_for_testing_only_32_chars",
                encryption_key=Fernet.generate_key().decode()
            )
            storage_manager = StorageManager(db_manager, security_config)
            
            # Initialize core AI components
            data_processor = DataProcessor(storage_manager)
            risk_assessor = RiskAssessor(storage_manager)
            compliance_validator = ComplianceValidator(storage_manager)
            insight_generator = InsightGenerator(storage_manager)
            scenario_analyzer = ScenarioAnalyzer(storage_manager)
            
            # Initialize learning components
            model_updater = ModelUpdater(storage_manager)
            feedback_learner = FeedbackLearner(storage_manager)
            
            # Initialize system reliability components
            load_manager = LoadManager()
            backup_manager = BackupManager()
            
            # Initialize decision support engine with correct dependencies
            decision_support_engine = DecisionSupportEngine(
                data_processor=data_processor,
                risk_assessor=risk_assessor,
                compliance_validator=compliance_validator,
                insight_generator=insight_generator
            )
            
            # Initialize main orchestrator
            from ..orchestrator import MarketPulseOrchestrator
            orchestrator = MarketPulseOrchestrator(
                data_processor=data_processor,
                risk_assessor=risk_assessor,
                compliance_validator=compliance_validator,
                insight_generator=insight_generator,
                decision_support_engine=decision_support_engine,
                scenario_analyzer=scenario_analyzer,
                model_updater=model_updater,
                feedback_learner=feedback_learner,
                storage_manager=storage_manager
            )
            
            # Store components
            self.components = {
                "storage_manager": storage_manager,
                "data_processor": data_processor,
                "risk_assessor": risk_assessor,
                "compliance_validator": compliance_validator,
                "insight_generator": insight_generator,
                "scenario_analyzer": scenario_analyzer,
                "decision_support_engine": decision_support_engine,
                "model_updater": model_updater,
                "feedback_learner": feedback_learner,
                "load_manager": load_manager,
                "backup_manager": backup_manager,
                "orchestrator": orchestrator
            }
            
            # Update global components for backward compatibility
            global components
            components.update(self.components)
            
            self.initialized = True
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            if testing_mode:
                # Create mock components for testing
                from unittest.mock import AsyncMock
                mock_components = {
                    "data_processor": AsyncMock(),
                    "risk_assessor": AsyncMock(),
                    "compliance_validator": AsyncMock(),
                    "insight_generator": AsyncMock(),
                    "scenario_analyzer": AsyncMock(),
                    "decision_support_engine": AsyncMock(),
                    "model_updater": AsyncMock(),
                    "feedback_learner": AsyncMock(),
                    "load_manager": AsyncMock(),
                    "backup_manager": AsyncMock(),
                    "orchestrator": AsyncMock()
                }
                self.components.update(mock_components)
                components.update(mock_components)
                logger.info("Mock components initialized for testing")
            else:
                raise
    
    def get_component(self, name: str):
        """Get a component by name."""
        return self.components.get(name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Get all initialized components."""
        return self.components.copy()
    
    async def shutdown_components(self):
        """Shutdown all components gracefully."""
        logger.info("Shutting down components...")
        
        # Shutdown components that need cleanup
        if "model_updater" in self.components:
            self.components["model_updater"].shutdown()
        
        if "feedback_learner" in self.components:
            self.components["feedback_learner"].shutdown()
        
        if "load_manager" in self.components:
            self.components["load_manager"].shutdown()
        
        if "backup_manager" in self.components:
            self.components["backup_manager"].shutdown()
        
        logger.info("All components shut down")

# Global component manager
component_manager = ComponentManager()

def initialize_components_for_testing():
    """Initialize components for testing environment."""
    import asyncio
    asyncio.run(component_manager.initialize_components(testing_mode=True))

# Include routers
app.include_router(insights_router, prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(data_router, prefix="/api/v1")
app.include_router(workflows_router, prefix="/api/v1")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        # Initialize all components
        await component_manager.initialize_components(testing_mode=False)
        logger.info("MarketPulse AI API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        # Don't raise in test environment
        if "pytest" not in str(e):
            raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await component_manager.shutdown_components()
    logger.info("MarketPulse AI API shut down successfully")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    # Convert detail to string if it's not already
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    
    error_response = ErrorResponse(
        error="HTTPException",
        message=message,
        request_id=str(uuid4())
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(error_response.model_dump())
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An internal server error occurred",
        details={"exception": str(exc)},
        request_id=str(uuid4())
    )
    
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(error_response.model_dump())
    )

# Health check endpoint
@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint."""
    response = APIResponse(
        success=True,
        data={"status": "healthy", "components": list(components.keys())},
        message="MarketPulse AI API is running",
        request_id=str(uuid4())
    )
    return JSONResponse(content=jsonable_encoder(response.model_dump()))

# System status endpoint
@app.get("/api/v1/system/status", response_model=APIResponse)
async def get_system_status():
    """Get comprehensive system status."""
    request_id = str(uuid4())
    
    try:
        all_components = component_manager.get_all_components()
        
        # Get component health status
        component_health = {}
        for name, component in all_components.items():
            try:
                if hasattr(component, 'get_system_status'):
                    component_health[name] = "healthy"
                else:
                    component_health[name] = "initialized"
            except Exception as e:
                component_health[name] = f"error: {str(e)}"
        
        # Get system metrics if load manager is available
        system_metrics = {}
        if "load_manager" in all_components:
            try:
                system_metrics = all_components["load_manager"].get_system_status()
            except Exception as e:
                logger.warning(f"Could not get system metrics: {e}")
        
        # Get backup status if backup manager is available
        backup_status = {}
        if "backup_manager" in all_components:
            try:
                backup_status = all_components["backup_manager"].get_system_status()
            except Exception as e:
                logger.warning(f"Could not get backup status: {e}")
        
        status = {
            "api_version": "1.0.0",
            "components_initialized": len(all_components),
            "component_health": component_health,
            "system_metrics": system_metrics,
            "backup_status": backup_status,
            "available_endpoints": {
                "data_ingestion": ["/api/v1/data/ingest", "/api/v1/data/patterns"],
                "insights": ["/api/v1/insights/generate", "/api/v1/insights/{product_id}"],
                "recommendations": [
                    "/api/v1/recommendations/generate",
                    "/api/v1/recommendations/optimize-discount",
                    "/api/v1/recommendations/{product_id}/impact"
                ]
            }
        }
        
        response = APIResponse(
            success=True,
            data=status,
            message="System status retrieved successfully",
            request_id=request_id
        )
        
        return JSONResponse(content=jsonable_encoder(response.model_dump()))
        
    except Exception as e:
        logger.error(f"System status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System status retrieval failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)