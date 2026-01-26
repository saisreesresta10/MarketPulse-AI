"""
Model Update and Continuous Learning System

Implements mechanisms for integrating new data into existing models,
tracking accuracy, and adapting to changing market conditions.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4, UUID
import numpy as np
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.models import SalesDataPoint, DemandPattern, ConfidenceLevel
from ..storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Model update operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ModelType(Enum):
    """Types of models that can be updated."""
    DEMAND_FORECASTING = "demand_forecasting"
    SEASONAL_PATTERNS = "seasonal_patterns"
    RISK_ASSESSMENT = "risk_assessment"
    PRICE_SENSITIVITY = "price_sensitivity"


@dataclass
class ModelVersion:
    """Represents a version of a model."""
    version_id: str
    model_type: ModelType
    created_at: datetime
    accuracy_score: float
    training_data_size: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False


@dataclass
class UpdateOperation:
    """Tracks a model update operation."""
    operation_id: str
    model_type: ModelType
    started_at: datetime
    status: UpdateStatus
    new_data_points: int = 0
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_version: Optional[str] = None


class ModelUpdater:
    """
    Manages continuous learning and model updates.
    
    Features:
    - New data integration into existing models
    - Accuracy tracking and performance monitoring
    - Market condition adaptation
    - Model versioning and rollback capabilities
    """
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        
        # Model versions and tracking
        self.model_versions: Dict[ModelType, List[ModelVersion]] = defaultdict(list)
        self.update_operations: Dict[str, UpdateOperation] = {}
        
        # Performance tracking
        self.accuracy_history: Dict[ModelType, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_metrics: Dict[ModelType, Dict[str, float]] = defaultdict(dict)
        
        # Update scheduling
        self.update_intervals: Dict[ModelType, int] = {
            ModelType.DEMAND_FORECASTING: 24,  # hours
            ModelType.SEASONAL_PATTERNS: 168,  # weekly
            ModelType.RISK_ASSESSMENT: 12,     # 12 hours
            ModelType.PRICE_SENSITIVITY: 48    # 2 days
        }
        
        # Threading for background updates
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_running = True
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background monitoring for model updates."""
        def monitoring_loop():
            while self.is_running:
                try:
                    # Check if any models need updates
                    for model_type in ModelType:
                        if self._should_update_model(model_type):
                            asyncio.run(self._schedule_model_update(model_type))
                    
                    # Clean up old operations
                    self._cleanup_old_operations()
                    
                    # Sleep for 1 hour before next check
                    threading.Event().wait(3600)
                    
                except Exception as e:
                    logger.error(f"Error in model update monitoring: {e}")
                    threading.Event().wait(1800)  # Wait 30 minutes on error
        
        threading.Thread(target=monitoring_loop, daemon=True).start()
    
    def _should_update_model(self, model_type: ModelType) -> bool:
        """Check if a model should be updated based on schedule and data availability."""
        # Get the latest version
        versions = self.model_versions.get(model_type, [])
        if not versions:
            return True  # No model exists, should create one
        
        latest_version = max(versions, key=lambda v: v.created_at)
        
        # Check if enough time has passed
        update_interval = self.update_intervals.get(model_type, 24)
        time_since_update = datetime.now(timezone.utc) - latest_version.created_at
        
        if time_since_update.total_seconds() < (update_interval * 3600):
            return False
        
        # Check if there's enough new data
        try:
            new_data_count = asyncio.run(self._count_new_data_since(
                model_type, latest_version.created_at
            ))
            return new_data_count >= 10  # Minimum 10 new data points
        except Exception as e:
            logger.error(f"Error checking new data for {model_type}: {e}")
            return False
    
    async def _count_new_data_since(self, model_type: ModelType, since: datetime) -> int:
        """Count new data points since the given timestamp."""
        # This would query the storage for new data
        # For now, return a mock count
        return 15  # Mock value
    
    async def _schedule_model_update(self, model_type: ModelType):
        """Schedule a model update operation."""
        operation_id = str(uuid4())
        operation = UpdateOperation(
            operation_id=operation_id,
            model_type=model_type,
            started_at=datetime.now(timezone.utc),
            status=UpdateStatus.PENDING
        )
        
        self.update_operations[operation_id] = operation
        
        # Submit to thread pool for processing
        self.executor.submit(self._execute_model_update, operation_id)
        
        logger.info(f"Scheduled model update for {model_type.value}: {operation_id}")
    
    def _execute_model_update(self, operation_id: str):
        """Execute a model update operation."""
        operation = self.update_operations[operation_id]
        operation.status = UpdateStatus.IN_PROGRESS
        
        try:
            # Get current model performance
            current_versions = self.model_versions.get(operation.model_type, [])
            if current_versions:
                latest_version = max(current_versions, key=lambda v: v.created_at)
                operation.accuracy_before = latest_version.accuracy_score
                operation.rollback_version = latest_version.version_id
            
            # Simulate model training with new data
            new_version = self._train_updated_model(operation.model_type)
            operation.new_data_points = new_version.training_data_size
            operation.accuracy_after = new_version.accuracy_score
            
            # Evaluate if the new model is better
            if self._should_deploy_new_version(operation.model_type, new_version):
                self._deploy_model_version(new_version)
                operation.status = UpdateStatus.COMPLETED
                logger.info(f"Model update completed: {operation_id}")
            else:
                operation.status = UpdateStatus.FAILED
                operation.error_message = "New model performance not better than current"
                logger.warning(f"Model update rejected due to poor performance: {operation_id}")
            
        except Exception as e:
            operation.status = UpdateStatus.FAILED
            operation.error_message = str(e)
            logger.error(f"Model update failed: {operation_id} - {e}")
        
        finally:
            operation.completed_at = datetime.now(timezone.utc)
    
    def _train_updated_model(self, model_type: ModelType) -> ModelVersion:
        """Train an updated model with new data."""
        # This is a simplified simulation of model training
        # In a real implementation, this would:
        # 1. Load existing model parameters
        # 2. Fetch new training data
        # 3. Retrain or fine-tune the model
        # 4. Evaluate performance
        
        version_id = str(uuid4())
        
        # Simulate training process
        import time
        time.sleep(1)  # Simulate training time
        
        # Mock accuracy improvement (in reality, this would be measured)
        base_accuracy = 0.75
        improvement = np.random.normal(0.05, 0.02)  # Small improvement with variance
        new_accuracy = min(0.95, max(0.5, base_accuracy + improvement))
        
        new_version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            created_at=datetime.now(timezone.utc),
            accuracy_score=new_accuracy,
            training_data_size=np.random.randint(100, 1000),
            parameters={
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32,
                "model_architecture": f"{model_type.value}_v2"
            },
            metadata={
                "training_duration_seconds": 60,
                "data_sources": ["sales_data", "market_signals"],
                "validation_method": "time_series_split"
            }
        )
        
        return new_version
    
    def _should_deploy_new_version(self, model_type: ModelType, new_version: ModelVersion) -> bool:
        """Determine if a new model version should be deployed."""
        current_versions = self.model_versions.get(model_type, [])
        
        if not current_versions:
            return True  # No existing model, deploy the new one
        
        # Get current active version
        active_versions = [v for v in current_versions if v.is_active]
        if not active_versions:
            return True
        
        current_version = active_versions[0]
        
        # Deploy if accuracy improved by at least 1%
        accuracy_threshold = 0.01
        accuracy_improvement = new_version.accuracy_score - current_version.accuracy_score
        
        return accuracy_improvement >= accuracy_threshold
    
    def _deploy_model_version(self, new_version: ModelVersion):
        """Deploy a new model version."""
        # Deactivate current version
        current_versions = self.model_versions.get(new_version.model_type, [])
        for version in current_versions:
            version.is_active = False
        
        # Activate new version
        new_version.is_active = True
        self.model_versions[new_version.model_type].append(new_version)
        
        # Update performance tracking
        self.accuracy_history[new_version.model_type].append({
            'timestamp': new_version.created_at,
            'accuracy': new_version.accuracy_score,
            'version_id': new_version.version_id
        })
        
        logger.info(f"Deployed new model version: {new_version.version_id} for {new_version.model_type.value}")
    
    async def integrate_new_data(self, data_points: List[SalesDataPoint]) -> Dict[str, Any]:
        """
        Integrate new data points and trigger model updates if needed.
        
        Args:
            data_points: New sales data points to integrate
            
        Returns:
            Integration summary with update recommendations
        """
        integration_summary = {
            "data_points_processed": len(data_points),
            "models_affected": [],
            "update_operations_triggered": [],
            "integration_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Store new data points
            for data_point in data_points:
                await self.storage_manager.store_sales_data(data_point)
            
            # Analyze which models might be affected
            affected_models = self._analyze_data_impact(data_points)
            integration_summary["models_affected"] = [m.value for m in affected_models]
            
            # Trigger updates for significantly affected models
            for model_type in affected_models:
                if self._data_impact_significant(model_type, data_points):
                    operation_id = await self._schedule_model_update(model_type)
                    integration_summary["update_operations_triggered"].append(operation_id)
            
            logger.info(f"Integrated {len(data_points)} new data points")
            return integration_summary
            
        except Exception as e:
            logger.error(f"Error integrating new data: {e}")
            integration_summary["error"] = str(e)
            return integration_summary
    
    def _analyze_data_impact(self, data_points: List[SalesDataPoint]) -> List[ModelType]:
        """Analyze which models might be impacted by new data."""
        affected_models = []
        
        # Check for seasonal patterns
        seasonal_indicators = any(dp.seasonal_event for dp in data_points)
        if seasonal_indicators:
            affected_models.append(ModelType.SEASONAL_PATTERNS)
        
        # Check for demand changes
        quantities = [dp.quantity_sold for dp in data_points]
        if quantities and (max(quantities) > 100 or min(quantities) == 0):
            affected_models.extend([
                ModelType.DEMAND_FORECASTING,
                ModelType.RISK_ASSESSMENT
            ])
        
        # Check for price sensitivity data
        price_variations = len(set(dp.selling_price for dp in data_points)) > 1
        if price_variations:
            affected_models.append(ModelType.PRICE_SENSITIVITY)
        
        return list(set(affected_models))  # Remove duplicates
    
    def _data_impact_significant(self, model_type: ModelType, data_points: List[SalesDataPoint]) -> bool:
        """Determine if the data impact is significant enough to trigger an update."""
        # Simple heuristic: if we have more than 20 new data points, it's significant
        return len(data_points) >= 20
    
    async def track_model_accuracy(self, model_type: ModelType, predictions: List[Any], 
                                 actual_values: List[Any]) -> Dict[str, float]:
        """
        Track model accuracy by comparing predictions with actual values.
        
        Args:
            model_type: Type of model being evaluated
            predictions: Model predictions
            actual_values: Actual observed values
            
        Returns:
            Accuracy metrics
        """
        if len(predictions) != len(actual_values):
            raise ValueError("Predictions and actual values must have the same length")
        
        # Calculate various accuracy metrics
        metrics = {}
        
        try:
            # Mean Absolute Error
            mae = np.mean(np.abs(np.array(predictions) - np.array(actual_values)))
            metrics["mean_absolute_error"] = float(mae)
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actual_values)) ** 2))
            metrics["root_mean_square_error"] = float(rmse)
            
            # Accuracy score (for classification-like problems)
            if all(isinstance(p, (int, bool)) for p in predictions):
                accuracy = np.mean(np.array(predictions) == np.array(actual_values))
                metrics["accuracy_score"] = float(accuracy)
            
            # Update performance tracking
            self.performance_metrics[model_type].update(metrics)
            
            # Store accuracy history
            self.accuracy_history[model_type].append({
                'timestamp': datetime.now(timezone.utc),
                'metrics': metrics,
                'sample_size': len(predictions)
            })
            
            logger.info(f"Updated accuracy metrics for {model_type.value}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {"error": str(e)}
    
    async def adapt_to_market_conditions(self, market_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt models based on changing market conditions.
        
        Args:
            market_signals: Dictionary of market condition indicators
            
        Returns:
            Adaptation summary
        """
        adaptation_summary = {
            "market_signals_processed": len(market_signals),
            "adaptations_made": [],
            "models_updated": [],
            "adaptation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Analyze market condition changes
            significant_changes = self._detect_market_changes(market_signals)
            
            for change_type, change_magnitude in significant_changes.items():
                # Determine which models need adaptation
                affected_models = self._get_models_for_market_change(change_type)
                
                for model_type in affected_models:
                    # Adjust model parameters or trigger retraining
                    adaptation_result = await self._adapt_model_to_change(
                        model_type, change_type, change_magnitude
                    )
                    
                    if adaptation_result["adapted"]:
                        adaptation_summary["adaptations_made"].append({
                            "model_type": model_type.value,
                            "change_type": change_type,
                            "adaptation_method": adaptation_result["method"]
                        })
                        adaptation_summary["models_updated"].append(model_type.value)
            
            logger.info(f"Market adaptation completed: {len(adaptation_summary['adaptations_made'])} adaptations made")
            return adaptation_summary
            
        except Exception as e:
            logger.error(f"Error in market adaptation: {e}")
            adaptation_summary["error"] = str(e)
            return adaptation_summary
    
    def _detect_market_changes(self, market_signals: Dict[str, Any]) -> Dict[str, float]:
        """Detect significant changes in market conditions."""
        significant_changes = {}
        
        # Example market signals and thresholds
        thresholds = {
            "inflation_rate": 0.02,      # 2% change
            "consumer_confidence": 0.1,   # 10% change
            "seasonal_index": 0.15,       # 15% change
            "competition_index": 0.05     # 5% change
        }
        
        for signal, value in market_signals.items():
            if signal in thresholds:
                # Compare with historical baseline (simplified)
                baseline = 1.0  # Normalized baseline
                change_magnitude = abs(value - baseline) / baseline
                
                if change_magnitude >= thresholds[signal]:
                    significant_changes[signal] = change_magnitude
        
        return significant_changes
    
    def _get_models_for_market_change(self, change_type: str) -> List[ModelType]:
        """Get models that should be adapted for a specific market change."""
        model_mapping = {
            "inflation_rate": [ModelType.PRICE_SENSITIVITY, ModelType.DEMAND_FORECASTING],
            "consumer_confidence": [ModelType.DEMAND_FORECASTING, ModelType.RISK_ASSESSMENT],
            "seasonal_index": [ModelType.SEASONAL_PATTERNS, ModelType.DEMAND_FORECASTING],
            "competition_index": [ModelType.PRICE_SENSITIVITY, ModelType.RISK_ASSESSMENT]
        }
        
        return model_mapping.get(change_type, [])
    
    async def _adapt_model_to_change(self, model_type: ModelType, change_type: str, 
                                   magnitude: float) -> Dict[str, Any]:
        """Adapt a specific model to a market change."""
        adaptation_result = {
            "adapted": False,
            "method": None,
            "parameters_changed": []
        }
        
        try:
            # Get current active model
            current_versions = self.model_versions.get(model_type, [])
            active_versions = [v for v in current_versions if v.is_active]
            
            if not active_versions:
                return adaptation_result
            
            current_version = active_versions[0]
            
            # Determine adaptation method based on change magnitude
            if magnitude > 0.2:  # Large change - retrain model
                await self._schedule_model_update(model_type)
                adaptation_result["adapted"] = True
                adaptation_result["method"] = "full_retrain"
                
            elif magnitude > 0.1:  # Medium change - adjust parameters
                # Simulate parameter adjustment
                new_parameters = current_version.parameters.copy()
                
                if change_type == "inflation_rate":
                    new_parameters["price_sensitivity_factor"] = new_parameters.get("price_sensitivity_factor", 1.0) * (1 + magnitude)
                elif change_type == "seasonal_index":
                    new_parameters["seasonal_weight"] = new_parameters.get("seasonal_weight", 1.0) * (1 + magnitude)
                
                # Create updated version with new parameters
                updated_version = ModelVersion(
                    version_id=str(uuid4()),
                    model_type=model_type,
                    created_at=datetime.now(timezone.utc),
                    accuracy_score=current_version.accuracy_score * 0.98,  # Slight decrease due to adaptation
                    training_data_size=current_version.training_data_size,
                    parameters=new_parameters,
                    metadata={
                        **current_version.metadata,
                        "adapted_from": current_version.version_id,
                        "adaptation_reason": f"{change_type}_change",
                        "adaptation_magnitude": magnitude
                    }
                )
                
                self._deploy_model_version(updated_version)
                adaptation_result["adapted"] = True
                adaptation_result["method"] = "parameter_adjustment"
                adaptation_result["parameters_changed"] = list(new_parameters.keys())
            
            return adaptation_result
            
        except Exception as e:
            logger.error(f"Error adapting model {model_type.value}: {e}")
            adaptation_result["error"] = str(e)
            return adaptation_result
    
    def get_model_performance_summary(self, model_type: Optional[ModelType] = None) -> Dict[str, Any]:
        """Get performance summary for models."""
        if model_type:
            model_types = [model_type]
        else:
            model_types = list(ModelType)
        
        summary = {
            "summary_timestamp": datetime.now(timezone.utc).isoformat(),
            "models": {}
        }
        
        for mt in model_types:
            model_summary = {
                "model_type": mt.value,
                "active_version": None,
                "total_versions": len(self.model_versions.get(mt, [])),
                "latest_accuracy": None,
                "accuracy_trend": None,
                "last_update": None,
                "performance_metrics": self.performance_metrics.get(mt, {})
            }
            
            # Get active version info
            versions = self.model_versions.get(mt, [])
            active_versions = [v for v in versions if v.is_active]
            if active_versions:
                active_version = active_versions[0]
                model_summary["active_version"] = active_version.version_id
                model_summary["latest_accuracy"] = active_version.accuracy_score
                model_summary["last_update"] = active_version.created_at.isoformat()
            
            # Calculate accuracy trend
            accuracy_history = list(self.accuracy_history.get(mt, []))
            if len(accuracy_history) >= 2:
                recent_accuracy = accuracy_history[-1].get("accuracy", 0)
                previous_accuracy = accuracy_history[-2].get("accuracy", 0)
                trend = "improving" if recent_accuracy > previous_accuracy else "declining"
                model_summary["accuracy_trend"] = trend
            
            summary["models"][mt.value] = model_summary
        
        return summary
    
    def rollback_model(self, model_type: ModelType, target_version_id: str) -> Dict[str, Any]:
        """Rollback a model to a previous version."""
        rollback_result = {
            "success": False,
            "model_type": model_type.value,
            "target_version": target_version_id,
            "rollback_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Find the target version
            versions = self.model_versions.get(model_type, [])
            target_version = None
            
            for version in versions:
                if version.version_id == target_version_id:
                    target_version = version
                    break
            
            if not target_version:
                rollback_result["error"] = f"Version {target_version_id} not found"
                return rollback_result
            
            # Deactivate current version
            for version in versions:
                version.is_active = False
            
            # Activate target version
            target_version.is_active = True
            
            rollback_result["success"] = True
            rollback_result["previous_accuracy"] = target_version.accuracy_score
            
            logger.info(f"Rolled back {model_type.value} to version {target_version_id}")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            rollback_result["error"] = str(e)
            return rollback_result
    
    def _cleanup_old_operations(self):
        """Clean up old update operations."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        operations_to_remove = []
        for op_id, operation in self.update_operations.items():
            if (operation.completed_at and operation.completed_at < cutoff_time):
                operations_to_remove.append(op_id)
        
        for op_id in operations_to_remove:
            del self.update_operations[op_id]
        
        if operations_to_remove:
            logger.info(f"Cleaned up {len(operations_to_remove)} old update operations")
    
    def get_update_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a model update operation."""
        operation = self.update_operations.get(operation_id)
        if not operation:
            return None
        
        return {
            "operation_id": operation_id,
            "model_type": operation.model_type.value,
            "status": operation.status.value,
            "started_at": operation.started_at.isoformat(),
            "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
            "new_data_points": operation.new_data_points,
            "accuracy_before": operation.accuracy_before,
            "accuracy_after": operation.accuracy_after,
            "error_message": operation.error_message
        }
    
    def shutdown(self):
        """Shutdown the model updater."""
        logger.info("Shutting down model updater")
        self.is_running = False
        self.executor.shutdown(wait=True)