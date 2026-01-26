"""
Feedback Learning System

Implements retailer feedback collection, processing, and integration
into model improvement workflows with seasonal model evolution.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4, UUID
import numpy as np
from collections import defaultdict, deque
import threading

from ..core.models import ConfidenceLevel
from ..storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    RECOMMENDATION_RATING = "recommendation_rating"
    INSIGHT_ACCURACY = "insight_accuracy"
    PREDICTION_OUTCOME = "prediction_outcome"
    USER_CORRECTION = "user_correction"
    BUSINESS_IMPACT = "business_impact"


class FeedbackSentiment(Enum):
    """Sentiment of feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class FeedbackEntry:
    """Represents a single feedback entry from a retailer."""
    feedback_id: str
    retailer_id: str
    feedback_type: FeedbackType
    target_id: str  # ID of recommendation, insight, etc.
    rating: Optional[float] = None  # 1-5 scale
    sentiment: Optional[FeedbackSentiment] = None
    text_feedback: Optional[str] = None
    structured_data: Dict[str, Any] = field(default_factory=dict)
    business_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed: bool = False
    impact_measured: bool = False


@dataclass
class SeasonalPattern:
    """Represents a seasonal pattern learned from feedback."""
    pattern_id: str
    season_name: str
    pattern_type: str
    confidence_score: float
    feedback_count: int
    accuracy_improvement: float
    last_updated: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)


class FeedbackLearner:
    """
    Manages feedback collection and learning from retailer interactions.
    
    Features:
    - Multi-channel feedback collection
    - Sentiment analysis and rating processing
    - Seasonal model evolution based on feedback
    - Performance monitoring and improvement tracking
    """
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        
        # Feedback storage and processing
        self.feedback_entries: Dict[str, FeedbackEntry] = {}
        self.feedback_queue: deque = deque()
        
        # Learning and improvement tracking
        self.seasonal_patterns: Dict[str, SeasonalPattern] = {}
        self.improvement_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.feedback_analytics: Dict[str, Any] = defaultdict(dict)
        
        # Processing configuration
        self.batch_size = 50
        self.processing_interval = 3600  # 1 hour
        self.min_feedback_for_learning = 10
        
        # Background processing
        self.is_running = True
        self._start_background_processing()
    
    def _start_background_processing(self):
        """Start background feedback processing."""
        def processing_loop():
            while self.is_running:
                try:
                    # Process pending feedback
                    asyncio.run(self._process_feedback_batch())
                    
                    # Update seasonal patterns
                    asyncio.run(self._update_seasonal_patterns())
                    
                    # Generate improvement recommendations
                    asyncio.run(self._generate_improvement_recommendations())
                    
                    # Sleep until next processing cycle
                    threading.Event().wait(self.processing_interval)
                    
                except Exception as e:
                    logger.error(f"Error in feedback processing loop: {e}")
                    threading.Event().wait(1800)  # Wait 30 minutes on error
        
        threading.Thread(target=processing_loop, daemon=True).start()
    
    async def collect_feedback(self, retailer_id: str, feedback_type: FeedbackType,
                             target_id: str, **kwargs) -> str:
        """
        Collect feedback from a retailer.
        
        Args:
            retailer_id: ID of the retailer providing feedback
            feedback_type: Type of feedback being provided
            target_id: ID of the item being rated (recommendation, insight, etc.)
            **kwargs: Additional feedback data (rating, text, etc.)
            
        Returns:
            Feedback ID for tracking
        """
        feedback_id = str(uuid4())
        
        feedback_entry = FeedbackEntry(
            feedback_id=feedback_id,
            retailer_id=retailer_id,
            feedback_type=feedback_type,
            target_id=target_id,
            rating=kwargs.get('rating'),
            sentiment=kwargs.get('sentiment'),
            text_feedback=kwargs.get('text_feedback'),
            structured_data=kwargs.get('structured_data', {}),
            business_context=kwargs.get('business_context', {})
        )
        
        # Store feedback
        self.feedback_entries[feedback_id] = feedback_entry
        self.feedback_queue.append(feedback_id)
        
        logger.info(f"Collected feedback: {feedback_id} from retailer {retailer_id}")
        return feedback_id
    
    async def rate_recommendation(self, retailer_id: str, recommendation_id: str,
                                rating: float, business_impact: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect rating feedback for a recommendation.
        
        Args:
            retailer_id: ID of the retailer
            recommendation_id: ID of the recommendation being rated
            rating: Rating from 1-5
            business_impact: Optional business impact data
            
        Returns:
            Feedback ID
        """
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        sentiment = FeedbackSentiment.POSITIVE if rating >= 4 else (
            FeedbackSentiment.NEGATIVE if rating <= 2 else FeedbackSentiment.NEUTRAL
        )
        
        return await self.collect_feedback(
            retailer_id=retailer_id,
            feedback_type=FeedbackType.RECOMMENDATION_RATING,
            target_id=recommendation_id,
            rating=rating,
            sentiment=sentiment,
            business_context=business_impact or {}
        )
    
    async def report_prediction_outcome(self, retailer_id: str, prediction_id: str,
                                      actual_outcome: Dict[str, Any], 
                                      predicted_outcome: Dict[str, Any]) -> str:
        """
        Report the actual outcome of a prediction for learning.
        
        Args:
            retailer_id: ID of the retailer
            prediction_id: ID of the prediction
            actual_outcome: What actually happened
            predicted_outcome: What was predicted
            
        Returns:
            Feedback ID
        """
        # Calculate accuracy metrics
        accuracy_data = self._calculate_prediction_accuracy(predicted_outcome, actual_outcome)
        
        return await self.collect_feedback(
            retailer_id=retailer_id,
            feedback_type=FeedbackType.PREDICTION_OUTCOME,
            target_id=prediction_id,
            structured_data={
                "actual_outcome": actual_outcome,
                "predicted_outcome": predicted_outcome,
                "accuracy_metrics": accuracy_data
            }
        )
    
    def _calculate_prediction_accuracy(self, predicted: Dict[str, Any], 
                                     actual: Dict[str, Any]) -> Dict[str, float]:
        """Calculate accuracy metrics for predictions."""
        accuracy_metrics = {}
        
        for key in predicted.keys():
            if key in actual:
                pred_val = predicted[key]
                actual_val = actual[key]
                
                if isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                    # Calculate percentage error
                    if actual_val != 0:
                        error = abs(pred_val - actual_val) / abs(actual_val)
                        accuracy_metrics[f"{key}_accuracy"] = max(0, 1 - error)
                    else:
                        accuracy_metrics[f"{key}_accuracy"] = 1.0 if pred_val == 0 else 0.0
        
        return accuracy_metrics
    
    async def submit_user_correction(self, retailer_id: str, item_id: str,
                                   correction_data: Dict[str, Any]) -> str:
        """
        Submit a user correction for model improvement.
        
        Args:
            retailer_id: ID of the retailer
            item_id: ID of the item being corrected
            correction_data: Correction information
            
        Returns:
            Feedback ID
        """
        return await self.collect_feedback(
            retailer_id=retailer_id,
            feedback_type=FeedbackType.USER_CORRECTION,
            target_id=item_id,
            structured_data=correction_data,
            sentiment=FeedbackSentiment.NEUTRAL
        )
    
    async def _process_feedback_batch(self):
        """Process a batch of pending feedback entries."""
        if not self.feedback_queue:
            return
        
        # Get batch of feedback to process
        batch_size = min(self.batch_size, len(self.feedback_queue))
        batch_ids = [self.feedback_queue.popleft() for _ in range(batch_size)]
        
        processed_count = 0
        for feedback_id in batch_ids:
            try:
                feedback = self.feedback_entries.get(feedback_id)
                if feedback and not feedback.processed:
                    await self._process_single_feedback(feedback)
                    feedback.processed = True
                    processed_count += 1
            except Exception as e:
                logger.error(f"Error processing feedback {feedback_id}: {e}")
        
        if processed_count > 0:
            logger.info(f"Processed {processed_count} feedback entries")
    
    async def _process_single_feedback(self, feedback: FeedbackEntry):
        """Process a single feedback entry."""
        # Update analytics
        self._update_feedback_analytics(feedback)
        
        # Extract learning signals
        learning_signals = self._extract_learning_signals(feedback)
        
        # Apply learning based on feedback type
        if feedback.feedback_type == FeedbackType.RECOMMENDATION_RATING:
            await self._learn_from_recommendation_rating(feedback, learning_signals)
        elif feedback.feedback_type == FeedbackType.PREDICTION_OUTCOME:
            await self._learn_from_prediction_outcome(feedback, learning_signals)
        elif feedback.feedback_type == FeedbackType.USER_CORRECTION:
            await self._learn_from_user_correction(feedback, learning_signals)
        
        # Store processed feedback
        await self._store_processed_feedback(feedback)
    
    def _update_feedback_analytics(self, feedback: FeedbackEntry):
        """Update feedback analytics and metrics."""
        feedback_type = feedback.feedback_type.value
        
        # Update counts
        if "total_count" not in self.feedback_analytics[feedback_type]:
            self.feedback_analytics[feedback_type]["total_count"] = 0
        self.feedback_analytics[feedback_type]["total_count"] += 1
        
        # Update ratings
        if feedback.rating is not None:
            if "ratings" not in self.feedback_analytics[feedback_type]:
                self.feedback_analytics[feedback_type]["ratings"] = []
            self.feedback_analytics[feedback_type]["ratings"].append(feedback.rating)
        
        # Update sentiment distribution
        if feedback.sentiment:
            sentiment_key = f"sentiment_{feedback.sentiment.value}"
            if sentiment_key not in self.feedback_analytics[feedback_type]:
                self.feedback_analytics[feedback_type][sentiment_key] = 0
            self.feedback_analytics[feedback_type][sentiment_key] += 1
    
    def _extract_learning_signals(self, feedback: FeedbackEntry) -> Dict[str, Any]:
        """Extract learning signals from feedback."""
        signals = {
            "feedback_strength": self._calculate_feedback_strength(feedback),
            "business_impact": feedback.business_context.get("impact_score", 0),
            "retailer_credibility": self._get_retailer_credibility(feedback.retailer_id),
            "seasonal_context": self._get_seasonal_context(feedback.created_at)
        }
        
        # Add type-specific signals
        if feedback.feedback_type == FeedbackType.PREDICTION_OUTCOME:
            accuracy_metrics = feedback.structured_data.get("accuracy_metrics", {})
            signals["prediction_accuracy"] = np.mean(list(accuracy_metrics.values())) if accuracy_metrics else 0
        
        return signals
    
    def _calculate_feedback_strength(self, feedback: FeedbackEntry) -> float:
        """Calculate the strength/weight of feedback."""
        strength = 0.5  # Base strength
        
        # Adjust based on rating
        if feedback.rating is not None:
            if feedback.rating >= 4 or feedback.rating <= 2:
                strength += 0.3  # Strong positive or negative
            else:
                strength += 0.1  # Neutral
        
        # Adjust based on text feedback presence
        if feedback.text_feedback:
            strength += 0.2
        
        # Adjust based on structured data
        if feedback.structured_data:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _get_retailer_credibility(self, retailer_id: str) -> float:
        """Get credibility score for a retailer based on feedback history."""
        # Count feedback from this retailer
        retailer_feedback = [f for f in self.feedback_entries.values() 
                           if f.retailer_id == retailer_id and f.processed]
        
        if len(retailer_feedback) < 5:
            return 0.5  # Default credibility for new retailers
        
        # Calculate credibility based on consistency and accuracy
        ratings = [f.rating for f in retailer_feedback if f.rating is not None]
        if ratings:
            rating_variance = np.var(ratings)
            # Lower variance = higher credibility
            credibility = max(0.3, 1.0 - (rating_variance / 4.0))
        else:
            credibility = 0.5
        
        return credibility
    
    def _get_seasonal_context(self, timestamp: datetime) -> Dict[str, Any]:
        """Get seasonal context for the feedback timestamp."""
        month = timestamp.month
        
        # Define seasons (Northern Hemisphere)
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "autumn"
        
        # Indian festival seasons (approximate)
        festival_season = None
        if month in [10, 11]:  # Diwali season
            festival_season = "diwali"
        elif month in [8, 9]:  # Ganesh Chaturthi
            festival_season = "ganesh"
        elif month in [3, 4]:  # Holi
            festival_season = "holi"
        
        return {
            "season": season,
            "month": month,
            "festival_season": festival_season
        }
    
    async def _learn_from_recommendation_rating(self, feedback: FeedbackEntry, 
                                              signals: Dict[str, Any]):
        """Learn from recommendation rating feedback."""
        if feedback.rating is None:
            return
        
        # Update recommendation performance tracking
        target_id = feedback.target_id
        rating = feedback.rating
        strength = signals["feedback_strength"]
        
        # Store learning for recommendation improvement
        learning_data = {
            "recommendation_id": target_id,
            "rating": rating,
            "weighted_rating": rating * strength,
            "business_context": feedback.business_context,
            "seasonal_context": signals["seasonal_context"],
            "retailer_credibility": signals["retailer_credibility"]
        }
        
        # Update seasonal patterns if applicable
        seasonal_context = signals["seasonal_context"]
        if seasonal_context.get("festival_season"):
            await self._update_seasonal_learning(
                seasonal_context["festival_season"],
                "recommendation_rating",
                rating,
                strength
            )
    
    async def _learn_from_prediction_outcome(self, feedback: FeedbackEntry, 
                                           signals: Dict[str, Any]):
        """Learn from prediction outcome feedback."""
        accuracy = signals.get("prediction_accuracy", 0)
        
        if accuracy > 0:
            # Update prediction model performance
            prediction_data = {
                "prediction_id": feedback.target_id,
                "accuracy": accuracy,
                "actual_outcome": feedback.structured_data.get("actual_outcome"),
                "predicted_outcome": feedback.structured_data.get("predicted_outcome"),
                "seasonal_context": signals["seasonal_context"]
            }
            
            # Learn seasonal prediction patterns
            seasonal_context = signals["seasonal_context"]
            await self._update_seasonal_learning(
                seasonal_context["season"],
                "prediction_accuracy",
                accuracy,
                signals["feedback_strength"]
            )
    
    async def _learn_from_user_correction(self, feedback: FeedbackEntry, 
                                        signals: Dict[str, Any]):
        """Learn from user correction feedback."""
        correction_data = feedback.structured_data
        
        # Extract correction patterns
        correction_learning = {
            "item_id": feedback.target_id,
            "correction_data": correction_data,
            "seasonal_context": signals["seasonal_context"],
            "strength": signals["feedback_strength"]
        }
        
        # Update model correction patterns
        await self._apply_user_corrections(correction_learning)
    
    async def _update_seasonal_learning(self, season_key: str, pattern_type: str,
                                      value: float, strength: float):
        """Update seasonal learning patterns."""
        pattern_id = f"{season_key}_{pattern_type}"
        
        if pattern_id not in self.seasonal_patterns:
            self.seasonal_patterns[pattern_id] = SeasonalPattern(
                pattern_id=pattern_id,
                season_name=season_key,
                pattern_type=pattern_type,
                confidence_score=0.5,
                feedback_count=0,
                accuracy_improvement=0.0,
                last_updated=datetime.now(timezone.utc)
            )
        
        pattern = self.seasonal_patterns[pattern_id]
        
        # Update pattern with new feedback
        pattern.feedback_count += 1
        pattern.last_updated = datetime.now(timezone.utc)
        
        # Update confidence and accuracy
        weighted_value = value * strength
        current_weight = pattern.confidence_score * pattern.feedback_count
        new_weight = current_weight + weighted_value
        pattern.confidence_score = new_weight / (pattern.feedback_count + 1)
        
        # Calculate accuracy improvement
        if pattern.feedback_count > 1:
            baseline_accuracy = 0.7  # Assumed baseline
            pattern.accuracy_improvement = max(0, pattern.confidence_score - baseline_accuracy)
        
        logger.info(f"Updated seasonal pattern {pattern_id}: confidence={pattern.confidence_score:.3f}")
    
    async def _apply_user_corrections(self, correction_learning: Dict[str, Any]):
        """Apply user corrections to improve models."""
        # This would integrate with the model updater to apply corrections
        # For now, we'll store the correction for future model updates
        correction_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correction": correction_learning
        }
        
        # Store correction for model updater to process
        logger.info(f"Applied user correction for item {correction_learning['item_id']}")
    
    async def _store_processed_feedback(self, feedback: FeedbackEntry):
        """Store processed feedback for future reference."""
        # In a real implementation, this would store to the database
        # For now, we'll just mark it as processed
        pass
    
    async def _update_seasonal_patterns(self):
        """Update and evolve seasonal patterns based on accumulated feedback."""
        current_time = datetime.now(timezone.utc)
        
        for pattern_id, pattern in self.seasonal_patterns.items():
            # Check if pattern needs evolution
            time_since_update = current_time - pattern.last_updated
            
            if (time_since_update.total_seconds() > 86400 and  # 24 hours
                pattern.feedback_count >= self.min_feedback_for_learning):
                
                # Evolve the pattern
                await self._evolve_seasonal_pattern(pattern)
    
    async def _evolve_seasonal_pattern(self, pattern: SeasonalPattern):
        """Evolve a seasonal pattern based on accumulated feedback."""
        # Calculate evolution parameters
        evolution_strength = min(1.0, pattern.feedback_count / 100.0)
        
        # Update pattern parameters
        if "evolution_factor" not in pattern.parameters:
            pattern.parameters["evolution_factor"] = 1.0
        
        pattern.parameters["evolution_factor"] *= (1 + evolution_strength * 0.1)
        pattern.parameters["last_evolution"] = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Evolved seasonal pattern {pattern.pattern_id} with strength {evolution_strength:.3f}")
    
    async def _generate_improvement_recommendations(self):
        """Generate recommendations for model improvements based on feedback."""
        recommendations = []
        
        # Analyze feedback patterns
        for feedback_type, analytics in self.feedback_analytics.items():
            if analytics.get("total_count", 0) >= self.min_feedback_for_learning:
                recommendation = await self._analyze_feedback_pattern(feedback_type, analytics)
                if recommendation:
                    recommendations.append(recommendation)
        
        if recommendations:
            logger.info(f"Generated {len(recommendations)} improvement recommendations")
    
    async def _analyze_feedback_pattern(self, feedback_type: str, 
                                      analytics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze feedback patterns to generate improvement recommendations."""
        total_count = analytics["total_count"]
        
        # Analyze ratings if available
        ratings = analytics.get("ratings", [])
        if ratings:
            avg_rating = np.mean(ratings)
            rating_std = np.std(ratings)
            
            if avg_rating < 3.0:  # Poor ratings
                return {
                    "type": "low_satisfaction",
                    "feedback_type": feedback_type,
                    "average_rating": avg_rating,
                    "sample_size": len(ratings),
                    "recommendation": "Review and improve model accuracy for this feedback type"
                }
            elif rating_std > 1.5:  # High variance
                return {
                    "type": "inconsistent_performance",
                    "feedback_type": feedback_type,
                    "rating_variance": rating_std,
                    "recommendation": "Investigate causes of inconsistent performance"
                }
        
        return None
    
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of feedback collected in the specified period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        
        recent_feedback = [
            f for f in self.feedback_entries.values()
            if f.created_at >= cutoff_time
        ]
        
        summary = {
            "period_days": days,
            "total_feedback": len(recent_feedback),
            "feedback_by_type": {},
            "average_ratings": {},
            "sentiment_distribution": {},
            "top_retailers": {},
            "seasonal_insights": {}
        }
        
        # Analyze by feedback type
        for feedback in recent_feedback:
            feedback_type = feedback.feedback_type.value
            
            if feedback_type not in summary["feedback_by_type"]:
                summary["feedback_by_type"][feedback_type] = 0
            summary["feedback_by_type"][feedback_type] += 1
            
            # Collect ratings
            if feedback.rating is not None:
                if feedback_type not in summary["average_ratings"]:
                    summary["average_ratings"][feedback_type] = []
                summary["average_ratings"][feedback_type].append(feedback.rating)
            
            # Collect sentiment
            if feedback.sentiment:
                sentiment_key = f"{feedback_type}_{feedback.sentiment.value}"
                if sentiment_key not in summary["sentiment_distribution"]:
                    summary["sentiment_distribution"][sentiment_key] = 0
                summary["sentiment_distribution"][sentiment_key] += 1
        
        # Calculate average ratings
        for feedback_type, ratings in summary["average_ratings"].items():
            summary["average_ratings"][feedback_type] = {
                "average": np.mean(ratings),
                "count": len(ratings),
                "std_dev": np.std(ratings)
            }
        
        # Get seasonal insights
        summary["seasonal_insights"] = {
            pattern_id: {
                "season": pattern.season_name,
                "confidence": pattern.confidence_score,
                "feedback_count": pattern.feedback_count,
                "accuracy_improvement": pattern.accuracy_improvement
            }
            for pattern_id, pattern in self.seasonal_patterns.items()
        }
        
        return summary
    
    def get_retailer_feedback_history(self, retailer_id: str) -> Dict[str, Any]:
        """Get feedback history for a specific retailer."""
        retailer_feedback = [
            f for f in self.feedback_entries.values()
            if f.retailer_id == retailer_id
        ]
        
        if not retailer_feedback:
            return {"retailer_id": retailer_id, "feedback_count": 0}
        
        ratings = [f.rating for f in retailer_feedback if f.rating is not None]
        
        history = {
            "retailer_id": retailer_id,
            "feedback_count": len(retailer_feedback),
            "average_rating": np.mean(ratings) if ratings else None,
            "feedback_types": list(set(f.feedback_type.value for f in retailer_feedback)),
            "first_feedback": min(f.created_at for f in retailer_feedback).isoformat(),
            "last_feedback": max(f.created_at for f in retailer_feedback).isoformat(),
            "credibility_score": self._get_retailer_credibility(retailer_id)
        }
        
        return history
    
    def shutdown(self):
        """Shutdown the feedback learner."""
        logger.info("Shutting down feedback learner")
        self.is_running = False