"""
Load Management and Queuing System

Implements request queuing, load balancing, and graceful degradation
for the MarketPulse AI API under high load conditions.
"""

import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from collections import deque
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels for queue management."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RequestStatus(Enum):
    """Request processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class QueuedRequest:
    """Represents a queued request."""
    request_id: str
    priority: RequestPriority
    handler: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RequestStatus = RequestStatus.QUEUED
    result: Optional[Any] = None
    error: Optional[Exception] = None
    estimated_completion_time: Optional[datetime] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    active_requests: int
    queued_requests: int
    requests_per_minute: float
    average_response_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LoadManager:
    """
    Manages request queuing, load balancing, and graceful degradation.
    
    Features:
    - Priority-based request queuing
    - Estimated completion time calculation
    - System resource monitoring
    - Graceful degradation under high load
    - Request timeout handling
    """
    
    def __init__(
        self,
        max_concurrent_requests: int = 10,
        max_queue_size: int = 100,
        request_timeout: int = 300,  # 5 minutes
        high_load_threshold: float = 0.8,
        critical_load_threshold: float = 0.95
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        self.high_load_threshold = high_load_threshold
        self.critical_load_threshold = critical_load_threshold
        
        # Request queues by priority
        self.queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        
        # Active requests
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.completed_requests: Dict[str, QueuedRequest] = {}
        
        # Metrics tracking
        self.metrics_history: List[SystemMetrics] = []
        self.request_times: deque = deque(maxlen=100)  # Last 100 request times
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.processing_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring and processing tasks."""
        # Start metrics collection
        threading.Thread(target=self._collect_metrics_loop, daemon=True).start()
        
        # Start request processor
        threading.Thread(target=self._process_requests_loop, daemon=True).start()
    
    def _collect_metrics_loop(self):
        """Background task to collect system metrics."""
        while True:
            try:
                metrics = self._collect_system_metrics()
                with self.metrics_lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 100 metrics
                    if len(self.metrics_history) > 100:
                        self.metrics_history.pop(0)
                
                time.sleep(10)  # Collect metrics every 10 seconds
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        with self.processing_lock:
            active_requests = len(self.active_requests)
            queued_requests = sum(len(queue) for queue in self.queues.values())
        
        # Calculate requests per minute
        current_time = time.time()
        recent_requests = [t for t in self.request_times if current_time - t < 60]
        requests_per_minute = len(recent_requests)
        
        # Calculate average response time
        if self.request_times:
            avg_response_time = sum(self.request_times) / len(self.request_times)
        else:
            avg_response_time = 0.0
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_requests=active_requests,
            queued_requests=queued_requests,
            requests_per_minute=requests_per_minute,
            average_response_time=avg_response_time
        )
    
    def _process_requests_loop(self):
        """Background task to process queued requests."""
        while True:
            try:
                request = self._get_next_request()
                if request:
                    self._execute_request(request)
                else:
                    time.sleep(0.1)  # Short sleep when no requests
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                time.sleep(1)
    
    def _get_next_request(self) -> Optional[QueuedRequest]:
        """Get the next request to process based on priority."""
        with self.processing_lock:
            # Check if we can process more requests
            if len(self.active_requests) >= self.max_concurrent_requests:
                return None
            
            # Get highest priority request
            for priority in reversed(list(RequestPriority)):
                if self.queues[priority]:
                    return self.queues[priority].popleft()
            
            return None
    
    def _execute_request(self, request: QueuedRequest):
        """Execute a queued request."""
        request.started_at = datetime.now(timezone.utc)
        request.status = RequestStatus.PROCESSING
        
        with self.processing_lock:
            self.active_requests[request.request_id] = request
        
        def run_request():
            try:
                # Execute the request handler
                if asyncio.iscoroutinefunction(request.handler):
                    # Handle async functions
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        request.handler(*request.args, **request.kwargs)
                    )
                    loop.close()
                else:
                    # Handle sync functions
                    result = request.handler(*request.args, **request.kwargs)
                
                request.result = result
                request.status = RequestStatus.COMPLETED
                
            except Exception as e:
                logger.error(f"Request {request.request_id} failed: {e}")
                request.error = e
                request.status = RequestStatus.FAILED
            
            finally:
                request.completed_at = datetime.now(timezone.utc)
                
                # Record request time
                if request.started_at:
                    processing_time = (request.completed_at - request.started_at).total_seconds()
                    self.request_times.append(processing_time)
                
                # Move to completed requests
                with self.processing_lock:
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]
                    self.completed_requests[request.request_id] = request
                    
                    # Keep only last 1000 completed requests
                    if len(self.completed_requests) > 1000:
                        oldest_id = min(self.completed_requests.keys(), 
                                      key=lambda k: self.completed_requests[k].completed_at)
                        del self.completed_requests[oldest_id]
        
        # Submit to thread pool
        self.executor.submit(run_request)
    
    async def queue_request(
        self,
        handler: Callable[..., Awaitable[Any]],
        *args,
        priority: RequestPriority = RequestPriority.NORMAL,
        **kwargs
    ) -> str:
        """
        Queue a request for processing.
        
        Args:
            handler: The async function to execute
            *args: Arguments for the handler
            priority: Request priority level
            **kwargs: Keyword arguments for the handler
            
        Returns:
            Request ID for tracking
            
        Raises:
            RuntimeError: If queue is full or system is overloaded
        """
        # Check system load
        current_metrics = self._collect_system_metrics()
        
        # Reject requests if system is critically overloaded
        if (current_metrics.cpu_usage > self.critical_load_threshold or 
            current_metrics.memory_usage > self.critical_load_threshold):
            raise RuntimeError("System critically overloaded - request rejected")
        
        # Check queue capacity
        total_queued = sum(len(queue) for queue in self.queues.values())
        if total_queued >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
        
        # Create request
        request = QueuedRequest(
            request_id=str(uuid4()),
            priority=priority,
            handler=handler,
            args=args,
            kwargs=kwargs
        )
        
        # Calculate estimated completion time
        request.estimated_completion_time = self._calculate_estimated_completion_time(priority)
        
        # Add to appropriate queue
        with self.processing_lock:
            self.queues[priority].append(request)
        
        logger.info(f"Queued request {request.request_id} with priority {priority.name}")
        return request.request_id
    
    def _calculate_estimated_completion_time(self, priority: RequestPriority) -> datetime:
        """Calculate estimated completion time for a request."""
        with self.processing_lock:
            # Count requests ahead in queue
            requests_ahead = 0
            for p in RequestPriority:
                if p.value >= priority.value:
                    requests_ahead += len(self.queues[p])
                else:
                    break
            
            # Estimate based on average processing time and queue position
            avg_processing_time = (
                sum(self.request_times) / len(self.request_times) 
                if self.request_times else 30.0  # Default 30 seconds
            )
            
            # Account for concurrent processing
            concurrent_factor = min(self.max_concurrent_requests, requests_ahead)
            estimated_seconds = (requests_ahead / max(concurrent_factor, 1)) * avg_processing_time
            
            return datetime.now(timezone.utc) + timedelta(seconds=estimated_seconds)
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a request."""
        # Check active requests
        with self.processing_lock:
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                return {
                    "request_id": request_id,
                    "status": request.status.value,
                    "created_at": request.created_at.isoformat(),
                    "started_at": request.started_at.isoformat() if request.started_at else None,
                    "estimated_completion_time": request.estimated_completion_time.isoformat() if request.estimated_completion_time else None
                }
            
            # Check completed requests
            if request_id in self.completed_requests:
                request = self.completed_requests[request_id]
                return {
                    "request_id": request_id,
                    "status": request.status.value,
                    "created_at": request.created_at.isoformat(),
                    "started_at": request.started_at.isoformat() if request.started_at else None,
                    "completed_at": request.completed_at.isoformat() if request.completed_at else None,
                    "result": request.result,
                    "error": str(request.error) if request.error else None
                }
            
            # Check queued requests
            for priority_queue in self.queues.values():
                for request in priority_queue:
                    if request.request_id == request_id:
                        return {
                            "request_id": request_id,
                            "status": request.status.value,
                            "created_at": request.created_at.isoformat(),
                            "estimated_completion_time": request.estimated_completion_time.isoformat() if request.estimated_completion_time else None,
                            "queue_position": self._get_queue_position(request_id)
                        }
        
        return None
    
    def _get_queue_position(self, request_id: str) -> int:
        """Get the position of a request in the queue."""
        position = 0
        for priority in reversed(list(RequestPriority)):
            for i, request in enumerate(self.queues[priority]):
                if request.request_id == request_id:
                    return position + i + 1
                position += 1
        return -1
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        current_metrics = self._collect_system_metrics()
        
        with self.processing_lock:
            queue_status = {
                priority.name: len(queue) 
                for priority, queue in self.queues.items()
            }
        
        return {
            "system_metrics": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "active_requests": current_metrics.active_requests,
                "queued_requests": current_metrics.queued_requests,
                "requests_per_minute": current_metrics.requests_per_minute,
                "average_response_time": current_metrics.average_response_time
            },
            "queue_status": queue_status,
            "load_level": self._get_load_level(current_metrics),
            "accepting_requests": current_metrics.cpu_usage < self.critical_load_threshold,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_queue_size": self.max_queue_size
        }
    
    def _get_load_level(self, metrics: SystemMetrics) -> str:
        """Determine current load level."""
        max_usage = max(metrics.cpu_usage, metrics.memory_usage)
        
        if max_usage >= self.critical_load_threshold:
            return "critical"
        elif max_usage >= self.high_load_threshold:
            return "high"
        elif max_usage >= 0.5:
            return "moderate"
        else:
            return "low"
    
    def enable_graceful_degradation(self) -> Dict[str, Any]:
        """Enable graceful degradation mode."""
        logger.warning("Enabling graceful degradation mode")
        
        # Reduce concurrent requests
        self.max_concurrent_requests = max(1, self.max_concurrent_requests // 2)
        
        # Clear low priority requests
        with self.processing_lock:
            cleared_count = len(self.queues[RequestPriority.LOW])
            self.queues[RequestPriority.LOW].clear()
        
        return {
            "degradation_enabled": True,
            "max_concurrent_requests": self.max_concurrent_requests,
            "low_priority_requests_cleared": cleared_count,
            "message": "System operating in degraded mode - only high priority requests accepted"
        }
    
    def disable_graceful_degradation(self, original_max_concurrent: int):
        """Disable graceful degradation mode."""
        logger.info("Disabling graceful degradation mode")
        self.max_concurrent_requests = original_max_concurrent
    
    def shutdown(self):
        """Shutdown the load manager."""
        logger.info("Shutting down load manager")
        self.executor.shutdown(wait=True)


# Global load manager instance
load_manager = LoadManager()