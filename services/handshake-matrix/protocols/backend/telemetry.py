"""
Telemetry module for the Affiliate Matrix backend.

This module provides telemetry and monitoring capabilities,
allowing the application to track metrics, events, and errors.
"""

import logging
import time
import functools
import json
from typing import Dict, Any, Optional, Callable, TypeVar, cast

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])


class TelemetryProvider:
    """
    Base telemetry provider interface.
    
    This class defines the interface for telemetry providers.
    Concrete implementations should override these methods.
    """
    
    def __init__(self):
        """Initialize the telemetry provider."""
        pass
    
    def track_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Track a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        pass
    
    def track_event(self, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Track an event.
        
        Args:
            name: Event name
            properties: Optional properties for the event
        """
        pass
    
    def track_exception(self, exception: Exception, properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Track an exception.
        
        Args:
            exception: Exception to track
            properties: Optional properties for the exception
        """
        pass
    
    def track_dependency(self, name: str, type: str, target: str, duration: float, success: bool) -> None:
        """
        Track a dependency call.
        
        Args:
            name: Dependency name
            type: Dependency type
            target: Dependency target
            duration: Call duration in milliseconds
            success: Whether the call was successful
        """
        pass


class ConsoleTelemetryProvider(TelemetryProvider):
    """
    Console telemetry provider.
    
    This provider logs telemetry data to the console.
    It's useful for development and debugging.
    """
    
    def track_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Log a metric to the console."""
        logger.info(f"METRIC: {name} = {value} {tags or ''}")
    
    def track_event(self, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Log an event to the console."""
        logger.info(f"EVENT: {name} {properties or ''}")
    
    def track_exception(self, exception: Exception, properties: Optional[Dict[str, Any]] = None) -> None:
        """Log an exception to the console."""
        logger.error(f"EXCEPTION: {type(exception).__name__}: {str(exception)} {properties or ''}")
    
    def track_dependency(self, name: str, type: str, target: str, duration: float, success: bool) -> None:
        """Log a dependency call to the console."""
        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"DEPENDENCY: {name} ({type}) -> {target} took {duration:.2f}ms - {status}")


class PrometheusProvider(TelemetryProvider):
    """
    Prometheus telemetry provider.
    
    This provider exposes metrics in Prometheus format.
    It requires the prometheus_client package.
    """
    
    def __init__(self):
        """Initialize the Prometheus provider."""
        try:
            import prometheus_client
            self.prometheus_client = prometheus_client
            
            # Create metrics
            self.metrics = {}
            self.counters = {}
            self.histograms = {}
            
            # Initialize default metrics
            self.counters['events'] = self.prometheus_client.Counter(
                'events_total', 'Total number of events', ['event_name']
            )
            self.counters['exceptions'] = self.prometheus_client.Counter(
                'exceptions_total', 'Total number of exceptions', ['exception_type']
            )
            self.histograms['dependency_duration'] = self.prometheus_client.Histogram(
                'dependency_duration_seconds', 'Dependency call duration in seconds',
                ['name', 'type', 'target', 'success']
            )
            
            logger.info("Prometheus telemetry provider initialized")
        except ImportError:
            logger.error("prometheus_client package not installed. Falling back to console provider.")
            self.prometheus_client = None
    
    def track_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Track a metric in Prometheus."""
        if not self.prometheus_client:
            return
        
        # Create gauge if it doesn't exist
        if name not in self.metrics:
            self.metrics[name] = self.prometheus_client.Gauge(
                name.replace('.', '_'), name, list(tags.keys()) if tags else []
            )
        
        # Set gauge value
        if tags:
            self.metrics[name].labels(**tags).set(value)
        else:
            self.metrics[name].set(value)
    
    def track_event(self, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Track an event in Prometheus."""
        if not self.prometheus_client:
            return
        
        # Increment event counter
        self.counters['events'].labels(event_name=name).inc()
    
    def track_exception(self, exception: Exception, properties: Optional[Dict[str, Any]] = None) -> None:
        """Track an exception in Prometheus."""
        if not self.prometheus_client:
            return
        
        # Increment exception counter
        self.counters['exceptions'].labels(exception_type=type(exception).__name__).inc()
    
    def track_dependency(self, name: str, type: str, target: str, duration: float, success: bool) -> None:
        """Track a dependency call in Prometheus."""
        if not self.prometheus_client:
            return
        
        # Record dependency duration
        self.histograms['dependency_duration'].labels(
            name=name, type=type, target=target, success=str(success)
        ).observe(duration / 1000.0)  # Convert ms to seconds


# Factory function to create telemetry provider
def create_telemetry_provider() -> TelemetryProvider:
    """
    Create a telemetry provider based on configuration.
    
    Returns:
        TelemetryProvider: Telemetry provider instance
    """
    if not settings.TELEMETRY_ENABLED:
        logger.info("Telemetry is disabled")
        return TelemetryProvider()  # No-op provider
    
    provider_type = settings.TELEMETRY_PROVIDER.lower()
    
    if provider_type == 'prometheus':
        return PrometheusProvider()
    elif provider_type == 'datadog':
        # Placeholder for DataDog provider
        logger.warning("DataDog provider not implemented, falling back to console")
        return ConsoleTelemetryProvider()
    else:
        # Default to console provider
        return ConsoleTelemetryProvider()


# Create global telemetry provider
telemetry = create_telemetry_provider()


# Middleware for request telemetry
class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking request telemetry.
    
    This middleware tracks request duration, status codes, and other metrics.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and track telemetry data."""
        start_time = time.time()
        
        # Track request start
        telemetry.track_event("http_request_start", {
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown"
        })
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate duration
            duration = (time.time() - start_time) * 1000  # ms
            
            # Track request completion
            telemetry.track_metric("http_request_duration", duration, {
                "method": request.method,
                "path": request.url.path,
                "status_code": str(response.status_code)
            })
            
            telemetry.track_event("http_request_complete", {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration
            })
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = (time.time() - start_time) * 1000  # ms
            
            # Track exception
            telemetry.track_exception(e, {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration
            })
            
            # Re-raise the exception
            raise


# Decorator for tracking function execution
def track_execution(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator for tracking function execution time and exceptions.
    
    Args:
        name: Optional name for the tracked function
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration = (time.time() - start_time) * 1000  # ms
                
                # Track successful execution
                telemetry.track_metric(f"function.{func_name}.duration", duration)
                
                return result
                
            except Exception as e:
                # Calculate duration
                duration = (time.time() - start_time) * 1000  # ms
                
                # Track exception
                telemetry.track_exception(e, {
                    "function": func_name,
                    "duration_ms": duration
                })
                
                # Re-raise the exception
                raise
        
        return cast(F, wrapper)
    
    return decorator


# Decorator for tracking async function execution
def track_async_execution(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator for tracking async function execution time and exceptions.
    
    Args:
        name: Optional name for the tracked function
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Calculate duration
                duration = (time.time() - start_time) * 1000  # ms
                
                # Track successful execution
                telemetry.track_metric(f"function.{func_name}.duration", duration)
                
                return result
                
            except Exception as e:
                # Calculate duration
                duration = (time.time() - start_time) * 1000  # ms
                
                # Track exception
                telemetry.track_exception(e, {
                    "function": func_name,
                    "duration_ms": duration
                })
                
                # Re-raise the exception
                raise
        
        return cast(F, wrapper)
    
    return decorator
