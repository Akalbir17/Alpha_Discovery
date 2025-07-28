"""
Monitoring Utilities
Provides monitoring and metrics tracking
"""

import logging
import time
from typing import Any, Callable, Dict, Optional
from functools import wraps
from datetime import datetime
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None

def set_metrics_collector(collector: MetricsCollector):
    """Set global metrics collector"""
    global _metrics_collector
    _metrics_collector = collector

def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get global metrics collector"""
    return _metrics_collector

def monitor_performance(operation_name: str = None,
                       track_latency: bool = True,
                       track_throughput: bool = True,
                       track_errors: bool = True):
    """
    Decorator for monitoring function performance
    
    Args:
        operation_name: Name of operation (defaults to function name)
        track_latency: Whether to track latency
        track_throughput: Whether to track throughput
        track_errors: Whether to track errors
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                if track_errors and _metrics_collector:
                    _metrics_collector.increment_counter(f"{op_name}_errors")
                raise
            finally:
                duration = time.time() - start_time
                
                if _metrics_collector:
                    if track_latency:
                        _metrics_collector.record_timing(op_name, duration)
                    
                    if track_throughput:
                        _metrics_collector.increment_counter(f"{op_name}_calls")
                        if success:
                            _metrics_collector.increment_counter(f"{op_name}_success")
                
                logger.debug(f"Operation {op_name} completed in {duration:.3f}s (success={success})")
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                if track_errors and _metrics_collector:
                    _metrics_collector.increment_counter(f"{op_name}_errors")
                raise
            finally:
                duration = time.time() - start_time
                
                if _metrics_collector:
                    if track_latency:
                        _metrics_collector.record_timing(op_name, duration)
                    
                    if track_throughput:
                        _metrics_collector.increment_counter(f"{op_name}_calls")
                        if success:
                            _metrics_collector.increment_counter(f"{op_name}_success")
                
                logger.debug(f"Operation {op_name} completed in {duration:.3f}s (success={success})")
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

def track_metrics(metric_name: str, value: Any, metric_type: str = "gauge"):
    """
    Track a metric value
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        metric_type: Type of metric (gauge, counter, histogram)
    """
    if not _metrics_collector:
        return
        
    if metric_type == "gauge":
        _metrics_collector.set_gauge(metric_name, float(value))
    elif metric_type == "counter":
        _metrics_collector.increment_counter(metric_name, int(value))
    elif metric_type == "histogram":
        _metrics_collector.record_histogram(metric_name, float(value))
    else:
        logger.warning(f"Unknown metric type: {metric_type}")

def track_business_metric(metric_name: str, value: float, tags: Dict[str, str] = None):
    """
    Track a business metric
    
    Args:
        metric_name: Name of the business metric
        value: Metric value
        tags: Optional tags for the metric
    """
    if _metrics_collector:
        full_name = f"business_{metric_name}"
        _metrics_collector.set_gauge(full_name, value)
        
        # Log business metric
        logger.info(f"Business metric: {metric_name} = {value}", extra={
            'metric_name': metric_name,
            'metric_value': value,
            'tags': tags or {},
            'timestamp': datetime.now()
        })

class MetricsContext:
    """Context manager for tracking metrics"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and _metrics_collector:
            duration = time.time() - self.start_time
            _metrics_collector.record_timing(self.operation_name, duration)
            
            if exc_type is not None:
                _metrics_collector.increment_counter(f"{self.operation_name}_errors")
            else:
                _metrics_collector.increment_counter(f"{self.operation_name}_success")

def create_metrics_context(operation_name: str) -> MetricsContext:
    """Create a metrics context manager"""
    return MetricsContext(operation_name) 