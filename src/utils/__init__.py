"""
Utility modules for the Alpha Discovery Platform - 2025 Edition

This package contains enhanced utility modules for:
- Advanced model management and LLM routing with 2025 models
- Comprehensive data processing helpers with ML enhancements
- Configuration management with environment-specific settings
- Error handling and logging with structured logging
- Performance monitoring and observability
- Security utilities and authentication helpers
- Common utilities with type safety and async support
"""

import logging
import structlog
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import functools
import time
import traceback
from pathlib import Path
import os

# Configure structured logging for 2025
def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None
) -> None:
    """Configure structured logging for the platform"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if format_type == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

# Performance monitoring decorator
def monitor_performance(func_name: Optional[str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = structlog.get_logger(__name__)
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    "Function executed successfully",
                    function=func_name or func.__name__,
                    execution_time=execution_time,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func_name or func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc()
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = structlog.get_logger(__name__)
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    "Function executed successfully",
                    function=func_name or func.__name__,
                    execution_time=execution_time,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func_name or func.__name__,
                    execution_time=execution_time,
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc()
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Error handling utilities
class AlphaDiscoveryError(Exception):
    """Base exception for Alpha Discovery platform"""
    def __init__(self, message: str, error_code: str = "UNKNOWN", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()

class ModelManagerError(AlphaDiscoveryError):
    """Error related to model management"""
    pass

class DataProcessingError(AlphaDiscoveryError):
    """Error related to data processing"""
    pass

class APIError(AlphaDiscoveryError):
    """Error related to API calls"""
    pass

class ConfigurationError(AlphaDiscoveryError):
    """Error related to configuration"""
    pass

def handle_errors(
    default_return=None,
    log_errors: bool = True,
    raise_on_error: bool = True
):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = structlog.get_logger(__name__)
            
            try:
                return await func(*args, **kwargs)
            except AlphaDiscoveryError as e:
                if log_errors:
                    logger.error(
                        "Alpha Discovery error occurred",
                        function=func.__name__,
                        error_code=e.error_code,
                        message=e.message,
                        details=e.details,
                        timestamp=e.timestamp.isoformat()
                    )
                if raise_on_error:
                    raise
                return default_return
            except Exception as e:
                if log_errors:
                    logger.error(
                        "Unexpected error occurred",
                        function=func.__name__,
                        error=str(e),
                        error_type=type(e).__name__,
                        traceback=traceback.format_exc()
                    )
                if raise_on_error:
                    raise AlphaDiscoveryError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        details={"original_error": str(e), "error_type": type(e).__name__}
                    )
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = structlog.get_logger(__name__)
            
            try:
                return func(*args, **kwargs)
            except AlphaDiscoveryError as e:
                if log_errors:
                    logger.error(
                        "Alpha Discovery error occurred",
                        function=func.__name__,
                        error_code=e.error_code,
                        message=e.message,
                        details=e.details,
                        timestamp=e.timestamp.isoformat()
                    )
                if raise_on_error:
                    raise
                return default_return
            except Exception as e:
                if log_errors:
                    logger.error(
                        "Unexpected error occurred",
                        function=func.__name__,
                        error=str(e),
                        error_type=type(e).__name__,
                        traceback=traceback.format_exc()
                    )
                if raise_on_error:
                    raise AlphaDiscoveryError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        details={"original_error": str(e), "error_type": type(e).__name__}
                    )
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Configuration utilities
def get_env_var(
    var_name: str,
    default: Optional[str] = None,
    required: bool = False,
    var_type: type = str
) -> Any:
    """Get environment variable with type conversion and validation"""
    value = os.getenv(var_name, default)
    
    if required and value is None:
        raise ConfigurationError(
            f"Required environment variable {var_name} is not set",
            error_code="MISSING_ENV_VAR",
            details={"variable": var_name}
        )
    
    if value is None:
        return None
    
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif var_type == int:
            return int(value)
        elif var_type == float:
            return float(value)
        elif var_type == list:
            return value.split(',')
        else:
            return var_type(value)
    except (ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Invalid value for environment variable {var_name}: {value}",
            error_code="INVALID_ENV_VAR",
            details={"variable": var_name, "value": value, "expected_type": var_type.__name__}
        )

# Async utilities
async def run_with_timeout(
    coro,
    timeout: float,
    default_return=None,
    timeout_message: str = "Operation timed out"
):
    """Run coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger = structlog.get_logger(__name__)
        logger.warning(
            "Operation timed out",
            timeout=timeout,
            message=timeout_message
        )
        return default_return

async def gather_with_error_handling(
    *coros,
    return_exceptions: bool = True,
    log_errors: bool = True
):
    """Gather coroutines with enhanced error handling"""
    logger = structlog.get_logger(__name__)
    
    results = await asyncio.gather(*coros, return_exceptions=return_exceptions)
    
    if log_errors:
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Coroutine failed in gather",
                    coroutine_index=i,
                    error=str(result),
                    error_type=type(result).__name__
                )
    
    return results

# Performance metrics
class PerformanceMetrics:
    """Performance metrics collector"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return {
            "metrics": self.metrics,
            "uptime": time.time() - self.start_time,
            "total_metrics": sum(len(values) for values in self.metrics.values())
        }
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        if name not in self.metrics:
            return {}
        
        values = [m["value"] for m in self.metrics[name]]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }

# Global performance metrics instance
performance_metrics = PerformanceMetrics()

# Health check utilities
class HealthChecker:
    """System health checker"""
    
    def __init__(self):
        self.checks = {}
    
    def add_check(self, name: str, check_func, timeout: float = 5.0):
        """Add a health check"""
        self.checks[name] = {
            "func": check_func,
            "timeout": timeout
        }
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_config in self.checks.items():
            try:
                result = await run_with_timeout(
                    check_config["func"](),
                    check_config["timeout"],
                    default_return={"healthy": False, "error": "Timeout"}
                )
                
                if isinstance(result, dict) and result.get("healthy", True):
                    results[name] = {"status": "healthy", "details": result}
                else:
                    results[name] = {"status": "unhealthy", "details": result}
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }

# Global health checker instance
health_checker = HealthChecker()

# Initialize logging on import
configure_logging()

# Import main components
from .model_manager import ModelManager, model_manager, ModelType, TaskType, ModelResponse

__all__ = [
    "ModelManager",
    "model_manager", 
    "ModelType",
    "TaskType",
    "ModelResponse",
    "configure_logging",
    "monitor_performance",
    "handle_errors",
    "AlphaDiscoveryError",
    "ModelManagerError",
    "DataProcessingError",
    "APIError",
    "ConfigurationError",
    "get_env_var",
    "run_with_timeout",
    "gather_with_error_handling",
    "PerformanceMetrics",
    "performance_metrics",
    "HealthChecker",
    "health_checker"
] 