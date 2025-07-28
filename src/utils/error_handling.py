"""
Error Handling Utilities
Provides error handling and exception management
"""

import logging
import traceback
from typing import Any, Callable, Dict, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

class AlphaDiscoveryError(Exception):
    """Base exception for Alpha Discovery system"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()

class MarketDataError(AlphaDiscoveryError):
    """Market data related errors"""
    pass

class TradingError(AlphaDiscoveryError):
    """Trading related errors"""
    pass

class RiskError(AlphaDiscoveryError):
    """Risk management related errors"""
    pass

class AgentError(AlphaDiscoveryError):
    """Agent related errors"""
    pass

def handle_errors(error_type: type = AlphaDiscoveryError, 
                 reraise: bool = True,
                 log_level: int = logging.ERROR):
    """
    Decorator for handling errors in functions
    
    Args:
        error_type: Type of error to catch
        reraise: Whether to reraise the error
        log_level: Log level for error logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_info = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'traceback': traceback.format_exc()
                }
                
                logger.log(log_level, f"Error in {func.__name__}: {str(e)}", extra=error_info)
                
                if reraise:
                    if isinstance(e, AlphaDiscoveryError):
                        raise
                    else:
                        raise error_type(f"Error in {func.__name__}: {str(e)}", details=error_info)
                        
                return None
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'traceback': traceback.format_exc()
                }
                
                logger.log(log_level, f"Error in {func.__name__}: {str(e)}", extra=error_info)
                
                if reraise:
                    if isinstance(e, AlphaDiscoveryError):
                        raise
                    else:
                        raise error_type(f"Error in {func.__name__}: {str(e)}", details=error_info)
                        
                return None
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context"""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'context': context or {}
    }
    
    if isinstance(error, AlphaDiscoveryError):
        error_info.update({
            'error_code': error.error_code,
            'details': error.details,
            'timestamp': error.timestamp
        })
    
    logger.error("System error occurred", extra=error_info)

def create_error_response(error: Exception, request_id: str = None) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        'error': True,
        'error_type': type(error).__name__,
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }
    
    if request_id:
        response['request_id'] = request_id
        
    if isinstance(error, AlphaDiscoveryError):
        response.update({
            'error_code': error.error_code,
            'details': error.details
        })
    
    return response 