"""
Sentry configuration for Alpha Discovery project
"""
import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
import logging

def init_sentry():
    """Initialize Sentry SDK with appropriate integrations"""
    
    # Get DSN from environment or use default
    dsn = os.getenv(
        'SENTRY_DSN', 
        'https://d79a766e2a1675178c3d17513713c872@o4509702748438528.ingest.us.sentry.io/4509702829834240'
    )
    
    # Environment detection
    environment = os.getenv('ENVIRONMENT', 'development')
    
    # Initialize Sentry with integrations
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=os.getenv('VERSION', '1.0.0'),
        
        # Set traces sample rate for performance monitoring
        traces_sample_rate=0.1,
        
        # Set profiles sample rate for performance profiling
        profiles_sample_rate=0.1,
        
        # Enable debug mode in development
        debug=environment == 'development',
        
        # Integrations
        integrations=[
            FastApiIntegration(),
            RedisIntegration(),
            SqlalchemyIntegration(),
            HttpxIntegration(),
            LoggingIntegration(
                level=logging.INFO,        # Capture info and above as breadcrumbs
                event_level=logging.ERROR  # Send errors as events
            ),
        ],
        
        # Before send callback to filter sensitive data
        before_send=before_send_filter,
        
        # Before breadcrumb callback to filter sensitive breadcrumbs
        before_breadcrumb=before_breadcrumb_filter,
    )
    
    # Set user context if available
    user_id = os.getenv('USER_ID')
    if user_id:
        sentry_sdk.set_user({"id": user_id})

def before_send_filter(event, hint):
    """Filter sensitive data before sending to Sentry"""
    
    # Filter out certain error types if needed
    if 'exception' in hint:
        exc_type = type(hint['exception']).__name__
        if exc_type in ['KeyboardInterrupt', 'SystemExit']:
            return None
    
    # Remove sensitive data from extra context
    if 'extra' in event:
        sensitive_keys = ['api_key', 'password', 'token', 'secret']
        for key in sensitive_keys:
            if key in event['extra']:
                event['extra'][key] = '[REDACTED]'
    
    return event

def before_breadcrumb_filter(breadcrumb, hint):
    """Filter sensitive data from breadcrumbs"""
    
    # Remove sensitive data from breadcrumb data
    if 'data' in breadcrumb:
        sensitive_keys = ['api_key', 'password', 'token', 'secret']
        for key in sensitive_keys:
            if key in breadcrumb['data']:
                breadcrumb['data'][key] = '[REDACTED]'
    
    return breadcrumb

def set_user_context(user_id: str, email: str = None, username: str = None):
    """Set user context for Sentry events"""
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        "username": username
    })

def set_tag(key: str, value: str):
    """Set a tag for Sentry events"""
    sentry_sdk.set_tag(key, value)

def set_context(name: str, data: dict):
    """Set context data for Sentry events"""
    sentry_sdk.set_context(name, data)

def capture_exception(exception: Exception, **kwargs):
    """Capture an exception with additional context"""
    sentry_sdk.capture_exception(exception, **kwargs)

def capture_message(message: str, level: str = "info", **kwargs):
    """Capture a message with specified level"""
    sentry_sdk.capture_message(message, level=level, **kwargs) 