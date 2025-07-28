"""
Configuration settings for Alpha Discovery API

This module contains configuration settings for the FastAPI server including:
- Environment variables
- Database settings
- Redis configuration
- Authentication settings
- Rate limiting configuration
- Monitoring settings

Author: Alpha Discovery Team
Date: 2025
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = Field(default="Alpha Discovery API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Database
    database_url: str = Field(default="sqlite:///alpha_discovery.db", env="DATABASE_URL")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Authentication
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate Limiting
    default_rate_limit: str = Field(default="100/hour", env="DEFAULT_RATE_LIMIT")
    burst_rate_limit: str = Field(default="10/minute", env="BURST_RATE_LIMIT")
    
    # CORS
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    allowed_methods: List[str] = Field(default=["*"], env="ALLOWED_METHODS")
    allowed_headers: List[str] = Field(default=["*"], env="ALLOWED_HEADERS")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Sentry
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_environment: str = Field(default="development", env="SENTRY_ENVIRONMENT")
    sentry_traces_sample_rate: float = Field(default=0.1, env="SENTRY_TRACES_SAMPLE_RATE")
    sentry_profiles_sample_rate: float = Field(default=0.1, env="SENTRY_PROFILES_SAMPLE_RATE")
    
    # Grafana
    grafana_host: str = Field(default="localhost", env="GRAFANA_HOST")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    grafana_api_key: Optional[str] = Field(default=None, env="GRAFANA_API_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Market Data
    market_data_provider: str = Field(default="mock", env="MARKET_DATA_PROVIDER")
    market_data_api_key: Optional[str] = Field(default=None, env="MARKET_DATA_API_KEY")
    
    # Performance
    max_request_size: int = Field(default=10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    log_level: str = "DEBUG"
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    log_level: str = "INFO"
    workers: int = 8
    allowed_origins: List[str] = ["https://yourdomain.com"]

class TestingSettings(Settings):
    """Testing environment settings"""
    database_url: str = "sqlite:///test_alpha_discovery.db"
    redis_db: int = 1
    debug: bool = True

def get_environment_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# API Key configuration
API_KEY_CONFIG = {
    "demo": {
        "key": "ak_demo_12345",
        "name": "Demo API Key",
        "permissions": ["read", "write", "stream"],
        "rate_limit": "1000/hour",
        "description": "Demo key for testing"
    },
    "production": {
        "key": "ak_prod_67890",
        "name": "Production API Key",
        "permissions": ["read", "write", "stream", "admin"],
        "rate_limit": "10000/hour",
        "description": "Production key for live trading"
    }
}

# Grafana dashboard configuration
GRAFANA_DASHBOARDS = {
    "alpha_discovery_overview": {
        "title": "Alpha Discovery Overview",
        "panels": [
            "API Request Rate",
            "Response Time",
            "Error Rate",
            "Active WebSocket Connections",
            "Alpha Discoveries",
            "Strategy Executions"
        ]
    },
    "performance_metrics": {
        "title": "Performance Metrics",
        "panels": [
            "Strategy Returns",
            "Risk Metrics",
            "Drawdown",
            "Sharpe Ratio",
            "Alpha Generation"
        ]
    },
    "system_health": {
        "title": "System Health",
        "panels": [
            "CPU Usage",
            "Memory Usage",
            "Database Connections",
            "Redis Connections",
            "Queue Length"
        ]
    }
}

# Prometheus metrics configuration
PROMETHEUS_METRICS = {
    "counters": [
        "api_requests_total",
        "alpha_discoveries_total",
        "strategy_executions_total",
        "websocket_connections_total",
        "errors_total"
    ],
    "histograms": [
        "api_request_duration_seconds",
        "alpha_discovery_duration_seconds",
        "strategy_execution_duration_seconds"
    ],
    "gauges": [
        "active_websocket_connections",
        "active_strategies",
        "system_memory_usage",
        "database_connections"
    ]
} 