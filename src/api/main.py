"""
FastAPI Server for Alpha Discovery Platform

This module provides a comprehensive REST API and WebSocket server for the Alpha Discovery platform:
- REST API endpoints for strategy execution and management
- WebSocket streaming for real-time signal updates
- API key authentication and authorization
- Grafana integration for monitoring
- Prometheus metrics collection
- Rate limiting and request throttling
- QuantLib-compatible response formats

Author: Alpha Discovery Team
Date: 2025
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Database and caching
import redis
import sqlite3
from sqlalchemy import create_engine, text

# Security
import hashlib
import secrets
from passlib.context import CryptContext

# Data handling
import pandas as pd
import numpy as np

# Alpha Discovery imports
import sys
sys.path.append('..')
from agents.microstructure_agent import MicrostructureAgent
from agents.altdata_agent import AlternativeDataAgent
from agents.regime_agent import MarketRegimeAgent
from agents.strategy_agent import StrategyAgent
from strategies.risk_manager import RiskManager
from analytics.performance import PerformanceAnalytics
from data.market_feeds import MarketDataPipeline

# Configure structured logging
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
ACTIVE_WEBSOCKETS = Gauge('websocket_connections_active', 'Active WebSocket connections')
ALPHA_DISCOVERIES = Counter('alpha_discoveries_total', 'Total alpha discoveries', ['strategy_type'])
STRATEGY_EXECUTIONS = Counter('strategy_executions_total', 'Total strategy executions', ['strategy_id'])

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API Models
class AlphaDiscoveryRequest(BaseModel):
    """Request model for alpha discovery"""
    market_data: Dict[str, Any] = Field(..., description="Market data for analysis")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    risk_limits: Dict[str, float] = Field(default_factory=dict, description="Risk limits")
    lookback_days: int = Field(default=252, ge=1, le=1000, description="Lookback period in days")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.999, description="Confidence level")
    
    @validator('confidence_level')
    def validate_confidence(cls, v):
        if not 0.5 <= v <= 0.999:
            raise ValueError('Confidence level must be between 0.5 and 0.999')
        return v

class StrategyRequest(BaseModel):
    """Request model for strategy execution"""
    strategy_id: str = Field(..., description="Strategy identifier")
    positions: Dict[str, float] = Field(..., description="Position allocations")
    benchmark: str = Field(default="SPY", description="Benchmark symbol")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")

class QuantLibResponse(BaseModel):
    """QuantLib-compatible response format"""
    success: bool = Field(..., description="Operation success status")
    data: Dict[str, Any] = Field(..., description="Response data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = Field(default="1.0", description="API version")

class AlphaSignal(BaseModel):
    """Alpha signal model"""
    signal_id: str = Field(..., description="Signal identifier")
    strategy_id: str = Field(..., description="Strategy identifier")
    asset: str = Field(..., description="Asset symbol")
    signal_strength: float = Field(..., ge=-1, le=1, description="Signal strength (-1 to 1)")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StrategyStatus(BaseModel):
    """Strategy status model"""
    strategy_id: str
    status: str
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    last_updated: str
    positions: Dict[str, float]

# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections for real-time streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # client_id -> [strategy_ids]
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = []
        ACTIVE_WEBSOCKETS.inc()
        logger.info("WebSocket connected", client_id=client_id)
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        ACTIVE_WEBSOCKETS.dec()
        logger.info("WebSocket disconnected", client_id=client_id)
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error("Error sending message", client_id=client_id, error=str(e))
                self.disconnect(client_id)
    
    async def broadcast_signal(self, signal: AlphaSignal):
        """Broadcast signal to subscribed clients"""
        message = signal.json()
        
        for client_id, strategy_ids in self.subscriptions.items():
            if signal.strategy_id in strategy_ids or not strategy_ids:  # Empty list means all strategies
                await self.send_personal_message(message, client_id)
    
    def subscribe(self, client_id: str, strategy_ids: List[str]):
        """Subscribe client to strategy signals"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] = strategy_ids
            logger.info("Client subscribed", client_id=client_id, strategy_ids=strategy_ids)

# Authentication
class AuthManager:
    """Handles API key authentication"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from database/config"""
        # In production, this would come from a secure database
        return {
            "ak_demo_12345": {
                "name": "Demo API Key",
                "permissions": ["read", "write", "stream"],
                "rate_limit": 1000,  # requests per hour
                "created_at": datetime.now().isoformat()
            },
            "ak_prod_67890": {
                "name": "Production API Key",
                "permissions": ["read", "write", "stream", "admin"],
                "rate_limit": 10000,
                "created_at": datetime.now().isoformat()
            }
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key info"""
        return self.api_keys.get(api_key)
    
    def check_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission"""
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return False
        return permission in key_info.get("permissions", [])
    
    def get_rate_limit(self, api_key: str) -> int:
        """Get rate limit for API key"""
        key_info = self.validate_api_key(api_key)
        return key_info.get("rate_limit", 100) if key_info else 100

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Alpha Discovery API server")
    
    # Initialize services
    await initialize_services()
    
    # Start background tasks
    asyncio.create_task(signal_generator())
    
    yield
    
    # Cleanup
    logger.info("Shutting down Alpha Discovery API server")
    await cleanup_services()

# Initialize FastAPI app
app = FastAPI(
    title="Alpha Discovery API",
    description="Comprehensive API for quantitative alpha discovery and strategy execution",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Global services
redis_client = None
auth_manager = None
connection_manager = ConnectionManager()
market_data_pipeline = None
microstructure_agent = None
altdata_agent = None
regime_agent = None
strategy_agent = None
risk_manager = None
performance_analytics = None

# Service initialization
async def initialize_services():
    """Initialize all services with parallel loading and background tasks"""
    global redis_client, auth_manager, market_data_pipeline
    global microstructure_agent, altdata_agent, regime_agent, strategy_agent
    global risk_manager, performance_analytics
    
    try:
        # Initialize Redis (essential service) - Fast
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test Redis connection
        redis_client.ping()
        logger.info("Redis connection established")
        
        # Initialize authentication (essential service) - Fast
        auth_manager = AuthManager(redis_client)
        logger.info("Authentication service initialized")
        
        # Start background model loading
        asyncio.create_task(initialize_ml_services_background())
        
        logger.info("Basic services initialized successfully - ML models loading in background")
        
    except Exception as e:
        logger.error("Failed to initialize basic services", error=str(e))
        raise

async def initialize_ml_services_background():
    """Initialize ML services in background with parallel loading"""
    global market_data_pipeline, microstructure_agent, altdata_agent, regime_agent, strategy_agent
    global risk_manager, performance_analytics
    
    try:
        logger.info("Starting background ML services initialization...")
        
        # Get configuration
        config = get_config('api')
        
        # Create tasks for parallel loading
        tasks = []
        
        # Task 1: Market Data Pipeline
        tasks.append(asyncio.create_task(
            initialize_component("Market Data Pipeline", lambda: MarketDataPipeline())
        ))
        
        # Task 2: Microstructure Agent
        tasks.append(asyncio.create_task(
            initialize_component("Microstructure Agent", lambda: MicrostructureAgent())
        ))
        
        # Task 3: Alternative Data Agent
        tasks.append(asyncio.create_task(
            initialize_component("Alternative Data Agent", lambda: AlternativeDataAgent())
        ))
        
        # Task 4: Regime Detection Agent
        tasks.append(asyncio.create_task(
            initialize_component("Regime Detection Agent", lambda: MarketRegimeAgent())
        ))
        
        # Task 5: Strategy Agent
        tasks.append(asyncio.create_task(
            initialize_component("Strategy Agent", lambda: StrategyAgent())
        ))
        
        # Task 6: Risk Manager
        tasks.append(asyncio.create_task(
            initialize_component("Risk Manager", lambda: RiskManager(config))
        ))
        
        # Task 7: Performance Analytics
        tasks.append(asyncio.create_task(
            initialize_component("Performance Analytics", lambda: PerformanceAnalytics())
        ))
        
        # Wait for all components to initialize
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assign results to global variables
        market_data_pipeline = results[0] if not isinstance(results[0], Exception) else None
        microstructure_agent = results[1] if not isinstance(results[1], Exception) else None
        altdata_agent = results[2] if not isinstance(results[2], Exception) else None
        regime_agent = results[3] if not isinstance(results[3], Exception) else None
        strategy_agent = results[4] if not isinstance(results[4], Exception) else None
        risk_manager = results[5] if not isinstance(results[5], Exception) else None
        performance_analytics = results[6] if not isinstance(results[6], Exception) else None
        
        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(f"ML services initialization completed: {successful} successful, {failed} failed")
        
        if failed > 0:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Component {i} failed to initialize: {result}")
        
    except Exception as e:
        logger.error("Failed to initialize ML services", error=str(e))
        raise

async def initialize_component(name: str, init_func):
    """Initialize a component with timeout and error handling"""
    try:
        logger.info(f"Initializing {name}...")
        start_time = time.time()
        
        # Run initialization with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(init_func), 
            timeout=60.0  # 60 second timeout per component
        )
        
        elapsed = time.time() - start_time
        logger.info(f"{name} initialized successfully in {elapsed:.2f}s")
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"{name} initialization timed out after 60s")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize {name}: {e}")
        raise

async def cleanup_services():
    """Cleanup services on shutdown"""
    try:
        if redis_client:
            redis_client.close()
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key and return user info"""
    api_key = credentials.credentials
    
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    
    key_info = auth_manager.validate_api_key(api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {"api_key": api_key, "key_info": key_info}

# Permission checking
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(user: dict = Depends(get_current_user)):
        if not auth_manager.check_permission(user["api_key"], permission):
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
        return user
    return permission_checker

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return QuantLibResponse(
        success=True,
        data={
            "service": "Alpha Discovery API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "discover_alpha": "POST /discover_alpha",
                "strategies": "GET /strategies",
                "stream_signals": "WS /stream/signals",
                "metrics": "GET /metrics",
                "health": "GET /health"
            }
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check basic service health
        services_status = {
            "redis": redis_client.ping() if redis_client else False,
            "authentication": auth_manager is not None,
            "market_data_pipeline": market_data_pipeline is not None,
            "microstructure_agent": microstructure_agent is not None,
            "altdata_agent": altdata_agent is not None,
            "regime_agent": regime_agent is not None,
            "strategy_agent": strategy_agent is not None,
            "risk_manager": risk_manager is not None,
            "performance_analytics": performance_analytics is not None
        }
        
        # Count loaded services
        loaded_services = sum(services_status.values())
        total_services = len(services_status)
        ml_services_loaded = sum([
            services_status["market_data_pipeline"],
            services_status["microstructure_agent"],
            services_status["altdata_agent"],
            services_status["regime_agent"],
            services_status["strategy_agent"],
            services_status["risk_manager"],
            services_status["performance_analytics"]
        ])
        
        # API is healthy if basic services are working
        basic_healthy = services_status["redis"] and services_status["authentication"]
        all_services_loaded = all(services_status.values())
        
        # Determine status
        if all_services_loaded:
            status = "fully_operational"
        elif basic_healthy:
            status = "partially_loaded"
        else:
            status = "degraded"
        
        return QuantLibResponse(
            success=basic_healthy,
            data={
                "status": status,
                "services": services_status,
                "loading_progress": {
                    "loaded": loaded_services,
                    "total": total_services,
                    "percentage": round((loaded_services / total_services) * 100, 1),
                    "ml_services_loaded": ml_services_loaded,
                    "ml_services_total": 7
                },
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return QuantLibResponse(
            success=False,
            data={"status": "unhealthy", "error": str(e)}
        )

@app.get("/startup-status")
async def startup_status():
    """Get detailed startup status and loading progress"""
    try:
        services_status = {
            "redis": {
                "loaded": redis_client.ping() if redis_client else False,
                "name": "Redis Connection",
                "priority": "critical"
            },
            "authentication": {
                "loaded": auth_manager is not None,
                "name": "Authentication Service",
                "priority": "critical"
            },
            "market_data_pipeline": {
                "loaded": market_data_pipeline is not None,
                "name": "Market Data Pipeline",
                "priority": "high"
            },
            "microstructure_agent": {
                "loaded": microstructure_agent is not None,
                "name": "Microstructure Agent",
                "priority": "high"
            },
            "altdata_agent": {
                "loaded": altdata_agent is not None,
                "name": "Alternative Data Agent",
                "priority": "medium"
            },
            "regime_agent": {
                "loaded": regime_agent is not None,
                "name": "Regime Detection Agent",
                "priority": "medium"
            },
            "strategy_agent": {
                "loaded": strategy_agent is not None,
                "name": "Strategy Agent",
                "priority": "high"
            },
            "risk_manager": {
                "loaded": risk_manager is not None,
                "name": "Risk Manager",
                "priority": "high"
            },
            "performance_analytics": {
                "loaded": performance_analytics is not None,
                "name": "Performance Analytics",
                "priority": "medium"
            }
        }
        
        # Calculate progress
        loaded_services = sum(1 for s in services_status.values() if s["loaded"])
        total_services = len(services_status)
        
        # Group by priority
        critical_loaded = sum(1 for s in services_status.values() 
                            if s["priority"] == "critical" and s["loaded"])
        critical_total = sum(1 for s in services_status.values() 
                           if s["priority"] == "critical")
        
        high_loaded = sum(1 for s in services_status.values() 
                         if s["priority"] == "high" and s["loaded"])
        high_total = sum(1 for s in services_status.values() 
                        if s["priority"] == "high")
        
        medium_loaded = sum(1 for s in services_status.values() 
                           if s["priority"] == "medium" and s["loaded"])
        medium_total = sum(1 for s in services_status.values() 
                          if s["priority"] == "medium")
        
        return QuantLibResponse(
            success=True,
            data={
                "overall_progress": {
                    "loaded": loaded_services,
                    "total": total_services,
                    "percentage": round((loaded_services / total_services) * 100, 1)
                },
                "priority_progress": {
                    "critical": {"loaded": critical_loaded, "total": critical_total},
                    "high": {"loaded": high_loaded, "total": high_total},
                    "medium": {"loaded": medium_loaded, "total": medium_total}
                },
                "services": services_status,
                "ready_for_requests": critical_loaded == critical_total,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error("Startup status check failed", error=str(e))
        return QuantLibResponse(
            success=False,
            data={"error": str(e)}
        )

@app.post("/discover_alpha")
@limiter.limit("10/minute")
async def discover_alpha(
    request: Request,
    alpha_request: AlphaDiscoveryRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_permission("write"))
):
    """Discover alpha signals using AI agents"""
    REQUEST_COUNT.labels(method="POST", endpoint="/discover_alpha", status="processing").inc()
    
    start_time = time.time()
    
    try:
        logger.info("Alpha discovery request received", 
                   user=user["key_info"]["name"],
                   lookback_days=alpha_request.lookback_days)
        
        # Extract market data
        market_data = pd.DataFrame(alpha_request.market_data)
        
        # Run alpha discovery pipeline
        discovery_results = await run_alpha_discovery_pipeline(
            market_data=market_data,
            strategy_params=alpha_request.strategy_params,
            risk_limits=alpha_request.risk_limits,
            lookback_days=alpha_request.lookback_days,
            confidence_level=alpha_request.confidence_level
        )
        
        # Track metrics
        ALPHA_DISCOVERIES.labels(strategy_type=discovery_results.get("strategy_type", "unknown")).inc()
        
        # Schedule background tasks
        background_tasks.add_task(
            log_alpha_discovery,
            user["api_key"],
            discovery_results
        )
        
        REQUEST_COUNT.labels(method="POST", endpoint="/discover_alpha", status="success").inc()
        REQUEST_DURATION.labels(method="POST", endpoint="/discover_alpha").observe(time.time() - start_time)
        
        return QuantLibResponse(
            success=True,
            data=discovery_results,
            metadata={
                "execution_time": time.time() - start_time,
                "user": user["key_info"]["name"]
            }
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/discover_alpha", status="error").inc()
        logger.error("Alpha discovery failed", error=str(e), user=user["key_info"]["name"])
        raise HTTPException(status_code=500, detail=f"Alpha discovery failed: {str(e)}")

@app.get("/strategies")
@limiter.limit("30/minute")
async def get_strategies(
    request: Request,
    strategy_id: Optional[str] = None,
    status: Optional[str] = None,
    user: dict = Depends(require_permission("read"))
):
    """Get strategy information"""
    REQUEST_COUNT.labels(method="GET", endpoint="/strategies", status="processing").inc()
    
    try:
        # Get strategy data
        strategies = await get_strategy_data(strategy_id, status)
        
        REQUEST_COUNT.labels(method="GET", endpoint="/strategies", status="success").inc()
        
        return QuantLibResponse(
            success=True,
            data={"strategies": strategies},
            metadata={"count": len(strategies)}
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="GET", endpoint="/strategies", status="error").inc()
        logger.error("Failed to get strategies", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get strategies: {str(e)}")

@app.post("/strategies/{strategy_id}/execute")
@limiter.limit("5/minute")
async def execute_strategy(
    request: Request,
    strategy_id: str,
    strategy_request: StrategyRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_permission("write"))
):
    """Execute a specific strategy"""
    REQUEST_COUNT.labels(method="POST", endpoint="/strategies/execute", status="processing").inc()
    
    start_time = time.time()
    
    try:
        logger.info("Strategy execution request", 
                   strategy_id=strategy_id,
                   user=user["key_info"]["name"])
        
        # Execute strategy
        execution_results = await execute_strategy_pipeline(
            strategy_id=strategy_id,
            positions=strategy_request.positions,
            benchmark=strategy_request.benchmark,
            start_date=strategy_request.start_date,
            end_date=strategy_request.end_date
        )
        
        # Track metrics
        STRATEGY_EXECUTIONS.labels(strategy_id=strategy_id).inc()
        
        # Schedule background tasks
        background_tasks.add_task(
            log_strategy_execution,
            strategy_id,
            user["api_key"],
            execution_results
        )
        
        REQUEST_COUNT.labels(method="POST", endpoint="/strategies/execute", status="success").inc()
        REQUEST_DURATION.labels(method="POST", endpoint="/strategies/execute").observe(time.time() - start_time)
        
        return QuantLibResponse(
            success=True,
            data=execution_results,
            metadata={
                "strategy_id": strategy_id,
                "execution_time": time.time() - start_time
            }
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/strategies/execute", status="error").inc()
        logger.error("Strategy execution failed", 
                    strategy_id=strategy_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Strategy execution failed: {str(e)}")

@app.get("/strategies/{strategy_id}/performance")
@limiter.limit("20/minute")
async def get_strategy_performance(
    request: Request,
    strategy_id: str,
    days: int = 30,
    user: dict = Depends(require_permission("read"))
):
    """Get strategy performance metrics"""
    try:
        # Get performance data
        performance_data = await get_performance_data(strategy_id, days)
        
        return QuantLibResponse(
            success=True,
            data=performance_data,
            metadata={"strategy_id": strategy_id, "period_days": days}
        )
        
    except Exception as e:
        logger.error("Failed to get performance data", 
                    strategy_id=strategy_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get performance data: {str(e)}")

@app.websocket("/stream/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signal streaming"""
    client_id = f"client_{int(time.time())}_{secrets.token_hex(4)}"
    
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Wait for client messages (subscription requests)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                strategy_ids = message.get("strategy_ids", [])
                connection_manager.subscribe(client_id, strategy_ids)
                
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "strategy_ids": strategy_ids,
                    "timestamp": datetime.now().isoformat()
                }))
                
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error("WebSocket error", client_id=client_id, error=str(e))
        connection_manager.disconnect(client_id)

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return JSONResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Background tasks and signal generation

async def signal_generator():
    """Background task to generate and broadcast signals"""
    while True:
        try:
            # Generate mock signals (in production, this would come from actual strategies)
            signal = AlphaSignal(
                signal_id=f"signal_{int(time.time())}_{secrets.token_hex(4)}",
                strategy_id="momentum_strategy",
                asset="AAPL",
                signal_strength=np.random.uniform(-1, 1),
                confidence=np.random.uniform(0.5, 1.0),
                metadata={
                    "regime": "normal",
                    "volatility": np.random.uniform(0.1, 0.3),
                    "market_cap": "large"
                }
            )
            
            await connection_manager.broadcast_signal(signal)
            await asyncio.sleep(5)  # Generate signal every 5 seconds
            
        except Exception as e:
            logger.error("Signal generation error", error=str(e))
            await asyncio.sleep(10)

async def log_alpha_discovery(api_key: str, results: Dict[str, Any]):
    """Log alpha discovery for audit trail"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "api_key": api_key,
            "action": "alpha_discovery",
            "results": results
        }
        
        # Store in Redis for audit
        if redis_client:
            redis_client.lpush("alpha_discovery_log", json.dumps(log_entry))
            redis_client.ltrim("alpha_discovery_log", 0, 1000)  # Keep last 1000 entries
            
    except Exception as e:
        logger.error("Failed to log alpha discovery", error=str(e))

async def log_strategy_execution(strategy_id: str, api_key: str, results: Dict[str, Any]):
    """Log strategy execution for audit trail"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy_id": strategy_id,
            "api_key": api_key,
            "action": "strategy_execution",
            "results": results
        }
        
        if redis_client:
            redis_client.lpush("strategy_execution_log", json.dumps(log_entry))
            redis_client.ltrim("strategy_execution_log", 0, 1000)
            
    except Exception as e:
        logger.error("Failed to log strategy execution", error=str(e))

# Core business logic

async def run_alpha_discovery_pipeline(
    market_data: pd.DataFrame,
    strategy_params: Dict[str, Any],
    risk_limits: Dict[str, float],
    lookback_days: int,
    confidence_level: float
) -> Dict[str, Any]:
    """Run the alpha discovery pipeline"""
    try:
        # Step 1: Market regime detection
        regime_analysis = await regime_agent.analyze_market_regime(market_data)
        
        # Step 2: Microstructure analysis
        microstructure_signals = await microstructure_agent.analyze_microstructure(market_data)
        
        # Step 3: Alternative data analysis
        altdata_signals = await altdata_agent.find_alpha_signals(market_data)
        
        # Step 4: Strategy synthesis
        strategy_recommendations = await strategy_agent.synthesize_strategy(
            regime_analysis,
            microstructure_signals,
            altdata_signals,
            strategy_params
        )
        
        # Step 5: Risk assessment
        risk_assessment = risk_manager.calculate_var(
            market_data,
            strategy_recommendations.get("positions", {}),
            confidence_level=confidence_level
        )
        
        # Step 6: Performance projection
        performance_projection = performance_analytics.calculate_alpha(
            pd.Series(strategy_recommendations.get("expected_returns", [])),
            pd.Series(market_data.get("benchmark_returns", []))
        )
        
        return {
            "strategy_type": "multi_factor_alpha",
            "regime_analysis": regime_analysis,
            "microstructure_signals": microstructure_signals,
            "altdata_signals": altdata_signals,
            "strategy_recommendations": strategy_recommendations,
            "risk_assessment": risk_assessment,
            "performance_projection": performance_projection,
            "confidence_level": confidence_level,
            "lookback_days": lookback_days
        }
        
    except Exception as e:
        logger.error("Alpha discovery pipeline failed", error=str(e))
        raise

async def execute_strategy_pipeline(
    strategy_id: str,
    positions: Dict[str, float],
    benchmark: str,
    start_date: Optional[str],
    end_date: Optional[str]
) -> Dict[str, Any]:
    """Execute strategy pipeline"""
    try:
        # Get historical data for backtesting
        # In production, this would fetch real market data
        mock_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 252))
        
        # Track performance
        performance_record = performance_analytics.track_performance(
            strategy_id=strategy_id,
            returns=mock_returns,
            positions=positions,
            benchmark_returns=benchmark_returns
        )
        
        # Calculate alpha
        alpha_analysis = performance_analytics.calculate_alpha(
            mock_returns,
            benchmark_returns,
            method='fama_french'
        )
        
        # Risk analysis
        risk_analysis = risk_manager.calculate_var(
            pd.DataFrame({"strategy": mock_returns, "benchmark": benchmark_returns}),
            positions
        )
        
        # Generate P&L
        pnl_analysis = performance_analytics.generate_pnl(strategy_id)
        
        return {
            "strategy_id": strategy_id,
            "execution_timestamp": datetime.now().isoformat(),
            "performance_record": {
                "pnl": performance_record.pnl,
                "returns": performance_record.returns,
                "alpha": performance_record.alpha,
                "sharpe_ratio": performance_record.sharpe_ratio,
                "drawdown": performance_record.drawdown
            },
            "alpha_analysis": alpha_analysis,
            "risk_analysis": risk_analysis,
            "pnl_analysis": pnl_analysis,
            "positions": positions,
            "benchmark": benchmark
        }
        
    except Exception as e:
        logger.error("Strategy execution pipeline failed", error=str(e))
        raise

async def get_strategy_data(strategy_id: Optional[str], status: Optional[str]) -> List[StrategyStatus]:
    """Get strategy data"""
    try:
        # Mock strategy data - in production, this would come from database
        strategies = [
            StrategyStatus(
                strategy_id="momentum_strategy",
                status="active",
                performance_metrics={
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "alpha": 0.08,
                    "max_drawdown": -0.05
                },
                risk_metrics={
                    "volatility": 0.12,
                    "var_95": -0.02,
                    "beta": 0.8
                },
                last_updated=datetime.now().isoformat(),
                positions={"AAPL": 0.3, "GOOGL": 0.2, "MSFT": 0.25, "CASH": 0.25}
            ),
            StrategyStatus(
                strategy_id="mean_reversion_strategy",
                status="active",
                performance_metrics={
                    "total_return": 0.12,
                    "sharpe_ratio": 0.9,
                    "alpha": 0.06,
                    "max_drawdown": -0.08
                },
                risk_metrics={
                    "volatility": 0.15,
                    "var_95": -0.025,
                    "beta": 1.1
                },
                last_updated=datetime.now().isoformat(),
                positions={"SPY": -0.2, "TLT": 0.4, "GLD": 0.3, "CASH": 0.5}
            )
        ]
        
        # Filter by strategy_id if provided
        if strategy_id:
            strategies = [s for s in strategies if s.strategy_id == strategy_id]
        
        # Filter by status if provided
        if status:
            strategies = [s for s in strategies if s.status == status]
        
        return strategies
        
    except Exception as e:
        logger.error("Failed to get strategy data", error=str(e))
        raise

async def get_performance_data(strategy_id: str, days: int) -> Dict[str, Any]:
    """Get performance data for strategy"""
    try:
        # Get performance summary
        performance_summary = performance_analytics.get_performance_summary(strategy_id)
        
        # Get recent alerts
        alerts = performance_analytics.get_alerts(strategy_id)
        
        return {
            "strategy_id": strategy_id,
            "period_days": days,
            "performance_summary": performance_summary,
            "alerts": alerts[-10:],  # Last 10 alerts
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get performance data", error=str(e))
        raise

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error("HTTP exception", 
                path=request.url.path, 
                status_code=exc.status_code, 
                detail=exc.detail)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=QuantLibResponse(
            success=False,
            data={"error": exc.detail},
            metadata={"status_code": exc.status_code}
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("Unhandled exception", 
                path=request.url.path, 
                error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content=QuantLibResponse(
            success=False,
            data={"error": "Internal server error"},
            metadata={"status_code": 500}
        ).dict()
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 