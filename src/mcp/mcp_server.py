"""
MCP (Model Context Protocol) Server

State-of-the-art MCP server implementation with:
- Latest MCP protocol features (2025)
- Async/await for high performance
- Integration with free AI models
- Real-time data streaming
- Tool discovery and execution
- Authentication management
- ML model services for heavy transformer models
- HTTP API endpoints for ML inference
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
import structlog

# HTTP server imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from ..utils.model_manager import model_manager, TaskType
from ..tools.market_data import MarketDataTool
from ..scrapers.reddit_scraper import RedditScraper
from ..tools.order_flow import OrderFlowTool
from .tool_definitions import ToolDefinitions
from .ml_services import ml_service

logger = structlog.get_logger(__name__)

# HTTP API Models
class SentimentRequest(BaseModel):
    text: str

class EmotionRequest(BaseModel):
    text: str

class NERRequest(BaseModel):
    text: str

class SimilarityRequest(BaseModel):
    text1: str
    text2: str

# Microstructure ML request models
class RegimeDetectionRequest(BaseModel):
    features: List[float]

class ImbalancePredictionRequest(BaseModel):
    features: List[float]

class LambdaPredictionRequest(BaseModel):
    features: List[float]

class FlowClassificationRequest(BaseModel):
    features: List[float]

# GLiNER NER request model (2025 state-of-the-art)
class GLiNERRequest(BaseModel):
    text: str
    entity_types: Optional[List[str]] = None

# Alternative Data Agent request models (Phase 3)
class FinancialSentimentRequest(BaseModel):
    text: str

class VisionAnalysisRequest(BaseModel):
    image_path: str
    analysis_type: str = "economic_activity"

# Advanced ML request models (Phase 4)
class MLPPredictionRequest(BaseModel):
    features: List[float]
    target_name: str = "risk"

class GradientBoostingRequest(BaseModel):
    features: List[float]

class RiskPCARequest(BaseModel):
    features: List[List[float]]

class ClusteringRequest(BaseModel):
    features: List[List[float]]
    n_clusters: int = 3

class AnomalyDetectionRequest(BaseModel):
    features: List[List[float]]

# Analytics/Performance request models (Phase 5.1)
class LinearRegressionRequest(BaseModel):
    X: List[List[float]]
    y: List[float]

class RidgeRegressionRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    alpha: float = 1.0

# Reinforcement Learning request models (Phase 5.1)
class RLTrainingRequest(BaseModel):
    model_type: str  # PPO, A2C, DQN
    total_timesteps: int = 10000

class RLPredictionRequest(BaseModel):
    model_type: str  # PPO, A2C, DQN
    observation: List[float]


class MCPMessageType(Enum):
    """MCP message types"""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    TOOLS_GET = "tools/get"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"
    PING = "ping"
    PONG = "pong"


@dataclass
class MCPMessage:
    """MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Tool call request"""
    name: str
    arguments: Dict[str, Any]
    call_id: str


class MCPServer:
    """
    State-of-the-art MCP server with latest protocol features.
    
    Features:
    - Async WebSocket server with high performance
    - Tool discovery and execution
    - Real-time data streaming
    - Authentication management
    - Integration with free AI models
    - Error handling and logging
    - Rate limiting and caching
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001, http_port: int = 8002):
        self.host = host
        self.port = port
        self.http_port = http_port
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, List[str]] = {}
        self.tool_definitions = ToolDefinitions()
        self.market_data_tool = MarketDataTool()
        self.reddit_scraper = RedditScraper()
        self.order_flow_tool = OrderFlowTool()
        self.rate_limits = {}
        self.cache = {}
        
        # Initialize HTTP app for ML services
        self.app = FastAPI(title="Alpha Discovery MCP ML Services", version="1.0.0")
        self._setup_http_routes()
        
        # Initialize tools
        self.tools = {
            "get_orderbook": self._get_orderbook,
            "get_reddit_sentiment": self._get_reddit_sentiment,
            "calculate_microstructure_features": self._calculate_microstructure_features,
            "detect_regime_change": self._detect_regime_change,
            "get_market_data": self._get_market_data,
            "analyze_order_flow": self._analyze_order_flow,
            "get_technical_indicators": self._get_technical_indicators,
            "stream_market_data": self._stream_market_data,
            "get_news_sentiment": self._get_news_sentiment,
            "calculate_risk_metrics": self._calculate_risk_metrics
        }
        
        logger.info("MCP Server initialized", host=host, port=port)
    
    async def start(self):
        """Start the MCP server with both WebSocket and HTTP endpoints"""
        try:
            # Initialize ML services
            logger.info("Initializing ML services...")
            await ml_service.initialize_models()
            logger.info("ML services initialized successfully")
            
            # Start HTTP server for ML services
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.http_port,
                log_level="info"
            )
            http_server = uvicorn.Server(config)
            
            # Start both servers concurrently
            async with websockets.serve(self._handle_client, self.host, self.port):
                logger.info("MCP WebSocket Server started", host=self.host, port=self.port)
                logger.info("MCP HTTP Server started", host=self.host, port=self.http_port)
                
                # Run HTTP server in background
                await asyncio.gather(
                    http_server.serve(),
                    asyncio.Future()  # Keep WebSocket server running
                )
                
        except Exception as e:
            logger.error("Failed to start MCP server", error=str(e))
            raise
    
    async def stop(self):
        """Stop the MCP server"""
        try:
            # Close all client connections
            for client_id, websocket in list(self.clients.items()):
                await websocket.close()
                logger.info("Client connection closed", client_id=client_id)
            
            # Clear subscriptions
            self.subscriptions.clear()
            self.clients.clear()
            
            logger.info("MCP Server stopped successfully")
        except Exception as e:
            logger.error("Error stopping MCP server", error=str(e))
            raise
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        
        try:
            logger.info("Client connected", client_id=client_id)
            
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    mcp_message = MCPMessage(**data)
                    
                    # Handle message
                    response = await self._process_message(mcp_message, client_id)
                    
                    # Send response
                    if response:
                        await websocket.send(json.dumps(asdict(response)))
                        
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON message", error=str(e))
                    error_response = self._create_error_response(
                        None, -32700, "Parse error", str(e)
                    )
                    await websocket.send(json.dumps(asdict(error_response)))
                    
                except Exception as e:
                    logger.error("Error processing message", error=str(e))
                    error_response = self._create_error_response(
                        mcp_message.id if hasattr(mcp_message, 'id') else None,
                        -32603, "Internal error", str(e)
                    )
                    await websocket.send(json.dumps(asdict(error_response)))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected", client_id=client_id)
        except Exception as e:
            logger.error("Client error", client_id=client_id, error=str(e))
        finally:
            # Cleanup
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
    
    async def _process_message(self, message: MCPMessage, client_id: str) -> Optional[MCPMessage]:
        """Process MCP message and return response"""
        try:
            if message.method == MCPMessageType.INITIALIZE.value:
                return await self._handle_initialize(message)
            elif message.method == MCPMessageType.TOOLS_LIST.value:
                return await self._handle_tools_list(message)
            elif message.method == MCPMessageType.TOOLS_CALL.value:
                return await self._handle_tools_call(message, client_id)
            elif message.method == MCPMessageType.TOOLS_GET.value:
                return await self._handle_tools_get(message)
            elif message.method == MCPMessageType.RESOURCES_LIST.value:
                return await self._handle_resources_list(message)
            elif message.method == MCPMessageType.RESOURCES_READ.value:
                return await self._handle_resources_read(message)
            elif message.method == MCPMessageType.RESOURCES_SUBSCRIBE.value:
                return await self._handle_resources_subscribe(message, client_id)
            elif message.method == MCPMessageType.RESOURCES_UNSUBSCRIBE.value:
                return await self._handle_resources_unsubscribe(message, client_id)
            elif message.method == MCPMessageType.PING.value:
                return self._create_pong_response(message.id)
            else:
                return self._create_error_response(
                    message.id, -32601, "Method not found", f"Unknown method: {message.method}"
                )
                
        except Exception as e:
            logger.error("Error processing message", error=str(e))
            return self._create_error_response(
                message.id, -32603, "Internal error", str(e)
            )
    
    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handle initialize request"""
        try:
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {
                        "subscribe": True,
                        "read": True
                    },
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "Alpha Discovery MCP Server",
                    "version": "1.0.0"
                }
            }
            
            return MCPMessage(
                id=message.id,
                result=result
            )
            
        except Exception as e:
            logger.error("Error in initialize", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    async def _handle_tools_list(self, message: MCPMessage) -> MCPMessage:
        """Handle tools list request"""
        try:
            tools = []
            for tool_name, tool_func in self.tools.items():
                tool_def = self.tool_definitions.get_tool_definition(tool_name)
                if tool_def:
                    tools.append(tool_def)
            
            return MCPMessage(
                id=message.id,
                result={"tools": tools}
            )
            
        except Exception as e:
            logger.error("Error in tools list", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    async def _handle_tools_call(self, message: MCPMessage, client_id: str) -> MCPMessage:
        """Handle tool call request"""
        try:
            params = message.params or {}
            tool_calls = params.get("calls", [])
            
            results = []
            for call in tool_calls:
                tool_name = call.get("name")
                arguments = call.get("arguments", {})
                call_id = call.get("id", str(uuid.uuid4()))
                
                if tool_name not in self.tools:
                    results.append({
                        "id": call_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool not found: {tool_name}"
                        }
                    })
                    continue
                
                try:
                    # Check rate limits
                    if not self._check_rate_limit(client_id, tool_name):
                        results.append({
                            "id": call_id,
                            "error": {
                                "code": -429,
                                "message": "Rate limit exceeded"
                            }
                        })
                        continue
                    
                    # Execute tool
                    tool_func = self.tools[tool_name]
                    result = await tool_func(**arguments)
                    
                    results.append({
                        "id": call_id,
                        "result": result
                    })
                    
                except Exception as e:
                    logger.error("Tool execution error", tool=tool_name, error=str(e))
                    results.append({
                        "id": call_id,
                        "error": {
                            "code": -32603,
                            "message": f"Tool execution failed: {str(e)}"
                        }
                    })
            
            return MCPMessage(
                id=message.id,
                result={"calls": results}
            )
            
        except Exception as e:
            logger.error("Error in tools call", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    async def _handle_tools_get(self, message: MCPMessage) -> MCPMessage:
        """Handle tool definition request"""
        try:
            params = message.params or {}
            tool_name = params.get("name")
            
            if not tool_name:
                return self._create_error_response(
                    message.id, -32602, "Invalid params", "Tool name required"
                )
            
            tool_def = self.tool_definitions.get_tool_definition(tool_name)
            if not tool_def:
                return self._create_error_response(
                    message.id, -32601, "Tool not found", f"Tool not found: {tool_name}"
                )
            
            return MCPMessage(
                id=message.id,
                result={"tool": tool_def}
            )
            
        except Exception as e:
            logger.error("Error in tools get", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    async def _handle_resources_list(self, message: MCPMessage) -> MCPMessage:
        """Handle resources list request"""
        try:
            resources = [
                {
                    "uri": "alpha-discovery://market-data",
                    "name": "Market Data Stream",
                    "description": "Real-time market data stream",
                    "mimeType": "application/json"
                },
                {
                    "uri": "alpha-discovery://sentiment-data",
                    "name": "Sentiment Data Stream",
                    "description": "Real-time sentiment analysis",
                    "mimeType": "application/json"
                },
                {
                    "uri": "alpha-discovery://order-flow",
                    "name": "Order Flow Data",
                    "description": "Real-time order flow analysis",
                    "mimeType": "application/json"
                }
            ]
            
            return MCPMessage(
                id=message.id,
                result={"resources": resources}
            )
            
        except Exception as e:
            logger.error("Error in resources list", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    async def _handle_resources_read(self, message: MCPMessage) -> MCPMessage:
        """Handle resource read request"""
        try:
            params = message.params or {}
            uri = params.get("uri")
            
            if not uri:
                return self._create_error_response(
                    message.id, -32602, "Invalid params", "Resource URI required"
                )
            
            # Get resource data based on URI
            if uri == "alpha-discovery://market-data":
                data = await self._get_market_data_stream()
            elif uri == "alpha-discovery://sentiment-data":
                data = await self._get_sentiment_data_stream()
            elif uri == "alpha-discovery://order-flow":
                data = await self._get_order_flow_stream()
            else:
                return self._create_error_response(
                    message.id, -32601, "Resource not found", f"Resource not found: {uri}"
                )
            
            return MCPMessage(
                id=message.id,
                result={
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(data)
                        }
                    ]
                }
            )
            
        except Exception as e:
            logger.error("Error in resources read", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    async def _handle_resources_subscribe(self, message: MCPMessage, client_id: str) -> MCPMessage:
        """Handle resource subscription request"""
        try:
            params = message.params or {}
            uris = params.get("uris", [])
            
            if not uris:
                return self._create_error_response(
                    message.id, -32602, "Invalid params", "Resource URIs required"
                )
            
            # Add to subscriptions
            if client_id not in self.subscriptions:
                self.subscriptions[client_id] = []
            
            for uri in uris:
                if uri not in self.subscriptions[client_id]:
                    self.subscriptions[client_id].append(uri)
            
            # Start streaming if not already running
            asyncio.create_task(self._start_streaming(client_id, uris))
            
            return MCPMessage(
                id=message.id,
                result={"subscribed": True}
            )
            
        except Exception as e:
            logger.error("Error in resources subscribe", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    async def _handle_resources_unsubscribe(self, message: MCPMessage, client_id: str) -> MCPMessage:
        """Handle resource unsubscription request"""
        try:
            params = message.params or {}
            uris = params.get("uris", [])
            
            if client_id in self.subscriptions:
                for uri in uris:
                    if uri in self.subscriptions[client_id]:
                        self.subscriptions[client_id].remove(uri)
            
            return MCPMessage(
                id=message.id,
                result={"unsubscribed": True}
            )
            
        except Exception as e:
            logger.error("Error in resources unsubscribe", error=str(e))
            return self._create_error_response(message.id, -32603, "Internal error", str(e))
    
    # Tool implementations
    async def _get_orderbook(self, symbol: str, exchange: str = "alpaca") -> Dict[str, Any]:
        """Get order book data"""
        try:
            orderbook = await self.market_data_tool.get_level2_data(symbol, exchange)
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now().isoformat(),
                "bids": orderbook.get("bids", []),
                "asks": orderbook.get("asks", []),
                "spread": orderbook.get("spread", 0),
                "bid_volume": orderbook.get("bid_volume", 0),
                "ask_volume": orderbook.get("ask_volume", 0)
            }
        except Exception as e:
            logger.error("Error getting orderbook", symbol=symbol, error=str(e))
            raise
    
    async def _get_reddit_sentiment(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get Reddit sentiment data"""
        try:
            sentiment = await self.reddit_scraper.get_sentiment(symbol, timeframe)
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "sentiment": sentiment.get("sentiment", 0),
                "volume": sentiment.get("volume", 0),
                "positive_ratio": sentiment.get("positive_ratio", 0),
                "negative_ratio": sentiment.get("negative_ratio", 0),
                "trending_topics": sentiment.get("trending_topics", [])
            }
        except Exception as e:
            logger.error("Error getting Reddit sentiment", symbol=symbol, error=str(e))
            raise
    
    async def _calculate_microstructure_features(self, symbol: str, timeframe: str = "1m") -> Dict[str, Any]:
        """Calculate microstructure features"""
        try:
            order_flow = await self.order_flow_tool.get_order_flow(symbol, timeframe)
            market_data = await self.market_data_tool.get_level2_data(symbol)
            
            features = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "bid_ask_spread": market_data.get("spread", 0),
                "order_imbalance": order_flow.get("order_imbalance", 0),
                "liquidity_score": order_flow.get("liquidity_score", 0),
                "market_impact": order_flow.get("market_impact", 0),
                "flow_pattern": order_flow.get("flow_pattern", "unknown"),
                "large_orders": len(order_flow.get("large_orders", []))
            }
            
            return features
        except Exception as e:
            logger.error("Error calculating microstructure features", symbol=symbol, error=str(e))
            raise
    
    async def _detect_regime_change(self, symbol: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Detect market regime changes"""
        try:
            # Use Claude for regime detection (best reasoning)
            prompt = f"""
            Analyze the market regime for {symbol} over the past {lookback_days} days.
            Consider volatility, trend strength, and mean reversion patterns.
            Return a JSON with regime type, confidence, and transition probability.
            """
            
            response = await model_manager.get_response(prompt, TaskType.REASONING)
            
            # Parse response and add market data
            market_data = await self.market_data_tool.get_current_price(symbol)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "regime_analysis": response.get("content", ""),
                "current_price": market_data.get("price", 0),
                "volatility": market_data.get("change_percent", 0),
                "lookback_days": lookback_days
            }
        except Exception as e:
            logger.error("Error detecting regime change", symbol=symbol, error=str(e))
            raise
    
    async def _get_market_data(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Get market data"""
        try:
            current_price = await self.market_data_tool.get_current_price(symbol)
            indicators = await self.market_data_tool.get_market_indicators(symbol, timeframe)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price.get("price", 0),
                "volume": current_price.get("volume", 0),
                "change": current_price.get("change", 0),
                "change_percent": current_price.get("change_percent", 0),
                "indicators": indicators
            }
        except Exception as e:
            logger.error("Error getting market data", symbol=symbol, error=str(e))
            raise
    
    async def _analyze_order_flow(self, symbol: str, timeframe: str = "1m") -> Dict[str, Any]:
        """Analyze order flow"""
        try:
            order_flow = await self.order_flow_tool.get_order_flow(symbol, timeframe)
            unusual_activity = await self.order_flow_tool.detect_unusual_activity(symbol)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "order_flow": order_flow,
                "unusual_activity": unusual_activity
            }
        except Exception as e:
            logger.error("Error analyzing order flow", symbol=symbol, error=str(e))
            raise
    
    async def _get_technical_indicators(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Get technical indicators"""
        try:
            indicators = await self.market_data_tool.get_market_indicators(symbol, timeframe)
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "indicators": indicators
            }
        except Exception as e:
            logger.error("Error getting technical indicators", symbol=symbol, error=str(e))
            raise
    
    async def _stream_market_data(self, symbol: str, duration_seconds: int = 60) -> Dict[str, Any]:
        """Stream market data for specified duration"""
        try:
            # This would implement real-time streaming
            # For now, return a single snapshot
            data = await self._get_market_data(symbol)
            return {
                "symbol": symbol,
                "stream_duration": duration_seconds,
                "data_points": [data],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Error streaming market data", symbol=symbol, error=str(e))
            raise
    
    async def _get_news_sentiment(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get news sentiment analysis"""
        try:
            # Use Gemini for news analysis (multimodal)
            prompt = f"""
            Analyze news sentiment for {symbol} over the past {timeframe}.
            Consider recent news articles, earnings reports, and market events.
            Return sentiment score and key news items.
            """
            
            response = await model_manager.get_response(prompt, TaskType.MULTIMODAL)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "sentiment_analysis": response.get("content", ""),
                "timeframe": timeframe
            }
        except Exception as e:
            logger.error("Error getting news sentiment", symbol=symbol, error=str(e))
            raise
    
    async def _calculate_risk_metrics(self, symbol: str, portfolio_value: float = 100000) -> Dict[str, Any]:
        """Calculate risk metrics"""
        try:
            # Use Claude for risk calculation (best reasoning)
            prompt = f"""
            Calculate risk metrics for {symbol} with portfolio value ${portfolio_value}.
            Consider volatility, VaR, maximum drawdown, and position sizing.
            Return comprehensive risk analysis.
            """
            
            response = await model_manager.get_response(prompt, TaskType.REASONING)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "risk_analysis": response.get("content", ""),
                "recommended_position_size": portfolio_value * 0.02  # 2% rule
            }
        except Exception as e:
            logger.error("Error calculating risk metrics", symbol=symbol, error=str(e))
            raise
    
    # Helper methods
    def _check_rate_limit(self, client_id: str, tool_name: str) -> bool:
        """Check rate limits for tool usage"""
        try:
            key = f"{client_id}:{tool_name}"
            now = datetime.now()
            
            if key not in self.rate_limits:
                self.rate_limits[key] = {"count": 0, "reset_time": now + timedelta(minutes=1)}
            
            limit_info = self.rate_limits[key]
            
            if now > limit_info["reset_time"]:
                limit_info["count"] = 0
                limit_info["reset_time"] = now + timedelta(minutes=1)
            
            # Allow 10 calls per minute per tool per client
            if limit_info["count"] >= 10:
                return False
            
            limit_info["count"] += 1
            return True
            
        except Exception as e:
            logger.error("Error checking rate limit", error=str(e))
            return True  # Allow if rate limiting fails
    
    def _create_error_response(self, message_id: Any, code: int, message: str, data: str = None) -> MCPMessage:
        """Create error response"""
        error = {"code": code, "message": message}
        if data:
            error["data"] = data
        
        return MCPMessage(
            id=message_id,
            error=error
        )
    
    def _create_pong_response(self, message_id: Any) -> MCPMessage:
        """Create pong response"""
        return MCPMessage(
            id=message_id,
            result={"pong": True}
        )
    
    # Streaming methods
    async def _start_streaming(self, client_id: str, uris: List[str]):
        """Start streaming data to client"""
        try:
            while client_id in self.clients:
                for uri in uris:
                    if uri == "alpha-discovery://market-data":
                        data = await self._get_market_data_stream()
                    elif uri == "alpha-discovery://sentiment-data":
                        data = await self._get_sentiment_data_stream()
                    elif uri == "alpha-discovery://order-flow":
                        data = await self._get_order_flow_stream()
                    else:
                        continue
                    
                    # Send streaming update
                    notification = {
                        "jsonrpc": "2.0",
                        "method": "resources/update",
                        "params": {
                            "uri": uri,
                            "content": data
                        }
                    }
                    
                    websocket = self.clients.get(client_id)
                    if websocket:
                        await websocket.send(json.dumps(notification))
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            logger.error("Error in streaming", client_id=client_id, error=str(e))
    
    async def _get_market_data_stream(self) -> Dict[str, Any]:
        """Get market data for streaming"""
        return {
            "type": "market_data",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "spy": {"price": 450.25, "change": 2.5},
                "qqq": {"price": 380.50, "change": -1.2},
                "tsla": {"price": 250.75, "change": 5.8}
            }
        }
    
    async def _get_sentiment_data_stream(self) -> Dict[str, Any]:
        """Get sentiment data for streaming"""
        return {
            "type": "sentiment_data",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "overall_sentiment": 0.15,
                "top_symbols": ["TSLA", "AAPL", "NVDA"],
                "trending_topics": ["earnings", "AI", "EV"]
            }
        }
    
    async def _get_order_flow_stream(self) -> Dict[str, Any]:
        """Get order flow data for streaming"""
        return {
            "type": "order_flow",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_volume": 1500000,
                "buy_volume": 800000,
                "sell_volume": 700000,
                "large_orders": 5
            }
        }

    def _setup_http_routes(self):
        """Set up FastAPI routes for ML services."""
        @self.app.post("/sentiment-analysis")
        async def sentiment_analysis(request: SentimentRequest):
            try:
                result = await ml_service.analyze_sentiment(request.text)
                return {"sentiment": result}
            except Exception as e:
                logger.error("Error in sentiment analysis", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/emotion-analysis")
        async def emotion_analysis(request: EmotionRequest):
            try:
                result = await ml_service.analyze_emotion(request.text)
                return {"emotion": result}
            except Exception as e:
                logger.error("Error in emotion analysis", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ner")
        async def ner(request: NERRequest):
            try:
                result = await ml_service.extract_entities(request.text)
                return {"entities": result}
            except Exception as e:
                logger.error("Error in NER", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/similarity")
        async def similarity(request: SimilarityRequest):
            try:
                result = await ml_service.calculate_similarity(request.text1, request.text2)
                return {"similarity": result}
            except Exception as e:
                logger.error("Error in similarity", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

        # Microstructure ML endpoints
        @self.app.post("/regime-detection")
        async def regime_detection(request: RegimeDetectionRequest):
            try:
                result = await ml_service.detect_regime(request.features)
                return {"regime": result}
            except Exception as e:
                logger.error("Error in regime detection", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/imbalance-prediction")
        async def imbalance_prediction(request: ImbalancePredictionRequest):
            try:
                result = await ml_service.predict_imbalance(request.features)
                return {"imbalance": result}
            except Exception as e:
                logger.error("Error in imbalance prediction", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/lambda-prediction")
        async def lambda_prediction(request: LambdaPredictionRequest):
            try:
                result = await ml_service.predict_lambda(request.features)
                return {"lambda": result}
            except Exception as e:
                logger.error("Error in lambda prediction", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/flow-classification")
        async def flow_classification(request: FlowClassificationRequest):
            try:
                result = await ml_service.classify_flow(request.features)
                return {"flow": result}
            except Exception as e:
                logger.error("Error in flow classification", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/ml-health")
        async def ml_health():
            try:
                status = ml_service.get_health_status()
                return {"ml_status": status}
            except Exception as e:
                logger.error("Error getting ML health status", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        # State-of-the-art ML endpoints (2025 upgrades)
        @self.app.post("/gliner-ner")
        async def gliner_ner(request: GLiNERRequest):
            try:
                result = await ml_service.extract_entities_gliner(request.text, request.entity_types)
                return {"entities": result}
            except Exception as e:
                logger.error("Error in GLiNER NER", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/xgboost-imbalance")
        async def xgboost_imbalance_prediction(request: ImbalancePredictionRequest):
            try:
                result = await ml_service.predict_imbalance_xgb(request.features)
                return {"imbalance": result}
            except Exception as e:
                logger.error("Error in XGBoost imbalance prediction", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/lightgbm-lambda")
        async def lightgbm_lambda_prediction(request: LambdaPredictionRequest):
            try:
                result = await ml_service.predict_lambda_lgb(request.features)
                return {"lambda": result}
            except Exception as e:
                logger.error("Error in LightGBM lambda prediction", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/catboost-flow")
        async def catboost_flow_classification(request: FlowClassificationRequest):
            try:
                result = await ml_service.classify_flow_catboost(request.features)
                return {"flow": result}
            except Exception as e:
                logger.error("Error in CatBoost flow classification", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/modern-regime")
        async def modern_regime_detection(request: RegimeDetectionRequest):
            try:
                result = await ml_service.detect_regime_modern(request.features)
                return {"regime": result}
            except Exception as e:
                logger.error("Error in modern regime detection", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        # Alternative Data Agent endpoints (Phase 3)
        @self.app.post("/financial-sentiment")
        async def financial_sentiment_analysis(request: FinancialSentimentRequest):
            try:
                result = await ml_service.analyze_financial_sentiment(request.text)
                return {"sentiment": result}
            except Exception as e:
                logger.error("Error in financial sentiment analysis", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/vision-analysis")
        async def vision_analysis(request: VisionAnalysisRequest):
            try:
                result = await ml_service.analyze_vision(request.image_path, request.analysis_type)
                return {"vision": result}
            except Exception as e:
                logger.error("Error in vision analysis", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        # Advanced ML endpoints (Phase 4)
        @self.app.post("/mlp-prediction")
        async def mlp_prediction(request: MLPPredictionRequest):
            try:
                result = await ml_service.predict_with_mlp(request.features, request.target_name)
                return {"prediction": result}
            except Exception as e:
                logger.error("Error in MLP prediction", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/gradient-boosting")
        async def gradient_boosting_prediction(request: GradientBoostingRequest):
            try:
                result = await ml_service.predict_with_gradient_boosting(request.features)
                return {"prediction": result}
            except Exception as e:
                logger.error("Error in Gradient Boosting prediction", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/risk-pca")
        async def risk_pca_analysis(request: RiskPCARequest):
            try:
                result = await ml_service.analyze_risk_pca(request.features)
                return {"risk_analysis": result}
            except Exception as e:
                logger.error("Error in PCA risk analysis", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/clustering")
        async def clustering_analysis(request: ClusteringRequest):
            try:
                result = await ml_service.perform_clustering(request.features, request.n_clusters)
                return {"clustering": result}
            except Exception as e:
                logger.error("Error in clustering analysis", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/anomaly-detection")
        async def anomaly_detection(request: AnomalyDetectionRequest):
            try:
                result = await ml_service.detect_anomalies(request.features)
                return {"anomaly_analysis": result}
            except Exception as e:
                logger.error("Error in anomaly detection", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        # Analytics/Performance endpoints (Phase 5.1)
        @self.app.post("/linear-regression")
        async def linear_regression(request: LinearRegressionRequest):
            try:
                result = await ml_service.perform_linear_regression(request.X, request.y)
                return {"regression_result": result}
            except Exception as e:
                logger.error("Error in linear regression", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ridge-regression")
        async def ridge_regression(request: RidgeRegressionRequest):
            try:
                result = await ml_service.perform_ridge_regression(request.X, request.y, request.alpha)
                return {"regression_result": result}
            except Exception as e:
                logger.error("Error in ridge regression", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        # Reinforcement Learning endpoints (Phase 5.1)
        @self.app.post("/rl-training")
        async def rl_training(request: RLTrainingRequest):
            try:
                result = await ml_service.train_rl_model(request.model_type, request.total_timesteps)
                return {"training_result": result}
            except Exception as e:
                logger.error("Error in RL training", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/rl-prediction")
        async def rl_prediction(request: RLPredictionRequest):
            try:
                result = await ml_service.predict_with_rl_model(request.model_type, request.observation)
                return {"prediction_result": result}
            except Exception as e:
                logger.error("Error in RL prediction", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))


# Global MCP server instance
mcp_server = MCPServer() 