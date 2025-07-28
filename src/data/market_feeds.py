"""
Real-time Market Data Pipeline for Alpha Discovery Platform

This pipeline connects to multiple exchanges, streams real-time data, normalizes it,
performs quality checks, stores in TimescaleDB, and publishes to Kafka for real-time processing.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict, deque
import uuid
import ccxt
import ccxt.pro as ccxtpro
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import psycopg2
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import redis
import websockets
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import warnings
warnings.filterwarnings('ignore')

# Import our components
from src.utils.model_manager import ModelManager
from src.utils.error_handling import handle_errors, AlphaDiscoveryError
from src.utils.monitoring import monitor_performance, track_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class MarketDataPoint(Base):
    __tablename__ = 'market_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    data_type = Column(String, nullable=False)  # orderbook, trade, ticker, ohlcv
    price = Column(Float)
    volume = Column(Float)
    side = Column(String)  # buy/sell for trades
    raw_data = Column(JSON)
    quality_score = Column(Float, default=1.0)
    processed = Column(Boolean, default=False)

class DataQualityMetrics(Base):
    __tablename__ = 'data_quality_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    metric_type = Column(String, nullable=False)  # latency, completeness, accuracy
    value = Column(Float, nullable=False)
    threshold = Column(Float)
    status = Column(String)  # good, warning, error

class ExchangeStatus(Base):
    __tablename__ = 'exchange_status'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String, nullable=False, unique=True)
    status = Column(String, nullable=False)  # connected, disconnected, error
    last_update = Column(DateTime, nullable=False)
    error_message = Column(String)
    reconnect_attempts = Column(Integer, default=0)

class DataType(Enum):
    """Data type classification"""
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    TICKER = "ticker"
    OHLCV = "ohlcv"
    FUNDING = "funding"
    LIQUIDATION = "liquidation"

class QualityStatus(Enum):
    """Data quality status"""
    GOOD = "good"
    WARNING = "warning"
    ERROR = "error"
    STALE = "stale"

@dataclass
class OrderBookData:
    """Normalized order book data"""
    exchange: str
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, volume)
    asks: List[Tuple[float, float]]  # (price, volume)
    sequence: Optional[int] = None
    checksum: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeData:
    """Normalized trade data"""
    exchange: str
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    side: str  # buy/sell
    trade_id: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataQualityCheck:
    """Data quality check result"""
    metric_type: str
    value: float
    threshold: float
    status: QualityStatus
    message: str
    timestamp: datetime

class ExchangeConnector:
    """Handles connection to individual exchanges"""
    
    def __init__(self, exchange_id: str, config: Dict[str, Any]):
        self.exchange_id = exchange_id
        self.config = config
        self.exchange = None
        self.ws_exchange = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.callbacks = {}
        self.subscriptions = set()
        
    async def connect(self):
        """Connect to exchange"""
        try:
            # Initialize REST API
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(self.config)
            
            # Initialize WebSocket API
            if hasattr(ccxtpro, self.exchange_id):
                ws_exchange_class = getattr(ccxtpro, self.exchange_id)
                self.ws_exchange = ws_exchange_class(self.config)
            
            # Test connection
            await self.exchange.load_markets()
            self.connected = True
            self.reconnect_attempts = 0
            
            logger.info(f"Connected to {self.exchange_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from exchange"""
        try:
            if self.ws_exchange:
                await self.ws_exchange.close()
            if self.exchange:
                await self.exchange.close()
            
            self.connected = False
            logger.info(f"Disconnected from {self.exchange_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.exchange_id}: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def subscribe_orderbook(self, symbol: str, callback: Callable):
        """Subscribe to order book updates"""
        try:
            if not self.ws_exchange:
                raise AlphaDiscoveryError(f"WebSocket not supported for {self.exchange_id}")
            
            self.callbacks[f"orderbook_{symbol}"] = callback
            self.subscriptions.add(f"orderbook_{symbol}")
            
            # Start streaming
            asyncio.create_task(self._stream_orderbook(symbol))
            
            logger.info(f"Subscribed to {symbol} orderbook on {self.exchange_id}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to orderbook for {symbol} on {self.exchange_id}: {e}")
            raise
    
    async def _stream_orderbook(self, symbol: str):
        """Stream order book data"""
        try:
            while self.connected and f"orderbook_{symbol}" in self.subscriptions:
                try:
                    orderbook = await self.ws_exchange.watch_order_book(symbol)
                    
                    # Normalize data
                    normalized_data = OrderBookData(
                        exchange=self.exchange_id,
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(orderbook['timestamp'] / 1000),
                        bids=[(float(bid[0]), float(bid[1])) for bid in orderbook['bids'][:20]],
                        asks=[(float(ask[0]), float(ask[1])) for ask in orderbook['asks'][:20]],
                        raw_data=orderbook
                    )
                    
                    # Call callback
                    callback = self.callbacks.get(f"orderbook_{symbol}")
                    if callback:
                        await callback(normalized_data)
                        
                except Exception as e:
                    logger.error(f"Error streaming orderbook for {symbol}: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Orderbook stream error for {symbol}: {e}")
    
    async def subscribe_trades(self, symbol: str, callback: Callable):
        """Subscribe to trade updates"""
        try:
            if not self.ws_exchange:
                raise AlphaDiscoveryError(f"WebSocket not supported for {self.exchange_id}")
            
            self.callbacks[f"trades_{symbol}"] = callback
            self.subscriptions.add(f"trades_{symbol}")
            
            # Start streaming
            asyncio.create_task(self._stream_trades(symbol))
            
            logger.info(f"Subscribed to {symbol} trades on {self.exchange_id}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to trades for {symbol} on {self.exchange_id}: {e}")
            raise
    
    async def _stream_trades(self, symbol: str):
        """Stream trade data"""
        try:
            while self.connected and f"trades_{symbol}" in self.subscriptions:
                try:
                    trades = await self.ws_exchange.watch_trades(symbol)
                    
                    for trade in trades:
                        # Normalize data
                        normalized_trade = TradeData(
                            exchange=self.exchange_id,
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000),
                            price=float(trade['price']),
                            volume=float(trade['amount']),
                            side=trade['side'],
                            trade_id=trade.get('id'),
                            raw_data=trade
                        )
                        
                        # Call callback
                        callback = self.callbacks.get(f"trades_{symbol}")
                        if callback:
                            await callback(normalized_trade)
                            
                except Exception as e:
                    logger.error(f"Error streaming trades for {symbol}: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Trade stream error for {symbol}: {e}")
    
    async def get_historical_ohlcv(self, symbol: str, timeframe: str, 
                                  since: datetime, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get historical OHLCV data"""
        try:
            if not self.exchange:
                raise AlphaDiscoveryError(f"Exchange {self.exchange_id} not connected")
            
            # Convert datetime to timestamp
            since_timestamp = int(since.timestamp() * 1000)
            
            # Fetch data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since_timestamp, limit)
            
            # Normalize data
            normalized_data = []
            for candle in ohlcv:
                normalized_data.append({
                    'exchange': self.exchange_id,
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise

class DataQualityChecker:
    """Performs data quality checks"""
    
    def __init__(self):
        self.quality_thresholds = {
            'latency_ms': 1000,  # 1 second max latency
            'completeness_pct': 95,  # 95% completeness
            'accuracy_score': 0.95,  # 95% accuracy
            'staleness_seconds': 10,  # 10 seconds max staleness
            'spread_ratio': 0.1,  # 10% max spread
            'volume_ratio': 0.1  # 10% volume deviation
        }
        
    def check_data_quality(self, data: Union[OrderBookData, TradeData]) -> List[DataQualityCheck]:
        """Perform comprehensive data quality checks"""
        try:
            checks = []
            
            # Latency check
            latency_check = self._check_latency(data)
            checks.append(latency_check)
            
            # Staleness check
            staleness_check = self._check_staleness(data)
            checks.append(staleness_check)
            
            # Data-specific checks
            if isinstance(data, OrderBookData):
                checks.extend(self._check_orderbook_quality(data))
            elif isinstance(data, TradeData):
                checks.extend(self._check_trade_quality(data))
            
            return checks
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return []
    
    def _check_latency(self, data: Union[OrderBookData, TradeData]) -> DataQualityCheck:
        """Check data latency"""
        try:
            current_time = datetime.now()
            latency_ms = (current_time - data.timestamp).total_seconds() * 1000
            
            threshold = self.quality_thresholds['latency_ms']
            status = QualityStatus.GOOD if latency_ms <= threshold else QualityStatus.WARNING
            
            return DataQualityCheck(
                metric_type='latency',
                value=latency_ms,
                threshold=threshold,
                status=status,
                message=f"Latency: {latency_ms:.0f}ms",
                timestamp=current_time
            )
            
        except Exception as e:
            logger.error(f"Latency check failed: {e}")
            return DataQualityCheck(
                metric_type='latency',
                value=0,
                threshold=0,
                status=QualityStatus.ERROR,
                message=f"Latency check error: {e}",
                timestamp=datetime.now()
            )
    
    def _check_staleness(self, data: Union[OrderBookData, TradeData]) -> DataQualityCheck:
        """Check data staleness"""
        try:
            current_time = datetime.now()
            staleness_seconds = (current_time - data.timestamp).total_seconds()
            
            threshold = self.quality_thresholds['staleness_seconds']
            status = QualityStatus.GOOD if staleness_seconds <= threshold else QualityStatus.STALE
            
            return DataQualityCheck(
                metric_type='staleness',
                value=staleness_seconds,
                threshold=threshold,
                status=status,
                message=f"Staleness: {staleness_seconds:.1f}s",
                timestamp=current_time
            )
            
        except Exception as e:
            logger.error(f"Staleness check failed: {e}")
            return DataQualityCheck(
                metric_type='staleness',
                value=0,
                threshold=0,
                status=QualityStatus.ERROR,
                message=f"Staleness check error: {e}",
                timestamp=datetime.now()
            )
    
    def _check_orderbook_quality(self, data: OrderBookData) -> List[DataQualityCheck]:
        """Check order book specific quality"""
        try:
            checks = []
            
            # Spread check
            if data.bids and data.asks:
                best_bid = max(data.bids, key=lambda x: x[0])[0]
                best_ask = min(data.asks, key=lambda x: x[0])[0]
                spread_ratio = (best_ask - best_bid) / best_bid
                
                threshold = self.quality_thresholds['spread_ratio']
                status = QualityStatus.GOOD if spread_ratio <= threshold else QualityStatus.WARNING
                
                checks.append(DataQualityCheck(
                    metric_type='spread',
                    value=spread_ratio,
                    threshold=threshold,
                    status=status,
                    message=f"Spread ratio: {spread_ratio:.4f}",
                    timestamp=datetime.now()
                ))
            
            # Depth check
            bid_depth = sum(bid[1] for bid in data.bids)
            ask_depth = sum(ask[1] for ask in data.asks)
            
            if bid_depth > 0 and ask_depth > 0:
                depth_imbalance = abs(bid_depth - ask_depth) / (bid_depth + ask_depth)
                
                checks.append(DataQualityCheck(
                    metric_type='depth_imbalance',
                    value=depth_imbalance,
                    threshold=0.8,  # 80% imbalance threshold
                    status=QualityStatus.GOOD if depth_imbalance <= 0.8 else QualityStatus.WARNING,
                    message=f"Depth imbalance: {depth_imbalance:.2f}",
                    timestamp=datetime.now()
                ))
            
            return checks
            
        except Exception as e:
            logger.error(f"Orderbook quality check failed: {e}")
            return []
    
    def _check_trade_quality(self, data: TradeData) -> List[DataQualityCheck]:
        """Check trade specific quality"""
        try:
            checks = []
            
            # Price reasonableness check
            if data.price > 0:
                checks.append(DataQualityCheck(
                    metric_type='price_validity',
                    value=1.0,
                    threshold=1.0,
                    status=QualityStatus.GOOD,
                    message="Price is valid",
                    timestamp=datetime.now()
                ))
            else:
                checks.append(DataQualityCheck(
                    metric_type='price_validity',
                    value=0.0,
                    threshold=1.0,
                    status=QualityStatus.ERROR,
                    message="Invalid price",
                    timestamp=datetime.now()
                ))
            
            # Volume reasonableness check
            if data.volume > 0:
                checks.append(DataQualityCheck(
                    metric_type='volume_validity',
                    value=1.0,
                    threshold=1.0,
                    status=QualityStatus.GOOD,
                    message="Volume is valid",
                    timestamp=datetime.now()
                ))
            else:
                checks.append(DataQualityCheck(
                    metric_type='volume_validity',
                    value=0.0,
                    threshold=1.0,
                    status=QualityStatus.ERROR,
                    message="Invalid volume",
                    timestamp=datetime.now()
                ))
            
            return checks
            
        except Exception as e:
            logger.error(f"Trade quality check failed: {e}")
            return []

class TimescaleDBManager:
    """Manages TimescaleDB for time-series data storage"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.connection_pool = None
        
    async def initialize(self):
        """Initialize TimescaleDB with hypertables"""
        try:
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Create hypertables for time-series data
            with self.engine.connect() as conn:
                # Market data hypertable
                conn.execute("""
                    SELECT create_hypertable('market_data', 'timestamp', 
                                           chunk_time_interval => INTERVAL '1 hour',
                                           if_not_exists => TRUE);
                """)
                
                # Data quality metrics hypertable
                conn.execute("""
                    SELECT create_hypertable('data_quality_metrics', 'timestamp',
                                           chunk_time_interval => INTERVAL '1 day',
                                           if_not_exists => TRUE);
                """)
                
                # Create indexes for better query performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_exchange_symbol 
                    ON market_data (exchange, symbol, timestamp DESC);
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_type 
                    ON market_data (data_type, timestamp DESC);
                """)
            
            logger.info("TimescaleDB initialized successfully")
            
        except Exception as e:
            logger.error(f"TimescaleDB initialization failed: {e}")
            raise
    
    async def store_market_data(self, data: Union[OrderBookData, TradeData]):
        """Store market data in TimescaleDB"""
        try:
            session = self.Session()
            
            # Determine data type and extract relevant fields
            if isinstance(data, OrderBookData):
                data_type = DataType.ORDERBOOK.value
                price = data.bids[0][0] if data.bids else None
                volume = sum(bid[1] for bid in data.bids) if data.bids else None
                side = None
            elif isinstance(data, TradeData):
                data_type = DataType.TRADE.value
                price = data.price
                volume = data.volume
                side = data.side
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Create market data record
            market_data = MarketDataPoint(
                exchange=data.exchange,
                symbol=data.symbol,
                timestamp=data.timestamp,
                data_type=data_type,
                price=price,
                volume=volume,
                side=side,
                raw_data=data.raw_data
            )
            
            session.add(market_data)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    async def store_quality_metrics(self, metrics: List[DataQualityCheck], 
                                   exchange: str, symbol: str):
        """Store data quality metrics"""
        try:
            session = self.Session()
            
            for metric in metrics:
                quality_metric = DataQualityMetrics(
                    exchange=exchange,
                    symbol=symbol,
                    timestamp=metric.timestamp,
                    metric_type=metric.metric_type,
                    value=metric.value,
                    threshold=metric.threshold,
                    status=metric.status.value
                )
                
                session.add(quality_metric)
            
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to store quality metrics: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
    
    async def get_historical_data(self, exchange: str, symbol: str, 
                                 data_type: DataType, start_time: datetime, 
                                 end_time: datetime) -> pd.DataFrame:
        """Get historical data from TimescaleDB"""
        try:
            query = """
                SELECT timestamp, price, volume, side, raw_data
                FROM market_data
                WHERE exchange = %s AND symbol = %s AND data_type = %s
                AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(
                query,
                self.engine,
                params=[exchange, symbol, data_type.value, start_time, end_time]
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()

class KafkaPublisher:
    """Publishes market data to Kafka for real-time processing"""
    
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        
    async def initialize(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
                compression_type='gzip'
            )
            
            logger.info("Kafka producer initialized")
            
        except Exception as e:
            logger.error(f"Kafka producer initialization failed: {e}")
            raise
    
    async def publish_market_data(self, data: Union[OrderBookData, TradeData]):
        """Publish market data to Kafka"""
        try:
            if not self.producer:
                raise AlphaDiscoveryError("Kafka producer not initialized")
            
            # Create message
            message = {
                'exchange': data.exchange,
                'symbol': data.symbol,
                'timestamp': data.timestamp.isoformat(),
                'data_type': 'orderbook' if isinstance(data, OrderBookData) else 'trade',
                'data': data.__dict__
            }
            
            # Determine topic
            topic = f"market_data_{data.exchange}_{data.symbol}".replace('/', '_')
            key = f"{data.exchange}_{data.symbol}"
            
            # Send message
            future = self.producer.send(topic, value=message, key=key)
            
            # Optional: Wait for acknowledgment
            # record_metadata = future.get(timeout=1)
            
        except Exception as e:
            logger.error(f"Failed to publish market data: {e}")
    
    async def close(self):
        """Close Kafka producer"""
        try:
            if self.producer:
                self.producer.close()
                logger.info("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")

class MarketDataPipeline:
    """
    Comprehensive Real-time Market Data Pipeline
    
    Connects to multiple exchanges, streams real-time data, normalizes it,
    performs quality checks, stores in TimescaleDB, and publishes to Kafka.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchanges = {}
        self.db_manager = None
        self.kafka_publisher = None
        self.quality_checker = DataQualityChecker()
        self.redis_client = None
        
        # State management
        self.running = False
        self.subscriptions = {}
        self.reconnect_tasks = {}
        
        # Performance tracking
        self.message_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.last_message_time = defaultdict(lambda: datetime.now())
        
        logger.info("MarketDataPipeline initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize database
            db_config = self.config.get('database', {})
            if db_config:
                self.db_manager = TimescaleDBManager(db_config['connection_string'])
                await self.db_manager.initialize()
            
            # Initialize Kafka
            kafka_config = self.config.get('kafka', {})
            if kafka_config:
                self.kafka_publisher = KafkaPublisher(kafka_config['bootstrap_servers'])
                await self.kafka_publisher.initialize()
            
            # Initialize Redis for caching
            redis_url = os.environ.get('REDIS_URL')
            if redis_url:
                self.redis_client = redis.from_url(redis_url)
            else:
            redis_config = self.config.get('redis', {})
            if redis_config:
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0)
                )
            
            # Initialize exchange connections
            exchange_configs = self.config.get('exchanges', {})
            for exchange_id, exchange_config in exchange_configs.items():
                connector = ExchangeConnector(exchange_id, exchange_config)
                await connector.connect()
                self.exchanges[exchange_id] = connector
            
            logger.info("MarketDataPipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise
    
    @handle_errors
    async def stream_orderbook(self, exchange: str, symbol: str):
        """Stream real-time order book data"""
        try:
            if exchange not in self.exchanges:
                raise AlphaDiscoveryError(f"Exchange {exchange} not configured")
            
            connector = self.exchanges[exchange]
            
            # Define callback for order book updates
            async def orderbook_callback(data: OrderBookData):
                try:
                    # Perform quality checks
                    quality_checks = self.quality_checker.check_data_quality(data)
                    
                    # Store in database
                    if self.db_manager:
                        await self.db_manager.store_market_data(data)
                        await self.db_manager.store_quality_metrics(quality_checks, exchange, symbol)
                    
                    # Publish to Kafka
                    if self.kafka_publisher:
                        await self.kafka_publisher.publish_market_data(data)
                    
                    # Cache in Redis
                    if self.redis_client:
                        cache_key = f"orderbook:{exchange}:{symbol}"
                        cache_data = {
                            'timestamp': data.timestamp.isoformat(),
                            'bids': data.bids[:5],  # Top 5 levels
                            'asks': data.asks[:5]
                        }
                        self.redis_client.setex(cache_key, 60, json.dumps(cache_data))
                    
                    # Update metrics
                    self.message_count[f"{exchange}_{symbol}_orderbook"] += 1
                    self.last_message_time[f"{exchange}_{symbol}_orderbook"] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error processing orderbook data: {e}")
                    self.error_count[f"{exchange}_{symbol}_orderbook"] += 1
            
            # Subscribe to order book updates
            await connector.subscribe_orderbook(symbol, orderbook_callback)
            
            # Store subscription
            self.subscriptions[f"{exchange}_{symbol}_orderbook"] = {
                'type': 'orderbook',
                'exchange': exchange,
                'symbol': symbol,
                'callback': orderbook_callback
            }
            
            logger.info(f"Started streaming orderbook for {symbol} on {exchange}")
            
        except Exception as e:
            logger.error(f"Failed to stream orderbook: {e}")
            raise
    
    @handle_errors
    async def stream_trades(self, exchange: str, symbol: str):
        """Stream real-time trade data"""
        try:
            if exchange not in self.exchanges:
                raise AlphaDiscoveryError(f"Exchange {exchange} not configured")
            
            connector = self.exchanges[exchange]
            
            # Define callback for trade updates
            async def trade_callback(data: TradeData):
                try:
                    # Perform quality checks
                    quality_checks = self.quality_checker.check_data_quality(data)
                    
                    # Store in database
                    if self.db_manager:
                        await self.db_manager.store_market_data(data)
                        await self.db_manager.store_quality_metrics(quality_checks, exchange, symbol)
                    
                    # Publish to Kafka
                    if self.kafka_publisher:
                        await self.kafka_publisher.publish_market_data(data)
                    
                    # Update metrics
                    self.message_count[f"{exchange}_{symbol}_trades"] += 1
                    self.last_message_time[f"{exchange}_{symbol}_trades"] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error processing trade data: {e}")
                    self.error_count[f"{exchange}_{symbol}_trades"] += 1
            
            # Subscribe to trade updates
            await connector.subscribe_trades(symbol, trade_callback)
            
            # Store subscription
            self.subscriptions[f"{exchange}_{symbol}_trades"] = {
                'type': 'trades',
                'exchange': exchange,
                'symbol': symbol,
                'callback': trade_callback
            }
            
            logger.info(f"Started streaming trades for {symbol} on {exchange}")
            
        except Exception as e:
            logger.error(f"Failed to stream trades: {e}")
            raise
    
    @handle_errors
    async def get_historical(self, exchange: str, symbol: str, 
                           timeframe: str, start_time: datetime, 
                           end_time: datetime) -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            if exchange not in self.exchanges:
                raise AlphaDiscoveryError(f"Exchange {exchange} not configured")
            
            connector = self.exchanges[exchange]
            
            # Get data from exchange
            historical_data = await connector.get_historical_ohlcv(
                symbol, timeframe, start_time, 1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Filter by end time
            if not df.empty:
                df = df[df['timestamp'] <= end_time]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    @handle_errors
    async def calculate_vwap(self, exchange: str, symbol: str, 
                           window_minutes: int = 60) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        try:
            # Get recent trade data from database
            if not self.db_manager:
                return None
            
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=window_minutes)
            
            df = await self.db_manager.get_historical_data(
                exchange, symbol, DataType.TRADE, start_time, end_time
            )
            
            if df.empty:
                return None
            
            # Calculate VWAP
            df['price_volume'] = df['price'] * df['volume']
            total_price_volume = df['price_volume'].sum()
            total_volume = df['volume'].sum()
            
            if total_volume > 0:
                vwap = total_price_volume / total_volume
                return float(vwap)
            
            return None
            
        except Exception as e:
            logger.error(f"VWAP calculation failed: {e}")
            return None
    
    async def handle_connection_loss(self, exchange: str):
        """Handle connection loss and reconnection"""
        try:
            logger.warning(f"Connection lost to {exchange}, attempting reconnection...")
            
            connector = self.exchanges[exchange]
            
            # Attempt reconnection with exponential backoff
            for attempt in range(connector.max_reconnect_attempts):
                try:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                    # Disconnect and reconnect
                    await connector.disconnect()
                    await connector.connect()
                    
                    # Resubscribe to all symbols
                    await self._resubscribe_after_reconnect(exchange)
                    
                    logger.info(f"Successfully reconnected to {exchange}")
                    return
                    
                except Exception as e:
                    logger.error(f"Reconnection attempt {attempt + 1} failed for {exchange}: {e}")
                    connector.reconnect_attempts += 1
            
            logger.error(f"Failed to reconnect to {exchange} after {connector.max_reconnect_attempts} attempts")
            
        except Exception as e:
            logger.error(f"Error handling connection loss for {exchange}: {e}")
    
    async def _resubscribe_after_reconnect(self, exchange: str):
        """Resubscribe to all symbols after reconnection"""
        try:
            # Find all subscriptions for this exchange
            exchange_subscriptions = {
                key: sub for key, sub in self.subscriptions.items()
                if sub['exchange'] == exchange
            }
            
            # Resubscribe
            for sub_key, subscription in exchange_subscriptions.items():
                if subscription['type'] == 'orderbook':
                    await self.stream_orderbook(subscription['exchange'], subscription['symbol'])
                elif subscription['type'] == 'trades':
                    await self.stream_trades(subscription['exchange'], subscription['symbol'])
            
            logger.info(f"Resubscribed to {len(exchange_subscriptions)} streams for {exchange}")
            
        except Exception as e:
            logger.error(f"Error resubscribing after reconnect: {e}")
    
    async def start_monitoring(self):
        """Start monitoring pipeline health"""
        try:
            while self.running:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for stale connections
                current_time = datetime.now()
                for key, last_time in self.last_message_time.items():
                    if (current_time - last_time).total_seconds() > 300:  # 5 minutes
                        logger.warning(f"Stale connection detected: {key}")
                        
                        # Extract exchange from key
                        exchange = key.split('_')[0]
                        if exchange in self.exchanges:
                            asyncio.create_task(self.handle_connection_loss(exchange))
                
                # Log statistics
                total_messages = sum(self.message_count.values())
                total_errors = sum(self.error_count.values())
                error_rate = total_errors / total_messages if total_messages > 0 else 0
                
                logger.info(f"Pipeline stats - Messages: {total_messages}, Errors: {total_errors}, Error rate: {error_rate:.2%}")
                
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    async def start(self):
        """Start the pipeline"""
        try:
            self.running = True
            
            # Start monitoring
            asyncio.create_task(self.start_monitoring())
            
            logger.info("MarketDataPipeline started")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            raise
    
    async def stop(self):
        """Stop the pipeline"""
        try:
            self.running = False
            
            # Disconnect from all exchanges
            for exchange_id, connector in self.exchanges.items():
                await connector.disconnect()
            
            # Close Kafka producer
            if self.kafka_publisher:
                await self.kafka_publisher.close()
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("MarketDataPipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status and statistics"""
        try:
            status = {
                'running': self.running,
                'exchanges': {
                    exchange_id: {
                        'connected': connector.connected,
                        'reconnect_attempts': connector.reconnect_attempts
                    }
                    for exchange_id, connector in self.exchanges.items()
                },
                'subscriptions': len(self.subscriptions),
                'message_counts': dict(self.message_count),
                'error_counts': dict(self.error_count),
                'last_message_times': {
                    key: time.isoformat() for key, time in self.last_message_time.items()
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}

# Example usage and configuration
async def main():
    """Example usage of MarketDataPipeline"""
    
    # Configuration
    config = {
        'exchanges': {
            'binance': {
                'apiKey': 'your_api_key',
                'secret': 'your_secret',
                'sandbox': True,
                'enableRateLimit': True
            },
            'coinbase': {
                'apiKey': 'your_api_key',
                'secret': 'your_secret',
                'passphrase': 'your_passphrase',
                'sandbox': True
            }
        },
        'database': {
            'connection_string': 'postgresql://user:password@localhost:5432/marketdata'
        },
        'kafka': {
            'bootstrap_servers': ['localhost:9092']
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    }
    
    # Initialize pipeline
    pipeline = MarketDataPipeline(config)
    
    try:
        # Initialize components
        await pipeline.initialize()
        
        # Start pipeline
        await pipeline.start()
        
        # Stream data for multiple symbols
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        for symbol in symbols:
            await pipeline.stream_orderbook('binance', symbol)
            await pipeline.stream_trades('binance', symbol)
        
        # Get historical data
        historical_data = await pipeline.get_historical(
            'binance', 'BTC/USDT', '1h', 
            datetime.now() - timedelta(days=7), 
            datetime.now()
        )
        
        print(f"Retrieved {len(historical_data)} historical data points")
        
        # Calculate VWAP
        vwap = await pipeline.calculate_vwap('binance', 'BTC/USDT', 60)
        print(f"VWAP: {vwap}")
        
        # Get pipeline status
        status = pipeline.get_pipeline_status()
        print(f"Pipeline status: {status}")
        
        # Run for a while
        await asyncio.sleep(60)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main()) 