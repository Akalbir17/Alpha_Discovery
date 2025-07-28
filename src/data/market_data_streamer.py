#!/usr/bin/env python3
"""
Market Data Streaming Service

Real-time market data streaming service that provides live price feeds,
volume data, and technical indicators for the Alpha Discovery platform.
"""

import asyncio
import json
import logging
import time
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import aiohttp
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from configs.config_loader import get_config
from src.utils.database import DatabaseManager
from src.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Single market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None

@dataclass
class StreamingState:
    """State management for streaming service."""
    is_running: bool = False
    connected_symbols: set = field(default_factory=set)
    last_update: Dict[str, datetime] = field(default_factory=dict)
    error_count: int = 0
    reconnect_count: int = 0
    data_points_received: int = 0
    callbacks: List[Callable] = field(default_factory=list)

class MarketDataStreamer:
    """Real-time market data streaming service."""
    
    def __init__(self):
        self.config = get_config('market_data')
        self.db_manager = DatabaseManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        
        self.state = StreamingState()
        self.data_queue = queue.Queue(maxsize=10000)
        self.price_cache = {}
        self.volume_cache = {}
        self.indicators_cache = {}
        
        # WebSocket connections
        self.websocket_connections = {}
        self.connection_lock = threading.Lock()
        
        # Data processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_tasks = []
        
        # Rate limiting
        self.rate_limiter = {}
        self.max_requests_per_second = 100
        
        logger.info("Market Data Streamer initialized")
    
    async def start_streaming(self, symbols: List[str]) -> None:
        """Start streaming market data for given symbols."""
        logger.info(f"Starting market data streaming for {len(symbols)} symbols")
        
        try:
            self.state.is_running = True
            
            # Start data processing task
            processing_task = asyncio.create_task(self._process_data_queue())
            self.processing_tasks.append(processing_task)
            
            # Start streaming for each provider
            if self.config.get('providers', {}).get('alpaca', {}).get('enabled', False):
                await self._start_alpaca_streaming(symbols)
            
            if self.config.get('providers', {}).get('polygon', {}).get('enabled', False):
                await self._start_polygon_streaming(symbols)
            
            if self.config.get('providers', {}).get('yahoo', {}).get('enabled', False):
                await self._start_yahoo_streaming(symbols)
            
            # Start heartbeat monitoring
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self.processing_tasks.append(heartbeat_task)
            
            logger.info("Market data streaming started successfully")
            
        except Exception as e:
            logger.error(f"Error starting market data streaming: {e}")
            raise
    
    async def stop_streaming(self) -> None:
        """Stop market data streaming."""
        logger.info("Stopping market data streaming...")
        
        self.state.is_running = False
        
        # Close WebSocket connections
        with self.connection_lock:
            for connection in self.websocket_connections.values():
                if connection and not connection.closed:
                    await connection.close()
            self.websocket_connections.clear()
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Market data streaming stopped")
    
    async def _start_alpaca_streaming(self, symbols: List[str]) -> None:
        """Start Alpaca market data streaming."""
        if not self.config.get('providers', {}).get('alpaca', {}).get('enabled', False):
            return
        
        logger.info("Starting Alpaca market data streaming...")
        
        alpaca_config = self.config['providers']['alpaca']
        ws_url = alpaca_config.get('ws_url', 'wss://stream.data.alpaca.markets/v2/iex')
        
        try:
            # Create WebSocket connection
            websocket = await websockets.connect(
                ws_url,
                extra_headers={
                    'Authorization': f"Bearer {alpaca_config.get('api_key', '')}"
                }
            )
            
            with self.connection_lock:
                self.websocket_connections['alpaca'] = websocket
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "key": alpaca_config.get('api_key', ''),
                "secret": alpaca_config.get('api_secret', '')
            }
            await websocket.send(json.dumps(auth_message))
            
            # Subscribe to symbols
            subscribe_message = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Start listening task
            listen_task = asyncio.create_task(self._listen_alpaca_stream(websocket))
            self.processing_tasks.append(listen_task)
            
            logger.info(f"Alpaca streaming started for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting Alpaca streaming: {e}")
            raise
    
    async def _listen_alpaca_stream(self, websocket) -> None:
        """Listen to Alpaca WebSocket stream."""
        try:
            async for message in websocket:
                if not self.state.is_running:
                    break
                
                try:
                    data = json.loads(message)
                    await self._process_alpaca_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from Alpaca: {message}")
                except Exception as e:
                    logger.error(f"Error processing Alpaca message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Alpaca WebSocket connection closed")
            if self.state.is_running:
                await self._reconnect_alpaca()
        except Exception as e:
            logger.error(f"Error in Alpaca stream listener: {e}")
    
    async def _process_alpaca_message(self, data: Dict[str, Any]) -> None:
        """Process Alpaca WebSocket message."""
        if isinstance(data, list):
            for item in data:
                await self._process_alpaca_item(item)
        else:
            await self._process_alpaca_item(data)
    
    async def _process_alpaca_item(self, item: Dict[str, Any]) -> None:
        """Process individual Alpaca data item."""
        msg_type = item.get('T')
        
        if msg_type == 't':  # Trade
            symbol = item.get('S')
            price = item.get('p')
            volume = item.get('s')
            timestamp = datetime.fromisoformat(item.get('t', '').replace('Z', '+00:00'))
            
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                volume=volume
            )
            
            await self._queue_data_point(data_point)
            
        elif msg_type == 'q':  # Quote
            symbol = item.get('S')
            bid = item.get('bp')
            ask = item.get('ap')
            bid_size = item.get('bs')
            ask_size = item.get('as')
            timestamp = datetime.fromisoformat(item.get('t', '').replace('Z', '+00:00'))
            
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=timestamp,
                price=(bid + ask) / 2 if bid and ask else None,
                volume=0,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size
            )
            
            await self._queue_data_point(data_point)
    
    async def _start_polygon_streaming(self, symbols: List[str]) -> None:
        """Start Polygon market data streaming."""
        if not self.config.get('providers', {}).get('polygon', {}).get('enabled', False):
            return
        
        logger.info("Starting Polygon market data streaming...")
        
        polygon_config = self.config['providers']['polygon']
        ws_url = polygon_config.get('ws_url', 'wss://socket.polygon.io/stocks')
        
        try:
            # Create WebSocket connection
            websocket = await websockets.connect(ws_url)
            
            with self.connection_lock:
                self.websocket_connections['polygon'] = websocket
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": polygon_config.get('api_key', '')
            }
            await websocket.send(json.dumps(auth_message))
            
            # Subscribe to symbols
            for symbol in symbols:
                subscribe_message = {
                    "action": "subscribe",
                    "params": f"T.{symbol},Q.{symbol}"
                }
                await websocket.send(json.dumps(subscribe_message))
            
            # Start listening task
            listen_task = asyncio.create_task(self._listen_polygon_stream(websocket))
            self.processing_tasks.append(listen_task)
            
            logger.info(f"Polygon streaming started for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting Polygon streaming: {e}")
            raise
    
    async def _listen_polygon_stream(self, websocket) -> None:
        """Listen to Polygon WebSocket stream."""
        try:
            async for message in websocket:
                if not self.state.is_running:
                    break
                
                try:
                    data = json.loads(message)
                    await self._process_polygon_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from Polygon: {message}")
                except Exception as e:
                    logger.error(f"Error processing Polygon message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Polygon WebSocket connection closed")
            if self.state.is_running:
                await self._reconnect_polygon()
        except Exception as e:
            logger.error(f"Error in Polygon stream listener: {e}")
    
    async def _process_polygon_message(self, data: List[Dict[str, Any]]) -> None:
        """Process Polygon WebSocket message."""
        for item in data:
            ev = item.get('ev')  # Event type
            
            if ev == 'T':  # Trade
                symbol = item.get('sym')
                price = item.get('p')
                volume = item.get('s')
                timestamp = datetime.fromtimestamp(item.get('t', 0) / 1000)
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=price,
                    volume=volume
                )
                
                await self._queue_data_point(data_point)
                
            elif ev == 'Q':  # Quote
                symbol = item.get('sym')
                bid = item.get('bp')
                ask = item.get('ap')
                bid_size = item.get('bs')
                ask_size = item.get('as')
                timestamp = datetime.fromtimestamp(item.get('t', 0) / 1000)
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=(bid + ask) / 2 if bid and ask else None,
                    volume=0,
                    bid=bid,
                    ask=ask,
                    bid_size=bid_size,
                    ask_size=ask_size
                )
                
                await self._queue_data_point(data_point)
    
    async def _start_yahoo_streaming(self, symbols: List[str]) -> None:
        """Start Yahoo Finance streaming (polling-based)."""
        if not self.config.get('providers', {}).get('yahoo', {}).get('enabled', False):
            return
        
        logger.info("Starting Yahoo Finance streaming...")
        
        # Yahoo doesn't have WebSocket, so we'll use polling
        polling_task = asyncio.create_task(self._poll_yahoo_data(symbols))
        self.processing_tasks.append(polling_task)
        
        logger.info(f"Yahoo streaming started for {len(symbols)} symbols")
    
    async def _poll_yahoo_data(self, symbols: List[str]) -> None:
        """Poll Yahoo Finance for market data."""
        yahoo_config = self.config['providers']['yahoo']
        poll_interval = yahoo_config.get('poll_interval', 5)  # seconds
        
        async with aiohttp.ClientSession() as session:
            while self.state.is_running:
                try:
                    for symbol in symbols:
                        if not self.state.is_running:
                            break
                        
                        # Rate limiting
                        if not self._check_rate_limit('yahoo', symbol):
                            continue
                        
                        # Fetch data
                        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                        params = {
                            'interval': '1m',
                            'range': '1d',
                            'includePrePost': 'true'
                        }
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                await self._process_yahoo_data(symbol, data)
                            else:
                                logger.warning(f"Yahoo API error for {symbol}: {response.status}")
                    
                    await asyncio.sleep(poll_interval)
                    
                except Exception as e:
                    logger.error(f"Error polling Yahoo data: {e}")
                    await asyncio.sleep(poll_interval * 2)  # Backoff on error
    
    async def _process_yahoo_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Process Yahoo Finance data."""
        try:
            result = data.get('chart', {}).get('result', [])
            if not result:
                return
            
            chart_data = result[0]
            timestamps = chart_data.get('timestamp', [])
            indicators = chart_data.get('indicators', {})
            quote = indicators.get('quote', [{}])[0]
            
            if not timestamps:
                return
            
            # Get latest data point
            latest_timestamp = timestamps[-1]
            latest_price = quote.get('close', [])[-1] if quote.get('close') else None
            latest_volume = quote.get('volume', [])[-1] if quote.get('volume') else None
            
            if latest_price is not None:
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(latest_timestamp),
                    price=latest_price,
                    volume=latest_volume or 0,
                    high=quote.get('high', [])[-1] if quote.get('high') else None,
                    low=quote.get('low', [])[-1] if quote.get('low') else None,
                    open=quote.get('open', [])[-1] if quote.get('open') else None
                )
                
                await self._queue_data_point(data_point)
                
        except Exception as e:
            logger.error(f"Error processing Yahoo data for {symbol}: {e}")
    
    def _check_rate_limit(self, provider: str, symbol: str) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        key = f"{provider}:{symbol}"
        
        if key not in self.rate_limiter:
            self.rate_limiter[key] = []
        
        # Clean old timestamps
        self.rate_limiter[key] = [
            ts for ts in self.rate_limiter[key] 
            if now - ts < 1.0  # Within last second
        ]
        
        # Check if we can make another request
        if len(self.rate_limiter[key]) < self.max_requests_per_second:
            self.rate_limiter[key].append(now)
            return True
        
        return False
    
    async def _queue_data_point(self, data_point: MarketDataPoint) -> None:
        """Queue data point for processing."""
        try:
            # Update cache
            self.price_cache[data_point.symbol] = data_point.price
            self.volume_cache[data_point.symbol] = data_point.volume
            self.state.last_update[data_point.symbol] = data_point.timestamp
            
            # Queue for processing
            if not self.data_queue.full():
                self.data_queue.put_nowait(data_point)
                self.state.data_points_received += 1
            else:
                logger.warning("Data queue is full, dropping data point")
            
            # Notify callbacks
            for callback in self.state.callbacks:
                try:
                    await callback(data_point)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error queuing data point: {e}")
    
    async def _process_data_queue(self) -> None:
        """Process queued data points."""
        while self.state.is_running:
            try:
                # Process batch of data points
                batch = []
                batch_size = 100
                
                for _ in range(batch_size):
                    try:
                        data_point = self.data_queue.get_nowait()
                        batch.append(data_point)
                    except queue.Empty:
                        break
                
                if batch:
                    await self._process_data_batch(batch)
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                logger.error(f"Error processing data queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_data_batch(self, batch: List[MarketDataPoint]) -> None:
        """Process a batch of data points."""
        try:
            # Save to database
            await self.db_manager.save_market_data_batch(batch)
            
            # Update metrics
            self.metrics_collector.record_market_data_batch(len(batch))
            
            # Calculate indicators for unique symbols
            symbols = list(set(point.symbol for point in batch))
            for symbol in symbols:
                await self._update_indicators(symbol)
                
        except Exception as e:
            logger.error(f"Error processing data batch: {e}")
    
    async def _update_indicators(self, symbol: str) -> None:
        """Update technical indicators for a symbol."""
        try:
            # Get recent price data
            recent_data = await self.db_manager.get_recent_market_data(symbol, hours=24)
            
            if len(recent_data) < 20:  # Need minimum data for indicators
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(recent_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Calculate indicators
            indicators = {}
            
            # Simple Moving Average
            indicators['sma_20'] = df['price'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = df['price'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # Exponential Moving Average
            indicators['ema_12'] = df['price'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['price'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            macd_line = indicators['ema_12'] - indicators['ema_26']
            signal_line = pd.Series([macd_line]).ewm(span=9).mean().iloc[0]
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = macd_line - signal_line
            
            # Bollinger Bands
            sma_20 = df['price'].rolling(window=20).mean()
            std_20 = df['price'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
            # Store indicators
            self.indicators_cache[symbol] = {
                'timestamp': datetime.now(),
                'indicators': indicators
            }
            
            # Save to database
            await self.db_manager.save_indicators(symbol, indicators)
            
        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}")
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor streaming health and reconnect if needed."""
        while self.state.is_running:
            try:
                now = datetime.now()
                
                # Check for stale data
                for symbol, last_update in self.state.last_update.items():
                    if now - last_update > timedelta(minutes=5):
                        logger.warning(f"Stale data for {symbol}, last update: {last_update}")
                
                # Check connection health
                await self._check_connection_health()
                
                # Log statistics
                logger.info(f"Streaming stats: {self.state.data_points_received} data points, "
                           f"{len(self.state.connected_symbols)} symbols, "
                           f"{self.state.error_count} errors")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(60)
    
    async def _check_connection_health(self) -> None:
        """Check health of WebSocket connections."""
        with self.connection_lock:
            for provider, connection in self.websocket_connections.items():
                if connection and connection.closed:
                    logger.warning(f"{provider} connection is closed, attempting reconnect")
                    if provider == 'alpaca':
                        await self._reconnect_alpaca()
                    elif provider == 'polygon':
                        await self._reconnect_polygon()
    
    async def _reconnect_alpaca(self) -> None:
        """Reconnect to Alpaca stream."""
        try:
            self.state.reconnect_count += 1
            logger.info(f"Reconnecting to Alpaca (attempt {self.state.reconnect_count})")
            
            # Wait before reconnecting
            await asyncio.sleep(5)
            
            # Get symbols from current state
            symbols = list(self.state.connected_symbols)
            if symbols:
                await self._start_alpaca_streaming(symbols)
                
        except Exception as e:
            logger.error(f"Error reconnecting to Alpaca: {e}")
    
    async def _reconnect_polygon(self) -> None:
        """Reconnect to Polygon stream."""
        try:
            self.state.reconnect_count += 1
            logger.info(f"Reconnecting to Polygon (attempt {self.state.reconnect_count})")
            
            # Wait before reconnecting
            await asyncio.sleep(5)
            
            # Get symbols from current state
            symbols = list(self.state.connected_symbols)
            if symbols:
                await self._start_polygon_streaming(symbols)
                
        except Exception as e:
            logger.error(f"Error reconnecting to Polygon: {e}")
    
    def add_callback(self, callback: Callable[[MarketDataPoint], None]) -> None:
        """Add callback for real-time data updates."""
        self.state.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MarketDataPoint], None]) -> None:
        """Remove callback."""
        if callback in self.state.callbacks:
            self.state.callbacks.remove(callback)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        return self.price_cache.get(symbol)
    
    def get_current_volume(self, symbol: str) -> Optional[int]:
        """Get current volume for a symbol."""
        return self.volume_cache.get(symbol)
    
    def get_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current indicators for a symbol."""
        indicator_data = self.indicators_cache.get(symbol)
        return indicator_data['indicators'] if indicator_data else None
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            'is_running': self.state.is_running,
            'connected_symbols': len(self.state.connected_symbols),
            'data_points_received': self.state.data_points_received,
            'error_count': self.state.error_count,
            'reconnect_count': self.state.reconnect_count,
            'queue_size': self.data_queue.qsize(),
            'last_updates': {
                symbol: timestamp.isoformat() 
                for symbol, timestamp in self.state.last_update.items()
            }
        } 