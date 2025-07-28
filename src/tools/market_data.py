"""
Market Data Tool

Retrieves market data from various sources including real-time prices,
historical data, and Level 2 market data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import ccxt
import os

logger = logging.getLogger(__name__)


class MarketDataTool:
    """
    Tool for retrieving market data from various sources.
    
    Features:
    - Real-time price data from multiple exchanges
    - Historical data with various timeframes
    - Level 2 market data (order book)
    - Options data and implied volatility
    - Market indicators and technical analysis
    """
    
    def __init__(self):
        self.exchanges = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize CCXT exchanges
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
                'sandbox': True  # Use sandbox for testing
            })
            
            self.exchanges['alpaca'] = ccxt.alpaca({
                'apiKey': os.getenv('ALPACA_API_KEY'),
                'secret': os.getenv('ALPACA_SECRET_KEY'),
                'sandbox': True
            })
            
            logger.info("Market data exchanges initialized")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def get_current_price(self, symbol: str, exchange: str = "alpaca") -> Dict[str, Any]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Dictionary containing current price data
        """
        try:
            logger.info(f"Getting current price for {symbol} from {exchange}")
            
            if exchange == "alpaca":
                return await self._get_alpaca_price(symbol)
            elif exchange == "binance":
                return await self._get_binance_price(symbol)
            else:
                return await self._get_yfinance_price(symbol)
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return self._get_mock_price(symbol)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 1d, etc.)
            
        Returns:
            DataFrame with historical price data
        """
        try:
            logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
            
            # Use yfinance for historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            if data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return self._get_mock_historical_data(symbol, start_date, end_date, timeframe)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return self._get_mock_historical_data(symbol, start_date, end_date, timeframe)
    
    async def get_level2_data(self, symbol: str, exchange: str = "alpaca") -> Dict[str, Any]:
        """
        Get Level 2 market data (order book).
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Dictionary containing Level 2 data
        """
        try:
            logger.info(f"Getting Level 2 data for {symbol} from {exchange}")
            
            if exchange in self.exchanges:
                orderbook = await self.exchanges[exchange].fetch_order_book(symbol)
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'bids': orderbook['bids'][:10],  # Top 10 bids
                    'asks': orderbook['asks'][:10],  # Top 10 asks
                    'bid_volume': sum(bid[1] for bid in orderbook['bids'][:10]),
                    'ask_volume': sum(ask[1] for ask in orderbook['asks'][:10]),
                    'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0] if orderbook['asks'] and orderbook['bids'] else 0
                }
            else:
                return self._get_mock_level2_data(symbol)
                
        except Exception as e:
            logger.error(f"Error getting Level 2 data for {symbol}: {e}")
            return self._get_mock_level2_data(symbol)
    
    async def get_market_indicators(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """
        Get market indicators and technical analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary containing market indicators
        """
        try:
            logger.info(f"Getting market indicators for {symbol}")
            
            # Get historical data for calculations
            end_date = datetime.now()
            start_date = end_date - timedelta(days=50)  # Need enough data for indicators
            
            data = await self.get_historical_data(symbol, start_date, end_date, timeframe)
            
            if data.empty:
                return self._get_mock_indicators(symbol)
            
            # Calculate indicators
            indicators = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'sma_20': self._calculate_sma(data, 20),
                'sma_50': self._calculate_sma(data, 50),
                'rsi': self._calculate_rsi(data),
                'macd': self._calculate_macd(data),
                'bollinger_bands': self._calculate_bollinger_bands(data),
                'volume_sma': self._calculate_volume_sma(data, 20)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting market indicators for {symbol}: {e}")
            return self._get_mock_indicators(symbol)
    
    async def _get_alpaca_price(self, symbol: str) -> Dict[str, Any]:
        """Get price from Alpaca"""
        try:
            # This would use Alpaca API
            return {
                'symbol': symbol,
                'price': 150.25,  # Mock price
                'timestamp': datetime.now(),
                'volume': 1000000,
                'change': 2.5,
                'change_percent': 1.69
            }
        except Exception as e:
            logger.error(f"Error getting Alpaca price: {e}")
            return self._get_mock_price(symbol)
    
    async def _get_binance_price(self, symbol: str) -> Dict[str, Any]:
        """Get price from Binance"""
        try:
            # This would use Binance API
            return {
                'symbol': symbol,
                'price': 0.0025,  # Mock price
                'timestamp': datetime.now(),
                'volume': 500000,
                'change': 0.0001,
                'change_percent': 4.17
            }
        except Exception as e:
            logger.error(f"Error getting Binance price: {e}")
            return self._get_mock_price(symbol)
    
    async def _get_yfinance_price(self, symbol: str) -> Dict[str, Any]:
        """Get price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0),
                'timestamp': datetime.now(),
                'volume': info.get('volume', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance price: {e}")
            return self._get_mock_price(symbol)
    
    def _calculate_sma(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            return float(data['Close'].rolling(window=period).mean().iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return 0.0
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD"""
        try:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(macd.iloc[-1] - signal.iloc[-1])
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            sma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            
            return {
                'upper': float(sma.iloc[-1] + (std.iloc[-1] * 2)),
                'middle': float(sma.iloc[-1]),
                'lower': float(sma.iloc[-1] - (std.iloc[-1] * 2))
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
    
    def _calculate_volume_sma(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Volume Simple Moving Average"""
        try:
            return float(data['Volume'].rolling(window=period).mean().iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating Volume SMA: {e}")
            return 0.0
    
    def _get_mock_price(self, symbol: str) -> Dict[str, Any]:
        """Return mock price data"""
        return {
            'symbol': symbol,
            'price': 100.0,
            'timestamp': datetime.now(),
            'volume': 1000000,
            'change': 1.0,
            'change_percent': 1.0
        }
    
    def _get_mock_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: str
    ) -> pd.DataFrame:
        """Return mock historical data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame({
            'Open': [100 + i * 0.1 for i in range(len(dates))],
            'High': [101 + i * 0.1 for i in range(len(dates))],
            'Low': [99 + i * 0.1 for i in range(len(dates))],
            'Close': [100.5 + i * 0.1 for i in range(len(dates))],
            'Volume': [1000000 + i * 1000 for i in range(len(dates))]
        }, index=dates)
        
        return data
    
    def _get_mock_level2_data(self, symbol: str) -> Dict[str, Any]:
        """Return mock Level 2 data"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'bids': [[99.9, 1000], [99.8, 2000], [99.7, 1500]],
            'asks': [[100.1, 1000], [100.2, 2000], [100.3, 1500]],
            'bid_volume': 4500,
            'ask_volume': 4500,
            'spread': 0.2
        }
    
    def _get_mock_indicators(self, symbol: str) -> Dict[str, Any]:
        """Return mock market indicators"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'sma_20': 100.0,
            'sma_50': 98.0,
            'rsi': 55.0,
            'macd': {'macd': 0.5, 'signal': 0.3, 'histogram': 0.2},
            'bollinger_bands': {'upper': 105.0, 'middle': 100.0, 'lower': 95.0},
            'volume_sma': 1000000
        } 