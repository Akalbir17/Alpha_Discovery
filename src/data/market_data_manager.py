"""
Market Data Manager
Manages market data streaming and provides unified interface for data access
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from .market_data_streamer import MarketDataStreamer
from .market_feeds import MarketDataPipeline

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Manages market data streaming and provides unified interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.streamer = MarketDataStreamer()
        self.feeds = {}
        self.is_running = False
        
    async def start(self):
        """Start market data streaming"""
        try:
            logger.info("Starting market data manager...")
            # Start with default symbols - can be made configurable
            default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            await self.streamer.start_streaming(default_symbols)
            self.is_running = True
            logger.info("Market data manager started successfully")
        except Exception as e:
            logger.error(f"Failed to start market data manager: {e}")
            raise
            
    async def stop(self):
        """Stop market data streaming"""
        try:
            logger.info("Stopping market data manager...")
            await self.streamer.stop_streaming()
            self.is_running = False
            logger.info("Market data manager stopped")
        except Exception as e:
            logger.error(f"Error stopping market data manager: {e}")
            
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for symbol"""
        price = self.streamer.get_current_price(symbol)
        volume = self.streamer.get_current_volume(symbol)
        indicators = self.streamer.get_indicators(symbol)
        
        if price is None:
            return None
            
        return {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'indicators': indicators,
            'timestamp': datetime.now().isoformat()
        }
        
    async def subscribe_symbol(self, symbol: str):
        """Subscribe to market data for symbol"""
        # This would need to be implemented in the streamer
        # For now, we'll just log it
        logger.info(f"Subscribing to symbol: {symbol}")
        
    async def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from market data for symbol"""
        # This would need to be implemented in the streamer
        # For now, we'll just log it
        logger.info(f"Unsubscribing from symbol: {symbol}")
        
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols"""
        stats = self.streamer.get_streaming_stats()
        return list(stats.get('connected_symbols', set()))
        
    def is_symbol_subscribed(self, symbol: str) -> bool:
        """Check if symbol is subscribed"""
        stats = self.streamer.get_streaming_stats()
        return symbol in stats.get('connected_symbols', set()) 