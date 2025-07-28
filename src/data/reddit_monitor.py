"""
Reddit Monitor
Wrapper for Reddit sentiment monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from .reddit_sentiment_monitor import RedditSentimentMonitor

logger = logging.getLogger(__name__)

class RedditMonitor:
    """Wrapper for Reddit sentiment monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sentiment_monitor = RedditSentimentMonitor()
        self.is_running = False
        
    async def start(self):
        """Start Reddit monitoring"""
        try:
            logger.info("Starting Reddit monitor...")
            await self.sentiment_monitor.start()
            self.is_running = True
            logger.info("Reddit monitor started successfully")
        except Exception as e:
            logger.error(f"Failed to start Reddit monitor: {e}")
            raise
            
    async def stop(self):
        """Stop Reddit monitoring"""
        try:
            logger.info("Stopping Reddit monitor...")
            await self.sentiment_monitor.stop()
            self.is_running = False
            logger.info("Reddit monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping Reddit monitor: {e}")
            
    async def get_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment data for symbol"""
        return await self.sentiment_monitor.get_sentiment(symbol)
        
    async def get_trending_symbols(self) -> List[str]:
        """Get trending symbols from Reddit"""
        return await self.sentiment_monitor.get_trending_symbols()
        
    def get_monitored_symbols(self) -> List[str]:
        """Get list of monitored symbols"""
        return self.sentiment_monitor.get_monitored_symbols() 