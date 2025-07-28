"""
Database Manager
Handles database connections and operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import sqlite3
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for handling database operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Handle both direct config and nested database config
        if config and 'path' in config:
            self.db_path = config.get('path', 'alpha_discovery.db')
        elif config and 'database' in config:
            self.db_path = config['database'].get('path', 'alpha_discovery.db')
        else:
            self.db_path = 'alpha_discovery.db'
        self.connection = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to database"""
        try:
            logger.info(f"Connecting to database: {self.db_path}")
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.is_connected = True
            await self._create_tables()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from database"""
        try:
            if self.connection:
                self.connection.close()
                self.is_connected = False
                logger.info("Database disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")
            
    async def _create_tables(self):
        """Create database tables"""
        try:
            cursor = self.connection.cursor()
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER,
                    timestamp DATETIME NOT NULL,
                    data TEXT
                )
            ''')
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    order_id TEXT,
                    metadata TEXT
                )
            ''')
            
            # Alpha discoveries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alpha_discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    alpha_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    factors TEXT,
                    timestamp DATETIME NOT NULL,
                    agent_id TEXT
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT
                )
            ''')
            
            self.connection.commit()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
            
    async def insert_market_data(self, symbol: str, price: float, volume: int, 
                                timestamp: datetime, data: Dict[str, Any] = None):
        """Insert market data"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO market_data (symbol, price, volume, timestamp, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, price, volume, timestamp, json.dumps(data) if data else None))
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error inserting market data: {e}")
            raise
            
    async def insert_trade(self, symbol: str, side: str, quantity: float, price: float,
                          timestamp: datetime, order_id: str = None, metadata: Dict[str, Any] = None):
        """Insert trade record"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, side, quantity, price, timestamp, order_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, side, quantity, price, timestamp, order_id, 
                  json.dumps(metadata) if metadata else None))
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error inserting trade: {e}")
            raise
            
    async def insert_alpha_discovery(self, symbol: str, alpha_score: float, confidence: float,
                                   factors: Dict[str, Any], timestamp: datetime, agent_id: str = None):
        """Insert alpha discovery"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO alpha_discoveries (symbol, alpha_score, confidence, factors, timestamp, agent_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, alpha_score, confidence, json.dumps(factors), timestamp, agent_id))
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error inserting alpha discovery: {e}")
            raise
            
    async def insert_performance_metric(self, metric_name: str, metric_value: float,
                                      timestamp: datetime, metadata: Dict[str, Any] = None):
        """Insert performance metric"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics (metric_name, metric_value, timestamp, metadata)
                VALUES (?, ?, ?, ?)
            ''', (metric_name, metric_value, timestamp, json.dumps(metadata) if metadata else None))
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error inserting performance metric: {e}")
            raise
            
    async def get_market_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get market data for symbol"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (symbol, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return []
            
    async def get_trades(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade records"""
        try:
            cursor = self.connection.cursor()
            if symbol:
                cursor.execute('''
                    SELECT * FROM trades 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
            
    async def get_alpha_discoveries(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alpha discoveries"""
        try:
            cursor = self.connection.cursor()
            if symbol:
                cursor.execute('''
                    SELECT * FROM alpha_discoveries 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM alpha_discoveries 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting alpha discoveries: {e}")
            return []
            
    async def get_performance_metrics(self, metric_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        try:
            cursor = self.connection.cursor()
            if metric_name:
                cursor.execute('''
                    SELECT * FROM performance_metrics 
                    WHERE metric_name = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (metric_name, limit))
            else:
                cursor.execute('''
                    SELECT * FROM performance_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return [] 