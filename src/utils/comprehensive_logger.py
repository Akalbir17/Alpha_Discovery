#!/usr/bin/env python3
"""
Comprehensive Logging System

Advanced logging system for the Alpha Discovery platform that tracks discoveries,
trades, performance metrics, and system events with structured logging and monitoring.
"""

import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import asyncio
from pathlib import Path
import gzip
import shutil
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback

from configs.config_loader import get_config
from src.utils.database import DatabaseManager

class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """Log categories for classification."""
    SYSTEM = "system"
    DISCOVERY = "discovery"
    TRADE = "trade"
    PERFORMANCE = "performance"
    RISK = "risk"
    MARKET_DATA = "market_data"
    SENTIMENT = "sentiment"
    ERROR = "error"
    AUDIT = "audit"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    data: Dict[str, Any]
    source: str
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'message': self.message,
            'data': self.data,
            'source': self.source,
            'correlation_id': self.correlation_id,
            'session_id': self.session_id,
            'user_id': self.user_id
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

class DiscoveryLogger:
    """Specialized logger for alpha discoveries."""
    
    def __init__(self, comprehensive_logger):
        self.logger = comprehensive_logger
        self.discovery_count = 0
        self.discovery_stats = {}
    
    def log_discovery_created(self, discovery: Dict[str, Any]) -> None:
        """Log discovery creation."""
        self.discovery_count += 1
        
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.DISCOVERY,
            message=f"Discovery created: {discovery['symbol']} - {discovery['discovery_type']}",
            data={
                'discovery_id': discovery['id'],
                'symbol': discovery['symbol'],
                'discovery_type': discovery['discovery_type'],
                'agent_source': discovery['agent_source'],
                'confidence': discovery['confidence'],
                'signal_strength': discovery['signal_strength'],
                'direction': discovery['direction'],
                'reasoning': discovery['reasoning'],
                'metadata': discovery.get('metadata', {})
            },
            source="discovery_engine",
            correlation_id=discovery['id']
        )
    
    def log_discovery_validated(self, discovery: Dict[str, Any], validation_score: float) -> None:
        """Log discovery validation."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.DISCOVERY,
            message=f"Discovery validated: {discovery['symbol']} - Score: {validation_score:.2f}",
            data={
                'discovery_id': discovery['id'],
                'symbol': discovery['symbol'],
                'validation_score': validation_score,
                'risk_score': discovery.get('risk_score', 0.0),
                'status': discovery['status']
            },
            source="validation_engine",
            correlation_id=discovery['id']
        )
    
    def log_discovery_executed(self, discovery: Dict[str, Any], execution_details: Dict[str, Any]) -> None:
        """Log discovery execution."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.DISCOVERY,
            message=f"Discovery executed: {discovery['symbol']} - {execution_details['order_type']}",
            data={
                'discovery_id': discovery['id'],
                'symbol': discovery['symbol'],
                'execution_price': execution_details['price'],
                'position_size': execution_details['quantity'],
                'order_id': execution_details['order_id'],
                'execution_time': execution_details['timestamp'],
                'fees': execution_details.get('fees', 0.0)
            },
            source="trading_engine",
            correlation_id=discovery['id']
        )
    
    def log_discovery_completed(self, discovery: Dict[str, Any], exit_reason: str, pnl: float) -> None:
        """Log discovery completion."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.DISCOVERY,
            message=f"Discovery completed: {discovery['symbol']} - {exit_reason} - P&L: {pnl:.2f}",
            data={
                'discovery_id': discovery['id'],
                'symbol': discovery['symbol'],
                'exit_reason': exit_reason,
                'closed_pnl': pnl,
                'max_profit': discovery.get('max_profit', 0.0),
                'max_loss': discovery.get('max_loss', 0.0),
                'duration_seconds': (datetime.now() - discovery['timestamp']).total_seconds()
            },
            source="discovery_engine",
            correlation_id=discovery['id']
        )

class TradeLogger:
    """Specialized logger for trades."""
    
    def __init__(self, comprehensive_logger):
        self.logger = comprehensive_logger
        self.trade_count = 0
        self.trade_volume = 0.0
    
    def log_trade_order(self, order: Dict[str, Any]) -> None:
        """Log trade order placement."""
        self.trade_count += 1
        
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.TRADE,
            message=f"Trade order placed: {order['symbol']} {order['side']} {order['quantity']}",
            data={
                'order_id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': order['quantity'],
                'order_type': order['order_type'],
                'price': order.get('price'),
                'stop_price': order.get('stop_price'),
                'time_in_force': order.get('time_in_force'),
                'status': order['status']
            },
            source="trading_engine",
            correlation_id=order['id']
        )
    
    def log_trade_execution(self, execution: Dict[str, Any]) -> None:
        """Log trade execution."""
        self.trade_volume += execution['quantity'] * execution['price']
        
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.TRADE,
            message=f"Trade executed: {execution['symbol']} {execution['side']} {execution['quantity']} @ {execution['price']}",
            data={
                'execution_id': execution['id'],
                'order_id': execution['order_id'],
                'symbol': execution['symbol'],
                'side': execution['side'],
                'quantity': execution['quantity'],
                'price': execution['price'],
                'fees': execution.get('fees', 0.0),
                'execution_time': execution['timestamp']
            },
            source="trading_engine",
            correlation_id=execution['order_id']
        )
    
    def log_trade_rejection(self, order: Dict[str, Any], reason: str) -> None:
        """Log trade rejection."""
        self.logger.log(
            level=LogLevel.WARNING,
            category=LogCategory.TRADE,
            message=f"Trade rejected: {order['symbol']} - {reason}",
            data={
                'order_id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': order['quantity'],
                'rejection_reason': reason,
                'order_type': order['order_type']
            },
            source="trading_engine",
            correlation_id=order['id']
        )

class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, comprehensive_logger):
        self.logger = comprehensive_logger
        self.performance_history = []
    
    def log_portfolio_snapshot(self, portfolio: Dict[str, Any]) -> None:
        """Log portfolio snapshot."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.PERFORMANCE,
            message=f"Portfolio snapshot - Value: ${portfolio['total_value']:,.2f}",
            data={
                'total_value': portfolio['total_value'],
                'cash_balance': portfolio['cash_balance'],
                'positions_value': portfolio['positions_value'],
                'daily_pnl': portfolio['daily_pnl'],
                'total_pnl': portfolio['total_pnl'],
                'active_positions': portfolio['active_positions'],
                'position_count': len(portfolio.get('positions', []))
            },
            source="portfolio_manager"
        )
    
    def log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.performance_history.append(metrics)
        
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.PERFORMANCE,
            message=f"Performance metrics - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
            data={
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': metrics.get('sortino_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'profit_factor': metrics.get('profit_factor', 0.0),
                'volatility': metrics.get('volatility', 0.0),
                'beta': metrics.get('beta', 0.0),
                'alpha': metrics.get('alpha', 0.0)
            },
            source="performance_tracker"
        )
    
    def log_risk_metrics(self, risk_metrics: Dict[str, Any]) -> None:
        """Log risk metrics."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.RISK,
            message=f"Risk metrics - VaR: {risk_metrics.get('var_95', 0):.2f}",
            data={
                'var_95': risk_metrics.get('var_95', 0.0),
                'var_99': risk_metrics.get('var_99', 0.0),
                'expected_shortfall': risk_metrics.get('expected_shortfall', 0.0),
                'portfolio_beta': risk_metrics.get('portfolio_beta', 0.0),
                'concentration_risk': risk_metrics.get('concentration_risk', 0.0),
                'leverage_ratio': risk_metrics.get('leverage_ratio', 0.0)
            },
            source="risk_manager"
        )

class SystemLogger:
    """Specialized logger for system events."""
    
    def __init__(self, comprehensive_logger):
        self.logger = comprehensive_logger
        self.system_events = []
    
    def log_system_startup(self, components: List[str]) -> None:
        """Log system startup."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message="Alpha Discovery system started",
            data={
                'components': components,
                'startup_time': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            source="orchestrator"
        )
    
    def log_system_shutdown(self, reason: str) -> None:
        """Log system shutdown."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message=f"Alpha Discovery system shutdown: {reason}",
            data={
                'shutdown_reason': reason,
                'shutdown_time': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.system_events[0]['timestamp']).total_seconds() if self.system_events else 0
            },
            source="orchestrator"
        )
    
    def log_component_status(self, component: str, status: str, details: Dict[str, Any] = None) -> None:
        """Log component status."""
        self.logger.log(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message=f"Component {component}: {status}",
            data={
                'component': component,
                'status': status,
                'details': details or {}
            },
            source=component
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log system error."""
        self.logger.log(
            level=LogLevel.ERROR,
            category=LogCategory.ERROR,
            message=f"System error: {str(error)}",
            data={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {}
            },
            source="system"
        )

class ComprehensiveLogger:
    """Main comprehensive logging system."""
    
    def __init__(self):
        self.config = get_config('logging')
        self.db_manager = DatabaseManager(self.config)
        
        # Initialize logging directory
        self.log_dir = Path(self.config.get('log_directory', '/var/log/alpha-discovery'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self.setup_loggers()
        
        # Specialized loggers
        self.discovery_logger = DiscoveryLogger(self)
        self.trade_logger = TradeLogger(self)
        self.performance_logger = PerformanceLogger(self)
        self.system_logger = SystemLogger(self)
        
        # Async processing
        self.log_queue = queue.Queue(maxsize=10000)
        self.processing_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.processing_thread.start()
        
        # Session tracking
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Statistics
        self.log_stats = {
            'total_logs': 0,
            'logs_by_level': {level.value: 0 for level in LogLevel},
            'logs_by_category': {category.value: 0 for category in LogCategory}
        }
        
        logger = logging.getLogger(__name__)
        logger.info("Comprehensive logging system initialized")
    
    def setup_loggers(self) -> None:
        """Setup Python logging configuration."""
        # Main application logger
        self.app_logger = logging.getLogger('alpha_discovery')
        self.app_logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.app_logger.addHandler(console_handler)
        
        # File handlers
        self.setup_file_handlers()
        
        # Database handler (custom)
        self.setup_database_handler()
    
    def setup_file_handlers(self) -> None:
        """Setup file logging handlers."""
        # Main log file
        main_handler = RotatingFileHandler(
            self.log_dir / 'alpha_discovery.log',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        main_handler.setLevel(logging.DEBUG)
        main_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        main_handler.setFormatter(main_formatter)
        self.app_logger.addHandler(main_handler)
        
        # Error log file
        error_handler = RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_formatter)
        self.app_logger.addHandler(error_handler)
        
        # Trade log file
        trade_handler = TimedRotatingFileHandler(
            self.log_dir / 'trades.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        trade_handler.setLevel(logging.INFO)
        trade_formatter = logging.Formatter(
            '%(asctime)s - TRADE - %(message)s'
        )
        trade_handler.setFormatter(trade_formatter)
        
        # Discovery log file
        discovery_handler = TimedRotatingFileHandler(
            self.log_dir / 'discoveries.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        discovery_handler.setLevel(logging.INFO)
        discovery_formatter = logging.Formatter(
            '%(asctime)s - DISCOVERY - %(message)s'
        )
        discovery_handler.setFormatter(discovery_formatter)
        
        # Performance log file
        performance_handler = TimedRotatingFileHandler(
            self.log_dir / 'performance.log',
            when='midnight',
            interval=1,
            backupCount=90
        )
        performance_handler.setLevel(logging.INFO)
        performance_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        performance_handler.setFormatter(performance_formatter)
    
    def setup_database_handler(self) -> None:
        """Setup database logging handler."""
        # Custom database handler for structured logs
        pass
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            data: Dict[str, Any] = None, source: str = "unknown",
            correlation_id: str = None, user_id: str = None) -> None:
        """Log a structured message."""
        try:
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                data=data or {},
                source=source,
                correlation_id=correlation_id,
                session_id=self.session_id,
                user_id=user_id
            )
            
            # Queue for async processing
            if not self.log_queue.full():
                self.log_queue.put_nowait(log_entry)
            else:
                # If queue is full, log directly to prevent blocking
                self._write_log_entry(log_entry)
            
            # Update statistics
            self.log_stats['total_logs'] += 1
            self.log_stats['logs_by_level'][level.value] += 1
            self.log_stats['logs_by_category'][category.value] += 1
            
        except Exception as e:
            # Fallback logging
            self.app_logger.error(f"Error in comprehensive logging: {e}")
    
    def _process_logs(self) -> None:
        """Process log entries asynchronously."""
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1.0)
                self._write_log_entry(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.app_logger.error(f"Error processing log entry: {e}")
    
    def _write_log_entry(self, log_entry: LogEntry) -> None:
        """Write log entry to all outputs."""
        try:
            # Write to Python logger
            python_level = getattr(logging, log_entry.level.value)
            self.app_logger.log(
                python_level,
                f"[{log_entry.category.value}] {log_entry.message} - {json.dumps(log_entry.data, default=str)}"
            )
            
            # Write to structured log file
            structured_log_file = self.log_dir / f"{log_entry.category.value}.jsonl"
            with open(structured_log_file, 'a') as f:
                f.write(log_entry.to_json() + '\n')
            
            # Write to database (async)
            asyncio.create_task(self._save_to_database(log_entry))
            
        except Exception as e:
            self.app_logger.error(f"Error writing log entry: {e}")
    
    async def _save_to_database(self, log_entry: LogEntry) -> None:
        """Save log entry to database."""
        try:
            await self.db_manager.save_log_entry(log_entry.to_dict())
        except Exception as e:
            self.app_logger.error(f"Error saving log to database: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'total_logs': self.log_stats['total_logs'],
            'logs_by_level': self.log_stats['logs_by_level'],
            'logs_by_category': self.log_stats['logs_by_category'],
            'queue_size': self.log_queue.qsize(),
            'session_id': self.session_id
        }
    
    async def query_logs(self, filters: Dict[str, Any] = None, 
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Query logs from database."""
        try:
            return await self.db_manager.query_logs(filters, limit)
        except Exception as e:
            self.app_logger.error(f"Error querying logs: {e}")
            return []
    
    def compress_old_logs(self, days_old: int = 7) -> None:
        """Compress old log files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for log_file in self.log_dir.glob('*.log.*'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()
                    self.app_logger.info(f"Compressed old log file: {log_file}")
            
        except Exception as e:
            self.app_logger.error(f"Error compressing old logs: {e}")
    
    def cleanup_old_logs(self, days_old: int = 30) -> None:
        """Clean up very old log files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for log_file in self.log_dir.glob('*.gz'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.app_logger.info(f"Deleted old log file: {log_file}")
            
        except Exception as e:
            self.app_logger.error(f"Error cleaning up old logs: {e}")
    
    def shutdown(self) -> None:
        """Shutdown logging system."""
        try:
            # Process remaining logs
            while not self.log_queue.empty():
                try:
                    log_entry = self.log_queue.get_nowait()
                    self._write_log_entry(log_entry)
                except queue.Empty:
                    break
            
            # Wait for processing thread
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            self.app_logger.info("Comprehensive logging system shutdown")
            
        except Exception as e:
            self.app_logger.error(f"Error shutting down logging system: {e}")

# Global logger instance
_comprehensive_logger = None

def get_comprehensive_logger() -> ComprehensiveLogger:
    """Get global comprehensive logger instance."""
    global _comprehensive_logger
    if _comprehensive_logger is None:
        _comprehensive_logger = ComprehensiveLogger()
    return _comprehensive_logger

def setup_comprehensive_logging() -> ComprehensiveLogger:
    """Setup comprehensive logging system."""
    return get_comprehensive_logger() 