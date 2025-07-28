#!/usr/bin/env python3
"""
Performance Reporting System

Comprehensive performance reporting system for the Alpha Discovery platform.
Tracks system metrics, trading performance, discovery effectiveness, and generates
detailed reports for monitoring and optimization.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import base64

from configs.config_loader import get_config
from src.utils.database import DatabaseManager
from src.utils.comprehensive_logger import get_comprehensive_logger
from src.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of performance reports."""
    SYSTEM_HEALTH = "system_health"
    TRADING_PERFORMANCE = "trading_performance"
    DISCOVERY_EFFECTIVENESS = "discovery_effectiveness"
    RISK_ANALYSIS = "risk_analysis"
    PORTFOLIO_SUMMARY = "portfolio_summary"
    AGENT_PERFORMANCE = "agent_performance"
    MARKET_DATA_QUALITY = "market_data_quality"
    COMPREHENSIVE = "comprehensive"

class ReportPeriod(Enum):
    """Report time periods."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    trading_metrics: Dict[str, Any] = field(default_factory=dict)
    discovery_metrics: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    portfolio_metrics: Dict[str, Any] = field(default_factory=dict)
    agent_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Performance report data structure."""
    report_id: str
    report_type: ReportType
    period: ReportPeriod
    start_time: datetime
    end_time: datetime
    generated_at: datetime
    metrics: PerformanceMetrics
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    charts: Dict[str, str] = field(default_factory=dict)  # Base64 encoded charts

class PerformanceReporter:
    """Comprehensive performance reporting system."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.config = get_config('monitoring')
        self.db_manager = DatabaseManager(self.config)
        self.logger = get_comprehensive_logger()
        self.metrics_collector = MetricsCollector(self.config)
        
        # Report configuration
        self.report_dir = Path(self.config.get('report_directory', '/var/log/alpha-discovery/reports'))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Reporting intervals
        self.report_intervals = {
            ReportPeriod.REAL_TIME: 30,      # 30 seconds
            ReportPeriod.HOURLY: 3600,       # 1 hour
            ReportPeriod.DAILY: 86400,       # 24 hours
            ReportPeriod.WEEKLY: 604800,     # 7 days
            ReportPeriod.MONTHLY: 2592000,   # 30 days
        }
        
        # Performance history
        self.performance_history = []
        self.report_history = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.reporting_tasks = []
        self.is_running = False
        
        # Metrics cache
        self.metrics_cache = {}
        self.last_metrics_update = datetime.now()
        
        # Chart styling
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("Performance Reporter initialized")
    
    async def start_reporting(self) -> None:
        """Start performance reporting."""
        logger.info("Starting performance reporting...")
        
        self.is_running = True
        
        # Start reporting tasks for different periods
        for period in [ReportPeriod.REAL_TIME, ReportPeriod.HOURLY, ReportPeriod.DAILY]:
            task = asyncio.create_task(self._reporting_loop(period))
            self.reporting_tasks.append(task)
        
        # Start metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.reporting_tasks.append(metrics_task)
        
        logger.info("Performance reporting started")
    
    async def stop_reporting(self) -> None:
        """Stop performance reporting."""
        logger.info("Stopping performance reporting...")
        
        self.is_running = False
        
        # Cancel reporting tasks
        for task in self.reporting_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.reporting_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Performance reporting stopped")
    
    async def _reporting_loop(self, period: ReportPeriod) -> None:
        """Reporting loop for a specific period."""
        interval = self.report_intervals.get(period, 3600)
        
        while self.is_running:
            try:
                # Generate report
                report = await self.generate_report(ReportType.COMPREHENSIVE, period)
                
                # Save report
                await self._save_report(report)
                
                # Check for alerts
                await self._check_alerts(report)
                
                # Log report generation
                logger.info(f"Generated {period.value} performance report")
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in {period.value} reporting loop: {e}")
                await asyncio.sleep(interval)
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = await self._collect_current_metrics()
                
                # Update cache
                self.metrics_cache = metrics
                self.last_metrics_update = datetime.now()
                
                # Add to history
                self.performance_history.append(metrics)
                
                # Limit history size
                if len(self.performance_history) > 10000:
                    self.performance_history = self.performance_history[-5000:]
                
                # Save to database
                await self.db_manager.save_performance_metrics(metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # System metrics
            system_metrics = await self._collect_system_metrics()
            
            # Trading metrics
            trading_metrics = await self._collect_trading_metrics()
            
            # Discovery metrics
            discovery_metrics = await self._collect_discovery_metrics()
            
            # Risk metrics
            risk_metrics = await self._collect_risk_metrics()
            
            # Portfolio metrics
            portfolio_metrics = await self._collect_portfolio_metrics()
            
            # Agent metrics
            agent_metrics = await self._collect_agent_metrics()
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                system_metrics=system_metrics,
                trading_metrics=trading_metrics,
                discovery_metrics=discovery_metrics,
                risk_metrics=risk_metrics,
                portfolio_metrics=portfolio_metrics,
                agent_metrics=agent_metrics
            )
            
        except Exception as e:
            logger.error(f"Error collecting current metrics: {e}")
            return PerformanceMetrics(timestamp=datetime.now())
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': load_avg
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent,
                    'io_read_bytes': disk_io.read_bytes if disk_io else 0,
                    'io_write_bytes': disk_io.write_bytes if disk_io else 0
                },
                'network': {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                },
                'process': {
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'cpu_percent': process_cpu,
                    'num_threads': process.num_threads()
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def _collect_trading_metrics(self) -> Dict[str, Any]:
        """Collect trading performance metrics."""
        try:
            # Get trading engine
            trading_engine = getattr(self.orchestrator, 'trading_engine', None)
            if not trading_engine:
                return {}
            
            # Get recent trades
            recent_trades = await self.db_manager.get_recent_trades(hours=24)
            
            if not recent_trades:
                return {'total_trades': 0, 'total_volume': 0.0}
            
            # Calculate metrics
            total_trades = len(recent_trades)
            total_volume = sum(trade.get('quantity', 0) * trade.get('price', 0) for trade in recent_trades)
            
            # Calculate P&L
            total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades if trade.get('pnl'))
            winning_trades = [trade for trade in recent_trades if trade.get('pnl', 0) > 0]
            losing_trades = [trade for trade in recent_trades if trade.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # Average metrics
            avg_win = sum(trade.get('pnl', 0) for trade in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(trade.get('pnl', 0) for trade in losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Risk metrics
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            return {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'total_pnl': total_pnl,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor,
                'largest_win': max([trade.get('pnl', 0) for trade in winning_trades], default=0),
                'largest_loss': min([trade.get('pnl', 0) for trade in losing_trades], default=0)
            }
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return {}
    
    async def _collect_discovery_metrics(self) -> Dict[str, Any]:
        """Collect discovery effectiveness metrics."""
        try:
            # Get discovery engine
            discovery_engine = getattr(self.orchestrator, 'discovery_engine', None)
            if not discovery_engine:
                return {}
            
            # Get discovery stats
            discovery_stats = discovery_engine.get_discovery_stats()
            
            # Get recent discoveries
            recent_discoveries = await self.db_manager.get_recent_discoveries(hours=24)
            
            # Calculate effectiveness metrics
            discovery_metrics = {
                'total_discoveries': discovery_stats.get('total_discoveries', 0),
                'validated_discoveries': discovery_stats.get('validated_discoveries', 0),
                'executed_discoveries': discovery_stats.get('executed_discoveries', 0),
                'successful_discoveries': discovery_stats.get('successful_discoveries', 0),
                'average_confidence': discovery_stats.get('average_confidence', 0),
                'average_return': discovery_stats.get('average_return', 0),
                'win_rate': discovery_stats.get('win_rate', 0),
                'active_discoveries': discovery_stats.get('active_discoveries', 0)
            }
            
            # Add recent discovery breakdown
            if recent_discoveries:
                discovery_types = {}
                for discovery in recent_discoveries:
                    discovery_type = discovery.get('discovery_type', 'unknown')
                    discovery_types[discovery_type] = discovery_types.get(discovery_type, 0) + 1
                
                discovery_metrics['recent_discoveries_by_type'] = discovery_types
            
            return discovery_metrics
            
        except Exception as e:
            logger.error(f"Error collecting discovery metrics: {e}")
            return {}
    
    async def _collect_risk_metrics(self) -> Dict[str, Any]:
        """Collect risk management metrics."""
        try:
            # Get risk manager
            risk_manager = getattr(self.orchestrator, 'risk_manager', None)
            if not risk_manager:
                return {}
            
            # Get current risk metrics
            risk_metrics = await risk_manager.get_current_metrics()
            
            return {
                'var_95': risk_metrics.get('var_95', 0),
                'var_99': risk_metrics.get('var_99', 0),
                'expected_shortfall': risk_metrics.get('expected_shortfall', 0),
                'portfolio_beta': risk_metrics.get('portfolio_beta', 0),
                'concentration_risk': risk_metrics.get('concentration_risk', 0),
                'leverage_ratio': risk_metrics.get('leverage_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': risk_metrics.get('sortino_ratio', 0)
            }
            
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
            return {}
    
    async def _collect_portfolio_metrics(self) -> Dict[str, Any]:
        """Collect portfolio performance metrics."""
        try:
            # Get portfolio manager
            portfolio_manager = getattr(self.orchestrator, 'portfolio_manager', None)
            if not portfolio_manager:
                return {}
            
            # Get portfolio status
            portfolio_status = await portfolio_manager.get_status()
            
            return {
                'total_value': portfolio_status.get('total_value', 0),
                'cash_balance': portfolio_status.get('cash_balance', 0),
                'positions_value': portfolio_status.get('positions_value', 0),
                'daily_pnl': portfolio_status.get('daily_pnl', 0),
                'total_pnl': portfolio_status.get('total_pnl', 0),
                'position_count': portfolio_status.get('position_count', 0),
                'buying_power': portfolio_status.get('buying_power', 0),
                'margin_used': portfolio_status.get('margin_used', 0)
            }
            
        except Exception as e:
            logger.error(f"Error collecting portfolio metrics: {e}")
            return {}
    
    async def _collect_agent_metrics(self) -> Dict[str, Any]:
        """Collect agent performance metrics."""
        try:
            agent_metrics = {}
            
            # Get agent performance from database
            agent_performance = await self.db_manager.get_agent_performance(hours=24)
            
            for agent_name, performance in agent_performance.items():
                agent_metrics[agent_name] = {
                    'discoveries_generated': performance.get('discoveries_generated', 0),
                    'discoveries_validated': performance.get('discoveries_validated', 0),
                    'discoveries_executed': performance.get('discoveries_executed', 0),
                    'success_rate': performance.get('success_rate', 0),
                    'average_confidence': performance.get('average_confidence', 0),
                    'average_return': performance.get('average_return', 0),
                    'response_time': performance.get('response_time', 0)
                }
            
            return agent_metrics
            
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
            return {}
    
    async def generate_report(self, report_type: ReportType, period: ReportPeriod,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> PerformanceReport:
        """Generate a performance report."""
        try:
            # Set time range
            if not end_time:
                end_time = datetime.now()
            
            if not start_time:
                if period == ReportPeriod.REAL_TIME:
                    start_time = end_time - timedelta(minutes=5)
                elif period == ReportPeriod.HOURLY:
                    start_time = end_time - timedelta(hours=1)
                elif period == ReportPeriod.DAILY:
                    start_time = end_time - timedelta(days=1)
                elif period == ReportPeriod.WEEKLY:
                    start_time = end_time - timedelta(weeks=1)
                elif period == ReportPeriod.MONTHLY:
                    start_time = end_time - timedelta(days=30)
                else:
                    start_time = end_time - timedelta(hours=1)
            
            # Generate report ID
            report_id = f"{report_type.value}_{period.value}_{int(time.time())}"
            
            # Get historical metrics
            historical_metrics = await self._get_historical_metrics(start_time, end_time)
            
            # Calculate aggregated metrics
            aggregated_metrics = self._aggregate_metrics(historical_metrics)
            
            # Generate summary
            summary = await self._generate_summary(report_type, aggregated_metrics)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(aggregated_metrics)
            
            # Check for alerts
            alerts = await self._generate_alerts(aggregated_metrics)
            
            # Generate charts
            charts = await self._generate_charts(historical_metrics, report_type)
            
            # Create report
            report = PerformanceReport(
                report_id=report_id,
                report_type=report_type,
                period=period,
                start_time=start_time,
                end_time=end_time,
                generated_at=datetime.now(),
                metrics=aggregated_metrics,
                summary=summary,
                recommendations=recommendations,
                alerts=alerts,
                charts=charts
            )
            
            # Add to history
            self.report_history.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def _get_historical_metrics(self, start_time: datetime, end_time: datetime) -> List[PerformanceMetrics]:
        """Get historical metrics for time range."""
        try:
            # Get from database
            db_metrics = await self.db_manager.get_performance_metrics(start_time, end_time)
            
            # Convert to PerformanceMetrics objects
            historical_metrics = []
            for metric in db_metrics:
                historical_metrics.append(PerformanceMetrics(
                    timestamp=metric['timestamp'],
                    system_metrics=metric.get('system_metrics', {}),
                    trading_metrics=metric.get('trading_metrics', {}),
                    discovery_metrics=metric.get('discovery_metrics', {}),
                    risk_metrics=metric.get('risk_metrics', {}),
                    portfolio_metrics=metric.get('portfolio_metrics', {}),
                    agent_metrics=metric.get('agent_metrics', {})
                ))
            
            return historical_metrics
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []
    
    def _aggregate_metrics(self, historical_metrics: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate historical metrics."""
        if not historical_metrics:
            return PerformanceMetrics(timestamp=datetime.now())
        
        # Get latest metrics as base
        latest_metrics = historical_metrics[-1]
        
        # Calculate aggregations
        aggregated = PerformanceMetrics(
            timestamp=latest_metrics.timestamp,
            system_metrics=latest_metrics.system_metrics,
            trading_metrics=latest_metrics.trading_metrics,
            discovery_metrics=latest_metrics.discovery_metrics,
            risk_metrics=latest_metrics.risk_metrics,
            portfolio_metrics=latest_metrics.portfolio_metrics,
            agent_metrics=latest_metrics.agent_metrics
        )
        
        # Add time-series aggregations
        if len(historical_metrics) > 1:
            # Calculate trends
            self._calculate_trends(aggregated, historical_metrics)
        
        return aggregated
    
    def _calculate_trends(self, aggregated: PerformanceMetrics, historical: List[PerformanceMetrics]) -> None:
        """Calculate trends from historical data."""
        try:
            # Extract time series data
            timestamps = [m.timestamp for m in historical]
            
            # System trends
            cpu_values = [m.system_metrics.get('cpu', {}).get('percent', 0) for m in historical]
            memory_values = [m.system_metrics.get('memory', {}).get('percent', 0) for m in historical]
            
            # Trading trends
            pnl_values = [m.trading_metrics.get('total_pnl', 0) for m in historical]
            trade_count_values = [m.trading_metrics.get('total_trades', 0) for m in historical]
            
            # Calculate trends (simple linear regression)
            if len(cpu_values) > 1:
                cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
                memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
                pnl_trend = np.polyfit(range(len(pnl_values)), pnl_values, 1)[0]
                
                # Add trends to aggregated metrics
                aggregated.system_metrics['cpu_trend'] = cpu_trend
                aggregated.system_metrics['memory_trend'] = memory_trend
                aggregated.trading_metrics['pnl_trend'] = pnl_trend
                
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
    
    async def _generate_summary(self, report_type: ReportType, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate report summary."""
        try:
            summary = {
                'report_type': report_type.value,
                'generated_at': datetime.now().isoformat(),
                'key_metrics': {}
            }
            
            if report_type in [ReportType.SYSTEM_HEALTH, ReportType.COMPREHENSIVE]:
                summary['key_metrics']['system'] = {
                    'cpu_usage': metrics.system_metrics.get('cpu', {}).get('percent', 0),
                    'memory_usage': metrics.system_metrics.get('memory', {}).get('percent', 0),
                    'disk_usage': metrics.system_metrics.get('disk', {}).get('percent', 0)
                }
            
            if report_type in [ReportType.TRADING_PERFORMANCE, ReportType.COMPREHENSIVE]:
                summary['key_metrics']['trading'] = {
                    'total_pnl': metrics.trading_metrics.get('total_pnl', 0),
                    'win_rate': metrics.trading_metrics.get('win_rate', 0),
                    'total_trades': metrics.trading_metrics.get('total_trades', 0)
                }
            
            if report_type in [ReportType.DISCOVERY_EFFECTIVENESS, ReportType.COMPREHENSIVE]:
                summary['key_metrics']['discovery'] = {
                    'total_discoveries': metrics.discovery_metrics.get('total_discoveries', 0),
                    'success_rate': metrics.discovery_metrics.get('win_rate', 0),
                    'average_confidence': metrics.discovery_metrics.get('average_confidence', 0)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
    
    async def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        try:
            # System recommendations
            cpu_usage = metrics.system_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = metrics.system_metrics.get('memory', {}).get('percent', 0)
            
            if cpu_usage > 80:
                recommendations.append("High CPU usage detected. Consider optimizing algorithms or scaling resources.")
            
            if memory_usage > 85:
                recommendations.append("High memory usage detected. Consider implementing memory optimization.")
            
            # Trading recommendations
            win_rate = metrics.trading_metrics.get('win_rate', 0)
            profit_factor = metrics.trading_metrics.get('profit_factor', 0)
            
            if win_rate < 0.4:
                recommendations.append("Low win rate detected. Review trading strategies and risk management.")
            
            if profit_factor < 1.0:
                recommendations.append("Profit factor below 1.0. Consider adjusting position sizing or stop losses.")
            
            # Discovery recommendations
            discovery_success = metrics.discovery_metrics.get('win_rate', 0)
            avg_confidence = metrics.discovery_metrics.get('average_confidence', 0)
            
            if discovery_success < 0.5:
                recommendations.append("Low discovery success rate. Review validation criteria and agent performance.")
            
            if avg_confidence < 0.6:
                recommendations.append("Low average confidence in discoveries. Consider improving signal quality.")
            
            # Risk recommendations
            max_drawdown = metrics.risk_metrics.get('max_drawdown', 0)
            sharpe_ratio = metrics.risk_metrics.get('sharpe_ratio', 0)
            
            if max_drawdown > 0.1:
                recommendations.append("High drawdown detected. Review risk management and position sizing.")
            
            if sharpe_ratio < 1.0:
                recommendations.append("Low Sharpe ratio. Consider improving risk-adjusted returns.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _generate_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance alerts."""
        alerts = []
        
        try:
            # Critical system alerts
            cpu_usage = metrics.system_metrics.get('cpu', {}).get('percent', 0)
            memory_usage = metrics.system_metrics.get('memory', {}).get('percent', 0)
            disk_usage = metrics.system_metrics.get('disk', {}).get('percent', 0)
            
            if cpu_usage > 95:
                alerts.append("CRITICAL: CPU usage above 95%")
            
            if memory_usage > 95:
                alerts.append("CRITICAL: Memory usage above 95%")
            
            if disk_usage > 90:
                alerts.append("WARNING: Disk usage above 90%")
            
            # Trading alerts
            daily_pnl = metrics.portfolio_metrics.get('daily_pnl', 0)
            total_pnl = metrics.portfolio_metrics.get('total_pnl', 0)
            
            if daily_pnl < -10000:  # Configurable threshold
                alerts.append("ALERT: Daily P&L below -$10,000")
            
            if total_pnl < -50000:  # Configurable threshold
                alerts.append("CRITICAL: Total P&L below -$50,000")
            
            # Risk alerts
            var_95 = metrics.risk_metrics.get('var_95', 0)
            leverage_ratio = metrics.risk_metrics.get('leverage_ratio', 0)
            
            if var_95 > 0.05:  # 5% VaR threshold
                alerts.append("ALERT: Value at Risk above 5%")
            
            if leverage_ratio > 2.0:
                alerts.append("WARNING: Leverage ratio above 2.0")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    async def _generate_charts(self, historical_metrics: List[PerformanceMetrics], 
                              report_type: ReportType) -> Dict[str, str]:
        """Generate charts for the report."""
        charts = {}
        
        try:
            if not historical_metrics:
                return charts
            
            # Extract time series data
            timestamps = [m.timestamp for m in historical_metrics]
            
            # System performance chart
            if report_type in [ReportType.SYSTEM_HEALTH, ReportType.COMPREHENSIVE]:
                cpu_values = [m.system_metrics.get('cpu', {}).get('percent', 0) for m in historical_metrics]
                memory_values = [m.system_metrics.get('memory', {}).get('percent', 0) for m in historical_metrics]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                ax1.plot(timestamps, cpu_values, label='CPU Usage %', color='blue')
                ax1.set_ylabel('CPU Usage (%)')
                ax1.set_title('System Performance')
                ax1.legend()
                ax1.grid(True)
                
                ax2.plot(timestamps, memory_values, label='Memory Usage %', color='red')
                ax2.set_ylabel('Memory Usage (%)')
                ax2.set_xlabel('Time')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                charts['system_performance'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # Trading performance chart
            if report_type in [ReportType.TRADING_PERFORMANCE, ReportType.COMPREHENSIVE]:
                pnl_values = [m.trading_metrics.get('total_pnl', 0) for m in historical_metrics]
                trade_counts = [m.trading_metrics.get('total_trades', 0) for m in historical_metrics]
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                ax1.plot(timestamps, pnl_values, label='Total P&L', color='green')
                ax1.set_ylabel('P&L ($)')
                ax1.set_title('Trading Performance')
                ax1.legend()
                ax1.grid(True)
                
                ax2.bar(timestamps, trade_counts, label='Trade Count', alpha=0.7)
                ax2.set_ylabel('Number of Trades')
                ax2.set_xlabel('Time')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                charts['trading_performance'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # Portfolio value chart
            if report_type in [ReportType.PORTFOLIO_SUMMARY, ReportType.COMPREHENSIVE]:
                portfolio_values = [m.portfolio_metrics.get('total_value', 0) for m in historical_metrics]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(timestamps, portfolio_values, label='Portfolio Value', color='purple', linewidth=2)
                ax.set_ylabel('Portfolio Value ($)')
                ax.set_xlabel('Time')
                ax.set_title('Portfolio Value Over Time')
                ax.legend()
                ax.grid(True)
                
                plt.tight_layout()
                charts['portfolio_value'] = self._fig_to_base64(fig)
                plt.close(fig)
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {}
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            return image_base64
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return ""
    
    async def _save_report(self, report: PerformanceReport) -> None:
        """Save report to file and database."""
        try:
            # Save to file
            report_file = self.report_dir / f"{report.report_id}.json"
            
            report_data = {
                'report_id': report.report_id,
                'report_type': report.report_type.value,
                'period': report.period.value,
                'start_time': report.start_time.isoformat(),
                'end_time': report.end_time.isoformat(),
                'generated_at': report.generated_at.isoformat(),
                'summary': report.summary,
                'recommendations': report.recommendations,
                'alerts': report.alerts,
                'charts': report.charts
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Save to database
            await self.db_manager.save_performance_report(report_data)
            
            logger.info(f"Performance report saved: {report.report_id}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    async def _check_alerts(self, report: PerformanceReport) -> None:
        """Check for alerts and send notifications."""
        try:
            if report.alerts:
                for alert in report.alerts:
                    # Log alert
                    if alert.startswith("CRITICAL"):
                        logger.critical(f"Performance Alert: {alert}")
                    elif alert.startswith("WARNING"):
                        logger.warning(f"Performance Alert: {alert}")
                    else:
                        logger.info(f"Performance Alert: {alert}")
                    
                    # Send notification (implement notification system)
                    # await self.send_notification(alert)
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def get_latest_report(self, report_type: ReportType = ReportType.COMPREHENSIVE) -> Optional[PerformanceReport]:
        """Get the latest performance report."""
        try:
            # Find latest report of specified type
            latest_report = None
            for report in reversed(self.report_history):
                if report.report_type == report_type:
                    latest_report = report
                    break
            
            return latest_report
            
        except Exception as e:
            logger.error(f"Error getting latest report: {e}")
            return None
    
    async def get_report_by_id(self, report_id: str) -> Optional[PerformanceReport]:
        """Get report by ID."""
        try:
            for report in self.report_history:
                if report.report_id == report_id:
                    return report
            
            # Try loading from database
            report_data = await self.db_manager.get_performance_report(report_id)
            if report_data:
                # Convert back to PerformanceReport object
                # Implementation depends on database schema
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting report by ID: {e}")
            return None
    
    def get_reporting_stats(self) -> Dict[str, Any]:
        """Get reporting statistics."""
        return {
            'is_running': self.is_running,
            'reports_generated': len(self.report_history),
            'last_metrics_update': self.last_metrics_update.isoformat(),
            'metrics_cache_size': len(self.metrics_cache),
            'performance_history_size': len(self.performance_history)
        } 