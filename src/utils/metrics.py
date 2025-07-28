"""
Metrics Collector
Collects and manages system metrics
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import threading

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.is_running = False
        self._lock = threading.Lock()
        
    async def start(self):
        """Start metrics collection"""
        try:
            logger.info("Starting metrics collector...")
            self.is_running = True
            
            # Start system metrics collection
            asyncio.create_task(self._collect_system_metrics())
            
            logger.info("Metrics collector started successfully")
        except Exception as e:
            logger.error(f"Failed to start metrics collector: {e}")
            raise
            
    async def stop(self):
        """Stop metrics collection"""
        try:
            logger.info("Stopping metrics collector...")
            self.is_running = False
            logger.info("Metrics collector stopped")
        except Exception as e:
            logger.error(f"Error stopping metrics collector: {e}")
            
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric"""
        with self._lock:
            self.counters[name] += value
            
    def set_gauge(self, name: str, value: float):
        """Set a gauge metric"""
        with self._lock:
            self.gauges[name] = value
            
    def record_histogram(self, name: str, value: float):
        """Record a histogram metric"""
        with self._lock:
            self.metrics[name].append({
                'value': value,
                'timestamp': datetime.now()
            })
            
    def record_timing(self, name: str, duration: float):
        """Record a timing metric"""
        self.record_histogram(f"{name}_duration", duration)
        
    def record_trade(self, execution_result: Dict[str, Any]):
        """Record a trade metric"""
        self.increment_counter('trades_total')
        self.increment_counter(f"trades_{execution_result['side']}")
        self.record_histogram('trade_size', execution_result['quantity'])
        self.record_histogram('trade_price', execution_result['price'])
        
    def get_counter(self, name: str) -> int:
        """Get counter value"""
        with self._lock:
            return self.counters.get(name, 0)
            
    def get_gauge(self, name: str) -> float:
        """Get gauge value"""
        with self._lock:
            return self.gauges.get(name, 0.0)
            
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            values = [m['value'] for m in self.metrics[name]]
            if not values:
                return {'count': 0, 'min': 0, 'max': 0, 'mean': 0, 'p95': 0, 'p99': 0}
                
            values.sort()
            count = len(values)
            
            return {
                'count': count,
                'min': values[0],
                'max': values[-1],
                'mean': sum(values) / count,
                'p95': values[int(count * 0.95)] if count > 0 else 0,
                'p99': values[int(count * 0.99)] if count > 0 else 0
            }
            
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {
                    name: self.get_histogram_stats(name)
                    for name in self.metrics.keys()
                },
                'timestamp': datetime.now()
            }
            
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self.is_running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge('system_cpu_percent', cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.set_gauge('system_memory_percent', memory.percent)
                self.set_gauge('system_memory_available', memory.available)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.set_gauge('system_disk_percent', (disk.used / disk.total) * 100)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.set_gauge('system_bytes_sent', net_io.bytes_sent)
                self.set_gauge('system_bytes_recv', net_io.bytes_recv)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
                
    def timing_context(self, name: str):
        """Context manager for timing operations"""
        return TimingContext(self, name)
        
class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_collector: MetricsCollector, name: str):
        self.metrics_collector = metrics_collector
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timing(self.name, duration) 