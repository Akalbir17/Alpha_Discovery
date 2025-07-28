"""
Performance Tracker
Tracks and analyzes system performance
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks and analyzes system performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[Dict[str, Any]] = []
        self.is_running = False
        
        # Performance thresholds
        self.latency_threshold = config.get('latency_threshold', 1.0)  # seconds
        self.error_rate_threshold = config.get('error_rate_threshold', 0.05)  # 5%
        self.throughput_threshold = config.get('throughput_threshold', 100)  # ops/sec
        
    async def start(self):
        """Start performance tracking"""
        try:
            logger.info("Starting performance tracker...")
            self.is_running = True
            
            # Start performance analysis
            asyncio.create_task(self._analyze_performance())
            
            logger.info("Performance tracker started successfully")
        except Exception as e:
            logger.error(f"Failed to start performance tracker: {e}")
            raise
            
    async def stop(self):
        """Stop performance tracking"""
        try:
            logger.info("Stopping performance tracker...")
            self.is_running = False
            logger.info("Performance tracker stopped")
        except Exception as e:
            logger.error(f"Error stopping performance tracker: {e}")
            
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation performance"""
        self.performance_data[operation].append({
            'duration': duration,
            'success': success,
            'timestamp': datetime.now()
        })
        
    def record_latency(self, operation: str, latency: float):
        """Record operation latency"""
        self.performance_data[f"{operation}_latency"].append({
            'value': latency,
            'timestamp': datetime.now()
        })
        
    def record_throughput(self, operation: str, count: int):
        """Record operation throughput"""
        self.performance_data[f"{operation}_throughput"].append({
            'count': count,
            'timestamp': datetime.now()
        })
        
    def get_performance_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance summary"""
        if operation:
            return self._get_operation_summary(operation)
        else:
            return self._get_overall_summary()
            
    def _get_operation_summary(self, operation: str) -> Dict[str, Any]:
        """Get summary for specific operation"""
        data = list(self.performance_data.get(operation, []))
        if not data:
            return {'operation': operation, 'no_data': True}
            
        # Calculate metrics
        durations = [d['duration'] for d in data]
        successes = [d['success'] for d in data]
        
        total_ops = len(data)
        successful_ops = sum(successes)
        error_rate = (total_ops - successful_ops) / total_ops if total_ops > 0 else 0
        
        return {
            'operation': operation,
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'error_rate': error_rate,
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'p95_duration': np.percentile(durations, 95),
            'p99_duration': np.percentile(durations, 99),
            'last_updated': datetime.now()
        }
        
    def _get_overall_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        operations = {}
        
        for operation, data in self.performance_data.items():
            if data:
                operations[operation] = self._get_operation_summary(operation)
                
        return {
            'operations': operations,
            'alert_count': len(self.alerts),
            'timestamp': datetime.now()
        }
        
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        return self.alerts.copy()
        
    def clear_alerts(self):
        """Clear performance alerts"""
        self.alerts.clear()
        
    async def _analyze_performance(self):
        """Analyze performance and generate alerts"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Analyze every minute
                
                # Check each operation for performance issues
                for operation, data in self.performance_data.items():
                    if not data:
                        continue
                        
                    # Get recent data (last 5 minutes)
                    recent_data = [
                        d for d in data
                        if d['timestamp'] > datetime.now() - timedelta(minutes=5)
                    ]
                    
                    if not recent_data:
                        continue
                        
                    # Check latency
                    durations = [d['duration'] for d in recent_data]
                    avg_latency = np.mean(durations)
                    
                    if avg_latency > self.latency_threshold:
                        self.alerts.append({
                            'type': 'high_latency',
                            'operation': operation,
                            'value': avg_latency,
                            'threshold': self.latency_threshold,
                            'timestamp': datetime.now()
                        })
                        
                    # Check error rate
                    successes = [d['success'] for d in recent_data]
                    error_rate = (len(successes) - sum(successes)) / len(successes)
                    
                    if error_rate > self.error_rate_threshold:
                        self.alerts.append({
                            'type': 'high_error_rate',
                            'operation': operation,
                            'value': error_rate,
                            'threshold': self.error_rate_threshold,
                            'timestamp': datetime.now()
                        })
                        
                # Keep only recent alerts (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.alerts = [
                    alert for alert in self.alerts
                    if alert['timestamp'] > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Error analyzing performance: {e}")
                
    def get_performance_trends(self, operation: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time"""
        data = list(self.performance_data.get(operation, []))
        if not data:
            return {'operation': operation, 'no_data': True}
            
        # Filter data by time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_data = [d for d in data if d['timestamp'] > cutoff_time]
        
        if not filtered_data:
            return {'operation': operation, 'no_data': True}
            
        # Group by hour
        hourly_data = defaultdict(list)
        for d in filtered_data:
            hour = d['timestamp'].replace(minute=0, second=0, microsecond=0)
            hourly_data[hour].append(d)
            
        # Calculate hourly metrics
        trends = []
        for hour, hour_data in sorted(hourly_data.items()):
            durations = [d['duration'] for d in hour_data]
            successes = [d['success'] for d in hour_data]
            
            trends.append({
                'hour': hour,
                'count': len(hour_data),
                'avg_duration': np.mean(durations),
                'error_rate': (len(successes) - sum(successes)) / len(successes),
                'throughput': len(hour_data) / 3600  # ops per second
            })
            
        return {
            'operation': operation,
            'trends': trends,
            'hours': hours,
            'generated_at': datetime.now()
        } 