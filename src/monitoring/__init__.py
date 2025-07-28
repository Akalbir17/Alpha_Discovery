"""
Monitoring and Observability Module - 2025 Edition

Comprehensive monitoring, metrics collection, and observability features for
the Alpha Discovery platform including:

- Real-time performance metrics
- Health checks and system status
- Distributed tracing and logging
- Alerting and notification systems
- Dashboard and visualization support
- Prometheus metrics integration
- Custom business metrics
- Error tracking and analysis
"""

import asyncio
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import structlog
from pathlib import Path
import os

# Prometheus metrics support
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of metrics supported"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Metric value with metadata"""
    value: Union[int, float, str]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition"""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: str  # healthy, unhealthy, unknown
    timestamp: datetime
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class MetricsCollector:
    """
    Advanced metrics collector with Prometheus integration
    """
    
    def __init__(self, enable_prometheus: bool = True):
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.metric_configs: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        self.lock = threading.Lock()
        
        # Prometheus metrics
        self.prometheus_metrics = {}
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        if self.enable_prometheus:
            self._initialize_prometheus()
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics"""
        try:
            # Core platform metrics
            self.prometheus_metrics = {
                'requests_total': Counter(
                    'alpha_discovery_requests_total',
                    'Total number of requests',
                    ['component', 'method', 'status']
                ),
                'request_duration_seconds': Histogram(
                    'alpha_discovery_request_duration_seconds',
                    'Request duration in seconds',
                    ['component', 'method']
                ),
                'active_connections': Gauge(
                    'alpha_discovery_active_connections',
                    'Number of active connections',
                    ['component']
                ),
                'model_requests_total': Counter(
                    'alpha_discovery_model_requests_total',
                    'Total model requests',
                    ['model', 'task_type', 'status']
                ),
                'model_response_time_seconds': Histogram(
                    'alpha_discovery_model_response_time_seconds',
                    'Model response time in seconds',
                    ['model', 'task_type']
                ),
                'model_tokens_total': Counter(
                    'alpha_discovery_model_tokens_total',
                    'Total tokens processed',
                    ['model', 'task_type']
                ),
                'cache_hits_total': Counter(
                    'alpha_discovery_cache_hits_total',
                    'Total cache hits',
                    ['component', 'cache_type']
                ),
                'cache_misses_total': Counter(
                    'alpha_discovery_cache_misses_total',
                    'Total cache misses',
                    ['component', 'cache_type']
                ),
                'errors_total': Counter(
                    'alpha_discovery_errors_total',
                    'Total errors',
                    ['component', 'error_type']
                ),
                'system_info': Info(
                    'alpha_discovery_system_info',
                    'System information'
                )
            }
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
            self.enable_prometheus = False
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ):
        """Register a new metric"""
        try:
            with self.lock:
                self.metric_configs[name] = {
                    "type": metric_type,
                    "description": description,
                    "labels": labels or [],
                    "buckets": buckets,
                    "created_at": datetime.now()
                }
            
            logger.info(f"Metric registered: {name} ({metric_type.value})")
            
        except Exception as e:
            logger.error(f"Failed to register metric {name}: {e}")
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float, str],
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value"""
        try:
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            with self.lock:
                self.metrics[name].append(metric_value)
                
                # Limit history size
                if len(self.metrics[name]) > 10000:
                    self.metrics[name] = self.metrics[name][-5000:]
            
            # Update Prometheus metrics if enabled
            if self.enable_prometheus:
                self._update_prometheus_metric(name, value, labels or {})
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def _update_prometheus_metric(self, name: str, value: Union[int, float, str], labels: Dict[str, str]):
        """Update Prometheus metric"""
        try:
            if name in self.prometheus_metrics:
                metric = self.prometheus_metrics[name]
                
                if hasattr(metric, 'labels'):
                    # Metric with labels
                    labeled_metric = metric.labels(**labels)
                    
                    if hasattr(labeled_metric, 'inc'):
                        labeled_metric.inc(value if isinstance(value, (int, float)) else 1)
                    elif hasattr(labeled_metric, 'set'):
                        labeled_metric.set(value if isinstance(value, (int, float)) else 0)
                    elif hasattr(labeled_metric, 'observe'):
                        labeled_metric.observe(value if isinstance(value, (int, float)) else 0)
                else:
                    # Metric without labels
                    if hasattr(metric, 'inc'):
                        metric.inc(value if isinstance(value, (int, float)) else 1)
                    elif hasattr(metric, 'set'):
                        metric.set(value if isinstance(value, (int, float)) else 0)
                    elif hasattr(metric, 'observe'):
                        metric.observe(value if isinstance(value, (int, float)) else 0)
            
        except Exception as e:
            logger.warning(f"Failed to update Prometheus metric {name}: {e}")
    
    def get_metric_values(
        self,
        name: str,
        since: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """Get metric values with optional filtering"""
        try:
            with self.lock:
                values = self.metrics.get(name, [])
            
            # Filter by time
            if since:
                values = [v for v in values if v.timestamp >= since]
            
            # Filter by labels
            if labels:
                filtered_values = []
                for value in values:
                    if all(value.labels.get(k) == v for k, v in labels.items()):
                        filtered_values.append(value)
                values = filtered_values
            
            return values
            
        except Exception as e:
            logger.error(f"Failed to get metric values for {name}: {e}")
            return []
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get metric summary statistics"""
        try:
            values = self.get_metric_values(name, since)
            
            if not values:
                return {"count": 0}
            
            numeric_values = [v.value for v in values if isinstance(v.value, (int, float))]
            
            if not numeric_values:
                return {
                    "count": len(values),
                    "latest": values[-1].value if values else None,
                    "latest_timestamp": values[-1].timestamp.isoformat() if values else None
                }
            
            return {
                "count": len(values),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "avg": sum(numeric_values) / len(numeric_values),
                "latest": values[-1].value,
                "latest_timestamp": values[-1].timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metric summary for {name}: {e}")
            return {"error": str(e)}
    
    def start_prometheus_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        try:
            if self.enable_prometheus:
                start_http_server(port)
                logger.info(f"Prometheus metrics server started on port {port}")
            else:
                logger.warning("Prometheus not available, metrics server not started")
                
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")


class AlertManager:
    """
    Advanced alert management system
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        self.lock = threading.Lock()
        
        # Start alert checking
        self._start_alert_checking()
    
    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,  # "greater_than", "less_than", "equals", "not_equals"
        threshold: float,
        severity: AlertSeverity,
        message_template: str,
        labels: Optional[Dict[str, str]] = None,
        cooldown_seconds: int = 300
    ):
        """Add an alert rule"""
        try:
            with self.lock:
                self.alert_rules[name] = {
                    "metric_name": metric_name,
                    "condition": condition,
                    "threshold": threshold,
                    "severity": severity,
                    "message_template": message_template,
                    "labels": labels or {},
                    "cooldown_seconds": cooldown_seconds,
                    "last_triggered": None
                }
            
            logger.info(f"Alert rule added: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add alert rule {name}: {e}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    def _start_alert_checking(self):
        """Start background alert checking"""
        def check_alerts():
            while True:
                try:
                    self._check_alert_rules()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Error in alert checking: {e}")
                    time.sleep(30)  # Wait longer on error
        
        thread = threading.Thread(target=check_alerts, daemon=True)
        thread.start()
    
    def _check_alert_rules(self):
        """Check all alert rules"""
        try:
            with self.lock:
                rules = dict(self.alert_rules)
            
            for rule_name, rule in rules.items():
                try:
                    self._check_single_rule(rule_name, rule)
                except Exception as e:
                    logger.error(f"Error checking alert rule {rule_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in alert rule checking: {e}")
    
    def _check_single_rule(self, rule_name: str, rule: Dict[str, Any]):
        """Check a single alert rule"""
        try:
            # Get recent metric values
            since = datetime.now() - timedelta(minutes=5)
            values = self.metrics_collector.get_metric_values(
                rule["metric_name"],
                since=since,
                labels=rule["labels"]
            )
            
            if not values:
                return
            
            # Get latest value
            latest_value = values[-1]
            
            if not isinstance(latest_value.value, (int, float)):
                return
            
            # Check condition
            triggered = False
            condition = rule["condition"]
            threshold = rule["threshold"]
            
            if condition == "greater_than" and latest_value.value > threshold:
                triggered = True
            elif condition == "less_than" and latest_value.value < threshold:
                triggered = True
            elif condition == "equals" and latest_value.value == threshold:
                triggered = True
            elif condition == "not_equals" and latest_value.value != threshold:
                triggered = True
            
            # Handle alert
            if triggered:
                self._trigger_alert(rule_name, rule, latest_value)
            else:
                self._resolve_alert(rule_name)
                
        except Exception as e:
            logger.error(f"Error checking rule {rule_name}: {e}")
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metric_value: MetricValue):
        """Trigger an alert"""
        try:
            now = datetime.now()
            
            # Check cooldown
            if rule["last_triggered"]:
                time_since_last = (now - rule["last_triggered"]).total_seconds()
                if time_since_last < rule["cooldown_seconds"]:
                    return
            
            # Create alert
            alert = Alert(
                name=rule_name,
                severity=rule["severity"],
                message=rule["message_template"].format(
                    metric_name=rule["metric_name"],
                    value=metric_value.value,
                    threshold=rule["threshold"]
                ),
                timestamp=now,
                metric_name=rule["metric_name"],
                metric_value=metric_value.value,
                threshold=rule["threshold"],
                labels=rule["labels"]
            )
            
            # Store alert
            with self.lock:
                self.active_alerts[rule_name] = alert
                self.alert_history.append(alert)
                self.alert_rules[rule_name]["last_triggered"] = now
            
            # Send notifications
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Notification handler failed: {e}")
            
            logger.warning(f"Alert triggered: {rule_name} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert {rule_name}: {e}")
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an alert"""
        try:
            with self.lock:
                if rule_name in self.active_alerts:
                    alert = self.active_alerts[rule_name]
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    del self.active_alerts[rule_name]
                    
                    logger.info(f"Alert resolved: {rule_name}")
                    
        except Exception as e:
            logger.error(f"Failed to resolve alert {rule_name}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        with self.lock:
            return self.alert_history[-limit:]


class HealthMonitor:
    """
    Comprehensive health monitoring system
    """
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.check_results: Dict[str, HealthCheck] = {}
        self.check_configs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Start health checking
        self._start_health_checking()
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable,
        interval_seconds: int = 30,
        timeout_seconds: int = 5,
        critical: bool = False
    ):
        """Register a health check"""
        try:
            with self.lock:
                self.health_checks[name] = check_function
                self.check_configs[name] = {
                    "interval_seconds": interval_seconds,
                    "timeout_seconds": timeout_seconds,
                    "critical": critical,
                    "last_check": None
                }
            
            logger.info(f"Health check registered: {name}")
            
        except Exception as e:
            logger.error(f"Failed to register health check {name}: {e}")
    
    def _start_health_checking(self):
        """Start background health checking"""
        def run_health_checks():
            while True:
                try:
                    self._run_all_checks()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Error in health checking: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=run_health_checks, daemon=True)
        thread.start()
    
    def _run_all_checks(self):
        """Run all health checks"""
        try:
            now = datetime.now()
            
            with self.lock:
                checks = dict(self.health_checks)
                configs = dict(self.check_configs)
            
            for name, check_func in checks.items():
                try:
                    config = configs[name]
                    
                    # Check if it's time to run this check
                    if config["last_check"]:
                        time_since_last = (now - config["last_check"]).total_seconds()
                        if time_since_last < config["interval_seconds"]:
                            continue
                    
                    # Run the check
                    self._run_single_check(name, check_func, config)
                    
                    # Update last check time
                    with self.lock:
                        self.check_configs[name]["last_check"] = now
                        
                except Exception as e:
                    logger.error(f"Error running health check {name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in health check execution: {e}")
    
    def _run_single_check(self, name: str, check_func: Callable, config: Dict[str, Any]):
        """Run a single health check"""
        try:
            start_time = time.time()
            
            # Run the check with timeout
            try:
                if asyncio.iscoroutinefunction(check_func):
                    # Async function
                    result = asyncio.run(
                        asyncio.wait_for(
                            check_func(),
                            timeout=config["timeout_seconds"]
                        )
                    )
                else:
                    # Sync function
                    result = check_func()
                
                response_time = time.time() - start_time
                
                # Process result
                if isinstance(result, dict):
                    status = result.get("status", "healthy")
                    details = result.get("details", {})
                    error = result.get("error")
                else:
                    status = "healthy" if result else "unhealthy"
                    details = {}
                    error = None
                
                health_check = HealthCheck(
                    name=name,
                    status=status,
                    timestamp=datetime.now(),
                    response_time=response_time,
                    details=details,
                    error=error
                )
                
            except asyncio.TimeoutError:
                health_check = HealthCheck(
                    name=name,
                    status="unhealthy",
                    timestamp=datetime.now(),
                    response_time=config["timeout_seconds"],
                    error="Check timed out"
                )
            
            except Exception as e:
                health_check = HealthCheck(
                    name=name,
                    status="unhealthy",
                    timestamp=datetime.now(),
                    response_time=time.time() - start_time,
                    error=str(e)
                )
            
            # Store result
            with self.lock:
                self.check_results[name] = health_check
            
            # Log if unhealthy
            if health_check.status != "healthy":
                logger.warning(f"Health check failed: {name} - {health_check.error}")
            
        except Exception as e:
            logger.error(f"Error in health check {name}: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        try:
            with self.lock:
                results = dict(self.check_results)
                configs = dict(self.check_configs)
            
            overall_healthy = True
            critical_issues = []
            
            for name, result in results.items():
                if result.status != "healthy":
                    overall_healthy = False
                    
                    if configs.get(name, {}).get("critical", False):
                        critical_issues.append(name)
            
            return {
                "overall_healthy": overall_healthy,
                "critical_issues": critical_issues,
                "checks": {name: asdict(result) for name, result in results.items()},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {"error": str(e)}


class MonitoringDashboard:
    """
    Monitoring dashboard and reporting system
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        health_monitor: HealthMonitor
    ):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.health_monitor = health_monitor
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get metrics summary
            metrics_summary = {}
            for metric_name in self.metrics_collector.metrics.keys():
                metrics_summary[metric_name] = self.metrics_collector.get_metric_summary(metric_name)
            
            # Get health status
            health_status = self.health_monitor.get_health_status()
            
            # Get alerts
            active_alerts = self.alert_manager.get_active_alerts()
            recent_alerts = self.alert_manager.get_alert_history(limit=20)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics_summary,
                "health": health_status,
                "alerts": {
                    "active": [asdict(alert) for alert in active_alerts],
                    "recent": [asdict(alert) for alert in recent_alerts]
                },
                "system_info": {
                    "prometheus_enabled": self.metrics_collector.enable_prometheus,
                    "total_metrics": len(self.metrics_collector.metrics),
                    "total_health_checks": len(self.health_monitor.health_checks),
                    "total_alert_rules": len(self.alert_manager.alert_rules)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    def generate_report(self, format_type: str = "json") -> str:
        """Generate monitoring report"""
        try:
            data = self.get_dashboard_data()
            
            if format_type == "json":
                return json.dumps(data, indent=2, default=str)
            elif format_type == "summary":
                return self._generate_summary_report(data)
            else:
                return json.dumps(data, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"
    
    def _generate_summary_report(self, data: Dict[str, Any]) -> str:
        """Generate summary text report"""
        try:
            lines = []
            lines.append("=== Alpha Discovery Platform Monitoring Report ===")
            lines.append(f"Generated: {data['timestamp']}")
            lines.append("")
            
            # Health summary
            health = data.get("health", {})
            lines.append("=== Health Status ===")
            lines.append(f"Overall Healthy: {health.get('overall_healthy', 'Unknown')}")
            
            critical_issues = health.get("critical_issues", [])
            if critical_issues:
                lines.append(f"Critical Issues: {', '.join(critical_issues)}")
            
            lines.append("")
            
            # Alerts summary
            alerts = data.get("alerts", {})
            active_alerts = alerts.get("active", [])
            lines.append("=== Alerts ===")
            lines.append(f"Active Alerts: {len(active_alerts)}")
            
            for alert in active_alerts:
                lines.append(f"  - {alert['name']}: {alert['message']} ({alert['severity']})")
            
            lines.append("")
            
            # Metrics summary
            metrics = data.get("metrics", {})
            lines.append("=== Key Metrics ===")
            
            for metric_name, summary in metrics.items():
                if summary.get("count", 0) > 0:
                    lines.append(f"{metric_name}: {summary.get('latest', 'N/A')} (avg: {summary.get('avg', 'N/A')})")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return f"Error generating summary report: {e}"


# Global monitoring instances
metrics_collector = MetricsCollector(get_config('monitoring'))
alert_manager = AlertManager(metrics_collector)
health_monitor = HealthMonitor()
dashboard = MonitoringDashboard(metrics_collector, alert_manager, health_monitor)

# Common notification handlers
def log_alert_handler(alert: Alert):
    """Log alert to structured logger"""
    logger.warning(
        "Alert triggered",
        alert_name=alert.name,
        severity=alert.severity.value,
        message=alert.message,
        metric_name=alert.metric_name,
        metric_value=alert.metric_value,
        threshold=alert.threshold
    )

def email_alert_handler(alert: Alert):
    """Email alert handler (placeholder)"""
    # Would integrate with email service
    logger.info(f"Email alert would be sent: {alert.name}")

def slack_alert_handler(alert: Alert):
    """Slack alert handler (placeholder)"""
    # Would integrate with Slack API
    logger.info(f"Slack alert would be sent: {alert.name}")

# Register default alert handlers
alert_manager.add_notification_handler(log_alert_handler)

# Common health checks
async def model_manager_health_check():
    """Health check for model manager"""
    try:
        from ..utils.model_manager import model_manager
        stats = model_manager.get_enhanced_stats()
        
        available_models = len(stats.get("available_models", []))
        
        return {
            "status": "healthy" if available_models > 0 else "unhealthy",
            "details": {
                "available_models": available_models,
                "total_models": stats.get("total_models", 0)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def redis_health_check():
    """Health check for Redis"""
    try:
        import redis
        from ..mcp.config import tool_config
        
        client = redis.Redis(
            host=tool_config.redis_host,
            port=tool_config.redis_port,
            db=tool_config.redis_db,
            socket_timeout=2
        )
        
        client.ping()
        
        return {
            "status": "healthy",
            "details": {"connection": "ok"}
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Register common health checks
health_monitor.register_health_check(
    "model_manager",
    model_manager_health_check,
    interval_seconds=30,
    critical=True
)

health_monitor.register_health_check(
    "redis",
    redis_health_check,
    interval_seconds=60,
    critical=False
)

# Export monitoring components
__all__ = [
    "MetricsCollector",
    "AlertManager", 
    "HealthMonitor",
    "MonitoringDashboard",
    "MetricType",
    "AlertSeverity",
    "MetricValue",
    "Alert",
    "HealthCheck",
    "metrics_collector",
    "alert_manager",
    "health_monitor",
    "dashboard"
] 