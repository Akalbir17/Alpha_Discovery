#!/usr/bin/env python3
"""
Alpha Discovery Health Check System

Comprehensive health check system for all services in the Alpha Discovery platform.
Provides detailed health status, dependency checks, and service monitoring.
"""

import os
import sys
import json
import time
import logging
import requests
import psycopg2
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status data structure."""
    service: str
    status: str  # healthy, unhealthy, warning, unknown
    response_time: float
    timestamp: datetime
    details: Dict
    dependencies: List[str]
    error_message: Optional[str] = None

class HealthChecker:
    """Comprehensive health checker for Alpha Discovery services."""
    
    def __init__(self):
        self.services = {
            'postgres': {
                'type': 'database',
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', 5432)),
                'check_func': self.check_postgres
            },
            'redis': {
                'type': 'cache',
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
                'check_func': self.check_redis
            },
            'timescaledb': {
                'type': 'database',
                'host': os.getenv('TIMESCALE_HOST', 'localhost'),
                'port': int(os.getenv('TIMESCALE_PORT', 5433)),
                'check_func': self.check_timescaledb
            },
            'api-gateway': {
                'type': 'service',
                'url': 'http://localhost:8000/health',
                'check_func': self.check_http_service
            },
            'dashboard': {
                'type': 'service',
                'url': 'http://localhost:8501/_stcore/health',
                'check_func': self.check_http_service
            },
            'market-data-service': {
                'type': 'service',
                'url': 'http://localhost:8001/health',
                'check_func': self.check_http_service
            },
            'trading-engine': {
                'type': 'service',
                'url': 'http://localhost:8002/health',
                'check_func': self.check_http_service
            },
            'risk-service': {
                'type': 'service',
                'url': 'http://localhost:8003/health',
                'check_func': self.check_http_service
            },
            'sentiment-service': {
                'type': 'service',
                'url': 'http://localhost:8004/health',
                'check_func': self.check_http_service
            },
            'prometheus': {
                'type': 'monitoring',
                'url': 'http://localhost:9090/-/healthy',
                'check_func': self.check_http_service
            },
            'grafana': {
                'type': 'monitoring',
                'url': 'http://localhost:3000/api/health',
                'check_func': self.check_http_service
            },
            'alertmanager': {
                'type': 'monitoring',
                'url': 'http://localhost:9093/-/healthy',
                'check_func': self.check_http_service
            }
        }
        
        self.dependencies = {
            'api-gateway': ['postgres', 'redis'],
            'dashboard': ['api-gateway'],
            'market-data-service': ['postgres', 'redis', 'timescaledb'],
            'trading-engine': ['postgres', 'redis', 'market-data-service'],
            'risk-service': ['postgres', 'redis', 'market-data-service'],
            'sentiment-service': ['postgres', 'redis'],
            'grafana': ['prometheus'],
            'alertmanager': ['prometheus']
        }
    
    def check_postgres(self, service_config: Dict) -> HealthStatus:
        """Check PostgreSQL health."""
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host=service_config['host'],
                port=service_config['port'],
                database=os.getenv('POSTGRES_DB', 'alpha_discovery'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD', ''),
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute('SELECT version();')
            version = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM pg_stat_activity;')
            active_connections = cursor.fetchone()[0]
            
            cursor.execute('SELECT pg_database_size(current_database());')
            db_size = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                service='postgres',
                status='healthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'version': version,
                    'active_connections': active_connections,
                    'database_size_bytes': db_size,
                    'host': service_config['host'],
                    'port': service_config['port']
                },
                dependencies=[]
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthStatus(
                service='postgres',
                status='unhealthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'host': service_config['host'],
                    'port': service_config['port']
                },
                dependencies=[],
                error_message=str(e)
            )
    
    def check_redis(self, service_config: Dict) -> HealthStatus:
        """Check Redis health."""
        start_time = time.time()
        
        try:
            r = redis.Redis(
                host=service_config['host'],
                port=service_config['port'],
                password=os.getenv('REDIS_PASSWORD'),
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            r.ping()
            
            # Get Redis info
            info = r.info()
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                service='redis',
                status='healthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'redis_version': info.get('redis_version'),
                    'connected_clients': info.get('connected_clients'),
                    'used_memory': info.get('used_memory'),
                    'used_memory_human': info.get('used_memory_human'),
                    'keyspace_hits': info.get('keyspace_hits'),
                    'keyspace_misses': info.get('keyspace_misses'),
                    'host': service_config['host'],
                    'port': service_config['port']
                },
                dependencies=[]
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthStatus(
                service='redis',
                status='unhealthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'host': service_config['host'],
                    'port': service_config['port']
                },
                dependencies=[],
                error_message=str(e)
            )
    
    def check_timescaledb(self, service_config: Dict) -> HealthStatus:
        """Check TimescaleDB health."""
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host=service_config['host'],
                port=service_config['port'],
                database=os.getenv('TIMESCALE_DB', 'market_data'),
                user=os.getenv('TIMESCALE_USER', 'timescale'),
                password=os.getenv('TIMESCALE_PASSWORD', ''),
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            
            # Check TimescaleDB extension
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
            timescale_version = cursor.fetchone()
            
            # Check hypertables
            cursor.execute("SELECT COUNT(*) FROM timescaledb_information.hypertables;")
            hypertable_count = cursor.fetchone()[0]
            
            # Check database size
            cursor.execute('SELECT pg_database_size(current_database());')
            db_size = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                service='timescaledb',
                status='healthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'timescaledb_version': timescale_version[0] if timescale_version else None,
                    'hypertable_count': hypertable_count,
                    'database_size_bytes': db_size,
                    'host': service_config['host'],
                    'port': service_config['port']
                },
                dependencies=[]
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthStatus(
                service='timescaledb',
                status='unhealthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'host': service_config['host'],
                    'port': service_config['port']
                },
                dependencies=[],
                error_message=str(e)
            )
    
    def check_http_service(self, service_config: Dict) -> HealthStatus:
        """Check HTTP service health."""
        service_name = [k for k, v in self.services.items() if v == service_config][0]
        start_time = time.time()
        
        try:
            response = requests.get(
                service_config['url'],
                timeout=10,
                headers={'User-Agent': 'Alpha-Discovery-Health-Check/1.0'}
            )
            
            response_time = time.time() - start_time
            
            # Determine status based on response
            if response.status_code == 200:
                status = 'healthy'
                try:
                    response_data = response.json()
                except:
                    response_data = {'status': 'ok'}
            else:
                status = 'unhealthy'
                response_data = {'status_code': response.status_code}
            
            return HealthStatus(
                service=service_name,
                status=status,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'url': service_config['url'],
                    'status_code': response.status_code,
                    'response_data': response_data
                },
                dependencies=self.dependencies.get(service_name, []),
                error_message=None if status == 'healthy' else f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthStatus(
                service=service_name,
                status='unhealthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'url': service_config['url']
                },
                dependencies=self.dependencies.get(service_name, []),
                error_message=str(e)
            )
    
    def check_system_resources(self) -> HealthStatus:
        """Check system resources."""
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = self.get_cpu_usage()
            
            # Memory usage
            memory_info = self.get_memory_usage()
            
            # Disk usage
            disk_info = self.get_disk_usage()
            
            # Load average
            load_avg = self.get_load_average()
            
            # Determine status based on resource usage
            status = 'healthy'
            warnings = []
            
            if cpu_percent > 80:
                status = 'warning'
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory_info['percent'] > 85:
                status = 'warning'
                warnings.append(f"High memory usage: {memory_info['percent']}%")
            
            if disk_info['percent'] > 90:
                status = 'warning'
                warnings.append(f"High disk usage: {disk_info['percent']}%")
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                service='system',
                status=status,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory': memory_info,
                    'disk': disk_info,
                    'load_average': load_avg,
                    'warnings': warnings
                },
                dependencies=[]
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthStatus(
                service='system',
                status='unhealthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={},
                dependencies=[],
                error_message=str(e)
            )
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if '%Cpu(s):' in line:
                    # Extract CPU usage from top output
                    parts = line.split(',')
                    for part in parts:
                        if 'us' in part:
                            return float(part.split('%')[0].strip())
            return 0.0
        except:
            return 0.0
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage information."""
        try:
            result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            mem_line = lines[1].split()
            total = int(mem_line[1])
            used = int(mem_line[2])
            free = int(mem_line[3])
            percent = (used / total) * 100
            
            return {
                'total_mb': total,
                'used_mb': used,
                'free_mb': free,
                'percent': round(percent, 2)
            }
        except:
            return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'percent': 0}
    
    def get_disk_usage(self) -> Dict:
        """Get disk usage information."""
        try:
            result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            disk_line = lines[1].split()
            total = disk_line[1]
            used = disk_line[2]
            free = disk_line[3]
            percent = int(disk_line[4].replace('%', ''))
            
            return {
                'total': total,
                'used': used,
                'free': free,
                'percent': percent
            }
        except:
            return {'total': '0G', 'used': '0G', 'free': '0G', 'percent': 0}
    
    def get_load_average(self) -> List[float]:
        """Get system load average."""
        try:
            result = subprocess.run(['uptime'], capture_output=True, text=True)
            output = result.stdout
            # Extract load average from uptime output
            load_part = output.split('load average: ')[1]
            loads = [float(x.strip()) for x in load_part.split(',')]
            return loads
        except:
            return [0.0, 0.0, 0.0]
    
    def check_docker_containers(self) -> HealthStatus:
        """Check Docker containers status."""
        start_time = time.time()
        
        try:
            result = subprocess.run(['docker', 'ps', '--format', 'json'], capture_output=True, text=True)
            containers = []
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    container = json.loads(line)
                    containers.append({
                        'name': container['Names'],
                        'status': container['Status'],
                        'image': container['Image'],
                        'ports': container.get('Ports', '')
                    })
            
            # Check for unhealthy containers
            unhealthy_containers = [c for c in containers if 'unhealthy' in c['status'].lower()]
            
            status = 'healthy' if not unhealthy_containers else 'warning'
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                service='docker',
                status=status,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'total_containers': len(containers),
                    'unhealthy_containers': len(unhealthy_containers),
                    'containers': containers
                },
                dependencies=[]
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthStatus(
                service='docker',
                status='unhealthy',
                response_time=response_time,
                timestamp=datetime.now(),
                details={},
                dependencies=[],
                error_message=str(e)
            )
    
    def check_all_services(self, parallel: bool = True) -> Dict[str, HealthStatus]:
        """Check all services health."""
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all health checks
                future_to_service = {}
                
                for service_name, service_config in self.services.items():
                    future = executor.submit(service_config['check_func'], service_config)
                    future_to_service[future] = service_name
                
                # Add system checks
                system_future = executor.submit(self.check_system_resources)
                future_to_service[system_future] = 'system'
                
                docker_future = executor.submit(self.check_docker_containers)
                future_to_service[docker_future] = 'docker'
                
                # Collect results
                for future in as_completed(future_to_service):
                    service_name = future_to_service[future]
                    try:
                        result = future.result()
                        results[service_name] = result
                    except Exception as e:
                        results[service_name] = HealthStatus(
                            service=service_name,
                            status='unhealthy',
                            response_time=0,
                            timestamp=datetime.now(),
                            details={},
                            dependencies=[],
                            error_message=str(e)
                        )
        else:
            # Sequential checks
            for service_name, service_config in self.services.items():
                results[service_name] = service_config['check_func'](service_config)
            
            results['system'] = self.check_system_resources()
            results['docker'] = self.check_docker_containers()
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthStatus]) -> str:
        """Get overall system status."""
        statuses = [result.status for result in results.values()]
        
        if 'unhealthy' in statuses:
            return 'unhealthy'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'healthy'
    
    def generate_report(self, results: Dict[str, HealthStatus]) -> Dict:
        """Generate comprehensive health report."""
        overall_status = self.get_overall_status(results)
        
        # Categorize services
        healthy_services = [name for name, result in results.items() if result.status == 'healthy']
        warning_services = [name for name, result in results.items() if result.status == 'warning']
        unhealthy_services = [name for name, result in results.items() if result.status == 'unhealthy']
        
        # Calculate average response time
        response_times = [result.response_time for result in results.values() if result.response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_services': len(results),
                'healthy_services': len(healthy_services),
                'warning_services': len(warning_services),
                'unhealthy_services': len(unhealthy_services),
                'average_response_time': round(avg_response_time, 3)
            },
            'services': {
                'healthy': healthy_services,
                'warning': warning_services,
                'unhealthy': unhealthy_services
            },
            'detailed_results': {name: asdict(result) for name, result in results.items()}
        }
    
    def save_report(self, report: Dict, filename: str = None):
        """Save health report to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"/var/log/alpha-discovery/health_report_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Health report saved to {filename}")

def main():
    """Main function to run health checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha Discovery Health Check')
    parser.add_argument('--service', help='Check specific service only')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--format', choices=['json', 'table'], default='json', help='Output format')
    parser.add_argument('--parallel', action='store_true', default=True, help='Run checks in parallel')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Interval for continuous monitoring (seconds)')
    
    args = parser.parse_args()
    
    health_checker = HealthChecker()
    
    if args.continuous:
        logger.info(f"Starting continuous health monitoring (interval: {args.interval}s)")
        while True:
            try:
                results = health_checker.check_all_services(parallel=args.parallel)
                report = health_checker.generate_report(results)
                
                if args.output:
                    health_checker.save_report(report, args.output)
                
                overall_status = report['overall_status']
                logger.info(f"Overall status: {overall_status}")
                
                if overall_status != 'healthy':
                    unhealthy = report['services']['unhealthy']
                    warning = report['services']['warning']
                    if unhealthy:
                        logger.error(f"Unhealthy services: {unhealthy}")
                    if warning:
                        logger.warning(f"Warning services: {warning}")
                
                time.sleep(args.interval)
                
            except KeyboardInterrupt:
                logger.info("Health monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(args.interval)
    else:
        # Single run
        if args.service:
            if args.service in health_checker.services:
                service_config = health_checker.services[args.service]
                result = service_config['check_func'](service_config)
                results = {args.service: result}
            else:
                logger.error(f"Unknown service: {args.service}")
                sys.exit(1)
        else:
            results = health_checker.check_all_services(parallel=args.parallel)
        
        report = health_checker.generate_report(results)
        
        if args.format == 'json':
            print(json.dumps(report, indent=2, default=str))
        elif args.format == 'table':
            print(f"Overall Status: {report['overall_status']}")
            print(f"Timestamp: {report['timestamp']}")
            print("\nService Status:")
            print("-" * 60)
            for service_name, result in results.items():
                status_symbol = "✓" if result.status == 'healthy' else "⚠" if result.status == 'warning' else "✗"
                print(f"{status_symbol} {service_name:<20} {result.status:<10} {result.response_time:.3f}s")
                if result.error_message:
                    print(f"  Error: {result.error_message}")
        
        if args.output:
            health_checker.save_report(report, args.output)
        
        # Exit with appropriate code
        if report['overall_status'] == 'unhealthy':
            sys.exit(1)
        elif report['overall_status'] == 'warning':
            sys.exit(2)
        else:
            sys.exit(0)

if __name__ == "__main__":
    main() 