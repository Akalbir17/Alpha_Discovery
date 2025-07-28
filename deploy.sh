#!/bin/bash

# Alpha Discovery Deployment Script
# Comprehensive deployment with Docker, migrations, health checks, monitoring, and blue-green deployment

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="alpha-discovery"
DEPLOYMENT_ID="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/var/log/${PROJECT_NAME}/deployment-${DEPLOYMENT_ID}.log"

# Environment settings
ENVIRONMENT="${ENVIRONMENT:-development}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-rolling}"  # rolling, blue-green, canary

# Service configuration
SERVICES=(
    "postgres"
    "redis"
    "timescaledb"
    "market-data-service"
    "trading-engine"
    "risk-service"
    "sentiment-service"
    "api-gateway"
    "dashboard"
    "monitoring"
)

# Health check configuration
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
HEALTH_CHECK_INTERVAL=10  # 10 seconds
MAX_RETRIES=3

# Blue-green deployment configuration
BLUE_GREEN_ENABLED=false
CURRENT_SLOT="${CURRENT_SLOT:-blue}"
TARGET_SLOT="${TARGET_SLOT:-green}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# LOGGING AND UTILITIES
# =============================================================================

# Setup logging
setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Error occurred in script at line $line_number (exit code: $exit_code)"
    
    if [[ "$DEPLOYMENT_TYPE" == "blue-green" ]]; then
        log_warning "Blue-green deployment failed, initiating rollback..."
        rollback_deployment
    fi
    
    cleanup
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f /tmp/${PROJECT_NAME}-*.tmp
    
    # Remove any dangling containers
    docker container prune -f || true
    
    # Remove unused networks
    docker network prune -f || true
}

# =============================================================================
# VALIDATION AND PREREQUISITES
# =============================================================================

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]] && ! groups | grep -q docker; then
        log_error "This script must be run as root or user must be in docker group"
        exit 1
    fi
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq" "nc")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check environment variables
    local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    # Check disk space (minimum 10GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        log_error "Insufficient disk space. At least 10GB required"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate configuration
validate_configuration() {
    log "Validating configuration..."
    
    # Check if configuration files exist
    local config_files=("configs/strategies.yaml" "configs/market_data.yaml" "configs/risk.yaml")
    for config_file in "${config_files[@]}"; do
        if [[ ! -f "$config_file" ]]; then
            log_error "Configuration file '$config_file' not found"
            exit 1
        fi
    done
    
    # Validate configuration using Python
    if ! python3 -c "
from configs.config_loader import ConfigLoader
try:
    config_loader = ConfigLoader()
    validation_results = config_loader.validate_all_configs()
    for config_name, errors in validation_results.items():
        if errors:
            print(f'Validation errors in {config_name}: {errors}')
            exit(1)
    print('Configuration validation passed')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    exit(1)
"; then
        log_success "Configuration validation passed"
    else
        log_error "Configuration validation failed"
        exit 1
    fi
}

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

build_docker_images() {
    log "Building Docker images..."
    
    # Build base image
    log "Building base image..."
    docker build -t "${DOCKER_REGISTRY}/${PROJECT_NAME}-base:${DOCKER_TAG}" \
        -f docker/Dockerfile.base . || {
        log_error "Failed to build base image"
        exit 1
    }
    
    # Build service images
    local services_to_build=("api" "trading-engine" "market-data" "risk-service" "sentiment-service" "dashboard")
    
    for service in "${services_to_build[@]}"; do
        log "Building $service image..."
        docker build -t "${DOCKER_REGISTRY}/${PROJECT_NAME}-${service}:${DOCKER_TAG}" \
            -f "docker/Dockerfile.${service}" . || {
            log_error "Failed to build $service image"
            exit 1
        }
    done
    
    # Tag images for blue-green deployment
    if [[ "$DEPLOYMENT_TYPE" == "blue-green" ]]; then
        for service in "${services_to_build[@]}"; do
            docker tag "${DOCKER_REGISTRY}/${PROJECT_NAME}-${service}:${DOCKER_TAG}" \
                "${DOCKER_REGISTRY}/${PROJECT_NAME}-${service}:${TARGET_SLOT}"
        done
    fi
    
    log_success "Docker images built successfully"
}

push_docker_images() {
    if [[ "$DOCKER_REGISTRY" != "localhost:5000" ]]; then
        log "Pushing Docker images to registry..."
        
        local services_to_push=("base" "api" "trading-engine" "market-data" "risk-service" "sentiment-service" "dashboard")
        
        for service in "${services_to_push[@]}"; do
            log "Pushing $service image..."
            docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}-${service}:${DOCKER_TAG}" || {
                log_error "Failed to push $service image"
                exit 1
            }
        done
        
        log_success "Docker images pushed successfully"
    else
        log "Using local registry, skipping push"
    fi
}

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

wait_for_database() {
    local db_host="${POSTGRES_HOST:-localhost}"
    local db_port="${POSTGRES_PORT:-5432}"
    local max_attempts=30
    local attempt=1
    
    log "Waiting for database to be ready..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if nc -z "$db_host" "$db_port"; then
            log_success "Database is ready"
            return 0
        fi
        
        log "Database not ready, attempt $attempt/$max_attempts"
        sleep 5
        ((attempt++))
    done
    
    log_error "Database failed to become ready after $max_attempts attempts"
    return 1
}

run_database_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    wait_for_database || exit 1
    
    # Run PostgreSQL migrations
    log "Running PostgreSQL migrations..."
    docker run --rm \
        --network="${PROJECT_NAME}_default" \
        -v "$SCRIPT_DIR/migrations:/migrations" \
        -e POSTGRES_HOST="${POSTGRES_HOST:-postgres}" \
        -e POSTGRES_PORT="${POSTGRES_PORT:-5432}" \
        -e POSTGRES_DB="${POSTGRES_DB:-alpha_discovery}" \
        -e POSTGRES_USER="${POSTGRES_USER:-postgres}" \
        -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
        "${DOCKER_REGISTRY}/${PROJECT_NAME}-base:${DOCKER_TAG}" \
        python /migrations/run_migrations.py || {
        log_error "PostgreSQL migrations failed"
        exit 1
    }
    
    # Run TimescaleDB setup
    log "Setting up TimescaleDB..."
    docker run --rm \
        --network="${PROJECT_NAME}_default" \
        -v "$SCRIPT_DIR/migrations:/migrations" \
        -e TIMESCALE_HOST="${TIMESCALE_HOST:-timescaledb}" \
        -e TIMESCALE_PORT="${TIMESCALE_PORT:-5432}" \
        -e TIMESCALE_DB="${TIMESCALE_DB:-market_data}" \
        -e TIMESCALE_USER="${TIMESCALE_USER:-timescale}" \
        -e TIMESCALE_PASSWORD="${TIMESCALE_PASSWORD}" \
        "${DOCKER_REGISTRY}/${PROJECT_NAME}-base:${DOCKER_TAG}" \
        python /migrations/setup_timescale.py || {
        log_error "TimescaleDB setup failed"
        exit 1
    }
    
    log_success "Database migrations completed successfully"
}

# =============================================================================
# SERVICE MANAGEMENT
# =============================================================================

start_infrastructure_services() {
    log "Starting infrastructure services..."
    
    # Start databases first
    local infrastructure_services=("postgres" "redis" "timescaledb")
    
    for service in "${infrastructure_services[@]}"; do
        log "Starting $service..."
        docker-compose -f "docker/docker-compose.${ENVIRONMENT}.yml" up -d "$service" || {
            log_error "Failed to start $service"
            exit 1
        }
    done
    
    # Wait for services to be ready
    sleep 30
    
    # Verify infrastructure services
    verify_infrastructure_services || exit 1
    
    log_success "Infrastructure services started successfully"
}

start_application_services() {
    log "Starting application services..."
    
    local app_services=("market-data-service" "trading-engine" "risk-service" "sentiment-service")
    
    for service in "${app_services[@]}"; do
        log "Starting $service..."
        docker-compose -f "docker/docker-compose.${ENVIRONMENT}.yml" up -d "$service" || {
            log_error "Failed to start $service"
            exit 1
        }
        
        # Wait a bit between services to avoid resource contention
        sleep 10
    done
    
    log_success "Application services started successfully"
}

start_frontend_services() {
    log "Starting frontend services..."
    
    local frontend_services=("api-gateway" "dashboard")
    
    for service in "${frontend_services[@]}"; do
        log "Starting $service..."
        docker-compose -f "docker/docker-compose.${ENVIRONMENT}.yml" up -d "$service" || {
            log_error "Failed to start $service"
            exit 1
        }
        
        sleep 5
    done
    
    log_success "Frontend services started successfully"
}

start_monitoring_services() {
    log "Starting monitoring services..."
    
    local monitoring_services=("prometheus" "grafana" "alertmanager" "node-exporter")
    
    for service in "${monitoring_services[@]}"; do
        log "Starting $service..."
        docker-compose -f "docker/docker-compose.monitoring.yml" up -d "$service" || {
            log_warning "Failed to start $service, continuing..."
        }
        
        sleep 5
    done
    
    log_success "Monitoring services started successfully"
}

# =============================================================================
# HEALTH CHECKS
# =============================================================================

check_service_health() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking health of $service_name..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            log_success "$service_name is healthy"
            return 0
        fi
        
        log "Health check failed for $service_name, attempt $attempt/$max_attempts"
        sleep "$HEALTH_CHECK_INTERVAL"
        ((attempt++))
    done
    
    log_error "$service_name health check failed after $max_attempts attempts"
    return 1
}

verify_infrastructure_services() {
    log "Verifying infrastructure services..."
    
    # Check PostgreSQL
    if ! docker exec "${PROJECT_NAME}_postgres_1" pg_isready -U "${POSTGRES_USER:-postgres}" > /dev/null 2>&1; then
        log_error "PostgreSQL is not ready"
        return 1
    fi
    
    # Check Redis
    if ! docker exec "${PROJECT_NAME}_redis_1" redis-cli ping > /dev/null 2>&1; then
        log_error "Redis is not ready"
        return 1
    fi
    
    # Check TimescaleDB
    if ! docker exec "${PROJECT_NAME}_timescaledb_1" pg_isready -U "${TIMESCALE_USER:-timescale}" > /dev/null 2>&1; then
        log_error "TimescaleDB is not ready"
        return 1
    fi
    
    log_success "Infrastructure services verified successfully"
    return 0
}

run_comprehensive_health_checks() {
    log "Running comprehensive health checks..."
    
    # Define health check endpoints
    local health_checks=(
        "api-gateway:http://localhost:8000/health"
        "dashboard:http://localhost:8501/_stcore/health"
        "market-data-service:http://localhost:8001/health"
        "trading-engine:http://localhost:8002/health"
        "risk-service:http://localhost:8003/health"
        "sentiment-service:http://localhost:8004/health"
    )
    
    local failed_checks=()
    
    for check in "${health_checks[@]}"; do
        IFS=':' read -r service_name health_url <<< "$check"
        if ! check_service_health "$service_name" "$health_url"; then
            failed_checks+=("$service_name")
        fi
    done
    
    if [[ ${#failed_checks[@]} -gt 0 ]]; then
        log_error "Health checks failed for: ${failed_checks[*]}"
        return 1
    fi
    
    log_success "All health checks passed"
    return 0
}

# =============================================================================
# MONITORING AND ALERTING
# =============================================================================

setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create monitoring configuration
    mkdir -p monitoring/prometheus
    mkdir -p monitoring/grafana
    mkdir -p monitoring/alertmanager
    
    # Generate Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
  
  - job_name: 'alpha-discovery-api'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'
  
  - job_name: 'alpha-discovery-trading'
    static_configs:
      - targets: ['trading-engine:8002']
    metrics_path: '/metrics'
  
  - job_name: 'alpha-discovery-risk'
    static_configs:
      - targets: ['risk-service:8003']
    metrics_path: '/metrics'
EOF

    # Generate alert rules
    cat > monitoring/prometheus/alert_rules.yml << EOF
groups:
  - name: alpha-discovery-alerts
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ \$labels.instance }} is down"
          description: "Service {{ \$labels.instance }} has been down for more than 1 minute"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on {{ \$labels.instance }}"
          description: "Error rate is {{ \$value }} errors per second"
      
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ \$labels.instance }}"
          description: "Memory usage is above 90%"
      
      - alert: TradingEngineDown
        expr: up{job="alpha-discovery-trading"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Trading engine is down"
          description: "Trading engine has been down for more than 30 seconds"
EOF

    # Generate Alertmanager configuration
    cat > monitoring/alertmanager/alertmanager.yml << EOF
global:
  smtp_smarthost: '${SMTP_SERVER:-localhost:587}'
  smtp_from: '${SMTP_FROM:-alerts@alphadiscovery.com}'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: '${ALERT_EMAIL:-admin@alphadiscovery.com}'
        subject: 'Alpha Discovery Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts'
        title: 'Alpha Discovery Alert'
        text: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

    log_success "Monitoring setup completed"
}

configure_log_aggregation() {
    log "Configuring log aggregation..."
    
    # Create log aggregation configuration
    mkdir -p logging/fluentd
    mkdir -p logging/elasticsearch
    
    # Generate Fluentd configuration
    cat > logging/fluentd/fluent.conf << EOF
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<source>
  @type tail
  path /var/log/alpha-discovery/*.log
  pos_file /var/log/fluentd/alpha-discovery.log.pos
  tag alpha-discovery.*
  format json
  time_format %Y-%m-%d %H:%M:%S
</source>

<filter alpha-discovery.**>
  @type record_transformer
  <record>
    hostname #{Socket.gethostname}
    environment ${ENVIRONMENT}
    service_name \${tag_parts[1]}
  </record>
</filter>

<match alpha-discovery.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
  logstash_prefix alpha-discovery
  include_tag_key true
  tag_key @log_name
  flush_interval 10s
</match>

<match **>
  @type stdout
</match>
EOF

    # Configure Docker logging driver
    cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "fluentd",
  "log-opts": {
    "fluentd-address": "localhost:24224",
    "tag": "alpha-discovery.{{.Name}}"
  }
}
EOF

    # Restart Docker daemon to apply logging configuration
    systemctl restart docker || log_warning "Failed to restart Docker daemon"
    
    log_success "Log aggregation configured"
}

# =============================================================================
# BLUE-GREEN DEPLOYMENT
# =============================================================================

setup_blue_green_deployment() {
    log "Setting up blue-green deployment..."
    
    BLUE_GREEN_ENABLED=true
    
    # Create blue-green configuration
    mkdir -p blue-green
    
    # Generate blue environment configuration
    cat > blue-green/docker-compose.blue.yml << EOF
version: '3.8'

services:
  api-gateway-blue:
    image: ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:blue
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=${ENVIRONMENT}
      - SLOT=blue
    networks:
      - alpha-discovery-blue
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api-blue.rule=Host(\`api-blue.alphadiscovery.com\`)"
      - "traefik.http.services.api-blue.loadbalancer.server.port=8000"

  dashboard-blue:
    image: ${DOCKER_REGISTRY}/${PROJECT_NAME}-dashboard:blue
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=${ENVIRONMENT}
      - SLOT=blue
    networks:
      - alpha-discovery-blue

networks:
  alpha-discovery-blue:
    driver: bridge
EOF

    # Generate green environment configuration
    cat > blue-green/docker-compose.green.yml << EOF
version: '3.8'

services:
  api-gateway-green:
    image: ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:green
    ports:
      - "8010:8000"
    environment:
      - ENVIRONMENT=${ENVIRONMENT}
      - SLOT=green
    networks:
      - alpha-discovery-green
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api-green.rule=Host(\`api-green.alphadiscovery.com\`)"
      - "traefik.http.services.api-green.loadbalancer.server.port=8000"

  dashboard-green:
    image: ${DOCKER_REGISTRY}/${PROJECT_NAME}-dashboard:green
    ports:
      - "8511:8501"
    environment:
      - ENVIRONMENT=${ENVIRONMENT}
      - SLOT=green
    networks:
      - alpha-discovery-green

networks:
  alpha-discovery-green:
    driver: bridge
EOF

    # Generate load balancer configuration
    cat > blue-green/traefik.yml << EOF
api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@alphadiscovery.com
      storage: acme.json
      httpChallenge:
        entryPoint: web
EOF

    log_success "Blue-green deployment setup completed"
}

deploy_to_slot() {
    local slot=$1
    log "Deploying to $slot slot..."
    
    # Start services in the target slot
    docker-compose -f "blue-green/docker-compose.${slot}.yml" up -d || {
        log_error "Failed to deploy to $slot slot"
        return 1
    }
    
    # Wait for services to be ready
    sleep 30
    
    # Run health checks on the target slot
    local slot_port
    if [[ "$slot" == "blue" ]]; then
        slot_port="8000"
    else
        slot_port="8010"
    fi
    
    if ! check_service_health "api-gateway-$slot" "http://localhost:$slot_port/health"; then
        log_error "Health check failed for $slot slot"
        return 1
    fi
    
    log_success "Successfully deployed to $slot slot"
    return 0
}

switch_traffic() {
    local from_slot=$1
    local to_slot=$2
    
    log "Switching traffic from $from_slot to $to_slot..."
    
    # Update load balancer configuration
    cat > blue-green/traefik-dynamic.yml << EOF
http:
  routers:
    api-router:
      rule: "Host(\`api.alphadiscovery.com\`)"
      service: "api-$to_slot"
      
  services:
    api-$to_slot:
      loadBalancer:
        servers:
          - url: "http://api-gateway-$to_slot:8000"
EOF

    # Apply the new configuration
    docker exec traefik kill -USR1 1 || {
        log_error "Failed to reload load balancer configuration"
        return 1
    }
    
    # Wait for traffic switch to take effect
    sleep 10
    
    # Verify traffic is flowing to the new slot
    if ! curl -f -s "http://api.alphadiscovery.com/health" > /dev/null; then
        log_error "Traffic switch verification failed"
        return 1
    fi
    
    log_success "Traffic switched successfully from $from_slot to $to_slot"
    return 0
}

# =============================================================================
# ROLLBACK FUNCTIONALITY
# =============================================================================

create_deployment_snapshot() {
    log "Creating deployment snapshot..."
    
    local snapshot_dir="/var/backups/${PROJECT_NAME}/snapshots/${DEPLOYMENT_ID}"
    mkdir -p "$snapshot_dir"
    
    # Save current configuration
    cp -r configs/ "$snapshot_dir/"
    
    # Save current Docker Compose files
    cp docker/*.yml "$snapshot_dir/"
    
    # Save current environment variables
    env | grep -E '^(POSTGRES_|REDIS_|JWT_|DOCKER_)' > "$snapshot_dir/environment.env"
    
    # Save current service states
    docker-compose ps > "$snapshot_dir/services.state"
    
    # Save database schema
    docker exec "${PROJECT_NAME}_postgres_1" pg_dump -s -U "${POSTGRES_USER:-postgres}" "${POSTGRES_DB:-alpha_discovery}" > "$snapshot_dir/database_schema.sql"
    
    log_success "Deployment snapshot created: $snapshot_dir"
    echo "$snapshot_dir" > /tmp/last_snapshot.txt
}

rollback_deployment() {
    log "Initiating deployment rollback..."
    
    local snapshot_dir
    if [[ -f /tmp/last_snapshot.txt ]]; then
        snapshot_dir=$(cat /tmp/last_snapshot.txt)
    else
        log_error "No snapshot found for rollback"
        return 1
    fi
    
    if [[ ! -d "$snapshot_dir" ]]; then
        log_error "Snapshot directory not found: $snapshot_dir"
        return 1
    fi
    
    log "Rolling back to snapshot: $snapshot_dir"
    
    # Stop current services
    docker-compose -f "docker/docker-compose.${ENVIRONMENT}.yml" down || true
    
    # Restore configuration
    cp -r "$snapshot_dir/configs/" .
    
    # Restore Docker Compose files
    cp "$snapshot_dir/"*.yml docker/
    
    # Restore environment variables
    source "$snapshot_dir/environment.env"
    
    # Restore services
    docker-compose -f "docker/docker-compose.${ENVIRONMENT}.yml" up -d || {
        log_error "Failed to restore services during rollback"
        return 1
    }
    
    # Wait for services to be ready
    sleep 60
    
    # Verify rollback
    if run_comprehensive_health_checks; then
        log_success "Rollback completed successfully"
        return 0
    else
        log_error "Rollback verification failed"
        return 1
    fi
}

# =============================================================================
# MAIN DEPLOYMENT FUNCTIONS
# =============================================================================

run_rolling_deployment() {
    log "Starting rolling deployment..."
    
    # Create deployment snapshot
    create_deployment_snapshot
    
    # Build and push images
    build_docker_images
    push_docker_images
    
    # Start infrastructure services
    start_infrastructure_services
    
    # Run database migrations
    run_database_migrations
    
    # Start application services
    start_application_services
    
    # Start frontend services
    start_frontend_services
    
    # Start monitoring services
    start_monitoring_services
    
    # Run comprehensive health checks
    run_comprehensive_health_checks || {
        log_error "Health checks failed, initiating rollback"
        rollback_deployment
        exit 1
    }
    
    log_success "Rolling deployment completed successfully"
}

run_blue_green_deployment() {
    log "Starting blue-green deployment..."
    
    # Setup blue-green deployment
    setup_blue_green_deployment
    
    # Create deployment snapshot
    create_deployment_snapshot
    
    # Build and push images
    build_docker_images
    push_docker_images
    
    # Deploy to target slot
    deploy_to_slot "$TARGET_SLOT" || {
        log_error "Deployment to $TARGET_SLOT slot failed"
        rollback_deployment
        exit 1
    }
    
    # Switch traffic to target slot
    switch_traffic "$CURRENT_SLOT" "$TARGET_SLOT" || {
        log_error "Traffic switch failed"
        rollback_deployment
        exit 1
    }
    
    # Run final health checks
    run_comprehensive_health_checks || {
        log_error "Final health checks failed"
        switch_traffic "$TARGET_SLOT" "$CURRENT_SLOT"
        rollback_deployment
        exit 1
    }
    
    # Clean up old slot
    log "Cleaning up $CURRENT_SLOT slot..."
    docker-compose -f "blue-green/docker-compose.${CURRENT_SLOT}.yml" down || true
    
    log_success "Blue-green deployment completed successfully"
}

# =============================================================================
# MAIN SCRIPT EXECUTION
# =============================================================================

main() {
    log "Starting Alpha Discovery deployment (ID: $DEPLOYMENT_ID)"
    log "Environment: $ENVIRONMENT"
    log "Deployment Type: $DEPLOYMENT_TYPE"
    
    # Setup logging
    setup_logging
    
    # Check prerequisites
    check_prerequisites
    
    # Validate configuration
    validate_configuration
    
    # Setup monitoring and logging
    setup_monitoring
    configure_log_aggregation
    
    # Run deployment based on type
    case "$DEPLOYMENT_TYPE" in
        "rolling")
            run_rolling_deployment
            ;;
        "blue-green")
            run_blue_green_deployment
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Final success message
    log_success "Deployment completed successfully!"
    log "Deployment ID: $DEPLOYMENT_ID"
    log "Services are available at:"
    log "  - API: http://localhost:8000"
    log "  - Dashboard: http://localhost:8501"
    log "  - Monitoring: http://localhost:3000"
    log "  - Logs: $LOG_FILE"
    
    # Cleanup
    cleanup
}

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

show_help() {
    cat << EOF
Alpha Discovery Deployment Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
  deploy          Run full deployment (default)
  rollback        Rollback to previous deployment
  health-check    Run health checks only
  build           Build Docker images only
  migrate         Run database migrations only
  
Options:
  -e, --environment ENV    Set environment (development, staging, production)
  -t, --type TYPE         Set deployment type (rolling, blue-green)
  -r, --registry URL      Set Docker registry URL
  -v, --verbose           Enable verbose logging
  -h, --help              Show this help message

Environment Variables:
  ENVIRONMENT             Deployment environment
  DEPLOYMENT_TYPE         Deployment type
  DOCKER_REGISTRY         Docker registry URL
  DOCKER_TAG              Docker image tag
  
Examples:
  $0 deploy                              # Rolling deployment to development
  $0 -e production -t blue-green deploy  # Blue-green deployment to production
  $0 rollback                            # Rollback to previous deployment
  $0 health-check                        # Run health checks only

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        deploy)
            COMMAND="deploy"
            shift
            ;;
        rollback)
            COMMAND="rollback"
            shift
            ;;
        health-check)
            COMMAND="health-check"
            shift
            ;;
        build)
            COMMAND="build"
            shift
            ;;
        migrate)
            COMMAND="migrate"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute command
case "${COMMAND:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        setup_logging
        rollback_deployment
        ;;
    "health-check")
        setup_logging
        run_comprehensive_health_checks
        ;;
    "build")
        setup_logging
        build_docker_images
        ;;
    "migrate")
        setup_logging
        run_database_migrations
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac 