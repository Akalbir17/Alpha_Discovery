# Multi-stage Dockerfile for Alpha Discovery Platform
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building with retry logic
RUN apt-get update && \
    for i in 1 2 3; do \
        apt-get install -y \
            build-essential \
            curl \
            git \
            libpq-dev \
            gcc \
            g++ \
            pkg-config \
            libyaml-dev \
            && break || sleep 30; \
    done && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with specific handling for problematic packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel && \
    pip install --no-cache-dir Cython && \
    pip install --no-cache-dir pyyaml==6.0.2 --no-build-isolation && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies with retry logic
RUN apt-get update && \
    for i in 1 2 3; do \
        apt-get install -y \
            libpq5 \
            curl \
            libyaml-0-2 \
            && break || sleep 30; \
    done && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r alpha && useradd -r -g alpha alpha

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/research && \
    chown -R alpha:alpha /app

# Switch to non-root user
USER alpha

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "run_alpha_discovery.py"] 