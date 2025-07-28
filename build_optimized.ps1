# Optimized Docker Build Script for Alpha Discovery
Write-Host "Cleaning up Docker system..." -ForegroundColor Yellow

# Stop and remove existing containers
docker compose down --volumes --remove-orphans

# Clean up Docker system (images, containers, networks, volumes)
docker system prune -a --volumes -f

# Clean up build cache
docker builder prune -a -f

Write-Host "Building optimized Docker images..." -ForegroundColor Green

# Build with optimizations
docker compose build --no-cache --parallel

Write-Host "Build completed!" -ForegroundColor Green
Write-Host "Starting services..." -ForegroundColor Blue

# Start services
docker compose up -d

Write-Host "Alpha Discovery is starting up!" -ForegroundColor Green
Write-Host "Monitor services with: docker compose ps" -ForegroundColor Cyan
Write-Host "View logs with: docker compose logs -f" -ForegroundColor Cyan 