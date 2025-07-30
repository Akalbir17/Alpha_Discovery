# 🚀 RunPod Deployment Guide for Alpha Discovery

## Overview
This guide will help you deploy the Alpha Discovery MCP server on RunPod.io with GPU acceleration for optimal ML performance.

## 📋 Prerequisites
- RunPod account (sign up at https://runpod.io)
- GitHub repository access: `https://github.com/Akalbir17/Alpha_Discovery.git`
- Basic familiarity with terminal commands

## 🎯 Step-by-Step Deployment

### Step 1: Launch RunPod Instance

1. **Go to RunPod.io** and sign in
2. **Click "Deploy"** → **"GPU Pods"**
3. **Select GPU Template:**
   - **Recommended**: RTX 4090 (24GB VRAM) - Best price/performance
   - **Premium**: A100 (40GB/80GB VRAM) - Maximum performance
4. **Choose Template:**
   - Select **"PyTorch"** template
   - Or use: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
5. **Configure Pod:**
   - **Container Disk**: 50GB minimum (recommended: 100GB)
   - **Volume**: Optional, but recommended for persistent storage
   - **Ports**: Expose ports 8001, 8002, 22 (SSH)
6. **Deploy Pod**

### Step 2: Connect to Your Pod

1. **Wait for pod to be "Running"**
2. **Connect via Web Terminal** (easiest) or SSH
3. **For Web Terminal**: Click "Connect" → "Start Web Terminal"

### Step 3: Install Dependencies and Setup

```bash
# Update system
apt update && apt upgrade -y

# Install essential tools
apt install -y git curl wget build-essential

# Install Docker (if not already installed)
apt install -y docker.io
systemctl start docker
systemctl enable docker

# Start Docker daemon in background
dockerd > /dev/null 2>&1 &
sleep 5
```

### Step 4: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/Akalbir17/Alpha_Discovery.git
cd Alpha_Discovery

# Verify files are present
ls -la
```

### Step 5: Build the GPU-Optimized Container

```bash
# Build the MCP server container
docker build -f Dockerfile.mcp -t alpha-discovery-mcp:latest .

# This will take 10-15 minutes as it downloads ML models
# Watch for successful model caching messages
```

### Step 6: Run the MCP Server

```bash
# Run with GPU access and port mapping
docker run -d \
  --name alpha-discovery-mcp \
  --gpus all \
  -p 8001:8001 \
  -p 8002:8002 \
  -e CUDA_VISIBLE_DEVICES=0 \
  --restart unless-stopped \
  alpha-discovery-mcp:latest

# Check if container is running
docker ps

# View logs to ensure successful startup
docker logs -f alpha-discovery-mcp
```

### Step 7: Verify Deployment

```bash
# Test HTTP ML endpoint
curl http://localhost:8002/ml-health

# Test WebSocket MCP endpoint (should return connection info)
curl http://localhost:8001/health

# Check GPU usage
nvidia-smi
```

## 🌐 Access Your Services

Once deployed, your services will be available at:

- **HTTP ML API**: `http://[POD_IP]:8002`
- **WebSocket MCP**: `ws://[POD_IP]:8001`
- **Health Check**: `http://[POD_IP]:8002/ml-health`

**Find your Pod IP**: In RunPod dashboard → Your Pod → "Connect" → Copy the IP address

## 🔧 Configuration for Production

### Environment Variables
Create a `.env` file in the container or pass environment variables:

```bash
# Optional: Create environment file
docker exec alpha-discovery-mcp bash -c 'cat > /app/.env << EOF
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8001
ML_API_HOST=0.0.0.0
ML_API_PORT=8002
CUDA_VISIBLE_DEVICES=0
TRANSFORMERS_CACHE=/app/cache/transformers
HF_HOME=/app/cache/huggingface
EOF'

# Restart container to apply changes
docker restart alpha-discovery-mcp
```

### Persistent Storage (Recommended)
To keep models cached between restarts:

```bash
# Create volume for model cache
docker volume create alpha-models-cache

# Run with persistent volume
docker run -d \
  --name alpha-discovery-mcp \
  --gpus all \
  -p 8001:8001 \
  -p 8002:8002 \
  -v alpha-models-cache:/app/cache \
  --restart unless-stopped \
  alpha-discovery-mcp:latest
```

## 🚨 Troubleshooting

### Common Issues:

1. **Docker not found**:
   ```bash
   apt update && apt install -y docker.io
   dockerd > /dev/null 2>&1 &
   ```

2. **CUDA version mismatch**:
   ```bash
   nvidia-smi  # Check CUDA version
   # Use compatible PyTorch base image
   ```

3. **Out of memory**:
   ```bash
   # Check GPU memory
   nvidia-smi
   # Reduce batch sizes or use smaller models
   ```

4. **Port conflicts**:
   ```bash
   # Check what's using ports
   netstat -tulpn | grep :8001
   netstat -tulpn | grep :8002
   ```

5. **Container won't start**:
   ```bash
   # Check logs
   docker logs alpha-discovery-mcp
   # Check container status
   docker ps -a
   ```

## 📊 Monitoring and Maintenance

### Check System Resources:
```bash
# GPU usage
nvidia-smi

# Container stats
docker stats alpha-discovery-mcp

# Disk usage
df -h

# Memory usage
free -h
```

### Update Deployment:
```bash
# Pull latest code
cd Alpha_Discovery
git pull origin main

# Rebuild container
docker build -f Dockerfile.mcp -t alpha-discovery-mcp:latest .

# Restart with new image
docker stop alpha-discovery-mcp
docker rm alpha-discovery-mcp
docker run -d --name alpha-discovery-mcp --gpus all -p 8001:8001 -p 8002:8002 alpha-discovery-mcp:latest
```

## 💰 Cost Optimization

- **Use Spot Instances** for development (cheaper but can be interrupted)
- **Stop pods** when not in use
- **Use Community Cloud** for lower rates
- **Monitor usage** in RunPod dashboard

## 🔗 Next Steps

1. **Configure your main application** to connect to: `http://[POD_IP]:8002`
2. **Set MCP_SERVER_URL** environment variable in your main app
3. **Test ML endpoints** with your trading strategies
4. **Monitor performance** and scale as needed

## 📞 Support

- **RunPod Documentation**: https://docs.runpod.io
- **Alpha Discovery Issues**: https://github.com/Akalbir17/Alpha_Discovery/issues
- **Community Discord**: Check RunPod community for help

---

🎉 **Congratulations!** Your GPU-accelerated Alpha Discovery MCP server is now running on RunPod!