# 📋 Alpha Discovery Deployment Checklist

**Print this out and check off each step as you complete it!**

---

## ☑️ **PHASE 1: RunPod Setup (15 minutes)**

### RunPod Account
- [ ] Create account at [runpod.io](https://runpod.io)
- [ ] Verify email address
- [ ] Add credit card for billing

### Create GPU Pod
- [ ] Go to RunPod Console → Deploy → Pods
- [ ] Select GPU: RTX 4090 (recommended) or RTX A5000
- [ ] Choose Template: "RunPod PyTorch 2.2.0"
- [ ] Set Pod Name: `alpha-discovery-ml`
- [ ] Set Container Disk: 50 GB
- [ ] Expose HTTP Ports: `8001,8002`
- [ ] Expose TCP Port: `22`
- [ ] Click "Deploy On-Demand"
- [ ] Wait for status: "Starting" → "Running" (2-5 minutes)

### Record Pod Information
Write down these details:
- [ ] Pod ID: `_________________`
- [ ] Public IP: `_________________`
- [ ] SSH Port: `_________________`

---

## ☑️ **PHASE 2: Deploy ML Server (20 minutes)**

### Connect to RunPod
- [ ] In RunPod console, click "Connect" on your pod
- [ ] Click "Start Web Terminal"
- [ ] Wait for terminal to load

### Install Dependencies
Copy/paste these commands **one at a time**:

```bash
# Update system
apt update && apt upgrade -y
```
- [ ] ✅ System updated

```bash
# Install tools
apt install -y git curl wget nano
```
- [ ] ✅ Tools installed

```bash
# Clone project (replace with your repo URL)
git clone https://github.com/your-username/alpha-discovery.git
cd alpha-discovery
```
- [ ] ✅ Project cloned

### Build ML Server
```bash
# Build Docker image (takes 10-15 minutes)
docker build -f Dockerfile.mcp -t alpha-discovery-mcp .
```
- [ ] ✅ Docker image built (no errors)

### Run ML Server
```bash
# Create cache directory
mkdir -p /app/cache

# Run the server
docker run -d \
  --name alpha-mcp-server \
  --gpus all \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /app/cache:/app/cache \
  --restart unless-stopped \
  alpha-discovery-mcp
```
- [ ] ✅ Container started

### Test ML Server
```bash
# Check if running
docker ps

# Test health endpoint
curl http://localhost:8002/ml-health
```
- [ ] ✅ Health check returns: `{"status": "healthy", "models_loaded": 15, "gpu_available": true}`

### Record ML Server URL
Your ML Server URL: `http://YOUR_POD_IP:8002`
- [ ] My ML Server URL: `http://_________________:8002`

---

## ☑️ **PHASE 3: Local Environment Setup (10 minutes)**

### Install Requirements
**Windows:**
- [ ] Install Docker Desktop from docker.com
- [ ] Install Git from git-scm.com
- [ ] Restart computer if required

**Mac:**
- [ ] Install Docker Desktop from docker.com
- [ ] Install Git: `brew install git`

**Linux:**
```bash
sudo apt install -y docker.io docker-compose git python3
sudo systemctl start docker
sudo usermod -aG docker $USER
```
- [ ] Log out and log back in

### Get Project Code
```bash
# Clone project locally
git clone https://github.com/your-username/alpha-discovery.git
cd alpha-discovery
```
- [ ] ✅ Project cloned locally

### Configure Environment
```bash
# Create environment file
cp configs/env.template .env

# Edit the file (use notepad on Windows, nano on Linux/Mac)
nano .env
```
- [ ] ✅ .env file created

### Update .env File
Edit your `.env` file with these values:
```bash
# CRITICAL: Replace with your actual RunPod IP
MCP_SERVER_URL=http://YOUR_POD_IP:8002

# Set secure passwords
POSTGRES_PASSWORD=your_secure_postgres_password
REDIS_PASSWORD=your_secure_redis_password
GRAFANA_PASSWORD=your_secure_grafana_password
```
- [ ] ✅ MCP_SERVER_URL set to: `http://_________________:8002`
- [ ] ✅ Passwords set

---

## ☑️ **PHASE 4: Deploy Main Application (15 minutes)**

### Test ML Server Connection
```bash
# Test from your local machine (replace IP)
curl http://YOUR_POD_IP:8002/ml-health
```
- [ ] ✅ Can reach ML server from local machine

### Build and Start Services
```bash
# Build images
docker-compose -f docker-compose.production.yml build
```
- [ ] ✅ Images built successfully

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d
```
- [ ] ✅ Services started

### Wait for Startup (2-3 minutes)
```bash
# Watch logs
docker-compose -f docker-compose.production.yml logs -f
```
Look for these messages:
- [ ] ✅ PostgreSQL: "database system is ready to accept connections"
- [ ] ✅ Redis: "Ready to accept connections"
- [ ] ✅ API: "Application startup complete"

Press `Ctrl+C` to stop watching logs.

---

## ☑️ **PHASE 5: Final Verification (5 minutes)**

### Test All Components
```bash
# Test API health
curl http://localhost:8000/health
```
- [ ] ✅ API health check passes

```bash
# Test ML connection through API
curl http://localhost:8000/ml-health
```
- [ ] ✅ API can reach ML server

### Run Verification Script
```bash
python3 verify_deployment.py
```
- [ ] ✅ All tests pass

### Open Dashboards
- [ ] Grafana: Open browser to `http://localhost:3000` (admin/your_password)
- [ ] Prometheus: Open browser to `http://localhost:9090`

---

## ☑️ **PHASE 6: You're Done! 🎉**

### Final Checklist
- [ ] ✅ RunPod ML server is running
- [ ] ✅ Local services are healthy
- [ ] ✅ API can connect to ML server
- [ ] ✅ Dashboards are accessible
- [ ] ✅ No error messages in logs

### Your System URLs
- **Main API**: `http://localhost:8000`
- **ML Server**: `http://YOUR_POD_IP:8002`
- **Grafana**: `http://localhost:3000`
- **Prometheus**: `http://localhost:9090`

---

## 🚨 **If Something Goes Wrong**

### Most Common Issues:
1. **"Cannot connect to ML server"**
   - [ ] Check RunPod pod is still running
   - [ ] Verify IP address in .env file
   - [ ] Test: `curl http://YOUR_POD_IP:8002/ml-health`

2. **"Database connection failed"**
   - [ ] Restart: `docker-compose -f docker-compose.production.yml restart postgres`

3. **"Service won't start"**
   - [ ] Check logs: `docker-compose -f docker-compose.production.yml logs SERVICE_NAME`

### Get Help
- [ ] Read `COMPLETE_DEPLOYMENT_WALKTHROUGH.md` for detailed troubleshooting
- [ ] Check all services: `docker-compose -f docker-compose.production.yml ps`
- [ ] View logs: `docker-compose -f docker-compose.production.yml logs`

---

## 💰 **Managing Costs**

### Daily Operations
- [ ] **Stop RunPod when not trading**: Saves $12-17/day
- [ ] **Monitor usage**: Check RunPod console regularly
- [ ] **Set spending limits**: In RunPod account settings

### Starting/Stopping
**To stop everything:**
```bash
# Stop local services
docker-compose -f docker-compose.production.yml down

# Stop RunPod (in RunPod console)
Click "Stop" on your pod
```

**To start everything:**
```bash
# Start RunPod (in RunPod console)
Click "Start" on your pod (takes 2-3 minutes)

# Start local services
docker-compose -f docker-compose.production.yml up -d
```

---

## 🎯 **SUCCESS! Your Alpha Discovery system is now running!**

**What you have:**
- ✅ GPU-accelerated ML server on RunPod
- ✅ Lightweight main application (fast startup)
- ✅ Real-time monitoring and dashboards
- ✅ Production-ready architecture
- ✅ Cost-optimized setup

**You're ready to start algorithmic trading!** 🚀 