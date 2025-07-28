# 🚀 Complete Alpha Discovery Deployment Walkthrough

**For Complete Beginners** - This guide assumes you've never used RunPod, Docker, or deployed a distributed system before.

## 📋 What We're Building

- **Main Application**: Runs on your local machine/server (lightweight, fast startup)
- **ML Server**: Runs on RunPod GPU cloud (handles all heavy AI/ML processing)
- **Communication**: Main app talks to ML server over the internet

---

## 🎯 **PHASE 1: RunPod Account & GPU Server Setup**

### Step 1.1: Create RunPod Account

1. **Go to RunPod**: Visit [https://runpod.io](https://runpod.io)
2. **Sign Up**: Click "Sign Up" and create an account
3. **Verify Email**: Check your email and verify your account
4. **Add Payment**: Add a credit card (you'll only pay for what you use)

### Step 1.2: Create GPU Pod

1. **Login to RunPod Console**: Go to [https://www.runpod.io/console](https://www.runpod.io/console)
2. **Click "Deploy"** in the top menu
3. **Choose "Pods"** (not Templates)
4. **Select GPU**: 
   - **Recommended**: RTX 4090 (best price/performance)
   - **Alternative**: RTX A5000 or RTX 3090
   - **Budget**: RTX 3080 (will work but slower)
5. **Choose Template**: Select "RunPod PyTorch 2.2.0"
6. **Configure Pod**:
   ```
   Pod Name: alpha-discovery-ml
   Container Disk: 50 GB (minimum)
   Volume Disk: 20 GB (optional but recommended)
   Expose HTTP Ports: 8001,8002
   Expose TCP Port: 22 (for SSH)
   ```
7. **Deploy Pod**: Click "Deploy On-Demand" (cheaper than Spot for beginners)

### Step 1.3: Wait for Pod to Start

- **Status Check**: Wait for status to change from "Starting" to "Running"
- **Time**: Usually takes 2-5 minutes
- **Get Connection Info**: Once running, note down:
  - **Pod ID**: (e.g., `abc123def456`)
  - **Public IP**: (e.g., `123.456.789.101`)
  - **SSH Port**: (e.g., `12345`)
  - **HTTP Ports**: 8001 and 8002

---

## 🎯 **PHASE 2: Deploy ML Server to RunPod**

### Step 2.1: Connect to Your RunPod

**Option A: Web Terminal (Easiest)**
1. In RunPod console, click "Connect" on your pod
2. Click "Start Web Terminal"
3. Wait for terminal to load

**Option B: SSH (Advanced)**
```bash
# Replace with your actual pod details
ssh root@123.456.789.101 -p 12345
```

### Step 2.2: Prepare the ML Server Code

In your RunPod terminal, run these commands **one by one**:

```bash
# Update system
apt update && apt upgrade -y

# Install git and other tools
apt install -y git curl wget nano

# Clone your project (replace with your actual repo)
git clone https://github.com/your-username/alpha-discovery.git
cd alpha-discovery

# Verify we have the right files
ls -la Dockerfile.mcp
ls -la src/mcp/
```

### Step 2.3: Build the ML Server Docker Image

```bash
# Build the Docker image (this will take 10-15 minutes)
docker build -f Dockerfile.mcp -t alpha-discovery-mcp .

# Check if build was successful
docker images | grep alpha-discovery-mcp
```

**What's Happening**: Docker is downloading and installing all the ML libraries and pre-caching the AI models.

### Step 2.4: Run the ML Server

```bash
# Create a directory for model cache
mkdir -p /app/cache

# Run the MCP server container
docker run -d \
  --name alpha-mcp-server \
  --gpus all \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /app/cache:/app/cache \
  --restart unless-stopped \
  alpha-discovery-mcp

# Check if it's running
docker ps

# Check logs to make sure it started correctly
docker logs alpha-mcp-server
```

### Step 2.5: Test Your ML Server

```bash
# Test if the server is responding
curl http://localhost:8002/ml-health

# You should see something like:
# {"status": "healthy", "models_loaded": 15, "gpu_available": true}
```

### Step 2.6: Get Your Public URL

Your ML server is now accessible at:
```
http://YOUR_POD_PUBLIC_IP:8002
```

**Example**: If your pod IP is `123.456.789.101`, your ML server URL is:
```
http://123.456.789.101:8002
```

**⚠️ IMPORTANT**: Write this URL down! You'll need it for the main application.

---

## 🎯 **PHASE 3: Set Up Your Local Development Environment**

### Step 3.1: Prepare Your Local Machine

**On Windows**:
```powershell
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/
# Install Git from https://git-scm.com/download/win
# Install Python from https://python.org (3.9 or higher)
```

**On macOS**:
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git and Python
brew install git python@3.9
```

**On Linux (Ubuntu)**:
```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose git python3 python3-pip

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add yourself to docker group
sudo usermod -aG docker $USER

# Log out and log back in for group changes to take effect
```

### Step 3.2: Clone and Set Up Project

```bash
# Clone the project (if you haven't already)
git clone https://github.com/your-username/alpha-discovery.git
cd alpha-discovery

# Create environment file
cp configs/env.template .env

# Edit the environment file
nano .env  # or use your preferred editor
```

### Step 3.3: Configure Environment Variables

Edit your `.env` file and add these critical settings:

```bash
# Replace with your actual RunPod IP
MCP_SERVER_URL=http://123.456.789.101:8002

# Database settings
POSTGRES_PASSWORD=your_secure_postgres_password
REDIS_PASSWORD=your_secure_redis_password

# Optional: Add Sentry for error tracking
SENTRY_DSN=your_sentry_dsn_if_you_have_one

# Grafana settings
GRAFANA_USER=admin
GRAFANA_PASSWORD=your_secure_grafana_password
```

**⚠️ CRITICAL**: Replace `123.456.789.101` with your actual RunPod IP!

---

## 🎯 **PHASE 4: Deploy Main Application Services**

### Step 4.1: Test ML Server Connection

First, let's make sure your local machine can reach your RunPod ML server:

```bash
# Test connection (replace with your actual IP)
curl http://123.456.789.101:8002/ml-health

# If this works, you should see:
# {"status": "healthy", "models_loaded": 15, "gpu_available": true}
```

**If this doesn't work**:
- Check your RunPod pod is still running
- Verify the IP address is correct
- Make sure ports 8001 and 8002 are exposed in RunPod

### Step 4.2: Build and Start Main Services

```bash
# Make sure you're in the project directory
cd alpha-discovery

# Build the main application images
docker-compose -f docker-compose.production.yml build

# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check if all services are running
docker-compose -f docker-compose.production.yml ps
```

### Step 4.3: Wait for Services to Start

```bash
# Watch the logs to see startup progress
docker-compose -f docker-compose.production.yml logs -f

# Press Ctrl+C to stop watching logs
```

**What you should see**:
- `postgres` should show "database system is ready to accept connections"
- `redis` should show "Ready to accept connections"
- `api` should show "Application startup complete"
- `worker` should show "Worker started successfully"

### Step 4.4: Verify Everything is Working

```bash
# Check API health
curl http://localhost:8000/health

# Check if API can reach ML server
curl http://localhost:8000/ml-health

# Check Grafana dashboard
# Open browser to http://localhost:3000
# Login: admin / your_grafana_password

# Check Prometheus metrics
# Open browser to http://localhost:9090
```

---

## 🎯 **PHASE 5: Final Testing & Verification**

### Step 5.1: Test the Complete System

Let's run a comprehensive test to make sure everything is connected:

```bash
# Create a test script
cat > test_system.py << 'EOF'
import requests
import json
import time

def test_system():
    print("🧪 Testing Alpha Discovery System...")
    
    # Test 1: Main API Health
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Main API is healthy")
        else:
            print("❌ Main API failed")
            return False
    except Exception as e:
        print(f"❌ Cannot reach main API: {e}")
        return False
    
    # Test 2: ML Server Connection
    try:
        response = requests.get("http://localhost:8000/ml-health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ML Server connected - {data.get('models_loaded', 0)} models loaded")
        else:
            print("❌ ML Server connection failed")
            return False
    except Exception as e:
        print(f"❌ Cannot reach ML server through API: {e}")
        return False
    
    # Test 3: Database Connection
    try:
        response = requests.get("http://localhost:8000/db-health")
        if response.status_code == 200:
            print("✅ Database connection working")
        else:
            print("⚠️ Database health check not available")
    except Exception as e:
        print(f"⚠️ Database test failed: {e}")
    
    print("\n🎉 System Test Complete!")
    print("🚀 Alpha Discovery is ready for trading!")
    return True

if __name__ == "__main__":
    test_system()
EOF

# Run the test
python3 test_system.py
```

### Step 5.2: Monitor System Performance

```bash
# Check resource usage
docker stats

# Check logs for any errors
docker-compose -f docker-compose.production.yml logs --tail=50

# Check specific service logs
docker-compose -f docker-compose.production.yml logs api
docker-compose -f docker-compose.production.yml logs worker
```

---

## 🎯 **PHASE 6: Daily Operations**

### Starting the System

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps
```

### Stopping the System

```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# Stop and remove volumes (careful!)
docker-compose -f docker-compose.production.yml down -v
```

### Checking Logs

```bash
# View all logs
docker-compose -f docker-compose.production.yml logs

# Follow logs in real-time
docker-compose -f docker-compose.production.yml logs -f

# Check specific service
docker-compose -f docker-compose.production.yml logs api
```

### Managing Your RunPod

- **Monitor Usage**: Check RunPod console for GPU usage and costs
- **Stop Pod**: When not trading, stop your pod to save money
- **Start Pod**: Restart when you need to trade (takes 2-3 minutes)

---

## 🚨 **Troubleshooting Common Issues**

### Issue 1: "Cannot connect to ML server"

**Symptoms**: API shows ML server as unavailable

**Solutions**:
1. Check if RunPod pod is still running
2. Verify IP address in `.env` file
3. Test direct connection: `curl http://YOUR_POD_IP:8002/ml-health`
4. Check RunPod firewall settings

### Issue 2: "Database connection failed"

**Symptoms**: API won't start, database errors in logs

**Solutions**:
```bash
# Restart database
docker-compose -f docker-compose.production.yml restart postgres

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres

# Reset database (WARNING: deletes all data)
docker-compose -f docker-compose.production.yml down -v
docker-compose -f docker-compose.production.yml up -d
```

### Issue 3: "Out of memory" on RunPod

**Symptoms**: ML server crashes, CUDA out of memory errors

**Solutions**:
1. Restart RunPod container: `docker restart alpha-mcp-server`
2. Use smaller batch sizes in ML processing
3. Upgrade to a GPU with more VRAM

### Issue 4: High RunPod costs

**Solutions**:
1. Stop pod when not trading
2. Use Spot instances (cheaper but can be interrupted)
3. Monitor usage in RunPod console
4. Set spending limits

---

## 💰 **Cost Management**

### RunPod Costs (Approximate)
- **RTX 4090**: $0.50-0.70/hour
- **RTX A5000**: $0.80-1.20/hour
- **RTX 3080**: $0.30-0.50/hour

### Cost Optimization Tips
1. **Stop when not trading**: Only run during market hours
2. **Use Spot instances**: 50-70% cheaper but can be interrupted
3. **Monitor usage**: Set up alerts in RunPod
4. **Batch processing**: Process multiple requests together

---

## 🎉 **Congratulations!**

You now have a fully operational Alpha Discovery trading system with:

✅ **GPU-accelerated ML server** running on RunPod  
✅ **Lightweight main application** with fast startup  
✅ **Real-time monitoring** with Grafana and Prometheus  
✅ **Production-ready architecture** with health checks  
✅ **Cost-optimized setup** paying only for GPU when needed  

**Your system is ready for algorithmic trading!** 🚀

---

## 📞 **Getting Help**

If you run into issues:
1. Check the logs first: `docker-compose logs`
2. Verify all services are running: `docker-compose ps`
3. Test ML server connection: `curl http://YOUR_POD_IP:8002/ml-health`
4. Check RunPod console for pod status
5. Review this guide step-by-step

**Remember**: The most common issue is incorrect IP address in the `.env` file! 