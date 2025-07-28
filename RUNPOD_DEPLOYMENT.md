# 🚀 RunPod Deployment Guide for Alpha Discovery MCP Server

This guide provides step-by-step instructions for deploying the GPU-accelerated Machine Learning (ML) services of the Alpha Discovery platform to a RunPod GPU Pod. This allows us to offload all heavy ML model inference to a dedicated, powerful, and cost-effective cloud GPU, keeping our main application containers lightweight and fast.

---

## 🎯 **Objective**

To deploy the **MCP Server** (`src/mcp/mcp_server.py`) as a standalone service on a RunPod GPU Pod, exposing its HTTP API (port 8002) to be consumed by the main Alpha Discovery application services (API, workers, agents) running elsewhere (e.g., locally or on a CPU-based cloud instance).

---

## 📋 **Prerequisites**

1.  **RunPod Account**: You need an active RunPod account with billing set up.
2.  **Docker**: Docker must be installed on your local machine to build the necessary container image.
3.  **Git**: Git must be installed to clone the project repository onto the RunPod instance.

---

##  Deployment Steps

### **Step 1: Select and Launch a RunPod GPU Pod**

1.  **Navigate to Secure Cloud**: Log in to your RunPod account and go to the **Secure Cloud** section to rent a GPU pod on-demand.
2.  **Choose a GPU**: Select a powerful GPU for our ML models. A good starting point with excellent price/performance is the **NVIDIA RTX 4090** or **RTX A5000**. For more demanding workloads, you can scale up to an A100.
3.  **Select a Template**: In the "Template" search bar, find and select the **`RunPod Pytorch 2.3.1`** template. This provides a pre-configured environment with CUDA, PyTorch, and other essential drivers.
4.  **Configure Pod Settings**:
    *   **Container Disk**: Allocate at least **30 GB**.
    *   **Volume Disk**: Allocate at least **50 GB**. This is where our project code and models will be stored persistently.
    *   **Exposed Ports**: Ensure the TCP ports `8001` (for WebSocket) and `8002` (for HTTP) are exposed.
5.  **Deploy**: Click "Deploy" and wait for the pod to initialize. Once it's ready, click "Connect" to see your connection options.

### **Step 2: Connect to the Pod and Set Up the Environment**

1.  **Start a Web Terminal**: From the "Connect" menu, choose "Start Web Terminal". This will open a terminal session directly in your browser.
2.  **Clone the Project**:
    ```bash
    git clone https://your-git-repository/alpha-discovery.git
    cd alpha-discovery
    ```
3.  **Verify Environment**: Check that `nvidia-smi` and `python` are working correctly.
    ```bash
    nvidia-smi  # Should show GPU details
    python --version # Should show a recent Python version
    ```

### **Step 3: Build and Run the MCP Server Docker Container**

We will use a dedicated Dockerfile (`Dockerfile.mcp`) to build our GPU-enabled MCP server.

1.  **Build the Docker Image**: From the root of the `alpha-discovery` directory inside your RunPod terminal, run the following command. This will build the Docker image using the GPU-specific Dockerfile.
    ```bash
    docker build -t alpha-discovery-mcp -f Dockerfile.mcp .
    ```
    *(Note: The `Dockerfile.mcp` will be created in the next step)*.

2.  **Run the Docker Container**: Once the build is complete, run the container. This command maps the exposed ports and ensures the container uses the GPU.
    ```bash
    docker run -d --name mcp-server --gpus all -p 8001:8001 -p 8002:8002 alpha-discovery-mcp
    ```
    *   `-d`: Run in detached mode.
    *   `--name mcp-server`: Assign a name to the container.
    *   `--gpus all`: **Crucial step** that makes the GPU available inside the container.
    *   `-p 8001:8001 -p 8002:8002`: Map the container ports to the host pod ports.

### **Step 4: Verify the Deployment**

1.  **Check Container Logs**:
    ```bash
    docker logs -f mcp-server
    ```
    You should see the log output indicating that the MCP server has started and is initializing the ML models successfully. Look for "All ML models initialized successfully on MCP server".

2.  **Access the Public Endpoint**:
    *   In your RunPod dashboard under "My Pods", find your running pod.
    *   You will see the public IP address and the mapped ports (e.g., `https://<your-pod-ip>:<mapped-port>`).
    *   The HTTP service will be available at port `8002`. You can test the health endpoint in your browser or with `curl`:
        ```bash
        curl http://<your-pod-ip>:8002/ml-health
        ```
        This should return a JSON response with the health status of all ML models.

---

## 🔌 **Connecting Your Main Application**

To connect your main Alpha Discovery application (running locally or on another server) to the new RunPod MCP server, you only need to set one environment variable:

```
export MCP_SERVER_URL="http://<your-pod-ip>:8002"
```

The `ml_client` in the application is already configured to use this environment variable to direct all ML inference requests to your RunPod instance.

Congratulations! You have successfully offloaded all heavy ML workloads to a dedicated cloud GPU, making your main application significantly faster and more efficient. 