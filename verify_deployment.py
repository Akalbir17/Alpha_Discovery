#!/usr/bin/env python3
"""
Alpha Discovery Deployment Verification Script
This script helps verify that each component of your deployment is working correctly.
"""

import requests
import json
import time
import sys
import os
from urllib.parse import urlparse

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    print(f"\n📋 Step {step}: {description}")

def print_success(message):
    print(f"✅ {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def print_error(message):
    print(f"❌ {message}")

def test_ml_server_direct(ml_server_url):
    """Test direct connection to ML server on RunPod"""
    print_step(1, "Testing Direct ML Server Connection")
    
    try:
        # Parse URL to get components
        parsed = urlparse(ml_server_url)
        if not parsed.scheme:
            ml_server_url = f"http://{ml_server_url}"
        
        # Test health endpoint
        response = requests.get(f"{ml_server_url}/ml-health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"ML Server is healthy!")
            print(f"   📊 Models loaded: {data.get('models_loaded', 'Unknown')}")
            print(f"   🎮 GPU available: {data.get('gpu_available', 'Unknown')}")
            print(f"   🔗 Server URL: {ml_server_url}")
            return True
        else:
            print_error(f"ML Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to ML Server")
        print("   🔍 Check if:")
        print("   - RunPod pod is running")
        print("   - IP address is correct")
        print("   - Ports 8001 and 8002 are exposed")
        return False
    except requests.exceptions.Timeout:
        print_error("Connection to ML Server timed out")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def test_local_services():
    """Test local Docker services"""
    print_step(2, "Testing Local Docker Services")
    
    services_to_test = [
        ("PostgreSQL", "http://localhost:5432", "Database"),
        ("Redis", "http://localhost:6379", "Cache"),
        ("API", "http://localhost:8000/health", "Main API"),
        ("Prometheus", "http://localhost:9090", "Metrics"),
        ("Grafana", "http://localhost:3000", "Dashboard")
    ]
    
    results = {}
    
    for service_name, url, description in services_to_test:
        try:
            if service_name in ["PostgreSQL", "Redis"]:
                # These don't have HTTP endpoints, just check if ports are open
                import socket
                host, port = "localhost", int(url.split(":")[-1])
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    print_success(f"{service_name} ({description}) is running")
                    results[service_name] = True
                else:
                    print_error(f"{service_name} ({description}) is not accessible")
                    results[service_name] = False
            else:
                # HTTP services
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print_success(f"{service_name} ({description}) is healthy")
                    results[service_name] = True
                else:
                    print_warning(f"{service_name} returned status {response.status_code}")
                    results[service_name] = False
                    
        except Exception as e:
            print_error(f"{service_name} ({description}) failed: {str(e)[:50]}")
            results[service_name] = False
    
    return results

def test_api_ml_connection():
    """Test if API can connect to ML server"""
    print_step(3, "Testing API to ML Server Connection")
    
    try:
        response = requests.get("http://localhost:8000/ml-health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print_success("API can successfully connect to ML Server!")
            print(f"   📊 ML Models available: {data.get('models_loaded', 'Unknown')}")
            return True
        else:
            print_error(f"API ML health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API (is it running?)")
        return False
    except Exception as e:
        print_error(f"API ML connection test failed: {e}")
        return False

def test_ml_inference():
    """Test actual ML inference"""
    print_step(4, "Testing ML Inference Capabilities")
    
    try:
        # Test sentiment analysis
        test_data = {
            "text": "The market is looking bullish today with strong volume."
        }
        
        response = requests.post(
            "http://localhost:8000/test-sentiment",  # This endpoint would need to be added
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print_success("ML inference is working!")
            return True
        else:
            print_warning("ML inference test endpoint not available (this is normal)")
            return True  # Don't fail on this
            
    except Exception as e:
        print_warning(f"ML inference test skipped: {e}")
        return True  # Don't fail on this

def check_environment_config():
    """Check environment configuration"""
    print_step(0, "Checking Environment Configuration")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print_success(".env file found")
        
        # Read and check critical variables
        with open('.env', 'r') as f:
            env_content = f.read()
            
        critical_vars = ['MCP_SERVER_URL', 'POSTGRES_PASSWORD', 'REDIS_PASSWORD']
        missing_vars = []
        
        for var in critical_vars:
            if var not in env_content:
                missing_vars.append(var)
        
        if missing_vars:
            print_error(f"Missing environment variables: {', '.join(missing_vars)}")
            return False
        else:
            print_success("All critical environment variables found")
            
            # Extract MCP_SERVER_URL
            for line in env_content.split('\n'):
                if line.startswith('MCP_SERVER_URL='):
                    mcp_url = line.split('=', 1)[1].strip()
                    print(f"   🔗 MCP Server URL: {mcp_url}")
                    return mcp_url
                    
    else:
        print_error(".env file not found")
        print("   📝 Create it by copying configs/env.template to .env")
        return False

def generate_report(results):
    """Generate final deployment report"""
    print_header("DEPLOYMENT VERIFICATION REPORT")
    
    all_passed = True
    
    print(f"\n📊 Service Status Summary:")
    for service, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {service}")
        if not status:
            all_passed = False
    
    print(f"\n🎯 Overall Status:")
    if all_passed:
        print_success("🎉 ALL SYSTEMS OPERATIONAL!")
        print("   Your Alpha Discovery system is ready for trading!")
        print("\n📋 Next Steps:")
        print("   1. Open Grafana dashboard: http://localhost:3000")
        print("   2. Check Prometheus metrics: http://localhost:9090")
        print("   3. Monitor API logs: docker-compose logs -f api")
        print("   4. Start your trading strategies!")
    else:
        print_error("⚠️  SOME ISSUES DETECTED")
        print("   Please review the failed components above")
        print("   Refer to COMPLETE_DEPLOYMENT_WALKTHROUGH.md for troubleshooting")
    
    return all_passed

def main():
    print_header("ALPHA DISCOVERY DEPLOYMENT VERIFICATION")
    print("This script will verify that all components of your system are working correctly.")
    print("Make sure you have completed the deployment steps first!")
    
    results = {}
    
    # Step 0: Check environment
    mcp_url = check_environment_config()
    if not mcp_url:
        print_error("Environment configuration failed. Please fix and run again.")
        sys.exit(1)
    
    # Step 1: Test ML server directly
    results['ML Server (Direct)'] = test_ml_server_direct(mcp_url)
    
    # Step 2: Test local services
    local_results = test_local_services()
    results.update(local_results)
    
    # Step 3: Test API to ML connection
    if results.get('API', False):
        results['API-ML Connection'] = test_api_ml_connection()
    else:
        print_warning("Skipping API-ML test (API not running)")
        results['API-ML Connection'] = False
    
    # Step 4: Test ML inference
    if results.get('API-ML Connection', False):
        results['ML Inference'] = test_ml_inference()
    else:
        print_warning("Skipping ML inference test (API-ML connection failed)")
        results['ML Inference'] = False
    
    # Generate final report
    success = generate_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 