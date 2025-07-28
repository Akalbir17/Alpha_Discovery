#!/usr/bin/env python3
"""
Startup script for Alpha Discovery Dashboard
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

def setup_environment():
    """Setup environment variables for dashboard"""
    
    # Set default environment variables if not already set
    env_vars = {
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': '6379',
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'alpha_discovery',
        'POSTGRES_USER': 'postgres',
        'POSTGRES_PASSWORD': 'postgres',
        'API_BASE_URL': 'http://localhost:8000',
        'REFRESH_INTERVAL': '5',
        'MAX_DATA_POINTS': '1000',
        'TIMEZONE': 'UTC',
        'CURRENCY_SYMBOL': '$'
    }
    
    for key, default_value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
    
    print("Environment variables set:")
    for key in env_vars:
        print(f"  {key}={os.environ[key]}")

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'redis',
        'psycopg2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    
    dashboard_path = current_dir / "app.py"
    
    # Streamlit configuration
    streamlit_config = {
        '--server.port': '8501',
        '--server.address': '0.0.0.0',
        '--server.headless': 'true',
        '--server.enableCORS': 'false',
        '--server.enableXsrfProtection': 'false',
        '--browser.gatherUsageStats': 'false'
    }
    
    # Build command
    cmd = ['streamlit', 'run', str(dashboard_path)]
    
    for key, value in streamlit_config.items():
        cmd.extend([key, value])
    
    print(f"Starting dashboard with command: {' '.join(cmd)}")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)

def main():
    """Main function"""
    
    print("=" * 60)
    print("🚀 Alpha Discovery Dashboard Launcher")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup environment
    print("\n1. Setting up environment...")
    setup_environment()
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Run dashboard
    print("\n3. Starting dashboard...")
    run_dashboard()

if __name__ == "__main__":
    main() 