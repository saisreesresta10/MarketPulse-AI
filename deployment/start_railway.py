#!/usr/bin/env python3
"""
MarketPulse AI - Railway Deployment Launcher

Starts both API server and Streamlit UI for Railway deployment.
"""

import subprocess
import threading
import time
import os
import sys
from pathlib import Path

def start_api_server():
    """Start the FastAPI server."""
    print("🚀 Starting MarketPulse AI API Server...")
    try:
        # Set environment variables for production
        os.environ["DATABASE_URL"] = "sqlite:///marketpulse.db"
        os.environ["ENVIRONMENT"] = "production"
        
        # Start API server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "marketpulse_ai.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8001"
        ])
    except Exception as e:
        print(f"❌ API Server error: {e}")

def start_streamlit_ui():
    """Start the Streamlit UI."""
    print("🎨 Starting Streamlit UI...")
    time.sleep(5)  # Give API server time to start
    
    try:
        # Get port from Railway environment
        port = os.environ.get("PORT", "8501")
        
        # Start Streamlit UI
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "true"
        ])
    except Exception as e:
        print(f"❌ Streamlit UI error: {e}")

def main():
    """Start both API server and UI for Railway."""
    print("🚀 MarketPulse AI - Railway Deployment")
    print("=" * 50)
    
    try:
        # Start API server in background thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # Start Streamlit UI (this will block and serve the main app)
        start_streamlit_ui()
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()