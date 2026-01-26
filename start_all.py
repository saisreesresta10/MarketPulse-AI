#!/usr/bin/env python3
"""
MarketPulse AI - Start Everything

Starts both the API server and Streamlit UI automatically.
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def start_api_server():
    """Start the API server in background."""
    print("🚀 Starting MarketPulse AI API Server...")
    try:
        # Start API server
        subprocess.run([sys.executable, "run.py"], cwd=Path.cwd())
    except Exception as e:
        print(f"❌ API Server error: {e}")

def start_streamlit_ui():
    """Start the Streamlit UI."""
    print("🎨 Starting Streamlit UI...")
    time.sleep(3)  # Give API server time to start
    
    try:
        # Install Streamlit requirements if needed
        try:
            import streamlit
        except ImportError:
            print("📦 Installing Streamlit requirements...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements-streamlit.txt"
            ], check=True)
        
        # Start Streamlit UI
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except Exception as e:
        print(f"❌ Streamlit UI error: {e}")

def main():
    """Start both API server and UI."""
    print("🚀 MarketPulse AI - Complete System Launcher")
    print("=" * 60)
    
    # Check if files exist
    if not Path("run.py").exists():
        print("❌ Error: run.py not found")
        sys.exit(1)
    
    if not Path("streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found")
        sys.exit(1)
    
    print("🔄 Starting API Server and UI...")
    print("📍 API will be at: http://localhost:8001")
    print("🎨 UI will be at: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop everything")
    print("-" * 60)
    
    try:
        # Start API server in background thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # Start Streamlit UI (this will block)
        start_streamlit_ui()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down MarketPulse AI...")
    except Exception as e:
        print(f"❌ Startup failed: {e}")

if __name__ == "__main__":
    main()