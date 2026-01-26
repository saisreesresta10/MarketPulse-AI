#!/usr/bin/env python3
"""
MarketPulse AI - Streamlit UI Launcher

Launch the beautiful Streamlit web interface for MarketPulse AI.
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import requests

def check_api_server():
    """Check if the MarketPulse AI API server is running."""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def install_streamlit_requirements():
    """Install Streamlit requirements if needed."""
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ Streamlit requirements already installed")
        return True
    except ImportError:
        print("📦 Installing Streamlit requirements...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements-streamlit.txt"
            ], check=True)
            print("✅ Streamlit requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit requirements")
            return False

def main():
    """Main function to launch the Streamlit UI."""
    print("🚀 MarketPulse AI - Streamlit UI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found")
        print("💡 Make sure you're running this from the project root directory")
        sys.exit(1)
    
    # Install requirements
    if not install_streamlit_requirements():
        sys.exit(1)
    
    # Check if API server is running
    print("🔍 Checking MarketPulse AI API server...")
    if not check_api_server():
        print("⚠️  MarketPulse AI API server is not running!")
        print("💡 Please start the API server first:")
        print("   python run.py")
        print("\n🔄 Waiting for API server to start...")
        
        # Wait for API server
        for i in range(30):  # Wait up to 30 seconds
            if check_api_server():
                print("✅ API server is now running!")
                break
            time.sleep(1)
            print(f"   Waiting... ({i+1}/30)")
        else:
            print("❌ API server is still not running")
            print("💡 Please start it manually: python run.py")
            print("   Then run this script again")
            sys.exit(1)
    else:
        print("✅ API server is running")
    
    # Launch Streamlit
    print("\n🌟 Launching MarketPulse AI Web Interface...")
    print("📍 UI will be available at: http://localhost:8501")
    print("🔗 API server running at: http://localhost:8001")
    print("\n⏹️  Press Ctrl+C to stop the UI")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit UI stopped by user")
    except Exception as e:
        print(f"❌ Failed to start Streamlit UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()