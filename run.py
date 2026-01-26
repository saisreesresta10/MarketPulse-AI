#!/usr/bin/env python3
"""
MarketPulse AI - Simple Startup Script

Quick way to start the MarketPulse AI system without installation.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'sqlalchemy',
        'cryptography'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Set up environment variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file from template...")
        example_file = Path(".env.example")
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
        else:
            # Create basic .env file
            with open(env_file, 'w') as f:
                f.write("DATABASE_URL=sqlite:///marketpulse.db\n")
                f.write("SECRET_KEY=dev-secret-key-change-in-production\n")
                f.write("ENVIRONMENT=development\n")
        print("✅ Environment file created")

def main():
    """Main startup function."""
    print("🚀 MarketPulse AI - Starting System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("marketpulse_ai").exists():
        print("❌ Error: marketpulse_ai directory not found")
        print("💡 Make sure you're running this from the project root directory")
        sys.exit(1)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All requirements satisfied")
    
    # Setup environment
    setup_environment()
    
    # Add current directory to Python path
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        print("🌟 Starting MarketPulse AI API Server...")
        print("📍 Server will be available at: http://localhost:8001")
        print("📚 API Documentation: http://localhost:8001/docs")
        print("🔍 Health Check: http://localhost:8001/health")
        print("\n⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the server
        import uvicorn
        uvicorn.run(
            "marketpulse_ai.api.main:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()