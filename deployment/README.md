# MarketPulse AI Deployment

This folder contains deployment configurations for various platforms.

## Folders:
- `railway/` - Railway deployment configuration
- `render/` - Render deployment configuration

## Files:
- `Procfile` - Heroku deployment configuration
- `runtime.txt` - Python runtime specification
- `requirements-railway.txt` - Railway-specific dependencies
- `start_railway.py` - Railway startup script

## Platform-specific Requirements:
- Streamlit Cloud: Use `requirements.txt` in root
- Railway: Use `requirements-railway.txt`
- Render: Use `requirements.txt` in root
- Heroku: Use `Procfile` and `runtime.txt`