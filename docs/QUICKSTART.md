# MarketPulse AI - Quick Start Guide

Welcome to MarketPulse AI! This guide will help you get the system up and running quickly.

## 🚀 Quick Setup

### 1. Install the Package

```bash
# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,test]"
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# DATABASE_URL=sqlite:///marketpulse.db
# SECRET_KEY=your-secret-key-here
```

### 3. Initialize Database

```bash
# Initialize database tables
marketpulse-ai init-db
```

### 4. Start the Server

```bash
# Start development server
marketpulse-ai serve --reload

# Or start with custom settings
marketpulse-ai serve --host 0.0.0.0 --port 8000 --reload
```

## 🎯 Access the System

Once the server is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **System Status**: http://localhost:8000/api/v1/system/status

## 🛠️ CLI Commands

### Server Management
```bash
# Start the API server
marketpulse-ai serve

# Start with auto-reload (development)
marketpulse-ai serve --reload

# Start on different port
marketpulse-ai serve --port 9000
```

### System Operations
```bash
# Check system status
marketpulse-ai status

# Run tests
marketpulse-ai test

# Run demo workflows
marketpulse-ai demo

# Show configuration
marketpulse-ai config

# Show version
marketpulse-ai version
```

### Database Operations
```bash
# Initialize database
marketpulse-ai init-db
```

## 📊 Testing the API

### Using the Web Interface
1. Go to http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"

### Using curl (Command Line)
```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/api/v1/system/status

# Generate insights (POST request)
curl -X POST "http://localhost:8000/api/v1/insights/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "retailer_id": "RETAILER_001",
    "product_ids": ["PROD_001", "PROD_002"],
    "analysis_type": "comprehensive"
  }'
```

### Using Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Generate insights
insight_data = {
    "retailer_id": "RETAILER_001",
    "product_ids": ["PROD_001"],
    "analysis_type": "comprehensive"
}
response = requests.post(
    "http://localhost:8000/api/v1/insights/generate",
    json=insight_data
)
print(response.json())
```

## 🔧 Development Setup

### Install Development Dependencies
```bash
pip install -e ".[dev,test]"
```

### Run Tests
```bash
# Run all tests
marketpulse-ai test

# Or use pytest directly
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=marketpulse_ai --cov-report=html
```

### Code Quality
```bash
# Format code
black marketpulse_ai/
isort marketpulse_ai/

# Lint code
flake8 marketpulse_ai/
mypy marketpulse_ai/
```

## 📁 Project Structure

```
MarketPulse_AI/
├── marketpulse_ai/           # Main package
│   ├── api/                  # FastAPI application
│   ├── components/           # AI components
│   ├── config/              # Configuration
│   ├── core/                # Core models
│   ├── storage/             # Data storage
│   └── cli.py               # Command line interface
├── tests/                   # Test suite
├── examples/                # Example scripts
├── .env.example            # Environment template
├── requirements.txt        # Dependencies
└── setup.py               # Package setup
```

## 🎯 Key Features

### Data Processing
- Ingest sales data from multiple sources
- Real-time data validation and cleaning
- Pattern recognition and anomaly detection

### AI Insights
- Demand forecasting
- Price optimization recommendations
- Risk assessment
- Compliance validation

### Decision Support
- Scenario analysis
- Business impact modeling
- Automated recommendations
- Performance tracking

### System Reliability
- Automated backups
- Load balancing
- Health monitoring
- Graceful error handling

## 🔍 Troubleshooting

### Common Issues

**Server won't start:**
```bash
# Check if port is in use
netstat -an | findstr :8000

# Try different port
marketpulse-ai serve --port 8001
```

**Database errors:**
```bash
# Reinitialize database
marketpulse-ai init-db
```

**Import errors:**
```bash
# Reinstall package
pip install -e .
```

**Test failures:**
```bash
# Check system status
marketpulse-ai status

# Run specific test
pytest tests/test_api.py -v
```

## 📚 Next Steps

1. **Explore the API**: Use the interactive docs at `/docs`
2. **Run the demo**: `marketpulse-ai demo`
3. **Check the examples**: Look at files in the `examples/` directory
4. **Read the specs**: Check `.kiro/specs/marketpulse-ai/` for detailed documentation
5. **Customize**: Modify configuration in `.env` file

## 🆘 Getting Help

- Check the API documentation at `/docs`
- Run `marketpulse-ai --help` for CLI help
- Look at example scripts in `examples/`
- Check system status with `marketpulse-ai status`

Happy coding! 🎉