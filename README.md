# MarketPulse AI

AI-powered decision-support copilot for India's MRP-based retail ecosystem.

## 🚀 Quick Start

### Option 1: Simple Run (Recommended for first time)
```bash
# Clone and navigate to project
cd MarketPulse_AI

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

### Option 2: Full Installation
```bash
# Install the package
pip install -e .

# Initialize database
marketpulse-ai init-db

# Start server
marketpulse-ai serve --reload
```

### Option 3: Direct Server Start
```bash
# Start with uvicorn directly
uvicorn marketpulse_ai.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 Access Your System

Once running, open your browser to:
- **📚 API Documentation**: http://localhost:8000/docs
- **🔍 Health Check**: http://localhost:8000/health
- **📊 System Status**: http://localhost:8000/api/v1/system/status

## 📋 Available Commands

```bash
# Server management
marketpulse-ai serve              # Start API server
marketpulse-ai serve --reload     # Start with auto-reload
marketpulse-ai status             # Check system status

# Testing and demo
marketpulse-ai test               # Run test suite
marketpulse-ai demo               # Run workflow demo

# Database
marketpulse-ai init-db            # Initialize database

# Information
marketpulse-ai config             # Show configuration
marketpulse-ai version            # Show version info
```

## Overview

MarketPulse AI is a comprehensive decision support system designed specifically for Indian retailers operating within the Maximum Retail Price (MRP) regulatory framework. The system analyzes historical sales data, seasonal patterns, and market signals to provide explainable insights and recommendations for inventory planning, discount timing, and risk management.

## Key Features

- **MRP-Compliant Recommendations**: All suggestions comply with Indian retail regulations
- **Explainable AI**: Clear, business-friendly explanations for all insights and recommendations
- **Seasonal Intelligence**: Advanced understanding of Indian festivals and seasonal patterns
- **Risk Assessment**: Proactive identification of inventory and demand risks
- **Scenario Analysis**: What-if modeling for strategic decision making
- **Data Security**: Enterprise-grade security with audit trails and compliance monitoring

## Architecture

The system is built with a modular architecture consisting of:

- **Data Processor**: Ingests and analyzes sales data, seasonal patterns, and market signals
- **Insight Generator**: Creates explainable insights with supporting evidence
- **Risk Assessor**: Identifies inventory and demand risks with early warning systems
- **Compliance Validator**: Ensures all recommendations comply with MRP regulations
- **Scenario Analyzer**: Generates what-if scenarios for decision planning
- **Decision Support Engine**: Orchestrates all components to provide comprehensive recommendations

## Quick Start

### Prerequisites

- Python 3.8 or higher
- SQLite (included) or PostgreSQL for production

### Installation

1. Clone the repository:
```bash
git clone https://github.com/marketpulse-ai/marketpulse-ai.git
cd marketpulse-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment configuration:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python -m marketpulse_ai.cli init-db
```

### Running Tests

Run the complete test suite:
```bash
pytest
```

Run specific test categories:
```bash
# Unit tests only
pytest -m unit

# Property-based tests
pytest -m property

# Integration tests
pytest -m integration
```

Run tests with coverage:
```bash
pytest --cov=marketpulse_ai --cov-report=html
```

### Development Setup

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

Set up pre-commit hooks:
```bash
pre-commit install
```

Run code quality checks:
```bash
# Format code
black marketpulse_ai tests

# Sort imports
isort marketpulse_ai tests

# Lint code
flake8 marketpulse_ai tests

# Type checking
mypy marketpulse_ai
```

## Configuration

MarketPulse AI uses environment-based configuration. Key settings include:

### Application Settings
- `MARKETPULSE_ENVIRONMENT`: Deployment environment (development, testing, staging, production)
- `MARKETPULSE_DEBUG`: Enable debug mode (true/false)
- `MARKETPULSE_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Database Settings
- `MARKETPULSE_DATABASE_URL`: Database connection URL
- `MARKETPULSE_DATABASE_ECHO`: Enable SQL query logging (true/false)

### Security Settings
- `MARKETPULSE_SECRET_KEY`: Application secret key (minimum 32 characters)
- `MARKETPULSE_ENABLE_DATA_ENCRYPTION`: Enable data encryption at rest (true/false)

### AI/ML Settings
- `MARKETPULSE_MODEL_CONFIDENCE_THRESHOLD`: Minimum confidence threshold (0.0-1.0)
- `MARKETPULSE_PROPERTY_TEST_ITERATIONS`: Property test iterations for validation

See `.env.example` for complete configuration options.

## Testing Strategy

MarketPulse AI uses a comprehensive testing approach:

### Unit Tests
- Test individual components and functions
- Validate business logic and edge cases
- Ensure proper error handling

### Property-Based Tests
- Use Hypothesis for comprehensive input validation
- Test universal properties across all possible inputs
- Validate correctness properties from the design document

### Integration Tests
- Test component interactions and data flow
- Validate end-to-end workflows
- Ensure system reliability and performance

### Test Categories
Tests are organized by markers:
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.property`: Property-based tests
- `@pytest.mark.security`: Security-related tests
- `@pytest.mark.compliance`: MRP compliance tests

## Project Structure

```
marketpulse_ai/
├── core/                   # Core data models and interfaces
│   ├── models.py          # Pydantic data models
│   └── interfaces.py      # Component interfaces
├── components/            # Implementation of core components
├── config/               # Configuration management
│   ├── settings.py       # Application settings
│   ├── database.py       # Database configuration
│   ├── security.py       # Security configuration
│   └── logging_config.py # Logging configuration
├── api/                  # FastAPI web service
├── utils/                # Utility functions and helpers
└── cli/                  # Command-line interface

tests/
├── conftest.py           # Pytest configuration and fixtures
├── test_core_models.py   # Core model tests
└── ...                   # Additional test modules
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Run code quality checks (`black`, `isort`, `flake8`, `mypy`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [https://marketpulse-ai.readthedocs.io/](https://marketpulse-ai.readthedocs.io/)
- Issues: [GitHub Issues](https://github.com/marketpulse-ai/marketpulse-ai/issues)
- Discussions: [GitHub Discussions](https://github.com/marketpulse-ai/marketpulse-ai/discussions)

## Acknowledgments

- Built for the Indian retail ecosystem
- Designed with MRP compliance as a core principle
- Focused on explainable AI for business decision support