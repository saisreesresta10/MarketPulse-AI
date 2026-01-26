"""
Setup configuration for MarketPulse AI package.

Defines package metadata, dependencies, and installation configuration
for the MarketPulse AI decision support system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from a file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and -r includes
                if line and not line.startswith('#') and not line.startswith('-r'):
                    requirements.append(line)
            return requirements
    return []

# Core requirements
install_requires = read_requirements("requirements.txt")

# Development requirements
dev_requires = read_requirements("requirements-dev.txt")

setup(
    name="marketpulse-ai",
    version="0.1.0",
    author="MarketPulse AI Team",
    author_email="team@marketpulse-ai.com",
    description="AI-powered decision-support copilot for India's MRP-based retail ecosystem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marketpulse-ai/marketpulse-ai",
    project_urls={
        "Bug Tracker": "https://github.com/marketpulse-ai/marketpulse-ai/issues",
        "Documentation": "https://marketpulse-ai.readthedocs.io/",
        "Source Code": "https://github.com/marketpulse-ai/marketpulse-ai",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires if install_requires else [],
    extras_require={
        "dev": dev_requires if dev_requires else [],
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "hypothesis>=6.92.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "marketpulse-ai=marketpulse_ai.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "marketpulse_ai": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai",
        "retail",
        "decision-support",
        "mrp",
        "india",
        "inventory-management",
        "demand-forecasting",
        "compliance",
        "analytics",
    ],
)