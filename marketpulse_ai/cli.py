"""
MarketPulse AI Command Line Interface

Provides command-line tools for running, managing, and interacting with
the MarketPulse AI system.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .api.main import app, component_manager
from .config.settings import get_settings

# Initialize CLI app and console
cli = typer.Typer(
    name="marketpulse-ai",
    help="MarketPulse AI - AI-powered retail decision support system",
    add_completion=False
)
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
):
    """Start the MarketPulse AI API server."""
    rprint(f"[bold green]🚀 Starting MarketPulse AI API Server[/bold green]")
    rprint(f"[blue]Host:[/blue] {host}")
    rprint(f"[blue]Port:[/blue] {port}")
    rprint(f"[blue]Reload:[/blue] {reload}")
    rprint(f"[blue]Workers:[/blue] {workers}")
    
    try:
        uvicorn.run(
            "marketpulse_ai.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        rprint("\n[yellow]👋 Server stopped by user[/yellow]")
    except Exception as e:
        rprint(f"[red]❌ Server failed to start: {e}[/red]")
        sys.exit(1)


@cli.command()
def status():
    """Check the status of MarketPulse AI components."""
    rprint("[bold blue]🔍 Checking MarketPulse AI System Status[/bold blue]")
    
    try:
        # Initialize components in testing mode for status check
        asyncio.run(component_manager.initialize_components(testing_mode=True))
        
        # Create status table
        table = Table(title="Component Status")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Description", style="white")
        
        components = component_manager.get_all_components()
        
        for name, component in components.items():
            status = "✅ Healthy" if component else "❌ Not Available"
            description = f"{name.replace('_', ' ').title()}"
            table.add_row(name, status, description)
        
        console.print(table)
        rprint(f"\n[green]✅ Total Components: {len(components)}[/green]")
        
    except Exception as e:
        rprint(f"[red]❌ Status check failed: {e}[/red]")
        sys.exit(1)


@cli.command()
def test():
    """Run the test suite."""
    rprint("[bold blue]🧪 Running MarketPulse AI Test Suite[/bold blue]")
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            rprint("[green]✅ All tests passed![/green]")
        else:
            rprint("[red]❌ Some tests failed[/red]")
            sys.exit(1)
            
    except FileNotFoundError:
        rprint("[red]❌ pytest not found. Install with: pip install pytest[/red]")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]❌ Test execution failed: {e}[/red]")
        sys.exit(1)


@cli.command()
def demo():
    """Run the end-to-end workflow demonstration."""
    rprint("[bold blue]🎯 Running MarketPulse AI Demo[/bold blue]")
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "examples/end_to_end_workflows_demo.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            rprint("[green]✅ Demo completed successfully![/green]")
        else:
            rprint("[yellow]⚠️ Demo completed with warnings[/yellow]")
            
    except Exception as e:
        rprint(f"[red]❌ Demo execution failed: {e}[/red]")
        sys.exit(1)


@cli.command()
def init_db():
    """Initialize the database with required tables."""
    rprint("[bold blue]🗄️ Initializing MarketPulse AI Database[/bold blue]")
    
    try:
        # Initialize database setup
        asyncio.run(_init_database())
        rprint("[green]✅ Database initialized successfully![/green]")
        
    except Exception as e:
        rprint(f"[red]❌ Database initialization failed: {e}[/red]")
        sys.exit(1)


async def _init_database():
    """Initialize database components."""
    from .config.database import DatabaseManager, DatabaseConfig
    from .config.security import SecurityConfig
    from .storage.storage_manager import StorageManager
    from cryptography.fernet import Fernet
    
    # Get settings
    settings = get_settings()
    
    # Setup database
    db_config = DatabaseConfig(url=settings.database_url)
    db_manager = DatabaseManager(db_config)
    
    # Setup security
    security_config = SecurityConfig(
        secret_key=settings.secret_key,
        encryption_key=Fernet.generate_key().decode()
    )
    
    # Initialize storage
    storage_manager = StorageManager(db_manager, security_config)
    
    rprint("[green]Database tables created successfully[/green]")


@cli.command()
def config():
    """Show current configuration."""
    rprint("[bold blue]⚙️ MarketPulse AI Configuration[/bold blue]")
    
    try:
        settings = get_settings()
        
        # Create config table
        table = Table(title="Configuration Settings")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        
        # Add configuration items (hide sensitive data)
        config_items = [
            ("Database URL", settings.database_url),
            ("Environment", getattr(settings, 'environment', 'development')),
            ("Debug Mode", str(getattr(settings, 'debug', False))),
            ("API Version", "1.0.0"),
        ]
        
        for key, value in config_items:
            # Hide sensitive information
            if "password" in key.lower() or "secret" in key.lower() or "key" in key.lower():
                value = "***hidden***"
            table.add_row(key, str(value))
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]❌ Configuration check failed: {e}[/red]")
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    rprint("[bold blue]📋 MarketPulse AI Version Information[/bold blue]")
    
    version_info = {
        "MarketPulse AI": "0.1.0",
        "Python": sys.version.split()[0],
        "Platform": sys.platform,
    }
    
    for key, value in version_info.items():
        rprint(f"[cyan]{key}:[/cyan] {value}")


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        rprint("\n[yellow]👋 Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        rprint(f"[red]❌ Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()