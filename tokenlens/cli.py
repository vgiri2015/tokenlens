import uvicorn
import click
from .main import app

@click.group()
def cli():
    """LLM Token Limits API CLI"""
    pass

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the LLM Token Limits API server"""
    uvicorn.run("tokenlens.main:app", host=host, port=port, reload=reload)

if __name__ == '__main__':
    cli()
