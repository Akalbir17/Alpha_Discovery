#!/usr/bin/env python3
"""
MCP Server Startup Script

Run the Alpha Discovery MCP server with proper configuration,
logging, and error handling.
"""

import asyncio
import os
import sys
import signal
import structlog
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.mcp_server import MCPServer
from utils.model_manager import model_manager

logger = structlog.get_logger(__name__)


async def main():
    """Main function to start the MCP server"""
    try:
        # Configure logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Load environment variables
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8001"))
        
        logger.info("Starting Alpha Discovery MCP Server", host=host, port=port)
        
        # Initialize model manager
        await model_manager.initialize()
        logger.info("Model manager initialized")
        
        # Create and start MCP server
        server = MCPServer(host=host, port=port)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal", signal=signum)
            asyncio.create_task(shutdown(server))
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the server
        await server.start()
        
    except Exception as e:
        logger.error("Failed to start MCP server", error=str(e))
        sys.exit(1)


async def shutdown(server):
    """Graceful shutdown of the MCP server"""
    try:
        logger.info("Shutting down MCP server...")
        
        # Close all client connections
        for client_id, websocket in server.clients.items():
            try:
                await websocket.close()
            except Exception as e:
                logger.error("Error closing client connection", client_id=client_id, error=str(e))
        
        # Clear client list
        server.clients.clear()
        server.subscriptions.clear()
        
        logger.info("MCP server shutdown complete")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))
    finally:
        # Exit the process
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        sys.exit(1) 