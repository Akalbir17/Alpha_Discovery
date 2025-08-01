#!/usr/bin/env python3
"""
Entry point for the MCP server when run as a module.

Usage:
    python -m src.mcp
"""

import asyncio
import logging
import sys
from .mcp_server import MCPServer

def main():
    """Main entry point for the MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start the MCP server
    server = MCPServer(host="0.0.0.0", port=8001, http_port=8002)
    
    try:
        print("Starting Alpha Discovery MCP Server...")
        print("WebSocket MCP Server: ws://0.0.0.0:8001")
        print("HTTP ML Services: http://0.0.0.0:8002")
        
        # Run the server
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        asyncio.run(server.stop())
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 