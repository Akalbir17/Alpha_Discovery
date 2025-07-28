"""
Alpha Discovery Platform - Multi-Agent Alpha Discovery System

A production-ready platform that leverages AI agents to identify, analyze, 
and execute trading opportunities across multiple markets and data sources.
"""

__version__ = "1.0.0"
__author__ = "Alpha Discovery Team"
__email__ = "team@alphadiscovery.com"

from . import agents
from . import tools
from . import mcp
from . import orchestration
from . import strategies
from . import data
from . import workers

__all__ = [
    "agents",
    "tools", 
    "mcp",
    "orchestration",
    "strategies",
    "data",
    "workers"
] 