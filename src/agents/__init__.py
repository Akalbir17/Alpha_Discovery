"""
AI Agents for Alpha Discovery Platform

This package contains specialized AI agents for different aspects of alpha discovery:
- Microstructure Agent: Analyzes order flow and market microstructure
- Alternative Data Agent: Processes social media and alternative data sources
- Regime Agent: Detects market regimes and adapts strategies
- Strategy Agent: Executes trading strategies based on agent consensus
"""

from .microstructure_agent import MicrostructureAgent
from .altdata_agent import AlternativeDataAgent
from .regime_agent import RegimeDetectionAgent
from .strategy_agent import StrategyAgent

__all__ = [
    "MicrostructureAgent",
    "AlternativeDataAgent", 
    "RegimeDetectionAgent",
    "StrategyAgent"
] 