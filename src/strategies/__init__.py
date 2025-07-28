"""
Trading Strategies and Risk Management

This package contains trading strategies and risk management tools:
- BacktestingEngine: Historical strategy validation and performance analysis
- Risk Manager: Position sizing, portfolio risk controls, and risk monitoring
"""

from .backtester import BacktestingEngine
from .risk_manager import RiskManager

__all__ = [
    "BacktestingEngine",
    "RiskManager"
] 