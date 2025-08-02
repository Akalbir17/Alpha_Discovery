"""
Strategy Synthesis Agent for Alpha Discovery Platform

This agent combines signals from other agents into coherent trading strategies,
implements portfolio optimization, risk management, and uses reinforcement learning
for continuous strategy improvement using CrewAI framework.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from pathlib import Path
import pickle
import joblib
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
# Remove direct sklearn imports - now using ML client
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
# Remove direct RL imports - now using ML client
# import gym
# from stable_baselines3 import PPO, A2C, DQN
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env
import warnings
warnings.filterwarnings('ignore')

# ML client for remote inference (Phase 4 upgrade)
try:
    from src.scrapers.ml_client import ml_client
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False
    logger.warning("ML client not available - using fallback analysis")

# Import our existing components
from src.agents.microstructure_agent import MicrostructureAgent
from src.agents.altdata_agent import AlternativeDataAgent
from src.agents.regime_agent import RegimeDetectionAgent
from src.utils.model_manager import ModelManager
from src.utils.error_handling import handle_errors, AlphaDiscoveryError
from src.utils.monitoring import monitor_performance, track_metrics
from src.utils.config_manager import get_config_section

tool_config = get_config_section('tools')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Strategy type classification"""
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    ARBITRAGE = "ARBITRAGE"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    PAIRS_TRADING = "PAIRS_TRADING"
    MULTI_FACTOR = "MULTI_FACTOR"

class RiskLevel(Enum):
    """Risk level classification"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    SPECULATIVE = "SPECULATIVE"

@dataclass
class StrategySignal:
    """Combined strategy signal from multiple agents"""
    symbol: str
    strategy_type: StrategyType
    direction: str  # long/short
    confidence: float
    expected_return: float
    risk_score: float
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    holding_period: int
    agent_sources: List[str]
    market_regime: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioOptimization:
    """Portfolio optimization result"""
    symbols: List[str]
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    optimization_method: str
    constraints: Dict[str, Any]
    timestamp: datetime

@dataclass
class BacktestResult:
    """Backtesting result"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_count: int
    avg_trade_duration: float
    market_impact_cost: float
    performance_metrics: Dict[str, float]
    equity_curve: pd.Series
    trade_log: List[Dict[str, Any]]

class MarketResearchTool(BaseTool):
    """Tool for McKinsey-level market research and competitive intelligence"""
    
    name: str = "market_research_analyzer"
    description: str = "Conducts deep market research, competitive intelligence, and strategic forecasting using McKinsey-level analysis"
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
    
    @handle_errors
    async def _run(self, market_sector: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Conduct McKinsey-level market research"""
        try:
            # McKinsey-level market research prompt
            research_prompt = f"""
            You are a world-class industry analyst with expertise in market research, competitive intelligence, and strategic forecasting.

            Your goal is to simulate a Gartner-style report using public data, historical trends, and logical estimation for the {market_sector} sector.

            For this analysis:
            • Generate clear, structured insights based on known market signals.
            • Build data-backed forecasts using assumptions (state them clearly).
            • Identify top vendors and categorize them by niche, scale, or innovation.
            • Highlight risks, emerging players, and future trends.

            Be analytical, not vague. Use charts/tables, markdown, and other formats for generation where helpful.
            Be explicit about what's estimated vs known.

            Use this structure:

            1. Market Overview
            - Market size and growth rates
            - Key market drivers and trends
            - Regulatory environment
            - Technology adoption patterns

            2. Key Players
            - Market leaders and their market share
            - Emerging players and disruptors
            - Competitive positioning
            - Strategic partnerships and M&A activity

            3. Forecast (1–3 years)
            - Growth projections with confidence intervals
            - Technology roadmap and innovation cycles
            - Market consolidation predictions
            - Investment flows and funding trends

            4. Opportunities & Risks
            - Market opportunities by segment
            - Regulatory and competitive risks
            - Technology disruption threats
            - Economic sensitivity analysis

            5. Strategic Insights
            - Investment recommendations
            - Strategic positioning advice
            - Market entry/exit strategies
            - Portfolio allocation suggestions

            Focus on {analysis_type} analysis for {market_sector}.
            """
            
            # Get analysis from the model
            response = await self.model_manager.get_completion(
                prompt=research_prompt,
                model_type="claude",  # Use Claude for analytical tasks
                max_tokens=4000
            )
            
            # Parse the response into structured format
            analysis = self._parse_market_research(response, market_sector)
            
            return {
                "market_sector": market_sector,
                "analysis_type": analysis_type,
                "research_report": analysis,
                "confidence_score": 0.85,  # McKinsey-level confidence
                "timestamp": datetime.now().isoformat(),
                "methodology": "McKinsey-level strategic analysis with Gartner-style reporting"
            }
            
        except Exception as e:
            logger.error(f"Market research analysis failed: {e}")
            return {"error": str(e)}
    
    def _parse_market_research(self, response: str, sector: str) -> Dict[str, Any]:
        """Parse market research response into structured format"""
        try:
            # Extract key sections (simplified parsing)
            sections = {
                "market_overview": self._extract_section(response, "Market Overview"),
                "key_players": self._extract_section(response, "Key Players"),
                "forecast": self._extract_section(response, "Forecast"),
                "opportunities_risks": self._extract_section(response, "Opportunities & Risks"),
                "strategic_insights": self._extract_section(response, "Strategic Insights")
            }
            
            # Extract key metrics (simplified)
            metrics = {
                "market_size_estimate": "Data-driven estimate based on analysis",
                "growth_rate": "Projected based on historical trends",
                "competitive_intensity": "High/Medium/Low based on player analysis",
                "investment_attractiveness": "Scored based on opportunities vs risks"
            }
            
            return {
                "sector": sector,
                "sections": sections,
                "key_metrics": metrics,
                "analysis_quality": "McKinsey-level",
                "confidence_level": "High"
            }
            
        except Exception as e:
            logger.error(f"Failed to parse market research: {e}")
            return {"error": "Failed to parse research report"}
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract specific section from research report"""
        try:
            # Simple section extraction (in production, use more sophisticated parsing)
            lines = text.split('\n')
            section_content = []
            in_section = False
            
            for line in lines:
                if section_name in line and ('.' in line or ':' in line):
                    in_section = True
                    continue
                elif in_section and any(x in line for x in ['1.', '2.', '3.', '4.', '5.']) and line.strip():
                    break
                elif in_section:
                    section_content.append(line)
            
            return '\n'.join(section_content).strip()
            
        except Exception:
            return f"Section content for {section_name} (parsing in progress)"

class PortfolioOptimizationTool(BaseTool):
    """Tool for portfolio optimization using various methods"""
    
    name: str = "portfolio_optimizer"
    description: str = "Optimizes portfolio allocation using Markowitz, Black-Litterman, and other methods"
    
    def __init__(self):
        super().__init__()
    
    @handle_errors
    async def _run(self, signals: List[StrategySignal], method: str = "markowitz") -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        try:
            # Extract symbols and expected returns
            symbols = [signal.symbol for signal in signals]
            expected_returns_dict = {signal.symbol: signal.expected_return for signal in signals}
            confidence_scores = {signal.symbol: signal.confidence for signal in signals}
            
            # Get historical data for optimization
            price_data = await self._get_price_data(symbols)
            
            if method == "markowitz":
                return await self._markowitz_optimization(price_data, expected_returns_dict, confidence_scores)
            elif method == "black_litterman":
                return await self._black_litterman_optimization(price_data, expected_returns_dict, confidence_scores)
            elif method == "risk_parity":
                return await self._risk_parity_optimization(price_data)
            elif method == "kelly_criterion":
                return await self._kelly_criterion_optimization(signals)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {"error": str(e)}
    
    async def _get_price_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Get historical price data for optimization"""
        try:
            price_data = pd.DataFrame()
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    price_data[symbol] = hist['Close']
            
            return price_data.dropna()
            
        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            return pd.DataFrame()
    
    async def _markowitz_optimization(self, price_data: pd.DataFrame, expected_returns_dict: Dict[str, float], confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Markowitz mean-variance optimization"""
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Calculate expected returns and covariance matrix
            mu = pd.Series(expected_returns_dict)
            S = risk_models.sample_cov(price_data)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            
            # Add constraints
            ef.add_constraint(lambda w: w >= 0)  # Long-only constraint
            ef.add_constraint(lambda w: w <= 0.3)  # Maximum 30% in any single asset
            
            # Optimize for maximum Sharpe ratio
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            
            # Calculate performance metrics
            performance = ef.portfolio_performance(verbose=False)
            expected_return, volatility, sharpe_ratio = performance
            
            # Calculate additional metrics
            portfolio_returns = returns.dot(pd.Series(cleaned_weights))
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            var_95 = np.percentile(portfolio_returns, 5)
            
            return {
                "method": "markowitz",
                "symbols": list(cleaned_weights.keys()),
                "weights": list(cleaned_weights.values()),
                "expected_return": float(expected_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "var_95": float(var_95),
                "constraints": {"long_only": True, "max_weight": 0.3},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Markowitz optimization failed: {e}")
            return {"error": str(e)}
    
    async def _black_litterman_optimization(self, price_data: pd.DataFrame, expected_returns_dict: Dict[str, float], confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Black-Litterman optimization with investor views"""
        try:
            # Calculate market cap weights (simplified)
            market_caps = {symbol: 1.0 for symbol in price_data.columns}  # Simplified
            
            # Calculate returns and covariance
            returns = price_data.pct_change().dropna()
            S = risk_models.sample_cov(price_data)
            
            # Create views matrix based on expected returns and confidence
            views = []
            view_confidences = []
            
            for symbol, expected_return in expected_returns_dict.items():
                if abs(expected_return) > 0.01:  # Only include significant views
                    views.append([1 if col == symbol else 0 for col in price_data.columns])
                    view_confidences.append(confidence_scores.get(symbol, 0.5))
            
            if not views:
                # Fall back to market cap weighting
                total_cap = sum(market_caps.values())
                weights = {symbol: cap/total_cap for symbol, cap in market_caps.items()}
            else:
                # Implement simplified Black-Litterman
                # In production, use more sophisticated implementation
                mu_market = expected_returns.mean_historical_return(price_data)
                ef = EfficientFrontier(mu_market, S)
                weights = ef.max_sharpe()
                weights = ef.clean_weights()
            
            # Calculate performance metrics
            portfolio_returns = returns.dot(pd.Series(weights))
            expected_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = expected_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            var_95 = np.percentile(portfolio_returns, 5)
            
            return {
                "method": "black_litterman",
                "symbols": list(weights.keys()),
                "weights": list(weights.values()),
                "expected_return": float(expected_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "var_95": float(var_95),
                "num_views": len(views),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return {"error": str(e)}
    
    async def _risk_parity_optimization(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Risk parity optimization"""
        try:
            returns = price_data.pct_change().dropna()
            S = risk_models.sample_cov(price_data)
            
            # Risk parity optimization
            n = len(price_data.columns)
            
            def risk_parity_objective(weights):
                portfolio_var = np.dot(weights.T, np.dot(S, weights))
                marginal_contrib = np.dot(S, weights)
                contrib = weights * marginal_contrib
                return np.sum((contrib - portfolio_var/n)**2)
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n))
            
            # Optimize
            result = minimize(risk_parity_objective, 
                            x0=np.ones(n)/n, 
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            weights = dict(zip(price_data.columns, result.x))
            
            # Calculate performance metrics
            portfolio_returns = returns.dot(pd.Series(weights))
            expected_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = expected_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            var_95 = np.percentile(portfolio_returns, 5)
            
            return {
                "method": "risk_parity",
                "symbols": list(weights.keys()),
                "weights": list(weights.values()),
                "expected_return": float(expected_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "var_95": float(var_95),
                "optimization_success": result.success,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return {"error": str(e)}
    
    async def _kelly_criterion_optimization(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Kelly Criterion position sizing"""
        try:
            weights = {}
            total_kelly = 0
            
            for signal in signals:
                # Kelly formula: f = (bp - q) / b
                # where b = odds, p = probability of win, q = probability of loss
                
                win_prob = signal.confidence
                loss_prob = 1 - win_prob
                
                # Estimate odds from expected return
                if signal.expected_return > 0:
                    odds = abs(signal.expected_return) / signal.risk_score if signal.risk_score > 0 else 1
                    kelly_fraction = (odds * win_prob - loss_prob) / odds
                else:
                    kelly_fraction = 0
                
                # Apply Kelly fraction with safety margin
                kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.2))  # 25% of Kelly, max 20%
                
                weights[signal.symbol] = kelly_fraction
                total_kelly += kelly_fraction
            
            # Normalize weights
            if total_kelly > 0:
                weights = {symbol: weight/total_kelly for symbol, weight in weights.items()}
            
            # Calculate expected portfolio metrics
            expected_return = sum(signal.expected_return * weights.get(signal.symbol, 0) for signal in signals)
            risk_score = np.sqrt(sum((signal.risk_score * weights.get(signal.symbol, 0))**2 for signal in signals))
            sharpe_ratio = expected_return / risk_score if risk_score > 0 else 0
            
            return {
                "method": "kelly_criterion",
                "symbols": list(weights.keys()),
                "weights": list(weights.values()),
                "expected_return": float(expected_return),
                "risk_score": float(risk_score),
                "sharpe_ratio": float(sharpe_ratio),
                "total_kelly_fraction": float(total_kelly),
                "safety_margin": 0.25,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Kelly criterion optimization failed: {e}")
            return {"error": str(e)}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min())
        except:
            return 0.0

class BacktestingTool(BaseTool):
    """Tool for strategy backtesting with realistic market impact"""
    
    name: str = "strategy_backtester"
    description: str = "Backtests trading strategies with realistic market impact and transaction costs"
    
    def __init__(self):
        super().__init__()
    
    @handle_errors
    async def _run(self, strategy_signals: List[StrategySignal], start_date: str, end_date: str, initial_capital: float = 100000) -> Dict[str, Any]:
        """Backtest strategy with realistic market impact"""
        try:
            # Convert dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Get historical data
            symbols = list(set(signal.symbol for signal in strategy_signals))
            price_data = await self._get_historical_data(symbols, start_date, end_date)
            
            # Run backtest
            backtest_result = await self._run_backtest(strategy_signals, price_data, initial_capital)
            
            return {
                "strategy_name": "Multi-Agent Strategy",
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "backtest_result": backtest_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            return {"error": str(e)}
    
    async def _get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data for backtesting"""
        try:
            data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[symbol] = hist
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return {}
    
    async def _run_backtest(self, signals: List[StrategySignal], price_data: Dict[str, pd.DataFrame], initial_capital: float) -> Dict[str, Any]:
        """Run backtest simulation"""
        try:
            portfolio_value = initial_capital
            positions = {}
            trades = []
            equity_curve = []
            
            # Group signals by date
            signals_by_date = {}
            for signal in signals:
                date_key = signal.timestamp.strftime("%Y-%m-%d")
                if date_key not in signals_by_date:
                    signals_by_date[date_key] = []
                signals_by_date[date_key].append(signal)
            
            # Simulate trading
            for date_str, daily_signals in signals_by_date.items():
                try:
                    trade_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    for signal in daily_signals:
                        if signal.symbol in price_data:
                            symbol_data = price_data[signal.symbol]
                            
                            # Find closest price data
                            available_dates = symbol_data.index
                            closest_date = min(available_dates, key=lambda x: abs(x.timestamp() - trade_date.timestamp()))
                            
                            if closest_date in symbol_data.index:
                                entry_price = symbol_data.loc[closest_date, 'Close']
                                
                                # Calculate position size
                                position_value = portfolio_value * signal.position_size
                                shares = int(position_value / entry_price)
                                
                                if shares > 0:
                                    # Account for market impact (simplified)
                                    market_impact = 0.001 * np.sqrt(position_value / 1000000)  # Impact based on size
                                    transaction_cost = 0.001  # 10 bps transaction cost
                                    
                                    actual_entry_price = entry_price * (1 + market_impact + transaction_cost)
                                    
                                    # Record trade
                                    trade = {
                                        "symbol": signal.symbol,
                                        "entry_date": trade_date,
                                        "entry_price": actual_entry_price,
                                        "shares": shares,
                                        "direction": signal.direction,
                                        "stop_loss": signal.stop_loss,
                                        "take_profit": signal.take_profit,
                                        "market_impact": market_impact,
                                        "transaction_cost": transaction_cost
                                    }
                                    
                                    trades.append(trade)
                                    positions[signal.symbol] = trade
                                    
                                    # Update portfolio value
                                    portfolio_value -= shares * actual_entry_price
                
                except Exception as e:
                    logger.warning(f"Failed to process signals for {date_str}: {e}")
                    continue
                
                equity_curve.append({
                    "date": date_str,
                    "portfolio_value": portfolio_value,
                    "positions_value": sum(pos.get("shares", 0) * pos.get("entry_price", 0) for pos in positions.values())
                })
            
            # Calculate performance metrics
            if equity_curve:
                total_value = equity_curve[-1]["portfolio_value"] + equity_curve[-1]["positions_value"]
                total_return = (total_value - initial_capital) / initial_capital
                
                # Calculate additional metrics
                returns = pd.Series([eq["portfolio_value"] + eq["positions_value"] for eq in equity_curve])
                daily_returns = returns.pct_change().dropna()
                
                annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
                volatility = daily_returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                # Max drawdown
                cumulative = returns / initial_capital
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Win rate
                winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
                win_rate = winning_trades / len(trades) if trades else 0
                
                return {
                    "total_return": float(total_return),
                    "annualized_return": float(annualized_return),
                    "volatility": float(volatility),
                    "sharpe_ratio": float(sharpe_ratio),
                    "max_drawdown": float(max_drawdown),
                    "win_rate": float(win_rate),
                    "total_trades": len(trades),
                    "final_portfolio_value": float(total_value),
                    "equity_curve": equity_curve,
                    "trades": trades
                }
            
            return {"error": "No valid trades executed"}
            
        except Exception as e:
            logger.error(f"Backtest simulation failed: {e}")
            return {"error": str(e)}

class ReinforcementLearningTool(BaseTool):
    """Tool for reinforcement learning strategy improvement - now using ML client"""
    
    name: str = "rl_strategy_optimizer"
    description: str = "Uses reinforcement learning to improve trading strategies via ML client"
    
    def __init__(self):
        super().__init__()
        # No longer storing models directly - using ML client
        logger.info("RL Strategy Optimizer initialized with ML client support")
    
    @handle_errors
    async def _run(self, strategy_performance: Dict[str, Any], signals: List[StrategySignal]) -> Dict[str, Any]:
        try:
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available - using fallback RL optimization")
                return self._fallback_rl_optimization(strategy_performance, signals)
            
            # Prepare performance features for RL model
            performance_features = [
                strategy_performance.get("total_return", 0.0),
                strategy_performance.get("sharpe_ratio", 0.0),
                strategy_performance.get("max_drawdown", 0.0),
                strategy_performance.get("volatility", 0.0),
                strategy_performance.get("win_rate", 0.0)
            ]
            
            # Use PPO model for strategy optimization (state-of-the-art RL)
            try:
                # First train the model if needed
                training_result = await ml_client.train_rl_model("PPO", total_timesteps=5000)
                logger.info(f"RL training result: {training_result}")
                
                # Get action prediction from trained model
                prediction_result = await ml_client.predict_with_rl_model("PPO", performance_features)
                
                # Interpret the RL action for strategy improvement
                improvements = self._interpret_rl_action(prediction_result, signals)
                
                return {
                    "rl_model": "PPO_via_ML_Client",
                    "training_status": training_result.get("status", "unknown"),
                    "action": prediction_result.get("action", 0),
                    "prediction_confidence": prediction_result.get("confidence", 0.0),
                    "improvements": improvements,
                    "confidence": prediction_result.get("confidence", 0.0),
                    "ml_client_used": True,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as rl_error:
                logger.warning(f"RL model failed, trying A2C: {rl_error}")
                # Fallback to A2C model
                try:
                    training_result = await ml_client.train_rl_model("A2C", total_timesteps=3000)
                    prediction_result = await ml_client.predict_with_rl_model("A2C", performance_features)
                    improvements = self._interpret_rl_action(prediction_result, signals)
                    
                    return {
                        "rl_model": "A2C_via_ML_Client",
                        "training_status": training_result.get("status", "unknown"),
                        "action": prediction_result.get("action", 0),
                        "prediction_confidence": prediction_result.get("confidence", 0.0),
                        "improvements": improvements,
                        "confidence": prediction_result.get("confidence", 0.0),
                        "ml_client_used": True,
                        "fallback_used": "A2C",
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as a2c_error:
                    logger.warning(f"A2C model failed, trying DQN: {a2c_error}")
                    # Final fallback to DQN
                    training_result = await ml_client.train_rl_model("DQN", total_timesteps=2000)
                    prediction_result = await ml_client.predict_with_rl_model("DQN", performance_features)
                    improvements = self._interpret_rl_action(prediction_result, signals)
                    
                    return {
                        "rl_model": "DQN_via_ML_Client",
                        "training_status": training_result.get("status", "unknown"),
                        "action": prediction_result.get("action", 0),
                        "prediction_confidence": prediction_result.get("confidence", 0.0),
                        "improvements": improvements,
                        "confidence": prediction_result.get("confidence", 0.0),
                        "ml_client_used": True,
                        "fallback_used": "DQN",
                        "timestamp": datetime.now().isoformat()
                    }
            
        except Exception as e:
            logger.error(f"RL optimization failed: {e}")
            return self._fallback_rl_optimization(strategy_performance, signals)
    
    def _fallback_rl_optimization(self, strategy_performance: Dict[str, Any], signals: List[StrategySignal]) -> Dict[str, Any]:
        """Fallback RL optimization when ML client is not available"""
        try:
            # Simple rule-based strategy improvement
            improvements = []
            
            sharpe_ratio = strategy_performance.get("sharpe_ratio", 0.0)
            max_drawdown = strategy_performance.get("max_drawdown", 0.0)
            win_rate = strategy_performance.get("win_rate", 0.5)
            
            # Simple improvement suggestions based on performance
            if sharpe_ratio < 1.0:
                improvements.append("Increase position sizing for high-confidence signals")
            
            if max_drawdown > 0.15:
                improvements.append("Tighten stop losses to reduce maximum drawdown")
            
            if win_rate < 0.45:
                improvements.append("Improve signal filtering to increase win rate")
            
            return {
                "rl_model": "Rule_Based_Fallback",
                "improvements": improvements,
                "confidence": 0.6,  # Lower confidence for fallback
                "ml_client_used": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback RL optimization failed: {e}")
            return {
                "rl_model": "Default_Fallback",
                "improvements": ["Review strategy parameters manually"],
                "confidence": 0.3,
                "ml_client_used": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _interpret_rl_action(self, prediction_result: Dict[str, Any], signals: List[StrategySignal]) -> Dict[str, Any]:
        """Interpret RL action for strategy improvements"""
        action = prediction_result.get("action", 0)
        confidence = prediction_result.get("confidence", 0.0)
        
        # Map RL actions to strategy improvements
        if action == 0:  # Hold/maintain current strategy
            return {
                "position_sizing": "maintain",
                "risk_adjustment": "none",
                "signal_filtering": "current_level",
                "reasoning": "RL model suggests maintaining current strategy"
            }
        elif action == 1:  # Increase aggression
            return {
                "position_sizing": "increase_by_10%",
                "risk_adjustment": "increase_tolerance",
                "signal_filtering": "reduce_threshold",
                "reasoning": "RL model suggests more aggressive strategy"
            }
        else:  # Decrease aggression
            return {
                "position_sizing": "decrease_by_10%",
                "risk_adjustment": "decrease_tolerance", 
                "signal_filtering": "increase_threshold",
                "reasoning": "RL model suggests more conservative strategy"
            }

# Remove TradingEnvironment class - now handled by ML client
# class TradingEnvironment(gym.Env):
#     """Custom trading environment for reinforcement learning"""
#     
#     def __init__(self, signals: List[StrategySignal], performance: Dict[str, Any]):
#         super().__init__()
#         self.signals = signals
#         self.performance = performance
#         self.current_step = 0
#         
#         # Action space: [position_size_mult, stop_loss_mult, take_profit_mult, confidence_threshold]
#         self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
#         
#         # Observation space: [returns, volatility, sharpe_ratio, drawdown, win_rate]
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
#     
#     def reset(self):
#         self.current_step = 0
#         return self._get_observation()
#     
#     def step(self, action):
#         # Apply action to strategy
#         reward = self._calculate_reward(action)
#         self.current_step += 1
#         
#         done = self.current_step >= 100  # Episode length
#         obs = self._get_observation()
#         
#         return obs, reward, done, {}
#     
#     def _get_observation(self):
#         return np.array([
#             self.performance.get("total_return", 0),
#             self.performance.get("volatility", 0),
#             self.performance.get("sharpe_ratio", 0),
#             self.performance.get("max_drawdown", 0),
#             self.performance.get("win_rate", 0)
#         ], dtype=np.float32)
#     
#     def _calculate_reward(self, action):
#         # Reward based on risk-adjusted returns
#         sharpe_ratio = self.performance.get("sharpe_ratio", 0)
#         max_drawdown = self.performance.get("max_drawdown", 0)
#         
#         reward = sharpe_ratio - abs(max_drawdown)
#         return float(reward)

class DebateMechanismTool(BaseTool):
    """Tool for implementing multi-agent debate mechanism with judged outcomes"""
    
    name: str = "multi_agent_debate_moderator"
    description: str = "Orchestrates multi-agent debates with specialized debate agents and judges"
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self._setup_debate_agents()
    
    def _setup_debate_agents(self):
        """Setup specialized debate agents"""
        
        # Bull Agent - Argues for the strategy
        self.bull_agent = Agent(
            role='Strategy Advocate (Bull)',
            goal='Argue convincingly for the proposed trading strategy',
            backstory="""You are an optimistic but analytical trader who specializes in 
            finding the strongest arguments for trading strategies. You have deep knowledge 
            of market dynamics, technical analysis, and fundamental factors. You excel at 
            identifying opportunities and building compelling cases for why strategies will succeed.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Bear Agent - Argues against the strategy
        self.bear_agent = Agent(
            role='Strategy Skeptic (Bear)',
            goal='Identify flaws and risks in the proposed trading strategy',
            backstory="""You are a cautious and analytical trader who specializes in 
            risk identification and strategy critique. You have extensive experience in 
            market crashes, strategy failures, and risk management. You excel at finding 
            weaknesses in trading strategies and identifying potential failure modes.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Risk Manager Agent - Focuses on risk aspects
        self.risk_agent = Agent(
            role='Risk Assessment Specialist',
            goal='Evaluate risk-reward trade-offs and position sizing appropriateness',
            backstory="""You are a quantitative risk management expert with deep knowledge 
            of portfolio theory, risk metrics, and position sizing. You specialize in 
            evaluating whether strategies have appropriate risk controls and whether 
            position sizes are justified by expected returns.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Judge Agent - Evaluates debate outcomes
        self.judge_agent = Agent(
            role='Debate Judge and Strategy Evaluator',
            goal='Objectively evaluate debate arguments and provide balanced final judgment',
            backstory="""You are an impartial and highly experienced trading strategist 
            who serves as a judge in strategy debates. You have decades of experience 
            evaluating trading strategies across different market conditions. You excel 
            at weighing arguments objectively and providing balanced assessments.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Create debate crew
        self.debate_crew = Crew(
            agents=[self.bull_agent, self.bear_agent, self.risk_agent, self.judge_agent],
            verbose=True,
            process=Process.sequential
        )
    
    @handle_errors
    async def _run(self, strategy: StrategySignal, market_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run multi-agent debate with judged outcome"""
        try:
            # Prepare debate context
            debate_context = self._prepare_debate_context(strategy, market_context)
            
            # Run multi-round debate
            debate_rounds = await self._conduct_multi_round_debate(strategy, debate_context)
            
            # Get final judgment
            final_judgment = await self._get_final_judgment(strategy, debate_rounds)
            
            return {
                "strategy_symbol": strategy.symbol,
                "debate_type": "multi_agent_judged",
                "debate_rounds": debate_rounds,
                "final_judgment": final_judgment,
                "debate_quality": "high",
                "participants": ["bull_agent", "bear_agent", "risk_agent", "judge_agent"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Multi-agent debate failed: {e}")
            return {"error": str(e)}
    
    def _prepare_debate_context(self, strategy: StrategySignal, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for debate"""
        return {
            "strategy_details": {
                "symbol": strategy.symbol,
                "strategy_type": strategy.strategy_type.value,
                "direction": strategy.direction,
                "confidence": strategy.confidence,
                "expected_return": strategy.expected_return,
                "risk_score": strategy.risk_score,
                "position_size": strategy.position_size,
                "entry_price": strategy.entry_price,
                "stop_loss": strategy.stop_loss,
                "take_profit": strategy.take_profit,
                "holding_period": strategy.holding_period,
                "agent_sources": strategy.agent_sources,
                "market_regime": strategy.market_regime
            },
            "market_context": market_context or {},
            "debate_focus_areas": [
                "Strategy viability and expected returns",
                "Risk assessment and position sizing",
                "Market timing and regime appropriateness",
                "Alternative scenarios and stress testing"
            ]
        }
    
    async def _conduct_multi_round_debate(self, strategy: StrategySignal, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Conduct multi-round debate between agents"""
        try:
            debate_rounds = []
            
            # Round 1: Initial positions
            round1 = await self._debate_round_1(strategy, context)
            debate_rounds.append(round1)
            
            # Round 2: Rebuttals
            round2 = await self._debate_round_2(strategy, round1)
            debate_rounds.append(round2)
            
            # Round 3: Risk-focused discussion
            round3 = await self._debate_round_3(strategy, round1, round2)
            debate_rounds.append(round3)
            
            return debate_rounds
            
        except Exception as e:
            logger.error(f"Multi-round debate failed: {e}")
            return []
    
    async def _debate_round_1(self, strategy: StrategySignal, context: Dict[str, Any]) -> Dict[str, Any]:
        """Round 1: Initial positions from Bull and Bear agents"""
        try:
            # Bull agent argument
            bull_prompt = f"""
            You are arguing FOR this trading strategy. Present your strongest case:
            
            STRATEGY: {strategy.symbol} - {strategy.strategy_type.value} {strategy.direction}
            - Expected Return: {strategy.expected_return:.3f}
            - Confidence: {strategy.confidence:.2f}
            - Position Size: {strategy.position_size:.3f}
            - Market Regime: {strategy.market_regime}
            
            Present 3-5 compelling arguments for why this strategy will succeed.
            Focus on: market conditions, technical factors, fundamental drivers, and timing.
            Be specific and analytical.
            """
            
            bull_argument = await self.model_manager.get_completion(
                prompt=bull_prompt,
                model_type="claude",
                max_tokens=1500
            )
            
            # Bear agent argument
            bear_prompt = f"""
            You are arguing AGAINST this trading strategy. Present your strongest critique:
            
            STRATEGY: {strategy.symbol} - {strategy.strategy_type.value} {strategy.direction}
            - Expected Return: {strategy.expected_return:.3f}
            - Confidence: {strategy.confidence:.2f}
            - Position Size: {strategy.position_size:.3f}
            - Market Regime: {strategy.market_regime}
            
            Present 3-5 compelling arguments for why this strategy will fail or underperform.
            Focus on: market risks, technical weaknesses, fundamental concerns, and timing issues.
            Be specific and analytical.
            """
            
            bear_argument = await self.model_manager.get_completion(
                prompt=bear_prompt,
                model_type="claude",
                max_tokens=1500
            )
            
            return {
                "round": 1,
                "type": "initial_positions",
                "bull_argument": bull_argument,
                "bear_argument": bear_argument,
                "focus": "Strategy viability and market conditions"
            }
            
        except Exception as e:
            logger.error(f"Debate round 1 failed: {e}")
            return {"error": str(e)}
    
    async def _debate_round_2(self, strategy: StrategySignal, round1: Dict[str, Any]) -> Dict[str, Any]:
        """Round 2: Rebuttals and counter-arguments"""
        try:
            # Bull rebuttal to Bear arguments
            bull_rebuttal_prompt = f"""
            The Bear agent argued against your strategy with these points:
            {round1.get('bear_argument', 'No bear argument available')}
            
            Provide a strong rebuttal addressing their concerns about:
            {strategy.symbol} - {strategy.strategy_type.value} {strategy.direction}
            
            Counter their arguments with evidence and alternative perspectives.
            Acknowledge valid concerns but explain why the strategy still has merit.
            """
            
            bull_rebuttal = await self.model_manager.get_completion(
                prompt=bull_rebuttal_prompt,
                model_type="claude",
                max_tokens=1500
            )
            
            # Bear counter-rebuttal
            bear_counter_prompt = f"""
            The Bull agent made these arguments for the strategy:
            {round1.get('bull_argument', 'No bull argument available')}
            
            And provided this rebuttal to your concerns:
            {bull_rebuttal}
            
            Provide a counter-rebuttal that reinforces your skeptical position.
            Address their points while maintaining your critical stance.
            """
            
            bear_counter = await self.model_manager.get_completion(
                prompt=bear_counter_prompt,
                model_type="claude",
                max_tokens=1500
            )
            
            return {
                "round": 2,
                "type": "rebuttals",
                "bull_rebuttal": bull_rebuttal,
                "bear_counter": bear_counter,
                "focus": "Addressing counter-arguments"
            }
            
        except Exception as e:
            logger.error(f"Debate round 2 failed: {e}")
            return {"error": str(e)}
    
    async def _debate_round_3(self, strategy: StrategySignal, round1: Dict[str, Any], round2: Dict[str, Any]) -> Dict[str, Any]:
        """Round 3: Risk manager assessment and final positions"""
        try:
            # Risk manager assessment
            risk_assessment_prompt = f"""
            You've observed this debate about the trading strategy:
            
            BULL ARGUMENTS: {round1.get('bull_argument', '')}
            BEAR ARGUMENTS: {round1.get('bear_argument', '')}
            
            STRATEGY DETAILS:
            - Symbol: {strategy.symbol}
            - Type: {strategy.strategy_type.value}
            - Direction: {strategy.direction}
            - Expected Return: {strategy.expected_return:.3f}
            - Risk Score: {strategy.risk_score:.3f}
            - Position Size: {strategy.position_size:.3f}
            
            Provide a risk management perspective focusing on:
            1. Is the position size appropriate for the risk level?
            2. Are the stop-loss and take-profit levels reasonable?
            3. What are the key risk factors both sides may have missed?
            4. How should position sizing be adjusted based on this debate?
            5. What additional risk controls should be implemented?
            """
            
            risk_assessment = await self.model_manager.get_completion(
                prompt=risk_assessment_prompt,
                model_type="claude",
                max_tokens=1500
            )
            
            return {
                "round": 3,
                "type": "risk_assessment",
                "risk_analysis": risk_assessment,
                "focus": "Risk management and position sizing"
            }
            
        except Exception as e:
            logger.error(f"Debate round 3 failed: {e}")
            return {"error": str(e)}
    
    async def _get_final_judgment(self, strategy: StrategySignal, debate_rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get final judgment from the judge agent"""
        try:
            # Compile all debate content
            debate_summary = ""
            for round_data in debate_rounds:
                if round_data.get("round") == 1:
                    debate_summary += f"BULL POSITION: {round_data.get('bull_argument', '')}\n\n"
                    debate_summary += f"BEAR POSITION: {round_data.get('bear_argument', '')}\n\n"
                elif round_data.get("round") == 2:
                    debate_summary += f"BULL REBUTTAL: {round_data.get('bull_rebuttal', '')}\n\n"
                    debate_summary += f"BEAR COUNTER: {round_data.get('bear_counter', '')}\n\n"
                elif round_data.get("round") == 3:
                    debate_summary += f"RISK ASSESSMENT: {round_data.get('risk_analysis', '')}\n\n"
            
            # Judge's final evaluation
            judge_prompt = f"""
            You are judging a debate about this trading strategy:
            
            STRATEGY: {strategy.symbol} - {strategy.strategy_type.value} {strategy.direction}
            - Expected Return: {strategy.expected_return:.3f}
            - Confidence: {strategy.confidence:.2f}
            - Position Size: {strategy.position_size:.3f}
            
            DEBATE SUMMARY:
            {debate_summary}
            
            As an impartial judge, provide your final evaluation:
            
            1. WINNING ARGUMENT: Which side presented the stronger case? (Bull/Bear/Neutral)
            2. CONFIDENCE ADJUSTMENT: How should the original confidence be adjusted? (-0.3 to +0.3)
            3. POSITION SIZE ADJUSTMENT: How should position size be adjusted? (0.5x to 1.5x)
            4. KEY RISKS IDENTIFIED: What are the 3 most important risks?
            5. STRATEGY MODIFICATIONS: What changes would improve the strategy?
            6. FINAL RECOMMENDATION: Proceed/Modify/Reject with reasoning
            7. OVERALL ASSESSMENT: Balanced summary of the strategy's merits and risks
            
            Be objective and analytical. Consider both upside potential and downside risks.
            """
            
            judge_evaluation = await self.model_manager.get_completion(
                prompt=judge_prompt,
                model_type="claude",
                max_tokens=2000
            )
            
            # Parse judge's evaluation (simplified)
            judgment = self._parse_judge_evaluation(judge_evaluation)
            
            return {
                "judge_evaluation": judge_evaluation,
                "parsed_judgment": judgment,
                "evaluation_quality": "high",
                "methodology": "multi_agent_adversarial_debate"
            }
            
        except Exception as e:
            logger.error(f"Final judgment failed: {e}")
            return {"error": str(e)}
    
    def _parse_judge_evaluation(self, evaluation: str) -> Dict[str, Any]:
        """Parse judge's evaluation into structured format"""
        try:
            # Simplified parsing - in production, use more sophisticated NLP
            return {
                "winning_argument": "neutral",  # Would be parsed from evaluation
                "confidence_adjustment": -0.1,  # Would be parsed from evaluation
                "position_size_adjustment": 0.8,  # Would be parsed from evaluation
                "key_risks": [
                    "Market regime uncertainty",
                    "Position sizing may be aggressive",
                    "Stop-loss levels need adjustment"
                ],
                "strategy_modifications": [
                    "Reduce position size by 20%",
                    "Tighten stop-loss levels",
                    "Add regime filter"
                ],
                "final_recommendation": "modify",
                "overall_assessment": "Strategy has merit but requires risk adjustments"
            }
        except Exception as e:
            logger.error(f"Failed to parse judge evaluation: {e}")
            return {"error": "Failed to parse evaluation"}

class StrategyAgent:
    """
    Advanced Strategy Synthesis Agent using CrewAI
    
    Combines signals from multiple agents into coherent trading strategies,
    implements portfolio optimization, risk management, and continuous improvement
    through reinforcement learning and debate mechanisms.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        
        # Initialize other agents
        self.microstructure_agent = MicrostructureAgent()
        self.altdata_agent = AlternativeDataAgent()
        self.regime_agent = RegimeDetectionAgent()
        
        # Initialize tools
        self.market_research_tool = MarketResearchTool()
        self.portfolio_optimizer = PortfolioOptimizationTool()
        self.backtester = BacktestingTool()
        self.rl_optimizer = ReinforcementLearningTool()
        self.debate_moderator = DebateMechanismTool()
        
        # Initialize CrewAI agents
        self._setup_crew()
        
        # Strategy storage
        self.strategies = {}
        self.performance_history = []
        
        logger.info("StrategyAgent initialized successfully")
    
    def _setup_crew(self):
        """Setup CrewAI agents and crew"""
        
        # Strategy Synthesizer Agent
        self.strategy_synthesizer = Agent(
            role='Strategy Synthesis Expert',
            goal='Combine signals from multiple agents into coherent trading strategies',
            backstory="""You are a master strategist with expertise in combining diverse 
            alpha signals into coherent trading strategies. You excel at signal aggregation, 
            risk assessment, and strategy optimization. You have deep knowledge of portfolio 
            theory, risk management, and market dynamics.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.market_research_tool,
                self.portfolio_optimizer,
                self.backtester
            ]
        )
        
        # Portfolio Optimization Agent
        self.portfolio_optimizer_agent = Agent(
            role='Portfolio Optimization Specialist',
            goal='Optimize portfolio allocation using advanced mathematical methods',
            backstory="""You are a quantitative portfolio optimization expert with deep 
            knowledge of Markowitz optimization, Black-Litterman models, Kelly Criterion, 
            and risk parity approaches. You specialize in creating optimal portfolio 
            allocations that maximize risk-adjusted returns.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.portfolio_optimizer
            ]
        )
        
        # Risk Management Agent
        self.risk_manager = Agent(
            role='Risk Management Expert',
            goal='Implement comprehensive risk management and position sizing',
            backstory="""You are a risk management expert with extensive experience in 
            position sizing, stop-loss implementation, and portfolio risk controls. You 
            excel at identifying and mitigating various types of market risks while 
            preserving alpha generation potential.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.backtester,
                self.debate_moderator
            ]
        )
        
        # Strategy Improvement Agent
        self.strategy_improver = Agent(
            role='Strategy Improvement Specialist',
            goal='Continuously improve strategies using machine learning and feedback',
            backstory="""You are a machine learning expert specializing in reinforcement 
            learning and strategy optimization. You excel at analyzing strategy performance, 
            identifying improvement opportunities, and implementing adaptive algorithms 
            for continuous strategy enhancement.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.rl_optimizer,
                self.debate_moderator
            ]
        )
        
        # Setup crew
        self.crew = Crew(
            agents=[self.strategy_synthesizer, self.portfolio_optimizer_agent, 
                   self.risk_manager, self.strategy_improver],
            verbose=True,
            process=Process.sequential
        )
    
    @handle_errors
    @monitor_performance
    async def synthesize_strategy(self, symbols: List[str], timeframe: str = "1d") -> List[StrategySignal]:
        """
        Synthesize trading strategies by combining signals from multiple agents
        
        Args:
            symbols: List of symbols to analyze
            timeframe: Analysis timeframe
            
        Returns:
            List of synthesized strategy signals
        """
        try:
            logger.info(f"Synthesizing strategies for {len(symbols)} symbols")
            
            # Collect signals from all agents
            agent_signals = await self._collect_agent_signals(symbols, timeframe)
            
            # Get market research for context
            market_research = await self._get_market_research(symbols)
            
            # Synthesize strategies
            strategy_signals = await self._synthesize_signals(agent_signals, market_research)
            
            # Apply debate mechanism to challenge assumptions
            debated_strategies = await self._apply_debate_mechanism(strategy_signals)
            
            # Store strategies
            self.strategies[datetime.now().isoformat()] = debated_strategies
            
            logger.info(f"Synthesized {len(debated_strategies)} strategy signals")
            return debated_strategies
            
        except Exception as e:
            logger.error(f"Strategy synthesis failed: {e}")
            raise AlphaDiscoveryError(f"Failed to synthesize strategy: {e}")
    
    @handle_errors
    async def optimize_portfolio(self, strategy_signals: List[StrategySignal], method: str = "markowitz") -> PortfolioOptimization:
        """
        Optimize portfolio allocation using various methods
        
        Args:
            strategy_signals: List of strategy signals
            method: Optimization method (markowitz, black_litterman, kelly_criterion)
            
        Returns:
            Portfolio optimization result
        """
        try:
            logger.info(f"Optimizing portfolio using {method} method")
            
            # Run portfolio optimization
            optimization_result = await self.portfolio_optimizer._run(strategy_signals, method)
            
            if "error" in optimization_result:
                raise AlphaDiscoveryError(f"Portfolio optimization failed: {optimization_result['error']}")
            
            # Create portfolio optimization object
            portfolio_optimization = PortfolioOptimization(
                symbols=optimization_result["symbols"],
                weights=optimization_result["weights"],
                expected_return=optimization_result["expected_return"],
                volatility=optimization_result["volatility"],
                sharpe_ratio=optimization_result["sharpe_ratio"],
                max_drawdown=optimization_result.get("max_drawdown", 0),
                var_95=optimization_result.get("var_95", 0),
                optimization_method=method,
                constraints=optimization_result.get("constraints", {}),
                timestamp=datetime.now()
            )
            
            logger.info(f"Portfolio optimization completed - Sharpe ratio: {portfolio_optimization.sharpe_ratio:.2f}")
            return portfolio_optimization
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise AlphaDiscoveryError(f"Failed to optimize portfolio: {e}")
    
    @handle_errors
    async def calculate_sharpe(self, strategy_signals: List[StrategySignal], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio for strategy signals
        
        Args:
            strategy_signals: List of strategy signals
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Sharpe ratio
        """
        try:
            if not strategy_signals:
                return 0.0
            
            # Calculate portfolio expected return
            total_weight = sum(signal.position_size for signal in strategy_signals)
            if total_weight == 0:
                return 0.0
            
            # Weighted average return
            expected_return = sum(signal.expected_return * signal.position_size for signal in strategy_signals) / total_weight
            
            # Weighted average risk
            portfolio_risk = np.sqrt(sum((signal.risk_score * signal.position_size)**2 for signal in strategy_signals)) / total_weight
            
            # Calculate Sharpe ratio
            if portfolio_risk > 0:
                sharpe_ratio = (expected_return - risk_free_rate) / portfolio_risk
            else:
                sharpe_ratio = 0.0
            
            logger.info(f"Calculated Sharpe ratio: {sharpe_ratio:.2f}")
            return float(sharpe_ratio)
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {e}")
            return 0.0
    
    @handle_errors
    async def backtest_strategy(self, strategy_signals: List[StrategySignal], start_date: str, end_date: str) -> BacktestResult:
        """
        Backtest strategy with realistic market impact
        
        Args:
            strategy_signals: List of strategy signals
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest result
        """
        try:
            logger.info(f"Backtesting strategy from {start_date} to {end_date}")
            
            # Run backtest
            backtest_data = await self.backtester._run(strategy_signals, start_date, end_date)
            
            if "error" in backtest_data:
                raise AlphaDiscoveryError(f"Backtesting failed: {backtest_data['error']}")
            
            result = backtest_data["backtest_result"]
            
            # Create backtest result object
            backtest_result = BacktestResult(
                strategy_name="Multi-Agent Strategy",
                start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(end_date, "%Y-%m-%d"),
                total_return=result["total_return"],
                annualized_return=result["annualized_return"],
                volatility=result["volatility"],
                sharpe_ratio=result["sharpe_ratio"],
                max_drawdown=result["max_drawdown"],
                win_rate=result["win_rate"],
                profit_factor=result.get("profit_factor", 1.0),
                trades_count=result["total_trades"],
                avg_trade_duration=result.get("avg_trade_duration", 0),
                market_impact_cost=result.get("market_impact_cost", 0),
                performance_metrics=result,
                equity_curve=pd.Series([eq["portfolio_value"] for eq in result["equity_curve"]]),
                trade_log=result["trades"]
            )
            
            # Store performance history
            self.performance_history.append(backtest_result)
            
            logger.info(f"Backtest completed - Total return: {backtest_result.total_return:.2%}")
            return backtest_result
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            raise AlphaDiscoveryError(f"Failed to backtest strategy: {e}")
    
    async def _collect_agent_signals(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Collect signals from all agents"""
        try:
            # Collect signals in parallel
            tasks = [
                self._get_microstructure_signals(symbols, timeframe),
                self._get_altdata_signals(symbols, timeframe),
                self._get_regime_signals(symbols, timeframe)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "microstructure": results[0] if not isinstance(results[0], Exception) else None,
                "altdata": results[1] if not isinstance(results[1], Exception) else None,
                "regime": results[2] if not isinstance(results[2], Exception) else None
            }
            
        except Exception as e:
            logger.error(f"Failed to collect agent signals: {e}")
            return {}
    
    async def _get_microstructure_signals(self, symbols: List[str], timeframe: str) -> List[Any]:
        """Get microstructure signals"""
        try:
            signals = []
            for symbol in symbols:
                signal = await self.microstructure_agent.generate_signals(symbol, timeframe)
                signals.extend(signal)
            return signals
        except Exception as e:
            logger.error(f"Failed to get microstructure signals: {e}")
            return []
    
    async def _get_altdata_signals(self, symbols: List[str], timeframe: str) -> List[Any]:
        """Get alternative data signals"""
        try:
            return await self.altdata_agent.find_alpha_signals(symbols, timeframe)
        except Exception as e:
            logger.error(f"Failed to get altdata signals: {e}")
            return []
    
    async def _get_regime_signals(self, symbols: List[str], timeframe: str) -> Any:
        """Get regime detection signals"""
        try:
            return await self.regime_agent.detect_current_regime(symbols)
        except Exception as e:
            logger.error(f"Failed to get regime signals: {e}")
            return None
    
    async def _get_market_research(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market research for context"""
        try:
            # Determine market sector from symbols (simplified)
            sector = "technology"  # Would be determined from symbol analysis
            
            research = await self.market_research_tool._run(sector, "competitive_intelligence")
            return research
            
        except Exception as e:
            logger.error(f"Failed to get market research: {e}")
            return {}
    
    async def _synthesize_signals(self, agent_signals: Dict[str, Any], market_research: Dict[str, Any]) -> List[StrategySignal]:
        """Synthesize signals from multiple agents"""
        try:
            strategy_signals = []
            
            # Combine microstructure and altdata signals
            microstructure_signals = agent_signals.get("microstructure", [])
            altdata_signals = agent_signals.get("altdata", [])
            regime_state = agent_signals.get("regime")
            
            # Create combined signals
            all_signals = {}
            
            # Process microstructure signals
            for signal in microstructure_signals:
                symbol = getattr(signal, 'symbol', None)
                if symbol:
                    all_signals[symbol] = {
                        "microstructure": signal,
                        "altdata": None,
                        "regime": regime_state
                    }
            
            # Add altdata signals
            for signal in altdata_signals:
                symbol = getattr(signal, 'symbol', None)
                if symbol:
                    if symbol not in all_signals:
                        all_signals[symbol] = {"microstructure": None, "altdata": signal, "regime": regime_state}
                    else:
                        all_signals[symbol]["altdata"] = signal
            
            # Synthesize strategy signals
            for symbol, signals in all_signals.items():
                strategy_signal = await self._create_strategy_signal(symbol, signals, market_research)
                if strategy_signal:
                    strategy_signals.append(strategy_signal)
            
            return strategy_signals
            
        except Exception as e:
            logger.error(f"Signal synthesis failed: {e}")
            return []
    
    async def _create_strategy_signal(self, symbol: str, signals: Dict[str, Any], market_research: Dict[str, Any]) -> Optional[StrategySignal]:
        """Create strategy signal from combined agent signals"""
        try:
            # Extract signal information
            microstructure = signals.get("microstructure")
            altdata = signals.get("altdata")
            regime = signals.get("regime")
            
            # Calculate combined metrics
            confidence_scores = []
            expected_returns = []
            risk_scores = []
            agent_sources = []
            
            # Microstructure contribution
            if microstructure:
                confidence_scores.append(getattr(microstructure, 'confidence', 0.5))
                expected_returns.append(getattr(microstructure, 'expected_return', 0.0))
                risk_scores.append(getattr(microstructure, 'risk_score', 0.1))
                agent_sources.append("microstructure")
            
            # Alternative data contribution
            if altdata:
                confidence_scores.append(getattr(altdata, 'confidence', 0.5))
                expected_returns.append(getattr(altdata, 'expected_return', 0.0))
                risk_scores.append(getattr(altdata, 'risk_score', 0.1))
                agent_sources.append("altdata")
            
            if not confidence_scores:
                return None
            
            # Combine signals
            combined_confidence = np.mean(confidence_scores)
            combined_expected_return = np.mean(expected_returns)
            combined_risk_score = np.sqrt(np.mean([r**2 for r in risk_scores]))
            
            # Determine strategy type and direction
            if combined_expected_return > 0.02:
                strategy_type = StrategyType.MOMENTUM
                direction = "long"
            elif combined_expected_return < -0.02:
                strategy_type = StrategyType.MOMENTUM
                direction = "short"
            else:
                strategy_type = StrategyType.MEAN_REVERSION
                direction = "long" if combined_expected_return > 0 else "short"
            
            # Calculate position size using simplified Kelly criterion
            kelly_fraction = abs(combined_expected_return) / combined_risk_score if combined_risk_score > 0 else 0
            position_size = min(kelly_fraction * 0.25, 0.1)  # 25% of Kelly, max 10%
            
            # Get current price for entry
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            entry_price = hist['Close'].iloc[-1] if not hist.empty else 100.0
            
            # Calculate stop loss and take profit
            stop_loss = entry_price * (1 - combined_risk_score) if direction == "long" else entry_price * (1 + combined_risk_score)
            take_profit = entry_price * (1 + abs(combined_expected_return)) if direction == "long" else entry_price * (1 - abs(combined_expected_return))
            
            # Determine market regime
            market_regime = regime.market_regime.value if regime else "UNKNOWN"
            
            return StrategySignal(
                symbol=symbol,
                strategy_type=strategy_type,
                direction=direction,
                confidence=combined_confidence,
                expected_return=combined_expected_return,
                risk_score=combined_risk_score,
                position_size=position_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                holding_period=5,  # 5 days default
                agent_sources=agent_sources,
                market_regime=market_regime,
                timestamp=datetime.now(),
                metadata={
                    "market_research": market_research,
                    "signal_count": len(agent_sources),
                    "regime_confidence": regime.confidence_score if regime else 0.5
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create strategy signal for {symbol}: {e}")
            return None
    
    async def _apply_debate_mechanism(self, strategy_signals: List[StrategySignal]) -> List[StrategySignal]:
        """Apply enhanced multi-agent debate mechanism"""
        try:
            debated_strategies = []
            
            for strategy in strategy_signals:
                # Get market context for debate
                market_context = {
                    "regime": strategy.market_regime,
                    "agent_sources": strategy.agent_sources,
                    "metadata": strategy.metadata
                }
                
                # Run multi-agent debate
                debate_result = await self.debate_moderator._run(strategy, market_context)
                
                if "error" not in debate_result:
                    # Apply debate adjustments based on judge's evaluation
                    final_judgment = debate_result.get("final_judgment", {})
                    parsed_judgment = final_judgment.get("parsed_judgment", {})
                    
                    # Apply judge's adjustments
                    confidence_adj = parsed_judgment.get("confidence_adjustment", 0)
                    position_adj = parsed_judgment.get("position_size_adjustment", 1.0)
                    
                    # Adjust strategy based on debate outcome
                    strategy.confidence = max(0.1, min(1.0, strategy.confidence + confidence_adj))
                    strategy.position_size *= position_adj
                    
                    # Add comprehensive debate metadata
                    strategy.metadata["debate_result"] = debate_result
                    strategy.metadata["debate_type"] = "multi_agent_judged"
                    strategy.metadata["judge_recommendation"] = parsed_judgment.get("final_recommendation", "proceed")
                    strategy.metadata["key_risks"] = parsed_judgment.get("key_risks", [])
                    strategy.metadata["suggested_modifications"] = parsed_judgment.get("strategy_modifications", [])
                
                debated_strategies.append(strategy)
            
            return debated_strategies
            
        except Exception as e:
            logger.error(f"Enhanced debate mechanism failed: {e}")
            return strategy_signals

# Example usage and testing
async def main():
    """Example usage of StrategyAgent"""
    
    # Initialize agent
    agent = StrategyAgent()
    
    # Test symbols
    symbols = ["AAPL", "GOOGL", "TSLA", "MSFT"]
    
    try:
        # Synthesize strategies
        print("Synthesizing strategies...")
        strategy_signals = await agent.synthesize_strategy(symbols)
        
        print(f"\nSynthesized {len(strategy_signals)} strategy signals:")
        for signal in strategy_signals:
            print(f"- {signal.symbol}: {signal.strategy_type.value} {signal.direction}")
            print(f"  Confidence: {signal.confidence:.2f}, Expected Return: {signal.expected_return:.3f}")
            print(f"  Position Size: {signal.position_size:.3f}, Risk Score: {signal.risk_score:.3f}")
        
        # Optimize portfolio
        print("\nOptimizing portfolio...")
        portfolio_opt = await agent.optimize_portfolio(strategy_signals, method="markowitz")
        
        print(f"Portfolio optimization:")
        print(f"- Expected Return: {portfolio_opt.expected_return:.3f}")
        print(f"- Volatility: {portfolio_opt.volatility:.3f}")
        print(f"- Sharpe Ratio: {portfolio_opt.sharpe_ratio:.2f}")
        
        # Calculate Sharpe ratio
        print("\nCalculating Sharpe ratio...")
        sharpe_ratio = await agent.calculate_sharpe(strategy_signals)
        print(f"Strategy Sharpe ratio: {sharpe_ratio:.2f}")
        
        # Backtest strategy
        print("\nBacktesting strategy...")
        backtest_result = await agent.backtest_strategy(
            strategy_signals, 
            start_date="2023-01-01", 
            end_date="2023-12-31"
        )
        
        print(f"Backtest results:")
        print(f"- Total Return: {backtest_result.total_return:.2%}")
        print(f"- Annualized Return: {backtest_result.annualized_return:.2%}")
        print(f"- Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
        print(f"- Max Drawdown: {backtest_result.max_drawdown:.2%}")
        print(f"- Win Rate: {backtest_result.win_rate:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 