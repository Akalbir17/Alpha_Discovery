"""
Advanced Backtesting Engine for Alpha Discovery Platform

This engine provides institutional-grade backtesting capabilities with realistic
market impact modeling, transaction costs, and comprehensive performance analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
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
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import vectorbt as vbt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
# Remove direct ML model imports - now using ML client
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import yfinance as yf
import QuantLib as ql
from arch import arch_model
import warnings
# Remove direct torch imports - now using ML client
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from transformers import pipeline, AutoTokenizer, AutoModel
# import tensorflow as tf

# ML client for remote inference (Phase 4 upgrade)
try:
    from src.scrapers.ml_client import ml_client
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False
    logger.warning("ML client not available - using fallback analysis")

# Import our utilities
from src.utils.error_handling import handle_errors
from src.utils.monitoring import monitor_performance
from src.utils.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add new AI-enhanced components - now using ML client

# Remove the AIMarketImpactModel class - now handled by ML client
# class AIMarketImpactModel(nn.Module):
#     """Neural network for market impact prediction"""
#     
#     def __init__(self, input_dim=10, hidden_dim=64):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
#     
#     def forward(self, x):
#         return self.network(x)

class LLMBacktestAnalyzer:
    """LLM-powered backtest analysis and insights"""
    
    def __init__(self):
        # No longer loading models directly - using ML client
        logger.info("LLM Backtest Analyzer initialized with ML client support")
        
    async def analyze_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze backtest performance using LLM"""
        try:
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available, using fallback analysis")
                return self._fallback_analysis(backtest_results)
            
            # Extract key metrics for analysis
            metrics = backtest_results.get('metrics', {})
            
            # Use ML client for advanced analysis
            analysis_text = f"""
            Backtest Performance Analysis:
            - Total Return: {metrics.get('total_return', 0):.2%}
            - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
            - Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
            - Win Rate: {metrics.get('win_rate', 0):.2%}
            """
            
            # Use sentiment analysis for performance interpretation
            sentiment_result = await ml_client.analyze_sentiment(analysis_text)
            
            return {
                "performance_summary": analysis_text,
                "sentiment_analysis": {
                    "label": sentiment_result.label,
                    "score": sentiment_result.score,
                    "confidence": sentiment_result.confidence
                },
                "analysis_type": "llm_enhanced",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(backtest_results)
    
    def _fallback_analysis(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when ML client is not available"""
        metrics = backtest_results.get('metrics', {})
        
        # Simple rule-based analysis
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        if total_return > 0.15 and sharpe_ratio > 1.5:
            performance_grade = "Excellent"
        elif total_return > 0.10 and sharpe_ratio > 1.0:
            performance_grade = "Good"
        elif total_return > 0.05 and sharpe_ratio > 0.5:
            performance_grade = "Fair"
        else:
            performance_grade = "Poor"
            
            return {
            "performance_summary": f"Strategy performance: {performance_grade}",
            "performance_grade": performance_grade,
            "analysis_type": "rule_based_fallback",
                "timestamp": datetime.now().isoformat()
            }
            
    async def generate_strategy_insights(self, signals: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy insights using pattern analysis"""
        try:
            if not ML_CLIENT_AVAILABLE:
                return {"insights": ["ML client not available"], "confidence": 0.3}
            
            # Analyze signal patterns
            signal_stats = {
                "total_signals": len(signals),
                "positive_signals": len(signals[signals['signal'] > 0]) if 'signal' in signals.columns else 0,
                "negative_signals": len(signals[signals['signal'] < 0]) if 'signal' in signals.columns else 0,
                "signal_strength_avg": float(signals['signal'].mean()) if 'signal' in signals.columns else 0
            }
            
            insights = []
            
            # Generate insights based on signal patterns
            if signal_stats["positive_signals"] > signal_stats["negative_signals"] * 2:
                insights.append("Strategy shows bullish bias - consider market regime analysis")
            
            if signal_stats["signal_strength_avg"] > 0.7:
                insights.append("High signal strength detected - strategy may be overconfident")
            
            return {
                "insights": insights,
                "signal_statistics": signal_stats,
                "confidence": 0.8,
                "analysis_method": "pattern_analysis",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal pattern analysis failed: {e}")
            return {"error": str(e)}

class AIRegimeDetector:
    """AI-powered market regime detection - now using ML client"""
    
    def __init__(self):
        # No longer loading models directly
        logger.info("AI Regime Detector initialized with ML client support")
        
    async def train_regime_model(self, market_data: pd.DataFrame) -> None:
        """Train AI model for regime detection using ML client"""
        try:
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available - regime detection disabled")
                return
            
            # Prepare features for regime detection
            features = self._create_regime_features(market_data)
            
            # Use ML client for clustering-based regime detection
            feature_list = features.values.tolist()
            clustering_result = await ml_client.perform_clustering(feature_list)
            
            logger.info(f"AI regime detection model trained via ML client - {clustering_result.n_clusters} regimes detected")
            
        except Exception as e:
            logger.error(f"Regime model training failed: {e}")
    
    async def predict_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict current market regime using ML client"""
        try:
            if not ML_CLIENT_AVAILABLE:
                return {"regime": "unknown", "confidence": 0.0}
            
            # Create features
            features = self._create_regime_features(market_data)
            
            # Use latest data point for prediction
            latest_features = features.iloc[-1].values.tolist()
            
            # Use ML client for anomaly detection to identify regime changes
            anomaly_result = await ml_client.detect_anomalies([latest_features])
            
            # Map anomaly score to regime
            if anomaly_result.risk_score > 0.7:
                regime = "crisis"
            elif anomaly_result.risk_score > 0.5:
                regime = "stressed"
            elif anomaly_result.risk_score > 0.3:
                regime = "volatile"
            else:
                regime = "normal"
            
            return {
                "regime": regime,
                "confidence": anomaly_result.confidence,
                "risk_score": anomaly_result.risk_score,
                "risk_factors": anomaly_result.risk_factors,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return {"regime": "unknown", "confidence": 0.0}
    
    def _create_regime_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for regime detection"""
        try:
            features = pd.DataFrame(index=market_data.index)
            
            # Price-based features
            if 'close' in market_data.columns:
                prices = market_data['close']
                features['returns'] = prices.pct_change()
                features['volatility'] = features['returns'].rolling(20).std()
                features['rsi'] = self._calculate_rsi(prices)
                
            # Volume features
            if 'volume' in market_data.columns:
                features['volume_ma'] = market_data['volume'].rolling(20).mean()
                features['volume_ratio'] = market_data['volume'] / features['volume_ma']
                
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature creation failed: {e}")
            # Return dummy features
            return pd.DataFrame(index=market_data.index, data={'dummy': 0})
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series(index=prices.index, data=50)  # Neutral RSI

class BacktestType(Enum):
    """Backtest type classification"""
    SIMPLE = "SIMPLE"
    WALK_FORWARD = "WALK_FORWARD"
    MONTE_CARLO = "MONTE_CARLO"
    COMBINATORIAL = "COMBINATORIAL"

class MarketImpactModel(Enum):
    """Market impact model types"""
    LINEAR = "LINEAR"
    SQUARE_ROOT = "SQUARE_ROOT"
    ALMGREN_CHRISS = "ALMGREN_CHRISS"
    CUSTOM = "CUSTOM"

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    market_impact_model: MarketImpactModel = MarketImpactModel.SQUARE_ROOT
    market_impact_coeff: float = 0.1
    max_position_size: float = 0.2  # 20% max position
    rebalance_frequency: str = "D"  # Daily rebalancing
    benchmark: str = "SPY"
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    monte_carlo_runs: int = 1000
    walk_forward_window: int = 252  # 1 year
    walk_forward_step: int = 21  # 1 month

@dataclass
class TradeExecution:
    """Trade execution details"""
    symbol: str
    timestamp: datetime
    side: str  # buy/sell
    quantity: float
    price: float
    commission: float
    slippage: float
    market_impact: float
    total_cost: float
    portfolio_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Trade metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float
    
    # Portfolio metrics
    beta: float
    alpha: float
    r_squared: float
    tracking_error: float
    
    # Additional metrics
    skewness: float
    kurtosis: float
    tail_ratio: float
    common_sense_ratio: float
    
    # Metadata
    start_date: datetime
    end_date: datetime
    trading_days: int
    benchmark: str
    
@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    period_start: datetime
    period_end: datetime
    in_sample_metrics: BacktestMetrics
    out_sample_metrics: BacktestMetrics
    decay_factor: float
    strategy_parameters: Dict[str, Any]

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    mean_return: float
    std_return: float
    percentile_returns: Dict[str, float]
    probability_of_loss: float
    expected_shortfall: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    simulation_paths: np.ndarray

class MarketImpactCalculator:
    """Calculates realistic market impact for trades"""
    
    def __init__(self, model_type: MarketImpactModel = MarketImpactModel.SQUARE_ROOT):
        self.model_type = model_type
        
    def calculate_impact(self, trade_size: float, avg_volume: float, 
                        volatility: float, participation_rate: float = 0.1) -> float:
        """Calculate market impact for a trade"""
        try:
            if self.model_type == MarketImpactModel.LINEAR:
                return self._linear_impact(trade_size, avg_volume, volatility)
            elif self.model_type == MarketImpactModel.SQUARE_ROOT:
                return self._square_root_impact(trade_size, avg_volume, volatility)
            elif self.model_type == MarketImpactModel.ALMGREN_CHRISS:
                return self._almgren_chriss_impact(trade_size, avg_volume, volatility, participation_rate)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Market impact calculation failed: {e}")
            return 0.0
    
    def _linear_impact(self, trade_size: float, avg_volume: float, volatility: float) -> float:
        """Linear market impact model"""
        volume_ratio = abs(trade_size) / avg_volume
        return 0.1 * volatility * volume_ratio
    
    def _square_root_impact(self, trade_size: float, avg_volume: float, volatility: float) -> float:
        """Square root market impact model (most common)"""
        volume_ratio = abs(trade_size) / avg_volume
        return 0.1 * volatility * np.sqrt(volume_ratio)
    
    def _almgren_chriss_impact(self, trade_size: float, avg_volume: float, 
                              volatility: float, participation_rate: float) -> float:
        """Almgren-Chriss market impact model"""
        # Simplified implementation
        temporary_impact = 0.5 * volatility * np.sqrt(participation_rate)
        permanent_impact = 0.1 * volatility * (abs(trade_size) / avg_volume)
        return temporary_impact + permanent_impact

class AIEnhancedMarketImpactCalculator(MarketImpactCalculator):
    """AI-enhanced market impact calculation"""
    
    def __init__(self, model_type: MarketImpactModel = MarketImpactModel.CUSTOM):
        super().__init__(model_type)
        # No longer loading models directly - using ML client
        logger.info("AI Enhanced Market Impact Calculator initialized with ML client support")
        
    async def train_impact_model(self, historical_data: pd.DataFrame) -> None:
        """Train AI model for market impact prediction using ML client"""
        try:
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available - AI impact model training disabled")
                return
            
            # Create features and targets from historical data
            features, targets = self._prepare_impact_training_data(historical_data)
            
            # Use ML client for training
            training_result = await ml_client.train_regression(features.values.tolist(), targets.tolist())
            
            logger.info(f"AI market impact model trained via ML client - R2 score: {training_result.r2_score:.2f}")
            
        except Exception as e:
            logger.error(f"AI impact model training failed: {e}")
    
    async def calculate_ai_impact(self, trade_size: float, market_features: Dict[str, float]) -> float:
        """Calculate market impact using AI model"""
        try:
            if not ML_CLIENT_AVAILABLE:
                # Fall back to traditional model
                return self.calculate_impact(
                    trade_size, 
                    market_features.get('avg_volume', 1000000),
                    market_features.get('volatility', 0.02)
                )
            
            # Prepare features
            feature_vector = np.array([[
                trade_size,
                market_features.get('avg_volume', 1000000),
                market_features.get('volatility', 0.02),
                market_features.get('bid_ask_spread', 0.001),
                market_features.get('market_cap', 1e9),
                market_features.get('momentum', 0),
                market_features.get('rsi', 50),
                market_features.get('volume_ratio', 1),
                market_features.get('time_of_day', 12),
                market_features.get('day_of_week', 3)
            ]])
            
            # Use ML client for prediction
            prediction_result = await ml_client.predict_regression(feature_vector.tolist())
            
            return max(0, prediction_result.prediction[0])  # Ensure non-negative impact
            
        except Exception as e:
            logger.error(f"AI impact calculation failed: {e}")
            return 0.001  # Default impact
    
    def _prepare_impact_training_data(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for impact model"""
        try:
            # This would use actual trade data with observed impacts
            # For now, we'll simulate training data
            n_samples = 1000
            
            features = np.random.rand(n_samples, 10)  # 10 features
            targets = np.random.rand(n_samples) * 0.01  # Impact targets
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            return np.array([]), np.array([])

class TransactionCostModel:
    """Models transaction costs including commissions and slippage"""
    
    def __init__(self, commission_rate: float = 0.001, slippage_rate: float = 0.0005):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
    def calculate_costs(self, trade_value: float, volatility: float = 0.2) -> Dict[str, float]:
        """Calculate transaction costs"""
        try:
            # Commission cost
            commission = abs(trade_value) * self.commission_rate
            
            # Slippage (increases with volatility)
            slippage = abs(trade_value) * self.slippage_rate * (1 + volatility)
            
            # Bid-ask spread cost (simplified)
            bid_ask_spread = abs(trade_value) * 0.0002 * (1 + volatility)
            
            return {
                "commission": commission,
                "slippage": slippage,
                "bid_ask_spread": bid_ask_spread,
                "total": commission + slippage + bid_ask_spread
            }
            
        except Exception as e:
            logger.error(f"Transaction cost calculation failed: {e}")
            return {"commission": 0, "slippage": 0, "bid_ask_spread": 0, "total": 0}

class MetricsCalculator:
    """Calculates comprehensive performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_comprehensive_metrics(self, returns: pd.Series, benchmark_returns: pd.Series,
                                      trades: List[TradeExecution], 
                                      portfolio_values: pd.Series) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            # Basic return metrics
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            excess_return = annualized_return - self.risk_free_rate
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            # Drawdown metrics
            dd_info = self._calculate_drawdown_metrics(portfolio_values)
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Risk-adjusted metrics
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            calmar_ratio = annualized_return / abs(dd_info["max_drawdown"]) if dd_info["max_drawdown"] != 0 else 0
            
            # Benchmark comparison
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)
            
            # Trade metrics
            trade_metrics = self._calculate_trade_metrics(trades)
            
            # Distribution metrics
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            tail_ratio = np.percentile(returns, 95) / abs(np.percentile(returns, 5))
            
            return BacktestMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                excess_return=excess_return,
                volatility=volatility,
                downside_deviation=downside_deviation,
                max_drawdown=dd_info["max_drawdown"],
                max_drawdown_duration=dd_info["max_duration"],
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=benchmark_metrics["information_ratio"],
                treynor_ratio=benchmark_metrics["treynor_ratio"],
                total_trades=trade_metrics["total_trades"],
                win_rate=trade_metrics["win_rate"],
                avg_win=trade_metrics["avg_win"],
                avg_loss=trade_metrics["avg_loss"],
                profit_factor=trade_metrics["profit_factor"],
                avg_trade_duration=trade_metrics["avg_trade_duration"],
                beta=benchmark_metrics["beta"],
                alpha=benchmark_metrics["alpha"],
                r_squared=benchmark_metrics["r_squared"],
                tracking_error=benchmark_metrics["tracking_error"],
                skewness=skewness,
                kurtosis=kurtosis,
                tail_ratio=tail_ratio,
                common_sense_ratio=tail_ratio,  # Simplified
                start_date=portfolio_values.index[0],
                end_date=portfolio_values.index[-1],
                trading_days=len(returns),
                benchmark="SPY"
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            raise AlphaDiscoveryError(f"Failed to calculate metrics: {e}")
    
    def _calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown metrics"""
        try:
            # Calculate drawdown
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            
            max_drawdown = drawdown.min()
            
            # Calculate drawdown duration
            drawdown_start = None
            max_duration = 0
            current_duration = 0
            
            for i, dd in enumerate(drawdown):
                if dd < 0:
                    if drawdown_start is None:
                        drawdown_start = i
                    current_duration += 1
                else:
                    if drawdown_start is not None:
                        max_duration = max(max_duration, current_duration)
                        drawdown_start = None
                        current_duration = 0
            
            return {
                "max_drawdown": max_drawdown,
                "max_duration": max_duration,
                "drawdown_series": drawdown
            }
            
        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}")
            return {"max_drawdown": 0, "max_duration": 0, "drawdown_series": pd.Series()}
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        try:
            # Align returns
            aligned_returns = returns.align(benchmark_returns, join='inner')
            strategy_returns = aligned_returns[0].dropna()
            bench_returns = aligned_returns[1].dropna()
            
            if len(strategy_returns) == 0 or len(bench_returns) == 0:
                return {
                    "beta": 0, "alpha": 0, "r_squared": 0, 
                    "tracking_error": 0, "information_ratio": 0, "treynor_ratio": 0
                }
            
            # Calculate beta
            covariance = np.cov(strategy_returns, bench_returns)[0, 1]
            benchmark_variance = np.var(bench_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Calculate alpha
            strategy_mean = strategy_returns.mean() * 252
            benchmark_mean = bench_returns.mean() * 252
            alpha = strategy_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
            
            # Calculate R-squared
            correlation = np.corrcoef(strategy_returns, bench_returns)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            # Calculate tracking error
            active_returns = strategy_returns - bench_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            
            # Calculate information ratio
            information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            # Calculate Treynor ratio
            treynor_ratio = (strategy_mean - self.risk_free_rate) / beta if beta > 0 else 0
            
            return {
                "beta": beta,
                "alpha": alpha,
                "r_squared": r_squared,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "treynor_ratio": treynor_ratio
            }
            
        except Exception as e:
            logger.error(f"Benchmark metrics calculation failed: {e}")
            return {
                "beta": 0, "alpha": 0, "r_squared": 0, 
                "tracking_error": 0, "information_ratio": 0, "treynor_ratio": 0
            }
    
    def _calculate_trade_metrics(self, trades: List[TradeExecution]) -> Dict[str, Any]:
        """Calculate trade-specific metrics"""
        try:
            if not trades:
                return {
                    "total_trades": 0, "win_rate": 0, "avg_win": 0, 
                    "avg_loss": 0, "profit_factor": 0, "avg_trade_duration": 0
                }
            
            # Group trades by symbol to calculate P&L
            positions = {}
            trade_pnls = []
            
            for trade in trades:
                symbol = trade.symbol
                if symbol not in positions:
                    positions[symbol] = {"quantity": 0, "avg_price": 0, "total_cost": 0}
                
                pos = positions[symbol]
                
                if trade.side == "buy":
                    if pos["quantity"] >= 0:  # Adding to long position
                        total_value = pos["quantity"] * pos["avg_price"] + trade.quantity * trade.price
                        pos["quantity"] += trade.quantity
                        pos["avg_price"] = total_value / pos["quantity"] if pos["quantity"] > 0 else 0
                    else:  # Covering short position
                        pnl = (pos["avg_price"] - trade.price) * min(abs(pos["quantity"]), trade.quantity)
                        trade_pnls.append(pnl)
                        pos["quantity"] += trade.quantity
                        
                else:  # sell
                    if pos["quantity"] <= 0:  # Adding to short position
                        total_value = abs(pos["quantity"]) * pos["avg_price"] + trade.quantity * trade.price
                        pos["quantity"] -= trade.quantity
                        pos["avg_price"] = total_value / abs(pos["quantity"]) if pos["quantity"] != 0 else 0
                    else:  # Selling long position
                        pnl = (trade.price - pos["avg_price"]) * min(pos["quantity"], trade.quantity)
                        trade_pnls.append(pnl)
                        pos["quantity"] -= trade.quantity
                
                pos["total_cost"] += trade.total_cost
            
            # Calculate metrics
            total_trades = len(trade_pnls)
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            gross_profit = sum(winning_trades)
            gross_loss = abs(sum(losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate average trade duration (simplified)
            avg_trade_duration = 1.0  # Would need more sophisticated tracking
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_trade_duration": avg_trade_duration
            }
            
        except Exception as e:
            logger.error(f"Trade metrics calculation failed: {e}")
            return {
                "total_trades": 0, "win_rate": 0, "avg_win": 0, 
                "avg_loss": 0, "profit_factor": 0, "avg_trade_duration": 0
            }

class StrategyDecayDetector:
    """Detects strategy decay and suggests adaptations"""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.performance_history = []
        
    def detect_decay(self, returns: pd.Series, window: int = 63) -> Dict[str, Any]:
        """Detect strategy decay using rolling performance analysis"""
        try:
            # Calculate rolling Sharpe ratio
            rolling_sharpe = returns.rolling(window).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            
            # Calculate trend in performance
            recent_performance = rolling_sharpe.tail(window * 2)
            if len(recent_performance) < 2:
                return {"decay_detected": False, "decay_factor": 0.0}
            
            # Linear regression to detect trend
            x = np.arange(len(recent_performance))
            y = recent_performance.values
            
            # Remove NaN values
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 2:
                return {"decay_detected": False, "decay_factor": 0.0}
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            # Calculate trend slope
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            
            # Detect decay
            decay_threshold = -0.01  # Negative trend threshold
            decay_detected = slope < decay_threshold
            decay_factor = abs(slope) if decay_detected else 0.0
            
            # Calculate statistical significance
            correlation = np.corrcoef(x_valid, y_valid)[0, 1]
            p_value = stats.pearsonr(x_valid, y_valid)[1]
            
            return {
                "decay_detected": decay_detected,
                "decay_factor": decay_factor,
                "slope": slope,
                "correlation": correlation,
                "p_value": p_value,
                "significance": p_value < 0.05,
                "recent_sharpe": rolling_sharpe.tail(window).mean(),
                "historical_sharpe": rolling_sharpe.head(window).mean()
            }
            
        except Exception as e:
            logger.error(f"Decay detection failed: {e}")
            return {"decay_detected": False, "decay_factor": 0.0, "error": str(e)}
    
    def suggest_adaptations(self, decay_info: Dict[str, Any]) -> List[str]:
        """Suggest adaptations based on decay analysis"""
        try:
            adaptations = []
            
            if decay_info.get("decay_detected", False):
                decay_factor = decay_info.get("decay_factor", 0)
                
                if decay_factor > 0.02:
                    adaptations.append("Consider reducing position sizes")
                    adaptations.append("Implement more frequent rebalancing")
                    adaptations.append("Review and update signal generation logic")
                
                if decay_factor > 0.05:
                    adaptations.append("Consider strategy retirement or major overhaul")
                    adaptations.append("Implement regime-aware position sizing")
                    adaptations.append("Add new alpha factors or data sources")
                
                if decay_info.get("significance", False):
                    adaptations.append("Decay trend is statistically significant")
                    adaptations.append("Implement adaptive parameters")
                    adaptations.append("Consider ensemble approach with multiple strategies")
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Adaptation suggestions failed: {e}")
            return ["Error generating adaptations"]

class BacktestingEngine:
    """
    Advanced Backtesting Engine using vectorbt
    
    Provides institutional-grade backtesting with realistic market impact,
    transaction costs, comprehensive metrics, and advanced analysis.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000
        )
        
        # Initialize components
        self.market_impact_calc = MarketImpactCalculator(self.config.market_impact_model)
        self.transaction_cost_model = TransactionCostModel(self.config.commission, self.config.slippage)
        self.metrics_calculator = MetricsCalculator(self.config.risk_free_rate)
        self.decay_detector = StrategyDecayDetector()
        
        # Storage
        self.backtest_results = {}
        self.trade_history = []
        
        logger.info("BacktestingEngine initialized successfully")
    
    @handle_errors
    @monitor_performance
    async def run_backtest(self, strategy_signals: pd.DataFrame, 
                          price_data: pd.DataFrame, 
                          backtest_type: BacktestType = BacktestType.SIMPLE) -> Dict[str, Any]:
        """
        Run comprehensive backtest
        
        Args:
            strategy_signals: DataFrame with trading signals
            price_data: DataFrame with OHLCV data
            backtest_type: Type of backtest to run
            
        Returns:
            Comprehensive backtest results
        """
        try:
            logger.info(f"Running {backtest_type.value} backtest")
            
            if backtest_type == BacktestType.SIMPLE:
                return await self._run_simple_backtest(strategy_signals, price_data)
            elif backtest_type == BacktestType.WALK_FORWARD:
                return await self._run_walk_forward_backtest(strategy_signals, price_data)
            elif backtest_type == BacktestType.MONTE_CARLO:
                return await self._run_monte_carlo_backtest(strategy_signals, price_data)
            elif backtest_type == BacktestType.COMBINATORIAL:
                return await self._run_combinatorial_backtest(strategy_signals, price_data)
            else:
                raise ValueError(f"Unsupported backtest type: {backtest_type}")
                
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise AlphaDiscoveryError(f"Backtest execution failed: {e}")
    
    async def _run_simple_backtest(self, strategy_signals: pd.DataFrame, 
                                  price_data: pd.DataFrame) -> Dict[str, Any]:
        """Run simple backtest with realistic costs"""
        try:
            # Prepare data
            close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, 0]
            volume_data = price_data['volume'] if 'volume' in price_data.columns else pd.Series(index=price_data.index, data=1000000)
            
            # Calculate volatility
            returns = close_prices.pct_change().dropna()
            volatility = returns.rolling(21).std()
            
            # Initialize portfolio
            portfolio_values = [self.config.initial_capital]
            positions = {}
            trades = []
            
            # Simulate trading
            for date in strategy_signals.index:
                if date not in close_prices.index:
                    continue
                    
                current_price = close_prices.loc[date]
                current_volume = volume_data.loc[date] if date in volume_data.index else 1000000
                current_volatility = volatility.loc[date] if date in volatility.index else 0.2
                
                # Get signal
                signal = strategy_signals.loc[date]
                
                # Calculate position size
                if hasattr(signal, 'position_size'):
                    target_position = signal.position_size
                elif 'position_size' in signal:
                    target_position = signal['position_size']
                else:
                    target_position = 0.05  # Default 5%
                
                # Current portfolio value
                current_portfolio_value = portfolio_values[-1]
                
                # Calculate trade size
                target_value = current_portfolio_value * target_position
                current_position_value = positions.get(signal.name, 0) * current_price
                trade_value = target_value - current_position_value
                
                if abs(trade_value) > current_portfolio_value * 0.01:  # Minimum trade size
                    # Calculate costs
                    market_impact = self.market_impact_calc.calculate_impact(
                        abs(trade_value) / current_price, current_volume, current_volatility
                    )
                    
                    transaction_costs = self.transaction_cost_model.calculate_costs(
                        trade_value, current_volatility
                    )
                    
                    # Execute trade
                    trade_quantity = trade_value / current_price
                    total_cost = transaction_costs['total'] + abs(trade_value) * market_impact
                    
                    # Record trade
                    trade = TradeExecution(
                        symbol=signal.name,
                        timestamp=date,
                        side="buy" if trade_value > 0 else "sell",
                        quantity=abs(trade_quantity),
                        price=current_price,
                        commission=transaction_costs['commission'],
                        slippage=transaction_costs['slippage'],
                        market_impact=abs(trade_value) * market_impact,
                        total_cost=total_cost,
                        portfolio_value=current_portfolio_value
                    )
                    
                    trades.append(trade)
                    
                    # Update position
                    if signal.name not in positions:
                        positions[signal.name] = 0
                    positions[signal.name] += trade_quantity
                    
                    # Update portfolio value
                    current_portfolio_value -= total_cost
                
                # Calculate current portfolio value
                total_position_value = sum(
                    pos * current_price for pos in positions.values()
                )
                
                portfolio_values.append(current_portfolio_value + total_position_value)
            
            # Create portfolio series
            portfolio_series = pd.Series(portfolio_values[1:], index=strategy_signals.index)
            
            # Calculate returns
            portfolio_returns = portfolio_series.pct_change().dropna()
            
            # Get benchmark data
            benchmark_returns = await self._get_benchmark_returns()
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                portfolio_returns, benchmark_returns, trades, portfolio_series
            )
            
            # Detect strategy decay
            decay_info = self.decay_detector.detect_decay(portfolio_returns)
            adaptations = self.decay_detector.suggest_adaptations(decay_info)
            
            return {
                "backtest_type": BacktestType.SIMPLE.value,
                "portfolio_values": portfolio_series,
                "returns": portfolio_returns,
                "trades": trades,
                "metrics": metrics,
                "decay_analysis": decay_info,
                "adaptations": adaptations,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Simple backtest failed: {e}")
            raise AlphaDiscoveryError(f"Simple backtest failed: {e}")
    
    async def _run_walk_forward_backtest(self, strategy_signals: pd.DataFrame, 
                                        price_data: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        try:
            logger.info("Running walk-forward analysis")
            
            # Setup time series split
            tscv = TimeSeriesSplit(n_splits=5)
            walk_forward_results = []
            
            for i, (train_idx, test_idx) in enumerate(tscv.split(strategy_signals)):
                # Split data
                train_signals = strategy_signals.iloc[train_idx]
                test_signals = strategy_signals.iloc[test_idx]
                
                train_prices = price_data.iloc[train_idx]
                test_prices = price_data.iloc[test_idx]
                
                # Run in-sample backtest
                in_sample_result = await self._run_simple_backtest(train_signals, train_prices)
                
                # Run out-of-sample backtest
                out_sample_result = await self._run_simple_backtest(test_signals, test_prices)
                
                # Calculate decay factor
                in_sample_sharpe = in_sample_result["metrics"].sharpe_ratio
                out_sample_sharpe = out_sample_result["metrics"].sharpe_ratio
                decay_factor = (in_sample_sharpe - out_sample_sharpe) / in_sample_sharpe if in_sample_sharpe != 0 else 0
                
                # Store results
                walk_forward_results.append(WalkForwardResult(
                    period_start=test_signals.index[0],
                    period_end=test_signals.index[-1],
                    in_sample_metrics=in_sample_result["metrics"],
                    out_sample_metrics=out_sample_result["metrics"],
                    decay_factor=decay_factor,
                    strategy_parameters={}
                ))
            
            # Aggregate results
            avg_decay = np.mean([r.decay_factor for r in walk_forward_results])
            stability_score = 1 - np.std([r.out_sample_metrics.sharpe_ratio for r in walk_forward_results])
            
            return {
                "backtest_type": BacktestType.WALK_FORWARD.value,
                "walk_forward_results": walk_forward_results,
                "average_decay": avg_decay,
                "stability_score": stability_score,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Walk-forward backtest failed: {e}")
            raise AlphaDiscoveryError(f"Walk-forward backtest failed: {e}")
    
    async def _run_monte_carlo_backtest(self, strategy_signals: pd.DataFrame, 
                                       price_data: pd.DataFrame) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        try:
            logger.info("Running Monte Carlo simulation")
            
            # Run base backtest
            base_result = await self._run_simple_backtest(strategy_signals, price_data)
            base_returns = base_result["returns"]
            
            # Monte Carlo parameters
            n_simulations = self.config.monte_carlo_runs
            simulation_results = []
            
            # Bootstrap simulation
            for i in range(n_simulations):
                # Bootstrap returns
                bootstrapped_returns = np.random.choice(
                    base_returns.dropna(), 
                    size=len(base_returns), 
                    replace=True
                )
                
                # Calculate cumulative returns
                cumulative_returns = (1 + pd.Series(bootstrapped_returns)).cumprod()
                final_return = cumulative_returns.iloc[-1] - 1
                
                simulation_results.append(final_return)
            
            # Calculate statistics
            simulation_array = np.array(simulation_results)
            
            monte_carlo_result = MonteCarloResult(
                mean_return=np.mean(simulation_array),
                std_return=np.std(simulation_array),
                percentile_returns={
                    "5%": np.percentile(simulation_array, 5),
                    "25%": np.percentile(simulation_array, 25),
                    "50%": np.percentile(simulation_array, 50),
                    "75%": np.percentile(simulation_array, 75),
                    "95%": np.percentile(simulation_array, 95)
                },
                probability_of_loss=np.mean(simulation_array < 0),
                expected_shortfall=np.mean(simulation_array[simulation_array < np.percentile(simulation_array, 5)]),
                confidence_intervals={
                    "90%": (np.percentile(simulation_array, 5), np.percentile(simulation_array, 95)),
                    "95%": (np.percentile(simulation_array, 2.5), np.percentile(simulation_array, 97.5))
                },
                simulation_paths=simulation_array
            )
            
            return {
                "backtest_type": BacktestType.MONTE_CARLO.value,
                "base_result": base_result,
                "monte_carlo_result": monte_carlo_result,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo backtest failed: {e}")
            raise AlphaDiscoveryError(f"Monte Carlo backtest failed: {e}")
    
    async def _run_combinatorial_backtest(self, strategy_signals: pd.DataFrame, 
                                         price_data: pd.DataFrame) -> Dict[str, Any]:
        """Run combinatorial backtest with parameter optimization"""
        try:
            logger.info("Running combinatorial backtest")
            
            # Parameter ranges to test
            param_ranges = {
                "commission": [0.0005, 0.001, 0.002],
                "slippage": [0.0002, 0.0005, 0.001],
                "max_position_size": [0.1, 0.2, 0.3],
                "rebalance_frequency": ["D", "W", "M"]
            }
            
            # Generate parameter combinations
            from itertools import product
            param_combinations = list(product(*param_ranges.values()))
            
            results = []
            
            for params in param_combinations[:20]:  # Limit to first 20 combinations
                # Create temporary config
                temp_config = BacktestConfig(
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    initial_capital=self.config.initial_capital,
                    commission=params[0],
                    slippage=params[1],
                    max_position_size=params[2],
                    rebalance_frequency=params[3]
                )
                
                # Create temporary engine
                temp_engine = BacktestingEngine(temp_config)
                
                # Run backtest
                result = await temp_engine._run_simple_backtest(strategy_signals, price_data)
                
                # Store result with parameters
                results.append({
                    "parameters": dict(zip(param_ranges.keys(), params)),
                    "metrics": result["metrics"],
                    "sharpe_ratio": result["metrics"].sharpe_ratio
                })
            
            # Find best parameters
            best_result = max(results, key=lambda x: x["sharpe_ratio"])
            
            return {
                "backtest_type": BacktestType.COMBINATORIAL.value,
                "all_results": results,
                "best_parameters": best_result["parameters"],
                "best_metrics": best_result["metrics"],
                "parameter_ranges": param_ranges,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Combinatorial backtest failed: {e}")
            raise AlphaDiscoveryError(f"Combinatorial backtest failed: {e}")
    
    @handle_errors
    async def calculate_metrics(self, portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series = None) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if benchmark_returns is None:
                benchmark_returns = await self._get_benchmark_returns()
            
            # Create dummy portfolio values for metrics calculation
            portfolio_values = (1 + portfolio_returns).cumprod() * self.config.initial_capital
            
            return self.metrics_calculator.calculate_comprehensive_metrics(
                portfolio_returns, benchmark_returns, [], portfolio_values
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            raise AlphaDiscoveryError(f"Failed to calculate metrics: {e}")
    
    @handle_errors
    async def generate_report(self, backtest_results: Dict[str, Any], 
                            output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive backtest report with visualizations"""
        try:
            logger.info("Generating backtest report")
            
            report = {
                "summary": self._generate_summary(backtest_results),
                "detailed_metrics": backtest_results.get("metrics"),
                "visualizations": await self._create_visualizations(backtest_results),
                "recommendations": self._generate_recommendations(backtest_results),
                "timestamp": datetime.now()
            }
            
            if output_path:
                # Save report
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                logger.info(f"Report saved to {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise AlphaDiscoveryError(f"Failed to generate report: {e}")
    
    async def _get_benchmark_returns(self) -> pd.Series:
        """Get benchmark returns"""
        try:
            # Download benchmark data
            benchmark_ticker = yf.Ticker(self.config.benchmark)
            benchmark_data = benchmark_ticker.history(
                start=self.config.start_date,
                end=self.config.end_date
            )
            
            return benchmark_data['Close'].pct_change().dropna()
            
        except Exception as e:
            logger.error(f"Benchmark data download failed: {e}")
            # Return dummy benchmark returns
            date_range = pd.date_range(self.config.start_date, self.config.end_date, freq='D')
            return pd.Series(np.random.normal(0.0005, 0.01, len(date_range)), index=date_range)
    
    def _generate_summary(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        try:
            metrics = backtest_results.get("metrics")
            if not metrics:
                return {"error": "No metrics available"}
            
            return {
                "strategy_performance": {
                    "total_return": f"{metrics.total_return:.2%}",
                    "annualized_return": f"{metrics.annualized_return:.2%}",
                    "volatility": f"{metrics.volatility:.2%}",
                    "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                    "max_drawdown": f"{metrics.max_drawdown:.2%}"
                },
                "risk_assessment": {
                    "var_95": f"{metrics.var_95:.2%}",
                    "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
                    "calmar_ratio": f"{metrics.calmar_ratio:.2f}"
                },
                "trading_activity": {
                    "total_trades": metrics.total_trades,
                    "win_rate": f"{metrics.win_rate:.2%}",
                    "profit_factor": f"{metrics.profit_factor:.2f}"
                }
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}
    
    async def _create_visualizations(self, backtest_results: Dict[str, Any]) -> Dict[str, str]:
        """Create visualization plots"""
        try:
            visualizations = {}
            
            # Portfolio value chart
            if "portfolio_values" in backtest_results:
                portfolio_values = backtest_results["portfolio_values"]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title='Portfolio Value Over Time',
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    template='plotly_white'
                )
                
                visualizations["portfolio_chart"] = fig.to_html()
            
            # Returns distribution
            if "returns" in backtest_results:
                returns = backtest_results["returns"]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=returns.values,
                    nbinsx=50,
                    name='Returns Distribution',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title='Returns Distribution',
                    xaxis_title='Daily Returns',
                    yaxis_title='Frequency',
                    template='plotly_white'
                )
                
                visualizations["returns_distribution"] = fig.to_html()
            
            # Drawdown chart
            if "portfolio_values" in backtest_results:
                portfolio_values = backtest_results["portfolio_values"]
                peak = portfolio_values.expanding().max()
                drawdown = (portfolio_values - peak) / peak
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red'),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title='Drawdown Analysis',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    template='plotly_white'
                )
                
                visualizations["drawdown_chart"] = fig.to_html()
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, backtest_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on backtest results"""
        try:
            recommendations = []
            metrics = backtest_results.get("metrics")
            
            if not metrics:
                return ["Unable to generate recommendations - no metrics available"]
            
            # Performance recommendations
            if metrics.sharpe_ratio < 1.0:
                recommendations.append("Consider improving risk-adjusted returns (Sharpe ratio < 1.0)")
            
            if metrics.max_drawdown < -0.2:
                recommendations.append("Implement stronger risk controls (Max drawdown > 20%)")
            
            if metrics.win_rate < 0.4:
                recommendations.append("Review signal quality (Win rate < 40%)")
            
            # Risk recommendations
            if metrics.volatility > 0.3:
                recommendations.append("Consider reducing position sizes (High volatility)")
            
            if metrics.sortino_ratio < 0.5:
                recommendations.append("Focus on reducing downside risk")
            
            # Trading recommendations
            if metrics.total_trades < 50:
                recommendations.append("Consider increasing trading frequency for better diversification")
            
            if metrics.profit_factor < 1.2:
                recommendations.append("Improve trade selection or exit strategies")
            
            # Decay recommendations
            if "decay_analysis" in backtest_results:
                decay_info = backtest_results["decay_analysis"]
                if decay_info.get("decay_detected", False):
                    recommendations.extend(backtest_results.get("adaptations", []))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Error generating recommendations"]

class AIEnhancedBacktestingEngine(BacktestingEngine):
    """AI-Enhanced Backtesting Engine with state-of-the-art models - now using ML client"""
    
    def __init__(self, config: BacktestConfig = None):
        super().__init__(config)
        
        # Initialize AI components - now using ML client
        self.llm_analyzer = LLMBacktestAnalyzer()
        self.ai_regime_detector = AIRegimeDetector()
        self.ai_impact_calculator = AIEnhancedMarketImpactCalculator()
        
        # Replace standard market impact calculator
        self.market_impact_calc = self.ai_impact_calculator
        
        logger.info("AI-Enhanced BacktestingEngine initialized with ML client support")
    
    async def run_ai_enhanced_backtest(self, strategy_signals: pd.DataFrame, 
                                     price_data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with AI enhancements using ML client"""
        try:
            logger.info("Running AI-enhanced backtest with ML client")
            
            # Train AI models using ML client
            await self.ai_regime_detector.train_regime_model(price_data)
            await self.ai_impact_calculator.train_impact_model(price_data)
            
            # Run standard backtest
            standard_results = await self.run_backtest(strategy_signals, price_data)
            
            # Add AI-powered analysis using ML client
            ai_analysis = await self.llm_analyzer.analyze_performance(standard_results)
            strategy_insights = await self.llm_analyzer.generate_strategy_insights(
                strategy_signals, standard_results
            )
            
            # Regime analysis using ML client
            regime_analysis = await self.ai_regime_detector.predict_regime(price_data)
            
            # Enhanced results
            enhanced_results = {
                **standard_results,
                "ai_analysis": ai_analysis,
                "strategy_insights": strategy_insights,
                "regime_analysis": regime_analysis,
                "ai_enhanced": True,
                "ml_client_used": ML_CLIENT_AVAILABLE,
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"AI-enhanced backtest failed: {e}")
            # Fall back to standard backtest
            return await self.run_backtest(strategy_signals, price_data)
    
    async def generate_ai_report(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered comprehensive report"""
        try:
            # Generate standard report
            standard_report = await self.generate_report(backtest_results)
            
            # Add AI-powered sections
            ai_sections = {
                "ai_performance_analysis": backtest_results.get("ai_analysis", {}),
                "strategy_insights": backtest_results.get("strategy_insights", {}),
                "regime_analysis": backtest_results.get("regime_analysis", {}),
                "ai_recommendations": await self._generate_ai_recommendations(backtest_results),
                "market_context": await self._generate_market_context(backtest_results)
            }
            
            # Combine reports
            enhanced_report = {
                **standard_report,
                "ai_sections": ai_sections,
                "report_type": "ai_enhanced",
                "ai_confidence": 0.85
            }
            
            return enhanced_report
            
        except Exception as e:
            logger.error(f"AI report generation failed: {e}")
            return await self.generate_report(backtest_results)
    
    async def _generate_ai_recommendations(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered recommendations"""
        try:
            metrics = backtest_results.get("metrics")
            if not metrics:
                return {"error": "No metrics available"}
            
            recommendation_prompt = f"""
            You are an AI portfolio manager analyzing a trading strategy. Based on these results,
            provide specific, actionable recommendations:
            
            PERFORMANCE:
            - Sharpe Ratio: {metrics.sharpe_ratio:.2f}
            - Max Drawdown: {metrics.max_drawdown:.2%}
            - Win Rate: {metrics.win_rate:.2%}
            - Volatility: {metrics.volatility:.2%}
            
            Provide recommendations in these categories:
            1. POSITION SIZING: Optimal position sizing adjustments
            2. RISK MANAGEMENT: Specific risk controls to implement
            3. SIGNAL OPTIMIZATION: How to improve signal quality
            4. EXECUTION: Trading execution improvements
            5. PORTFOLIO CONSTRUCTION: Portfolio-level enhancements
            
            Be specific with numbers and actionable steps.
            """
            
            recommendations = await ml_client.get_completion(
                prompt=recommendation_prompt,
                model_type="llm", # Assuming LLM client for recommendations
                max_tokens=2000
            )
            
            return {
                "ai_recommendations": recommendations,
                "recommendation_type": "comprehensive_optimization",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"AI recommendations failed: {e}")
            return {"error": str(e)}
    
    async def _generate_market_context(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market context analysis"""
        try:
            context_prompt = f"""
            Analyze the market context for this trading strategy performance:
            
            STRATEGY PERFORMANCE:
            - Period: {backtest_results.get('config', {}).start_date} to {backtest_results.get('config', {}).end_date}
            - Total Return: {backtest_results.get('metrics', {}).total_return:.2%}
            - Sharpe Ratio: {backtest_results.get('metrics', {}).sharpe_ratio:.2f}
            
            Provide context on:
            1. MARKET CONDITIONS: What market conditions existed during this period?
            2. REGIME IMPACT: How did different market regimes affect performance?
            3. COMPARATIVE PERFORMANCE: How does this compare to market benchmarks?
            4. FUTURE OUTLOOK: What market conditions would favor/hurt this strategy?
            
            Provide specific, insightful analysis.
            """
            
            context = await ml_client.get_completion(
                prompt=context_prompt,
                model_type="llm", # Assuming LLM client for market context
                max_tokens=1500
            )
            
            return {
                "market_context": context,
                "context_type": "comprehensive_market_analysis",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market context generation failed: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of BacktestingEngine"""
    
    # Create configuration
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Initialize engine
    engine = BacktestingEngine(config)
    
    try:
        # Create dummy strategy signals
        date_range = pd.date_range(config.start_date, config.end_date, freq='D')
        strategy_signals = pd.DataFrame({
            'position_size': np.random.uniform(0.01, 0.1, len(date_range))
        }, index=date_range)
        
        # Create dummy price data
        price_data = pd.DataFrame({
            'close': 100 * (1 + np.random.normal(0.001, 0.02, len(date_range))).cumprod(),
            'volume': np.random.uniform(1000000, 5000000, len(date_range))
        }, index=date_range)
        
        # Run simple backtest
        print("Running simple backtest...")
        results = await engine.run_backtest(strategy_signals, price_data, BacktestType.SIMPLE)
        
        print(f"Backtest Results:")
        print(f"- Total Return: {results['metrics'].total_return:.2%}")
        print(f"- Annualized Return: {results['metrics'].annualized_return:.2%}")
        print(f"- Sharpe Ratio: {results['metrics'].sharpe_ratio:.2f}")
        print(f"- Max Drawdown: {results['metrics'].max_drawdown:.2%}")
        print(f"- Total Trades: {results['metrics'].total_trades}")
        
        # Generate report
        print("\nGenerating report...")
        report = await engine.generate_report(results)
        
        print(f"Report generated with {len(report['visualizations'])} visualizations")
        print(f"Recommendations: {len(report['recommendations'])}")
        
        # Run Monte Carlo simulation
        print("\nRunning Monte Carlo simulation...")
        mc_results = await engine.run_backtest(strategy_signals, price_data, BacktestType.MONTE_CARLO)
        
        mc_result = mc_results['monte_carlo_result']
        print(f"Monte Carlo Results:")
        print(f"- Mean Return: {mc_result.mean_return:.2%}")
        print(f"- Probability of Loss: {mc_result.probability_of_loss:.2%}")
        print(f"- 95% Confidence Interval: {mc_result.confidence_intervals['95%']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 